"""utils.line

LINE delivery helper.

This repo supports two delivery paths:

1) Cloudflare Worker (preferred)
   - Env: WORKER_URL (or LINE_WORKER_URL), WORKER_AUTH_TOKEN (or LINE_WORKER_AUTH_TOKEN)
   - POST JSON: {"text": "..."}  -> text only
   - POST multipart/form-data: field "image" (+ optional "text"/"caption", "key")
     -> worker uploads image to R2 and pushes via LINE Messaging API (push).

2) LINE Notify (fallback)
   - Env: LINE_NOTIFY_TOKEN

Important:
- In LINE Messaging API image messages, previewImageUrl has a 1MB limit.
- This repo's worker currently uses the *same* URL for previewImageUrl and originalContentUrl.
  Therefore any uploaded image must be <= ~1MB or the LINE API may reject the push.

To avoid "LINE push failed" in production, this module automatically shrinks images to
<= LINE_IMAGE_MAX_BYTES (default: 1,000,000 bytes) before uploading.

"""

from __future__ import annotations

import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover
    requests = None
    _requests_import_error = e
else:
    _requests_import_error = None

try:
    # Pillow is installed in this project (used for table image rendering).
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


@dataclass
class LineResult:
    ok: bool
    status: int
    reason: str
    image_url: Optional[str] = None


_DEFAULT_LINE_IMAGE_MAX_BYTES = 1_000_000


def _env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _line_image_max_bytes() -> int:
    # Allow overrides in emergencies.
    # - LINE_IMAGE_MAX_BYTES is preferred
    # - LINE_PREVIEW_MAX_BYTES kept for compatibility
    return _env_int("LINE_IMAGE_MAX_BYTES", _env_int("LINE_PREVIEW_MAX_BYTES", _DEFAULT_LINE_IMAGE_MAX_BYTES))


def _ensure_line_image_bytes(src_path: str, max_bytes: int) -> Tuple[str, bool]:
    """Return a file path that is <= max_bytes.

    If src_path already fits, returns (src_path, False).
    Otherwise, writes a reduced PNG into /tmp and returns (tmp_path, True).

    Strategy (in order):
      1) Lossless PNG re-save with strong compression.
      2) Palette quantization (256 colors, no dithering).
      3) Progressive downscale + compress.

    For the report table images (flat UI + text), this typically keeps readability.
    """

    try:
        if os.path.getsize(src_path) <= max_bytes:
            return src_path, False
    except Exception:
        return src_path, False

    if Image is None:
        return src_path, False

    try:
        img = Image.open(src_path)
        img.load()
    except Exception:
        return src_path, False

    tmp_base = f"/tmp/line_img_{uuid.uuid4().hex}"

    def _save_png(im: "Image.Image", out_path: str) -> bool:
        try:
            im.save(out_path, format="PNG", optimize=True, compress_level=9)
            return os.path.getsize(out_path) <= max_bytes
        except Exception:
            return False

    # (1) Lossless re-save (sometimes original isn't fully optimized)
    tmp1 = tmp_base + ".png"
    if _save_png(img, tmp1):
        return tmp1, True

    # Prepare an RGB base for quantize/downscale
    try:
        if img.mode in ("RGBA", "LA") or ("transparency" in getattr(img, "info", {})):
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            base = bg.convert("RGB")
        else:
            base = img.convert("RGB")
    except Exception:
        base = img

    # (2) Quantize (palette PNG) is very effective for UI-like images
    try:
        q = base.quantize(colors=256, dither=Image.Dither.NONE)
        tmp2 = tmp_base + "_q.png"
        if _save_png(q, tmp2):
            return tmp2, True
    except Exception:
        pass

    # (3) Downscale progressively until it fits
    w, h = base.size

    # Pillow >= 9 uses Image.Resampling.*
    try:
        lanczos = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        lanczos = getattr(Image, "LANCZOS", 1)  # type: ignore

    for scale in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
    ]:
        try:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = base.resize((nw, nh), resample=lanczos)
        except Exception:
            continue

        tmp3 = tmp_base + f"_{int(scale * 100)}.png"
        if _save_png(resized, tmp3):
            return tmp3, True

        # Also try quantized resized
        try:
            rq = resized.quantize(colors=256, dither=Image.Dither.NONE)
            tmp4 = tmp_base + f"_{int(scale * 100)}_q.png"
            if _save_png(rq, tmp4):
                return tmp4, True
        except Exception:
            pass

    return src_path, False


def _auth_headers(worker_auth_token: str | None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    token = (worker_auth_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _post_text(worker_url: str, headers: Dict[str, str], text: str, timeout: float) -> LineResult:
    if requests is None:  # pragma: no cover
        return LineResult(False, 0, f"requests import error: {_requests_import_error}")

    try:
        r = requests.post(worker_url, json={"text": text}, headers=headers, timeout=timeout)
        if r.ok:
            return LineResult(True, r.status_code, "ok")
        return LineResult(False, r.status_code, f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        return LineResult(False, 0, f"exception: {e}")


def _upload_image(
    worker_url: str,
    headers: Dict[str, str],
    image_path: str,
    caption: str | None,
    image_key: str | None,
    timeout: float,
) -> LineResult:
    if requests is None:  # pragma: no cover
        return LineResult(False, 0, f"requests import error: {_requests_import_error}")

    if not os.path.exists(image_path):
        return LineResult(False, 0, f"image not found: {image_path}")

    # MIME for the multipart upload (worker stores content-type in R2)
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "application/octet-stream"

    filename = os.path.basename(image_path)
    key = image_key or filename

    data = {"key": key}
    # Worker reads text from either "text" or "caption". Use "text".
    if caption is not None:
        data["text"] = caption

    try:
        with open(image_path, "rb") as f:
            files = {"image": (filename, f, mime)}
            r = requests.post(worker_url, headers=headers, data=data, files=files, timeout=timeout)
        if not r.ok:
            return LineResult(False, r.status_code, f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            js = r.json()
        except Exception:
            js = {}
        ok = bool(js.get("ok")) if isinstance(js, dict) else True
        image_url = js.get("imageUrl") if isinstance(js, dict) else None
        if ok:
            return LineResult(True, r.status_code, "ok", image_url=image_url)
        # worker returns ok:false with 200 sometimes; keep reason.
        reason = js.get("error") if isinstance(js, dict) else "unknown"
        return LineResult(False, r.status_code, f"worker error: {reason}")
    except Exception as e:
        return LineResult(False, 0, f"exception: {e}")


def send_line(
    text: str,
    *,
    image_path: Optional[str] = None,
    image_caption: Optional[str] = None,
    image_key: Optional[str] = None,
    force_text: bool = False,
    timeout: float = 30.0,
) -> Dict[str, object]:
    """Send to LINE.

    Returns a dict for main.py to decide whether to fail the job.

    Keys:
      ok: bool
      skipped: bool
      reason: str
      text_ok: bool
      image_ok: bool
      image_url: str|None
    """

    message = text or ""

    # For images we usually want a short caption as the message (instead of the full report).
    # main.py passes report text for the first image, and captions for the others.
    # Keep backward-compat: only override when a non-empty caption is provided.
    if image_path and image_caption and not force_text:
        message = image_caption

    token = (os.getenv("LINE_NOTIFY_TOKEN") or "").strip()
    worker_url = (os.getenv("WORKER_URL") or os.getenv("LINE_WORKER_URL") or "").strip()
    worker_auth_token = (os.getenv("WORKER_AUTH_TOKEN") or os.getenv("LINE_WORKER_AUTH_TOKEN") or "").strip()

    if not (token or worker_url):
        return {
            "ok": False,
            "skipped": True,
            "reason": "LINE_NOTIFY_TOKEN and WORKER_URL are not set",
            "text_ok": False,
            "image_ok": False,
            "image_url": None,
        }

    # Prefer worker when configured.
    if worker_url:
        headers = _auth_headers(worker_auth_token)

        had_text = bool((message or "").strip())
        text_ok = True
        image_ok = False
        image_url: Optional[str] = None
        reason_parts = []

        # 1) Try image upload (worker will also push caption text if provided).
        if image_path and not force_text:
            # main.py may provide a stable key (e.g. report_table_YYYY-MM-DD.png)
            # to make retries idempotent. Otherwise fallback to basename.
            img_key = (str(image_key).strip() if image_key else "") or os.path.basename(image_path)
            max_bytes = _line_image_max_bytes()
            upload_path, is_tmp = _ensure_line_image_bytes(image_path, max_bytes=max_bytes)
            if upload_path != image_path:
                try:
                    orig_sz = os.path.getsize(image_path)
                    new_sz = os.path.getsize(upload_path)
                    print(f"[INFO] LINE image shrink: {orig_sz} -> {new_sz} bytes (limit={max_bytes})")
                except Exception:
                    pass

            res_img = _upload_image(worker_url, headers, upload_path, caption=message, image_key=img_key, timeout=timeout)
            if is_tmp:
                try:
                    os.remove(upload_path)
                except Exception:
                    pass

            image_ok = res_img.ok
            image_url = res_img.image_url
            if not res_img.ok:
                reason_parts.append(f"image: {res_img.reason}")

                # If image fails, fall back to posting text so at least something arrives.
                if had_text:
                    res_text = _post_text(worker_url, headers, message, timeout=timeout)
                    text_ok = res_text.ok
                    if not res_text.ok:
                        reason_parts.append(f"text: {res_text.reason}")
                else:
                    text_ok = True

            else:
                # Image upload succeeded. If we had text, worker already delivered it.
                text_ok = True

        else:
            # Text-only path
            res_text = _post_text(worker_url, headers, message, timeout=timeout)
            text_ok = res_text.ok
            if not res_text.ok:
                reason_parts.append(f"text: {res_text.reason}")

        ok_all = (text_ok if had_text else True) and (image_ok if (image_path and not force_text) else True)
        return {
            "ok": bool(ok_all),
            "skipped": False,
            "reason": "; ".join(reason_parts) if reason_parts else "ok",
            "text_ok": bool(text_ok),
            "image_ok": bool(image_ok),
            "image_url": image_url,
        }

    # --- LINE Notify fallback (no worker) ---
    if requests is None:  # pragma: no cover
        return {
            "ok": False,
            "skipped": False,
            "reason": f"requests import error: {_requests_import_error}",
            "text_ok": False,
            "image_ok": False,
            "image_url": None,
        }

    api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}

    # LINE Notify requires a message field; send a single space for "image only".
    notify_message = message if (message or "").strip() else " "

    try:
        if image_path and not force_text and os.path.exists(image_path):
            max_bytes = _line_image_max_bytes()
            upload_path, is_tmp = _ensure_line_image_bytes(image_path, max_bytes=max_bytes)
            files = {"imageFile": open(upload_path, "rb")}
            data = {"message": notify_message}
            r = requests.post(api, headers=headers, data=data, files=files, timeout=timeout)
            try:
                files["imageFile"].close()
            except Exception:
                pass
            if is_tmp:
                try:
                    os.remove(upload_path)
                except Exception:
                    pass
            ok = bool(r.ok)
            return {
                "ok": ok,
                "skipped": False,
                "reason": "ok" if ok else f"HTTP {r.status_code}: {r.text[:200]}",
                "text_ok": ok,
                "image_ok": ok,
                "image_url": None,
            }

        # Text only
        r = requests.post(api, headers=headers, data={"message": notify_message}, timeout=timeout)
        ok = bool(r.ok)
        return {
            "ok": ok,
            "skipped": False,
            "reason": "ok" if ok else f"HTTP {r.status_code}: {r.text[:200]}",
            "text_ok": ok,
            "image_ok": False,
            "image_url": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "skipped": False,
            "reason": f"exception: {e}",
            "text_ok": False,
            "image_ok": False,
            "image_url": None,
        }
