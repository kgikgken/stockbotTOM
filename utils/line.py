"""LINE delivery helper.

This project supports two delivery backends:

1) A custom "worker" endpoint (recommended for GitHub Actions) that forwards
   text/images to LINE.
   - Config via env:
       WORKER_URL         (required)
       WORKER_AUTH_TOKEN  (optional)

   The worker implementations seen in the wild vary. This module is intentionally
   backward-compatible with both styles:

   A. Single endpoint:
      - POST JSON  {"text": "..."}
      - POST multipart form-data with fields:
          image=<file>, text=<caption>, key=<optional key>

   B. Split endpoints:
      - POST <WORKER_URL>/notify with JSON {"text": "..."}
      - POST <WORKER_URL>/image with multipart.

2) LINE Notify API directly.
   - Config via env:
       LINE_NOTIFY_TOKEN

This module returns structured results instead of raising, so callers can decide
whether to fail the workflow.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

LINE_NOTIFY_API = "https://notify-api.line.me/api/notify"


def _env_truthy(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _headers_bearer(token: Optional[str], extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if extra:
        h.update(extra)
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _post_with_retries(
    *,
    url: str,
    headers: Dict[str, str],
    timeout: float,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> Tuple[bool, int, str]:
    """POST wrapper with minimal retries for transient failures."""

    last_status = 0
    last_text = ""
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, params=params, data=data, json=json, files=files, timeout=timeout)
            last_status = int(getattr(r, "status_code", 0) or 0)
            last_text = getattr(r, "text", "") or ""

            if 200 <= last_status < 300:
                return True, last_status, last_text

            # Retry on rate limit / transient server errors
            if last_status in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(1.0 * (2**attempt))
                continue

            return False, last_status, last_text
        except Exception as e:
            last_status = 0
            last_text = str(e)
            if attempt < max_retries - 1:
                time.sleep(1.0 * (2**attempt))
                continue
            return False, last_status, last_text

    return False, last_status, last_text


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _ensure_image_under_limit(path: Path, *, max_bytes: int, workdir: Path) -> Path:
    """Ensure the file is below max_bytes.

    Strategy:
    - If already small enough: return original.
    - If Pillow is unavailable: return original (best effort).
    - Try re-saving PNG with higher compression.
    - If still too big, convert to JPEG and downscale gradually.

    Returns a path to the (possibly) re-encoded file.
    """

    try:
        if path.stat().st_size <= max_bytes:
            return path
    except Exception:
        return path

    if Image is None:
        return path

    workdir.mkdir(parents=True, exist_ok=True)

    try:
        im = Image.open(path)
    except Exception:
        return path

    # 1) If PNG, try to re-save optimized PNG first
    if path.suffix.lower() == ".png":
        try:
            tmp_png = workdir / f"{path.stem}.line.png"
            im.save(tmp_png, format="PNG", optimize=True, compress_level=9)
            if tmp_png.stat().st_size <= max_bytes:
                return tmp_png
        except Exception:
            pass

    # 2) Convert to JPEG with gradual downscale/quality
    try:
        rgb = im.convert("RGB")
    except Exception:
        return path

    w0, h0 = rgb.size
    # Guard against weird sizes
    if w0 <= 0 or h0 <= 0:
        return path

    for scale in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]:
        w = max(1, int(w0 * scale))
        h = max(1, int(h0 * scale))
        if (w, h) != rgb.size:
            try:
                resized = rgb.resize((w, h))
            except Exception:
                resized = rgb
        else:
            resized = rgb

        for quality in [92, 88, 85, 80, 75]:
            try:
                tmp_jpg = workdir / f"{path.stem}.line.q{quality}.s{int(scale*100)}.jpg"
                resized.save(tmp_jpg, format="JPEG", quality=quality, optimize=True)
                if tmp_jpg.stat().st_size <= max_bytes:
                    return tmp_jpg
            except Exception:
                continue

    return path


def _normalize_worker_urls(worker_url: str) -> Tuple[str, str]:
    """Return (raw_url, base_url).

    raw_url: as provided (rstrip '/')
    base_url: without trailing '/notify' or '/image' if present.
    """
    raw = worker_url.rstrip("/")
    base = raw
    if base.endswith("/notify") or base.endswith("/image"):
        base = base.rsplit("/", 1)[0]
    return raw, base


def _worker_post_text(worker_url: str, auth_token: Optional[str], text: str, timeout: float) -> Tuple[bool, int, str]:
    raw, base = _normalize_worker_urls(worker_url)

    headers = _headers_bearer(auth_token, {"Content-Type": "application/json"})

    candidates = []
    # Try raw first (supports both base and explicit /notify)
    candidates.append(raw)
    # Then try base/notify
    if base and (base + "/notify") not in candidates:
        candidates.append(base + "/notify")

    payloads = [
        {"text": text},
        {"message": text},
    ]

    last: Tuple[bool, int, str] = (False, 0, "")
    for url in candidates:
        for payload in payloads:
            ok, status, body = _post_with_retries(url=url, headers=headers, json=payload, timeout=timeout)
            if ok:
                return True, status, body
            last = (ok, status, body)
            # If endpoint doesn't exist, try next url
            if status in {404, 405}:
                break

    return last


def _worker_post_image(
    worker_url: str,
    auth_token: str,
    image_path: str,
    caption: str,
    image_key: str,
    timeout: int = 20,
) -> Tuple[bool, int, str]:
    """Post an image to the worker backend.

    We may try multiple URL/payload variants for backward compatibility.
    Do NOT reuse a single file handle across attempts (it gets consumed on the
    first request). We send bytes instead so retries stay correct.
    """

    worker_url = worker_url.rstrip("/")
    raw, base = _normalize_worker_urls(worker_url)

    candidates = [f"{base}{_LINE_WORKER_IMAGE_PATH}", f"{base}/image", raw]

    caption = (caption or "").strip()
    key = (image_key or "").strip()

    payloads: List[Tuple[str, Dict[str, str]]] = [
        ("params", {"caption": caption, "key": key}),
        ("data", {"text": caption, "key": key}),
        ("data", {"text": caption}),
        ("data", {"key": key}),
    ]

    last: Tuple[bool, int, str] = (False, 0, "")

    max_bytes = int(os.getenv("LINE_IMAGE_MAX_BYTES", "900000"))
    workdir = Path(os.getenv("LINE_IMAGE_WORKDIR", "/tmp/stockbot_line_img"))
    p = _ensure_image_under_limit(Path(image_path), max_bytes=max_bytes, workdir=workdir)
    basename = p.name
    mime = _mime_for_path(str(p))
    try:
        img_bytes = p.read_bytes()
    except Exception as e:
        return False, 0, f"read_image_failed: {e}"

    for url in candidates:
        for mode, payload in payloads:
            files = {"image": (basename, img_bytes, mime)}
            kwargs = {"params": dict(payload)} if mode == "params" else {"data": dict(payload)}

            ok, status, body = _post_with_retries(
                url=url,
                headers=_auth_headers(auth_token),
                timeout=timeout,
                files=files,
                **kwargs,
            )
            if ok:
                return True, status, body

            last = (ok, status, body)

            # If endpoint is wrong, move to next url
            if status in {404, 405}:
                break

    return last


def _notify_post_text(token: str, text: str, timeout: float) -> Tuple[bool, int, str]:
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": text}
    return _post_with_retries(url=LINE_NOTIFY_API, headers=headers, data=data, timeout=timeout)


def _notify_post_image(
    notify_token: str,
    caption: str,
    image_path: str,
    timeout: int = 20,
) -> Tuple[bool, int, str]:
    url = "https://notify-api.line.me/api/notify"

    max_bytes = int(os.getenv("LINE_IMAGE_MAX_BYTES", "900000"))
    workdir = Path(os.getenv("LINE_IMAGE_WORKDIR", "/tmp/stockbot_line_img"))
    p = _ensure_image_under_limit(Path(image_path), max_bytes=max_bytes, workdir=workdir)
    basename = p.name
    mime = _mime_for_path(str(p))

    try:
        img_bytes = p.read_bytes()
    except Exception as e:
        return False, 0, f"read_image_failed: {e}"

    files = {"imageFile": (basename, img_bytes, mime)}

    payload = {
        # LINE Notify requires a message field; send a single space if empty.
        "message": caption if caption else " ",
    }

    return _post_with_retries(
        url=url,
        headers={"Authorization": f"Bearer {notify_token}"},
        timeout=timeout,
        data=payload,
        files=files,
    )


def send_line(
    text: str = "",
    *,
    image_path: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    image_caption: Optional[str] = None,
    image_key: Optional[str] = None,
    force_text: bool = False,
    force_image: bool = False,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Send a LINE notification.

    - If image_path/image_paths is provided and image delivery succeeds, this
      function **does not** send the text by default (same behaviour as the
      2026-02-24 working version).
    - If image delivery fails, it falls back to sending text (unless force_image).

    Returns a dict with keys: ok, text_ok, image_ok, skipped, reason.
    """

    # Backward compatibility: allow list via image_paths
    if image_paths and not image_path:
        # Send first image with text fallback, remaining images without.
        overall_ok = True
        any_image_fail = False
        any_text_sent = False
        for i, p in enumerate(image_paths):
            sub_text = text if i == 0 else ""
            res = send_line(
                sub_text,
                image_path=p,
                image_caption=image_caption,
                image_key=image_key if i == 0 else None,
                force_text=force_text,
                force_image=force_image,
                timeout=timeout,
            )
            overall_ok = overall_ok and bool(res.get("ok"))
            any_image_fail = any_image_fail or (not bool(res.get("image_ok", True)))
            any_text_sent = any_text_sent or bool(res.get("text_ok"))
        return {
            "ok": overall_ok,
            "text_ok": any_text_sent,
            "image_ok": not any_image_fail,
            "skipped": False,
            "reason": "batch",
        }

    send_text = _env_truthy("LINE_SEND_TEXT", True)
    send_image = _env_truthy("LINE_SEND_IMAGE", True)

    if force_text:
        send_image = False
        send_text = True
    if force_image:
        send_text = False
        send_image = True

    worker_url = os.getenv("WORKER_URL")
    worker_token = os.getenv("WORKER_AUTH_TOKEN")
    notify_token = os.getenv("LINE_NOTIFY_TOKEN")

    backend = "none"
    if worker_url:
        backend = "worker"
    elif notify_token:
        backend = "notify"

    if backend == "none":
        return {
            "ok": False,
            "text_ok": False,
            "image_ok": False,
            "skipped": True,
            "reason": "WORKER_URL and LINE_NOTIFY_TOKEN are not set",
        }

    caption = (image_caption if image_caption is not None else "").strip()

    ok_image = True
    ok_text = True

    # --- Image first (if requested) ---
    if send_image and image_path:
        path = Path(image_path)
        if not path.exists():
            ok_image = False
            status, body = 0, f"image file not found: {image_path}"
        else:
            if backend == "worker":
                ok, status, body = _worker_post_image(worker_url, worker_token, path, caption, image_key, timeout)
            else:
                ok, status, body = _notify_post_image(notify_token, path, caption, timeout)  # type: ignore[arg-type]
            ok_image = bool(ok)

        if not ok_image:
            print(f"LINE image delivery error: {status} {body}")

            # Fallback to text when image failed (unless force_image)
            if (not force_image) and send_text and text.strip():
                if backend == "worker":
                    ok, status, body = _worker_post_text(worker_url, worker_token, text, timeout)
                else:
                    ok, status, body = _notify_post_text(notify_token, text, timeout)  # type: ignore[arg-type]
                ok_text = bool(ok)
                if not ok_text:
                    print(f"LINE text delivery error (fallback): {status} {body}")
                return {
                    "ok": ok_text,
                    "text_ok": ok_text,
                    "image_ok": False,
                    "skipped": False,
                    "reason": "image_failed_text_fallback",
                }

            return {
                "ok": False,
                "text_ok": False,
                "image_ok": False,
                "skipped": False,
                "reason": "image_failed",
            }

        # Image succeeded. Skip text unless force_text.
        if not force_text:
            return {
                "ok": True,
                "text_ok": True,
                "image_ok": True,
                "skipped": False,
                "reason": "image_only",
            }

    # --- Text (normal path) ---
    if send_text and text.strip():
        if backend == "worker":
            ok, status, body = _worker_post_text(worker_url, worker_token, text, timeout)
        else:
            ok, status, body = _notify_post_text(notify_token, text, timeout)  # type: ignore[arg-type]
        ok_text = bool(ok)
        if not ok_text:
            print(f"LINE text delivery error: {status} {body}")
        return {
            "ok": ok_text,
            "text_ok": ok_text,
            "image_ok": True,
            "skipped": False,
            "reason": "text_only",
        }

    # Nothing to send
    return {
        "ok": True,
        "text_ok": True,
        "image_ok": True,
        "skipped": True,
        "reason": "no_content",
    }
