"""LINE delivery helper.

Primary backend: custom Cloudflare Worker (recommended for GitHub Actions).
Fallback backend: LINE Notify (legacy).

This module is intentionally defensive because multiple worker shapes have been
used during the project history.

Supported worker styles
-----------------------
A) Single endpoint
   - POST <WORKER_URL> JSON {"text": "..."}
   - POST <WORKER_URL> multipart form-data with image + text/key fields

B) Split endpoints
   - POST <WORKER_URL>/notify JSON {"text": "..."}
   - POST <WORKER_URL>/upload multipart form-data with image + text/key fields
   - Some historical variants used /image instead of /upload

The function returns a structured dict and should not raise in normal operation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

LINE_NOTIFY_API = "https://notify-api.line.me/api/notify"
DEFAULT_IMAGE_MAX_BYTES = 950_000


def _env_truthy(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _normalize_worker_urls(worker_url: str) -> Tuple[str, str]:
    raw = str(worker_url or "").rstrip("/")
    base = raw
    for suffix in ("/notify", "/upload", "/image"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return raw, base


def _auth_headers(token: Optional[str], extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if extra:
        headers.update(extra)
    if token:
        # Historical worker variants have used either x-auth-token or Bearer.
        headers.setdefault("Authorization", f"Bearer {token}")
        headers.setdefault("x-auth-token", token)
    return headers


def _mime_for_path(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _post_with_retries(
    *,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> Tuple[bool, int, str]:
    last_status = 0
    last_text = ""
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url,
                headers=headers or {},
                params=params,
                data=data,
                json=json,
                files=files,
                timeout=timeout,
            )
            last_status = int(getattr(r, "status_code", 0) or 0)
            last_text = (getattr(r, "text", "") or "")[:500]
            if 200 <= last_status < 300:
                return True, last_status, last_text
            # Retry only for transient cases.
            if last_status in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(0.8 * (2**attempt))
                continue
            return False, last_status, last_text
        except Exception as e:  # pragma: no cover - network/runtime dependent
            last_status = 0
            last_text = str(e)
            if attempt < max_retries - 1:
                time.sleep(0.8 * (2**attempt))
                continue
            return False, last_status, last_text
    return False, last_status, last_text


def _ensure_image_under_limit(path: Path, *, max_bytes: int, workdir: Path) -> Path:
    """Best-effort shrinker for LINE preview/worker image limits.

    Returns original path if already small enough or if Pillow is unavailable.
    """
    try:
        if path.stat().st_size <= max_bytes:
            return path
    except Exception:
        return path

    if Image is None:
        return path

    try:
        im = Image.open(path)
    except Exception:
        return path

    workdir.mkdir(parents=True, exist_ok=True)

    # 1) Try optimized PNG first (best for table images)
    if path.suffix.lower() == ".png":
        try:
            tmp_png = workdir / f"{path.stem}.line.png"
            im.save(tmp_png, format="PNG", optimize=True, compress_level=9)
            if tmp_png.stat().st_size <= max_bytes:
                return tmp_png
        except Exception:
            pass

    # 2) Quantize to palette PNG (works very well for charts/tables)
    try:
        palette = im.convert("P", palette=Image.ADAPTIVE, colors=256)  # type: ignore[attr-defined]
        tmp_q = workdir / f"{path.stem}.line.pal.png"
        palette.save(tmp_q, format="PNG", optimize=True, compress_level=9)
        if tmp_q.stat().st_size <= max_bytes:
            return tmp_q
    except Exception:
        pass

    # 3) Fall back to JPEG with gradual downscale
    try:
        rgb = im.convert("RGB")
    except Exception:
        return path

    w0, h0 = rgb.size
    if w0 <= 0 or h0 <= 0:
        return path

    for scale in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        w = max(1, int(w0 * scale))
        h = max(1, int(h0 * scale))
        cur = rgb if (w, h) == rgb.size else rgb.resize((w, h))
        for quality in (90, 85, 80, 75, 70):
            try:
                tmp_jpg = workdir / f"{path.stem}.line.s{int(scale*100)}.q{quality}.jpg"
                cur.save(tmp_jpg, format="JPEG", quality=quality, optimize=True)
                if tmp_jpg.stat().st_size <= max_bytes:
                    return tmp_jpg
            except Exception:
                continue
    return path


def _worker_post_text(worker_url: str, auth_token: Optional[str], text: str, timeout: float) -> Tuple[bool, int, str]:
    raw, base = _normalize_worker_urls(worker_url)
    urls = _unique_keep_order([raw, f"{base}/notify"])
    payloads = [{"text": text}, {"message": text}]
    last = (False, 0, "")
    for url in urls:
        for payload in payloads:
            ok, status, body = _post_with_retries(
                url=url,
                headers=_auth_headers(auth_token, {"Content-Type": "application/json"}),
                json=payload,
                timeout=timeout,
            )
            if ok:
                return True, status, body
            last = (ok, status, body)
            if status in {404, 405}:
                break
    return last


def _worker_post_image(
    worker_url: str,
    auth_token: Optional[str],
    image_path: str,
    caption: str,
    image_key: str,
    timeout: float = 30.0,
) -> Tuple[bool, int, str]:
    raw, base = _normalize_worker_urls(worker_url)
    urls = _unique_keep_order([
        f"{base}/upload",  # current/expected worker path
        raw,               # legacy single-endpoint multipart
        f"{base}/image",  # older compatibility path
    ])

    # Some worker variants require a non-empty text field even for image-only messages.
    caption_safe = (caption or "").strip() or "\u200b"
    key = (image_key or "").strip() or Path(image_path).name

    p = Path(image_path)
    if not p.exists():
        return False, 0, f"image file not found: {image_path}"

    max_bytes = int(float(os.getenv("LINE_IMAGE_MAX_BYTES", str(DEFAULT_IMAGE_MAX_BYTES)) or DEFAULT_IMAGE_MAX_BYTES))
    p2 = _ensure_image_under_limit(p, max_bytes=max_bytes, workdir=Path("out") / "_line_tmp")

    try:
        img_bytes = p2.read_bytes()
    except Exception as e:
        return False, 0, f"read_image_failed: {e}"

    basename = p2.name
    mime = _mime_for_path(p2)

    file_field_candidates = ["image", "file", "imageFile"]
    payload_variants: List[Tuple[str, Dict[str, str]]] = [
        ("data", {"text": caption_safe, "key": key}),
        ("data", {"text": caption_safe, "image_key": key}),
        ("data", {"caption": caption_safe, "image_key": key}),
        ("data", {"message": caption_safe, "key": key}),
        ("data", {"message": caption_safe, "image_key": key}),
        ("data", {"text": caption_safe}),
        ("data", {"caption": caption_safe}),
        ("data", {"message": caption_safe}),
        ("params", {"caption": caption_safe, "key": key}),
        ("params", {"text": caption_safe, "key": key}),
    ]

    last = (False, 0, "")
    for url in urls:
        for file_field in file_field_candidates:
            for mode, payload in payload_variants:
                files = {file_field: (basename, img_bytes, mime)}
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
                if status in {404, 405}:
                    break
            if last[1] in {404, 405}:
                break
    return last


def _notify_post_text(token: str, text: str, timeout: float) -> Tuple[bool, int, str]:
    return _post_with_retries(
        url=LINE_NOTIFY_API,
        headers={"Authorization": f"Bearer {token}"},
        data={"message": text},
        timeout=timeout,
    )


def _notify_post_image(token: str, image_path: str, caption: str, timeout: float) -> Tuple[bool, int, str]:
    p = Path(image_path)
    if not p.exists():
        return False, 0, f"image file not found: {image_path}"
    try:
        img_bytes = p.read_bytes()
    except Exception as e:
        return False, 0, f"read_image_failed: {e}"
    files = {"imageFile": (p.name, img_bytes, _mime_for_path(p))}
    payload = {"message": caption if caption else " "}
    return _post_with_retries(
        url=LINE_NOTIFY_API,
        headers={"Authorization": f"Bearer {token}"},
        data=payload,
        files=files,
        timeout=timeout,
    )


def send_line(
    text: str = "",
    *,
    image_path: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    image_caption: Optional[str] = None,
    image_key: Optional[str] = None,
    force_text: bool = False,
    force_image: bool = False,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Send a LINE notification.

    Behavior:
    - If image(s) are requested and the first image succeeds, text is not sent by
      default (same UX as the working 2026-02-24 setup).
    - If an image fails and text is available, text is sent as a fallback unless
      force_image=True.

    Returns a structured dict:
      ok, text_ok, image_ok, skipped, reason, backend, status_code, body
    """

    # Batch helper: send first image with fallback text, remaining images image-only.
    if image_paths and not image_path:
        overall_ok = True
        image_ok_all = True
        text_ok_any = False
        statuses: List[int] = []
        reasons: List[str] = []
        for idx, p in enumerate(image_paths):
            res = send_line(
                text if idx == 0 else "",
                image_path=p,
                image_caption=image_caption,
                image_key=image_key if idx == 0 else None,
                force_text=force_text,
                force_image=force_image,
                timeout=timeout,
            )
            overall_ok = overall_ok and bool(res.get("ok", False))
            image_ok_all = image_ok_all and bool(res.get("image_ok", False))
            text_ok_any = text_ok_any or bool(res.get("text_ok", False))
            statuses.append(int(res.get("status_code", 0) or 0))
            reasons.append(str(res.get("reason", "")))
        return {
            "ok": overall_ok,
            "text_ok": text_ok_any,
            "image_ok": image_ok_all,
            "skipped": False,
            "reason": ",".join([r for r in reasons if r]),
            "backend": "batch",
            "status_code": max(statuses) if statuses else 0,
        }

    send_image = _env_truthy("LINE_SEND_IMAGE", True)
    send_text = _env_truthy("LINE_SEND_TEXT", False)
    if force_text:
        send_text = True
        send_image = False
    if force_image:
        send_image = True
        send_text = False

    worker_url = (os.getenv("WORKER_URL") or os.getenv("LINE_WORKER_URL") or "").strip()
    worker_token = (
        os.getenv("WORKER_AUTH_TOKEN")
        or os.getenv("LINE_WORKER_AUTH_TOKEN")
        or os.getenv("UPLOAD_TOKEN")
        or os.getenv("AUTH_TOKEN")
        or ""
    ).strip() or None
    notify_token = (
        os.getenv("LINE_NOTIFY_TOKEN")
        or os.getenv("LINE_NOTIFY_ACCESS_TOKEN")
        or os.getenv("LINE_TOKEN")
        or ""
    ).strip()

    if worker_url:
        backend = "worker"
    elif notify_token:
        backend = "notify"
    else:
        return {
            "ok": False,
            "text_ok": False,
            "image_ok": False,
            "skipped": True,
            "reason": "WORKER_URL and LINE_NOTIFY_TOKEN/LINE_TOKEN are not set",
            "backend": "none",
            "status_code": 0,
            "body": "",
        }

    caption = (image_caption or "").strip()

    # 1) Try image first (daily ops: images only)
    if send_image and image_path:
        if backend == "worker":
            ok, status, body = _worker_post_image(worker_url, worker_token, image_path, caption, image_key or "", timeout)
        else:
            ok, status, body = _notify_post_image(notify_token, image_path, caption, timeout)

        if ok:
            if force_text and text.strip():
                # Explicitly requested text too
                if backend == "worker":
                    t_ok, t_status, t_body = _worker_post_text(worker_url, worker_token, text, timeout)
                else:
                    t_ok, t_status, t_body = _notify_post_text(notify_token, text, timeout)
                return {
                    "ok": bool(t_ok),
                    "text_ok": bool(t_ok),
                    "image_ok": True,
                    "skipped": False,
                    "reason": "image_then_text" if t_ok else "image_ok_text_failed",
                    "backend": backend,
                    "status_code": int(t_status or status or 0),
                    "body": t_body or body,
                }
            return {
                "ok": True,
                "text_ok": True,
                "image_ok": True,
                "skipped": False,
                "reason": "image_only",
                "backend": backend,
                "status_code": int(status or 0),
                "body": body,
            }

        # Image failed -> optional fallback to text
        print(f"LINE image delivery error: {status} {body}")
        if (not force_image) and text.strip():
            if backend == "worker":
                t_ok, t_status, t_body = _worker_post_text(worker_url, worker_token, text, timeout)
            else:
                t_ok, t_status, t_body = _notify_post_text(notify_token, text, timeout)
            return {
                "ok": bool(t_ok),
                "text_ok": bool(t_ok),
                "image_ok": False,
                "skipped": False,
                "reason": "image_failed_text_fallback" if t_ok else "image_failed_text_failed",
                "backend": backend,
                "status_code": int(t_status or status or 0),
                "body": t_body or body,
            }
        return {
            "ok": False,
            "text_ok": False,
            "image_ok": False,
            "skipped": False,
            "reason": "image_failed",
            "backend": backend,
            "status_code": int(status or 0),
            "body": body,
        }

    # 2) Text-only path
    if send_text and text.strip():
        if backend == "worker":
            ok, status, body = _worker_post_text(worker_url, worker_token, text, timeout)
        else:
            ok, status, body = _notify_post_text(notify_token, text, timeout)
        return {
            "ok": bool(ok),
            "text_ok": bool(ok),
            "image_ok": True,
            "skipped": False,
            "reason": "text_only" if ok else "text_failed",
            "backend": backend,
            "status_code": int(status or 0),
            "body": body,
        }

    return {
        "ok": True,
        "text_ok": True,
        "image_ok": True,
        "skipped": True,
        "reason": "no_content",
        "backend": backend,
        "status_code": 0,
        "body": "",
    }
