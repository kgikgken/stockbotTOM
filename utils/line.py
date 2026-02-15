from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import requests


def _get_worker_url() -> str:
    return (os.getenv("WORKER_URL") or "").strip().rstrip("/")


def _get_auth_headers() -> dict:
    """Optional auth header for WORKER_URL.

    If you set WORKER_AUTH_TOKEN (recommended), requests will include:
      Authorization: Bearer <token>
    """
    # Support multiple env names across environments.
    token = (
        os.getenv("WORKER_AUTH_TOKEN")
        or os.getenv("WORKER_TOKEN")
        or os.getenv("WORKER_AUTH")
        or os.getenv("AUTH_TOKEN")
        or ""
    ).strip()
    if not token:
        return {}
    # Some worker versions check X-Auth-Token, others Authorization.
    return {"Authorization": f"Bearer {token}", "X-Auth-Token": token}


def send_line_text(text: str, *, chunk_size: int = 3800) -> Tuple[bool, str]:
    """Send plain text to WORKER_URL.

    Returns:
      (ok, detail) where detail is last status/err message.
    """
    url = _get_worker_url()
    if not url:
        return False, "WORKER_URL is not set"

    headers = {"Content-Type": "application/json", **_get_auth_headers()}
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    last_detail = ""
    ok_all = True

    for chunk in chunks:
        sent = False
        for attempt in range(3):
            try:
                res = requests.post(url, json={"text": chunk}, headers=headers, timeout=30)
                last_detail = f"HTTP {res.status_code}"
                if 200 <= res.status_code < 300:
                    sent = True
                    break
                # non-2xx
                last_detail = f"HTTP {res.status_code}: {res.text[:200]}"
            except Exception as e:
                last_detail = f"EXC: {repr(e)}"
            if attempt < 2:
                time.sleep(1.0 + attempt)
        if not sent:
            ok_all = False
            break

    return ok_all, last_detail or ("ok" if ok_all else "failed")


def send_line_image(
    image_path: str | os.PathLike,
    *,
    caption: str = "",
    key: str | None = None,
) -> Tuple[bool, str]:
    """Upload an image to WORKER_URL and let the worker push it to LINE as an image message.

    Returns:
      (ok, detail)
    """
    url = _get_worker_url()
    if not url:
        return False, "WORKER_URL is not set"

    p = Path(image_path)
    if not p.exists():
        return False, f"image not found: {p}"

    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "application/octet-stream"

    data: dict[str, str] = {}
    if caption:
        data["text"] = caption
    if key:
        data["key"] = key

    headers = _get_auth_headers()
    last_detail = ""

    # Newer worker versions accept multipart POSTs to root.
    # If you explicitly need /upload, set WORKER_UPLOAD_SUFFIX=/upload.
    upload_suffix = (os.getenv("WORKER_UPLOAD_SUFFIX") or "").strip()
    upload_url = url + upload_suffix

    for attempt in range(3):
        try:
            with p.open("rb") as f:
                files = {"image": (p.name, f, mime)}
                res = requests.post(upload_url, data=data, files=files, headers=headers, timeout=60)
            last_detail = f"HTTP {res.status_code}: {res.text[:200]}"
            if 200 <= res.status_code < 300:
                return True, last_detail
        except Exception as e:
            last_detail = f"EXC: {repr(e)}"
        if attempt < 2:
            time.sleep(1.0 + attempt)

    return False, last_detail or "failed"


def send_line(
    text: str,
    *,
    image_path: Optional[str] = None,
    image_caption: Optional[str] = None,
    image_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper.

    - Always attempts to send text
    - If image_path is provided and exists, uploads the image too.

    Returns:
      dict with keys: text_ok, text_detail, image_ok, image_detail
    """
    # Prefer image first so it appears at the top of the chat.
    image_ok = None
    image_detail = ""
    if image_path:
        print(f"[LINE] image: try upload ({image_path})")
        image_ok, image_detail = send_line_image(image_path, caption=image_caption or "", key=image_key)
        print(f"[LINE] image: {'OK' if image_ok else 'FAIL'} ({image_detail})")

    print("[LINE] text: try send")
    text_ok, text_detail = send_line_text(text)
    print(f"[LINE] text: {'OK' if text_ok else 'FAIL'} ({text_detail})")

    return {
        "text_ok": bool(text_ok),
        "text_detail": text_detail,
        "image_ok": None if image_ok is None else bool(image_ok),
        "image_detail": image_detail,
    }
