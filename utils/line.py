from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Optional

import requests


def _get_worker_url() -> str:
    return (os.getenv("WORKER_URL") or "").strip().rstrip("/")


def _get_auth_headers() -> dict:
    """Optional auth header for WORKER_URL.

    If you set WORKER_AUTH_TOKEN (recommended), requests will include:
      Authorization: Bearer <token>
    """

    token = (os.getenv("WORKER_AUTH_TOKEN") or os.getenv("WORKER_TOKEN") or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def send_line_text(text: str, *, chunk_size: int = 3800) -> None:
    """Send plain text to WORKER_URL.

    WORKER_URL is expected to forward the message to LINE.
    """

    url = _get_worker_url()
    if not url:
        print("[LINE] WORKER_URL is not set; skip sending.")
        return

    headers = {"Content-Type": "application/json", **_get_auth_headers()}
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
    for chunk in chunks:
        # Retry a couple of times because GitHub Actions -> Worker can be flaky.
        for attempt in range(3):
            try:
                res = requests.post(url, json={"text": chunk}, headers=headers, timeout=30)
                print("[LINE RESULT]", res.status_code, res.text[:200])
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 + attempt)
                else:
                    print(f"[LINE ERROR] failed to send text: {e}")


def send_line_image(
    image_path: str | os.PathLike,
    *,
    caption: str = "",
    key: str | None = None,
) -> None:
    """Upload an image to WORKER_URL and let the worker push it to LINE as an image message.

    The worker should accept multipart/form-data:
      - text: optional caption
      - key: optional storage key (e.g., "report_table_YYYY-MM-DD.png")
      - image: file

    Notes:
      - LINE Messaging API requires https image URLs.
        The worker is expected to upload the image (e.g. to R2) and send
        the image message using that public URL.
    """

    url = _get_worker_url()
    if not url:
        print("[LINE] WORKER_URL is not set; skip sending image.")
        return

    p = Path(image_path)
    if not p.exists():
        print(f"[LINE] image not found: {p}")
        return

    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "application/octet-stream"

    data: dict[str, str] = {}
    if caption:
        data["text"] = caption
    if key:
        data["key"] = key

    headers = _get_auth_headers()
    # NOTE: for retries, reopen the file each time so the stream position is always correct.
    for attempt in range(3):
        try:
            with p.open("rb") as f:
                files = {"image": (p.name, f, mime)}
                res = requests.post(url, data=data, files=files, headers=headers, timeout=60)
            print("[LINE IMAGE RESULT]", res.status_code, res.text[:200])
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0 + attempt)
            else:
                print(f"[LINE IMAGE ERROR] failed to send image: {e}")


def send_line(
    text: str,
    *,
    image_path: Optional[str] = None,
    image_caption: Optional[str] = None,
    image_key: Optional[str] = None,
) -> None:
    """Convenience wrapper.

    - Always sends text.
    - If image_path is provided and exists, uploads the image too.
    """

    send_line_text(text)
    if image_path:
        send_line_image(image_path, caption=image_caption or "", key=image_key)
