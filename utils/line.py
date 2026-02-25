"""LINE delivery helper (via Cloudflare Worker).

This repository is designed to run on GitHub Actions.

- Text delivery:
    POST JSON to WORKER_URL: {"text": "..."}

- Image delivery:
    POST multipart/form-data to WORKER_URL with field "image" (file)
    and optional fields:
        - text: optional caption
        - key : optional storage key (e.g. report_table_YYYY-MM-DD.png)

The Cloudflare Worker then stores the image (e.g. R2) and pushes it to LINE.

Env vars used:
- WORKER_URL: Cloudflare Worker endpoint
- WORKER_AUTH_TOKEN: Bearer token (required for image uploads if Worker enforces auth)

Important:
- This module MUST NOT raise; it returns a status dict so the caller can decide
  whether to fail the workflow.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


def _auth_headers() -> Dict[str, str]:
    token = os.getenv("WORKER_AUTH_TOKEN") or os.getenv("WORKER_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _chunk_text(text: str, max_len: int = 4500) -> List[str]:
    """Split text into chunks to avoid LINE message length errors."""

    if not text:
        return []

    text = text.replace("\r\n", "\n")

    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for line in text.split("\n"):
        add_len = len(line) + (1 if buf else 0)
        if buf_len + add_len <= max_len:
            if buf:
                buf.append("\n")
                buf_len += 1
            buf.append(line)
            buf_len += len(line)
            continue

        if buf:
            chunks.append("".join(buf))
            buf = []
            buf_len = 0

        while len(line) > max_len:
            chunks.append(line[:max_len])
            line = line[max_len:]
        buf.append(line)
        buf_len = len(line)

    if buf:
        chunks.append("".join(buf))

    return chunks


def _post_text(worker_url: str, text: str, timeout: int = 25) -> Tuple[bool, str]:
    """Send one text chunk."""

    try:
        r = requests.post(
            worker_url,
            json={"text": text},
            headers={"Content-Type": "application/json", **_auth_headers()},
            timeout=timeout,
        )
        if r.ok:
            return True, ""
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _upload_image(
    worker_url: str,
    image_path: str,
    *,
    caption: str = "",
    key: Optional[str] = None,
    timeout: int = 40,
) -> Tuple[bool, str]:
    """Upload one image (multipart/form-data)."""

    if not image_path or not os.path.exists(image_path):
        return False, f"image not found: {image_path}"

    data: Dict[str, str] = {}
    if caption:
        data["text"] = caption
    if key:
        data["key"] = key

    headers = _auth_headers()

    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f)}
            r = requests.post(worker_url, files=files, data=data, headers=headers, timeout=timeout)
        if r.ok:
            return True, ""
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def send_line(
    text: str,
    *,
    image_path: Optional[str] = None,
    image_caption: str = "",
    image_key: Optional[str] = None,
    force_text: bool = False,
) -> Dict[str, Any]:
    """Send to LINE via Worker.

    Args:
        text: message body (used for text-only send, or as fallback when image upload fails)
        image_path: optional local image path (PNG) to upload
        image_caption: optional short caption to attach (Worker sends it after the image)
        image_key: optional storage key passed to Worker
        force_text: if True, ignore image_path and send text only

    Returns:
        dict with keys: ok, skipped, reason, image_ok, text_ok
    """

    worker_url = os.getenv("WORKER_URL")
    if not worker_url:
        print("[WARN] WORKER_URL is not set. Skipping LINE notify.")
        return {
            "ok": False,
            "skipped": True,
            "reason": "WORKER_URL is not set",
            "image_ok": False,
            "text_ok": False,
        }

    image_ok = True
    text_ok = True
    reason_parts: List[str] = []

    # 1) Image upload (optional)
    if image_path and (not force_text):
        ok, err = _upload_image(worker_url, image_path, caption=image_caption.strip(), key=image_key)
        if not ok:
            image_ok = False
            reason_parts.append(f"image: {err}")

            # Fallback to text (if provided)
            txt = (text or "").strip()
            if txt:
                text_ok = True
                for chunk in _chunk_text(txt):
                    ok2, err2 = _post_text(worker_url, chunk)
                    if not ok2:
                        text_ok = False
                        reason_parts.append(f"text: {err2}")
                        break
                    time.sleep(0.2)
            else:
                text_ok = False
        else:
            # image sent successfully; text is not sent by default in this mode
            text_ok = True

    # 2) Text-only mode
    else:
        txt = (text or "").strip()
        if txt:
            for chunk in _chunk_text(txt):
                ok, err = _post_text(worker_url, chunk)
                if not ok:
                    text_ok = False
                    reason_parts.append(f"text: {err}")
                    break
                time.sleep(0.2)
        else:
            # Nothing to send (treat as OK)
            text_ok = True

    # Overall success condition:
    # - If we attempted image delivery and it succeeded -> success (even if we didn't send text).
    # - If image delivery failed but we had non-empty text and successfully fell back to text -> success.
    # - Otherwise (text-only) -> success iff text_ok.
    ok_all = True
    if image_path and (not force_text):
        if image_ok:
            ok_all = True
        else:
            # We only consider the fallback successful if there was actually text to send.
            had_text = bool((text or "").strip())
            ok_all = had_text and bool(text_ok)
    else:
        ok_all = bool(text_ok)

    return {
        "ok": bool(ok_all),
        "skipped": False,
        "reason": " / ".join(reason_parts),
        "image_ok": bool(image_ok) if (image_path and not force_text) else True,
        "text_ok": bool(text_ok),
    }
