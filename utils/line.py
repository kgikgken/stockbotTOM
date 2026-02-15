# utils/line.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests


def _normalize_url(url: str) -> str:
    url = url.strip()
    return url[:-1] if url.endswith("/") else url


def send_line_text(text: str, worker_url: Optional[str] = None, auth_token: Optional[str] = None) -> requests.Response:
    """
    Send plain text to Cloudflare Worker (which pushes to LINE).
    """
    if not worker_url:
        worker_url = os.getenv("WORKER_URL") or os.getenv("WORKER_ENDPOINT")
    if not worker_url:
        raise RuntimeError("WORKER_URL is not set")

    worker_url = _normalize_url(worker_url)
    auth_token = auth_token or os.getenv("WORKER_AUTH_TOKEN")

    headers = {"content-type": "application/json; charset=utf-8"}
    if auth_token:
        headers["x-auth-token"] = auth_token

    payload = {"text": text}
    res = requests.post(worker_url, json=payload, headers=headers, timeout=60)
    return res


def send_line_image(
    image_path: str,
    text: str = "",
    worker_url: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> requests.Response:
    """
    Upload image to Cloudflare Worker via multipart/form-data.
    Worker stores it to R2, then pushes image + optional text to LINE.
    """
    if not worker_url:
        worker_url = os.getenv("WORKER_URL") or os.getenv("WORKER_ENDPOINT")
    if not worker_url:
        raise RuntimeError("WORKER_URL is not set")

    worker_url = _normalize_url(worker_url)
    auth_token = auth_token or os.getenv("WORKER_AUTH_TOKEN")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    headers = {}
    if auth_token:
        headers["x-auth-token"] = auth_token

    # Put a deterministic key into R2: reports/<filename>
    data = {
        "text": text,
        "key": path.name,
    }

    with path.open("rb") as f:
        files = {
            "image": (path.name, f, "image/png"),
        }
        res = requests.post(worker_url, data=data, files=files, headers=headers, timeout=120)
    return res