from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


try:
    import requests
except Exception:  # pragma: no cover - requests should exist in runtime
    requests = None  # type: ignore


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _upload_image(worker_url: str, path: str) -> str | None:
    if requests is None:
        return None
    upload_url = worker_url.rstrip("/") + "/upload"
    auth = os.getenv("WORKER_AUTH_TOKEN", "").strip()
    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    p = Path(path)
    if not p.exists():
        return None
    with p.open("rb") as f:
        files = {"file": (p.name, f, "image/png")}
        data = {"path": p.name}
        r = requests.post(upload_url, files=files, data=data, headers=headers, timeout=40)
    if 200 <= r.status_code < 300:
        try:
            payload = r.json()
            return str(payload.get("url") or "") or None
        except Exception:
            return None
    return None


def send_line(
    text: str = "",
    image_paths: Iterable[str] | None = None,
    image_caption: str = "",
    force_image: bool = False,
    force_text: bool = False,
) -> Dict:
    worker_url = os.getenv("WORKER_URL", "").strip()
    images = [str(p) for p in (image_paths or []) if str(p).strip()]
    if not worker_url:
        if text:
            print(text)
        return {
            "ok": True,
            "text_ok": bool(text or not force_text),
            "image_ok": bool(not images or not force_image),
            "reason": "WORKER_URL missing; printed to stdout",
            "uploaded": [],
        }
    if requests is None:
        return {
            "ok": False,
            "text_ok": False,
            "image_ok": False,
            "reason": "requests unavailable",
            "uploaded": [],
        }

    uploaded: List[str] = []
    image_ok = True
    if images:
        for path in images:
            url = _upload_image(worker_url, path)
            if url:
                uploaded.append(url)
            else:
                image_ok = False
    payload = {
        "text": text,
        "imageUrls": uploaded,
        "imageCaption": image_caption,
    }
    push_url = worker_url.rstrip("/") + "/push"
    r = requests.post(push_url, json=payload, timeout=30)
    ok = 200 <= r.status_code < 300
    body = (r.text or "")[:500]
    return {
        "ok": ok and (image_ok or not force_image),
        "text_ok": ok if text or force_text else True,
        "image_ok": image_ok,
        "status_code": r.status_code,
        "body": body,
        "uploaded": uploaded,
        "reason": "ok" if ok else "push_failed",
    }


send = send_line
send_line_message = send_line
