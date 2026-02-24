# utils/line.py
# Minimal LINE Notify wrapper.
#
# Default behavior (for daily ops): send images, skip text.
# In error/fail-safe mode you can force text delivery via send_line(..., force_text=True).

from __future__ import annotations

import os
from typing import Optional, Sequence

import requests

from utils.util import env_truthy


def send_line(
    text: str = "",
    image_paths: Optional[Sequence[str]] = None,
    *,
    # Backward/forward compatibility:
    # - Some callers historically used `image_path="..."` (single image).
    # - Current implementation prefers `image_paths=[...]`.
    # Accept both to avoid runtime TypeError in CI.
    image_path: Optional[str] = None,
    force_text: bool = False,
    force_image: bool = False,
) -> None:
    """Send a LINE Notify message.

    Args:
        text: Message text.
        image_paths: Local file paths to images to send.
        image_path: Single local image path (alias for image_paths=[...]).
        force_text: If True, send text even when LINE_SEND_TEXT is false.
        force_image: If True, send images even when LINE_SEND_IMAGE is false.

    Notes:
        - Uses LINE_NOTIFY_TOKEN env.
        - Env switches:
          - LINE_SEND_IMAGE (default: True)
          - LINE_SEND_TEXT  (default: False)
    """

    token = os.getenv("LINE_NOTIFY_TOKEN", "").strip()
    if not token:
        print("[WARN] LINE_NOTIFY_TOKEN is not set. Skipping LINE notify.")
        return

    # Default: image-only. Allow force_* override for fail-safe notifications.
    send_image = env_truthy("LINE_SEND_IMAGE", True) or force_image
    send_text = env_truthy("LINE_SEND_TEXT", False) or force_text

    headers = {"Authorization": f"Bearer {token}"}

    # Normalize image path inputs (preserve order; de-dup)
    paths: list[str] = []
    if image_paths:
        paths.extend([p for p in image_paths if p])
    if image_path:
        paths.append(image_path)

    seen: set[str] = set()
    norm_paths: list[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        norm_paths.append(p)

    # 1) Send images first (if enabled)
    if send_image and norm_paths:
        for path in norm_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    files = {"imageFile": f}
                    payload = {"message": ""}  # must include message key
                    r = requests.post(
                        "https://notify-api.line.me/api/notify",
                        headers=headers,
                        data=payload,
                        files=files,
                        timeout=30,
                    )
                if r.status_code != 200:
                    print(f"[WARN] LINE image notify failed: {r.status_code} {r.text}")
            except Exception as e:
                print(f"[WARN] LINE image notify exception: {e}")

    # 2) Send text (if enabled)
    if send_text and text.strip():
        try:
            payload = {"message": text}
            r = requests.post(
                "https://notify-api.line.me/api/notify",
                headers=headers,
                data=payload,
                timeout=30,
            )
            if r.status_code != 200:
                print(f"[WARN] LINE text notify failed: {r.status_code} {r.text}")
        except Exception as e:
            print(f"[WARN] LINE text notify exception: {e}")
