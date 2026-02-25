# utils/line.py
# Minimal LINE Notify wrapper.
#
# Default behavior (for daily ops): send images, skip text.
# In error/fail-safe mode you can force text delivery via send_line(..., force_text=True).
#
# Compatibility:
# - Accept both image_path and image_paths.
# - Accept image_caption (caption attached to the image upload).
# - Ignore unexpected kwargs (warn) to prevent CI/runtime crashes when callers evolve.

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

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
    # Some callers pass `image_caption` (caption text for image upload).
    # LINE Notify requires the "message" field even for image uploads.
    image_caption: str = "",
    force_text: bool = False,
    force_image: bool = False,
    # Swallow extra kwargs for compatibility (print a warning so bugs are still visible).
    **_ignored: Any,
) -> None:
    """Send a LINE Notify message (text and/or images).

    Args:
        text:
            Text body.
            - Normal mode: used as fallback when images fail to send.
            - force_text=True: always sent (for fail-safe/error notifications).
        image_paths: Local file paths to images to send.
        image_path: Single local image path (alias for image_paths=[...]).
        image_caption:
            Caption attached to the image upload.
            LINE Notify requires the "message" field even for image uploads.
        force_text: Force text sending regardless of LINE_SEND_TEXT.
        force_image: Force image sending regardless of LINE_SEND_IMAGE.

    Env:
        LINE_NOTIFY_TOKEN: required.
        LINE_SEND_IMAGE: default True.
        LINE_SEND_TEXT : default False.

    Notes:
        - When images are provided and LINE_SEND_TEXT is false, `text` is used only as a
          fallback if *no* image could be delivered (prevents "text + image spam").
        - Unknown kwargs are ignored with a warning to avoid runtime crashes.
    """

    if _ignored:
        try:
            keys = ", ".join(sorted(_ignored.keys()))
            print(f"[WARN] send_line: ignoring unknown kwargs: {keys}")
        except Exception:
            pass

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

    image_attempted = False
    image_sent = False

    # 1) Send images first (if enabled)
    if send_image and norm_paths:
        image_attempted = True
        for idx, path in enumerate(norm_paths):
            if not path or not os.path.exists(path):
                continue

            # Attach caption only to the first image to avoid repetition.
            cap = str(image_caption or "") if idx == 0 else ""

            try:
                with open(path, "rb") as f:
                    files = {"imageFile": f}
                    payload = {"message": cap}  # must include message key
                    r = requests.post(
                        "https://notify-api.line.me/api/notify",
                        headers=headers,
                        data=payload,
                        files=files,
                        timeout=30,
                    )
                if r.status_code == 200:
                    image_sent = True
                else:
                    print(f"[WARN] LINE image notify failed: {r.status_code} {r.text}")
            except Exception as e:
                print(f"[WARN] LINE image notify exception: {e}")

    # 2) Send text (if enabled OR fail-safe fallback)
    # - force_text: always
    # - send_text (LINE_SEND_TEXT): enabled
    # - fallback: image attempted but none succeeded -> send text even if LINE_SEND_TEXT is false
    should_send_text = False
    if force_text:
        should_send_text = True
    elif send_text:
        should_send_text = True
    elif image_attempted and (not image_sent) and text.strip():
        should_send_text = True

    if should_send_text and text.strip():
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
