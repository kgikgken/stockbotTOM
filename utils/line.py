from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import ParseResult, urlparse, urlunparse


try:
    import requests
except Exception:  # pragma: no cover - requests should exist in runtime
    requests = None  # type: ignore


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _replace_path(parsed: ParseResult, new_path: str) -> str:
    safe_path = new_path if new_path.startswith("/") else f"/{new_path}" if new_path else "/"
    return urlunparse(parsed._replace(path=safe_path, params="", query="", fragment=""))


def _resolve_worker_endpoints(worker_url: str) -> Dict[str, List[str]]:
    raw = str(worker_url or "").strip()
    if not raw:
        return {"push": [], "upload": [], "base": []}
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return {"push": [], "upload": [], "base": []}

    path = (parsed.path or "").rstrip("/")
    terminal = ""
    base_path = path
    for suffix in ("/push", "/upload"):
        if path.endswith(suffix):
            terminal = suffix[1:]
            base_path = path[: -len(suffix)]
            break

    push_paths: List[str] = []
    upload_paths: List[str] = []
    base_urls: List[str] = []

    if terminal == "push":
        push_paths.append(path or "/push")
    if terminal == "upload":
        upload_paths.append(path or "/upload")

    if base_path:
        push_paths.append(f"{base_path}/push")
        upload_paths.append(f"{base_path}/upload")
        base_urls.append(_replace_path(parsed, base_path))
    else:
        base_urls.append(_replace_path(parsed, "/"))

    push_paths.append("/push")
    upload_paths.append("/upload")

    if path and terminal not in {"push", "upload"}:
        push_paths.append(path)
        base_urls.append(_replace_path(parsed, path))

    return {
        "push": [_replace_path(parsed, p) for p in _dedupe_keep_order(push_paths)],
        "upload": [_replace_path(parsed, p) for p in _dedupe_keep_order(upload_paths)],
        "base": _dedupe_keep_order(base_urls),
    }


def _pick_response_text(resp) -> str:
    try:
        return (resp.text or "")[:500]
    except Exception:
        return ""


def _post_first_success(urls: Sequence[str], *, json_payload=None, files=None, data=None, headers=None, timeout: int = 30):
    if requests is None:
        return None, "", "requests unavailable"
    last_resp = None
    last_url = ""
    last_err = ""
    for url in urls:
        last_url = url
        try:
            resp = requests.post(url, json=json_payload, files=files, data=data, headers=headers, timeout=timeout)
            last_resp = resp
            if 200 <= resp.status_code < 300:
                return resp, url, ""
            last_err = f"HTTP {resp.status_code}: {_pick_response_text(resp)}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return last_resp, last_url, last_err


def _upload_images(worker_url: str, image_paths: Sequence[str]) -> Tuple[List[str], List[str], str, str]:
    endpoints = _resolve_worker_endpoints(worker_url)
    upload_urls = endpoints.get("upload", [])
    auth = os.getenv("WORKER_AUTH_TOKEN", "").strip()
    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    uploaded: List[str] = []
    failures: List[str] = []
    last_target = ""
    last_error = ""
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            failures.append(f"{p.name}: file_missing")
            continue
        with p.open("rb") as f:
            files = {"file": (p.name, f, "image/png")}
            data = {"path": p.name}
            resp, used_url, err = _post_first_success(
                upload_urls,
                files=files,
                data=data,
                headers=headers,
                timeout=40,
            )
        last_target = used_url or last_target
        last_error = err or last_error
        if resp is None or not (200 <= resp.status_code < 300):
            failures.append(f"{p.name}: {err or 'upload_failed'}")
            continue
        try:
            payload = resp.json()
        except Exception:
            failures.append(f"{p.name}: invalid_json")
            continue
        url = str(payload.get("url") or "").strip()
        if not url:
            failures.append(f"{p.name}: missing_url")
            continue
        uploaded.append(url)
    return uploaded, failures, last_target, last_error


def _push_messages(worker_url: str, payload: Dict, timeout: int = 30) -> Dict:
    endpoints = _resolve_worker_endpoints(worker_url)
    push_urls = endpoints.get("push", [])
    resp, used_url, err = _post_first_success(push_urls, json_payload=payload, timeout=timeout)
    if resp is None:
        return {
            "ok": False,
            "status_code": None,
            "body": err,
            "push_url": used_url,
            "worker_ok": False,
            "line_ok": False,
        }

    body = _pick_response_text(resp)
    worker_ok = 200 <= resp.status_code < 300
    line_ok = worker_ok
    line_status = None
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            if "ok" in parsed:
                line_ok = bool(parsed.get("ok"))
            if parsed.get("status") is not None:
                line_status = parsed.get("status")
            if parsed.get("body"):
                body = str(parsed.get("body"))[:500]
    except Exception:
        parsed = None

    return {
        "ok": worker_ok and line_ok,
        "status_code": resp.status_code,
        "line_status": line_status,
        "body": body,
        "push_url": used_url,
        "worker_ok": worker_ok,
        "line_ok": line_ok,
        "raw_json": parsed if isinstance(parsed, dict) else None,
    }


def send_line(
    text: str = "",
    image_paths: Iterable[str] | None = None,
    image_caption: str = "",
    force_image: bool = False,
    force_text: bool = False,
) -> Dict:
    worker_url = os.getenv("WORKER_URL", "").strip()
    images = [str(p) for p in (image_paths or []) if str(p).strip()]
    text = str(text or "")
    text_requested = bool(text.strip() or force_text)
    image_requested = bool(images or force_image)

    if not worker_url:
        if text.strip():
            print(text)
        return {
            "ok": False,
            "text_ok": False,
            "image_ok": False,
            "text_requested": text_requested,
            "image_requested": image_requested,
            "reason": "WORKER_URL missing; printed to stdout only",
            "uploaded": [],
            "upload_failures": ["WORKER_URL missing"] if images else [],
            "stdout_fallback": bool(text.strip()),
        }
    if requests is None:
        return {
            "ok": False,
            "text_ok": False,
            "image_ok": False,
            "text_requested": text_requested,
            "image_requested": image_requested,
            "reason": "requests unavailable",
            "uploaded": [],
            "upload_failures": ["requests unavailable"] if images else [],
            "stdout_fallback": False,
        }

    text_result = {
        "ok": not text_requested,
        "status_code": None,
        "body": "",
        "push_url": "",
        "line_status": None,
    }
    if text_requested:
        text_result = _push_messages(worker_url, {"text": text, "imageUrls": [], "imageCaption": image_caption})

    uploaded: List[str] = []
    upload_failures: List[str] = []
    upload_url = ""
    upload_error = ""
    if images:
        uploaded, upload_failures, upload_url, upload_error = _upload_images(worker_url, images)

    image_result = {
        "ok": not image_requested,
        "status_code": None,
        "body": "",
        "push_url": "",
        "line_status": None,
    }
    partial_image_ok = False
    if images and uploaded:
        image_result = _push_messages(worker_url, {"text": "", "imageUrls": uploaded, "imageCaption": image_caption})
        partial_image_ok = bool(image_result.get("ok")) and bool(uploaded)
    elif image_requested:
        image_result = {
            "ok": False,
            "status_code": None,
            "body": upload_error or "no_images_uploaded",
            "push_url": "",
            "line_status": None,
        }

    text_ok = bool(text_result.get("ok")) if text_requested else True
    image_ok = True
    if image_requested:
        image_ok = bool(image_result.get("ok")) and len(uploaded) == len(images)

    any_delivery = False
    if text_requested and text_ok:
        any_delivery = True
    if image_requested and partial_image_ok:
        any_delivery = True
    if not text_requested and not image_requested:
        any_delivery = True

    ok = any_delivery
    if force_text and not text_ok:
        ok = False
    if force_image and not image_ok:
        ok = False

    reasons: List[str] = []
    if text_requested and not text_ok:
        reasons.append(f"text_push_failed:{text_result.get('status_code')}")
    if image_requested and not image_ok:
        if upload_failures:
            reasons.extend(upload_failures)
        elif image_result.get("body"):
            reasons.append(f"image_push_failed:{image_result.get('body')}")
    if not reasons:
        reasons.append("ok")

    return {
        "ok": ok,
        "text_ok": text_ok,
        "image_ok": image_ok,
        "partial_image_ok": partial_image_ok,
        "text_requested": text_requested,
        "image_requested": image_requested,
        "requested_images": len(images),
        "uploaded_images": len(uploaded),
        "uploaded": uploaded,
        "upload_failures": upload_failures,
        "upload_url": upload_url,
        "upload_error": upload_error,
        "text_status_code": text_result.get("status_code"),
        "image_status_code": image_result.get("status_code"),
        "text_push_url": text_result.get("push_url"),
        "image_push_url": image_result.get("push_url"),
        "text_body": str(text_result.get("body") or "")[:500],
        "image_body": str(image_result.get("body") or "")[:500],
        "line_text_status": text_result.get("line_status"),
        "line_image_status": image_result.get("line_status"),
        "reason": "; ".join(reasons),
        "worker_base_candidates": _resolve_worker_endpoints(worker_url).get("base", []),
        "worker_push_candidates": _resolve_worker_endpoints(worker_url).get("push", []),
        "worker_upload_candidates": _resolve_worker_endpoints(worker_url).get("upload", []),
        "stdout_fallback": False,
    }


send = send_line
send_line_message = send_line
