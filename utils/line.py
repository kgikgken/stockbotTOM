from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import ParseResult, urlparse, urlunparse


try:
    import requests
except Exception:  # pragma: no cover - requests should exist in runtime
    requests = None  # type: ignore


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        clean = str(value).strip()
        if clean:
            return clean
    return ""


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _replace_path(parsed: ParseResult, new_path: str) -> str:
    safe_path = new_path if new_path.startswith("/") else f"/{new_path}" if new_path else "/"
    return urlunparse(parsed._replace(path=safe_path, params="", query="", fragment=""))


def _worker_url() -> str:
    return _env_first("WORKER_URL", "WORKER_BASE_URL", "PUBLIC_BASE_URL")


def _worker_auth_token() -> str:
    return _env_first(
        "WORKER_AUTH_TOKEN",
        "WORKER_TOKEN",
        "UPLOAD_TOKEN",
        "AUTH_TOKEN",
        "PUSH_TOKEN",
    )


def _resolve_worker_endpoints(worker_url: str) -> Dict[str, List[str]]:
    raw = str(worker_url or "").strip()
    if not raw:
        return {"push": [], "upload": [], "base": [], "legacy": []}
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return {"push": [], "upload": [], "base": [], "legacy": []}

    path = (parsed.path or "").rstrip("/")
    terminal = ""
    base_path = path
    for suffix in ("/push", "/upload", "/health"):
        if path.endswith(suffix):
            terminal = suffix[1:]
            base_path = path[: -len(suffix)]
            break

    push_paths: List[str] = []
    upload_paths: List[str] = []
    base_urls: List[str] = []
    legacy_urls: List[str] = []

    if terminal == "push":
        push_paths.append(path or "/push")
    if terminal == "upload":
        upload_paths.append(path or "/upload")

    if base_path:
        push_paths.append(f"{base_path}/push")
        upload_paths.append(f"{base_path}/upload")
        base_urls.append(_replace_path(parsed, base_path))
        legacy_urls.append(_replace_path(parsed, base_path))
    else:
        base_urls.append(_replace_path(parsed, "/"))
        legacy_urls.append(_replace_path(parsed, "/"))

    push_paths.append("/push")
    upload_paths.append("/upload")
    legacy_urls.append(_replace_path(parsed, "/"))

    if path and terminal not in {"push", "upload", "health"}:
        push_paths.append(path)
        base_urls.append(_replace_path(parsed, path))
        legacy_urls.append(_replace_path(parsed, path))

    return {
        "push": [_replace_path(parsed, p) for p in _dedupe_keep_order(push_paths)],
        "upload": [_replace_path(parsed, p) for p in _dedupe_keep_order(upload_paths)],
        "base": _dedupe_keep_order(base_urls),
        "legacy": _dedupe_keep_order(legacy_urls),
    }


def _pick_response_text(resp) -> str:
    try:
        return (resp.text or "")[:500]
    except Exception:
        return ""


def _worker_auth_headers() -> Dict[str, str]:
    auth = _worker_auth_token()
    if not auth:
        return {}
    return {
        "Authorization": f"Bearer {auth}",
        "X-Auth-Token": auth,
    }


def _line_api_base_url() -> str:
    return _env_first("LINE_API_BASE_URL") or "https://api.line.me"


def _line_credentials() -> Tuple[str, str]:
    token = _env_first(
        "LINE_CHANNEL_ACCESS_TOKEN",
        "LINE_TOKEN",
        "LINE_ACCESS_TOKEN",
        "CHANNEL_ACCESS_TOKEN",
    )
    to = _env_first(
        "LINE_USER_ID",
        "LINE_TO",
        "LINE_TARGET_ID",
        "TARGET_ID",
        "USER_ID",
        "TO",
    )
    return token, to


def _line_direct_available() -> bool:
    token, to = _line_credentials()
    return bool(token and to and requests is not None)


def _parse_worker_response(resp) -> Tuple[bool, object, str]:
    body = _pick_response_text(resp)
    try:
        parsed = resp.json()
    except Exception:
        parsed = None

    if 200 <= int(getattr(resp, "status_code", 0) or 0) < 300:
        if isinstance(parsed, dict) and parsed.get("ok") is False:
            return False, parsed, str(parsed.get("error") or parsed.get("body") or body)
        return True, parsed, body

    if isinstance(parsed, dict):
        msg = str(parsed.get("error") or parsed.get("detail") or parsed.get("body") or body)
    else:
        msg = body
    return False, parsed, msg


def _post_first_success(
    urls: Sequence[str],
    *,
    json_payload=None,
    files=None,
    data=None,
    headers=None,
    timeout: int = 30,
):
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
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue

        ok, _parsed, msg = _parse_worker_response(resp)
        if ok:
            return resp, url, ""
        last_err = f"HTTP {resp.status_code}: {msg}"
    return last_resp, last_url, last_err


def _upload_images_v2(worker_url: str, image_paths: Sequence[str]) -> Tuple[List[str], List[str], str, str]:
    endpoints = _resolve_worker_endpoints(worker_url)
    upload_urls = endpoints.get("upload", [])
    headers = _worker_auth_headers()

    uploaded: List[str] = []
    failures: List[str] = []
    last_target = ""
    last_error = ""
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            failures.append(f"{p.name}: file_missing")
            continue

        success = False
        for upload_url in upload_urls:
            last_target = upload_url or last_target
            try:
                with p.open("rb") as f:
                    files = {"file": (p.name, f, "image/png")}
                    data = {"path": p.name}
                    resp = requests.post(upload_url, files=files, data=data, headers=headers, timeout=40)  # type: ignore[arg-type]
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            ok, parsed, msg = _parse_worker_response(resp)
            if not ok:
                last_error = f"HTTP {resp.status_code}: {msg}"
                continue

            url = ""
            if isinstance(parsed, dict):
                url = str(parsed.get("url") or "").strip()
            if not url:
                last_error = "missing_url"
                continue

            uploaded.append(url)
            success = True
            last_error = ""
            break

        if not success:
            failures.append(f"{p.name}: {last_error or 'upload_failed'}")
    return uploaded, failures, last_target, last_error


def _push_messages_v2(worker_url: str, payload: Dict, timeout: int = 30) -> Dict:
    endpoints = _resolve_worker_endpoints(worker_url)
    push_urls = endpoints.get("push", [])
    headers = _worker_auth_headers()
    resp, used_url, err = _post_first_success(push_urls, json_payload=payload, headers=headers, timeout=timeout)
    if resp is None:
        return {
            "ok": False,
            "status_code": None,
            "body": err,
            "push_url": used_url,
            "worker_ok": False,
            "line_ok": False,
            "mode": "worker_v2",
        }

    ok, parsed, msg = _parse_worker_response(resp)
    line_ok = ok
    line_status = None
    if isinstance(parsed, dict):
        if parsed.get("status") is not None:
            line_status = parsed.get("status")
        elif parsed.get("line_status") is not None:
            line_status = parsed.get("line_status")
        if parsed.get("ok") is not None:
            line_ok = bool(parsed.get("ok"))

    return {
        "ok": ok and line_ok,
        "status_code": resp.status_code,
        "line_status": line_status,
        "body": msg,
        "push_url": used_url,
        "worker_ok": ok,
        "line_ok": line_ok,
        "raw_json": parsed if isinstance(parsed, dict) else None,
        "mode": "worker_v2",
    }


def _push_legacy_text(worker_url: str, text: str, timeout: int = 30) -> Dict:
    endpoints = _resolve_worker_endpoints(worker_url)
    legacy_urls = endpoints.get("legacy", []) or endpoints.get("base", [])
    headers = _worker_auth_headers()
    resp, used_url, err = _post_first_success(legacy_urls, json_payload={"text": text}, headers=headers, timeout=timeout)
    if resp is None:
        return {
            "ok": False,
            "status_code": None,
            "body": err,
            "push_url": used_url,
            "line_status": None,
            "mode": "worker_legacy",
        }

    ok, parsed, msg = _parse_worker_response(resp)
    line_status = resp.status_code
    if isinstance(parsed, dict) and parsed.get("status") is not None:
        line_status = parsed.get("status")

    return {
        "ok": ok,
        "status_code": resp.status_code,
        "body": msg,
        "push_url": used_url,
        "line_status": line_status,
        "raw_json": parsed if isinstance(parsed, dict) else None,
        "mode": "worker_legacy",
    }


def _legacy_image_post(
    worker_url: str,
    path: str,
    *,
    text: str = "",
    timeout: int = 40,
) -> Dict:
    endpoints = _resolve_worker_endpoints(worker_url)
    legacy_urls = endpoints.get("legacy", []) or endpoints.get("base", [])
    headers = _worker_auth_headers()
    p = Path(path)
    if not p.exists():
        return {
            "ok": False,
            "status_code": None,
            "body": "file_missing",
            "push_url": "",
            "line_status": None,
            "mode": "worker_legacy",
            "url": "",
        }

    last_err = ""
    last_url = ""
    last_resp = None
    for legacy_url in legacy_urls:
        last_url = legacy_url
        try:
            with p.open("rb") as f:
                files = {"image": (p.name, f, "image/png")}
                data = {"key": p.name}
                if text.strip():
                    data["text"] = text
                resp = requests.post(legacy_url, files=files, data=data, headers=headers, timeout=timeout)  # type: ignore[arg-type]
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue

        last_resp = resp
        ok, parsed, msg = _parse_worker_response(resp)
        if ok:
            image_url = ""
            if isinstance(parsed, dict):
                image_url = str(parsed.get("url") or "").strip()
            return {
                "ok": True,
                "status_code": resp.status_code,
                "body": msg,
                "push_url": legacy_url,
                "line_status": resp.status_code,
                "mode": "worker_legacy",
                "raw_json": parsed if isinstance(parsed, dict) else None,
                "url": image_url,
            }
        last_err = f"HTTP {resp.status_code}: {msg}"

    status_code = None if last_resp is None else last_resp.status_code
    return {
        "ok": False,
        "status_code": status_code,
        "body": last_err,
        "push_url": last_url,
        "line_status": status_code,
        "mode": "worker_legacy",
        "url": "",
    }


def _send_images_legacy(worker_url: str, image_paths: Sequence[str], *, text: str = "") -> Dict:
    uploaded: List[str] = []
    failures: List[str] = []
    results: List[Dict] = []
    text_ok = not bool(text.strip())
    first_text = text if text.strip() else ""

    for idx, path in enumerate(image_paths):
        payload_text = first_text if idx == 0 and first_text else ""
        result = _legacy_image_post(worker_url, path, text=payload_text)
        results.append(result)
        if result.get("ok"):
            url = str(result.get("url") or "").strip()
            if url:
                uploaded.append(url)
            if payload_text:
                text_ok = True
        else:
            failures.append(f"{Path(path).name}: {result.get('body') or result.get('status_code') or 'legacy_upload_failed'}")

    partial_ok = any(bool(r.get("ok")) for r in results)
    image_ok = partial_ok and len([r for r in results if r.get("ok")]) == len(image_paths)
    last = results[-1] if results else {
        "status_code": None,
        "push_url": "",
        "body": "",
        "mode": "worker_legacy",
    }
    return {
        "ok": partial_ok,
        "text_ok": text_ok,
        "image_ok": image_ok,
        "partial_image_ok": partial_ok,
        "uploaded": uploaded,
        "upload_failures": failures,
        "status_code": last.get("status_code"),
        "push_url": last.get("push_url"),
        "body": last.get("body"),
        "mode": "worker_legacy",
        "details": results,
    }


def _build_text_messages(text: str) -> List[Dict[str, str]]:
    clean = str(text or "")
    if not clean.strip():
        return []
    return [{"type": "text", "text": clean[:5000]}]


def _build_image_messages(image_urls: Sequence[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for url in list(image_urls)[:5]:
        clean = str(url or "").strip()
        if not clean:
            continue
        messages.append(
            {
                "type": "image",
                "originalContentUrl": clean,
                "previewImageUrl": clean,
            }
        )
    return messages


def _direct_push(messages: Sequence[Dict[str, str]], timeout: int = 30) -> Dict:
    token, to = _line_credentials()
    if requests is None:
        return {
            "ok": False,
            "status_code": None,
            "body": "requests unavailable",
            "push_url": "",
            "line_status": None,
            "mode": "direct",
        }
    if not token or not to:
        return {
            "ok": False,
            "status_code": None,
            "body": "LINE credentials missing",
            "push_url": "",
            "line_status": None,
            "mode": "direct",
        }
    if not messages:
        return {
            "ok": True,
            "status_code": 200,
            "body": "no_messages",
            "push_url": f"{_line_api_base_url().rstrip('/')}/v2/bot/message/push",
            "line_status": 200,
            "mode": "direct",
        }

    url = f"{_line_api_base_url().rstrip('/')}/v2/bot/message/push"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
    }
    try:
        resp = requests.post(url, json={"to": to, "messages": list(messages)}, headers=headers, timeout=timeout)
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "body": f"{type(exc).__name__}: {exc}",
            "push_url": url,
            "line_status": None,
            "mode": "direct",
        }

    return {
        "ok": 200 <= resp.status_code < 300,
        "status_code": resp.status_code,
        "body": _pick_response_text(resp),
        "push_url": url,
        "line_status": resp.status_code,
        "mode": "direct",
    }


def send_line(
    text: str = "",
    image_paths: Iterable[str] | None = None,
    image_caption: str = "",
    force_image: bool = False,
    force_text: bool = False,
) -> Dict:
    worker_url = _worker_url()
    images = [str(p) for p in (image_paths or []) if str(p).strip()]
    text = str(text or "")
    text_requested = bool(text.strip() or force_text)
    image_requested = bool(images or force_image)

    worker_available = bool(worker_url)
    direct_available = _line_direct_available()

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
            "worker_available": worker_available,
            "direct_available": direct_available,
        }

    text_result = {
        "ok": not text_requested,
        "status_code": None,
        "body": "",
        "push_url": "",
        "line_status": None,
        "mode": "none",
    }
    text_fallback_result = None
    text_legacy_result = None

    if text_requested:
        if worker_available:
            text_result = _push_messages_v2(worker_url, {"text": text, "imageUrls": [], "imageCaption": image_caption})
            if not bool(text_result.get("ok")):
                text_legacy_result = _push_legacy_text(worker_url, text)
                if bool(text_legacy_result.get("ok")):
                    text_result = text_legacy_result
        if (not bool(text_result.get("ok"))) and direct_available:
            text_fallback_result = _direct_push(_build_text_messages(text))
            if bool(text_fallback_result.get("ok")):
                text_result = text_fallback_result

    uploaded: List[str] = []
    upload_failures: List[str] = []
    upload_url = ""
    upload_error = ""
    image_result = {
        "ok": not image_requested,
        "status_code": None,
        "body": "",
        "push_url": "",
        "line_status": None,
        "mode": "none",
    }
    image_fallback_result = None
    legacy_image_result = None
    partial_image_ok = False

    if images and worker_available:
        uploaded, upload_failures, upload_url, upload_error = _upload_images_v2(worker_url, images)
        if uploaded:
            image_result = _push_messages_v2(worker_url, {"text": "", "imageUrls": uploaded, "imageCaption": image_caption})
            if (not bool(image_result.get("ok"))) and direct_available:
                image_fallback_result = _direct_push(_build_image_messages(uploaded))
                if bool(image_fallback_result.get("ok")):
                    image_result = image_fallback_result
            partial_image_ok = bool(image_result.get("ok")) and bool(uploaded)
        if not partial_image_ok:
            legacy_text = text if text_requested and not bool(text_result.get("ok")) else ""
            legacy_image_result = _send_images_legacy(worker_url, images, text=legacy_text)
            if bool(legacy_image_result.get("ok")):
                image_result = {
                    "ok": legacy_image_result.get("ok"),
                    "status_code": legacy_image_result.get("status_code"),
                    "body": legacy_image_result.get("body"),
                    "push_url": legacy_image_result.get("push_url"),
                    "line_status": legacy_image_result.get("status_code"),
                    "mode": "worker_legacy",
                }
                partial_image_ok = bool(legacy_image_result.get("partial_image_ok"))
                uploaded = list(legacy_image_result.get("uploaded") or uploaded)
                upload_failures = list(legacy_image_result.get("upload_failures") or [])
                if legacy_text and bool(legacy_image_result.get("text_ok")):
                    text_result = {
                        "ok": True,
                        "status_code": legacy_image_result.get("status_code"),
                        "body": legacy_image_result.get("body"),
                        "push_url": legacy_image_result.get("push_url"),
                        "line_status": legacy_image_result.get("status_code"),
                        "mode": "worker_legacy",
                    }
    elif images:
        upload_error = "WORKER_URL missing"
        upload_failures = [f"{Path(p).name}: WORKER_URL missing" for p in images]

    text_ok = bool(text_result.get("ok")) if text_requested else True
    image_ok = True
    if image_requested:
        if legacy_image_result is not None and bool(legacy_image_result.get("ok")):
            image_ok = bool(legacy_image_result.get("image_ok"))
            partial_image_ok = bool(legacy_image_result.get("partial_image_ok"))
        else:
            image_ok = bool(image_result.get("ok")) and len(uploaded) == len(images)
            if not partial_image_ok:
                partial_image_ok = bool(image_result.get("ok")) and bool(uploaded)

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
        reasons.append(f"text_push_failed:{text_result.get('status_code')}:{text_result.get('body')}")
    if image_requested and not image_ok:
        if upload_failures:
            reasons.extend(upload_failures)
        elif image_result.get("body"):
            reasons.append(f"image_push_failed:{image_result.get('status_code')}:{image_result.get('body')}")
    if not ok:
        if not worker_available:
            reasons.append("worker_missing")
        if not direct_available:
            reasons.append("direct_line_unavailable")
    if not reasons:
        reasons.append("ok")

    if not ok and text.strip():
        print(text)

    endpoints = _resolve_worker_endpoints(worker_url)
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
        "text_mode": text_result.get("mode"),
        "image_mode": image_result.get("mode"),
        "text_fallback": text_fallback_result,
        "text_legacy": text_legacy_result,
        "image_fallback": image_fallback_result,
        "legacy_image": legacy_image_result,
        "reason": "; ".join(_dedupe_keep_order([r for r in reasons if r])),
        "worker_available": worker_available,
        "direct_available": direct_available,
        "worker_auth_used": bool(_worker_auth_token()),
        "worker_base_candidates": endpoints.get("base", []),
        "worker_legacy_candidates": endpoints.get("legacy", []),
        "worker_push_candidates": endpoints.get("push", []),
        "worker_upload_candidates": endpoints.get("upload", []),
        "stdout_fallback": bool(not ok and text.strip()),
    }


def summarize_line_result(result: Dict) -> Dict:
    keys = [
        "ok",
        "text_ok",
        "image_ok",
        "partial_image_ok",
        "reason",
        "text_mode",
        "image_mode",
        "text_status_code",
        "image_status_code",
        "text_push_url",
        "image_push_url",
        "upload_url",
        "worker_available",
        "direct_available",
        "worker_auth_used",
    ]
    return {key: result.get(key) for key in keys}


send = send_line
send_line_message = send_line
