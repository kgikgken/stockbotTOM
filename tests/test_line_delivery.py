from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.line import _resolve_worker_endpoints, send_line


class _V2Handler(BaseHTTPRequestHandler):
    state: Dict[str, int] = {
        "push_calls": 0,
        "upload_calls": 0,
        "line_calls": 0,
    }

    def log_message(self, format, *args):  # noqa: A003
        return

    def _json(self, payload: Dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        path = self.path

        if path == "/bad/push":
            self._json({"ok": False, "error": "wrong_path"}, 404)
            return
        if path == "/bad/upload":
            self._json({"ok": False, "error": "wrong_path"}, 404)
            return
        if path == "/push":
            self.__class__.state["push_calls"] += 1
            if self.headers.get("Authorization") != "Bearer secret":
                self._json({"ok": False, "error": "unauthorized"}, 401)
                return
            payload = json.loads(body.decode("utf-8") or "{}")
            if not payload.get("text") and not payload.get("imageUrls"):
                self._json({"ok": True, "skipped": True}, 200)
                return
            self._json({"ok": True, "status": 200, "body": "ok"}, 200)
            return
        if path == "/upload":
            self.__class__.state["upload_calls"] += 1
            if self.headers.get("Authorization") != "Bearer secret":
                self._json({"ok": False, "error": "unauthorized"}, 401)
                return
            if b"PNGDATA" not in body:
                self._json({"ok": False, "error": "missing_bytes"}, 400)
                return
            host = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
            self._json({"ok": True, "url": f"{host}/img/test.png"}, 200)
            return
        if path == "/v2/bot/message/push":
            self.__class__.state["line_calls"] += 1
            if self.headers.get("Authorization") != "Bearer directtoken":
                self._json({"message": "bad token"}, 401)
                return
            payload = json.loads(body.decode("utf-8") or "{}")
            if payload.get("to") != "directuser":
                self._json({"message": "bad to"}, 400)
                return
            self._json({}, 200)
            return

        self._json({"ok": False, "error": "not_found"}, 404)




class _PrefixedLegacyLikeHandler(BaseHTTPRequestHandler):
    state: Dict[str, int] = {
        "push_calls": 0,
        "upload_calls": 0,
        "img_root_calls": 0,
        "img_prefixed_calls": 0,
    }

    def log_message(self, format, *args):  # noqa: A003
        return

    def _json(self, payload: Dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        host = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
        if self.path.startswith("/img/"):
            self.__class__.state["img_root_calls"] += 1
            self.send_response(404)
            self.end_headers()
            return
        if self.path.startswith("/stockbot/img/"):
            self.__class__.state["img_prefixed_calls"] += 1
            body = b"PNGDATA"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json({"ok": False, "error": "not_found", "host": host}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        host = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
        if self.path == "/stockbot/push":
            self.__class__.state["push_calls"] += 1
            if self.headers.get("Authorization") != "Bearer secret":
                self._json({"ok": False, "error": "unauthorized"}, 401)
                return
            payload = json.loads(body.decode("utf-8") or "{}")
            image_urls = payload.get("imageUrls") or []
            if image_urls:
                if any(not str(u).startswith(f"{host}/stockbot/img/") for u in image_urls):
                    self._json({"ok": False, "error": "bad_image_url", "imageUrls": image_urls}, 400)
                    return
            self._json({"ok": True, "status": 200, "body": "ok"}, 200)
            return
        if self.path == "/stockbot/upload":
            self.__class__.state["upload_calls"] += 1
            if self.headers.get("Authorization") != "Bearer secret":
                self._json({"ok": False, "error": "unauthorized"}, 401)
                return
            if b"PNGDATA" not in body:
                self._json({"ok": False, "error": "missing_bytes"}, 400)
                return
            self._json(
                {
                    "ok": True,
                    "key": "reports/test.png",
                    "url": f"{host}/img/reports%2Ftest.png",
                },
                200,
            )
            return
        self._json({"ok": False, "error": "not_found"}, 404)


class _LegacyHandler(BaseHTTPRequestHandler):
    state: Dict[str, int] = {
        "text_calls": 0,
        "upload_calls": 0,
    }

    def log_message(self, format, *args):  # noqa: A003
        return

    def _json(self, payload: Dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        content_type = self.headers.get("Content-Type", "")
        if not self.path.startswith("/"):
            self._json({"ok": False, "error": "bad_path"}, 404)
            return

        if content_type.startswith("application/json"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if str(payload.get("text") or "").strip():
                self.__class__.state["text_calls"] += 1
                self._json({"ok": True, "line": {}}, 200)
                return
            self._json({"ok": False, "error": "missing text"}, 400)
            return

        if content_type.startswith("multipart/form-data"):
            auth = self.headers.get("Authorization")
            xauth = self.headers.get("X-Auth-Token")
            if auth != "Bearer legacysecret" and xauth != "legacysecret":
                self._json({"ok": False, "error": "unauthorized"}, 401)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if b'name="image"' not in body:
                self._json({"ok": False, "error": "missing image"}, 400)
                return
            if b"PNGDATA" not in body:
                self._json({"ok": False, "error": "missing_bytes"}, 400)
                return
            self.__class__.state["upload_calls"] += 1
            host = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
            self._json({"ok": True, "key": "legacy/test.png", "url": f"{host}/img/legacy/test.png"}, 200)
            return

        self._json({"ok": False, "error": "not_found"}, 404)


def _start_server(handler) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _assert_contains(got, expected):
    if expected not in got:
        raise AssertionError(f"expected {expected!r} in {got!r}")


def _with_env(env: Dict[str, str]):
    class _EnvCtx:
        def __enter__(self):
            self._old = {k: os.environ.get(k) for k in env}
            for k, v in env.items():
                os.environ[k] = v
            return self

        def __exit__(self, exc_type, exc, tb):
            for k, old in self._old.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
            return False

    return _EnvCtx()


def test_endpoint_resolution() -> None:
    root = _resolve_worker_endpoints("https://bot.example.workers.dev")
    _assert_contains(root["push"], "https://bot.example.workers.dev/push")
    _assert_contains(root["upload"], "https://bot.example.workers.dev/upload")
    _assert_contains(root["legacy"], "https://bot.example.workers.dev/")

    full_push = _resolve_worker_endpoints("https://bot.example.workers.dev/push")
    _assert_contains(full_push["push"], "https://bot.example.workers.dev/push")
    _assert_contains(full_push["upload"], "https://bot.example.workers.dev/upload")
    _assert_contains(full_push["legacy"], "https://bot.example.workers.dev/")

    prefixed = _resolve_worker_endpoints("https://example.com/stockbot")
    _assert_contains(prefixed["push"], "https://example.com/stockbot/push")
    _assert_contains(prefixed["upload"], "https://example.com/stockbot/upload")
    _assert_contains(prefixed["legacy"], "https://example.com/stockbot")

    prefixed_full = _resolve_worker_endpoints("https://example.com/stockbot/push")
    _assert_contains(prefixed_full["push"], "https://example.com/stockbot/push")
    _assert_contains(prefixed_full["upload"], "https://example.com/stockbot/upload")
    _assert_contains(prefixed_full["legacy"], "https://example.com/stockbot")


def test_worker_auth_and_retry() -> None:
    server, thread = _start_server(_V2Handler)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        tmp.write(b"PNGDATA")
        tmp.flush()
        tmp.close()
        with _with_env(
            {
                "WORKER_URL": f"{host}/bad/push",
                "WORKER_AUTH_TOKEN": "secret",
            }
        ):
            result = send_line("hello", image_paths=[tmp.name], force_text=True)
        if not result.get("ok"):
            raise AssertionError(result)
        if result.get("text_mode") != "worker_v2":
            raise AssertionError(result)
        if result.get("image_mode") != "worker_v2":
            raise AssertionError(result)
        if result.get("text_push_url") != f"{host}/push":
            raise AssertionError(result)
        if result.get("uploaded_images") != 1:
            raise AssertionError(result)
        if result.get("reason") != "ok":
            raise AssertionError(result)
    finally:
        os.unlink(tmp.name)
        server.shutdown()
        thread.join(timeout=2)


def test_legacy_worker_compat_and_auth_alias() -> None:
    server, thread = _start_server(_LegacyHandler)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        for tmp in (tmp1, tmp2):
            tmp.write(b"PNGDATA")
            tmp.flush()
            tmp.close()
        with _with_env(
            {
                "WORKER_URL": host,
                "UPLOAD_TOKEN": "legacysecret",
            }
        ):
            result = send_line("legacy hello", image_paths=[tmp1.name, tmp2.name], force_text=True)
        if not result.get("ok"):
            raise AssertionError(result)
        if result.get("text_ok") is not True or result.get("image_ok") is not True:
            raise AssertionError(result)
        if result.get("image_mode") != "worker_legacy":
            raise AssertionError(result)
        if result.get("text_mode") not in {"worker_v2", "worker_legacy"}:
            raise AssertionError(result)
        if result.get("worker_auth_used") is not True:
            raise AssertionError(result)
        if result.get("upload_failures"):
            raise AssertionError(result)
    finally:
        for tmp in (tmp1, tmp2):
            os.unlink(tmp.name)
        server.shutdown()
        thread.join(timeout=2)




def test_prefixed_public_image_url_rewrite() -> None:
    server, thread = _start_server(_PrefixedLegacyLikeHandler)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        tmp.write(b"PNGDATA")
        tmp.flush()
        tmp.close()
        with _with_env(
            {
                "WORKER_URL": f"{host}/stockbot",
                "WORKER_AUTH_TOKEN": "secret",
            }
        ):
            result = send_line("hello", image_paths=[tmp.name], force_text=True)
        if not result.get("ok"):
            raise AssertionError(result)
        if result.get("uploaded_images") != 1:
            raise AssertionError(result)
        uploaded = list(result.get("uploaded") or [])
        expected = f"{host}/stockbot/img/reports%2Ftest.png"
        if uploaded != [expected]:
            raise AssertionError(result)
        if result.get("image_mode") != "worker_v2":
            raise AssertionError(result)
    finally:
        os.unlink(tmp.name)
        server.shutdown()
        thread.join(timeout=2)


def test_direct_line_fallback_without_worker() -> None:
    server, thread = _start_server(_V2Handler)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with _with_env(
            {
                "WORKER_URL": "",
                "LINE_API_BASE_URL": host,
                "LINE_ACCESS_TOKEN": "directtoken",
                "LINE_TARGET_ID": "directuser",
            }
        ):
            result = send_line("direct hello", force_text=True)
        if not result.get("ok"):
            raise AssertionError(result)
        if result.get("text_mode") != "direct":
            raise AssertionError(result)
        if result.get("text_push_url") != f"{host}/v2/bot/message/push":
            raise AssertionError(result)
    finally:
        server.shutdown()
        thread.join(timeout=2)


def main() -> None:
    test_endpoint_resolution()
    test_worker_auth_and_retry()
    test_legacy_worker_compat_and_auth_alias()
    test_direct_line_fallback_without_worker()
    print({"ok": True, "v2_state": _V2Handler.state, "legacy_state": _LegacyHandler.state})


if __name__ == "__main__":
    main()
