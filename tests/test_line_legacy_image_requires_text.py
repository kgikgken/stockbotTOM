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

from utils.line import send_line


class _LegacyTextRequiredHandler(BaseHTTPRequestHandler):
    state: Dict[str, int] = {
        "upload_calls": 0,
        "push_calls": 0,
        "base_json_calls": 0,
        "legacy_multipart_calls": 0,
        "legacy_multipart_text_calls": 0,
        "img_calls": 0,
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
            self.__class__.state["img_calls"] += 1
            body = b"PNGDATA"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json({"ok": False, "error": "not_found", "host": host}, 404)

    def do_POST(self):  # noqa: N802
        host = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        if self.path == "/upload":
            self.__class__.state["upload_calls"] += 1
            if b"PNGDATA" not in body:
                self._json({"ok": False, "error": "missing_bytes"}, 400)
                return
            self._json({"ok": True, "key": "reports/test.png", "url": f"{host}/img/reports%2Ftest.png"}, 200)
            return

        if self.path == "/push":
            self.__class__.state["push_calls"] += 1
            self._json({"ok": False, "error": "not_found"}, 404)
            return

        if self.path == "/" and content_type.startswith("application/json"):
            self.__class__.state["base_json_calls"] += 1
            payload = json.loads(body.decode("utf-8") or "{}")
            if payload.get("imageUrls"):
                self._json({"ok": False, "error": "No text"}, 400)
                return
            if str(payload.get("text") or "").strip():
                self._json({"ok": True, "status": 200, "body": "ok"}, 200)
                return
            self._json({"ok": False, "error": "No text"}, 400)
            return

        if self.path == "/" and content_type.startswith("multipart/form-data"):
            self.__class__.state["legacy_multipart_calls"] += 1
            if b"PNGDATA" not in body:
                self._json({"ok": False, "error": "missing_bytes"}, 400)
                return
            if b'name="text"' not in body:
                self._json({"ok": False, "error": "No text"}, 400)
                return
            self.__class__.state["legacy_multipart_text_calls"] += 1
            self._json({"ok": True, "key": "reports/test.png", "url": f"{host}/img/reports%2Ftest.png"}, 200)
            return

        self._json({"ok": False, "error": "not_found"}, 404)


def _start_server(handler) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


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


def test_image_push_prefers_legacy_multipart_for_base_worker_url() -> None:
    server, thread = _start_server(_LegacyTextRequiredHandler)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        tmp.write(b"PNGDATA")
        tmp.flush()
        tmp.close()
        with _with_env({"WORKER_URL": host}):
            result = send_line("", image_paths=[tmp.name], force_image=True)
        if not result.get("ok"):
            raise AssertionError(result)
        if result.get("image_mode") != "worker_legacy":
            raise AssertionError(result)
        if result.get("uploaded_images") != 1:
            raise AssertionError(result)
        if result.get("reason") != "ok":
            raise AssertionError(result)
        if _LegacyTextRequiredHandler.state["push_calls"] != 0:
            raise AssertionError(_LegacyTextRequiredHandler.state)
        if _LegacyTextRequiredHandler.state["base_json_calls"] != 0:
            raise AssertionError(_LegacyTextRequiredHandler.state)
        if _LegacyTextRequiredHandler.state["upload_calls"] != 0:
            raise AssertionError(_LegacyTextRequiredHandler.state)
        if _LegacyTextRequiredHandler.state["legacy_multipart_calls"] != 2:
            raise AssertionError(_LegacyTextRequiredHandler.state)
        if _LegacyTextRequiredHandler.state["legacy_multipart_text_calls"] != 1:
            raise AssertionError(_LegacyTextRequiredHandler.state)
    finally:
        os.unlink(tmp.name)
        server.shutdown()
        thread.join(timeout=2)


if __name__ == "__main__":
    test_image_push_prefers_legacy_multipart_for_base_worker_url()
    print({"ok": True, "state": _LegacyTextRequiredHandler.state})
