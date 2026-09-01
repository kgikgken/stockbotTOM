"""LINE への push（docs/SCREENER.md §4.1）。

既存の Cloudflare Worker（`src/worker.js`）をそのまま使う。Worker は
`POST /`（JSON `{"text": ...}`）でテキストを LINE に push する。**Worker・
`wrangler.toml`・Secrets 名は変更しない**（CLAUDE.md「LINE 経路を変えない」）。

環境変数:
- `WORKER_URL`        … Worker のエンドポイント。未設定なら送信せずスキップする
- `WORKER_AUTH_TOKEN` … 任意。Worker 側で `UPLOAD_TOKEN` を設定している場合に必要

送信の失敗はここでは握り潰さず、呼び出し側に結果を返す。ワークフロー上は配信ステップを
コミットより後ろに置いてあるので、ここが失敗しても screen と resolve の結果は残る（§4.4）。
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_TIMEOUT = 30


def worker_url() -> Optional[str]:
    url = os.getenv("WORKER_URL", "").strip()
    return url or None


def auth_token() -> Optional[str]:
    token = os.getenv("WORKER_AUTH_TOKEN", "").strip()
    return token or None


def push_text(text: str, url: Optional[str] = None, token: Optional[str] = None,
              timeout: int = DEFAULT_TIMEOUT, post=None) -> dict:
    """Worker 経由で LINE にテキストを push する。

    post を渡すとその関数を使う（テストでネットワークを使わないため。既定は
    requests.post の遅延 import）。

    戻り値: {"sent": bool, "status": HTTP ステータス or None, "reason": 説明}
    URL 未設定は「送らなかった」であって失敗ではないので sent=False / reason で返す。
    """
    url = url or worker_url()
    if not url:
        return {"sent": False, "status": None, "reason": "WORKER_URL 未設定のため送信しない"}
    if not text:
        return {"sent": False, "status": None, "reason": "本文が空のため送信しない"}

    if post is None:
        import requests  # 遅延 import（テストと DRYRUN では不要）

        post = requests.post

    headers = {"Content-Type": "application/json"}
    token = token if token is not None else auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = post(url, json={"text": text}, headers=headers, timeout=timeout)
    status = int(getattr(r, "status_code", 0))
    ok = 200 <= status < 300
    body = ""
    try:
        body = str(r.text)[:200]
    except Exception:
        pass
    return {"sent": ok, "status": status,
            "reason": "送信成功" if ok else f"Worker が {status} を返した: {body}"}
