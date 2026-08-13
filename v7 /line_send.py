"""LINE配信 — Cloudflare Worker 経由.

プロトコル(src/worker.js と対・自立版: 旧mispricingから逐語移植):
  POST {WORKER_URL}/upload   multipart(file=PNG, caption任意) → R2保存 → LINEへimage push
  POST {WORKER_URL}/         JSON {"text": ...}              → LINEへtext push
認証: WORKER_AUTH_TOKEN があれば Authorization: Bearer で送る(Worker側 UPLOAD_TOKEN/AUTH_TOKEN)。
LINE_DRY_RUN=1 で実送信せずpayloadを表示(配信経路のテスト用)。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def send_line(text: str = "", image_paths: List[str] | None = None,
              image_caption: str = "", **_kw) -> Dict:
    worker_url = os.getenv("WORKER_URL", "").rstrip("/")
    token = os.getenv("WORKER_AUTH_TOKEN", "").strip()
    dry = _truthy("LINE_DRY_RUN", False)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    result: Dict = {"ok": False, "text_ok": False, "image_ok": False,
                    "backend": "worker", "status_code": 0, "images": []}

    if dry:
        print("=== LINE_DRY_RUN ===")
        print(f"WORKER_URL={worker_url or '(未設定)'} auth={'yes' if token else 'no'}")
        for p in (image_paths or []):
            print(f"[dry] POST /upload  file={p} ({Path(p).stat().st_size/1024:.0f}KB) caption={image_caption!r}")
        if text:
            print(f"[dry] POST /  text({len(text)}字) 先頭200字:\n{text[:200]}")
        result.update(ok=True, text_ok=bool(text), image_ok=bool(image_paths),
                      backend="dry_run", skipped=True)
        return result

    if not worker_url:
        print(text or "(no content)")
        result.update(ok=True, text_ok=True, backend="stdout", skipped=True,
                      reason="WORKER_URL未設定")
        return result

    import requests

    sent_any_image = False
    for p in (image_paths or []):
        try:
            with open(p, "rb") as fh:
                r = requests.post(
                    f"{worker_url}/upload",
                    files={"file": (Path(p).name, fh, "image/png")},
                    data={"caption": image_caption} if image_caption else {},
                    headers=headers, timeout=60,
                )
            ok = 200 <= r.status_code < 300
            result["images"].append({"path": p, "status": r.status_code,
                                     "body": r.text[:200]})
            sent_any_image = sent_any_image or ok
            if not ok:
                print(f"[WARN] upload failed {r.status_code}: {r.text[:200]}")
        except Exception as e:
            result["images"].append({"path": p, "error": str(e)})
            print(f"[WARN] upload error: {e}")
    result["image_ok"] = sent_any_image

    if text and (not sent_any_image):
        try:
            r = requests.post(worker_url + "/", json={"text": text[:4800]},
                              headers=headers, timeout=30)
            result["status_code"] = r.status_code
            result["text_ok"] = 200 <= r.status_code < 300
            if not result["text_ok"]:
                print(f"[WARN] text push failed {r.status_code}: {r.text[:200]}")
        except Exception as e:
            result["reason"] = str(e)
            print(f"[WARN] text push error: {e}")

    result["ok"] = result["image_ok"] or result["text_ok"]
    if not result["ok"]:
        result.setdefault("reason", "image/text とも配信失敗")
    return result
