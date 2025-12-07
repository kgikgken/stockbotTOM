#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests


# ======================================
# Worker URL（GitHub Secrets: WORKER_URL）
# ======================================
WORKER_URL = os.getenv("WORKER_URL")


# ======================================
# LINE通知関数
# ======================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定")
        print(text)
        return
    try:
        requests.post(
            WORKER_URL,
            json={"text": text},
            timeout=10,
        )
    except Exception as e:
        print(f"[ERROR] send_line failed: {e}")
        print(text)


# ======================================
# メイン処理
# ======================================
def main():
    # ===========================
    # ここにあなたの本体ロジックを入れる
    # ===========================

    # ↓ とりあえず仮
    # final_text に「日報全文」を入れる
    final_text = "TEST from main.py"

    # ===========================
    # LINE送信
    # ===========================
    send_line(final_text)


if __name__ == "__main__":
    main()