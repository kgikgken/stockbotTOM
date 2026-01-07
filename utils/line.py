# ============================================
# utils/line.py
# LINE Notify 送信ユーティリティ
# - 失敗しにくい最小構成
# - 1メッセージ完結（分割なし）
# ============================================

import os
import requests
import sys
from typing import Optional


# --------------------------------------------
# 環境変数
# --------------------------------------------
LINE_NOTIFY_TOKEN = os.getenv("LINE_NOTIFY_TOKEN")
LINE_NOTIFY_API = "https://notify-api.line.me/api/notify"


# --------------------------------------------
# 送信本体
# --------------------------------------------
def send_line_message(message: str, silent: bool = False) -> bool:
    """
    LINE Notify にメッセージを送信する
    戻り値:
        True  -> 成功
        False -> 失敗
    """

    if not LINE_NOTIFY_TOKEN:
        print("[LINE] ERROR: LINE_NOTIFY_TOKEN is not set", file=sys.stderr)
        return False

    headers = {
        "Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"
    }

    payload = {
        "message": message
    }

    if silent:
        payload["notificationDisabled"] = True

    try:
        response = requests.post(
            LINE_NOTIFY_API,
            headers=headers,
            data=payload,
            timeout=10,
        )

        if response.status_code != 200:
            print(
                f"[LINE] ERROR: status={response.status_code}, body={response.text}",
                file=sys.stderr,
            )
            return False

        return True

    except Exception as e:
        print(f"[LINE] EXCEPTION: {e}", file=sys.stderr)
        return False


# --------------------------------------------
# 安全ラッパー（main から呼ぶ用）
# --------------------------------------------
def send_daily_report(report_text: str) -> None:
    """
    日報送信専用ラッパー
    失敗しても例外は投げない（ワークフローを落とさない）
    """
    ok = send_line_message(report_text)
    if not ok:
        print("[LINE] Failed to send daily report", file=sys.stderr)