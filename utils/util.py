from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional


# ============================================================
# JST
# ============================================================
JST = timezone(timedelta(hours=9))


# ============================================================
# 今日の日付 (YYYY-MM-DD)
# ============================================================
def jst_today_str() -> str:
    """
    例: "2025-12-08"
    日報タイトルやLINE通知の日付用。
    Cloudflare/GitHub ActionsはUTCのため必須。
    """
    return datetime.now(JST).date().strftime("%Y-%m-%d")


def jst_today_datetime() -> datetime:
    """
    JSTの現在日時を返す (datetime)
    """
    return datetime.now(JST)


# ============================================================
# フォーマットヘルパー
# ============================================================
def fmt_price(v: float, digits: int = 1) -> str:
    """
    数値を小数付きで整形
    digits=1 → 1381.0
    """
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "-"


def fmt_int(v: float) -> str:
    """
    数値を整数フォーマット
    """
    try:
        return f"{int(round(v)):,}"
    except Exception:
        return "-"


def fmt_pct(v: float, digits: int = 1) -> str:
    """
    0.1234 → "+12.3%"
    -0.055 → "-5.5%"
    """
    try:
        return f"{v*100:+.{digits}f}%"
    except Exception:
        return "-"


# ============================================================
# 環境変数
# ============================================================
def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    GitHub Actions / Cloudflare Worker の環境変数取得
    テスト時も default が使えるようにしてある。
    """
    return os.getenv(name, default)


# ============================================================
# LINE送信用の安全チェック
# ============================================================
def safe_text(text: str) -> str:
    """
    LINE用テキストで危険な制御文字や None を安全化。
    None → ""
    """
    if text is None:
        return ""
    return str(text).replace("\x00", "")


# ============================================================
# デバッグ用ログ
# ============================================================
def log(msg: str) -> None:
    """
    GitHub Actions / Cloudflare のログに出す。
    ローカルだと print
    """
    try:
        print(msg, flush=True)
    except Exception:
        pass