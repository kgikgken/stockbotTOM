# utils/util.py
# ============================================================
# 共通ユーティリティ
# ============================================================

from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional


# ============================================================
# JST 関連
# ============================================================
JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    """JST の現在時刻（datetime）"""
    return datetime.now(JST)


def jst_today_date() -> datetime.date:
    """JST の今日の日付（date）"""
    return jst_now().date()


def jst_today_str(fmt: str = "%Y-%m-%d") -> str:
    """JST 今日の日付を文字列で返す"""
    return jst_now().strftime(fmt)


def jst_time_str(fmt: str = "%H:%M:%S") -> str:
    """JST 時刻の文字列"""
    return jst_now().strftime(fmt)


# ============================================================
# 日付ユーティリティ
# ============================================================
def days_between(d1: datetime.date, d2: datetime.date) -> int:
    """日付の差（絶対値）"""
    try:
        return abs((d1 - d2).days)
    except Exception:
        return 9999


def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> Optional[datetime.date]:
    """安全な日付変換"""
    try:
        return datetime.strptime(date_str, fmt).date()
    except Exception:
        return None


# ============================================================
# 環境変数
# ============================================================
def env(key: str, default: str = "") -> str:
    """環境変数取得（空なら default）"""
    v = os.getenv(key, "")
    return v if v else default


# ============================================================
# 安全な数値変換
# ============================================================
def to_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        if not (x == x):  # NaN
            return default
        return x
    except Exception:
        return default


def to_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


# ============================================================
# フォーマット系
# ============================================================
def fmt_num(v: float, decimals: int = 1) -> str:
    try:
        return f"{v:.{decimals}f}"
    except Exception:
        return f"{v}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    """0.08 → +8.0%"""
    try:
        return f"{v*100:+.{decimals}f}%"
    except Exception:
        return "-"


def fmt_yen(v: float) -> str:
    """金額を3桁区切り"""
    try:
        return f"{int(round(v)):,}円"
    except Exception:
        return "-"


# ============================================================
# Core：RR計算（Rベース）
# ============================================================
def calc_rr(tp_pct: float, sl_pct: float) -> float:
    """
    RR = 期待利益 / 期待損失 = TP% / |SL%|
    R基準に統一
    """
    try:
        if sl_pct >= 0:
            return 0.0
        return float(tp_pct / abs(sl_pct))
    except Exception:
        return 0.0


# ============================================================
# テキスト整形
# ============================================================
def indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


# ============================================================
# 安全ログ
# ============================================================
def safe_print(*args, **kwargs):
    """
    GitHub Actions で化けないように安全出力
    """
    try:
        print(*args, **kwargs, flush=True)
    except Exception:
        pass