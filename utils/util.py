from __future__ import annotations

"""
共通ユーティリティ

※ここは「依存される側」なので慎重に。
　副作用ゼロ・日付関連・軽量関数だけ置く。
"""

import os
from datetime import datetime, timedelta, timezone


# ============================================================
# JST 今日の日付（文字列）
# ============================================================
def jst_now() -> datetime:
    """JST の現在時刻"""
    return datetime.now(timezone(timedelta(hours=9)))


def jst_today_str() -> str:
    """
    📅 2025-12-08 みたいな形式
    ※日報タイトルで使う
    """
    d = jst_now()
    return d.strftime("%Y-%m-%d")


def jst_today_date():
    """date オブジェクト（後で地合い補正などで使う）"""
    return jst_now().date()


# ============================================================
# ENV 読み取り（安全ラッパー）
# ============================================================
def env(name: str, default: str = "") -> str:
    """
    環境変数読み取り（None 時は default）
    ※Worker／Github Actions 両対応
    """
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip()


# ============================================================
# 数値 utilities
# ============================================================
def clamp(v: float, lo: float, hi: float) -> float:
    """lo <= v <= hi に収める"""
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def to_int(v: float, default: int = 0) -> int:
    """安全に int 化"""
    try:
        return int(round(v))
    except Exception:
        return default


# ============================================================
# フォーマット utilities
# ============================================================
def fmt_percent(x: float) -> str:
    """
    0.085 → '+8.5%'
    -0.022 → '-2.2%'
    """
    try:
        pct = float(x) * 100
        if pct >= 0:
            return f"+{pct:.1f}%"
        return f"{pct:.1f}%"
    except Exception:
        return "+0.0%"


def fmt_price(x: float) -> str:
    """価格を小数1桁に統一"""
    try:
        return f"{float(x):.1f}"
    except Exception:
        return "0.0"


# ============================================================
# RR 判定
# ============================================================
def rr_comment(rr: float) -> str:
    """
    RR の定性的評価（LINEの説明用）
    """
    try:
        r = float(rr)
    except Exception:
        return ""

    if r >= 3.0:
        return "RR非常に高い（本命波）"
    if r >= 2.0:
        return "RR高い（狙い目）"
    if r >= 1.5:
        return "RR普通（状況次第）"
    return "RR低い（除外推奨）"