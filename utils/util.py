from __future__ import annotations

from datetime import datetime, timedelta, timezone


# ============================================================
# JST日付文字列（例: "2025-12-05"）
# ============================================================
def jst_today_str() -> str:
    """
    Japan Standard Time ベースの「YYYY-MM-DD」文字列を返す。
    """
    dt = datetime.now(timezone(timedelta(hours=9)))
    return dt.strftime("%Y-%m-%d")


# ============================================================
# JST日付（dateオブジェクト）
# ============================================================
def jst_today_date():
    """
    JSTベースの date オブジェクトを返す。
    """
    dt = datetime.now(timezone(timedelta(hours=9)))
    return dt.date()


# ============================================================
# 安全なfloat化（NoneやNaN対応）
# ============================================================
def safe_float(v, default=0.0) -> float:
    """
    NaNやNoneが来ても安全にfloatに変換する。
    """
    try:
        f = float(v)
        if f != f:  # NaNチェック
            return default
        return f
    except Exception:
        return default


# ============================================================
# INTフォーマット（例: 1234567 → "1,234,567"）
# ============================================================
def fmt_int(v) -> str:
    """
    金額や株数などINTをカンマ区切りで返す。
    """
    try:
        return f"{int(v):,}"
    except Exception:
        return "-"


# ============================================================
# %表示（例: 0.053 → "+5.3%"）
# ============================================================
def fmt_pct(v) -> str:
    """
    0.0123 → "+1.2%"
    -0.045 → "-4.5%"
    None/NaN → "-"
    """
    try:
        f = float(v)
        if f != f:  # NaN
            return "-"
        return f"{f*100:+.1f}%"
    except Exception:
        return "-"