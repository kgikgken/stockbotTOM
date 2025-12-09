from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ============================================================
# 内部ヘルパー
# ============================================================

def _safe_last(series: pd.Series, default: float = np.nan) -> float:
    if series is None or len(series) == 0:
        return float(default)
    v = series.iloc[-1]
    try:
        v = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _calc_vola20(close: pd.Series) -> float:
    if close is None or len(close) < 21:
        return float("nan")
    ret = close.pct_change(fill_method=None)
    v = ret.rolling(20).std().iloc[-1]
    try:
        v = float(v)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


def _calc_rsi14(close: pd.Series) -> float:
    if close is None or len(close) <= 15:
        return float("nan")
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    gain = up.rolling(14).mean()
    loss = down.rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    try:
        v = float(v)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


def _calc_off_high_pct(close: pd.Series) -> float:
    """60日高値からの位置（％）"""
    if close is None or len(close) < 60:
        return float("nan")
    tail = close.tail(60)
    rolling_high = float(tail.max())
    last = float(tail.iloc[-1])
    if rolling_high <= 0:
        return float("nan")
    return (last / rolling_high - 1.0) * 100.0


def _base_tp_sl_from_vola(vola: float) -> Tuple[float, float]:
    """
    ボラからTP/SLの素を決める（％）
    戻り値: (tp_pct, sl_pct<0)
    """
    if not np.isfinite(vola):
        # データ不足時のデフォルト
        return 10.0, -4.0

    # ざっくり：ボラ小さい → tpもslも小さめ
    if vola < 0.015:
        return 7.0, -3.0
    if vola < 0.03:
        return 10.0, -4.0
    if vola < 0.06:
        return 13.0, -5.0
    # 超ボラ高
    return 18.0, -7.0


def _completion_factor(
    rsi: float,
    off_high: float,
    mkt_score: int,
    vola: float,
) -> float:
    """
    波が「ちゃんと予定通り完成するか」の係数。
    0.6〜1.3 の間にクリップ。
    """
    f = 1.0

    # RSI：程よい押し目は完成しやすい
    if np.isfinite(rsi):
        if 35 <= rsi <= 55:
            f += 0.15
        elif 30 <= rsi < 35 or 55 < rsi <= 60:
            f += 0.05
        elif rsi < 25 or rsi > 70:
            f -= 0.15

    # 高値からの下げ幅
    if np.isfinite(off_high):
        # -20〜-5% の押しは完成しやすい
        if -20 <= off_high <= -5:
            f += 0.15
        # 高値更新直後や深すぎる押しは失敗しやすい
        elif off_high > 5 or off_high < -30:
            f -= 0.1

    # 地合い
    if mkt_score >= 70:
        f += 0.1
    elif mkt_score < 45:
        f -= 0.15

    # ボラ：高すぎるとブレやすい / 低すぎてもダマシが増える
    if np.isfinite(vola):
        if vola > 0.06:
            f -= 0.1
        elif vola < 0.015:
            f += 0.05

    f = float(np.clip(f, 0.6, 1.3))
    return f


# ============================================================
# 公開API
# ============================================================

def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> Tuple[float, float, float]:
    """
    TP/SL/RR を決める中枢ロジック（％・R）

    戻り値:
      tp_pct: 利確目安（+◯％）
      sl_pct: 損切り目安（-◯％）
      rr:     期待RR（◯R）
    """
    if hist is None or len(hist) < 40:
        # データが少ない銘柄は標準パラメータ
        return 8.0, -4.0, 2.0

    close = hist["Close"].astype(float)

    vola = _calc_vola20(close)
    rsi = _calc_rsi14(close)
    off_high = _calc_off_high_pct(close)

    tp_pct, sl_pct = _base_tp_sl_from_vola(vola)
    if sl_pct >= 0:
        sl_pct = -4.0

    rr_raw = tp_pct / abs(sl_pct) if abs(sl_pct) > 1e-9 else 2.0

    cfac = _completion_factor(
        rsi=rsi,
        off_high=off_high,
        mkt_score=int(mkt_score),
        vola=vola,
    )

    rr = rr_raw * cfac

    # 軽くクリップ（あまりにバカでかいRRは信用しない）
    rr = float(np.clip(rr, 1.0, 5.0))

    return float(tp_pct), float(sl_pct), float(round(rr, 2))