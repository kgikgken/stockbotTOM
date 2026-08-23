"""地合いゲージとブレス（DESIGN.md §8.1 / TASKS.md T-207）。

6点ゲージ（順位には使わない。表示・検証の集計用）:
- 指数（TOPIX、取れなければ日経225。フォールバックは data.yf_fetch.fetch_index が
  既に行う。ここでは解決済みの指数終値 Series を受け取るだけ）について
  Close>SMA25 / Close>SMA75 / Close>SMA200 / SMA75 上向き（15日前比） の4点
- ブレス: breadth_75 ≥ 0.5 / breadth_200 ≥ 0.5 の2点
- 6点中 5〜6 = 強、3〜4 = 中、0〜2 = 弱

条件が定義できない場合（SMA の warmup 未了など）はその点を「満たさない」（0点）と
扱う（欠測を有利にも不利にも倒さない安全側のデフォルト）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .indicators import sma

REGIME_STRONG = "強"
REGIME_MID = "中"
REGIME_WEAK = "弱"

SLOPE_LOOKBACK = 15  # DESIGN.md §8.1: SMA75 上向き（15 日前比）


def compute_breadth(ohlcv: Dict[str, pd.DataFrame], asof: pd.Timestamp) -> tuple[float, float, int]:
    """ユニバース全体のブレス（DESIGN.md §5 D8 breadth_75 / breadth_200）。

    各銘柄は自身の履歴だけで SMA75/SMA200 を計算し、asof 以前の最終営業日の値で
    判定する（未来参照なし）。SMA が定義できない銘柄（履歴不足）は分母から除く。

    戻り値: (breadth_75, breadth_200, n_counted_200)。対象銘柄が無い場合は NaN。
    """
    above75 = above200 = counted75 = counted200 = 0
    for df in ohlcv.values():
        if df is None or len(df) == 0:
            continue
        sub = df[df.index <= asof]
        if len(sub) == 0:
            continue
        close = sub["Close"]
        c = float(close.iloc[-1])
        s75 = sma(close, 75).iloc[-1]
        s200 = sma(close, 200).iloc[-1]
        if not np.isnan(s75):
            counted75 += 1
            if c > s75:
                above75 += 1
        if not np.isnan(s200):
            counted200 += 1
            if c > s200:
                above200 += 1
    breadth_75 = (above75 / counted75) if counted75 else np.nan
    breadth_200 = (above200 / counted200) if counted200 else np.nan
    return breadth_75, breadth_200, counted200


def regime_gauge(idx_close: pd.Series, t_pos: int, breadth_75: float,
                 breadth_200: float) -> dict:
    """DESIGN.md §8.1 の6点ゲージ。t_pos は idx_close 上の位置（0始まり）。

    戻り値: {"score": 0-6, "level": 強/中/弱, 各条件の bool}
    """
    c_arr = idx_close.to_numpy(dtype=float)
    sma25 = sma(idx_close, 25).to_numpy(dtype=float)
    sma75 = sma(idx_close, 75).to_numpy(dtype=float)
    sma200 = sma(idx_close, 200).to_numpy(dtype=float)

    c = c_arr[t_pos]
    points = {
        "close_gt_sma25": bool(not np.isnan(sma25[t_pos]) and c > sma25[t_pos]),
        "close_gt_sma75": bool(not np.isnan(sma75[t_pos]) and c > sma75[t_pos]),
        "close_gt_sma200": bool(not np.isnan(sma200[t_pos]) and c > sma200[t_pos]),
    }
    if t_pos - SLOPE_LOOKBACK >= 0 and not np.isnan(sma75[t_pos]) and not np.isnan(sma75[t_pos - SLOPE_LOOKBACK]):
        points["sma75_rising"] = bool(sma75[t_pos] > sma75[t_pos - SLOPE_LOOKBACK])
    else:
        points["sma75_rising"] = False
    points["breadth_75_ge_half"] = bool(not np.isnan(breadth_75) and breadth_75 >= 0.5)
    points["breadth_200_ge_half"] = bool(not np.isnan(breadth_200) and breadth_200 >= 0.5)

    score = sum(1 for v in points.values() if v)
    if score >= 5:
        level = REGIME_STRONG
    elif score >= 3:
        level = REGIME_MID
    else:
        level = REGIME_WEAK

    return {"score": score, "level": level, **points}
