"""押し目の構造と状態（DESIGN.md §4 / TASKS.md T-203）。

- 構造（§4.1）: H0（直前 60 本の最高値である、直近の確定スイング高値）、L0（h0 より
  前の直近の確定スイング安値）、Lp（押し目区間の最安値）、R（戻り高値）と基本量
  （depth_pct, depth_atr, retrace, position, dev5）
- 押し目の条件（§4.2）と状態（§4.4、5状態）: 完了 → 失効 → ブレイク → 反発開始 → 形成中
  の優先順で毎日ちょうど一つ判定する
- 浅押し・深押しフラグ（§4.4 末尾）

引数の high/low/close/sma5/sma200/atr14 は 0 始まりの位置（swings.py と同じ規約）で
扱う整列済み pandas.Series。alternated_swings は swings.alternate_swings() の出力。

H0 選定について: 「確定済みスイング高値のうち最新」かつ「直前60本の最高値」という
2条件は、直近の確定スイング高値が60本条件を満たさない場合（その直前に、まだ超えられて
いないより高い確定スイング高値が60本以内に存在する場合）に、そのより高い方を H0 と
みなすべきだと読み、新しい方から遡って最初に60本条件を満たす確定スイング高値を探す
実装にした（該当が無ければ構造なし）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .swings import SWING_HIGH, SWING_LOW, swings_as_of

STATE_COMPLETE = "完了"
STATE_INVALID = "失効"
STATE_BREAK = "ブレイク"
STATE_BOUNCE = "反発開始"
STATE_FORMING = "形成中"
# DESIGN.md §4.4 の5状態には含まれない。H0/L0（有効な脚）が定義できない場合の宣言的な値
STATE_NO_STRUCTURE = "no_structure"

H0_LOOKBACK = 60  # DESIGN.md §4.1: H0.high = max(High[h0-60..h0])

RESULT_FIELDS = [
    "state", "h0", "l0", "lp",
    "h0_high", "l0_low", "lp_value", "r",
    "leg", "leg_bars", "d",
    "depth_pct", "depth_atr", "retrace", "position", "dev5",
    "is_shallow", "is_deep",
]


def find_h0(alternated_swings: pd.DataFrame, high: pd.Series, t_pos: int, k: int,
           lookback: int = H0_LOOKBACK) -> Optional[int]:
    """T 時点で確定済みの直近スイング高値のうち、直前 lookback 本の最高値であるものを探す。

    新しい方から遡り、最初に条件を満たしたインデックスを返す（無ければ None）。
    """
    confirmed = swings_as_of(alternated_swings, t_pos)
    highs = confirmed[confirmed["kind"] == SWING_HIGH].sort_values("index", ascending=False)
    h = high.to_numpy(dtype=float)
    for idx in highs["index"]:
        idx = int(idx)
        window_start = max(0, idx - lookback)
        prior = h[window_start:idx]
        if prior.size == 0 or h[idx] >= prior.max():
            return idx
    return None


def find_l0(alternated_swings: pd.DataFrame, t_pos: int, h0: int) -> Optional[int]:
    """h0 より前の直近の確定スイング安値。無ければ None。"""
    confirmed = swings_as_of(alternated_swings, t_pos)
    lows = confirmed[(confirmed["kind"] == SWING_LOW) & (confirmed["index"] < h0)]
    if lows.empty:
        return None
    return int(lows["index"].max())


def _trigger_fired(close: np.ndarray, sma5: np.ndarray, high: np.ndarray, t: int) -> bool:
    """反発トリガー（§4.3）。t-2 が存在しない先頭付近は発火しない。"""
    if t < 2:
        return False
    trig_a = close[t] > high[t - 1] and close[t] > sma5[t]
    trig_b = (sma5[t] > sma5[t - 1]) and (sma5[t - 1] <= sma5[t - 2]) and (close[t] > sma5[t])
    return bool(trig_a or trig_b)


def _no_structure() -> dict:
    return {
        "state": STATE_NO_STRUCTURE,
        "h0": None, "l0": None, "lp": None,
        "h0_high": np.nan, "l0_low": np.nan, "lp_value": np.nan, "r": np.nan,
        "leg": np.nan, "leg_bars": np.nan, "d": np.nan,
        "depth_pct": np.nan, "depth_atr": np.nan, "retrace": np.nan,
        "position": np.nan, "dev5": np.nan,
        "is_shallow": False, "is_deep": False,
    }


def pullback_state(high: pd.Series, low: pd.Series, close: pd.Series,
                   sma5: pd.Series, sma200: pd.Series, atr14: pd.Series,
                   alternated_swings: pd.DataFrame, t_pos: int, k: int,
                   shallow_depth: float = 0.03, deep_depth: float = 0.10,
                   pullback_max_days: int = 25, pullback_max_depth: float = 0.25,
                   h0_lookback: int = H0_LOOKBACK) -> dict:
    """T（t_pos）時点の押し目構造・基本量・状態を計算する（DESIGN.md §4）。

    引数の shallow_depth/deep_depth/pullback_max_days/pullback_max_depth は
    DESIGN.md §12 の事前登録パラメータ（config.Settings の同名フィールド）。

    戻り値: RESULT_FIELDS の dict。構造が定義できない（H0/L0 が無い、leg_bars<3）
    場合は state=STATE_NO_STRUCTURE で他フィールドは NaN/None（例外にしない）。
    """
    h0 = find_h0(alternated_swings, high, t_pos, k, h0_lookback)
    if h0 is None:
        return _no_structure()
    l0 = find_l0(alternated_swings, t_pos, h0)
    if l0 is None:
        return _no_structure()
    leg_bars = h0 - l0
    if leg_bars < 3:
        return _no_structure()

    h_arr = high.to_numpy(dtype=float)
    l_arr = low.to_numpy(dtype=float)
    c_arr = close.to_numpy(dtype=float)
    sma5_arr = sma5.to_numpy(dtype=float)

    h0_high = float(h_arr[h0])
    l0_low = float(l_arr[l0])
    leg = h0_high - l0_low
    d = t_pos - h0

    window_low = l_arr[h0 + 1: t_pos + 1]
    if window_low.size == 0:
        # 押し目区間がまだ無い（T が h0 以前、または h0 当日）。基本量は未定義
        return _no_structure()
    lp_rel = int(np.argmin(window_low))
    lp = h0 + 1 + lp_rel
    lp_value = float(window_low[lp_rel])

    atr_h0 = float(atr14.iloc[h0])
    depth_pct = (h0_high - lp_value) / h0_high
    depth_atr = (h0_high - lp_value) / atr_h0 if np.isfinite(atr_h0) and atr_h0 > 0 else np.nan
    retrace = (h0_high - lp_value) / leg if leg else np.nan
    close_t = float(c_arr[t_pos])
    denom = h0_high - lp_value
    position = (close_t - lp_value) / denom if np.isfinite(denom) and denom > 0 else np.nan
    sma5_t = sma5_arr[t_pos]
    dev5 = (close_t / sma5_t - 1) if np.isfinite(sma5_t) and sma5_t > 0 else np.nan

    if lp < t_pos - 1:
        r_window = h_arr[lp + 1: t_pos]
        r_value = float(r_window.max()) if r_window.size else np.nan
    else:
        r_value = np.nan

    is_shallow = bool(depth_pct < shallow_depth)
    is_deep = bool(depth_pct > deep_depth)

    result = {
        "h0": h0, "l0": l0, "lp": lp,
        "h0_high": h0_high, "l0_low": l0_low, "lp_value": lp_value, "r": r_value,
        "leg": leg, "leg_bars": leg_bars, "d": d,
        "depth_pct": depth_pct, "depth_atr": depth_atr, "retrace": retrace,
        "position": position, "dev5": dev5,
        "is_shallow": is_shallow, "is_deep": is_deep,
    }

    # ---- 状態（§4.4）: 完了 → 失効 → ブレイク → 反発開始 → 形成中 の優先順 ----
    if h_arr[t_pos] > h0_high:
        result["state"] = STATE_COMPLETE
        return result

    max_high_since_h0 = float(h_arr[h0 + 1: t_pos + 1].max())
    sma200_t = float(sma200.iloc[t_pos])
    cond_close_sma200 = (not np.isnan(sma200_t)) and (close_t >= sma200_t)
    valid_4_2 = (
        (2 <= d <= pullback_max_days)
        and (lp_value > l0_low)
        and (max_high_since_h0 < h0_high)
        and (depth_pct <= pullback_max_depth)
        and cond_close_sma200
    )
    if not valid_4_2:
        result["state"] = STATE_INVALID
        return result

    if not np.isnan(r_value) and close_t > r_value:
        result["state"] = STATE_BREAK
        return result

    bounced = False
    for t_trig in (t_pos - 2, t_pos - 1, t_pos):
        if t_trig <= h0 or t_trig >= len(close):
            continue
        if _trigger_fired(c_arr, sma5_arr, h_arr, t_trig) and lp <= t_trig:
            bounced = True
            break
    result["state"] = STATE_BOUNCE if bounced else STATE_FORMING
    return result
