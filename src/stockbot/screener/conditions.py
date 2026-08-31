"""19 条件のブール判定（docs/SCREENER.md §2）。

A 群（ユニバース）・B 群（トレンド）・C 群（押し目の形）・D 群（反発）・E 群（除外）の
19 条件を、T（判定日）の引けまでのデータだけで判定する。全条件 AND で候補とする。
順位もスコアも作らない（§2.2）。

**欠損は全条件で不成立**（§2.3）。判定できないものは配信しない。唯一の例外は A4 で、
決算発表予定日が取れない場合だけ成立扱いにし、`a4_earnings_unknown` を立てる
（取れないだけで全部落ちると候補がゼロになるため。カードに「決算日未取得」と出す）。

E1 だけはその日の候補集合に依存するため、ここでは判定しない（`screen.py` が
A〜D・E2・E3 の通過集合に対して後から付ける）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.jpx_lists import next_earnings_business_days
from ..features.dimensions import landing_ma, ma_values_at
from ..features.indicators import atr_wilder, sma

# ---- 閾値（docs/SCREENER.md §2。ここ以外に閾値を置かない）----
A1_MIN_ADV_JPY = 3e8          # 20日平均売買代金 ≥ 3億円
A1_ADV_WINDOW = 20
A2_MIN_CLOSE = 300.0          # 終値 ≥ 300円
A3_MIN_BARS = 250             # 履歴 ≥ 250営業日
A4_EARNINGS_WITHIN = 5        # 5営業日以内に決算発表なし
B2_SLOPE_LOOKBACK = 20        # SMA200[T] > SMA200[T-20]
B4_HIGH_BARS = 252            # 過去252日の最高値 = max(High[T-251..T])
B4_RATIO = 0.75
C1_DEPTH_MIN, C1_DEPTH_MAX = 0.03, 0.08
C3_DURATION_MIN, C3_DURATION_MAX = 2, 12
D3_POSITION_MIN = 0.50
D4_MA_DIST_MAX = 1.0
D4_ALLOWED_MA = ("SMA5", "SMA25", "SMA75")   # 200日線で止まったものは候補にしない
E2_DEV5_MAX = 0.02
E3_LP_AGE_MAX = 5             # T − lp ≤ 5
RS60_LOOKBACK = 60            # d2_rs60（DESIGN.md §5 D2）と同じ 60 営業日

GROUPS = {
    "A": ["A1", "A2", "A3", "A4"],
    "B": ["B1", "B2", "B3", "B4"],
    "C": ["C1", "C2", "C3", "C4"],
    "D": ["D1", "D2", "D3", "D4"],
    "E": ["E1", "E2", "E3"],
}
CONDITION_IDS = [cid for ids in GROUPS.values() for cid in ids]
# E1 以外（銘柄ごとに閉じて判定できるもの）
SELF_CONTAINED_IDS = [cid for cid in CONDITION_IDS if cid != "E1"]

# 判定に使った量。記録・カード表示・あとからの突合に使う（判定そのものには使わない）
DIAGNOSTIC_COLS = [
    "adv_jpy", "close_t", "bars", "earnings_days", "a4_earnings_unknown",
    "rs60", "landing_ma", "landing_ma_value", "landing_dist_atr", "lp_age",
]


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float, np.floating, np.integer))
               and np.isfinite(v) for v in values)


def rs60(close: pd.Series, idx_close: Optional[pd.Series], t_pos: int,
         n: int = RS60_LOOKBACK) -> float:
    """相対力 = ln(Close[T]/Close[T−n]) − ln(IDX[T]/IDX[T−n])（DESIGN.md §5 D2 の d2_rs60）。

    idx_close は銘柄の日付に整列済みの指数終値。定義できなければ NaN。
    """
    if idx_close is None or t_pos - n < 0 or t_pos >= len(close):
        return np.nan
    c = close.to_numpy(dtype=float)
    x = idx_close.to_numpy(dtype=float)
    if len(x) != len(c):
        return np.nan
    c_t, c_p, x_t, x_p = c[t_pos], c[t_pos - n], x[t_pos], x[t_pos - n]
    if not _finite(c_t, c_p, x_t, x_p) or min(c_t, c_p, x_t, x_p) <= 0:
        return np.nan
    return float(np.log(c_t / c_p) - np.log(x_t / x_p))


def average_turnover(close: pd.Series, volume: pd.Series, t_pos: int,
                     window: int = A1_ADV_WINDOW) -> float:
    """20日平均売買代金（終値×出来高で近似。universe.build.liquidity_stats と同じ式）。

    窓が満たない（t_pos+1 < window）場合は NaN（A1 は不成立になる）。
    """
    start = t_pos - window + 1
    if start < 0 or t_pos >= len(close):
        return np.nan
    turnover = (close.iloc[start: t_pos + 1].to_numpy(dtype=float)
                * volume.iloc[start: t_pos + 1].to_numpy(dtype=float))
    if not np.isfinite(turnover).all():
        return np.nan
    return float(turnover.mean())


def evaluate_conditions(
    ticker: str, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    pullback_result: dict, t_pos: int,
    idx_close: Optional[pd.Series] = None,
    earnings_schedule: Optional[pd.DataFrame] = None,
) -> dict:
    """1 銘柄 1 日ぶんの条件判定（E1 を除く 18 条件）と、判定に使った量を返す。

    high/low/close/volume は 0 始まりの位置で扱う整列済み pandas.Series
    （features 内の他モジュールと同じ規約）。pullback_result は
    features.pullback.pullback_state() の戻り値。idx_close は銘柄の日付に整列済みの
    指数終値（E1 の母集団に使う rs60 用）。

    戻り値: {"ticker", "A1".."E3"（E1 は pd.NA）, DIAGNOSTIC_COLS...}。
    E1 以外の 18 条件が全て True の行だけが E1 の母集団になる。
    """
    out: dict = {"ticker": ticker}
    c = close.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    close_t = c[t_pos] if 0 <= t_pos < len(c) else np.nan

    sma5 = sma(close, 5).to_numpy(dtype=float)
    sma75 = sma(close, 75).to_numpy(dtype=float)
    sma200 = sma(close, 200).to_numpy(dtype=float)

    # ---------------------------------------------------------------- A 群
    adv = average_turnover(close, volume, t_pos)
    out["A1"] = bool(_finite(adv) and adv >= A1_MIN_ADV_JPY)
    out["A2"] = bool(_finite(close_t) and close_t >= A2_MIN_CLOSE)
    out["A3"] = bool(t_pos + 1 >= A3_MIN_BARS)

    earnings_days = None
    if earnings_schedule is not None and len(earnings_schedule):
        earnings_days = next_earnings_business_days(earnings_schedule, close.index[t_pos], ticker)
    a4_unknown = earnings_days is None
    # A4 だけは欠損を成立扱いにする（§2.3 の唯一の例外）
    out["A4"] = True if a4_unknown else bool(earnings_days > A4_EARNINGS_WITHIN)

    # ---------------------------------------------------------------- B 群
    s75_t = sma75[t_pos] if t_pos < len(sma75) else np.nan
    s200_t = sma200[t_pos] if t_pos < len(sma200) else np.nan
    s200_p = sma200[t_pos - B2_SLOPE_LOOKBACK] if t_pos - B2_SLOPE_LOOKBACK >= 0 else np.nan
    out["B1"] = bool(_finite(s75_t, s200_t) and s75_t > s200_t)
    out["B2"] = bool(_finite(s200_t, s200_p) and s200_t > s200_p)
    out["B3"] = bool(_finite(close_t, s200_t) and close_t >= s200_t)

    start = t_pos - (B4_HIGH_BARS - 1)
    if start >= 0 and t_pos < len(h):
        window = h[start: t_pos + 1]
        out["B4"] = bool(_finite(close_t) and np.isfinite(window).all()
                         and close_t >= B4_RATIO * float(window.max()))
    else:
        out["B4"] = False   # 252 本に満たない = 判定できない → 不成立

    # ---------------------------------------------------------------- C 群
    h0 = pullback_result.get("h0")
    lp = pullback_result.get("lp")
    depth_pct = pullback_result.get("depth_pct", np.nan)
    lp_value = pullback_result.get("lp_value", np.nan)
    l0_low = pullback_result.get("l0_low", np.nan)
    h0_high = pullback_result.get("h0_high", np.nan)
    duration = pullback_result.get("d")

    out["C1"] = bool(_finite(depth_pct) and C1_DEPTH_MIN <= depth_pct <= C1_DEPTH_MAX)
    out["C2"] = bool(_finite(lp_value, l0_low) and lp_value > l0_low)
    out["C3"] = bool(duration is not None and _finite(duration)
                     and C3_DURATION_MIN <= duration <= C3_DURATION_MAX)
    if h0 is not None and _finite(h0_high) and h0 + 1 <= t_pos < len(h):
        since_h0 = h[h0 + 1: t_pos + 1]
        out["C4"] = bool(np.isfinite(since_h0).all() and float(since_h0.max()) < h0_high)
    else:
        out["C4"] = False

    # ---------------------------------------------------------------- D 群
    s5_t = sma5[t_pos] if t_pos < len(sma5) else np.nan
    out["D1"] = bool(_finite(close_t, s5_t) and close_t > s5_t)
    prev_high = h[t_pos - 1] if t_pos - 1 >= 0 else np.nan
    out["D2"] = bool(_finite(close_t, prev_high) and close_t > prev_high)

    position = pullback_result.get("position", np.nan)
    out["D3"] = bool(_finite(position) and position >= D3_POSITION_MIN)

    atr_t = float(atr_wilder(high, low, close, 14).iloc[t_pos])
    landing = {"landing_ma": None, "landing_ma_value": np.nan, "dist_atr": np.nan}
    if lp is not None:
        landing = landing_ma(lp_value, ma_values_at(close, int(lp)), atr_t)
    out["D4"] = bool(landing["landing_ma"] in D4_ALLOWED_MA
                     and _finite(landing["dist_atr"])
                     and landing["dist_atr"] <= D4_MA_DIST_MAX)

    # ---------------------------------------------------------------- E 群
    out["E1"] = pd.NA   # その日の候補集合が要る（screen.apply_e1）
    dev5 = pullback_result.get("dev5", np.nan)
    out["E2"] = bool(_finite(dev5) and dev5 <= E2_DEV5_MAX)
    lp_age = (t_pos - int(lp)) if lp is not None else np.nan
    out["E3"] = bool(_finite(lp_age) and lp_age <= E3_LP_AGE_MAX)

    # ---------------------------------------------------------------- 診断
    out.update({
        "adv_jpy": adv,
        "close_t": close_t,
        "bars": int(t_pos + 1),
        "earnings_days": earnings_days if earnings_days is not None else np.nan,
        "a4_earnings_unknown": a4_unknown,
        "rs60": rs60(close, idx_close, t_pos),
        "landing_ma": landing["landing_ma"] or "",
        "landing_ma_value": landing["landing_ma_value"],
        "landing_dist_atr": landing["dist_atr"],
        "lp_age": lp_age,
    })
    return out


def passes_self_contained(row) -> bool:
    """E1 以外の 18 条件が全て True か（E1 の母集団の定義、docs/SCREENER.md §2.5）。

    欠損（pd.NA / NaN）は不成立として扱う（§2.6）。素朴に bool() すると pd.NA で
    例外になり、NaN は真になってしまう。
    """
    for cid in SELF_CONTAINED_IDS:
        value = row[cid]
        if not isinstance(value, (bool, np.bool_)) or not value:
            return False
    return True
