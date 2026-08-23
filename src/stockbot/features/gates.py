"""ゲート G0〜G3（DESIGN.md §2 / TASKS.md T-208）。

4 つのハード条件（ハード条件は増やさない、§2 冒頭）:
- G0: ユニバース通過、かつ T から保有上限（label_n 営業日、DESIGN.md §12 の事前登録
  パラメータ）以内に決算発表が無い。予定日が取れない場合（決算スケジュールが無い、
  または当該銘柄の予定日が見つからない）はこの条件を省略して True とし、
  g0_earnings_unknown を立てる（決算データの欠測を不利に倒さない）
- G1: SMA75[T] > SMA200[T] かつ SMA200[T] > SMA200[T−30]
- G2: Close[T] ≥ SMA200[T]
- G3: Close[T] ≥ 0.75 × max(High[T−251..T])（52週高値から25%以内、Minervini）

25日線は条件にしない（DESIGN.md §2 末尾。押し目は25日線を割るのが普通）。

各条件は指標が未定義（warmup 未了）の場合は「満たさない」（False）として扱う
（欠測を有利にも不利にも倒さない安全側のデフォルト。pullback.py/regime.py と同じ方針）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.jpx_lists import next_earnings_business_days

G3_LOOKBACK = 251  # DESIGN.md §2 G3: max(High[T-251..T])
G1_SLOPE_LOOKBACK = 30  # DESIGN.md §2 G1: SMA200[T] > SMA200[T-30]

GATE_COLS = ["g0", "g1", "g2", "g3", "gate_pass", "g0_earnings_unknown"]


def evaluate_gates(
    close: pd.Series, high: pd.Series, sma75: pd.Series, sma200: pd.Series,
    t_pos: int, passes_universe: bool, label_n: int,
    earnings_schedule: Optional[pd.DataFrame] = None, ticker: Optional[str] = None,
) -> dict:
    """T（t_pos）時点のゲート G0〜G3 を判定する（DESIGN.md §2）。

    close/high/sma75/sma200 は 0 始まりの位置（features 内の他モジュールと同じ規約）で
    扱う整列済み pandas.Series。sma75/sma200 は features.indicators.sma() の出力を渡す。

    戻り値: {"g0","g1","g2","g3","gate_pass"（4つの AND）,"g0_earnings_unknown"}
    """
    c = close.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    s75 = sma75.to_numpy(dtype=float)
    s200 = sma200.to_numpy(dtype=float)

    # ---- G0: ユニバース通過 かつ 決算発表が label_n 営業日以内に無い ----
    g0_earnings_unknown = False
    earnings_near = False
    if earnings_schedule is not None and ticker is not None and len(earnings_schedule):
        asof = close.index[t_pos]
        days = next_earnings_business_days(earnings_schedule, asof, ticker)
        if days is None:
            g0_earnings_unknown = True
        else:
            earnings_near = days <= label_n
    else:
        g0_earnings_unknown = True
    g0 = bool(passes_universe) and (g0_earnings_unknown or not earnings_near)

    # ---- G1: 中期・長期トレンドが上向き ----
    g1 = False
    if t_pos - G1_SLOPE_LOOKBACK >= 0:
        s75_t, s200_t, s200_prev = s75[t_pos], s200[t_pos], s200[t_pos - G1_SLOPE_LOOKBACK]
        if not (np.isnan(s75_t) or np.isnan(s200_t) or np.isnan(s200_prev)):
            g1 = bool(s75_t > s200_t and s200_t > s200_prev)

    # ---- G2: 長期トレンド維持 ----
    g2 = bool(not np.isnan(s200[t_pos]) and c[t_pos] >= s200[t_pos])

    # ---- G3: 52週高値から25%以内 ----
    window_start = max(0, t_pos - G3_LOOKBACK)
    window = h[window_start: t_pos + 1]
    g3 = bool(window.size and c[t_pos] >= 0.75 * window.max())

    gate_pass = g0 and g1 and g2 and g3
    return {
        "g0": g0, "g1": g1, "g2": g2, "g3": g3,
        "gate_pass": gate_pass, "g0_earnings_unknown": g0_earnings_unknown,
    }
