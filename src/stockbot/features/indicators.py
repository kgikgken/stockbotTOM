"""基礎指標（DESIGN.md §0, §5, TASKS.md T-201）。

- SMA_n: 終値の単純移動平均。n は 5/25/75/200 のみ（§0）
- TR / ATR14: True Range と Wilder の平滑化 ATR（§0）。ATR14 の初期値は最初の
  14 本の TR の単純平均（教科書的な Wilder 法。pandas の ewm 近似は使わない
  ―― seed が異なり、小さな既知系列での手計算と一致しなくなるため）
- ATR5 / ATR20: 単純移動平均の TR（Wilder ではない。d5_atr_ratio 用、§5）
- BB幅: (20, 2σ) のバンド幅 / SMA20（d5_bbw_pct の元系列、§5）。標準偏差は
  母標準偏差（ddof=0）
- 週足集約: W-FRI（金曜終わりの週）に集約し、T を含む週は未確定として除外する
  （§0, §11）

すべて rolling/recursive-causal（各行の計算はその行までのデータのみに依存）で、
系列を T で切って計算した結果と、全期間で計算して T 行を取り出した結果が一致する
（再計算一致テスト、§11）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMA_PERIODS = (5, 25, 75, 200)


def sma(close: pd.Series, n: int) -> pd.Series:
    """終値の単純移動平均（§0）。窓が満たない先頭 n-1 行は NaN。"""
    return close.rolling(n, min_periods=n).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """TR = max(H−L, |H−C_prev|, |L−C_prev|)（§0）。先頭行は C_prev が無いため NaN。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    if len(tr):
        tr.iloc[0] = np.nan
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder の平滑化 ATR（§0）。

    最初の有効値（先頭の TR n 本ぶんが揃った行）は TR の単純平均、以降は
    ATR[i] = (ATR[i-1] * (n-1) + TR[i]) / n の再帰。
    """
    tr = true_range(high, low, close)
    values = tr.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    start = 0
    while start < len(values) and np.isnan(values[start]):
        start += 1
    seed_idx = start + n - 1
    if seed_idx < len(values):
        out[seed_idx] = values[start:start + n].mean()
        for i in range(seed_idx + 1, len(values)):
            out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return pd.Series(out, index=tr.index)


def atr_simple(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """単純移動平均の ATR（Wilder ではない。§5 d5_atr_ratio = ATR5/ATR20 の注記）。"""
    tr = true_range(high, low, close)
    return tr.rolling(n, min_periods=n).mean()


def bb_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """ボリンジャーバンド幅 (n, kσ) / SMA_n（§5 d5_bbw_pct の元系列）。

    標準偏差は母標準偏差（ddof=0、Bollinger の原義）。
    """
    mid = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return (2 * k * std) / mid


def weekly_ohlcv(daily: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """日足を W-FRI 週足に集約する（§0, §11）。

    列は Open/High/Low/Close（+ あれば Volume）。集約は
    Open=最初/High=最大/Low=最小/Close=最後/Volume=合計。
    T（asof）を含む週は、asof が金曜であっても未確定として常に除外する
    （CLAUDE.md「週足は T を含む週を使わない」）。
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    cols = [c for c in agg if c in daily.columns]
    w = daily[cols].resample("W-FRI").agg({c: agg[c] for c in cols})
    ohlc_cols = [c for c in ("Open", "High", "Low", "Close") if c in w.columns]
    w = w.dropna(subset=ohlc_cols, how="any")

    asof = pd.Timestamp(asof).normalize()
    cutoff_friday = asof if asof.weekday() == 4 else asof + pd.offsets.Week(weekday=4)
    return w[w.index < cutoff_friday]
