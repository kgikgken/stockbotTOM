"""swing2w用データ層 — 実データはmomentum/mispricingの汎用フェッチを再利用。
DRYRUN合成データは低回転率(エンジンR)・高回転率(エンジンM)の両方を作り分ける。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from momentum.data import fetch_ohlcv as _fetch_ohlcv_momentum  # 汎用実データ取得(fetch_failures記録込み)を再利用


def fetch_ohlcv(tickers: List[str], history_days: int, dryrun: bool = False) -> Tuple[Dict[str, pd.DataFrame], dict]:
    if dryrun:
        return _synthetic(tickers)
    return _fetch_ohlcv_momentum(tickers, history_days, dryrun=False)


def _mk_series_swing(seed: int, n: int, mode: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    if end.dayofweek >= 5:
        end -= pd.tseries.offsets.BDay(1)
    idx = pd.bdate_range(end=end, periods=n)

    if mode == "low_turnover_oversold":
        # 低〜中回転率(出来高少なめ)。決定的な緩やかな上昇トレンド+直近5日で業種内相対に売られ過ぎ
        t = np.arange(n)
        trend = 1000.0 * np.exp(0.0003 * t)
        close = trend * (1 + rng.normal(0, 0.006, n))
        v = rng.integers(500_000, 800_000, n).astype(float)  # 低回転率(ただし流動性フィルター5億円は超える水準)
        base = float(close[-6])
        close[-5:] = base * np.linspace(1.0, 0.93, 5)  # 直近5日で-7%程度の急な押し目
    elif mode == "high_turnover_gap":
        # 高回転率(出来高多め)。平常運転+直近2日以内に決算ギャップ想定の単日急騰+出来高急増
        ret = rng.normal(0.0002, 0.014, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(2_000_000, 4_000_000, n).astype(float)  # 高回転率
        gap_pos = n - 2  # 2営業日前にギャップ
        base = float(close[gap_pos - 1])
        close[gap_pos] = base * 1.06  # +6%ギャップ
        close[gap_pos + 1:] = close[gap_pos] * (1 + rng.normal(0.001, 0.010, n - gap_pos - 1))
        v[gap_pos] = v[gap_pos - 1] * 3.0  # 出来高急増
    elif mode == "high_turnover_breakout":
        # 高回転率。52週高値をブレイクする形状+出来高確認
        t = np.arange(n)
        trend = 1000.0 * np.exp(0.0009 * t)
        close = trend * (1 + rng.normal(0, 0.010, n))
        v = rng.integers(2_000_000, 4_000_000, n).astype(float)
        # 直近の高値を明確に超える形にする(直前259日の最大値より当日を高くする)
        past_max = float(close[-260:-1].max())
        close[-1] = past_max * 1.02
        v[-1] = v[-2] * 2.2
    elif mode == "tob_pattern":
        drift, noise = 0.0002, 0.016
        ret = rng.normal(drift, noise, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(1_000_000, 2_000_000, n).astype(float)
        jump_day = -40
        pre_base = float(close[jump_day - 1])
        close[jump_day] = pre_base * 1.28
        pinned = float(close[jump_day])
        close[jump_day + 1:] = pinned * (1 + np.linspace(0, 0, n - (n + jump_day + 1)))
        v[jump_day] = float(v[jump_day - 5:jump_day].mean()) * 5.0
    else:
        ret = rng.normal(0.0002, 0.016, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(500_000, 1_500_000, n).astype(float)

    hl_scale = np.full(n, 0.006)
    if mode == "tob_pattern":
        hl_scale[-40 + 1:] = 0.0008
    o = close * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 1, n)) * hl_scale)
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 1, n)) * hl_scale)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close, "Volume": v}, index=idx)


def _synthetic(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], dict]:
    # 0,1: 同一セクターにlow_turnover_oversoldを2つ(セクター分散キャップ検証用)
    # 2,3: 同様にhigh_turnover_gapを2つ / 4: high_turnover_breakout / 5,6: tob_pattern
    modes = ["low_turnover_oversold", "low_turnover_oversold", "high_turnover_gap", "high_turnover_gap",
             "high_turnover_breakout", "tob_pattern", "tob_pattern"] + ["normal"] * 5
    out: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers[:60]):
        out[t] = _mk_series_swing(seed=311 + i, n=420,
                                  mode=modes[i % len(modes)] if i < 12 else "normal")
    meta = {
        "data_total": len(tickers[:60]), "data_ok": len(out),
        "data_coverage": 1.0 if tickers else 0.0,
        "source": "SYNTHETIC (DRYRUN・実データではない)",
        "fetched_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST"),
        "dryrun": True,
    }
    return out, meta
