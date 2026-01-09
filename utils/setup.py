from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SetupResult:
    setup: str                 # "A1" / "A2" / "B" / "NONE"
    entry_low: float
    entry_high: float
    entry_mid: float
    stop_seed: float           # initial stop seed (finalized later)
    breakout_line: float       # for B
    gu: bool
    note: str


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def atr14(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 16:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    v = tr.rolling(14).mean().iloc[-1]
    return _safe_float(v, np.nan)


def _ma(series: pd.Series, w: int) -> float:
    if series is None or len(series) < w:
        return _safe_float(series.iloc[-1]) if series is not None and len(series) else np.nan
    return _safe_float(series.rolling(w).mean().iloc[-1], np.nan)


def _rsi(close: pd.Series, n: int = 14) -> float:
    if close is None or len(close) < n + 5:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return _safe_float(rsi.iloc[-1], np.nan)


def detect_setup(hist: pd.DataFrame) -> SetupResult:
    """
    仕様のSetup判定（順張りのみ）

    A1（最優先）
      - Close > SMA20 > SMA50
      - SMA20 上向き
      - 押し：SMA20 ± 0.5ATR
      - RSI 40〜60
      - 押しで出来高減 → 反発（proxy）

    A2（許容）
      - SMA20〜SMA50 押し（やや深い）
      - 形OKなら返す（最終採用はRR/EV/Rdayで絞る）

    B（ブレイク）
      - HH20 ブレイク
      - 出来高 ≥ 1.5×平均
      - 当日成行禁止（必ず押し待ち）→ entry はブレイクライン±0.3ATR

    GU判定
      - Open > 前日終値 + 1ATR → 即IN禁止（寄り後再判定）
    """
    if hist is None or hist.empty or len(hist) < 80:
        return SetupResult("NONE", 0, 0, 0, 0, 0, False, "data_short")

    df = hist.copy()
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    c = _safe_float(close.iloc[-1], np.nan)
    o = _safe_float(open_.iloc[-1], np.nan)
    prev_c = _safe_float(close.iloc[-2], np.nan) if len(close) >= 2 else np.nan

    atr = atr14(df)
    if not (np.isfinite(c) and np.isfinite(atr) and atr > 0):
        return SetupResult("NONE", 0, 0, 0, 0, 0, False, "bad_price")

    sma20 = _ma(close, 20)
    sma50 = _ma(close, 50)

    # SMA20 slope (5d up?)
    slope_ok = False
    if len(close) >= 26:
        sma20_series = close.rolling(20).mean()
        a = _safe_float(sma20_series.iloc[-6], np.nan)
        b = _safe_float(sma20_series.iloc[-1], np.nan)
        slope_ok = bool(np.isfinite(a) and np.isfinite(b) and a > 0 and (b / a - 1.0) > 0)

    rsi = _rsi(close, 14)
    rsi_ok = bool(np.isfinite(rsi) and 40 <= rsi <= 60)

    gu = bool(np.isfinite(o) and np.isfinite(prev_c) and o > prev_c + atr)

    # Volume contraction proxy on pullback (last 3 vs previous 10)
    v3 = _safe_float(vol.tail(3).mean(), np.nan)
    v10 = _safe_float(vol.tail(13).head(10).mean(), np.nan)
    vol_contract = bool(np.isfinite(v3) and np.isfinite(v10) and v10 > 0 and v3 < v10 * 0.90)

    rebound = bool(np.isfinite(c) and np.isfinite(prev_c) and c > prev_c)

    # --- B: HH20 breakout
    hh20 = _safe_float(high.rolling(20).max().iloc[-2], np.nan)  # yesterday HH20
    breakout_today = bool(np.isfinite(c) and np.isfinite(hh20) and c > hh20)
    vavg20 = _safe_float(vol.rolling(20).mean().iloc[-1], np.nan)
    vol_ok = bool(np.isfinite(vol.iloc[-1]) and np.isfinite(vavg20) and vavg20 > 0 and float(vol.iloc[-1]) >= 1.5 * vavg20)

    if breakout_today and vol_ok and np.isfinite(hh20):
        entry_low = float(hh20 - 0.3 * atr)
        entry_high = float(hh20 + 0.3 * atr)
        entry_mid = float(hh20)
        stop_seed = float(entry_mid - 1.2 * atr)
        return SetupResult("B", entry_low, entry_high, entry_mid, stop_seed, float(hh20), gu, "HH20_breakout")

    # --- A1 / A2
    a_trend = bool(np.isfinite(c) and np.isfinite(sma20) and np.isfinite(sma50) and c > sma20 > sma50 and slope_ok)

    entry_low_a1 = float(sma20 - 0.5 * atr)
    entry_high_a1 = float(sma20 + 0.5 * atr)
    entry_mid_a1 = float(sma20)

    pullback_near_a1 = bool(c <= entry_high_a1 * 1.01 and c >= entry_low_a1 * 0.95)

    if a_trend and pullback_near_a1 and rsi_ok and (vol_contract or rebound):
        stop_seed = float(entry_mid_a1 - 1.2 * atr)
        return SetupResult("A1", entry_low_a1, entry_high_a1, entry_mid_a1, stop_seed, 0.0, gu, "pullback_SMA20")

    # A2: deeper pullback around SMA20..SMA50
    a2_ok = bool(np.isfinite(c) and np.isfinite(sma20) and np.isfinite(sma50) and (sma20 >= sma50 or c > sma50))
    entry_low_a2 = float(min(sma20, sma50) - 0.3 * atr)
    entry_high_a2 = float(max(sma20, sma50) + 0.3 * atr)
    entry_mid_a2 = float((sma20 + sma50) / 2.0) if np.isfinite(sma20) and np.isfinite(sma50) else float(sma20)
    pullback_near_a2 = bool(c <= entry_high_a2 * 1.01 and c >= entry_low_a2 * 0.95)

    if a2_ok and pullback_near_a2 and (np.isfinite(rsi) and 35 <= rsi <= 65):
        stop_seed = float(entry_mid_a2 - 1.2 * atr)
        return SetupResult("A2", entry_low_a2, entry_high_a2, entry_mid_a2, stop_seed, 0.0, gu, "deep_pullback")

    return SetupResult("NONE", 0, 0, 0, 0, 0, gu, "no_setup")
