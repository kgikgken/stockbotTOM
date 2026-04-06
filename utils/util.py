from __future__ import annotations

import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


# ---------- env / date helpers ----------

def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_date() -> date:
    return jst_now().date()


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


# ---------- numeric helpers ----------

def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def percentile_rank(values: Sequence[float], target: float) -> float:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0 or not np.isfinite(target):
        return float("nan")
    return float((arr <= target).mean() * 100.0)


def rolling_median_ratio(series: pd.Series, window: int = 60) -> float:
    if series is None or series.empty or len(series) < max(window, 5):
        return float("nan")
    base = series.rolling(window).median().iloc[-1]
    last = series.iloc[-1]
    if not (np.isfinite(base) and np.isfinite(last) and base != 0):
        return float("nan")
    return float(last / base)


# ---------- market microstructure helpers ----------

def tick_size_jpx(price: float) -> float:
    """Approximate TSE tick size by price band."""
    p = safe_float(price)
    if not np.isfinite(p) or p <= 0:
        return 1.0
    if p < 1_000:
        return 1.0
    if p < 3_000:
        return 1.0
    if p < 5_000:
        return 5.0
    if p < 30_000:
        return 10.0
    if p < 50_000:
        return 50.0
    if p < 300_000:
        return 100.0
    if p < 500_000:
        return 500.0
    return 1_000.0


def round_to_tick(price: float, price_ref: float, mode: str = "nearest") -> float:
    tick = tick_size_jpx(price_ref)
    if tick <= 0:
        return float(price)
    if mode == "down":
        return math.floor(price / tick) * tick
    if mode == "up":
        return math.ceil(price / tick) * tick
    return round(price / tick) * tick


# ---------- indicators ----------

def atr14(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or df.empty or len(df) < period * 2:
        return pd.Series(dtype=float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).sum() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).sum() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period, min_periods=period).mean()


def efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    if close is None or close.empty:
        return pd.Series(dtype=float)
    direction = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period, min_periods=period).sum()
    return direction / volatility.replace(0, np.nan)


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or df.empty or len(df) < period:
        return pd.Series(dtype=float)
    atr = atr14(df)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    denom = (hh - ll).replace(0, np.nan)
    value = 100 * np.log10(atr.rolling(period, min_periods=period).sum() / denom) / np.log10(period)
    return value


def bb_width_ratio(close: pd.Series, ma_window: int = 20, lookback: int = 60) -> float:
    if close is None or close.empty or len(close) < max(ma_window + 5, lookback):
        return float("nan")
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    std = close.rolling(ma_window, min_periods=ma_window).std(ddof=0)
    width = (4.0 * std) / ma.replace(0, np.nan)
    return rolling_median_ratio(width.dropna(), window=lookback)


def down_up_volume_ratio(close: pd.Series, volume: pd.Series, lookback: int = 10) -> float:
    if close is None or volume is None or len(close) < lookback + 1:
        return float("nan")
    delta = close.diff()
    recent = delta.iloc[-lookback:]
    vol_recent = volume.iloc[-lookback:]
    down = vol_recent.where(recent < 0, 0.0).sum()
    up = vol_recent.where(recent > 0, 0.0).sum()
    if up <= 0:
        return float("inf") if down > 0 else 1.0
    return float(down / up)


def is_abnormal_stock(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 60:
        return True
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    if close.iloc[-1] <= 50:
        return True
    if vol.tail(20).median() <= 0:
        return True
    if close.tail(20).pct_change().abs().max() > 0.40:
        return True
    return False


# ---------- data download ----------

def _chunked(items: Sequence[str], size: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in wanted:
        if col not in out.columns:
            if col == "Adj Close" and "Close" in out.columns:
                out[col] = out["Close"]
            elif col == "Volume":
                out[col] = 0.0
            else:
                out[col] = np.nan
    out = out[wanted]
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def download_history_bulk(
    tickers: Sequence[str],
    period: str = "3y",
    interval: str = "1d",
    chunk_size: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV from yfinance in chunks.

    The function is intentionally tolerant: unavailable symbols are skipped.
    """
    symbols = [str(t).strip() for t in tickers if str(t).strip()]
    if not symbols:
        return {}

    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for batch in _chunked(symbols, chunk_size):
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception:
            continue
        if raw is None or getattr(raw, "empty", True):
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = set(str(x) for x in raw.columns.get_level_values(0))
            if any(symbol in lvl0 for symbol in batch):
                for symbol in batch:
                    try:
                        frame = raw[symbol].dropna(how="all")
                    except Exception:
                        frame = None
                    if frame is not None and not frame.empty:
                        result[symbol] = _normalize_history_frame(frame)
            else:
                # Layout: fields first, symbols second.
                for symbol in batch:
                    try:
                        frame = raw.xs(symbol, axis=1, level=1).dropna(how="all")
                    except Exception:
                        frame = None
                    if frame is not None and not frame.empty:
                        result[symbol] = _normalize_history_frame(frame)
        else:
            if len(batch) == 1:
                result[batch[0]] = _normalize_history_frame(raw.dropna(how="all"))

    return result


# ---------- text helpers ----------

def human_yen(value: float) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "-"
    if abs(v) >= 100_000_000:
        return f"{v/100_000_000:.1f}億"
    if abs(v) >= 10_000:
        return f"{v/10_000:.1f}万"
    return f"{v:,.0f}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
