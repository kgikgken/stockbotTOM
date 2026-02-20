from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Silence noisy yfinance logs by default.
# (GitHub Actions のログが yfinance の警告で埋まるのを避ける)
import logging


def env_truthy(name: str, default: bool = False) -> bool:
    """Parse a boolean-like environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def tick_size_jpx(price: float) -> float:
    """Return the JPX tick size (呼値) for a given price.

    This follows JPX's official *"その他の銘柄"* table.

    Why this matters:
      - Wrong tick sizes can produce invalid limit prices (order rejected).
      - Even if accepted, coarse/incorrect rounding can shift entry/SL/zone and
        distort the intended risk.
    """

    try:
        p = float(price)
    except Exception:
        return 1.0
    if not np.isfinite(p) or p <= 0:
        return 1.0

    # JPX 呼値（その他の銘柄）
    # https://www.jpx.co.jp/equities/trading/domestic/01.html
    if p <= 3000:
        return 1.0
    if p <= 5000:
        return 5.0
    if p <= 30000:
        return 10.0
    if p <= 50000:
        return 50.0
    if p <= 300000:
        return 100.0
    if p <= 500000:
        return 500.0
    if p <= 3000000:
        return 1000.0
    if p <= 5000000:
        return 5000.0
    if p <= 30000000:
        return 10000.0
    if p <= 50000000:
        return 50000.0
    return 100000.0


def floor_to_tick(price: float, tick: float) -> float:
    """Floor price to the nearest valid tick."""

    try:
        p = float(price)
        t = float(tick)
        if not (np.isfinite(p) and np.isfinite(t)) or t <= 0:
            return p
        return float(np.floor(p / t) * t)
    except Exception:
        return float(price)


def ceil_to_tick(price: float, tick: float) -> float:
    """Ceil price to the nearest valid tick."""

    try:
        p = float(price)
        t = float(tick)
        if not (np.isfinite(p) and np.isfinite(t)) or t <= 0:
            return p
        return float(np.ceil(p / t) * t)
    except Exception:
        return float(price)

JST = timezone(timedelta(hours=9))

def jst_now() -> datetime:
    return datetime.now(JST)

def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")

def jst_today_date():
    return jst_now().date()

def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
    dt_str = (dt_str or "").strip()
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()

    if dt_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str and time_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=JST)
        except Exception:
            return None

    return None

def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))

@dataclass
class OHLCV:
    df: pd.DataFrame

def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    out = []
    for t in tickers:
        t = str(t).strip()
        if t:
            out.append(t)
    seen=set()
    uniq=[]
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq

def download_history_bulk(
    tickers: Iterable[str],
    period: str = "260d",
    auto_adjust: bool = True,
    group_size: int = 200,
    pause_sec: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    """
    yfinance.download を chunk で回す。
    返り値: {ticker: df} (Open/High/Low/Close/Volume)
    """
    tickers = _normalize_tickers(tickers)
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    # yfinance は欠損ティッカーが混ざると大量の WARNING/ERROR を標準出力に出す。
    # Actions のログ可読性のため、デフォルトで抑制する。
    quiet = env_truthy("YFINANCE_QUIET", default=True)
    if quiet:
        logging.getLogger("yfinance").setLevel(logging.ERROR)
    else:
        logging.getLogger("yfinance").setLevel(logging.INFO)

    import contextlib
    import io

    for i in range(0, len(tickers), group_size):
        chunk = tickers[i:i + group_size]
        try:
            if quiet:
                _buf_out = io.StringIO()
                _buf_err = io.StringIO()
                with contextlib.redirect_stdout(_buf_out), contextlib.redirect_stderr(_buf_err):
                    data = yf.download(
                        tickers=" ".join(chunk),
                        period=period,
                        auto_adjust=auto_adjust,
                        progress=False,
                        threads=True,
                        group_by="ticker",
                    )
            else:
                data = yf.download(
                    tickers=" ".join(chunk),
                    period=period,
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
        except Exception:
            time.sleep(pause_sec)
            continue

        if data is None or getattr(data, "empty", True):
            time.sleep(pause_sec)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for t in chunk:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t].copy().dropna(how="all")
                if df is not None and not df.empty:
                    out[t] = df
        else:
            t = chunk[0]
            df = data.copy().dropna(how="all")
            if df is not None and not df.empty:
                out[t] = df

        time.sleep(pause_sec)

    return out

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr14(df: pd.DataFrame) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(14).mean()

def adv20(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Volume" not in df.columns:
        return np.nan
    v = df["Volume"].astype(float)
    c = df["Close"].astype(float)
    val = (v * c).rolling(20).mean()
    return safe_float(val.iloc[-1], np.nan)

def atr_pct_last(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan
    a = atr14(df)
    c = df["Close"].astype(float)
    if len(a) == 0:
        return np.nan
    return safe_float((a.iloc[-1] / (c.iloc[-1] + 1e-9)) * 100.0, np.nan)


def efficiency_ratio(close: pd.Series, window: int = 60) -> float:
    """Kaufman Efficiency Ratio (ER).

    ER = |close(t) - close(t-window)| / sum(|diff(close)|)

    Range: [0, 1]
      - 1.0: straight / efficient trend
      - 0.0: choppy / mean-reverting
    """
    try:
        c = pd.Series(close).astype(float)
    except Exception:
        return float("nan")

    if c is None or len(c) < window + 1:
        return float("nan")

    seg = c.iloc[-(window + 1):]
    net = float(abs(seg.iloc[-1] - seg.iloc[0]))
    denom = float(seg.diff().abs().sum())
    if denom <= 0:
        return 0.0
    return float(net / denom)


def choppiness_index(df: pd.DataFrame, window: int = 14) -> float:
    """Choppiness Index (CHOP).

    Higher -> choppy / ranging. Lower -> trending.
    """
    try:
        if df is None or len(df) < window + 1:
            return float("nan")
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

        tr_sum = tr.rolling(window).sum()
        hh = high.rolling(window).max()
        ll = low.rolling(window).min()
        rng = (hh - ll).replace(0, np.nan)

        x = (tr_sum / rng).replace([np.inf, -np.inf], np.nan)
        chop = 100.0 * np.log10(x) / np.log10(float(window))
        return float(chop.iloc[-1])
    except Exception:
        return float("nan")


def adx(df: pd.DataFrame, window: int = 14) -> float:
    """Average Directional Index (ADX)."""
    try:
        if df is None or len(df) < window * 2 + 2:
            return float("nan")

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        alpha = 1.0 / float(window)
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-9)
        minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-9)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx_s = dx.ewm(alpha=alpha, adjust=False).mean()
        return float(adx_s.iloc[-1])
    except Exception:
        return float("nan")

def returns(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    return c.pct_change(fill_method=None)

def corr_60d(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    ra = returns(df_a).tail(60)
    rb = returns(df_b).tail(60)
    x = pd.concat([ra, rb], axis=1).dropna()
    if len(x) < 20:
        return np.nan
    return float(x.iloc[:, 0].corr(x.iloc[:, 1]))

def is_abnormal_stock(df: pd.DataFrame) -> bool:
    """
    異常変動の簡易除外（完全ではない）
    - 直近5日で|日次|>12%が3本以上
    - 直近3日連続で|日次|>12%
    """
    if df is None or len(df) < 10:
        return False
    r = returns(df).tail(5).abs()
    extreme = int((r > 0.12).sum())
    last3 = returns(df).tail(3).abs()
    last3_ext = bool((last3 > 0.12).all())
    return bool(extreme >= 3 or last3_ext)
