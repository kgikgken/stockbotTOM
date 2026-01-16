from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

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

    for i in range(0, len(tickers), group_size):
        chunk = tickers[i:i + group_size]
        try:
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
