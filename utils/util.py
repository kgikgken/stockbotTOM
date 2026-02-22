from __future__ import annotations

import os
import time
import random
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
    """JPX tick size (呼値の単位) for the standard 'その他銘柄' table.

    Notes
    -----
    * JPX has different tick tables for some liquid constituents / ETFs.
      This function intentionally follows the *standard/others* table.
      If a symbol actually has *smaller* tick sizes, using this larger tick
      is still a valid price (multiple of the smaller tick) and avoids
      "order rejected due to invalid tick" incidents.
    * Price is assumed to be JPY (Tokyo Stock Exchange cash equities).
    """
    try:
        p = float(price)
    except Exception:
        return 1.0

    if not np.isfinite(p) or p <= 0:
        return 1.0

    # ---- Standard/others table (JPY) ----
    # <=1,000: 1
    if p <= 1000:
        return 1.0
    # 1,000< - 3,000: 1
    if p <= 3000:
        return 1.0
    # 3,000< - 5,000: 5
    if p <= 5000:
        return 5.0
    # 5,000< - 10,000: 10
    if p <= 10000:
        return 10.0
    # 10,000< - 30,000: 10
    if p <= 30000:
        return 10.0
    # 30,000< - 50,000: 50
    if p <= 50000:
        return 50.0
    # 50,000< - 100,000: 100
    if p <= 100000:
        return 100.0
    # 100,000< - 300,000: 100
    if p <= 300000:
        return 100.0
    # 300,000< - 500,000: 500
    if p <= 500000:
        return 500.0
    # 500,000< - 1,000,000: 1,000
    if p <= 1000000:
        return 1000.0
    # 1,000,000< - 3,000,000: 1,000
    if p <= 3000000:
        return 1000.0
    # 3,000,000< - 5,000,000: 5,000
    if p <= 5000000:
        return 5000.0
    # 5,000,000< - 10,000,000: 10,000
    if p <= 10000000:
        return 10000.0
    # 10,000,000< - 30,000,000: 10,000
    if p <= 30000000:
        return 10000.0
    # 30,000,000< - 50,000,000: 50,000
    if p <= 50000000:
        return 50000.0

    # >50,000,000
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

def append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    """Append a row to a CSV file (create with header if missing).

    This is used for lightweight logging (e.g., probability calibration logs)
    without adding any new dependencies.
    """
    try:
        p = Path(path)
        if p.parent and str(p.parent) not in ('.', ''):
            p.parent.mkdir(parents=True, exist_ok=True)
        exists = p.exists()
        with p.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, '') for k in fieldnames})
    except Exception:
        # never break the screener because of logging
        return


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
    tickers: list[str],
    period: str = "260d",
    interval: str = "1d",
    *,
    group_size: int = 200,
    pause_sec: float = 1.0,
    auto_adjust: bool = True,
    min_bars: int = 60,
    cache_dir: str | None = None,
    cache_max_age_days: int | None = None,
    retries: int | None = None,
    retry_base_sec: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Bulk download OHLCV via yfinance with retries + per-symbol cache.

    Why this exists
    ---------------
    * Yahoo/yfinance is prone to rate limits and intermittent empty responses.
    * A cache (optionally persisted via CI cache) dramatically reduces API hits.
    * Retries + backoff improves resilience without changing the strategy logic.

    Returns
    -------
    dict[symbol, DataFrame]
        Each DF has columns: Open, High, Low, Close, Volume.
    """

    if not tickers:
        return {}

    # de-duplicate while keeping order
    seen: set[str] = set()
    syms: list[str] = []
    for t in tickers:
        s = str(t).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        syms.append(s)

    if not syms:
        return {}

    # defaults from env
    if cache_dir is None:
        cache_dir = os.environ.get("YF_CACHE_DIR", "out/cache/yf")
    if cache_max_age_days is None:
        cache_max_age_days = int(os.environ.get("YF_CACHE_MAX_AGE_DAYS", "7"))
    if retries is None:
        retries = int(os.environ.get("YF_RETRY", "3"))
    if retry_base_sec is None:
        retry_base_sec = float(os.environ.get("YF_RETRY_BASE_SEC", "1.5"))

    use_cache = _env_truthy("YF_CACHE", True)

    cache_root = Path(cache_dir) if cache_dir else None
    if use_cache and cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    def _cache_path(sym: str) -> Path:
        safe = re.sub(r"[^0-9A-Za-z._^=\-]+", "_", sym)
        adj = "adj" if auto_adjust else "raw"
        return cache_root / f"{safe}_{period}_{interval}_{adj}.pkl"  # type: ignore[arg-type]

    def _cache_get(sym: str) -> pd.DataFrame | None:
        if not use_cache or cache_root is None:
            return None
        p = _cache_path(sym)
        if not p.exists():
            return None
        if cache_max_age_days and cache_max_age_days > 0:
            age_sec = max(0.0, time.time() - p.stat().st_mtime)
            if age_sec > float(cache_max_age_days) * 86400.0:
                return None
        try:
            df = pd.read_pickle(p)
            if df is None or df.empty:
                return None
            if len(df) < min_bars:
                return None
            return df
        except Exception:
            return None

    def _cache_put(sym: str, df: pd.DataFrame) -> None:
        if not use_cache or cache_root is None:
            return
        try:
            p = _cache_path(sym)
            df.to_pickle(p)
        except Exception:
            pass

    def _yf_download(chunk: list[str]) -> pd.DataFrame | None:
        # retry/backoff on hard failures
        last_err: Exception | None = None
        for attempt in range(max(0, retries) + 1):
            try:
                data = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                return data
            except Exception as e:
                last_err = e
                if attempt >= max(0, retries):
                    break
                sleep_s = float(retry_base_sec) * (2.0 ** attempt)
                # small jitter to avoid thundering herd in CI
                sleep_s += random.random() * 0.25
                time.sleep(sleep_s)
        if last_err is not None:
            logger.warning(f"yfinance download failed (chunk={len(chunk)}): {last_err}")
        return None

    # 1) try cache first
    out: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for s in syms:
        cached = _cache_get(s)
        if cached is not None:
            out[s] = cached
        else:
            need.append(s)

    # 2) download missing in chunks
    for i in range(0, len(need), max(1, group_size)):
        chunk = need[i : i + max(1, group_size)]
        if not chunk:
            continue

        data = _yf_download(chunk)
        if data is None or getattr(data, "empty", True):
            # fallback: try per-ticker (salvage)
            for sym in chunk:
                d1 = _yf_download([sym])
                if d1 is None or getattr(d1, "empty", True):
                    continue
                # normalize
                if isinstance(getattr(d1, "columns", None), pd.MultiIndex):
                    try:
                        df1 = d1[sym].copy().dropna(how="all")
                    except Exception:
                        continue
                else:
                    df1 = d1.copy().dropna(how="all")
                if df1 is None or df1.empty or len(df1) < min_bars:
                    continue
                out[sym] = df1
                _cache_put(sym, df1)
            time.sleep(pause_sec)
            continue

        # normalize: multi/single
        if isinstance(getattr(data, "columns", None), pd.MultiIndex):
            for sym in chunk:
                try:
                    df = data[sym].copy().dropna(how="all")
                except Exception:
                    continue
                if df is None or df.empty or len(df) < min_bars:
                    continue
                out[sym] = df
                _cache_put(sym, df)
        else:
            # sometimes yfinance returns a single-table even when chunk>1 (rare)
            if len(chunk) == 1:
                sym = chunk[0]
                df = data.copy().dropna(how="all")
                if df is not None and (not df.empty) and len(df) >= min_bars:
                    out[sym] = df
                    _cache_put(sym, df)

        time.sleep(pause_sec)

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
