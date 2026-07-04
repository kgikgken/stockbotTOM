"""OHLCV data layer — yfinance batched fetch, or synthetic data in DRYRUN.

出典表記: yfinance = Yahoo Finance 系の単一ソース。v4.1の規則上、
ここから算出した指標はすべて「未確認(単一ソース)」であり、確定判定には
ユーザーのブローカー画面照合(独立2ソース化)が必要。
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def fetch_ohlcv(tickers: List[str], history_days: int, dryrun: bool = False,
                ) -> Tuple[Dict[str, pd.DataFrame], dict]:
    if dryrun:
        return _synthetic(tickers)

    import yfinance as yf

    out: Dict[str, pd.DataFrame] = {}
    chunks = [tickers[i:i + 200] for i in range(0, len(tickers), 200)]
    period = f"{max(history_days, 400)}d"
    for ci, chunk in enumerate(chunks):
        for attempt in range(3):
            try:
                raw = yf.download(
                    tickers=" ".join(chunk), period=period, interval="1d",
                    group_by="ticker", auto_adjust=False, actions=False,
                    threads=True, progress=False,
                )
                break
            except Exception:
                if attempt == 2:
                    raw = None
                else:
                    time.sleep(3 * (attempt + 1))
        if raw is None or len(raw) == 0:
            continue
        if len(chunk) == 1:
            df = raw.dropna(how="all")
            if len(df):
                out[chunk[0]] = df
        else:
            for t in chunk:
                try:
                    df = raw[t].dropna(how="all")
                except Exception:
                    continue
                if len(df) >= 60:
                    out[t] = df
        time.sleep(1.0)

    meta = _coverage(tickers, out, source="yfinance(Yahoo Finance) 単一ソース")
    return out, meta


def _coverage(tickers, out, source):
    total = len(tickers)
    ok = len(out)
    return {
        "data_total": total,
        "data_ok": ok,
        "data_coverage": (ok / total) if total else 0.0,
        "source": source,
        "fetched_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST"),
    }


# ---------------------------------------------------------------- synthetic

def _mk_series(seed: int, n: int, mode: str) -> pd.DataFrame:
    """Deterministic OHLCV. mode: normal / oversold / overbought / crash_event / pump_event"""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    if end.dayofweek >= 5:  # 土日は直前の営業日に丸める(bdate_rangeの本数ズレ対策)
        end = end - pd.tseries.offsets.BDay(1)
    idx = pd.bdate_range(end=end, periods=n)
    drift = 0.0002
    ret = rng.normal(drift, 0.018, n)

    if mode == "oversold":
        ret[-30:] -= 0.009                      # ゆるやかな売られ過ぎ(ニュース無=E想定)
    elif mode == "overbought":
        ret[-25:] += 0.011
    elif mode == "crash_event":
        ret[-4] = -0.16                          # 直近に急落イベント(A疑い想定)
        ret[-3:] -= 0.01
    elif mode == "pump_event":
        ret[-3] = 0.17
        ret[-2:] += 0.012

    close = 1000.0 * np.exp(np.cumsum(ret))
    o = close * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(200_000, 900_000, n).astype(float)
    if mode == "crash_event":
        v[-4] *= 6.0
    if mode == "pump_event":
        v[-3] *= 6.0
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close, "Volume": v}, index=idx)


def _synthetic(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], dict]:
    modes = ["oversold", "crash_event", "overbought", "pump_event"] + ["normal"] * 8
    out: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers[:60]):
        out[t] = _mk_series(seed=42 + i, n=420, mode=modes[i % len(modes)] if i < 12 else "normal")
    meta = _coverage(tickers[:60], out, source="SYNTHETIC (DRYRUN・実データではない)")
    meta["dryrun"] = True
    return out, meta


def fetch_macro(dryrun: bool = False) -> dict:
    """STEP1 用の市場データ。欠落は None → 暫定扱い。"""
    if dryrun:
        return {
            "spx": +0.4, "ndx": +0.7, "dji": +0.2, "sox": +1.8,
            "usdjpy_1d": +0.2, "usdjpy_5d": +0.8, "usdjpy": 158.3,
            "nkfut": +0.6, "n225_rv20": 24.5, "vix": 15.2,
            "fetched_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST"),
            "provisional": [], "dryrun": True,
        }

    import yfinance as yf

    def last_chg(tk, days=1):
        try:
            h = yf.download(tk, period="3mo", interval="1d", progress=False, auto_adjust=False)
            c = h["Close"].dropna()
            if hasattr(c, "columns"):
                c = c.iloc[:, 0]
            if len(c) < days + 1:
                return None, None
            chg = (float(c.iloc[-1]) / float(c.iloc[-1 - days]) - 1.0) * 100.0
            return chg, float(c.iloc[-1])
        except Exception:
            return None, None

    prov = []
    spx, _ = last_chg("^GSPC"); ndx, _ = last_chg("^IXIC"); dji, _ = last_chg("^DJI")
    sox, _ = last_chg("^SOX")
    uj1, uj = last_chg("USDJPY=X", 1)
    uj5, _ = last_chg("USDJPY=X", 5)
    nk, _ = last_chg("NIY=F")
    if nk is None:
        nk, _ = last_chg("NKD=F")
    _, vix = last_chg("^VIX")

    # 日経VI: 手動値(env) > 実現ボラproxy(^N225 20日)
    rv = None
    try:
        h = yf.download("^N225", period="4mo", interval="1d", progress=False, auto_adjust=False)
        c = h["Close"].dropna()
        if hasattr(c, "columns"):
            c = c.iloc[:, 0]
        lr = np.log(c / c.shift(1)).dropna()
        if len(lr) >= 20:
            rv = float(lr.iloc[-20:].std(ddof=0) * np.sqrt(252) * 100.0)
    except Exception:
        rv = None

    for k, v in [("SPX", spx), ("NDX", ndx), ("SOX", sox), ("USDJPY", uj1),
                 ("日経先物", nk), ("N225実現ボラ", rv)]:
        if v is None:
            prov.append(k)

    return {
        "spx": spx, "ndx": ndx, "dji": dji, "sox": sox,
        "usdjpy_1d": uj1, "usdjpy_5d": uj5, "usdjpy": uj,
        "nkfut": nk, "n225_rv20": rv, "vix": vix,
        "fetched_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST"),
        "provisional": prov, "dryrun": False,
    }
