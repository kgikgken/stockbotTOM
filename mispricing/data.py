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

    period = f"{max(history_days, 400)}d"

    def _download_chunk(chunk: List[str], attempts: int, base_backoff: float):
        for attempt in range(attempts):
            try:
                return yf.download(
                    tickers=" ".join(chunk), period=period, interval="1d",
                    group_by="ticker", auto_adjust=False, actions=False,
                    threads=True, progress=False,
                )
            except Exception:
                if attempt == attempts - 1:
                    return None
                time.sleep(base_backoff * (attempt + 1))
        return None

    def _extract(raw, chunk: List[str], out: Dict[str, pd.DataFrame]):
        if raw is None or len(raw) == 0:
            return
        if len(chunk) == 1:
            df = raw.dropna(how="all")
            if len(df):
                out[chunk[0]] = df
            return
        for t in chunk:
            try:
                df = raw[t].dropna(how="all")
            except Exception:
                continue
            if len(df) >= 60:
                out[t] = df

    out: Dict[str, pd.DataFrame] = {}
    chunks = [tickers[i:i + 200] for i in range(0, len(tickers), 200)]
    for chunk in chunks:
        raw = _download_chunk(chunk, attempts=3, base_backoff=3)
        _extract(raw, chunk, out)
        time.sleep(1.0)

    # ★2巡目(新規): 1巡目で欠落した銘柄だけ、小さいチャンクで再取得を試みる。
    # チャンク単位の一時的な失敗(該当チャンク内の1銘柄の問題が全体に波及する等)による
    # 取りこぼしを回収する狙い。1巡目の成功分の挙動は変えない(追加のみ・既存動作は不変)。
    first_pass_missing = [t for t in tickers if t not in out]
    if first_pass_missing:
        retry_chunks = [first_pass_missing[i:i + 50] for i in range(0, len(first_pass_missing), 50)]
        for chunk in retry_chunks:
            raw = _download_chunk(chunk, attempts=2, base_backoff=4)
            _extract(raw, chunk, out)
            time.sleep(1.5)

    meta = _coverage(tickers, out, source="yfinance(Yahoo Finance) 単一ソース")
    meta["recovered_2nd_pass"] = len(first_pass_missing) - len([t for t in first_pass_missing if t not in out])
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
        ret[-3:] -= 0.045                       # 直近3日で急落(ゆるやかな30日下げでなく鋭い押し目)
    elif mode == "overbought":
        ret[-3:] += 0.05
    elif mode == "crash_event":
        ret[-4] = -0.22                          # 深い急落(オーバーシュート要件を確実にクリア)
        ret[-3:-1] -= 0.01
    elif mode == "pump_event":
        ret[-3] = 0.20
        ret[-2:] += 0.012

    close = 1000.0 * np.exp(np.cumsum(ret))

    if mode in ("oversold", "crash_event", "overbought", "pump_event"):
        base = float(close[-13])
        if mode == "oversold":
            close[-13:-5] = base * (1.0 + np.linspace(0, 0.012, 8))
            peak = base * 1.02
            close[-5] = peak
            close[-4:] = peak * np.array([0.97, 0.935, 0.90, 0.865])
        elif mode == "crash_event":
            close[-13:-5] = base * (1.0 + np.linspace(0, 0.01, 8))
            peak = base * 1.012
            close[-5] = peak
            close[-4] = peak * 0.80                       # 単日急落(-20%)
            close[-3:] = close[-4] * np.array([0.99, 1.00, 0.985])
        elif mode == "overbought":
            close[-13:-5] = base * (1.0 - np.linspace(0, 0.012, 8))
            trough = base * 0.98
            close[-5] = trough
            close[-4:] = trough * np.array([1.03, 1.065, 1.10, 1.135])
        elif mode == "pump_event":
            close[-13:-5] = base * (1.0 - np.linspace(0, 0.01, 8))
            trough = base * 0.988
            close[-5] = trough
            close[-4] = trough * 1.22                      # 単日急騰(+22%)
            close[-3:] = close[-4] * np.array([1.01, 1.00, 1.015])

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
