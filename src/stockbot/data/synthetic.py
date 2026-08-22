"""DRYRUN 用の合成日足。ネットワーク無しでパイプライン全体を通すために使う。

銘柄ごとに「上昇トレンド＋押し目」「横ばい」「下落」「分割あり」「ギャップあり」を作り分ける。
統計的な現実性は求めない。後段の形状検出が動くことだけを保証する。
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .yf_fetch import COLS


def _bdays(n: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    end = (end or pd.Timestamp.today()).normalize()
    idx = pd.bdate_range(end=end, periods=n + 5)
    return idx[-n:]


def make_synthetic(tickers: Iterable[str], n_bars: int = 400, seed: int = 0,
                   end: pd.Timestamp | None = None) -> Dict[str, pd.DataFrame]:
    idx = _bdays(n_bars, end)
    out: Dict[str, pd.DataFrame] = {}
    kinds = ("pullback", "flat", "down", "pullback_split", "gap")
    for t in tickers:
        # 銘柄ごとに決定的な乱数列（再実行しても同じ系列 → 改訂検出の偽陽性を防ぐ）
        h = int.from_bytes(__import__("hashlib").md5(f"{seed}:{t}".encode()).digest()[:4], "little")
        rng = np.random.default_rng(h)
        kind = kinds[h % len(kinds)]
        base = float(rng.uniform(300, 8000))
        drift = {"pullback": 0.0012, "pullback_split": 0.0012, "flat": 0.0, "down": -0.0008, "gap": 0.0006}[kind]
        vol = float(rng.uniform(0.012, 0.025))
        ret = rng.normal(drift, vol, n_bars)
        if kind in ("pullback", "pullback_split"):
            # 末尾 12 本に押し目（緩い下落）、その後 3 本で反発の兆し
            ret[-15:-3] = rng.normal(-0.006, vol * 0.6, 12)
            ret[-3:] = rng.normal(0.008, vol * 0.6, 3)
        if kind == "gap":
            ret[-25] += 0.08
        close = base * np.exp(np.cumsum(ret))
        o = close * (1 + rng.normal(0, vol * 0.3, n_bars))
        h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, vol * 0.4, n_bars)))
        lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, vol * 0.4, n_bars)))
        v = rng.lognormal(mean=np.log(rng.uniform(5e4, 2e6)), sigma=0.5, size=n_bars)
        if kind in ("pullback", "pullback_split"):
            v[-15:-3] *= 0.6          # 押し目で出来高が枯れる
            v[-3:] *= 1.8             # 反発で増える
        if kind == "gap":
            v[-25] *= 3.0
        df = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": close, "Volume": v,
                           "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
        df.index.name = "Date"
        if kind == "pullback_split":
            # 100 本前に 1:2 分割。Yahoo 流に調整済みの系列として、分割イベントだけ記録する
            df.iloc[n_bars - 100, df.columns.get_loc("Stock Splits")] = 2.0
        out[t] = df[COLS]
    return out


def synthetic_listed(n: int = 60) -> pd.DataFrame:
    """合成ユニバース（JPX 正規化後の列）。"""
    rows = []
    sectors = ["食料品", "化学", "機械", "電気機器", "小売業", "情報・通信業", "サービス業"]
    for i in range(n):
        code = f"{1301 + i * 7:04d}"
        rows.append({"date": "", "code": code, "ticker": f"{code}.T", "name": f"合成{code}",
                     "market": "プライム（内国株式）" if i % 3 else "スタンダード（内国株式）",
                     "sector33_code": "", "sector33": sectors[i % len(sectors)],
                     "sector17_code": "", "sector17": "", "size_code": "", "size": "",
                     "is_equity": True})
    return pd.DataFrame(rows)
