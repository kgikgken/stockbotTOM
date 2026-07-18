"""topdown/universe.py — universe_jpx.csvの読込+ティッカー正規化(自立版・旧mispricingから逐語移植)."""

from __future__ import annotations

import re

import pandas as pd


def norm_ticker(s: str) -> str:
    s = str(s).strip().upper()
    if s.endswith(".T"):
        return s
    m = re.fullmatch(r"(\d{3}[0-9A-Z])T", s)   # 例: 186AT → 186A.T / 8035T → 8035.T
    if m:
        return m.group(1) + ".T"
    if re.fullmatch(r"\d{3}[0-9A-Z]", s):
        return s + ".T"
    return s


def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["ticker"] = df["ticker"].map(norm_ticker)
    for col in ("name", "sector", "market"):
        if col not in df.columns:
            df[col] = ""
    return df
