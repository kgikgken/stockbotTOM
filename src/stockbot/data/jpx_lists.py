"""JPX 公開リスト（SPEC §4 ソース）。

1. 上場銘柄一覧 data_j.xls
   列: 日付, コード, 銘柄名, 市場・商品区分, 33業種コード, 33業種区分, 17業種コード, 17業種区分, 規模コード, 規模区分
   取得に失敗したら旧リポジトリの universe_jpx.csv を種として使う。

2. 決算発表予定日 … JPX の公開ページは書式が変わり得るため、本モジュールでは
   reference/earnings_schedule.csv (code,date) を読む関数だけを提供する。
   自動取得は取得経路を確認してから別途追加する（SPEC §8 未決）。

3. 上場廃止銘柄一覧 … 同様に reference/delistings.csv (code,name,date,reason) を読む。
   生存バイアスの上限評価（SPEC §4 緩和策 1）に使う。
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import pandas as pd

LISTED_COLS = ["date", "code", "ticker", "name", "market", "sector33_code", "sector33",
               "sector17_code", "sector17", "size_code", "size", "is_equity"]

_JP_COLS = {
    "日付": "date", "コード": "code", "銘柄名": "name", "市場・商品区分": "market",
    "33業種コード": "sector33_code", "33業種区分": "sector33",
    "17業種コード": "sector17_code", "17業種区分": "sector17",
    "規模コード": "size_code", "規模区分": "size",
}

# 株式以外（市場・商品区分の文字列で判定）
NON_EQUITY_PATTERNS = ("ETF", "ETN", "REIT", "インフラファンド", "ベンチャーファンド",
                       "カントリーファンド", "出資証券", "PRO Market", "PRO MARKET")


def norm_ticker(code: str) -> str:
    """'1301' / '130A' / '1301T' / '1301.T' → '1301.T'。5桁コード（優先株等）はそのまま返す。"""
    s = str(code).strip().upper()
    if s.endswith(".T"):
        return s
    m = re.fullmatch(r"(\d{3}[0-9A-Z])T", s)
    if m:
        return m.group(1) + ".T"
    if re.fullmatch(r"\d{3}[0-9A-Z]", s):
        return s + ".T"
    return s


def is_equity_market(market: str) -> bool:
    m = str(market)
    if any(p in m for p in NON_EQUITY_PATTERNS):
        return False
    return ("内国株式" in m) or ("外国株式" in m) or m in ("Prime", "Standard", "Growth")


def normalize_listed(raw: pd.DataFrame) -> pd.DataFrame:
    """JPX の列名（日本語）または旧 universe_jpx.csv の列名を LISTED_COLS に揃える。"""
    df = raw.copy()
    df = df.rename(columns={k: v for k, v in _JP_COLS.items() if k in df.columns})
    if "ticker" in df.columns and "code" not in df.columns:
        df["code"] = df["ticker"].astype(str).str.replace(".T", "", regex=False)
    if "sector" in df.columns and "sector33" not in df.columns:
        df["sector33"] = df["sector"]
    for c in LISTED_COLS:
        if c not in df.columns:
            df[c] = "" if c != "is_equity" else True
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["ticker"] = df["code"].map(norm_ticker)
    df["is_equity"] = df["market"].map(is_equity_market) & (df["code"].str.len() == 4)
    if df["date"].astype(str).str.len().gt(0).any():
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce").dt.strftime("%Y-%m-%d")
    return df[LISTED_COLS].drop_duplicates(subset=["code"]).reset_index(drop=True)


def load_listed(source: str | Path, timeout: int = 60) -> pd.DataFrame:
    """URL または ローカルの .xls/.xlsx/.csv から上場銘柄一覧を読み、正規化して返す。"""
    src = str(source)
    if src.startswith("http://") or src.startswith("https://"):
        import requests  # 遅延 import

        r = requests.get(src, timeout=timeout, headers={"User-Agent": "stockbotTOM/0.2"})
        r.raise_for_status()
        raw = pd.read_excel(io.BytesIO(r.content))
    elif src.lower().endswith((".xls", ".xlsx")):
        raw = pd.read_excel(src)
    else:
        raw = pd.read_csv(src, dtype=str).fillna("")
    return normalize_listed(raw)


def load_listed_with_fallback(url: str, seed_csv: Path, log=print) -> tuple[pd.DataFrame, str]:
    """JPX から取得し、失敗したら種 CSV を使う。戻り値: (listed, source_label)"""
    try:
        df = load_listed(url)
        if len(df) > 1000:
            return df, "jpx"
        log(f"[listed] JPX の行数が少なすぎる({len(df)}) → 種CSVにフォールバック")
    except Exception as e:  # ネットワーク・書式変更
        log(f"[listed] JPX 取得失敗: {type(e).__name__}: {e} → 種CSVにフォールバック")
    return load_listed(seed_csv), "seed"


def load_earnings_schedule(path: Path) -> pd.DataFrame:
    """reference/earnings_schedule.csv (code,date[,name]) → DataFrame[ticker,date]。無ければ空。"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "date"])
    df = pd.read_csv(p, dtype=str).fillna("")
    df["ticker"] = df["code"].map(norm_ticker)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["date"])[["ticker", "date"]].drop_duplicates()


def load_delistings(path: Path) -> pd.DataFrame:
    """reference/delistings.csv (code,name,date,reason) → DataFrame。無ければ空。"""
    p = Path(path)
    cols = ["ticker", "name", "date", "reason"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(p, dtype=str).fillna("")
    df["ticker"] = df["code"].map(norm_ticker)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df[cols]


def next_earnings_days(schedule: pd.DataFrame, asof: pd.Timestamp, ticker: str) -> Optional[int]:
    """asof 以降で最も近い決算発表日までの暦日数。無ければ None。"""
    if schedule is None or len(schedule) == 0:
        return None
    s = schedule[(schedule["ticker"] == ticker) & (schedule["date"] >= asof)]
    if len(s) == 0:
        return None
    return int((s["date"].min() - asof).days)
