"""ユニバース構築（SPEC §1）。

  1. JPX 上場銘柄一覧 → 株式のみ（ETF/ETN/REIT/インフラファンド/出資証券/PRO Market を除外、
     5桁コードの優先株等を除外）
  2. 日足から流動性統計を計算し、
       20日平均売買代金 ≥ MIN_ADV_JPY（既定 2億円）
       株価 ≥ MIN_PRICE（既定 200円）
       履歴 ≥ MIN_HISTORY_BARS（既定 250本）
     を満たす銘柄を採用
  3. 分割の整合性に疑いがある銘柄（adjust.py の suspected_unrecorded_split）は除外

市場区分は問わない（SPEC の決定）。区分は統計用に保持する。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

STATS_COLS = ["ticker", "last_date", "bars", "last_close", "adv_jpy", "adv_shares", "turnover_last"]


def equities_only(listed: pd.DataFrame) -> pd.DataFrame:
    return listed[listed["is_equity"].astype(bool)].reset_index(drop=True)


def liquidity_stats(ohlcv: Dict[str, pd.DataFrame], adv_window: int = 20) -> pd.DataFrame:
    """銘柄ごとの流動性統計。売買代金は 終値×出来高 で近似する（yfinance に売買代金は無い）。"""
    rows = []
    for t, df in ohlcv.items():
        if df is None or len(df) == 0:
            continue
        tail = df.tail(adv_window)
        turnover = tail["Close"] * tail["Volume"]
        rows.append({
            "ticker": t,
            "last_date": df.index[-1],
            "bars": int(len(df)),
            "last_close": float(df["Close"].iloc[-1]),
            "adv_jpy": float(turnover.mean()) if len(tail) else np.nan,
            "adv_shares": float(tail["Volume"].mean()) if len(tail) else np.nan,
            "turnover_last": float(turnover.iloc[-1]) if len(tail) else np.nan,
        })
    return pd.DataFrame(rows, columns=STATS_COLS)


def apply_filters(stats: pd.DataFrame, min_adv_jpy: float, min_price: float,
                  min_history_bars: int, exclude: Iterable[str] = (),
                  asof: pd.Timestamp | None = None, max_staleness_days: int = 7) -> pd.DataFrame:
    """各条件のフラグと総合 passes を付ける。

    ok_fresh: 最終足が asof から max_staleness_days 以内（保存データに残った上場廃止・取得失敗銘柄を落とす）
    """
    s = stats.copy()
    ex = set(exclude)
    s["ok_adv"] = s["adv_jpy"] >= min_adv_jpy
    s["ok_price"] = s["last_close"] >= min_price
    s["ok_history"] = s["bars"] >= min_history_bars
    s["ok_split"] = ~s["ticker"].isin(ex)
    if asof is not None and len(s):
        limit = pd.Timestamp(asof).tz_localize(None).normalize() - pd.Timedelta(days=max_staleness_days)
        s["ok_fresh"] = pd.to_datetime(s["last_date"]) >= limit
    else:
        s["ok_fresh"] = True
    s["passes"] = s["ok_adv"] & s["ok_price"] & s["ok_history"] & s["ok_split"] & s["ok_fresh"]
    return s


def build_universe(listed: pd.DataFrame, stats: pd.DataFrame, min_adv_jpy: float,
                   min_price: float, min_history_bars: int,
                   exclude: Iterable[str] = (), asof: pd.Timestamp | None = None,
                   max_staleness_days: int = 7) -> pd.DataFrame:
    """上場一覧（株式のみ）と流動性統計を結合し、採用フラグ付きの表を返す。"""
    eq = equities_only(listed)[["ticker", "code", "name", "market", "sector33", "size"]]
    f = apply_filters(stats, min_adv_jpy, min_price, min_history_bars, exclude, asof, max_staleness_days)
    u = eq.merge(f, on="ticker", how="left")
    for c in ("ok_adv", "ok_price", "ok_history", "ok_split", "ok_fresh", "passes"):
        u[c] = u[c].fillna(False).astype(bool)
    return u.sort_values(["passes", "adv_jpy"], ascending=[False, False]).reset_index(drop=True)


def save_universe(u: pd.DataFrame, universe_dir: Path, asof: pd.Timestamp) -> tuple[Path, Path]:
    universe_dir = Path(universe_dir)
    universe_dir.mkdir(parents=True, exist_ok=True)
    dated = universe_dir / f"universe_{pd.Timestamp(asof).strftime('%Y-%m-%d')}.csv"
    latest = universe_dir / "latest.csv"
    u.to_csv(dated, index=False, encoding="utf-8-sig")
    u.to_csv(latest, index=False, encoding="utf-8-sig")
    return dated, latest


def load_latest_universe(universe_dir: Path) -> pd.DataFrame | None:
    p = Path(universe_dir) / "latest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"code": str})


def summarize(u: pd.DataFrame) -> dict:
    return {
        "equities": int(len(u)),
        "with_data": int(u["bars"].notna().sum()) if "bars" in u else 0,
        "passes": int(u["passes"].sum()),
        "fail_adv": int((~u["ok_adv"] & u["bars"].notna()).sum()),
        "fail_price": int((~u["ok_price"] & u["bars"].notna()).sum()),
        "fail_history": int((~u["ok_history"] & u["bars"].notna()).sum()),
        "fail_split": int((~u["ok_split"] & u["bars"].notna()).sum()),
        "fail_fresh": int((~u["ok_fresh"] & u["bars"].notna()).sum()),
        "by_market": {str(k): int(v) for k, v in u[u["passes"]]["market"].value_counts().items()},
    }
