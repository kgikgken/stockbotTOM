"""topdown 保有ポジションの監視 — 固定利確・初期ストップ・時間ストップ(保有期間タグ別)の到達判定.

positions_topdown.csv(専用ファイル・旧システムのpositions.csvとは別)で管理する。
列: ticker,shares,entry_price,entry_date,stop_price,target_price,hold_tag
hold_tag: 「短期スイング」(時間ストップ5営業日) / 「スイング」(同10営業日)。空欄はスイング扱い。
自動決済はしない。到達時は警告のみ(発注は人間が行う)。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .data import fetch_ohlcv

POSITIONS_PATH = "positions_topdown.csv"
COLS = ["ticker", "shares", "entry_price", "entry_date", "stop_price", "target_price", "hold_tag"]


def load_positions_topdown(path: str = POSITIONS_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(p, dtype=str).fillna("")
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    # ティッカー正規化(例: 186AT → 186A.T)。表記ゆれによるデータ取得不可を予防する。
    from .universe import norm_ticker
    df["ticker"] = df["ticker"].map(norm_ticker)
    return df


def _to_float(v) -> float | None:
    try:
        s = str(v).strip()
        return float(s) if s and s.lower() != "nan" else None
    except Exception:
        return None


def _bdays_since(entry_date: str, today: str) -> int | None:
    try:
        d0, d1 = pd.Timestamp(entry_date).date(), pd.Timestamp(today).date()
    except Exception:
        return None
    return int(np.busday_count(d0, d1)) if d1 >= d0 else None


def check_held_positions(pos_df: pd.DataFrame, universe: pd.DataFrame, cfg: Config, today: str) -> List[dict]:
    if pos_df is None or len(pos_df) == 0:
        return []
    tickers = [str(t).strip() for t in pos_df["ticker"].tolist() if str(t).strip()]
    ohlcv, _ = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    uni_idx = universe.drop_duplicates("ticker").set_index("ticker") if len(universe) else None

    alerts: List[dict] = []
    for _, p in pos_df.iterrows():
        ticker = str(p["ticker"]).strip()
        if not ticker:
            continue
        code = ticker.replace(".T", "")
        name = str(uni_idx.loc[ticker]["name"]) if (uni_idx is not None and ticker in uni_idx.index) else ticker
        try:
            df = ohlcv.get(ticker)
            if df is None:
                alerts.append({"code": code, "name": name, "hit": None,
                               "note": "データ取得不可(コード表記・上場廃止等、要確認)"})
                continue
            close_s = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
            if not len(close_s):
                alerts.append({"code": code, "name": name, "hit": None,
                               "note": f"データ不足(算出不可・取得{len(df)}行)"})
                continue
            close_now = float(close_s.iloc[-1])

            notes, hit = [], None
            target = _to_float(p.get("target_price"))
            stop = _to_float(p.get("stop_price"))
            if target is not None and close_now >= target:
                hit = "target"
                notes.append(f"利確目標({target:,.0f}円)到達(終値{close_now:,.0f}円)。手仕舞い検討")
            elif stop is not None and close_now <= stop:
                hit = "stop"
                notes.append(f"初期ストップ({stop:,.0f}円)到達(終値{close_now:,.0f}円)。手仕舞い検討")

            tag = str(p.get("hold_tag", "")).strip() or "スイング"
            limit = cfg.time_stop_short_swing if tag == "短期スイング" else cfg.time_stop_swing
            entry_date = str(p.get("entry_date", "")).strip()
            days = _bdays_since(entry_date, today) if entry_date else None
            if hit is None and days is not None and days >= limit:
                hit = "time"
                notes.append(f"時間ストップ到達({tag}: {days}営業日経過・上限{limit})。手仕舞い検討")
            elif days is None and not entry_date:
                notes.append("entry_date未記録のため時間ストップ判定不可(要追記)")

            alerts.append({"code": code, "name": name, "hit": hit, "close": close_now,
                           "days_held": days,
                           "note": " / ".join(notes) if notes else "利確・ストップ・時間ストップいずれも未到達(平常)"})
        except Exception as e:
            print(f"[WARN] topdown保有チェックで例外(ticker={ticker}): {type(e).__name__}: {e}")
            alerts.append({"code": code, "name": name, "hit": None,
                           "note": "処理中に例外(コード表記等を要確認・Actionsログ参照)"})
    return alerts
