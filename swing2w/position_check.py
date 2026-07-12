"""swing2w 保有ポジションの監視 — 固定利確・初期ストップ・時間ストップの到達判定.

★momentum系の positions.csv とは別ファイル(positions_swing2w.csv)で管理する。
「別枠」というユーザー指定を、保有記録のレベルでも明確に分離するため。
自動決済はしない。到達時は警告のみ(発注は人間が行う)。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .data import fetch_ohlcv
from .indicators import compute_swing_features

POSITIONS_PATH = "positions_swing2w.csv"


def load_positions_swing2w(path: str = POSITIONS_PATH) -> pd.DataFrame:
    p = Path(path)
    cols = ["ticker", "shares", "entry_price", "entry_date", "stop_price", "target_price"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(p, dtype=str).fillna("")
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def _business_days_since(entry_date: str, today: str) -> int | None:
    try:
        d0 = pd.Timestamp(entry_date).date()
        d1 = pd.Timestamp(today).date()
    except Exception:
        return None
    if d1 < d0:
        return None
    return int(np.busday_count(d0, d1))


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
        urow = uni_idx.loc[ticker] if (uni_idx is not None and ticker in uni_idx.index) else None
        name = str(urow["name"]) if urow is not None else ticker
        code = ticker.replace(".T", "")

        df = ohlcv.get(ticker)
        if df is None:
            alerts.append({"code": code, "name": name, "hit": None,
                           "note": "データ取得不可(コード不一致・上場廃止等、要確認)"})
            continue
        feat = compute_swing_features(df, cfg)
        close_now = feat["close"] if feat else (float(df["Close"].dropna().iloc[-1]) if len(df["Close"].dropna()) else None)
        if close_now is None:
            alerts.append({"code": code, "name": name, "hit": None,
                           "note": f"データ不足(算出不可・取得{len(df)}行)"})
            continue

        notes = []
        hit = None
        breakeven_due = False

        entry_price = _to_float(p.get("entry_price"))
        target_price = _to_float(p.get("target_price"))
        stop_price = _to_float(p.get("stop_price"))
        if target_price is not None and close_now >= target_price:
            hit = "target"
            notes.append(f"利確目標({target_price:,.0f}円)に到達(終値{close_now:,.0f}円)。手仕舞い検討")
        elif stop_price is not None and close_now <= stop_price:
            hit = "stop"
            notes.append(f"初期ストップ({stop_price:,.0f}円)に到達(終値{close_now:,.0f}円)。手仕舞い検討")

        # ★指示: 精査で発覚した未実装機能を実装。1R到達でストップを建値へ移動する(利益の確保)。
        # entry_price/stop_priceの両方が記録されており、まだ建値に移動していない(stop<entry)場合のみ判定。
        if hit is None and entry_price is not None and stop_price is not None and entry_price > stop_price:
            risk_w = entry_price - stop_price
            current_r = (close_now - entry_price) / risk_w if risk_w > 0 else 0.0
            if current_r >= cfg.breakeven_trigger_r:
                breakeven_due = True
                notes.append(f"{current_r:.1f}R到達 — ストップを建値({entry_price:,.0f}円)へ引き上げ検討"
                            f"(現在のストップ{stop_price:,.0f}円のまま据え置き中)")

        entry_date = str(p.get("entry_date", "")).strip()
        days_held = _business_days_since(entry_date, today) if entry_date else None
        if hit is None and days_held is not None and days_held >= cfg.time_stop_days:
            hit = "time"
            notes.append(f"時間ストップ到達(建玉から{days_held}営業日経過・上限{cfg.time_stop_days}営業日)。手仕舞い検討")
        elif days_held is None and entry_date == "":
            notes.append("entry_date未記録のため時間ストップ判定不可(要追記)")

        alerts.append({
            "code": code, "name": name, "hit": hit, "close": close_now,
            "days_held": days_held, "breakeven_due": breakeven_due,
            "note": " / ".join(notes) if notes else "利確・ストップ・時間ストップいずれも未到達(平常)",
        })
    return alerts


def _to_float(v) -> float | None:
    try:
        s = str(v).strip()
        return float(s) if s and s.lower() != "nan" else None
    except Exception:
        return None
