"""universe_jpx.csv / positions.csv / events.csv の読込."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

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


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "shares", "entry_price", "stop_price"])
    df = pd.read_csv(p)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].map(norm_ticker)
    if "stop_price" not in df.columns:
        df["stop_price"] = float("nan")
    return df


def open_risk_from_positions(pos: pd.DataFrame, equity: float) -> tuple[float | None, str]:
    """既存ポジの総オープンリスク%(stop_price 登録がある場合のみ算定可能)."""
    if pos is None or len(pos) == 0:
        return 0.0, "既存ポジ: なし"
    if equity <= 0:
        return None, f"既存ポジ{len(pos)}件 — ACCOUNT_EQUITY未設定のためリスク%算定不可"
    if pos["stop_price"].isna().all():
        return None, f"既存ポジ{len(pos)}件 — positions.csv に stop_price 列が無くリスク%算定不可(要追記)"
    risk = 0.0
    for _, r in pos.iterrows():
        try:
            if pd.notna(r["stop_price"]):
                risk += max(0.0, (float(r["entry_price"]) - float(r["stop_price"]))) * float(r["shares"])
        except Exception:
            continue
    return risk / equity * 100.0, f"既存ポジ{len(pos)}件のオープンリスク {risk/equity*100.0:.2f}%(stop登録分)"


def load_week_events(path: str = "events.csv", today: date | None = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    today = today or date.today()
    end = today + timedelta(days=7)
    try:
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        w = df[(df["date"] >= today) & (df["date"] <= end)].sort_values("date")
        return [f"{r['date'].strftime('%m/%d')} {r['label']}" for _, r in w.iterrows()]
    except Exception:
        return []
