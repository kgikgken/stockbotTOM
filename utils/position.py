from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from utils.util import safe_float


POSITION_ALIASES = {
    "ticker": ["ticker", "code", "symbol"],
    "name": ["name", "company", "銘柄"],
    "qty": ["qty", "quantity", "shares"],
    "avg_price": ["avg_price", "avg", "cost", "entry_price"],
    "last_price": ["last_price", "last", "current_price", "price"],
    "sl": ["sl", "stop", "stop_price"],
    "tp1": ["tp1", "take_profit", "target"],
    "side": ["side", "direction"],
}


def _pick_column(df: pd.DataFrame, names: list[str]) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def load_positions(csv_path: str | Path = "positions.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "qty", "avg_price", "last_price", "sl", "tp1", "side"])
    out = pd.DataFrame()
    for std_name, aliases in POSITION_ALIASES.items():
        col = _pick_column(df, aliases)
        if col is not None:
            out[std_name] = df[col]
        else:
            out[std_name] = np.nan
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip()
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out["side"] = out["side"].fillna("long").astype(str).str.strip()
    for col in ["qty", "avg_price", "last_price", "sl", "tp1"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["ticker"] != ""].reset_index(drop=True)
    return out


def analyze_positions(
    pos_df: pd.DataFrame,
    mkt_score: int,
    macro_on: bool,
    new_tickers: Iterable[str] | None = None,
) -> Tuple[str, pd.DataFrame]:
    df = normalize_positions(pos_df)
    if df.empty:
        return "保有なし", pd.DataFrame(columns=["ticker", "name", "avg", "last", "pnl_pct", "sl", "tp1", "R", "flag"])

    new_set = {str(x).strip() for x in (new_tickers or [])}
    rows = []
    pnl_values: list[float] = []
    for _, row in df.iterrows():
        avg_px = safe_float(row.get("avg_price"))
        last_px = safe_float(row.get("last_price"), avg_px)
        sl = safe_float(row.get("sl"))
        tp1 = safe_float(row.get("tp1"))
        pnl_pct = (last_px / avg_px - 1.0) * 100.0 if avg_px > 0 and np.isfinite(last_px) else float("nan")
        pnl_values.append(pnl_pct)
        r_multiple = float("nan")
        if avg_px > 0 and np.isfinite(sl) and avg_px > sl:
            r_multiple = (last_px - avg_px) / (avg_px - sl)
        flag_parts = []
        if str(row.get("ticker")) in new_set:
            flag_parts.append("NEW")
        if macro_on:
            flag_parts.append("macro")
        if int(mkt_score) < 45:
            flag_parts.append("defense")
        rows.append(
            {
                "ticker": str(row.get("ticker")),
                "name": str(row.get("name") or ""),
                "avg": avg_px,
                "last": last_px,
                "pnl_pct": pnl_pct,
                "sl": sl,
                "tp1": tp1,
                "R": r_multiple,
                "flag": ",".join(flag_parts) if flag_parts else "-",
            }
        )

    out = pd.DataFrame(rows)
    mean_pnl = float(np.nanmean(pnl_values)) if pnl_values else float("nan")
    winners = int(sum(1 for x in pnl_values if np.isfinite(x) and x > 0))
    total = len(pnl_values)
    text = f"保有 {total}件 / 勝ち {winners}/{total} / 平均 {mean_pnl:+.1f}%"
    return text, out
