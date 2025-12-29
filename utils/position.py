import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_positions(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def analyze_positions(pos_df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """positions.csv を軽く要約。total_asset はあれば列から推定。"""
    if pos_df is None or pos_df.empty:
        return "- なし", 0.0

    # total_asset を推定（あれば）
    total_asset = 0.0
    for col in ("total_asset", "asset", "equity"):
        if col in pos_df.columns:
            try:
                v = float(pos_df[col].dropna().iloc[0])
                if np.isfinite(v) and v > 0:
                    total_asset = v
                    break
            except Exception:
                pass

    lines = []
    # 想定列：ticker, pnl_pct, rr
    for _, r in pos_df.iterrows():
        t = str(r.get("ticker", "")).strip()
        if not t:
            continue
        pnl = r.get("pnl_pct", r.get("pnl", ""))
        rr = r.get("rr", "")
        try:
            pnl_f = float(pnl)
            pnl_s = f"{pnl_f:.2f}%"
        except Exception:
            pnl_s = str(pnl) if str(pnl) else "n/a"
        rr_s = f"{float(rr):.2f}R" if str(rr).strip() not in ("", "nan", "None") else ""
        if rr_s:
            lines.append(f"- {t}: 損益 {pnl_s} RR:{rr_s}")
        else:
            lines.append(f"- {t}: 損益 {pnl_s}")

    if not lines:
        lines = ["- なし"]

    return "\n".join(lines), float(total_asset)