from __future__ import annotations

import os
import pandas as pd


def load_positions(csv_path: str = "positions.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["ticker", "rr", "adj_ev"])
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        if "ticker" not in df.columns:
            return pd.DataFrame(columns=["ticker", "rr", "adj_ev"])
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "rr", "adj_ev"])
