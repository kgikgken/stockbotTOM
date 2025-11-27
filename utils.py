
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- load_universe ---
def load_universe(path="universe_jpx.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            return df["ticker"].astype(str).tolist()
    return [
        "6920.T", "8035.T", "4502.T", "9984.T", "8316.T",
        "7203.T", "6861.T", "4063.T", "7735.T", "9433.T"
    ]

# placeholder, user should insert rest of utils here manually.
