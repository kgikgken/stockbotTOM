from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_blackouts_from_env() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    raw = os.getenv("SCREEN_BLACKOUT_CODES", "").strip()
    if raw:
        for part in raw.split(","):
            code = part.strip()
            if code:
                mapping[code] = "env:blackout"

    path = Path(os.getenv("BLACKOUTS_CSV", "data/blackouts.csv"))
    if path.exists():
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                ticker = str(row.get("ticker", "")).strip()
                if not ticker:
                    continue
                reason = str(row.get("reason", "blackout")).strip() or "blackout"
                mapping[ticker] = reason
        except Exception:
            pass
    return mapping


def blackout_reason(ticker: str, blackouts: Optional[Dict[str, str]] = None) -> str | None:
    mapping = blackouts or load_blackouts_from_env()
    code = str(ticker).strip()
    return mapping.get(code)
