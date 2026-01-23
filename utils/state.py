from __future__ import annotations

import json
import os
from typing import Any, Dict

DEFAULT_STATE_PATH = os.getenv("STOCKBOTTOM_STATE_PATH", "state.json")


def load_state(path: str = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, Any], path: str = DEFAULT_STATE_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        return
