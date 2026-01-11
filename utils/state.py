from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any

from utils.util import jst_now

def _week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"

def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"market_scores": [], "weekly_key": _week_key(jst_now()), "weekly_new": 0, "delta3d": 0.0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if not isinstance(st, dict):
            raise ValueError("state not dict")
        st.setdefault("market_scores", [])
        st.setdefault("weekly_key", _week_key(jst_now()))
        st.setdefault("weekly_new", 0)
        st.setdefault("delta3d", 0.0)
        return st
    except Exception:
        return {"market_scores": [], "weekly_key": _week_key(jst_now()), "weekly_new": 0, "delta3d": 0.0}

def _calc_delta3d(scores) -> float:
    try:
        s = [float(x) for x in scores if x is not None]
        if len(s) < 4:
            return 0.0
        return float(s[-1] - s[-4])
    except Exception:
        return 0.0

def update_state_after_run(path: str, state: Dict[str, Any], mkt_score: int) -> None:
    try:
        now = jst_now()
        wk = _week_key(now)
        if state.get("weekly_key") != wk:
            state["weekly_key"] = wk
            state["weekly_new"] = 0

        ms = state.get("market_scores", [])
        if not isinstance(ms, list):
            ms = []
        ms.append(int(mkt_score))
        ms = ms[-14:]
        state["market_scores"] = ms
        state["delta3d"] = _calc_delta3d(ms)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception:
        return
