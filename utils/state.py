from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.util import jst_now, safe_float

STATE_PATH = "state.json"

def _default_state() -> Dict[str, Any]:
    return {
        "version": "v2.3",
        "week_id": "",
        "weekly_new_count": 0,
        "market_scores": [],  # [{"date": "...", "score": int}]
        "cooldowns": {
            "tier0_exception_until": None,
            "distortion_until": None,
        },
        "paper_trades": {
            "tier0_exception": [],
            "distortion": [],
        },
        # Snapshot of last-seen open positions (tickers). Used to infer weekly new count
        # without requiring an explicit 'open_date' column.
        "positions_last": [],
    }

def load_state(path: str = STATE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return _default_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if not isinstance(st, dict):
            return _default_state()
        d = _default_state()
        d.update(st)
        d.setdefault("cooldowns", _default_state()["cooldowns"])
        d.setdefault("paper_trades", _default_state()["paper_trades"])
        d.setdefault("positions_last", _default_state()["positions_last"])
        if not isinstance(d.get("positions_last"), list):
            d["positions_last"] = []
        return d
    except Exception:
        return _default_state()

def save_state(state: Dict[str, Any], path: str = STATE_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _week_id(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

def update_week(state: Dict[str, Any]) -> None:
    wid = _week_id(jst_now())
    if state.get("week_id") != wid:
        state["week_id"] = wid
        state["weekly_new_count"] = 0

def inc_weekly_new(state: Dict[str, Any], n: int = 1) -> None:
    state["weekly_new_count"] = int(state.get("weekly_new_count", 0)) + int(n)


def update_weekly_from_positions(state: Dict[str, Any], current_tickers: List[str]) -> int:
    """Infer weekly new positions by diffing today's positions against the last snapshot.

    Rationale:
      - This project does not execute orders itself.
      - The only reliable source of "what was actually opened" is the user's positions.csv.
      - By snapshotting tickers, we can increment weekly_new_count when new tickers appear.

    Returns:
      number of newly detected tickers
    """
    cur = [str(x).strip() for x in (current_tickers or []) if str(x).strip()]
    cur_set = set(cur)
    prev = state.get("positions_last", [])
    prev_set = set([str(x).strip() for x in prev]) if isinstance(prev, list) else set()
    new = sorted(list(cur_set - prev_set))
    if new:
        inc_weekly_new(state, n=len(new))
    # Always refresh snapshot (closures should be reflected too).
    state["positions_last"] = sorted(list(cur_set))
    return int(len(new))
def weekly_left(state: Dict[str, Any], max_new: int = 3) -> Tuple[int, int]:
    used = int(state.get("weekly_new_count", 0))
    return used, max_new

def add_market_score(state: Dict[str, Any], date_str: str, score: int) -> float:
    lst = state.get("market_scores", [])
    if not isinstance(lst, list):
        lst = []
    lst = [x for x in lst if x.get("date") != date_str]
    lst.append({"date": date_str, "score": int(score)})
    lst = sorted(lst, key=lambda x: x.get("date", ""))[-10:]
    state["market_scores"] = lst
    if len(lst) >= 4:
        return float(int(lst[-1]["score"]) - int(lst[-4]["score"]))
    return 0.0

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def in_cooldown(state: Dict[str, Any], key: str) -> bool:
    dt = _parse_iso(state.get("cooldowns", {}).get(key))
    return bool(dt is not None and jst_now() < dt)

def set_cooldown_days(state: Dict[str, Any], key: str, days: int) -> None:
    until = jst_now() + timedelta(days=int(days))
    state.setdefault("cooldowns", {})[key] = until.isoformat()

def _trim(trades: List[Dict[str, Any]], keep: int = 80) -> List[Dict[str, Any]]:
    return trades[-keep:] if len(trades) > keep else trades

def record_paper_trade(
    state: Dict[str, Any],
    bucket: str,
    ticker: str,
    date_str: str,
    entry: float,
    sl: float,
    tp2: float,
    expected_r: float,
) -> None:
    tr = {
        "ticker": ticker,
        "open_date": date_str,
        "entry": float(entry),
        "sl": float(sl),
        "tp2": float(tp2),
        "expected_r": float(expected_r),
        "status": "OPEN",
        "close_date": None,
        "realized_r": None,
    }
    pt = state.setdefault("paper_trades", {}).setdefault(bucket, [])
    if isinstance(pt, list):
        pt.append(tr)
        state["paper_trades"][bucket] = _trim(pt)

def update_paper_trades_with_ohlc(
    state: Dict[str, Any],
    bucket: str,
    ohlc_map: Dict[str, pd.DataFrame],
    today_str: str,
) -> None:
    pt = state.get("paper_trades", {}).get(bucket, [])
    if not isinstance(pt, list) or not pt:
        return

    for tr in pt:
        if tr.get("status") != "OPEN":
            continue
        t = tr.get("ticker")
        df = ohlc_map.get(t)
        if df is None or df.empty:
            continue

        try:
            sub = df[df.index.strftime("%Y-%m-%d") >= tr.get("open_date", "")]
        except Exception:
            sub = df
        if sub is None or sub.empty:
            continue

        entry = safe_float(tr.get("entry"), np.nan)
        sl = safe_float(tr.get("sl"), np.nan)
        tp2 = safe_float(tr.get("tp2"), np.nan)
        if not (np.isfinite(entry) and np.isfinite(sl) and np.isfinite(tp2) and entry > sl and tp2 > entry):
            continue

        rr = (tp2 - entry) / max(entry - sl, 1e-9)
        hit_tp = bool((sub["High"].astype(float) >= tp2).any())
        hit_sl = bool((sub["Low"].astype(float) <= sl).any())

        if hit_sl:
            tr["status"] = "CLOSED"
            tr["close_date"] = today_str
            tr["realized_r"] = -1.0
        elif hit_tp:
            tr["status"] = "CLOSED"
            tr["close_date"] = today_str
            tr["realized_r"] = float(rr)

    state["paper_trades"][bucket] = _trim(pt)

def kpi_distortion(state: Dict[str, Any]) -> Dict[str, float]:
    pt = state.get("paper_trades", {}).get("distortion", [])
    if not isinstance(pt, list) or not pt:
        return {"median_r": 0.0, "exp_gap": 0.0, "neg_streak": 0.0, "count": 0.0}

    closed = [x for x in pt if x.get("status") == "CLOSED" and x.get("realized_r") is not None]
    if not closed:
        return {"median_r": 0.0, "exp_gap": 0.0, "neg_streak": 0.0, "count": 0.0}

    last20 = closed[-20:]
    r = np.array([safe_float(x.get("realized_r"), 0.0) for x in last20], dtype=float)
    e = np.array([safe_float(x.get("expected_r"), 0.0) for x in last20], dtype=float)

    median_r = float(np.median(r)) if len(r) else 0.0
    exp_gap = float(np.median(r - e)) if len(r) and len(e) else 0.0

    neg_streak = 0
    for x in reversed(last20):
        rr = safe_float(x.get("realized_r"), 0.0)
        if rr < 0:
            neg_streak += 1
        else:
            break

    return {"median_r": median_r, "exp_gap": exp_gap, "neg_streak": float(neg_streak), "count": float(len(last20))}
