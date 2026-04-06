from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from utils.util import env_int, jst_today_date


STATE_PATH = Path("state.json")


@dataclass
class WeeklyState:
    iso_year: int
    iso_week: int
    count: int = 0


def _current_week_key(today: date | None = None) -> tuple[int, int]:
    d = today or jst_today_date()
    iso = d.isocalendar()
    return int(iso.year), int(iso.week)


def load_state(path: str | Path = STATE_PATH) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict, path: str | Path = STATE_PATH) -> None:
    p = Path(path)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def update_week(state: Dict, today: date | None = None) -> None:
    year, week = _current_week_key(today)
    if state.get("week_year") != year or state.get("week_no") != week:
        state["week_year"] = year
        state["week_no"] = week
        state["weekly_new_count"] = 0
        state["weekly_new_tickers"] = []


def weekly_left(state: Dict, max_new: int) -> Tuple[int, int]:
    used = int(state.get("weekly_new_count", 0) or 0)
    return used, int(max_new)


def update_weekly_from_positions(state: Dict, tickers: Iterable[str]) -> None:
    current = sorted({str(x).strip() for x in tickers if str(x).strip()})
    prev = sorted({str(x).strip() for x in state.get("positions_last", []) if str(x).strip()})
    new_names = sorted(set(current) - set(prev))
    if new_names:
        seen = set(str(x).strip() for x in state.get("weekly_new_tickers", []))
        for ticker in new_names:
            if ticker not in seen:
                seen.add(ticker)
                state["weekly_new_count"] = int(state.get("weekly_new_count", 0) or 0) + 1
        state["weekly_new_tickers"] = sorted(seen)


def add_market_score(state: Dict, today_str: str, score: int) -> int:
    history = list(state.get("market_score_history", []))
    history = [x for x in history if isinstance(x, dict) and "date" in x and "score" in x]
    history = [x for x in history if x.get("date") != today_str]
    history.append({"date": today_str, "score": int(score)})
    history = sorted(history, key=lambda x: str(x.get("date")))[-10:]
    state["market_score_history"] = history
    if len(history) >= 4:
        try:
            return int(history[-1]["score"]) - int(history[-4]["score"])
        except Exception:
            return 0
    return 0


def in_cooldown(state: Dict, ticker: str, today: date | None = None) -> bool:
    d = today or jst_today_date()
    cool = state.get("cooldown", {})
    raw = str(cool.get(ticker, ""))
    if not raw:
        return False
    try:
        until = date.fromisoformat(raw)
    except Exception:
        return False
    return d <= until


def set_cooldown_days(state: Dict, ticker: str, days: int, today: date | None = None) -> None:
    d = today or jst_today_date()
    until = d + timedelta(days=max(0, int(days)))
    cool = dict(state.get("cooldown", {}))
    cool[str(ticker)] = until.isoformat()
    state["cooldown"] = cool


# Compatibility helpers kept intentionally lightweight.

def record_paper_trade(state: Dict, trade: Dict) -> None:
    trades = list(state.get("paper_trades", []))
    trades.append(dict(trade))
    state["paper_trades"] = trades[-200:]


def update_paper_trades_with_ohlc(state: Dict, _ohlc_map: Dict) -> None:
    # Reserved for future KPI tracking. Kept as a no-op for compatibility.
    return None


def kpi_distortion(_state: Dict) -> float:
    return 0.0
