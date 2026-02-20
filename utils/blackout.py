from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class BlackoutEvent:
    """A manually-maintained blackout event (earnings / major event / etc.).

    We intentionally keep this simple and offline-friendly:
      - Users can maintain a small CSV exported from broker/IR calendar.
      - The screener can avoid trading around these events.

    CSV format (header is optional):

        date,ticker,reason
        2026-02-19,7599.T,earnings

    Notes:
      - `ticker` can be '*' or 'ALL' to apply to all symbols.
      - Empty / invalid lines are ignored.
    """

    d: date
    ticker: str
    reason: str


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def load_blackout_csv(path: str) -> List[BlackoutEvent]:
    """Load blackout events from a CSV.

    If the file does not exist, returns an empty list.
    """

    p = Path(path)
    if not p.exists():
        return []

    events: List[BlackoutEvent] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            # Try DictReader first (header present), fallback to plain CSV.
            sample = f.read(2048)
            f.seek(0)
            has_header = "date" in sample.lower() and "ticker" in sample.lower()

            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    d = _parse_date(str(row.get("date", "")))
                    t = str(row.get("ticker", "")).strip()
                    r = str(row.get("reason", "") or row.get("kind", "") or "event").strip()
                    if not d or not t:
                        continue
                    events.append(BlackoutEvent(d=d, ticker=t, reason=r or "event"))
            else:
                reader2 = csv.reader(f)
                for row in reader2:
                    if not row:
                        continue
                    d = _parse_date(row[0] if len(row) > 0 else "")
                    t = (row[1] if len(row) > 1 else "").strip()
                    r = (row[2] if len(row) > 2 else "event").strip() or "event"
                    if not d or not t:
                        continue
                    events.append(BlackoutEvent(d=d, ticker=t, reason=r))
    except Exception:
        # Be conservative: never crash the bot on a bad local file.
        return []

    return events


def blackout_reason(
    ticker: str,
    today: date,
    events: Iterable[BlackoutEvent],
    before_days: int = 1,
    after_days: int = 1,
) -> Optional[str]:
    """Return blackout reason if `ticker` is within blackout window."""

    t = (ticker or "").strip().upper()
    if not t:
        return None

    try:
        b = int(before_days)
    except Exception:
        b = 1
    try:
        a = int(after_days)
    except Exception:
        a = 1

    for ev in events:
        ev_t = (ev.ticker or "").strip().upper()
        if ev_t in ("*", "ALL") or ev_t == t:
            if ev.d is None:
                continue
            if (ev.d - timedelta(days=b)) <= today <= (ev.d + timedelta(days=a)):
                r = (ev.reason or "event").strip()
                return r
    return None


def load_blackouts_from_env(today_str: str) -> tuple[List[BlackoutEvent], int, int]:
    """Load blackouts based on environment variables.

    Env:
      - BLACKOUT_CSV: path to blackout csv (default: data/blackout.csv if exists)
      - BLACKOUT_BEFORE_DAYS: default 1
      - BLACKOUT_AFTER_DAYS: default 1

    Returns:
      (events, before_days, after_days)
    """

    _ = today_str  # reserved (future: per-day overrides)

    path = os.getenv("BLACKOUT_CSV", "").strip()
    if not path:
        # Default path if file exists.
        if Path("data/blackout.csv").exists():
            path = "data/blackout.csv"

    events: List[BlackoutEvent] = load_blackout_csv(path) if path else []

    try:
        before_days = int(os.getenv("BLACKOUT_BEFORE_DAYS", "1"))
    except Exception:
        before_days = 1
    try:
        after_days = int(os.getenv("BLACKOUT_AFTER_DAYS", "1"))
    except Exception:
        after_days = 1

    return events, before_days, after_days
