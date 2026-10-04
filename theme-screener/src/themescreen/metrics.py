"""Pure computations for the industry map. No network access.

Definitions (README.md "Definitions"):
- trading value = Close x Volume of a daily bar (yfinance, auto_adjust=False)
- 60-day median trading value = median over the last 60 sessions up to asof
- latest operating loss = annual "Operating Income" of the latest fiscal
  period end is < 0
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

WINDOW_SESSIONS: int = 60
MIN_MEDIAN_TRADING_VALUE_JPY: float = 1e8

PROCESSES: tuple[str, ...] = (
    "露光", "成膜", "エッチング", "洗浄", "検査",
    "搬送", "部材", "ガス", "薬液", "ウェハ",
)


@dataclass(frozen=True)
class Computed:
    """A computed value, or None with the reason it could not be computed."""

    value: float | None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def market_cap(info: dict | None) -> Computed:
    """Market cap in JPY from yfinance info (README.md "Definitions")."""
    if not info or info.get("quoteType") in (None, "NONE"):
        return Computed(None, "yfinance が info を返さない")
    if info.get("currency") != "JPY":
        return Computed(None, f"通貨が JPY でない ({info.get('currency')})")
    v = info.get("marketCap")
    if not _is_number(v) or v <= 0:
        return Computed(None, "marketCap なし")
    return Computed(float(v))


def median_trading_value(history: pd.DataFrame, asof: str,
                         window: int = WINDOW_SESSIONS) -> Computed:
    """Median of Close x Volume over the last `window` sessions on or before asof.

    history: columns date (YYYY-MM-DD str), close, volume. Rows after asof are
    ignored. Rows with NaN close/volume are dropped. Fewer than `window`
    remaining rows -> not computable (README.md "Definitions").
    """
    if history is None or history.empty:
        return Computed(None, "日足なし")
    h = history.loc[history["date"] <= asof, ["date", "close", "volume"]].dropna()
    h = h.sort_values("date")
    if len(h) < window:
        return Computed(None, f"日足が{window}本未満 ({len(h)}本)")
    tail = h.iloc[-window:]
    return Computed(float((tail["close"] * tail["volume"]).median()))


def latest_annual(series: dict[str, float | None] | None) -> tuple[str | None, Computed]:
    """Value at the latest fiscal period end of an annual line item.

    series: {period_end (YYYY-MM-DD): value}. The latest period end is used even
    if its value is missing; it does not fall back to an older period, because
    an older period is not the latest one (README.md "Definitions").
    """
    if not series:
        return None, Computed(None, "通期の損益計算書に項目なし")
    latest = max(series)
    v = series[latest]
    if not _is_number(v):
        return latest, Computed(None, f"直近期 {latest} の値が欠損")
    return latest, Computed(float(v))


def next_period_end_passed(period_end: str | None, asof: str) -> tuple[str, int] | None:
    """The fiscal period end one year after `period_end`, if it is on or before asof.

    Returns (next period end, days from it to asof), or None. This only states
    that a later fiscal year has ended; whether its results are published is
    not judged here (README.md "Definitions").
    """
    if not period_end:
        return None
    nxt = pd.Timestamp(period_end) + pd.DateOffset(years=1)
    if nxt.is_month_end is False and pd.Timestamp(period_end).is_month_end:
        nxt = nxt + pd.offsets.MonthEnd(0)
    a = pd.Timestamp(asof)
    if nxt > a:
        return None
    return nxt.strftime("%Y-%m-%d"), int((a - nxt).days)


def cutoff(tv: Computed, op: Computed) -> tuple[str, list[str]]:
    """Apply the two cutoffs. Returns (status, exclusion reasons).

    - 60-day median trading value < 100M JPY -> excluded
    - latest operating income < 0 -> excluded
    status: "掲載" (both computable, neither triggered), "除外" (any computable
    cutoff triggered), "判定不能" (nothing triggered but an input is missing).
    """
    reasons: list[str] = []
    if tv.ok and tv.value < MIN_MEDIAN_TRADING_VALUE_JPY:
        reasons.append("売買代金1億円未満")
    if op.ok and op.value < 0:
        reasons.append("直近期営業赤字")
    if reasons:
        return "除外", reasons
    if not tv.ok or not op.ok:
        return "判定不能", []
    return "掲載", []
