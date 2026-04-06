from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from utils.table_image import save_table_image
from utils.util import ensure_dir, human_yen, safe_float


@dataclass
class ReportBundle:
    text: str
    assets: Dict[str, str]


def _cand_df(cands: List[Dict], lane_label: str) -> pd.DataFrame:
    rows = []
    for c in cands:
        rows.append(
            {
                "Ticker": c.get("ticker", ""),
                "Name": c.get("name", ""),
                "Sector": c.get("sector", ""),
                "Setup": c.get("setup", ""),
                "Entry": c.get("order_text", ""),
                "SL": safe_float(c.get("stop")),
                "TP1": safe_float(c.get("tp1")),
                "R": safe_float(c.get("rr")),
                "Score": safe_float(c.get("score")),
                "Why": c.get("why", lane_label),
            }
        )
    return pd.DataFrame(rows)


def _summary_df(
    market: Dict,
    delta3: int,
    futures_chg: float,
    risk_on: bool,
    macro_on: bool,
    events_lines: Iterable[str],
    no_trade: bool,
    weekly_used: int,
    weekly_max: int,
    leverage: float,
    policy_lines: Iterable[str],
    pos_text: str,
    positions_df: pd.DataFrame,
    trend_count: int,
    leader_count: int,
) -> pd.DataFrame:
    rows = [
        {"Section": "Market", "Detail": f"score {market.get('score', '-')}, regime {market.get('label', '-')}, Δ3 {delta3:+d}"},
        {"Section": "Futures", "Detail": f"{futures_chg:+.2f}% / {'risk-on' if risk_on else 'risk-off'}"},
        {"Section": "Macro", "Detail": "warning on" if macro_on else "normal"},
        {"Section": "Weekly", "Detail": f"new {weekly_used}/{weekly_max} | leverage {leverage:.2f}x | {'watch-only' if no_trade else 'active'}"},
        {"Section": "Lanes", "Detail": f"trend {trend_count} | leaders {leader_count}"},
        {"Section": "Positions", "Detail": pos_text},
    ]
    for line in list(market.get("lines", []))[:3]:
        rows.append({"Section": "Market detail", "Detail": str(line)})
    for line in list(events_lines)[:4]:
        rows.append({"Section": "Event", "Detail": str(line)})
    for line in list(policy_lines)[:5]:
        rows.append({"Section": "Policy", "Detail": str(line)})
    if positions_df is not None and not positions_df.empty:
        for _, row in positions_df.head(6).iterrows():
            detail = (
                f"avg {safe_float(row.get('avg')):,.0f} / last {safe_float(row.get('last')):,.0f} "
                f"/ PnL {safe_float(row.get('pnl_pct')):+.1f}% / SL {safe_float(row.get('sl')):,.0f}"
            )
            rows.append({"Section": f"Pos {row.get('ticker')}", "Detail": detail})
    return pd.DataFrame(rows)


def build_report(
    *,
    today_str: str,
    market: Dict,
    delta3: int,
    futures_chg: float,
    risk_on: bool,
    macro_on: bool,
    events_lines: Iterable[str],
    no_trade: bool,
    weekly_used: int,
    weekly_max: int,
    leverage: float,
    policy_lines: Iterable[str],
    trend_candidates: List[Dict],
    leader_candidates: List[Dict],
    pos_text: str,
    positions_df: pd.DataFrame,
    out_dir: str | Path = "out",
) -> ReportBundle:
    out = ensure_dir(out_dir)

    trend_df = _cand_df(trend_candidates, "trend")
    leaders_df = _cand_df(leader_candidates, "leaders")
    summary_df = _summary_df(
        market,
        delta3,
        futures_chg,
        risk_on,
        macro_on,
        events_lines,
        no_trade,
        weekly_used,
        weekly_max,
        leverage,
        policy_lines,
        pos_text,
        positions_df,
        len(trend_candidates),
        len(leader_candidates),
    )

    summary_png = save_table_image(
        summary_df,
        out / f"report_table_{today_str}.png",
        title=f"stockbotTOM Summary | {today_str}",
        subtitle="Order summary / regime / policies / positions",
        notes=["saucer lane removed", "dual lanes = trend + leaders"],
        max_rows=24,
    )
    trend_png = save_table_image(
        trend_df,
        out / f"report_table_{today_str}_trend.png",
        title=f"順張りスクリーニング | {today_str}",
        subtitle="Existing momentum lane retained",
        notes=["A1 / A1-Strong / A2 / B", "execution plan reused from existing setup family"],
        empty_text="trend candidates not found",
    )
    leaders_png = save_table_image(
        leaders_df,
        out / f"report_table_{today_str}_leaders.png",
        title=f"Leaders スクリーニング | {today_str}",
        subtitle="Research-based lane: RS + 52W high + contraction + liquidity",
        notes=["primary triggers = leaders-only trend follow", "saucer moved to quality bonus, not primary lane"],
        empty_text="leader candidates not found",
    )

    trend_df.to_csv(out / f"trend_candidates_{today_str}.csv", index=False)
    leaders_df.to_csv(out / f"leaders_candidates_{today_str}.csv", index=False)
    summary_df.to_csv(out / f"summary_{today_str}.csv", index=False)

    lines = [
        f"stockbotTOM {today_str}",
        f"Market {market.get('score', '-')} ({market.get('label', '-')}) | Δ3 {delta3:+d} | Futures {futures_chg:+.2f}% {'risk-on' if risk_on else 'risk-off'}",
        f"Weekly {weekly_used}/{weekly_max} | {'watch-only' if no_trade else 'active'} | leverage {leverage:.2f}x",
        f"Trend lane {len(trend_candidates)} candidates | Leaders lane {len(leader_candidates)} candidates",
        f"Positions: {pos_text}",
    ]
    if macro_on:
        lines.append("Macro warning: on")
    for line in list(events_lines)[:3]:
        lines.append(f"Event: {line}")
    for lane_name, lane_cands in (("Trend", trend_candidates), ("Leaders", leader_candidates)):
        if lane_cands:
            top = lane_cands[0]
            lines.append(
                f"{lane_name} top: {top.get('ticker')} {top.get('setup')} {top.get('order_text')} / SL {safe_float(top.get('stop')):,.0f} / R {safe_float(top.get('rr')):.2f}"
            )
    text = "\n".join(lines)
    return ReportBundle(
        text=text,
        assets={
            "summary_png": summary_png,
            "trend_png": trend_png,
            "leaders_png": leaders_png,
        },
    )
