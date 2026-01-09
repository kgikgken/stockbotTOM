from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np

from utils.rr_ev import rr_min_by_market


def _fmt_price(x: float) -> str:
    try:
        if not np.isfinite(float(x)):
            return "-"
        return f"{float(x):,.1f}"
    except Exception:
        return "-"


def _fmt(x: float, nd: int = 2) -> str:
    try:
        if not np.isfinite(float(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def _fmt_pct(x: float, nd: int = 1) -> str:
    try:
        if not np.isfinite(float(x)):
            return "-"
        return f"{float(x):+.{nd}f}%"
    except Exception:
        return "-"


def recommend_leverage(mkt_score: int, macro_on: bool) -> float:
    # base
    if mkt_score >= 70:
        lev = 1.8
    elif mkt_score >= 60:
        lev = 1.5
    elif mkt_score >= 55:
        lev = 1.3
    elif mkt_score >= 45:
        lev = 1.1
    else:
        lev = 1.0
    # macro suppress
    if macro_on:
        lev = min(1.1, lev)
    return float(lev)


def market_judge_jp(mkt_score: int, delta3d: float) -> str:
    if mkt_score < 45:
        return "地合いNG（新規ゼロが基本）"
    if delta3d <= -5 and mkt_score < 55:
        return "地合い悪化中（新規ゼロ寄り）"
    if mkt_score >= 70:
        return "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return "やや強気（押し目のみ）"
    if mkt_score >= 55:
        return "中立（厳選）"
    return "弱め（新規かなり絞る）"


def build_report(
    today_str: str,
    today_date: date,
    mkt: Dict[str, object],
    macro_on: bool,
    event_warnings: List[str],
    weekly_new_count: Optional[int],
    total_asset: float,
    positions_text: str,
    screening: Dict[str, object],
) -> str:
    mkt_score = int(mkt.get("score", 50))
    delta3d = float(mkt.get("delta3d", 0.0))
    judge = market_judge_jp(mkt_score, delta3d)

    lev = recommend_leverage(mkt_score, macro_on)
    rr_min = rr_min_by_market(mkt_score)

    no_trade = bool(screening.get("no_trade", False))
    reasons = screening.get("no_trade_reasons", []) or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    candidates = screening.get("candidates", []) or []
    stats = screening.get("stats", {}) or {}

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    # Header
    if no_trade:
        lines.append("🚫 新規：NO-TRADE DAY")
        if reasons:
            lines.append("理由：" + " / ".join([str(x) for x in reasons]))
    else:
        lines.append("✅ 新規：OK（指値のみ / 追いかけ禁止 / 現値IN禁止）")
    lines.append("")
    lines.append(f"地合い：{mkt_score}（{mkt.get('comment','')}）  ΔMarketScore_3d:{_fmt(delta3d,1)}")
    lines.append(f"相場判断：{judge}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    if weekly_new_count is None:
        lines.append("週次新規：- / 3")
    else:
        lines.append(f"週次新規：{weekly_new_count} / 3")
    lines.append(f"推奨レバ：{lev:.1f}x")
    lines.append(f"RR下限：{rr_min:.1f}  AdjEV下限：0.50  R/day下限：0.50")
    lines.append("")

    # Events
    lines.append("⚠ イベント")
    for w in event_warnings:
        lines.append(w)
    lines.append("")

    # Candidates
    if candidates:
        title = "📌 候補（参考：見送り）" if no_trade else "🏆 候補（1〜7営業日）"
        lines.append(title)
        lines.append("")
        for c in candidates:
            t = c.get("ticker", "")
            name = c.get("name", "")
            setup = c.get("setup", "")
            sector = c.get("sector", "不明")

            entry_low = float(c.get("entry_low", 0.0))
            entry_high = float(c.get("entry_high", 0.0))
            sl = float(c.get("sl", 0.0))
            tp1 = float(c.get("tp1", 0.0))
            tp2 = float(c.get("tp2", 0.0))

            rr = float(c.get("rr", 0.0))
            adjev = float(c.get("adjev", 0.0))
            rday = float(c.get("r_per_day", 0.0))
            edays = float(c.get("expected_days", 0.0))
            gu = bool(c.get("gu", False))

            if no_trade:
                action = "見送り（NO-TRADE）"
            else:
                # 監視は最小限：GUのみ寄り後再判定
                action = "見送り（GU→寄り後再判定）" if gu else "指値（Entry帯で待つ）"

            lines.append(f"- {t} {name} [{sector}]")
            lines.append(f"  Setup:{setup}  行動:{action}")
            lines.append(f"  Entry帯:{_fmt_price(entry_low)}〜{_fmt_price(entry_high)}")
            lines.append(f"  RR:{_fmt(rr,2)}  AdjEV:{_fmt(adjev,2)}  R/day:{_fmt(rday,2)}  ExpectedDays:{_fmt(edays,1)}")
            lines.append(f"  SL:{_fmt_price(sl)}  TP1:{_fmt_price(tp1)}  TP2:{_fmt_price(tp2)}")
            lines.append("")
    else:
        lines.append("🏆 候補")
        lines.append("- 該当なし")
        lines.append("")

    # Positions
    lines.append("📊 ポジション")
    lines.append(positions_text.strip() if positions_text else "ノーポジション")

    # Debug (one-liner)
    try:
        lines.append("")
        lines.append(
            f"（debug）raw:{stats.get('raw_count','-')} final:{stats.get('final_count','-')} "
            f"avgAdjEV:{_fmt(stats.get('avg_adjev',0.0),2)} GU:{_fmt(stats.get('gu_ratio',0.0),2)} rrMin:{_fmt(stats.get('rr_min',0.0),2)}"
        )
    except Exception:
        pass

    return "\n".join(lines)
