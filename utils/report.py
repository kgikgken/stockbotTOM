from __future__ import annotations

from typing import List, Dict
from datetime import date

def _fmt_range(a: float, b: float) -> str:
    return f"{a:,.1f}〜{b:,.1f}"

def _fmt_price(x: float) -> str:
    return f"{x:,.1f}"

def build_report(
    today_str: str,
    today_date: date,
    mkt: Dict[str, object],
    macro_on: bool,
    event_warnings: List[str],
    weekly_new_count: int,
    total_asset: float,
    positions_text: str,
    screening: Dict[str, object],
) -> str:
    mkt_score = int(mkt.get("score", 50) or 50)
    delta3 = float(mkt.get("delta3", 0.0) or 0.0)
    regime = str(mkt.get("regime", "")) or str(mkt.get("comment", ""))

    rr_min = float(mkt.get("rr_min", 2.2) or 2.2)
    ev_min = float(mkt.get("adjev_min", 0.5) or 0.5)
    rday_min = float(mkt.get("rday_min", 0.5) or 0.5)
    lev = float(mkt.get("lev", 1.0) or 1.0)

    no_trade = bool(screening.get("no_trade", False))
    reasons = screening.get("no_trade_reasons", []) or []

    if no_trade:
        new_label = "🛑 NO（新規ゼロ）"
    else:
        new_label = "⚠ 慎重（指値のみ / 追いかけ禁止 / 現値IN禁止）" if macro_on else "✅ OK（指値のみ / 追いかけ禁止 / 現値IN禁止）"

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append(f"新規：{new_label}")
    lines.append("")
    lines.append(f"地合い：{mkt_score}（{regime}）  ΔMarketScore_3d:{delta3:.1f}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"週次新規：{int(weekly_new_count)} / 3")
    lines.append(f"推奨レバ：{lev:.1f}x")
    lines.append(f"RR下限：{rr_min:.1f}  AdjEV下限：{ev_min:.2f}  R/day下限：{rday_min:.2f}")
    lines.append("")

    lines.append("🛑 本日の方針")
    lines.append("・現値IN禁止")
    lines.append("・Entry帯に来なければ新規なし")
    lines.append("・GU銘柄は寄り後再判定のみ")
    if no_trade and reasons:
        lines.append("・NO-TRADE理由：" + ", ".join([str(r) for r in reasons]))
    lines.append("")

    lines.append("⚠ イベント")
    if event_warnings:
        for ev in event_warnings:
            lines.append(ev)
    else:
        lines.append("- 特になし")
    lines.append("")

    cands = screening.get("candidates", []) or []
    lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
    if not cands:
        lines.append("- 該当なし")
        lines.append("")
    else:
        lines.append("")
        for i, c in enumerate(cands[:5], 1):
            ticker = str(c.get("ticker", ""))
            name = str(c.get("name", ""))
            sector = str(c.get("sector", ""))
            setup = str(c.get("setup", ""))
            action = str(c.get("action", ""))
            entry_lo = float(c.get("entry_lo", 0.0) or 0.0)
            entry_hi = float(c.get("entry_hi", 0.0) or 0.0)
            rr = float(c.get("rr", 0.0) or 0.0)
            adjev = float(c.get("adjev", 0.0) or 0.0)
            rday = float(c.get("rday", 0.0) or 0.0)
            edays = float(c.get("expected_days", 0.0) or 0.0)
            sl = float(c.get("sl", 0.0) or 0.0)
            tp1 = float(c.get("tp1", 0.0) or 0.0)
            tp2 = float(c.get("tp2", 0.0) or 0.0)
            gu = bool(c.get("gu", False))

            lines.append(f"{i}. {ticker} {name} [{sector}]")
            lines.append(f"  Setup:{setup}  行動:{action}" + ("  （GU）" if gu else ""))
            lines.append(f"  Entry帯:{_fmt_range(entry_lo, entry_hi)}")
            lines.append(f"  RR:{rr:.2f}  AdjEV:{adjev:.2f}  R/day:{rday:.2f}  ExpectedDays:{edays:.1f}")
            lines.append(f"  SL:{_fmt_price(sl)}  TP1:{_fmt_price(tp1)}  TP2:{_fmt_price(tp2)}")
            lines.append("")

    lines.append("📊 ポジション")
    lines.append(positions_text.strip() if positions_text else "ノーポジション")
    lines.append("")

    stats = screening.get("stats", {}) or {}
    raw_n = int(stats.get("raw_n", 0) or 0)
    final_n = int(stats.get("final_n", len(cands)) or len(cands))
    avg_adjev = float(stats.get("avg_adjev", 0.0) or 0.0)
    gu_ratio = float(stats.get("gu_ratio", 0.0) or 0.0)
    lines.append(f"(debug) raw:{raw_n} final:{final_n} avgAdjEV:{avg_adjev:.2f} GU:{gu_ratio:.2f} rrMin:{rr_min:.2f}")

    return "\n".join(lines)
