from __future__ import annotations

from typing import List, Dict, Any
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
    event_items: List[Dict[str, Any]],
    weekly_new: int,
    positions_text: str,
    screening: Dict[str, object],
) -> str:
    mkt_score = int(mkt.get("score", 50) or 50)
    delta3 = float(mkt.get("delta3", 0.0) or 0.0)
    regime = str(mkt.get("regime", "")) or "中立"
    lev = float(mkt.get("lev", 1.0) or 1.0)

    rr_min = float(mkt.get("rr_min", 2.2) or 2.2)
    ev_min = 0.50
    rday_min = 0.50

    no_trade = bool(screening.get("no_trade", False))
    reasons = screening.get("no_trade_reasons", []) or []

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    if macro_on and event_items:
        lines.append("⚠ 本日は重要イベント警戒日")
        lines.append("")
        lines.append("対象イベント：")
        for it in event_items:
            lines.append(f"・{it.get('label','')}（{it.get('dt_str','')}）")
        lines.append("")
        lines.append("🛑 本日の方針（イベント警戒）")
        lines.append("・新規は指値のみ（現値IN禁止）")
        lines.append("・ロットは通常の50%以下を推奨")
        lines.append("・エントリーは寄り後の値動き確認が前提")
        lines.append("・条件未達なら見送りを優先")
        lines.append("")

    if no_trade:
        new_label = "🛑 NO（新規ゼロ）"
    else:
        new_label = "✅ OK（指値のみ / 追いかけ禁止 / 現値IN禁止）"
    lines.append(f"新規：{new_label}")
    lines.append("")

    lines.append(f"地合い：{mkt_score}（{regime}）  ΔMarketScore_3d:{delta3:.1f}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"週次新規：{int(weekly_new)} / 3")
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
    if event_items:
        for it in event_items:
            when = it.get("when", "")
            lines.append(f"⚠ {it.get('label','')}（{it.get('dt_str','')} / {when}）")
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
            lines.append(f"{i}. {c.get('ticker','')} {c.get('name','')} [{c.get('sector','')}]")
            lines.append(f"  Setup:{c.get('setup','')}  行動:{c.get('action','')}")
            lines.append(f"  Entry帯:{_fmt_range(float(c.get('entry_lo',0.0)), float(c.get('entry_hi',0.0)))}")
            lines.append(f"  RR:{float(c.get('rr',0.0)):.2f}  AdjEV:{float(c.get('adjev',0.0)):.2f}  R/day:{float(c.get('rday',0.0)):.2f}  ExpectedDays:{float(c.get('expected_days',0.0)):.1f}")
            lines.append(f"  SL:{_fmt_price(float(c.get('sl',0.0)))}  TP1:{_fmt_price(float(c.get('tp1',0.0)))}  TP2:{_fmt_price(float(c.get('tp2',0.0)))}")
            lines.append("")

    lines.append("📊 ポジション")
    lines.append(positions_text.strip() if positions_text else "ノーポジション")
    lines.append("")

    stats = screening.get("stats", {}) or {}
    lines.append(
        f"(debug) raw:{int(stats.get('raw_n',0))} final:{int(stats.get('final_n',0))} "
        f"avgAdjEV:{float(stats.get('avg_adjev',0.0)):.2f} GU:{float(stats.get('gu_ratio',0.0)):.2f} rrMin:{float(stats.get('rr_min',rr_min)):.2f}"
    )
    return "\n".join(lines)
