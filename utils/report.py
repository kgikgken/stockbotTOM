from __future__ import annotations

from typing import Dict, Any, List

def build_report(
    *,
    today_str: str,
    today_date,
    mkt: Dict[str, Any],
    delta3d: float,
    macro_on: bool,
    macro_lines: List[str],
    futures_risk_on: bool,
    state: Dict[str, Any],
    screen: Dict[str, Any],
    pos_text: str,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))
    rr_min = float(screen.get("meta", {}).get("rr_min", 2.2))

    weekly_new = int(state.get("weekly_new", 0) or 0)
    weekly_cap = 3

    lev = 1.0
    if mkt_score >= 70:
        lev = 1.7
    elif mkt_score >= 60:
        lev = 1.4
    elif mkt_score >= 50:
        lev = 1.1
    else:
        lev = 1.0
    if macro_on:
        lev = min(lev, 1.1)

    fut = float(mkt.get("futures_1d", 0.0) or 0.0)
    fut_sym = str(mkt.get("futures_symbol", "") or "")
    fut_txt = ""
    if abs(fut) > 0:
        fut_txt = f"  先物:{fut:+.2f}%"
        if fut_sym:
            fut_txt += f"({fut_sym})"
        if futures_risk_on:
            fut_txt += " Risk-ON"

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    if macro_on:
        lines.append("⚠ 本日は重要イベント警戒日")
        if futures_risk_on:
            lines.append("※ 先物Risk-ONにつき、警戒しつつ最大5まで表示")
        lines.append("")
        lines.append("対象イベント：")
        for x in (macro_lines or []):
            lines.append(x)
        lines.append("")
        lines.append("🛑 本日の方針（イベント警戒）")
        lines.append("・新規は指値のみ（現値IN禁止）")
        lines.append("・ロットは通常の50%以下を推奨")
        lines.append("・TP2は控えめ（伸ばし過ぎない）")
        lines.append("・GU銘柄は寄り後再判定のみ")
        lines.append("")

    no_trade = bool(screen.get("no_trade", False))
    reason = str(screen.get("reason", ""))

    lines.append(f"新規：{'🛑 NO（新規ゼロ）' if no_trade else '✅ OK（指値のみ / 現値IN禁止）'}")
    lines.append("")
    lines.append(f"地合い：{mkt_score}（{mkt_comment}）  ΔMarketScore_3d:{float(delta3d):.1f}{fut_txt}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"週次新規：{weekly_new} / {weekly_cap}")
    lines.append(f"推奨レバ：{lev:.1f}x")
    lines.append(f"RR下限：{rr_min:.1f}  AdjEV下限：0.50  R/day下限：0.50")
    lines.append("")

    if no_trade:
        lines.append("🛑 本日の方針")
        lines.append("・現値IN禁止")
        lines.append("・Entry帯に来なければ新規なし")
        lines.append("・GU銘柄は寄り後再判定のみ")
        if reason:
            lines.append(f"・NO-TRADE理由：{reason}")
        lines.append("")

    lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
    cands = screen.get("candidates", []) or []
    if not cands:
        lines.append("- 該当なし")
    else:
        for c in cands:
            lines.append(f"- {c['ticker']} {c['name']} [{c.get('sector','不明')}]")
            lines.append(f"  Setup:{c['setup']}  行動:{c['action']}")
            lines.append(f"  Entry帯:{c['entry_lo']:,.1f}〜{c['entry_hi']:,.1f}")
            lines.append(f"  RR:{c['rr']:.2f}  AdjEV:{c['adjev']:.2f}  R/day:{c['rday']:.2f}  ExpectedDays:{c['expected_days']:.1f}")
            lines.append(f"  SL:{c['sl']:,.1f}  TP1:{c['tp1']:,.1f}  TP2:{c['tp2']:,.1f}")
            lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines).rstrip() + "\n"
