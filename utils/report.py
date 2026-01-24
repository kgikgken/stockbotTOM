from __future__ import annotations

from typing import Dict, List, Optional

from utils.util import fmt_yen, fmt_ratio


def build_header(meta: Dict) -> str:
    lines = []
    lines.append(f"📅 {meta['date']} stockbotTOM 日報")
    lines.append("")
    lines.append(f"新規：{meta['new_trade_flag']}")
    lines.append("")
    lines.append(f"地合い：{meta['market_score']}（{meta['market_label']}）  ΔMarketScore_3d:{meta['delta_3d']:.1f}  先物:{meta['futures']}")
    lines.append(f"Macro警戒：{meta['macro_caution']}")
    lines.append(f"週次新規：{meta['weekly_new']}")
    lines.append(f"推奨レバ：{meta['leverage']}")
    lines.append("▶ フィルター条件")
    lines.append(f"・RR 下限：{meta['rr_min']}")
    lines.append(f"・期待値（補正）下限：{meta['adjev_min']}")
    lines.append("・回転効率 下限：Setup別")
    return "\n".join(lines)


def build_event_block(events: List[Dict], risk_on_note: Optional[str] = None) -> str:
    if not events:
        return ""
    lines = []
    lines.append("")
    lines.append("⚠ 本日は重要イベント警戒日")
    if risk_on_note:
        lines.append(risk_on_note)
    lines.append("")
    lines.append("対象イベント：")
    for e in events:
        when = e["dt"].strftime("%Y-%m-%d %H:%M JST")
        lines.append(f"・{e['name']}（{when}）")
    lines.append("")
    lines.append("🛑 本日の方針（イベント警戒）")
    lines.append("・新規は指値のみ（現値IN禁止）")
    lines.append("・ロットは通常の50%以下を推奨")
    lines.append("・TP2は控えめ（伸ばし過ぎない）")
    lines.append("・GU銘柄は寄り後再判定のみ")
    return "\n".join(lines)


def build_policy_block(lines_in: List[str]) -> str:
    if not lines_in:
        return ""
    out = ["", "🛑 本日の方針"]
    out += [f"・{x}" for x in lines_in]
    return "\n".join(out)


def build_candidates_block(cands: List[Dict]) -> str:
    lines = []
    lines.append("")
    lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
    if not cands:
        lines.append("- 該当なし")
        return "\n".join(lines)

    for c in cands:
        lines.append(f"■ {c['ticker']} {c['name']}（{c.get('sector','不明')}）")
        lines.append("")
        lines.append("【形・行動】")
        lines.append(f"・形：{c['setup_label']}")
        lines.append("・行動：指値で待つ（現値IN禁止）")
        lines.append("")
        lines.append("【エントリー】")
        lines.append(f"・指値目安（中央）：{fmt_yen(c['entry'])}")
        lines.append(f"・損切り：{fmt_yen(c['sl'])}")
        lines.append("")
        lines.append("【利確目標】")
        lines.append(f"・利確①：{fmt_yen(c['tp1'])}")
        lines.append(f"・利確②：{fmt_yen(c['tp2'])}")
        lines.append("")
        lines.append("【指標（参考）】")
        lines.append(f"・RR：{fmt_ratio(c['rr'], 2)}")
        lines.append(f"・期待値（補正）：{fmt_ratio(c['adj_ev'], 2)}")
        lines.append(f"・回転効率（目安）：{fmt_ratio(c['rday'], 2)}")
        lines.append(f"・想定日数（中央値）：{fmt_ratio(c['expected_days'], 1)}日")
        lines.append("")

    lines.append("※ 用語：期待値（補正）=想定期待R（補正後）／回転効率=1日あたり想定R")
    return "\n".join(lines).rstrip()


def build_positions_block(pos_lines: List[str]) -> str:
    lines = []
    lines.append("")
    lines.append("📊 ポジション")
    if not pos_lines:
        lines.append("- なし")
        return "\n".join(lines)
    for p in pos_lines:
        lines.append(p)
    return "\n".join(lines)


def build_report(meta: Dict, events: List[Dict], policy_lines: List[str], cands: List[Dict], pos_lines: List[str]) -> str:
    parts = [build_header(meta)]
    ev = build_event_block(events, meta.get("risk_on_note"))
    if ev:
        parts.append(ev)
    pol = build_policy_block(policy_lines)
    if pol:
        parts.append(pol)
    parts.append(build_candidates_block(cands))
    parts.append(build_positions_block(pos_lines))
    return "\n".join(parts).strip()
