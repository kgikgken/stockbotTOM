from __future__ import annotations

from typing import Dict, List


def build_report(result: Dict) -> str:
    market = result["market"]
    events = result["events"]
    no_trade = result["no_trade"]
    cands: List[Dict] = result["candidates"]
    watch: List[Dict] = result["watchlist"]

    lines: List[str] = []

    # --- Header ---
    lines.append("📅 stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if no_trade:
        lines.append(f"🚫 新規見送り（{no_trade}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {market['score']}点 ({market['comment']})")
    lines.append(f"- ΔMarketScore_3d: {market['delta_3d']}")
    lines.append(f"- レバ: {market['leverage']}")
    lines.append("")

    # --- Events ---
    lines.append("⚠ イベント")
    if events:
        for e in events:
            lines.append(f"- {e}")
    else:
        lines.append("- 特になし")
    lines.append("")

    # --- Candidates ---
    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if cands:
        avg_rr = sum(c["rr"] for c in cands) / len(cands)
        avg_ev = sum(c["adj_ev"] for c in cands) / len(cands)
        avg_rpd = sum(c["r_per_day"] for c in cands) / len(cands)

        lines.append(
            f"  候補数:{len(cands)}銘柄 / 平均RR:{avg_rr:.2f} / 平均AdjEV:{avg_ev:.2f} / 平均R/day:{avg_rpd:.2f}"
        )
        lines.append("")

        for c in cands:
            lines.append(f"- {c['ticker']} [{c['sector']}]")
            lines.append(
                f"  形:{c['setup_type']} RR:{c['rr']:.2f} AdjEV:{c['adj_ev']:.2f} R/day:{c['r_per_day']:.2f}"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} 帯:{c['entry_low']:.1f}〜{c['entry_high']:.1f} GU:{'Y' if c['gu'] else 'N'}"
            )
            lines.append(
                f"  STOP:{c['stop']:.1f} TP1:{c['tp1']:.1f} TP2:{c['tp2']:.1f} 行動:{c['action']}"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # --- Watchlist ---
    lines.append("🧠 監視リスト")
    if watch:
        for w in watch[:10]:
            lines.append(f"- {w['ticker']} 理由:{w['reason']}")
    else:
        lines.append("- なし")

    return "\n".join(lines)