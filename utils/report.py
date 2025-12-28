from __future__ import annotations


def build_report(
    date_str,
    market,
    swing,
    watch,
    positions
) -> str:
    lines = []

    lines.append(f"📅 {date_str} stockbotTOM 日報\n")

    if market["no_trade"]:
        lines.append(f"🚫 新規見送り（{market['no_trade_reason']}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {market['score']}点 ({market['comment']})")
    lines.append(f"- ΔMarketScore_3d: {market['delta_3d']}")
    lines.append(f"- レバ: {market['leverage']}\n")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")

    if swing:
        for c in swing:
            lines.append(
                f"- {c['ticker']} [{c['sector']}] "
                f"形:{c['setup']} RR:{c['rr']:.2f} "
                f"AdjEV:{c['adj_ev']:.2f} R/day:{c['r_per_day']:.2f}\n"
                f"  IN:{c['entry']:.1f} STOP:{c['stop']:.1f} "
                f"TP1:{c['tp1']:.1f} TP2:{c['tp2']:.1f} "
                f"行動:{c['action']}"
            )
    else:
        lines.append("- 該当なし")

    if watch:
        lines.append("\n🧠 監視リスト")
        for w in watch:
            lines.append(f"- {w['ticker']} 理由:{w['reason']}")

    lines.append("\n📊 ポジション")
    lines.append(positions)

    return "\n".join(lines)