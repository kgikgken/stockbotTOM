def build_report(today_str, market, swing, sectors, events, pos_text, total_asset):
    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    lines.append(f"- 地合い: {market['score']}点 ({market['comment']})")
    lines.append(f"- MAX建玉: 約{int(total_asset):,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(sectors, 1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")

    lines.append("")
    lines.append("⚠ イベント")
    lines.extend(events)

    lines.append("")
    lines.append("🏆 Swing（順張りのみ）")
    for c in swing:
        lines.append(
            f"- {c['ticker']} [{c['sector']}] "
            f"RR:{c['rr']:.2f} EV:{c['ev']:.2f} R/day:{c['r_day']:.2f} 行動:{c['action']}"
        )

    lines.append("")
    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)