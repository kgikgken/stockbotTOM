from __future__ import annotations

import numpy as np

from utils.position import lot_accident_warning


def _fmt_float(x, nd=2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def build_report(
    today_str: str,
    today_date,
    mkt: dict,
    delta3d: int,
    weekly_used: int,
    weekly_limit: int,
    macro_danger: bool,
    sectors,
    event_lines,
    swing: dict,
    pos_text: str,
    total_asset: float,
    regime_mul: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    comment = str(mkt.get("comment", "中立"))

    picked = swing.get("picked", [])
    stats = swing.get("stats", {})

    # NO-TRADE機械化（ここは “出力理由” のみ。判定は report側でも再確認する）
    no_trade = False
    reasons = []
    if mkt_score < 45:
        no_trade = True
        reasons.append("MarketScore<45")
    if delta3d <= -5 and mkt_score < 55:
        no_trade = True
        reasons.append("Δ3d<=-5 & MarketScore<55")
    if macro_danger:
        no_trade = True
        reasons.append("イベント接近")
    if weekly_used >= weekly_limit:
        no_trade = True
        reasons.append("週次制限")
    if float(stats.get("avg_adj_ev", 0.0)) < 0.3 and picked:
        no_trade = True
        reasons.append("平均AdjEV<0.3")

    # レバ（reportは表示だけ）
    if no_trade:
        lev = 1.0
        lev_comment = "守り（新規禁止）"
    else:
        if mkt_score >= 70:
            lev = 2.0
            lev_comment = "強気（押し目＋一部ブレイク）"
        elif mkt_score >= 60:
            lev = 1.7
            lev_comment = "やや強気（押し目メイン）"
        elif mkt_score >= 50:
            lev = 1.3
            lev_comment = "中立（厳選・押し目中心）"
        elif mkt_score >= 40:
            lev = 1.1
            lev_comment = "やや守り（新規ロット小さめ）"
        else:
            lev = 1.0
            lev_comment = "守り"

    max_pos = int(round(total_asset * lev))

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    if no_trade:
        lines.append(f"🚫 新規見送り（{' & '.join(reasons)}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {mkt_score}点 ({comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3d:+d}")
    lines.append(f"- 週次新規カウント: {weekly_used} / {weekly_limit}")
    if macro_danger:
        lines.append("- マクロ警戒: ON（イベント接近）")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s, p) in enumerate(sectors, 1):
            lines.append(f"{i}. {s} ({p:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(event_lines if event_lines else ["- 特になし"])
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if picked:
        lines.append(
            f"  候補数:{len(picked)}銘柄 / 平均RR:{_fmt_float(stats.get('avg_rr',0),2)} / 平均EV:{_fmt_float(stats.get('avg_ev',0),2)}"
            f" / 平均AdjEV:{_fmt_float(stats.get('avg_adj_ev',0),2)} / 平均R/day:{_fmt_float(stats.get('avg_rday',0),2)}"
        )
        lines.append("")
        for c in picked:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  形:{c['setup']}  RR:{c['rr']:.2f}  AdjEV:{c['adj_ev']:.2f}  R/day:{c['r_per_day']:.2f}")
            lines.append(
                f"  IN:{c['in_center']:.1f}（帯:{c['in_low']:.1f}〜{c['in_high']:.1f}） 現在:{c['price_now']:.1f}  ATR:{c['atr']:.1f}  GU:{'Y' if c['gu'] else 'N'}"
            )
            lines.append(
                f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}  ExpectedDays:{c['expected_days']:.1f}  行動:{c['action']}"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # 監視
    watch = swing.get("watch", [])
    if watch:
        lines.append("🧠 監視リスト（今日は入らない）")
        for w in watch:
            if "drop_reason" in w:
                lines.append(f"- {w.get('ticker','')} {w.get('name','')} [{w.get('sector','')}] 理由:{w['drop_reason']}")
            else:
                lines.append(f"- {w.get('ticker','')} {w.get('name','')} [{w.get('sector','')}] 理由:監視")
        lines.append("")

    # ロット事故警告
    warn = lot_accident_warning(picked, total_asset=total_asset, risk_per_trade=0.015)
    if warn:
        lines.append(warn)
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)