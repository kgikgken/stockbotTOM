# ============================================
# utils/report.py
# 日報レポート生成（Swing専用）
# ============================================

from typing import List, Dict
from utils.util import jst_today_str
from utils.position import check_risk_warning


# --------------------------------------------
# ヘッダ生成
# --------------------------------------------
def build_header(report: Dict) -> str:
    lines = []

    lines.append(f"📅 {jst_today_str()} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if report["no_trade"]:
        lines.append(f"🚫 新規見送り（{report['no_trade_reason']}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {report['market_score']}点（{report['market_label']}）")
    lines.append(f"- 地合い変化: Δ{report['delta_market']:+d}")
    lines.append(f"- 相場判断: {report['market_trend']}")
    lines.append(f"- 週次新規回数: {report['weekly_new']} / {report['weekly_limit']}")

    if report["macro_risk"]:
        lines.append("- マクロ警戒: ON")
    else:
        lines.append("- マクロ警戒: OFF")

    lines.append(f"- 推奨レバレッジ: {report['leverage']}倍")
    lines.append(f"- 最大建玉目安: 約{int(report['max_position']):,}円")

    return "\n".join(lines)


# --------------------------------------------
# セクター表示
# --------------------------------------------
def build_sector_section(sectors: List[Dict]) -> str:
    lines = []
    lines.append("")
    lines.append("📈 セクター動向（直近5日）")

    for i, s in enumerate(sectors, start=1):
        lines.append(f"{i}. {s['name']} ({s['return']:+.2f}%)")

    return "\n".join(lines)


# --------------------------------------------
# イベント表示
# --------------------------------------------
def build_event_section(events: List[str]) -> str:
    lines = []
    lines.append("")
    lines.append("⚠ 重要イベント")

    if not events:
        lines.append("- 特になし")
    else:
        for e in events:
            lines.append(f"⚠ {e}")

    return "\n".join(lines)


# --------------------------------------------
# Swing候補表示
# --------------------------------------------
def build_candidates_section(summary: Dict, candidates: List[Dict]) -> str:
    lines = []
    lines.append("")
    lines.append("🏆 Swing候補（順張りのみ / 追いかけ禁止 / 速度重視）")
    lines.append(
        f"  候補数:{summary['count']}銘柄  "
        f"平均RR:{summary['avg_rr']:.2f} / "
        f"平均EV:{summary['avg_ev']:.2f} / "
        f"平均補正EV:{summary['avg_adj_ev']:.2f} / "
        f"平均R/日:{summary['avg_r_day']:.2f}"
    )

    if summary.get("limit_reason"):
        lines.append(f"※ {summary['limit_reason']}")

    for c in candidates:
        lines.append("")
        lines.append(f"- {c['ticker']} {c['name']}［{c['sector']}］")
        lines.append(
            f"  型:{c['setup']}  RR:{c['rr']:.2f}  "
            f"補正EV:{c['adj_ev']:.2f}  R/日:{c['r_day']:.2f}"
        )
        lines.append(
            f"  エントリー:{c['entry']:.1f}"
            f"（範囲:{c['entry_low']:.1f}〜{c['entry_high']:.1f}） "
            f"現在値:{c['price']:.1f}  ATR:{c['atr']:.1f}  GU:{c['gu']}"
        )
        lines.append(
            f"  損切:{c['stop']:.1f}  "
            f"利確1:{c['tp1']:.1f}  利確2:{c['tp2']:.1f}  "
            f"想定日数:{c['days']:.1f}"
        )
        lines.append(f"  行動:{c['action']}")

    return "\n".join(lines)


# --------------------------------------------
# 監視リスト
# --------------------------------------------
def build_watch_section(watchlist: List[Dict]) -> str:
    lines = []
    lines.append("")
    lines.append("🧠 監視リスト（今日は入らない）")

    for w in watchlist:
        lines.append(
            f"- {w['ticker']} {w['name']}［{w['sector']}］ "
            f"理由:{w['reason']}"
        )

    return "\n".join(lines)


# --------------------------------------------
# ポジション・リスク表示
# --------------------------------------------
def build_position_section(
    positions: List[Dict],
    account_size: float
) -> str:
    lines = []
    lines.append("")
    lines.append("📊 保有ポジション")

    if not positions:
        lines.append("- なし")
    else:
        for p in positions:
            lines.append(
                f"- {p['ticker']}: 損益 {p.get('pnl', 'n/a')}"
            )

    risk = check_risk_warning(positions, account_size)
    if risk["warning"]:
        lines.append("")
        lines.append(
            f"⚠ ロット事故警告："
            f"想定最大損失 ≈ {int(risk['total_risk']):,}円"
            f"（資産比 {risk['risk_ratio']}%）"
        )

    return "\n".join(lines)


# --------------------------------------------
# レポート統合
# --------------------------------------------
def build_report(
    header: Dict,
    sectors: List[Dict],
    events: List[str],
    summary: Dict,
    candidates: List[Dict],
    watchlist: List[Dict],
    positions: List[Dict],
    account_size: float
) -> str:
    sections = [
        build_header(header),
        build_sector_section(sectors),
        build_event_section(events),
        build_candidates_section(summary, candidates),
        build_watch_section(watchlist),
        build_position_section(positions, account_size),
    ]

    return "\n".join(sections)