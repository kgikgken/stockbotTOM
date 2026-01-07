# ============================================
# utils/report.py
# LINE送信用レポート生成
# ============================================

from __future__ import annotations

from typing import List, Dict
from utils.util import jst_today_str


# --------------------------------------------
# ヘッダ生成
# --------------------------------------------
def build_header(context: Dict) -> str:
    """
    日報ヘッダ部分
    """
    lines = []
    lines.append(f"📅 {jst_today_str()} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if context.get("no_trade"):
        lines.append("🚫 新規見送り（条件未達）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {context.get('market_score')}点")
    lines.append(f"- 地合い変化: Δ{context.get('delta_3d')}")
    lines.append(f"- 相場判断: {context.get('market_view')}")
    lines.append(f"- 週次新規回数: {context.get('weekly_count')} / {context.get('weekly_limit')}")
    lines.append(f"- マクロ警戒: {'ON' if context.get('macro_risk') else 'OFF'}")
    lines.append(f"- 推奨レバレッジ: {context.get('leverage')}倍")
    lines.append(f"- 最大建玉目安: 約{context.get('max_position'):,}円")

    return "\n".join(lines)


# --------------------------------------------
# セクター表示
# --------------------------------------------
def build_sector_block(sectors: List[str]) -> str:
    lines = []
    lines.append("")
    lines.append("📈 セクター動向（直近5日）")
    for i, s in enumerate(sectors, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


# --------------------------------------------
# イベント表示
# --------------------------------------------
def build_event_block(events: List[str]) -> str:
    if not events:
        return "\n⚠ 重要イベント\n- 特になし"

    lines = []
    lines.append("")
    lines.append("⚠ 重要イベント")
    for e in events:
        lines.append(f"⚠ {e}")
    return "\n".join(lines)


# --------------------------------------------
# Swing候補
# --------------------------------------------
def build_candidates_block(
    candidates: List[Dict],
    summary: Dict,
) -> str:
    lines = []
    lines.append("")
    lines.append("🏆 Swing候補（順張りのみ / 追いかけ禁止 / 速度重視）")
    lines.append(
        f"  候補数:{summary['count']}銘柄  "
        f"平均RR:{summary['avg_rr']} / "
        f"平均EV:{summary['avg_ev']} / "
        f"平均補正EV:{summary['avg_adj_ev']} / "
        f"平均R/日:{summary['avg_r_day']}"
    )

    for c in candidates:
        lines.append("")
        lines.append(f"- {c['ticker']} {c['name']}［{c['sector']}］")
        lines.append(
            f"  型:{c['setup']}  RR:{c['rr']}  補正EV:{c['adj_ev']}  R/日:{c['r_day']}  特性:{c['macro']}"
        )
        lines.append(
            f"  エントリー:{c['entry']}（範囲:{c['entry_low']}〜{c['entry_high']}） "
            f"現在値:{c['price']}  ATR:{c['atr']}  GU:{'あり' if c['gu'] else 'なし'}"
        )
        lines.append(
            f"  損切:{c['stop']}  利確1:{c['tp1']}  利確2:{c['tp2']}  想定日数:{c['days']}"
        )
        lines.append(f"  行動:{c['action']}")

    return "\n".join(lines)


# --------------------------------------------
# 監視リスト
# --------------------------------------------
def build_watch_block(watches: List[Dict]) -> str:
    if not watches:
        return ""

    lines = []
    lines.append("")
    lines.append("🧠 監視リスト（今日は入らない）")
    for w in watches:
        lines.append(
            f"- {w['ticker']}［{w['sector']}］ 理由:{w['reason']}"
        )
    return "\n".join(lines)


# --------------------------------------------
# ポジション
# --------------------------------------------
def build_position_block(position_info: Dict) -> str:
    lines = []
    lines.append("")
    lines.append("📊 保有ポジション")
    if position_info.get("count", 0) == 0:
        lines.append("- なし")
    else:
        lines.append(
            f"- 保有数:{position_info['count']} / "
            f"想定最大損失:{position_info['max_loss']}円 "
            f"(資産比 {position_info['risk_ratio']*100:.1f}%)"
        )
        if position_info.get("warning"):
            lines.append("⚠ ロット事故警告")
    return "\n".join(lines)


# --------------------------------------------
# 全体まとめ
# --------------------------------------------
def build_report(
    context: Dict,
    sectors: List[str],
    events: List[str],
    candidates: List[Dict],
    summary: Dict,
    watches: List[Dict],
    position_info: Dict,
) -> str:
    blocks = [
        build_header(context),
        build_sector_block(sectors),
        build_event_block(events),
        build_candidates_block(candidates, summary),
        build_watch_block(watches),
        build_position_block(position_info),
    ]
    return "\n".join(blocks)