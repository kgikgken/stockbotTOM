# ============================================
# utils/report.py
# LINE 送信用レポート生成（最終完成版）
# ============================================

from typing import List, Dict
from utils.util import jst_today_str


# --------------------------------------------
# 共通フォーマット補助
# --------------------------------------------
def _fmt(v, nd=2):
    if v is None:
        return "-"
    try:
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)
    except Exception:
        return "-"


def _safe_str(s):
    if s is None:
        return "-"
    return str(s)


# --------------------------------------------
# ヘッダー
# --------------------------------------------
def build_header(info: Dict) -> str:
    lines = []
    lines.append(f"📅 {jst_today_str()} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if info.get("no_trade", False):
        lines.append(f"🚫 新規見送り（{_safe_str(info.get('no_trade_reason'))}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {_fmt(info.get('market_score'),0)}点")
    lines.append(f"- 地合い変化: Δ{_fmt(info.get('delta_market'),0)}")
    lines.append(f"- 相場判断: {_safe_str(info.get('market_view'))}")
    lines.append(f"- 週次新規回数: {_fmt(info.get('weekly_count'),0)} / {_fmt(info.get('weekly_limit'),0)}")

    if info.get("macro_risk"):
        lines.append("- マクロ警戒: ON")

    lines.append(f"- 推奨レバレッジ: {_fmt(info.get('leverage'),1)}倍")
    lines.append(f"- 最大建玉目安: 約{_fmt(info.get('max_position'),0)}円")

    return "\n".join(lines)


# --------------------------------------------
# セクター
# --------------------------------------------
def build_sector(sectors: List[Dict]) -> str:
    lines = []
    lines.append("")
    lines.append("📈 セクター動向（直近5日）")

    if not sectors:
        lines.append("- 該当なし")
        return "\n".join(lines)

    for i, s in enumerate(sectors, 1):
        lines.append(f"{i}. {_safe_str(s.get('name'))} (+{_fmt(s.get('ret'),2)}%)")

    return "\n".join(lines)


# --------------------------------------------
# イベント
# --------------------------------------------
def build_events(events: List[Dict]) -> str:
    lines = []
    lines.append("")
    lines.append("⚠ 重要イベント")

    if not events:
        lines.append("- 特になし")
        return "\n".join(lines)

    for e in events:
        lines.append(f"⚠ {_safe_str(e.get('name'))}（{_safe_str(e.get('date'))}）")

    return "\n".join(lines)


# --------------------------------------------
# Swing 候補
# --------------------------------------------
def build_candidates(cands: List[Dict], summary: Dict) -> str:
    lines = []
    lines.append("")
    lines.append("🏆 Swing候補（順張りのみ / 追いかけ禁止 / 速度重視）")

    lines.append(
        f"  候補数:{_fmt(summary.get('count'),0)}銘柄  "
        f"平均RR:{_fmt(summary.get('avg_rr'))} / "
        f"平均EV:{_fmt(summary.get('avg_ev'))} / "
        f"平均補正EV:{_fmt(summary.get('avg_adj_ev'))} / "
        f"平均R/日:{_fmt(summary.get('avg_r_day'))}"
    )

    if not cands:
        lines.append("")
        lines.append("- 該当なし")
        return "\n".join(lines)

    for c in cands:
        lines.append("")
        lines.append(
            f"- {_safe_str(c.get('code'))} {_safe_str(c.get('name'))}［{_safe_str(c.get('sector'))}］"
        )
        lines.append(
            f"  型:{_safe_str(c.get('setup'))}  "
            f"RR:{_fmt(c.get('rr'))}  "
            f"補正EV:{_fmt(c.get('adj_ev'))}  "
            f"R/日:{_fmt(c.get('r_day'))}"
        )
        lines.append(
            f"  エントリー:{_fmt(c.get('entry'))}"
            f"（範囲:{_fmt(c.get('in_low'))}〜{_fmt(c.get('in_high'))}） "
            f"現在値:{_fmt(c.get('price'))}  ATR:{_fmt(c.get('atr'))}  "
            f"GU:{'あり' if c.get('gu') else 'なし'}"
        )
        lines.append(
            f"  損切:{_fmt(c.get('stop'))}  "
            f"利確1:{_fmt(c.get('tp1'))}  "
            f"利確2:{_fmt(c.get('tp2'))}  "
            f"想定日数:{_fmt(c.get('days'))}"
        )
        lines.append(f"  行動:{_safe_str(c.get('action'))}")

    return "\n".join(lines)


# --------------------------------------------
# 監視リスト（任意）
# --------------------------------------------
def build_watchlist(watch: List[Dict]) -> str:
    if not watch:
        return ""

    lines = []
    lines.append("")
    lines.append("🧠 監視リスト（今日は入らない）")

    for w in watch:
        lines.append(
            f"- {_safe_str(w.get('code'))} {_safe_str(w.get('name'))}［{_safe_str(w.get('sector'))}］ "
            f"理由:{_safe_str(w.get('reason'))}"
        )

    return "\n".join(lines)


# --------------------------------------------
# メイン組み立て
# --------------------------------------------
def build_report(
    header_info: Dict,
    sectors: List[Dict],
    events: List[Dict],
    candidates: List[Dict],
    summary: Dict,
    watchlist: List[Dict],
) -> str:

    parts = [
        build_header(header_info),
        build_sector(sectors),
        build_events(events),
        build_candidates(candidates, summary),
        build_watchlist(watchlist),
    ]

    # None / 空 を完全排除
    parts = [p for p in parts if p and isinstance(p, str)]

    return "\n".join(parts)