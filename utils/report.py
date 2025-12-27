from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.7, "やや強気（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（厳選・押し目中心）"
    if mkt_score >= 45:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規かなり絞る）"

def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))

def _jp_action(a: str) -> str:
    if a == "EXEC_NOW":
        return "即IN可"
    if a == "LIMIT_WAIT":
        return "指値待ち"
    return "監視のみ"

def build_report(today_str: str, mkt: Dict, swing_result: Dict, events: List[str], pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    delta3d = int(mkt.get("delta3d", 0))
    mkt_comment = str(mkt.get("comment", "中立"))

    lev, lev_comment = recommend_leverage(mkt_score)
    max_pos = calc_max_position(total_asset, lev)

    sectors_top = swing_result.get("sectors_top", [])
    cands = swing_result.get("candidates", [])
    watch = swing_result.get("watchlist", [])

    trade_ok = bool(swing_result.get("trade_ok", False))
    no_trade_reason = str(swing_result.get("no_trade_reason", ""))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    lines.append("✅ 新規可（条件クリア）" if trade_ok else f"🚫 新規見送り（{no_trade_reason}）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3d:+d}")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors_top:
        for i, (s_name, pct) in enumerate(sectors_top):
            lines.append(f"{i+1}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    for ev in events:
        lines.append(ev)
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if cands:
        avg_rr = float(np.mean([c["rr"] for c in cands]))
        avg_ev = float(np.mean([c["ev"] for c in cands]))
        avg_adjev = float(np.mean([c["adjev"] for c in cands]))
        avg_rpd = float(np.mean([c["r_per_day"] for c in cands]))
        lines.append(f"  候補数:{len(cands)}銘柄 / 平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f} / 平均AdjEV:{avg_adjev:.2f} / 平均R/day:{avg_rpd:.2f}")
        lines.append("")
        for c in cands:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  形:{c['setup']}  RR:{c['rr']:.2f}  AdjEV:{c['adjev']:.2f}  R/day:{c['r_per_day']:.2f}")
            lines.append(f"  IN:{c['in_center']:.1f}（帯:{c['in_low']:.1f}〜{c['in_high']:.1f}） 現在:{c['price_now']:.1f}  ATR:{c['atr']:.1f}  GU:{'Y' if c['gu_flag'] else 'N'}")
            lines.append(f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}  ExpectedDays:{c['expected_days']:.1f}  行動:{_jp_action(c['action'])}")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    if watch:
        lines.append("🧠 監視リスト（今日は入らない）")
        for w in watch:
            lines.append(f"- {w['ticker']} {w.get('name','')} [{w.get('sector','')}] 理由:{w.get('reason','')}")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)
