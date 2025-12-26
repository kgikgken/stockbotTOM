from __future__ import annotations

from typing import List, Dict
import numpy as np

from utils.market import MarketContext
from utils.events import EventContext

def _fmt_yen(x: float) -> str:
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return "0"

def build_report(today_str: str,
                 mkt: MarketContext,
                 ev_ctx: EventContext,
                 sector_tops: List[tuple],
                 result: dict,
                 pos_text: str,
                 max_position_yen: float) -> str:
    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if result.get("trade_allowed", True):
        lines.append("✅ 新規可（条件クリア）")
    else:
        reasons = " / ".join(result.get("no_trade_reasons", []) or ["条件該当"])
        lines.append(f"🚫 本日は新規見送り（理由：{reasons}）")

    lines.append(f"- 地合い: {mkt.score}点 ({mkt.comment})")
    lines.append(f"- ΔMarketScore_3d: {mkt.delta_3d:+d}")
    lines.append(f"- レバ: {mkt.lev:.1f}倍（{mkt.lev_comment}）")
    lines.append(f"- MAX建玉: 約{_fmt_yen(max_position_yen)}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sector_tops:
        for i, (s_name, pct) in enumerate(sector_tops):
            lines.append(f"{i+1}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    for w in ev_ctx.warnings:
        lines.append(w)
    lines.append("")

    picks = result.get("picks", [])
    watch = result.get("watch", [])

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if picks:
        rr_vals = [p["rr"] for p in picks]
        ev_vals = [p["ev"] for p in picks]
        adj_vals = [p["adj_ev"] for p in picks]
        rday_vals = [p["r_per_day"] for p in picks]
        lines.append(f"  候補数:{len(picks)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f} / 平均EV:{float(np.mean(ev_vals)):.2f} / 平均AdjEV:{float(np.mean(adj_vals)):.2f} / 平均R/day:{float(np.mean(rday_vals)):.2f}")
        lines.append("")
        for p in picks:
            lines.append(f"- {p['ticker']} {p['name']} [{p['sector']}]")
            lines.append(f"  形:{p['setup']}  RR:{p['rr']:.2f}  AdjEV:{p['adj_ev']:.2f}  R/day:{p['r_per_day']:.2f}")
            lines.append(f"  IN:{p['in_center']:.1f}（帯:{p['in_low']:.1f}〜{p['in_high']:.1f}） 現在:{p['price']:.1f}  ATR:{p['atr']:.1f}  GU:{'Y' if p['gu'] else 'N'}")
            lines.append(f"  STOP:{p['stop']:.1f}  TP1:{p['tp1']:.1f}  TP2:{p['tp2']:.1f}  ExpectedDays:{p['expected_days']:.1f}  行動:{p['action_jp']}")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    if watch:
        lines.append("🧠 監視リスト（今日は入らない）")
        for w in watch[:10]:
            reason = w.get("watch_reason") or w.get("reject_reason") or "条件"
            lines.append(f"- {w['ticker']} {w['name']} [{w['sector']}] 形:{w.get('setup','-')} RR:{w.get('rr',0):.2f} R/day:{w.get('r_per_day',0):.2f} 理由:{reason} 行動:{w.get('action_jp','監視のみ')} GU:{'Y' if w.get('gu',False) else 'N'}")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)
