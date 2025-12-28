from __future__ import annotations

from typing import Dict, List
import numpy as np

from utils.util import jst_today_str, jst_today_date
from utils.position import load_positions, analyze_positions
from utils.market import calc_max_position
from utils.sector import top_sectors_5d


POSITIONS_PATH = "positions.csv"


def _fmt_yen(x: float) -> str:
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return "0"


def build_report(market: Dict, result: Dict) -> str:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt_score = int(market.get("score", 50))
    delta3d = int(market.get("delta3d", 0))
    mkt_comment = str(market.get("comment", "中立"))

    lev = float(market.get("lev", 1.3))
    lev_comment = str(market.get("lev_comment", "中立（厳選・押し目中心）"))

    # positions
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    max_pos = calc_max_position(total_asset, lev)

    sector_top5 = result.get("sector_top5") or top_sectors_5d(5)
    events = result.get("events") or ["- 特になし"]

    no_trade = bool(result.get("no_trade", False))
    nt_reasons = result.get("no_trade_reasons", [])
    nt_reason_text = " & ".join(nt_reasons) if nt_reasons else "条件該当"

    final = result.get("final", []) or []
    watch = result.get("watch", []) or []

    avg_rr = float(result.get("avg_rr", 0.0))
    avg_ev = float(result.get("avg_ev", 0.0))
    avg_adj_ev = float(result.get("avg_adj_ev", 0.0))
    avg_r_day = float(result.get("avg_r_day", 0.0))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    if no_trade:
        lines.append(f"🚫 新規見送り（{nt_reason_text}）")
    else:
        lines.append("✅ 新規可（条件クリア）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3d:+d}")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{_fmt_yen(max_pos)}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sector_top5:
        for i, (s, pct) in enumerate(sector_top5, start=1):
            lines.append(f"{i}. {s} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    for ev in events:
        lines.append(ev)
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if final:
        lines.append(
            f"  候補数:{len(final)}銘柄 / 平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f} / 平均AdjEV:{avg_adj_ev:.2f} / 平均R/day:{avg_r_day:.2f}"
        )
        lines.append("")
        for c in final:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  形:{c['setup']}  RR:{c['rr']:.2f}  AdjEV:{c['adj_ev']:.2f}  R/day:{c['r_day']:.2f}")
            lines.append(
                f"  IN:{c['in_center']:.1f}（帯:{c['in_low']:.1f}〜{c['in_high']:.1f}） 現在:{(c['price_now'] if c['price_now'] is not None else float('nan')):.1f}  ATR:{c['atr']:.1f}  GU:{c['gu']}"
            )
            lines.append(
                f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}  ExpectedDays:{c['exp_days']:.1f}  行動:{_action_jp(c['action'])}"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    lines.append("🧠 監視リスト（今日は入らない）")
    if watch:
        for w in watch[:10]:
            setup = w.get("setup", "-")
            rr = float(w.get("rr", 0.0))
            rd = float(w.get("r_day", 0.0))
            reason = str(w.get("reason", ""))
            act = _action_jp(str(w.get("action", "WATCH_ONLY")))
            gu = str(w.get("gu", "N"))
            lines.append(f"- {w['ticker']} {w['name']} [{w['sector']}] 形:{setup} RR:{rr:.2f} R/day:{rd:.2f} 理由:{reason} 行動:{act} GU:{gu}")
    else:
        lines.append("- 特になし")
    lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)


def _action_jp(action: str) -> str:
    if action == "EXEC_NOW":
        return "即IN可"
    if action == "LIMIT_WAIT":
        return "指値待ち"
    return "監視のみ"