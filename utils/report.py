from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from utils.util import fmt_yen


def _fmt_pct(p: float) -> str:
    return f"{p:+.1f}%"


def build_report(
    today_str: str,
    market_ctx: Dict,
    sectors: List[Tuple[str, float]],
    events: List[str],
    screening: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(market_ctx["score"])
    mkt_comment = str(market_ctx["comment"])
    delta3d = int(market_ctx.get("delta3d", 0))

    # apply leverage adjustment by lot warning
    lev_base = float(market_ctx["lev"])
    lev = float(screening["lot"].get("lev_adj", lev_base))
    lev_comment = str(market_ctx["lev_comment"])
    if screening["lot"].get("lev_reason"):
        lev_comment = lev_comment + f" / {screening['lot']['lev_reason']}"

    max_pos = int(round(total_asset * lev))

    no_trade = bool(screening["no_trade"])
    no_trade_reasons = screening.get("no_trade_reasons", [])

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if no_trade:
        rs = " & ".join(no_trade_reasons) if no_trade_reasons else "条件該当"
        lines.append(f"🚫 新規見送り（{rs}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3d:+d}")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s, pct) in enumerate(sectors, start=1):
            lines.append(f"{i}. {s} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    for ev in events:
        lines.append(ev)
    lines.append("")

    # Swing block
    stats = screening["stats"]
    final = screening["final"]

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")

    if final:
        lines.append(
            f"  候補数:{stats['count']}銘柄 / 平均RR:{stats['avg_rr']:.2f} / 平均EV:{stats['avg_ev']:.2f} / 平均AdjEV:{stats['avg_adj_ev']:.2f} / 平均R/day:{stats['avg_r_per_day']:.2f}"
        )
        lines.append("")
        if no_trade:
            lines.append("※ 本日は新規禁止。下記は監視・指値設計のみ。")
            lines.append("")

        for c in final:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  形:{c['setup']}  RR:{c['rr']:.2f}  AdjEV:{c['adj_ev']:.2f}  R/day:{c['r_per_day']:.2f}")
            lines.append(
                f"  IN:{c['entry']:.1f}（帯:{c['in_low']:.1f}〜{c['in_high']:.1f}） 現在:{c['price_now']:.1f}  ATR:{c['atr']:.1f}  GU:{c['gu']}"
            )
            lines.append(
                f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}  ExpectedDays:{c['expected_days']:.1f}  行動:{c['action']}"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # Watch
    watch = screening.get("watch", [])
    lines.append("🧠 監視リスト（今日は入らない）")
    if watch:
        for w in watch:
            t = w.get("ticker", "")
            n = w.get("name", "")
            s = w.get("sector", "")
            r = w.get("watch_reason", "")
            if s:
                lines.append(f"- {t} {n} [{s}] 理由:{r}")
            else:
                lines.append(f"- {t} {n} 理由:{r}")
    else:
        lines.append("- 特になし")
    lines.append("")

    # Lot accident warning
    lot = screening["lot"]
    loss_yen = lot["total_loss_yen"]
    loss_ratio = lot["loss_ratio"] * 100.0
    if loss_ratio >= 8.0:
        lines.append(f"⚠ ロット事故警告：想定最大損失 ≈ {int(round(loss_yen)):,}円（資産比 {loss_ratio:.2f}%）")
        lines.append("")

    # Positions
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)