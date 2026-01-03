from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np


def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def build_report(
    today_str: str,
    mkt: Dict[str, Any],
    lev: float,
    lev_comment: str,
    max_position: int,
    sectors: List[Tuple[str, float]],
    events: List[str],
    swing: Dict[str, Any],
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))
    delta3d = float(mkt.get("delta3d", 0.0))

    no_trade = bool(swing.get("no_trade", False))
    no_trade_reason = str(swing.get("no_trade_reason", ""))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    if no_trade:
        lines.append(f"🚫 新規見送り（{no_trade_reason}）")
    else:
        lines.append("✅ 新規可（条件クリア）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3d:+.0f}")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_position:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s_name, pct) in enumerate(sectors[:5]):
            lines.append(f"{i+1}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    for ev in events:
        lines.append(ev)
    lines.append("")

    # --- Swing ---
    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    cands = swing.get("candidates", []) or []
    watch = swing.get("watch", []) or []

    if cands:
        lines.append(
            f"  候補数:{len(cands)}銘柄 / 平均RR:{float(swing.get('avg_rr',0)):.2f} / "
            f"平均EV:{float(swing.get('avg_ev',0)):.2f} / 平均AdjEV:{float(swing.get('avg_adjev',0)):.2f} / "
            f"平均R/day:{float(swing.get('avg_rpd',0)):.2f}"
        )
        lines.append("")
        for c in cands:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(
                f"  形:{c.get('setup','-')}  RR:{c.get('rr',0):.2f}  AdjEV:{c.get('adjev',0):.2f}  R/day:{c.get('r_per_day',0):.2f}"
            )
            lines.append(
                f"  IN:{c.get('in_center',0):.1f}（帯:{c.get('in_low',0):.1f}〜{c.get('in_high',0):.1f}） "
                f"現在:{c.get('price_now',0):.1f}  ATR:{c.get('atr',0):.1f}  GU:{c.get('gu','N')}"
            )
            lines.append(
                f"  STOP:{c.get('stop',0):.1f}  TP1:{c.get('tp1',0):.1f}  TP2:{c.get('tp2',0):.1f}  "
                f"ExpectedDays:{c.get('exp_days',0):.1f}  行動:{'即IN可' if c.get('action_code')=='EXEC_NOW' else '指値待ち'}"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # --- Watch ---
    if watch:
        lines.append("🧠 監視リスト（今日は入らない）")
        for w in watch[:10]:
            t = w.get("ticker", "")
            sec = w.get("sector", "不明")
            reason = w.get("reject_reason", "")
            setup = w.get("setup", "-")
            rr = w.get("rr", 0.0)
            rpd = w.get("r_per_day", 0.0)
            gu = w.get("gu", "N")
            lines.append(f"- {t} [{sec}] 形:{setup} RR:{rr:.2f} R/day:{rpd:.2f} 理由:{reason} GU:{gu}")
        lines.append("")

    # --- Lot warning (placeholder: main側で拡張できる) ---
    if swing.get("lot_warning_text"):
        lines.append(str(swing["lot_warning_text"]))
        lines.append("")

    # --- Positions ---
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)