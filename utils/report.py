from __future__ import annotations

from typing import List

from utils.market import MarketSnapshot
from utils.screener import ScreeningResult
from utils.util import fmt_yen, fmt_pct


def _market_text(score: float) -> str:
    if score >= 70:
        return "強い"
    if score >= 55:
        return "中立"
    return "弱い"


def build_line_report(today_str: str, market: MarketSnapshot, screening: ScreeningResult, positions_text: str) -> str:
    lines: List[str] = []

    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    # Event warning block
    if market.macro_on and market.events:
        lines.append("⚠ 本日は重要イベント警戒日")
        if market.risk_text == "Risk-ON":
            lines.append("※ 先物Risk-ONにつき、警戒しつつ最大5まで表示")
        lines.append("")
        lines.append("対象イベント：")
        for ev in market.events:
            if "\n" in ev.name:
                name = ev.name.split("\n")[0]
            else:
                name = ev.name
            # 仕様：イベント名は正確に。時刻がある場合だけ付与
            if ev.dt_jst.endswith("00:00"):
                lines.append(f"・{name}")
            else:
                lines.append(f"・{name}（{ev.dt_jst}）")
        lines.append("")

        lines.append("🛑 本日の方針（イベント警戒）")
        lines.append("・新規は指値のみ（現値IN禁止）")
        lines.append("・ロットは通常の50%以下を推奨")
        lines.append("・TP2は控えめ（伸ばし過ぎない）")
        lines.append("・GU銘柄は寄り後再判定のみ")
        lines.append("")

    # Allow new
    new_txt = "✅ OK（指値のみ / 現値IN禁止）" if screening.allow_new else "🛑 NO（新規ゼロ）"
    lines.append(f"新規：{new_txt}")
    lines.append("")

    # Market
    lines.append(
        f"地合い：{int(round(market.market_score))}（{_market_text(market.market_score)}）  "
        f"ΔMarketScore_3d:{market.delta_score_3d:+.1f}  "
        f"先物:{fmt_pct(market.futures_pct)}({market.futures_symbol})"
    )
    lines.append(f"Macro警戒：{'ON' if market.macro_on else 'OFF'}")
    lines.append("週次新規：0 / 3")
    lines.append("推奨レバ：1.1x")
    lines.append(f"RR下限：{screening.rr_min:.1f}  期待効率下限：{screening.adj_ev_min:.2f}  速度下限：Setup別")
    lines.append("")

    # Rules (always)
    lines.append("🛑 本日の方針")
    lines.append("・新規は指値のみ（現値IN禁止）")
    if not screening.allow_new and screening.no_trade_reason:
        lines.append(f"・見送り理由：{screening.no_trade_reason}")
    lines.append("")

    # Candidates
    lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
    if not screening.candidates:
        lines.append("- 該当なし")
    else:
        for c in screening.candidates:
            # Entryは中央のみ表示（仕様）
            entry_txt = f"{fmt_yen(c.entry_center)} 円"
            action = "指値（Entry帯で待つ）"
            if market.macro_on:
                action = "指値（ロット50%・TP2控えめ）"
            lines.append(f"- {c.ticker} {c.name} [{c.sector}]")
            lines.append(f"  Setup:{c.setup}  行動:{action}")
            lines.append(f"  Entry価格:{entry_txt}")
            lines.append(
                f"  RR:{c.rr:.2f}  期待効率:{c.adj_ev:.2f}  速度:{c.speed:.2f}  想定日数:{c.exp_days:.1f}日"
            )
            lines.append(
                f"  損切り:{fmt_yen(c.sl)} 円  利確①:{fmt_yen(c.tp1)} 円  利確②:{fmt_yen(c.tp2)} 円"
            )
            lines.append("")

    # Positions
    lines.append("📊 ポジション")
    lines.append(positions_text)

    return "\n".join(lines).rstrip() + "\n"
