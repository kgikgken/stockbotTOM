from __future__ import annotations

from typing import List, Dict

import numpy as np


def _yn(ok: bool) -> str:
    return "✅ OK" if ok else "🛑 NO"


def _fmt_yen(x: float) -> str:
    try:
        if x >= 10000:
            return f"{int(round(x)):,}"
        return f"{x:.0f}"
    except Exception:
        return str(x)


def _fmt_x(x: float) -> str:
    try:
        return f"{x:.2f}"
    except Exception:
        return str(x)


def build_report(
    today_str: str,
    market: Dict,
    delta3d: float,
    futures_pct: float | None,
    risk_on: bool,
    macro_caution: bool,
    weekly_new: int,
    weekly_new_max: int,
    leverage: float,
    rr_min: float,
    pos_text: str,
    events_lines: List[str],
    cands: List[Dict],
    no_trade_reason: str | None = None,
) -> str:
    mkt_score = int(market.get("score", 50))
    mkt_comment = str(market.get("comment", ""))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    # Event header
    if macro_caution and events_lines:
        lines.append("⚠ 本日は重要イベント警戒日")
        if risk_on:
            lines.append("※ 先物Risk-ONのため、警戒しつつ最大5まで表示")
        lines.append("")
        lines.append("対象イベント：")
        for ev in events_lines:
            lines.append(f"・{ev}")
        lines.append("")
        lines.append("🛑 本日の方針（イベント警戒）")
        lines.append("・新規は指値のみ（成行・追いかけ禁止）")
        lines.append("・ロットは通常の50%以下を推奨")
        lines.append("・GU銘柄は寄り後再判定のみ")
        lines.append("")

    allow_new = True
    if no_trade_reason:
        allow_new = False

    lines.append(f"新規：{_yn(allow_new)}（指値のみ / 現値IN禁止）")
    lines.append("")

    fut_s = ""
    if futures_pct is not None and np.isfinite(futures_pct):
        fut_s = f"  先物:{futures_pct:+.2f}%" + (" Risk-ON" if risk_on else "")

    lines.append(f"地合い：{mkt_score}（{mkt_comment}）  ΔMarketScore_3d:{delta3d:+.1f}{fut_s}")
    lines.append(f"Macro警戒：{'ON' if macro_caution else 'OFF'}")
    lines.append(f"週次新規：{weekly_new} / {weekly_new_max}")
    lines.append(f"推奨レバ：{leverage:.1f}x")
    lines.append(f"RR下限：{rr_min:.1f}  期待値下限：0.50  効率下限：Setup別")
    lines.append("")

    lines.append("🛑 本日の方針")
    lines.append("・新規は指値のみ（現値IN禁止）")
    if no_trade_reason:
        lines.append(f"・NO-TRADE理由：{no_trade_reason}")
    lines.append("")

    lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
    if not cands:
        lines.append("- 該当なし")
    else:
        for c in cands:
            setup = str(c.get("setup", ""))
            act = str(c.get("action", "指値"))
            lines.append(f"- {c['ticker']} {c.get('name','')} [{c.get('sector','')}]")
            lines.append(f"  形:{setup}  行動:{act}")
            # Display only entry center
            lines.append(f"  指値目安:{_fmt_yen(float(c['entry']))} 円")
            lines.append(
                f"  RR:{_fmt_x(float(c['rr']))}  期待値:{_fmt_x(float(c['adjev']))}  効率:{_fmt_x(float(c['rday']))}  想定日数:{_fmt_x(float(c['expected_days']))}日"
            )
            lines.append(
                f"  損切り:{_fmt_yen(float(c['sl']))} 円  利確①:{_fmt_yen(float(c['tp1']))} 円  利確②:{_fmt_yen(float(c['tp2']))} 円"
            )
            lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    # Glossary (short)
    lines.append("")
    lines.append("補足：")
    lines.append("・期待値：1回のトレードで増える/減る“R”の見込み（0.50以上のみ）")
    lines.append("・効率：期待値を“日数”で割った回転効率（高いほど資金が速く増えやすい）")

    return "\n".join(lines)
