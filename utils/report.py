from __future__ import annotations

from typing import Dict, List

from utils.screen_logic import rr_min_by_market

def _fmt_yen(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"

def build_report(
    today_str: str,
    market: Dict,
    delta3: float,
    futures_chg: float,
    risk_on: bool,
    macro_on: bool,
    events_lines: List[str],
    no_trade: bool,
    weekly_used: int,
    weekly_max: int,
    leverage: float,
    policy_lines: List[str],
    cands: List[Dict],
    pos_text: str,
) -> str:
    mkt_score = int(market.get("score", 50))
    mkt_comment = str(market.get("comment", "中立"))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    if macro_on:
        lines.append("⚠ 本日は重要イベント警戒日")
        if risk_on:
            lines.append("※ 先物Risk-ONにつき、警戒しつつ最大5まで表示")
        lines.append("")
        lines.append("対象イベント：")
        for ev in events_lines:
            if ev.startswith("⚠ "):
                lines.append("・" + ev.replace("⚠ ", "").split("（")[0])
        lines.append("")
        lines.append("🛑 本日の方針（イベント警戒）")
        lines.append("・新規は指値のみ（現値IN禁止）")
        lines.append("・ロットは通常の50%以下を推奨")
        lines.append("・TP2は控えめ（伸ばし過ぎない）")
        lines.append("・GU銘柄は寄り後再判定のみ")
        lines.append("")

    if no_trade and not cands:
        lines.append("新規：🛑 NO（新規ゼロ）")
    else:
        lines.append("新規：✅ OK（指値のみ / 現値IN禁止）")
    lines.append("")

    fut_txt = f"  先物:{futures_chg:+.2f}%(NKD=F) {'Risk-ON' if risk_on else ''}".rstrip()
    lines.append(f"地合い：{mkt_score}（{mkt_comment}）  ΔMarketScore_3d:{delta3:.1f}{fut_txt}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"週次新規：{weekly_used} / {weekly_max}")
    lines.append(f"推奨レバ：{leverage:.1f}x")
    lines.append("")

    if policy_lines:
        lines.append("🛑 本日の方針")
        for p in policy_lines:
            lines.append("・" + p)
        if no_trade and not cands:
            lines.append("・NO-TRADE理由：地合い条件 or 例外停止")
        lines.append("")

    lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
    if cands:
        for c in cands:
            setup = str(c.get("setup", ""))
            setup_label = setup
            if setup == "A1":
                setup_label = "A1（標準押し目）"
            elif setup == "A1-Strong":
                setup_label = "A1-Strong（強押し目）"

            sector = str(c.get("sector", ""))
            lines.append(f"■ {c['ticker']} {c['name']}（{sector}）")
            lines.append("")

            # 行動（裁量排除）
            action = "指値で待つ（現値IN禁止）"
            if c.get("gu"):
                action = "寄り後に再判定（GU）"

            entry_price = float(c.get("entry_price", (float(c["entry_low"]) + float(c["entry_high"])) / 2.0))

            # 4ブロック

            lines.append("【エントリー】")
            lines.append(f"・指値目安（中央）：{_fmt_yen(entry_price)} 円")
            lines.append(f"・損切り：{_fmt_yen(c['sl'])} 円")
            lines.append("")

            lines.append("【利確目標】")
            lines.append(f"・利確①：{_fmt_yen(c['tp1'])} 円、②：{_fmt_yen(c['tp2'])} 円")
            lines.append("")

            lines.append("【指標（参考）】")
            lines.append(f"・CAGR寄与度（/日）：{c.get('cagr', 0.0):.2f}")
            lines.append(f"・到達確率（目安）：{c.get('p_hit', 0.0):.3f}")
            lines.append(f"・期待R×到達確率：{c.get('adj_ev', 0.0):.2f}")
            lines.append(f"・RR（TP1基準）：{c.get('rr', 0.0):.2f}")
            lines.append(f"・想定日数（中央値）：{c.get('expected_days', 0.0):.1f}日")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    # Summary: execution list (central limit prices) for ALL displayed candidates.
    # Keep the same order as the main list (top -> bottom) for operational consistency.
    if cands:
        lines.append("")
        lines.append("まとめ")
        for c in cands:
            ticker = str(c.get("ticker", "")).strip()
            name = str(c.get("name", "")).strip()
            sector = str(c.get("sector", "")).strip()
            entry = c.get("entry_price", c.get("entry", None))
            try:
                entry = float(entry) if entry is not None else None
            except Exception:
                entry = None
            if entry is None:
                lo = c.get("entry_low", None)
                hi = c.get("entry_high", None)
                if lo is not None and hi is not None:
                    entry = (float(lo) + float(hi)) / 2.0
            title = f"■ {ticker} {name}（{sector}）" if (name and sector) else f"■ {ticker}"
            lines.append(title)
            if entry is not None:
                lines.append(f"・指値目安：{_fmt_yen(entry)} 円")

    return "\n".join(lines)