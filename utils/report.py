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
    lines.append(f"RR下限：{rr_min_by_market(mkt_score):.1f}  期待値（補正）下限：0.50  回転効率（R/日）下限：Setup別")
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
            action = "指値で待つ（現値では入らない）"
            if macro_on:
                action = "指値で待つ（ロット50%・TP2控えめ）"
            if c.get("gu"):
                action = "寄り後に再判定（GU）"

            entry_mid = float(c.get("entry_mid", (float(c["entry_low"]) + float(c["entry_high"])) / 2.0))

            # 1銘柄=4ブロック（縦配置）
            lines.append(f"■ {c['ticker']} {c['name']}（{c['sector']}）")
            lines.append("")

            lines.append("【形・行動】")
            # Setupは英字のまま残しつつ、日本語補足を添える
            setup = str(c.get("setup", "-")).strip()
            jp = "押し目" if setup.startswith("A1") else ("押し目（深め）" if setup == "A2" else ("ブレイク" if setup == "B" else ""))
            if jp:
                lines.append(f"・形：{setup}（{jp}）")
            else:
                lines.append(f"・形：{setup}")
            lines.append(f"・行動：{action}")
            lines.append("")

            lines.append("【エントリー】")
            lines.append(f"・指値目安（中央）：{_fmt_yen(entry_mid)} 円")
            lines.append(f"・損切り：{_fmt_yen(c['sl'])} 円")
            lines.append("")

            lines.append("【利確目標】")
            lines.append(f"・利確①：{_fmt_yen(c['tp1'])} 円")
            lines.append(f"・利確②：{_fmt_yen(c['tp2'])} 円")
            lines.append("")

            lines.append("【指標（参考）】")
            lines.append(f"・RR：{c['rr']:.2f}")
            lines.append(f"・期待値（補正）：{c['adj_ev']:.2f}")
            lines.append(f"・回転効率：{c['rday']:.2f}")
            lines.append(f"・想定日数：{c['expected_days']:.1f}日")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    lines.append("※ 用語：期待値（補正）=想定期待R（補正後）／回転効率=1日あたり想定R")
    lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)