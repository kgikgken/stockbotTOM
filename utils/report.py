from __future__ import annotations

from typing import Dict, List

from utils.screen_logic import rr_min_by_market
from utils.util import safe_float

def _fmt_fmt_yen(x: float) -> str:
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
    saucers: List[Dict] | None = None,
) -> str:
    mkt_score = int(market.get("score", 50))
    mkt_comment = str(market.get("comment", "中立"))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    # Macro day preface (keep strict; do not promote market-in on event days)
    if macro_on:
        lines.append("⚠ 本日は重要イベント警戒日")
        if risk_on:
            lines.append("※ 先物Risk-ONにつき、警戒しつつ最大5まで表示")
        lines.append("")
        if events_lines:
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

    # Header
    if no_trade and not cands:
        lines.append("新規：🛑 NO（新規ゼロ）")
    else:
        lines.append("新規：✅ OK（指値 / 現値INは銘柄別）")
    lines.append("")

    fut_txt = f"  先物:{futures_chg:+.2f}%(NKD=F) {'Risk-ON' if risk_on else ''}".rstrip()
    lines.append(f"地合い：{mkt_score}（{mkt_comment}）  ΔMarketScore_3d:{delta3:.1f}{fut_txt}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"週次新規：{weekly_used} / {weekly_max}")
    lines.append(f"推奨レバ：{leverage:.1f}x")
    lines.append("")

    # Policy
    lines.append("🛑 本日の方針")
    if policy_lines:
        for p in policy_lines:
            if p.strip():
                lines.append("・" + p.strip().lstrip("・"))
    else:
        lines.append("・新規は指値のみ（現値IN禁止）")
    lines.append("")

    # Candidates
    if cands:
        lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
        for c in cands:
            ticker = str(c.get("ticker", ""))
            name = str(c.get("name", ticker))
            sector = str(c.get("sector", ""))
            entry_mode = str(c.get("entry_mode", "LIMIT"))
            suffix = "（現値IN可）" if (entry_mode == "MARKET_OK" and not macro_on) else ""
            lines.append(f"■ {ticker} {name}（{sector}）{suffix}")
            lines.append("")
            # Entry
            lines.append("【エントリー】")
            lines.append(f"・指値目安（中央）：{_fmt_yen(c.get('entry_price', (c.get('entry_low',0)+c.get('entry_high',0))/2.0))} 円")
            lines.append(f"・損切り：{_fmt_yen(c.get('sl', 0.0))} 円")
            lines.append("")
            # Targets (single line)
            lines.append("【利確目標】")
            lines.append(f"・利確①：{_fmt_yen(c.get('tp1', 0.0))} 円、②：{_fmt_yen(c.get('tp2', 0.0))} 円")
            lines.append("")
            # Indicators
            lines.append("【指標（参考）】")
            lines.append(f"・CAGR寄与度（/日）：{c.get('cagr', 0.0):.2f}")
            lines.append(f"・到達確率（目安）：{c.get('p_hit', 0.0):.3f}")
            lines.append(f"・期待R×到達確率：{c.get('exp_r', 0.0):.2f}")
            lines.append(f"・RR（TP1基準）：{c.get('rr', 0.0):.2f}")
            lines.append(f"・想定日数（中央値）：{c.get('expected_days', 0.0):.1f}日")
            lines.append("")
    else:
        lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
        lines.append("該当なし")
        lines.append("")

    # Positions (as-is; already unified in latest spec for audit, if enabled upstream)
    if pos_text.strip():
        lines.append("📊 ポジション")
        lines.append(pos_text.rstrip())
        lines.append("")

    # Summary (all displayed cands, in order)
    if cands:
        lines.append("まとめ")
        for c in cands:
            ticker = str(c.get("ticker", ""))
            name = str(c.get("name", ticker))
            sector = str(c.get("sector", ""))
            entry = _fmt_yen(c.get("entry_price", (c.get("entry_low",0)+c.get("entry_high",0))/2.0))
            lines.append(f"■ {ticker}.T {name}（{sector}）")
            lines.append(f"・指値目安：{entry} 円")
        lines.append("")

    # Saucer bucket (separate; requested to be at the very end)
    if saucers:
        lines.append("🥣 ソーサー枠（週足/月足）最大5")
        for s in saucers[:5]:
            ticker = str(s.get("ticker", ""))
            name = str(s.get("name", ticker))
            sector = str(s.get("sector", ""))
            tf = "週足" if str(s.get("timeframe", "W")) == "W" else "月足"
            rim = _fmt_yen(s.get("entry_price", s.get("rim", 0.0)))
            last = safe_float(s.get("last", 0.0), 0.0)
            rim_f = safe_float(s.get("rim", 0.0), 0.0)
            prog = (last / rim_f) if rim_f > 0 else 0.0
            lines.append(f"■ {ticker} {name}（{sector}）[{tf}]")
            lines.append(f"・指値目安（リム）：{rim} 円（進捗 {prog*100:.0f}%）")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"