from __future__ import annotations

from typing import Dict, List, Optional

from utils.screen_logic import rr_min_fixed, ev_min_fixed, strategy_caps

def _pct(x: float) -> str:
    return f"{x:+.2f}%"

def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"

def _yen(x: float) -> str:
    try:
        return f"{int(round(float(x))):,} 円"
    except Exception:
        return f"{x}"

def _header(market: Dict) -> List[str]:
    lines: List[str] = []
    date = market.get("date", "")
    ms = float(market.get("market_score", 50.0))
    ms_label = market.get("market_label", "")
    d3 = float(market.get("delta_3d", 0.0))
    fut_pct = float(market.get("futures_pct", 0.0))
    fut_ticker = market.get("futures_ticker", "NKD=F")
    macro_on = bool(market.get("macro_on", False))
    lev = float(market.get("leverage", 1.0))
    idx_vol = str(market.get("index_vol_regime", "MID"))

    lines.append(f"📅 {date} stockbotTOM 日報")
    lines.append("")
    lines.append("新規：✅ OK（指値のみ / 現値IN禁止）")
    lines.append("")
    lines.append(f"地合い：{int(round(ms))}（{ms_label}）  ΔMarketScore_3d:{_fmt(d3,1)}  先物:{_pct(fut_pct)}({fut_ticker})")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"推奨レバ：{_fmt(lev,1)}x")
    lines.append("▶ フィルター条件")
    lines.append(f"・RR 下限：{_fmt(rr_min_fixed(),1)}")
    lines.append(f"・期待値（補正）下限：{_fmt(ev_min_fixed(),2)}")
    lines.append("・回転効率 下限：Setup別")
    lines.append("")
    caps = strategy_caps(ms, idx_vol)
    lines.append("▶ 戦略別 最大枠（本日）")
    lines.append(f"・押し目：0〜{caps['PULLBACK']} / 初動：0〜{caps['BREAKOUT']} / 歪み：0〜{caps['DISTORT']}")
    lines.append("")
    lines.append("🛑 本日の方針")
    lines.append("・新規は指値のみ（現値IN禁止）")
    lines.append("・追いかけ禁止（GU銘柄は寄り後再判定のみ）")
    lines.append("")
    return lines

def _event_block(events: List[Dict]) -> List[str]:
    if not events:
        return []
    lines = ["⚠ 重要イベント（注意喚起）"]
    for e in events:
        name = str(e.get("name", "")).strip()
        ts = str(e.get("timestamp_jst", "")).strip()
        if name:
            lines.append(f"・{name}（{ts}）" if ts else f"・{name}")
    lines.append("")
    return lines

def _cand_block(c: Dict) -> List[str]:
    t = c.get("ticker", "")
    name = c.get("name", "")
    sec = c.get("sector", "")
    setup_jp = c.get("setup_jp", c.get("setup", ""))
    action = c.get("action", "指値で待つ（現値IN禁止）")
    entry = float(c.get("entry_mid", c.get("entry", 0.0)))
    sl = float(c.get("sl", 0.0))
    tp1 = float(c.get("tp1", 0.0))
    tp2 = float(c.get("tp2", 0.0))

    rr = float(c.get("rr", 0.0))
    ev = float(c.get("exp_value", 0.0))
    rot = float(c.get("rotation_eff", 0.0))
    days = float(c.get("expected_days", 0.0))
    cagr = float(c.get("cagr_pt", 0.0))

    lines: List[str] = []
    lines.append(f"■ {t} {name}（{sec}）")
    lines.append("")
    lines.append("【形・行動】")
    lines.append(f"・形：{setup_jp}")
    lines.append(f"・行動：{action}")
    lines.append("")
    lines.append("【エントリー】")
    lines.append(f"・指値目安（中央）：{_yen(entry)}")
    lines.append(f"・損切り：{_yen(sl)}")
    lines.append("")
    lines.append("【利確目標】")
    lines.append(f"・利確①：{_yen(tp1)}")
    lines.append(f"・利確②：{_yen(tp2)}")
    lines.append("")
    lines.append("【指標】")
    lines.append(f"・CAGR寄与度：{_fmt(cagr,1)} pt")
    lines.append(f"・RR（TP2基準）：{_fmt(rr,2)}")
    lines.append(f"・期待値（補正）：{_fmt(ev,2)}")
    lines.append(f"・回転効率（R/日）：{_fmt(rot,2)}")
    lines.append(f"・想定日数（中央値）：{_fmt(days,1)}日")
    lines.append("")
    return lines

def build_report(
    market: Dict,
    candidates: List[Dict],
    positions: List[Dict],
    events: Optional[List[Dict]] = None,
) -> str:
    lines: List[str] = []
    lines += _header(market)
    if events:
        lines += _event_block(events)

    lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
    if not candidates:
        lines.append("- 該当なし")
        lines.append("")
    else:
        for c in candidates:
            lines += _cand_block(c)

    lines.append("📊 ポジション")
    if not positions:
        lines.append("- なし")
    else:
        for p in positions:
            t = p.get("ticker", "")
            lines.append(f"■ {t}")
            if "rr" in p:
                lines.append(f"・RR：{p.get('rr')}")
            if "exp_value" in p:
                lines.append(f"・期待値（補正）：{p.get('exp_value')}")
            elif "adj_ev" in p:
                lines.append(f"・期待値（補正）：{p.get('adj_ev')}")
    lines.append("")
    lines.append("※ 用語：期待値（補正）= TP1基準の期待R×到達確率 ／ 回転効率（R/日）= 期待値（補正）÷想定日数 ／ CAGR寄与度=100×回転効率−時間ペナルティ")
    return "\n".join(lines)
