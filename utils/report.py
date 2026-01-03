from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from utils.events import build_event_warnings, event_risk_multiplier
from utils.market import recommend_leverage
from utils.sector import compute_sector_5d_rank
from utils.screener import run_swing_screening
from utils.util import apply_weekly_new_trade_limit


def _fmt_price(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.1f}"


def _fmt_float(x: float, nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


def build_report_text(
    today_str: str,
    today_date,
    market: Dict,
    positions_text: str,
    total_asset: float,
    universe_path: str,
    events_path: str,
) -> str:
    mkt_score = int(market.get("score", 50))
    delta3d = int(market.get("delta3d", 0))
    mkt_comment = str(market.get("comment", "中立"))

    sectors_top, sector_rank_map = compute_sector_5d_rank(universe_path, top_n=5)
    event_lines, event_near = build_event_warnings(today_date, events_path)
    regime_mult = event_risk_multiplier(event_near)

    # baseline NO-TRADE
    no_trade_reasons: List[str] = []
    if mkt_score < 45:
        no_trade_reasons.append("MarketScore<45")
    if delta3d <= -5 and mkt_score < 55:
        no_trade_reasons.append("Δ3d<=-5 & MarketScore<55")

    baseline_new_ok = (len(no_trade_reasons) == 0)

    # screening (even if no-trade, we still compute but label as watch)
    core, watch = run_swing_screening(
        today_date=today_date,
        universe_path=universe_path,
        market=market,
        sector_rank_map=sector_rank_map,
        regime_multiplier=regime_mult,
    )

    # avg AdjEV for decision
    avg_adj_ev = float(np.mean([c["adj_ev"] for c in core])) if core else 0.0

    if event_near and avg_adj_ev < 1.2:
        no_trade_reasons.append("イベント接近 & 平均AdjEV<1.2")

    baseline_new_ok = (len(no_trade_reasons) == 0)

    # weekly limit (spec⑥)
    weekly = apply_weekly_new_trade_limit(today_date=today_date, baseline_new_ok=baseline_new_ok, weekly_cap=3)
    if weekly.weekly_block:
        no_trade_reasons.append("週次新規制限")

    new_ok = (len(no_trade_reasons) == 0)

    # leverage
    lev, lev_comment = recommend_leverage(mkt_score)
    # If weekly blocked or no-trade, lev keeps but report indicates no new.
    max_pos = _calc_max_position(total_asset, lev)

    # header
    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if new_ok:
        lines.append("✅ 新規可（条件クリア）")
    else:
        if no_trade_reasons:
            lines.append(f"🚫 新規見送り（{' & '.join(no_trade_reasons)}）")
        else:
            lines.append("🚫 新規見送り")

    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3d:+d}")
    lines.append(f"- 週次新規カウント: {weekly.weekly_count} / {weekly.weekly_cap}")
    if event_near:
        lines.append("- マクロ警戒: ON（イベント接近）")
    else:
        lines.append("- マクロ警戒: OFF")

    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    # sectors
    lines.append("📈 セクター（5日）")
    if sectors_top:
        for i, (s_name, pct) in enumerate(sectors_top, start=1):
            lines.append(f"{i}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    # events
    lines.append("⚠ イベント")
    for ev in event_lines:
        lines.append(ev)
    lines.append("")

    # core
    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if core:
        avg_rr = float(np.mean([c["rr"] for c in core]))
        avg_ev = float(np.mean([c["ev"] for c in core]))
        avg_adj = float(np.mean([c["adj_ev"] for c in core]))
        avg_rday = float(np.mean([c["r_per_day"] for c in core]))
        lines.append(
            f"  候補数:{len(core)}銘柄 / 平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f} / 平均AdjEV:{avg_adj:.2f} / 平均R/day:{avg_rday:.2f}"
        )
        lines.append("")
        for c in core:
            action = c["action"]
            # if no-trade, force watch
            if not new_ok:
                action = "WATCH_ONLY"

            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(
                f"  形:{c['setup']}  RR:{_fmt_float(c['rr'],2)}  AdjEV:{_fmt_float(c['adj_ev'],2)}  R/day:{_fmt_float(c['r_per_day'],2)}  Macro:{c.get('macro_tag','other')}"
            )
            lines.append(
                f"  IN:{_fmt_price(c['in_center'])}（帯:{_fmt_price(c['in_low'])}〜{_fmt_price(c['in_high'])}） 現在:{_fmt_price(c['price'])}  ATR:{_fmt_price(c['atr'])}  GU:{c['gu']}"
            )
            lines.append(
                f"  STOP:{_fmt_price(c['stop'])}  TP1:{_fmt_price(c['tp1'])}  TP2:{_fmt_price(c['tp2'])}  ExpectedDays:{_fmt_float(c['expected_days'],1)}  行動:{('監視のみ' if action=='WATCH_ONLY' else ('即IN可' if action=='EXEC_NOW' else '指値待ち'))}"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # watch
    lines.append("🧠 監視リスト（今日は入らない）")
    if watch:
        for w in watch[:10]:
            t = w.get("ticker", "-")
            sec = w.get("sector", "")
            rsn = w.get("reason", "")
            setup = w.get("setup", "-")
            rr = w.get("rr", 0.0)
            rday = w.get("r_per_day", 0.0)
            gu = w.get("gu", "N")
            if setup != "-" and rr:
                lines.append(f"- {t} [{sec}] 形:{setup} RR:{_fmt_float(rr,2)} R/day:{_fmt_float(rday,2)} 理由:{rsn} GU:{gu}")
            else:
                lines.append(f"- {t} [{sec}] 理由:{rsn}")
    else:
        lines.append("- なし")
    lines.append("")

    # positions
    lines.append("📊 ポジション")
    lines.append(positions_text.strip() if positions_text else "ノーポジション")

    return "\n".join(lines)