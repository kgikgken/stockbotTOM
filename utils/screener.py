from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.util import jst_now, jst_today_str
from utils.market import compute_market_score
from utils.events import load_events, upcoming_important
from utils.position import load_positions
from utils.screen_logic import load_universe, build_raw_candidates
from utils.setup import build_setup_info, liquidity_filters, rday_min_by_setup
from utils.rr_ev import adj_ev
from utils.diversify import apply_basic_diversify
from utils.report import build_report


def _macro_caution(events: List[Dict]) -> bool:
    return len(events) > 0


def _rr_min(market_score: int) -> float:
    if market_score >= 75:
        return 1.8
    if market_score >= 55:
        return 1.9
    if market_score >= 45:
        return 2.1
    return 2.2


def _leverage(market_score: int, macro_caution: bool, risk_on: bool) -> str:
    lev = 1.1 if market_score >= 55 else 1.0
    if macro_caution and not risk_on:
        lev = min(lev, 1.0)
    return f"{lev:.1f}x"


def _new_trade_flag(_: bool) -> str:
    return "✅ OK（指値のみ / 現値IN禁止）"


def run_screen() -> str:
    now = jst_now()

    mk = compute_market_score()
    events_df = load_events("events.csv")
    upcoming = upcoming_important(events_df, now, horizon_days=2)
    macro_caution = _macro_caution(upcoming)

    rr_min = _rr_min(mk.score)
    adjev_min = 0.50

    uni = load_universe("universe_jpx.csv")
    raw, _debug = build_raw_candidates(uni)

    cands: List[Dict] = []
    for r in raw:
        setup = build_setup_info(r)
        if not setup:
            continue
        if not liquidity_filters(r):
            continue

        rday_min = rday_min_by_setup(setup)
        if not np.isfinite(r.get("rday", float("nan"))) or float(r["rday"]) < rday_min:
            continue

        if not np.isfinite(r.get("rr", float("nan"))) or float(r["rr"]) < rr_min:
            continue

        tp2 = float(r["tp2"])
        if macro_caution:
            # イベント警戒日はTP2を控えめに（伸ばしすぎない）
            tp2 = r["tp1"] + 0.40 * (r["tp1"] - r["entry"])

        # --- スコアの中核：CAGR寄与度 = (期待R × 到達確率) ÷ 想定日数
        # 期待RはTP1基準（固定）。分割利確・粘りはスコア外。
        expected_r = float(r.get("rr", float("nan")))  # TP1到達時のR
        expd = float(r.get("expected_days", float("nan")))
        if macro_caution and expd == expd:
            expd = expd * 1.10

        # 到達確率（簡易・再現性重視）。Setupの基礎確率 + 形の強さで微調整。
        base_p = {"A1-Strong": 0.55, "A1": 0.50, "B": 0.42, "D": 0.35}.get(setup, 0.45)
        pb = float(r.get("pullback_score", 0.0))
        if setup in ("A1", "A1-Strong"):
            base_p += (pb - 75.0) * 0.002
        p_reach = max(0.25, min(0.65, base_p))

        # 期待値（補正）: EV(R) = 期待R×p - 1×(1-p)
        aev = expected_r * p_reach - (1.0 - p_reach)
        # 最低限の環境補正（ゲートではなく、期待値の摩耗を見積もる）
        if macro_caution:
            aev *= 0.85
        if mk.risk_on:
            aev *= 1.05

        if not (aev == aev) or aev < adjev_min:
            continue

        # 回転効率（R/日）: 期待R×p ÷ 想定日数
        rday = (expected_r * p_reach) / expd if (expd == expd and expd > 0) else float("nan")

        # 想定日数ペナルティ（完全機械化）
        if expd == expd and expd >= 6.0:
            continue
        penalty = 0.0
        if expd == expd and expd >= 5.0:
            penalty = 10.0
        elif expd == expd and expd >= 4.0:
            penalty = 5.0

        label = "A1-Strong（強押し目）" if setup == "A1-Strong" else ("A1（標準押し目）" if setup == "A1" else ("B（初動ブレイク）" if setup == "B" else "D（需給歪み）"))

        cagr_score = (rotation_eff * 100.0) - penalty
        rank = cagr_score
        cands.append(
            {
                **r,
                "setup": setup,
                "setup_label": label,
                "tp2": float(tp2),
                "rr": float(rr_tp1),
                "expected_days": float(expd),
                "rday": float(rotation_eff),
                "adj_ev": float(adj_ev),
                "rank": float(rank),
            }
        )

    cands.sort(key=lambda x: x.get("rank", -1e9), reverse=True)
    cands = apply_basic_diversify(cands, max_per_sector=2)
    cands = cands[:5]

    posdf = load_positions("positions.csv")
    pos_lines: List[str] = []
    if not posdf.empty:
        for _, p in posdf.head(3).iterrows():
            t = p.get("ticker", "-")
            rr = p.get("rr", "")
            aev = p.get("adj_ev", "")
            pos_lines.append(f"■ {t}")
            if rr != "":
                pos_lines.append(f"・RR：{rr}")
            if aev != "":
                pos_lines.append(f"・期待値（補正）：{aev}")

    policy_lines = ["新規は指値のみ（現値IN禁止）"]
    if macro_caution:
        policy_lines += ["ロットは通常の50%以下", "TP2は控えめ", "GUは寄り後再判定"]

    futures = "-"
    if np.isfinite(mk.futures_pct):
        sign = "+" if mk.futures_pct >= 0 else ""
        futures = f"{sign}{mk.futures_pct:.2f}%({mk.futures_ticker})"
        if mk.risk_on:
            futures += " Risk-ON"

    risk_on_note = None
    if macro_caution and mk.risk_on:
        risk_on_note = "※ 先物Risk-ONにつき、警戒しつつ最大5まで表示"

    meta = {
        "date": jst_today_str(),
        "new_trade_flag": _new_trade_flag(macro_caution),
        "market_score": mk.score,
        "market_label": mk.score_label,
        "delta_3d": mk.delta_3d,
        "futures": futures,
        "macro_caution": "ON" if macro_caution else "OFF",
        "weekly_new": "0 / 3",
        "leverage": _leverage(mk.score, macro_caution, mk.risk_on),
        "rr_min": f"{rr_min:.1f}",
        "adjev_min": f"{adjev_min:.2f}",
        "risk_on_note": risk_on_note,
    }

    return build_report(meta, upcoming, policy_lines, cands, pos_lines)
