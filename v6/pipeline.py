"""v6 パイプライン — Layer 0〜5を直列に通し、通知候補を組み立てる。

中核原則: 各レイヤーはAND条件。スコアは通過可否に一切関与せず、順序決定のみに使う。
通知が0件の日があるのは正常(最大5件は目標値ではなく上限値)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .filters import (weekly_stage, rs_vs_index, trend_template,
                      rs_performance, rs_ratings, daily_stage, atr_wilder)
from .setups import detect_setup, SETUP_VCP, SETUP_PULL, SETUP_PIVOT
from .risk import build_risk_plan, portfolio_heat, drawdown_state, gate_new_entries


@dataclass
class CandidateV6:
    ticker: str
    code: str
    name: str
    sector: str
    setup: str
    # 各レイヤーの値
    stage_w: int = 0
    slope_w: float = 0.0
    rs_rating: float = 0.0
    rs_excess: float = 0.0
    stage_d: int = 0
    stage1_days: int = 0
    stage_days: int = 0      # 現ステージの連続滞在日数(ステージ6でも記録)
    # セットアップ詳細
    detail: dict = field(default_factory=dict)
    # 執行
    order_type: str = ""
    entry: float = 0.0
    stop: float = 0.0
    stop_pct: float = 0.0
    stop_kind: str = ""
    shares: int = 0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    target_2r: float = 0.0
    expire_date: str = ""
    # スコア(順序決定のみ)
    score: int = 0
    score_parts: List[str] = field(default_factory=list)
    # 注意喚起
    flags: List[str] = field(default_factory=list)
    tradable: bool = True    # ステージ6は通知するが発注禁止


def _entry_price(setup: dict, cfg) -> tuple[str, float]:
    """セットアップ別の注文種別と価格(仕様書§7: 成行は一切使わない)。"""
    if setup["setup"] in (SETUP_VCP, SETUP_PIVOT):
        return "逆指値買い", setup["pivot"] * cfg.entry_stop_buy_mult
    # 押し目: 25日線と「高値-1ATR」の高い方(深く待って約定率を下げない)
    ma = setup.get("sma_pull")
    alt = setup["recent_high"] - 1.0 * setup["atr"]
    return "指値買い", max(ma, alt) if ma is not None else alt


def _score(c: CandidateV6, sector_rs_top: bool, rr_ratio: float | None, cfg) -> None:
    """通知順序の決定にのみ使う。通過可否には一切関与しない。"""
    pts, parts = 0, []
    base = {SETUP_VCP: 3, SETUP_PULL: 2, SETUP_PIVOT: 1}.get(c.setup, 0)
    pts += base; parts.append(f"{c.setup}+{base}")

    if c.rs_rating >= 90:
        pts += 2; parts.append(f"RS{c.rs_rating:.0f}+2")
    elif c.rs_rating >= 80:
        pts += 1; parts.append(f"RS{c.rs_rating:.0f}+1")

    if c.stage_d == 1 and 0 < c.stage1_days <= 3:
        pts += 1; parts.append(f"ステージ1入り{c.stage1_days}日+1")
    elif c.stage_d == 1 and c.stage1_days > 40:
        pts -= 1; parts.append(f"ステージ1が{c.stage1_days}日と長い-1")

    if c.setup == SETUP_VCP and c.detail.get("n_contractions", 0) >= 3:
        pts += 1; parts.append(f"収縮{c.detail['n_contractions']}回+1")

    if sector_rs_top:
        pts += 1; parts.append("業種RS上位+1")

    if rr_ratio is not None and rr_ratio >= 3.0:
        pts += 1; parts.append(f"RR{rr_ratio:.1f}+1")

    c.score, c.score_parts = pts, parts


def run_pipeline(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
                 index_daily: pd.DataFrame, cfg, today: str,
                 positions: list | None = None,
                 equity: float | None = None,
                 peak_equity: float | None = None) -> dict:
    """Layer 0〜5を直列に通す。各レイヤーの通過数も記録して返す。"""
    equity = equity if equity is not None else cfg.equity
    peak_equity = peak_equity if peak_equity is not None else equity
    positions = positions or []

    counts = {"universe": 0, "layer0": 0, "layer1": 0, "layer2": 0,
              "layer3": 0, "layer4": 0, "layer5": 0}
    drops = []          # 各レイヤーで落ちた理由の記録
    watchlist = []      # Layer2通過 = 監視ユニバース

    # ---- Layer 0: 流動性ユニバース ----
    l0 = []
    for _, row in universe.iterrows():
        counts["universe"] += 1
        tkr = str(row["ticker"]).strip()
        df = ohlcv.get(tkr)
        if df is None or len(df) < cfg.min_listed_days:
            continue
        c = df["Close"].dropna()
        v = df["Volume"].dropna()
        if len(c) < cfg.min_listed_days or len(v) < 21:
            continue
        close_now = float(c.iloc[-1])
        if not (cfg.min_price <= close_now <= cfg.max_price):
            continue
        adv = float((c * v).iloc[-21:-1].mean())
        if adv < cfg.min_adv_jpy:
            continue
        l0.append({"row": row.to_dict(), "df": df, "close": close_now, "adv": adv})
    counts["layer0"] = len(l0)

    # ---- Layer 1: 週足ステージ2 + 対指数相対強度 ----
    l1 = []
    for it in l0:
        ws = weekly_stage(it["df"], cfg)
        if ws is None or ws["stage_w"] != 2:
            continue
        ex = rs_vs_index(it["df"], index_daily, cfg.rs_vs_index_weeks)
        if ex is None or ex <= 0:
            continue      # 指数に勝っていない銘柄はここで落とす
        it.update(ws); it["rs_excess"] = ex
        l1.append(it)
    counts["layer1"] = len(l1)

    # ---- Layer 2: トレンドテンプレート8条件 ----
    # ★RSレーティングは「ユニバース全体」でパーセンタイルを取る(IBD方式)。
    # トレンドテンプレート通過後の少数銘柄内で順位付けすると、母集団が数銘柄しかない
    # 場合にパーセンタイルが25/50/75/100等の粗い値にしかならず、閾値70が意味を失う。
    # 市場全体の中で相対的に強いかを見るのがRSレーティングの本来の定義なので、
    # Layer0(流動性通過)の全銘柄を母集団とする。
    perf_all = {}
    for it in l0:
        perf_all[it["row"]["ticker"]] = rs_performance(it["df"])
    ratings = rs_ratings(perf_all)

    tt_ok = []
    for it in l1:
        tt = trend_template(it["df"], cfg)
        if tt is None or not tt["tt_pass_7"]:
            continue
        it.update(tt)
        tt_ok.append(it)
    l2 = []
    for it in tt_ok:
        rr = ratings.get(it["row"]["ticker"])
        if rr is None or rr < cfg.rs_rating_min:
            continue
        it["rs_rating"] = rr
        l2.append(it)
        watchlist.append({
            "ticker": it["row"]["ticker"], "code": it["row"]["ticker"].replace(".T", ""),
            "name": it["row"].get("name", ""), "sector": it["row"].get("sector", ""),
            "rs_rating": round(rr, 1), "stage_w": it["stage_w"],
            "pct_from_high": round(it.get("pct_from_high") or 0, 4),
        })
    counts["layer2"] = len(l2)

    # 業種ごとのRS中央値(スコア用)
    sec_rs = {}
    for it in l2:
        sec_rs.setdefault(it["row"].get("sector") or "不明", []).append(it["rs_rating"])
    sec_med = {k: pd.Series(v).median() for k, v in sec_rs.items()}
    top_secs = set()
    if sec_med:
        cut = pd.Series(sec_med).quantile(0.75)
        top_secs = {k for k, v in sec_med.items() if v >= cut}

    # ---- Layer 3: 日足ステージ1(発注可) or 6(監視のみ) ----
    l3 = []
    for it in l2:
        ds = daily_stage(it["df"], cfg)
        if ds is None or ds["stage_d"] not in (1, 6):
            continue
        it.update(ds)
        l3.append(it)
    counts["layer3"] = len(l3)

    # ---- Layer 4: セットアップ点灯 ----
    l4 = []
    for it in l3:
        su = detect_setup(it["df"], it["stage_d"], cfg)
        if su is None:
            continue
        it["setup_detail"] = su
        l4.append(it)
    counts["layer4"] = len(l4)

    # ---- Layer 5: 執行価格・リスク計算・ポートフォリオ制約 ----
    heat = portfolio_heat(positions, equity)
    dd = drawdown_state(equity, peak_equity, cfg)
    can_new, gate_reason = gate_new_entries(heat, dd, cfg)

    cands: List[CandidateV6] = []
    for it in l4:
        su = it["setup_detail"]
        row = it["row"]
        order_type, entry = _entry_price(su, cfg)
        atr = su.get("atr")
        if atr is None:
            a = atr_wilder(it["df"]["High"], it["df"]["Low"], it["df"]["Close"], cfg.atr_period)
            atr = float(a.iloc[-1]) if not pd.isna(a.iloc[-1]) else 0.0
        if atr <= 0:
            continue

        risk_mult = dd["risk_mult"] if dd["risk_mult"] > 0 else 1.0
        plan = build_risk_plan(entry, su["recent_swing_low"], atr, equity, cfg,
                               risk_pct_override=cfg.risk_per_trade * risk_mult)
        if plan.rejected:
            drops.append({"code": row["ticker"].replace(".T", ""),
                          "name": row.get("name", ""), "layer": "Layer5(リスク)",
                          "reason": plan.rejected})
            continue

        c = CandidateV6(
            ticker=row["ticker"], code=row["ticker"].replace(".T", ""),
            name=row.get("name", ""), sector=row.get("sector") or "不明",
            setup=su["setup"],
            stage_w=it["stage_w"], slope_w=it["slope_w"],
            rs_rating=it["rs_rating"], rs_excess=it["rs_excess"],
            stage_d=it["stage_d"], stage1_days=it.get("stage1_days", 0),
            stage_days=it.get("stage_days", 0),
            detail=su,
            order_type=order_type, entry=plan.entry, stop=plan.stop,
            stop_pct=plan.stop_pct, stop_kind=plan.stop_kind,
            shares=plan.shares, risk_amount=plan.risk_amount, risk_pct=plan.risk_pct,
            target_2r=plan.target_2r,
            expire_date=_bday_str(today, cfg.order_expire_days),
            tradable=(it["stage_d"] == 1),
        )
        if it["stage_d"] == 6:
            c.flags.append("日足ステージ6(上昇の始まり) — 監視のみ。ステージ1移行後に発注可")

        # リスクリワード(直近スイング高値まで)
        rr = None
        try:
            hi = float(it["df"]["High"].iloc[-60:].max())
            if plan.entry > plan.stop and hi > plan.entry:
                rr = (hi - plan.entry) / (plan.entry - plan.stop)
        except Exception:
            pass
        _score(c, c.sector in top_secs, rr, cfg)
        cands.append(c)

    # 並べ替え(スコア降順)→ 業種上限 → 件数上限
    cands.sort(key=lambda x: -x.score)
    picked, sec_used = [], {}
    for c in cands:
        if len(picked) >= cfg.max_notify:
            break
        if sec_used.get(c.sector, 0) >= cfg.max_per_sector:
            drops.append({"code": c.code, "name": c.name, "layer": "Layer5(業種分散)",
                          "reason": f"同一業種({c.sector})は{cfg.max_per_sector}社まで"})
            continue
        picked.append(c)
        sec_used[c.sector] = sec_used.get(c.sector, 0) + 1
    counts["layer5"] = len(picked)

    if not can_new:
        for c in picked:
            c.tradable = False
            c.flags.append(gate_reason)

    return {
        "picked": picked, "counts": counts, "drops": drops,
        "watchlist": watchlist,
        "heat": heat, "dd": dd, "can_new": can_new, "gate_reason": gate_reason,
        "equity": equity,
    }


def _bday_str(today: str, n: int) -> str:
    try:
        return (pd.Timestamp(today) + pd.tseries.offsets.BDay(n)).strftime("%m/%d")
    except Exception:
        return ""
