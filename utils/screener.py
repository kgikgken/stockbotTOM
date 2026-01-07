# ============================================
# utils/screener.py
# 銘柄スクリーニング本体
# ============================================

import pandas as pd

from utils.features import (
    calc_trend_features,
    detect_setup_a1_a2,
    detect_setup_b,
)
from utils.entry import calc_entry_zone, decide_action
from utils.rr_ev import (
    calc_rr,
    estimate_pwin,
    calc_ev,
    adjust_ev_by_market,
    pass_rr_filter,
    pass_ev_filter,
    calc_speed_metrics,
    pass_speed_filter,
)
from utils.util import clamp


# --------------------------------------------
# 単銘柄スクリーニング
# --------------------------------------------
def screen_one_stock(
    df: pd.DataFrame,
    meta: dict,
    market_ctx: dict,
    sector_rank: int,
    macro_risk: bool,
):
    """
    df: 日足データ
    meta: 銘柄メタ情報（code, name, sector 等）
    market_ctx:
        - market_score
        - delta_market
    """

    # ==============================
    # 特徴量
    # ==============================
    feat = calc_trend_features(df)
    if feat is None:
        return None

    # ==============================
    # セットアップ判定
    # ==============================
    setup_type = None

    a_type = detect_setup_a1_a2(df, feat)
    if a_type:
        setup_type = a_type  # "A1" or "A2"
    else:
        if detect_setup_b(df, feat):
            setup_type = "B"

    if setup_type is None:
        return None

    # ==============================
    # エントリー帯
    # ==============================
    entry_info = calc_entry_zone(df, setup_type, feat)
    if entry_info is None:
        return None

    entry = entry_info["entry"]
    entry_low = entry_info["entry_low"]
    entry_high = entry_info["entry_high"]
    stop = entry_info["stop"]
    tp1 = entry_info["tp1"]
    tp2 = entry_info["tp2"]
    atr = entry_info["atr"]
    gu_flag = entry_info["gu_flag"]

    # ==============================
    # RR
    # ==============================
    rr = calc_rr(entry, stop, tp2)
    if not pass_rr_filter(rr, market_ctx["market_score"]):
        return None

    # ==============================
    # 勝率 proxy
    # ==============================
    pwin = estimate_pwin({
        "trend_strength": feat["trend_strength"],
        "rs_strength": feat["rs_strength"],
        "sector_rank": sector_rank,
        "volume_quality": feat["volume_quality"],
        "gu_flag": gu_flag,
        "liquidity": feat["liquidity"],
    })

    # ==============================
    # EV / 補正EV
    # ==============================
    ev = calc_ev(rr, pwin)
    adj_ev = adjust_ev_by_market(
        ev,
        market_ctx["market_score"],
        market_ctx["delta_market"],
        macro_risk
    )

    if not pass_ev_filter(adj_ev, threshold=0.5):
        return None

    # ==============================
    # 速度評価
    # ==============================
    speed = calc_speed_metrics(entry, tp2, atr, rr)
    if not pass_speed_filter(
        speed["expected_days"],
        speed["r_day"]
    ):
        return None

    # ==============================
    # 行動判定
    # ==============================
    action = decide_action(
        current_price=df["Close"].iloc[-1],
        entry_center=entry,
        atr=atr,
        gu_flag=gu_flag,
    )

    # ==============================
    # 結果
    # ==============================
    return {
        "code": meta["code"],
        "name": meta.get("name", ""),
        "sector": meta.get("sector", ""),
        "setup": setup_type,
        "entry": entry,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "ev": ev,
        "adj_ev": adj_ev,
        "expected_days": speed["expected_days"],
        "r_day": speed["r_day"],
        "atr": atr,
        "gu_flag": gu_flag,
        "action": action,
        "sector_rank": sector_rank,
    }


# --------------------------------------------
# 複数銘柄スクリーニング
# --------------------------------------------
def run_screening(
    stock_data: dict,
    meta_data: dict,
    market_ctx: dict,
    sector_ranks: dict,
    macro_risk: bool,
    max_candidates: int,
):
    """
    stock_data: { code: df }
    meta_data: { code: meta }
    sector_ranks: { sector: rank }
    """

    results = []

    for code, df in stock_data.items():
        meta = meta_data.get(code)
        if meta is None:
            continue

        sector = meta.get("sector", "")
        sector_rank = sector_ranks.get(sector, 99)

        r = screen_one_stock(
            df=df,
            meta=meta,
            market_ctx=market_ctx,
            sector_rank=sector_rank,
            macro_risk=macro_risk,
        )
        if r:
            results.append(r)

    # ==============================
    # 並び替え（最重要）
    # ==============================
    results.sort(
        key=lambda x: (
            -x["adj_ev"],        # 補正EV最優先
            -x["r_day"],         # 速度
            -x["rr"],            # RR
        )
    )

    return results[:max_candidates]