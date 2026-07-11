"""swing2w スクリーニング本体 — 回転率二分 → エンジンR/M判定 → 自前の3枠へ絞り込み.

ポジション構成のハード制約(モメンタム系とは別枠・ユーザー明示指定):
- 実保有は最大3銘柄(このエンジン専用の別枠)
- 同一業種は1銘柄まで(このエンジン内で独立判定)
- リスク%は固定0.5%
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .indicators import compute_swing_features, compute_sector_relative_z, tob_suspect
from .classify import classify_engine_r, classify_engine_m, Candidate


def build_pool(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame], cfg: Config):
    eligible: List[dict] = []
    tob_rejects: List[dict] = []
    for _, row in universe.iterrows():
        tkr = str(row["ticker"]).strip()
        df = ohlcv.get(tkr)
        if df is None:
            continue
        feat = compute_swing_features(df, cfg)
        if feat is None:
            continue
        if feat["close"] < cfg.min_price or feat["adv20_jpy"] < cfg.min_adv_jpy:
            continue

        is_tob, tob_reason = tob_suspect(df, cfg)
        if is_tob:
            tob_rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                                "stage": "TOB疑い", "reason": tob_reason, "close": round(feat["close"], 1)})
            continue

        eligible.append({"row": row.to_dict(), "feat": feat})

    # ★回転率二分(この設計の核心)。ADVを回転率の代理指標として使用。
    advs = sorted(it["feat"]["adv20_jpy"] for it in eligible)
    if advs:
        low_cut = advs[int(len(advs) * cfg.turnover_low_pct)] if len(advs) > 0 else 0
        high_cut = advs[min(int(len(advs) * cfg.turnover_high_pct), len(advs) - 1)] if len(advs) > 0 else 0
    else:
        low_cut = high_cut = 0

    low_turnover = [it for it in eligible if it["feat"]["adv20_jpy"] <= low_cut]
    high_turnover = [it for it in eligible if it["feat"]["adv20_jpy"] >= high_cut]

    # 業種内相対z(エンジンR用)は全母集団(eligible)を基準に計算し、低回転率側にのみ適用する
    # (セクター内の比較対象を回転率で削らないため。統計的な安定性を優先)。
    sector_z = compute_sector_relative_z(eligible, cfg)

    stats = {
        "universe_considered": len(eligible), "tob_excluded": len(tob_rejects),
        "low_turnover_n": len(low_turnover), "high_turnover_n": len(high_turnover),
        "turnover_low_cut_oku": low_cut / 1e8, "turnover_high_cut_oku": high_cut / 1e8,
    }
    return low_turnover, high_turnover, sector_z, stats, tob_rejects, eligible


def run_screen(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame], cfg: Config) -> dict:
    low_turnover, high_turnover, sector_z, pool_stats, tob_rejects, eligible = build_pool(universe, ohlcv, cfg)

    fired: List[Candidate] = []
    for item in low_turnover:
        c = classify_engine_r(item, sector_z, cfg)
        if c is not None:
            fired.append(c)
    for item in high_turnover:
        c = classify_engine_m(item, cfg)
        if c is not None:
            fired.append(c)

    fired.sort(key=lambda x: -x.score)

    picked: List[Candidate] = []
    overflow: List[Candidate] = []
    sector_used: Dict[str, int] = {}
    rejects: List[dict] = list(tob_rejects)

    for c in fired:
        if len(picked) >= cfg.max_positions:
            rejects.append({"code": c.code, "name": c.name, "stage": "上限",
                            "reason": f"実保有上限{cfg.max_positions}銘柄に到達 → 参考層",
                            "close": round(c.feat["close"], 1)})
            overflow.append(c)
            continue
        n_sector = sector_used.get(c.sector, 0)
        if n_sector >= cfg.max_per_sector:
            rejects.append({"code": c.code, "name": c.name, "stage": "セクター分散",
                            "reason": f"同一業種({c.sector})は{cfg.max_per_sector}銘柄まで(swing2w内) → 参考層",
                            "close": round(c.feat["close"], 1)})
            overflow.append(c)
            continue

        c.risk_pct = cfg.risk_pct_fixed
        if cfg.account_equity > 0:
            risk_amount = cfg.account_equity * c.risk_pct / 100
            c.shares = int(risk_amount / c.risk_w // 100 * 100)
            if c.shares * c.entry < cfg.min_exec_jpy:
                rejects.append({"code": c.code, "name": c.name, "stage": "サイジング",
                                "reason": f"サイズ過小(≈{c.shares*c.entry/1e4:.0f}万円)につき見送り",
                                "close": round(c.feat["close"], 1)})
                continue

        picked.append(c)
        sector_used[c.sector] = n_sector + 1

    watch = overflow[: cfg.max_watch]
    total_risk = sum(c.risk_pct for c in picked)
    n_r = sum(1 for c in fired if c.engine == "R")
    n_m = sum(1 for c in fired if c.engine == "M")

    stats = {
        **pool_stats, "fired": len(fired), "fired_r": n_r, "fired_m": n_m,
        "picked": len(picked), "watch": len(watch), "rejected": len(rejects),
        "total_risk": total_risk, "risk_cap": cfg.total_risk_cap,
    }
    return {"picked": picked, "watch": watch, "rejects": rejects, "stats": stats, "eligible": eligible}
