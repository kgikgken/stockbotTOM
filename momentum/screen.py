"""モメンタム・スクリーニング本体 — 広い候補プール→3状態分類→アクション候補への絞り込み.

ポジション構成のハード制約(ユーザー明示指定・裁量による例外なし):
- 実保有は最大3銘柄
- 同一業種は1銘柄まで
- リスク%は確信度に関わらず固定0.5%
- レジームが防御モードなら新規は例外なくゼロ
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import Config
from .indicators import compute_momentum_features, momentum_score
from .classify import classify_state, build_candidate, Candidate


def build_pool(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               bench_logclose, cfg: Config) -> tuple[list[dict], dict]:
    """全ユニバースを走査し特徴量を計算、モメンタムスコア上位cfg.pool_size銘柄をプールとする。"""
    scored: List[dict] = []
    considered = 0
    for _, row in universe.iterrows():
        tkr = str(row["ticker"]).strip()
        df = ohlcv.get(tkr)
        if df is None:
            continue
        feat = compute_momentum_features(df, bench_logclose, cfg)
        if feat is None:
            continue
        if feat["close"] < cfg.min_price or feat["adv20_jpy"] < cfg.min_adv_jpy:
            continue
        considered += 1
        score = momentum_score(feat, cfg)
        scored.append({"row": row.to_dict(), "feat": feat, "score": score})

    scored.sort(key=lambda x: -x["score"])
    pool = scored[: cfg.pool_size]
    stats = {"universe_considered": considered, "pool_size": len(pool)}
    return pool, stats


def run_screen(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               bench_logclose, regime: dict, cfg: Config) -> dict:
    pool, pool_stats = build_pool(universe, ohlcv, bench_logclose, cfg)

    state_count = {"A": 0, "B": 0, "C": 0}
    fired: List[Candidate] = []
    for item in pool:
        state = classify_state(item["feat"], cfg)
        if state is None:
            continue
        state_count[state] = state_count.get(state, 0) + 1
        c = build_candidate(item["row"], item["feat"], state, cfg)
        if c is not None:
            fired.append(c)

    fired.sort(key=lambda x: -x.score)

    picked: List[Candidate] = []
    sector_used: Dict[str, int] = {}
    rejects: List[dict] = []
    blocked_reason = None

    if not regime.get("attack", False):
        blocked_reason = f"レジーム防御モードにつき新規エントリー全停止({regime.get('detail','')})"
        for c in fired:
            rejects.append({"code": c.code, "name": c.name, "stage": "レジーム",
                            "reason": blocked_reason, "close": round(c.feat["close"], 1)})
    else:
        for c in fired:
            if len(picked) >= cfg.max_positions:
                rejects.append({"code": c.code, "name": c.name, "stage": "上限",
                                "reason": f"実保有上限{cfg.max_positions}銘柄に到達",
                                "close": round(c.feat["close"], 1)})
                continue
            n_sector = sector_used.get(c.sector, 0)
            if n_sector >= cfg.max_per_sector:
                rejects.append({"code": c.code, "name": c.name, "stage": "セクター分散",
                                "reason": f"同一業種({c.sector})は{cfg.max_per_sector}銘柄まで",
                                "close": round(c.feat["close"], 1)})
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

    total_risk = sum(c.risk_pct for c in picked)
    stats = {
        **pool_stats,
        "state_count": state_count,
        "fired": len(fired),
        "picked": len(picked),
        "rejected": len(rejects),
        "total_risk": total_risk,
        "risk_cap": cfg.total_risk_cap,
        "blocked_reason": blocked_reason,
    }
    return {"picked": picked, "rejects": rejects, "stats": stats, "pool_stats": pool_stats}
