"""モメンタム・スクリーニング本体 — 広い候補プール→3状態分類→アクション候補への絞り込み.

ポジション構成のハード制約(ユーザー明示指定・裁量による例外なし):
- 実保有は最大3銘柄
- 同一業種は1銘柄まで
- リスク%は確信度に関わらず固定0.5%(レジームに関わらず変更しない・ユーザー明示指定)
- レジームが防御モードでも新規エントリーはブロックしない(ユーザー判断で撤廃)。
  個別シグナルに期待値があるという前提で銘柄選定は通常どおり実行し、防御モード中は
  候補・レポートに注意フラグを付けるのみ(警告に留め、機械的な足切りはしない)。
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import Config
from .indicators import compute_momentum_features, compute_pool_scores, tob_suspect
from .classify import classify_state, build_candidate, Candidate


def build_pool(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               bench_logclose, cfg: Config) -> tuple[list[dict], dict, list[dict]]:
    """全ユニバースを走査し特徴量を計算。TOB疑いを除外した後の母集団全体でz-score化して
    総合スコアを付与し(指示①)、上位cfg.pool_size銘柄をプールとする。"""
    eligible: List[dict] = []
    tob_rejects: List[dict] = []
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

        is_tob, tob_reason = tob_suspect(df, cfg)
        if is_tob:
            tob_rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                                "stage": "TOB疑い", "reason": tob_reason,
                                "close": round(feat["close"], 1)})
            continue

        eligible.append({"row": row.to_dict(), "feat": feat})

    # ★指示①: z-score化は母集団(流動性・TOB除外通過の全銘柄)全体に対して行う。
    # 先にpool_sizeで絞ってからz化すると、既に上位のものだけの分布になり歪む。
    eligible = compute_pool_scores(eligible, cfg)
    eligible.sort(key=lambda x: -x["score"])
    pool = eligible[: cfg.pool_size]
    # ★指示⑨(診断のみ・スコア式は変更しない): 候補プールの業種内訳を可視化する。
    # 業種モメンタム(Moskowitz & Grinblatt 1999)が実在するシグナルの可能性があるため、
    # 偏りが見えても現時点ではスコアのセクター中立化は行わない。
    sector_counts: Dict[str, int] = {}
    for it in pool:
        sec = it["row"].get("sector") or "不明"
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
    top_sectors = sorted(sector_counts.items(), key=lambda x: -x[1])[: cfg.sector_diag_top_n]

    stats = {"universe_considered": len(eligible), "pool_size": len(pool), "tob_excluded": len(tob_rejects),
             "top_sectors": top_sectors}
    return pool, stats, tob_rejects, eligible


def run_screen(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               bench_logclose, regime: dict, cfg: Config) -> dict:
    pool, pool_stats, tob_rejects, eligible = build_pool(universe, ohlcv, bench_logclose, cfg)

    state_count = {"A": 0, "B": 0, "C": 0}
    fired: List[Candidate] = []
    for item in pool:
        state = classify_state(item["feat"], cfg)
        if state is None:
            continue
        state_count[state] = state_count.get(state, 0) + 1
        c = build_candidate(item["row"], item["feat"], state, item["score"], cfg)
        if c is not None:
            fired.append(c)

    fired.sort(key=lambda x: -x.score)

    # ★指示②(sizing_mode=atr_scaledの時のみ使用): fired銘柄群のATR%中央値を基準値とする
    median_atr_pct = None
    if cfg.sizing_mode == "atr_scaled" and fired:
        atr_pcts = [c.feat["atr"] / c.feat["close"] * 100 for c in fired if c.feat["close"] > 0]
        if atr_pcts:
            s = sorted(atr_pcts)
            median_atr_pct = s[len(s) // 2] if len(s) % 2 else (s[len(s)//2 - 1] + s[len(s)//2]) / 2

    picked: List[Candidate] = []
    overflow: List[Candidate] = []
    sector_used: Dict[str, int] = {}
    rejects: List[dict] = list(tob_rejects)
    regime_caution = None

    # ★変更: レジーム防御モードはもはや新規エントリーの機械的ブロックではない(ユーザー判断で撤廃)。
    # 銘柄自身の状態A/Bシグナルに期待値があるという前提で、位置構築(3銘柄上限・セクター分散・
    # 固定リスク%)は通常どおり実行する。防御モード中は各候補に注意フラグを付け、
    # レポート全体にも警告バナーを出す(リスク%は縮小しない・ユーザー明示指定)。
    if not regime.get("attack", False):
        regime_caution = f"レジーム防御モード({regime.get('detail','')}) — 相場全体の地合いに注意。銘柄選定自体は通常どおり実行"

    for c in fired:
        if regime_caution:
            c.flags.append("⚠相場全体が防御モード。個別シグナルの期待値はレジーム条件付きである点に留意(通常より慎重に)")
        if len(picked) >= cfg.max_positions:
            rejects.append({"code": c.code, "name": c.name, "stage": "上限",
                            "reason": f"実保有上限{cfg.max_positions}銘柄に到達 → 参考層",
                            "close": round(c.feat["close"], 1)})
            overflow.append(c)
            continue
        n_sector = sector_used.get(c.sector, 0)
        if n_sector >= cfg.max_per_sector:
            rejects.append({"code": c.code, "name": c.name, "stage": "セクター分散",
                            "reason": f"同一業種({c.sector})は{cfg.max_per_sector}銘柄まで → 参考層",
                            "close": round(c.feat["close"], 1)})
            overflow.append(c)
            continue

        c.risk_pct = cfg.risk_pct_fixed
        if cfg.sizing_mode == "atr_scaled" and median_atr_pct:
            this_atr_pct = c.feat["atr"] / c.feat["close"] * 100 if c.feat["close"] > 0 else median_atr_pct
            factor = median_atr_pct / this_atr_pct if this_atr_pct > 0 else 1.0
            factor = max(cfg.atr_scale_min, min(cfg.atr_scale_max, factor))
            c.risk_pct = cfg.risk_pct_fixed * factor
            c.flags.append(f"ATR連動サイジング: 基準ATR%比 係数{factor:.2f}倍(指示②・確認要)")
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
    if cfg.sizing_mode == "atr_scaled" and total_risk > cfg.total_risk_cap and total_risk > 0:
        shrink = cfg.total_risk_cap / total_risk
        for c in picked:
            c.risk_pct *= shrink
            if cfg.account_equity > 0 and c.risk_w > 0:
                c.shares = int(cfg.account_equity * c.risk_pct / 100 / c.risk_w // 100 * 100)
        total_risk = sum(c.risk_pct for c in picked)
    stats = {
        **pool_stats,
        "state_count": state_count,
        "fired": len(fired),
        "picked": len(picked),
        "watch": len(watch),
        "rejected": len(rejects),
        "total_risk": total_risk,
        "risk_cap": cfg.total_risk_cap,
        "regime_caution": regime_caution,
    }
    return {"picked": picked, "watch": watch, "rejects": rejects, "stats": stats,
            "pool_stats": pool_stats, "eligible": eligible}
