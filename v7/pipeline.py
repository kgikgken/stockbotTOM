"""v7.1 パイプライン — L0〜L5の統合とカバレッジ判定。

第6部の時点整合が本モジュールの中核的な責務:
  規則1: DCS内は t−n に揃える。最も遅い次元にタイムスタンプを合わせる
  規則5: 出力に基準時点を明記する

これを怠ると④⑤(スイングポイント確定待ち)を当日情報として扱うことになり、
先読みバイアスが発生する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .core import to_weekly
from .dimensions import (dim1_trend, dim2_relative_raw, dim2_relative,
                         dim3_supply_demand, dim4_time, dim5_volatility, detect_base)
from .gates import (l0_regime, l0_distribution_days, build_universe,
                    build_market_index, RISK_ON, NEUTRAL, RISK_OFF)

DIM_NAMES = {1: "①トレンド", 2: "②相対力", 3: "③需給", 4: "④時間", 5: "⑤収縮"}


@dataclass
class CandidateV7:
    ticker: str
    code: str
    name: str
    sector: str
    dims: Dict[int, float | None] = field(default_factory=dict)
    parts: Dict[int, dict] = field(default_factory=dict)
    coverage: float = 0.0
    satisfied: List[int] = field(default_factory=list)
    missing: List[int] = field(default_factory=list)
    dcs: float = 0.0
    category: str = ""          # 本命 / 監視 / 除外
    maturity_weeks: int | None = None   # ④欠落時の成熟予測(§9.3.1)
    notes: List[str] = field(default_factory=list)


def _align(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """時点整合(§6.2 規則1)。全次元を t−n の断面で計算するため末尾n本を落とす。

    ★これが先読みバイアスを構造的に防ぐ唯一の仕組み。
    fractal(n)のスイング点は t+n まで確定しないため、
    ①②③だけ当日値を使うと基準時点が混在する。
    """
    return df.iloc[:-lag] if (lag > 0 and len(df) > lag) else df


def run_pipeline(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
                 index_daily: pd.DataFrame, cfg, today: str) -> dict:
    counts = {"universe": len(universe), "l1": 0, "scored": 0,
              "main": 0, "watch": 0, "excluded": 0}

    # ---- L0 レジーム判定 ----
    regime = l0_regime(index_daily, cfg)
    dd = l0_distribution_days(index_daily, cfg)

    if regime["state"] == RISK_OFF:
        # §5.1.6「L0がOFFの時に候補ゼロを出すのはバグではなく仕様である」
        return {"regime": regime, "dd": dd, "counts": counts,
                "main": [], "watch": [], "drops": [],
                "halted": True,
                "asof": {"dcs": None, "price": None, "regime": regime.get("asof")}}

    # ---- L1 ユニバース ----
    l1, drops = build_universe(universe, ohlcv, cfg)
    counts["l1"] = len(l1)
    if not l1:
        return {"regime": regime, "dd": dd, "counts": counts,
                "main": [], "watch": [], "drops": drops, "halted": False,
                "asof": {"dcs": None, "price": None, "regime": regime.get("asof")}}

    # ---- 市場ファクター(自前構築・§10.4.3) ----
    aligned = {x["row"]["ticker"]: _align(x["df"], cfg.align_lag_days) for x in l1}
    market = build_market_index(aligned)

    # ---- L2 次元スコア算出(全次元を t−n で計算) ----
    enabled = [d for d, on in [(1, cfg.dim1_enabled), (2, cfg.dim2_enabled),
                               (3, cfg.dim3_enabled), (4, cfg.dim4_enabled),
                               (5, cfg.dim5_enabled)] if on]

    # ②はユニバース内パーセンタイルが必要なため2段階
    raw2, tmp = {}, []
    for x in l1:
        tkr = x["row"]["ticker"]
        df = aligned[tkr]
        base = detect_base(df, cfg)
        item = {"x": x, "df": df, "base": base, "tkr": tkr}
        if 2 in enabled and market is not None:
            raw2[tkr] = dim2_relative_raw(df, market, cfg)
        tmp.append(item)

    # 残差モメンタムのパーセンタイル
    resid_vals = {k: v["resid_mom"] for k, v in raw2.items() if v.get("resid_mom") is not None}
    resid_pct = _to_percentile(resid_vals)

    # セクター相対(2-b: セクター内順位 / 2-c: セクター自体の順位)
    sec_of = {i["tkr"]: (i["x"]["row"].get("sector") or "不明") for i in tmp}
    sec_inner_pct, sec_self_pct = _sector_percentiles(resid_vals, sec_of)

    cands: List[CandidateV7] = []
    for it in tmp:
        x, df, base, tkr = it["x"], it["df"], it["base"], it["tkr"]
        row = x["row"]
        c = CandidateV7(ticker=tkr, code=tkr.replace(".T", ""),
                        name=row.get("name", ""), sector=sec_of[tkr])

        if 1 in enabled:
            r = dim1_trend(df, cfg)
            c.dims[1] = r["score"]; c.parts[1] = r["parts"]
            if r.get("reason"): c.notes.append(f"①{r['reason']}")
        if 2 in enabled:
            r = dim2_relative(raw2.get(tkr, {}), resid_pct.get(tkr),
                              sec_inner_pct.get(tkr), sec_self_pct.get(tkr), cfg)
            c.dims[2] = r["score"]; c.parts[2] = r["parts"]
            if r.get("reason"): c.notes.append(f"②{r['reason']}")
        if 3 in enabled:
            r = dim3_supply_demand(df, cfg)
            c.dims[3] = r["score"]; c.parts[3] = r["parts"]
        if 4 in enabled:
            r = dim4_time(df, cfg, base)
            c.dims[4] = r["score"]; c.parts[4] = r["parts"]
        if 5 in enabled:
            r = dim5_volatility(df, cfg, base)
            c.dims[5] = r["score"]; c.parts[5] = r["parts"]

        # ---- L3 カバレッジ判定 + DCS ----
        vals = [v for v in c.dims.values() if v is not None]
        if not vals:
            continue
        c.satisfied = [d for d, v in c.dims.items() if v is not None and v >= cfg.theta]
        c.missing = [d for d, v in c.dims.items() if v is None or v < cfg.theta]
        c.coverage = len(c.satisfied) / len(enabled) if enabled else 0.0
        c.dcs = float(np.mean(vals)) * 100.0

        if c.coverage >= cfg.coverage_main:
            c.category = "本命"
        elif c.coverage >= cfg.coverage_watch:
            c.category = "監視"
        else:
            c.category = "除外"

        # §9.3.1 成熟予測: ④のみ欠落なら充足までの推定週数を出す
        if c.missing == [4] and base and base.get("weeks") is not None:
            need = cfg.d4_base_ideal_weeks - int(base["weeks"])
            c.maturity_weeks = max(0, need)

        cands.append(c)

    counts["scored"] = len(cands)

    # ---- L4/L5 は yfinance では実装不可(config.data_limitations に明記) ----

    main = sorted([c for c in cands if c.category == "本命"], key=lambda z: -z.dcs)
    watch = sorted([c for c in cands if c.category == "監視"], key=lambda z: -z.dcs)
    counts["main"] = len(main)
    counts["watch"] = len(watch)
    counts["excluded"] = sum(1 for c in cands if c.category == "除外")

    # ---- 基準時点(§6.2 規則5) ----
    dcs_asof = price_asof = None
    if l1:
        try:
            sample = aligned[l1[0]["row"]["ticker"]]
            dcs_asof = str(sample.index[-1].date())
            price_asof = str(l1[0]["df"].index[-1].date())
        except Exception:
            pass

    return {
        "regime": regime, "dd": dd, "counts": counts,
        "main": main[: cfg.max_output], "watch": watch[: cfg.max_watch_output],
        "drops": drops, "halted": False,
        "enabled_dims": enabled,
        "asof": {"dcs": dcs_asof, "price": price_asof, "regime": regime.get("asof")},
        "market_built": market is not None,
    }


def _to_percentile(values: dict) -> dict:
    if not values:
        return {}
    s = pd.Series(values)
    return {k: float(v) for k, v in (s.rank(pct=True) * 100.0).items()}


def _sector_percentiles(resid: dict, sector_of: dict) -> tuple[dict, dict]:
    """2-b: セクター内での順位 / 2-c: セクター自体の順位。

    §9.2: セクター相対力にはTOPIX-17相当の粗い分類が望ましいが、
    yfinance運用では universe_jpx.csv の33業種を使う。
    構成銘柄の少ない業種はノイジーになるため、5銘柄未満の業種は
    セクター自体の順位付けから除外する。
    """
    inner, self_pct = {}, {}
    by_sec: dict = {}
    for tkr, v in resid.items():
        by_sec.setdefault(sector_of.get(tkr, "不明"), {})[tkr] = v

    sec_median = {}
    for sec, m in by_sec.items():
        if len(m) >= 2:
            inner.update(_to_percentile(m))
        if len(m) >= 5:
            sec_median[sec] = float(np.median(list(m.values())))

    if sec_median:
        sp = _to_percentile(sec_median)
        for tkr, sec in sector_of.items():
            if sec in sp:
                self_pct[tkr] = sp[sec]
    return inner, self_pct
