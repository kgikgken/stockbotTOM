"""プール正規化と合成（DESIGN.md §6 / TASKS.md T-301）。

- 基準集合（§6.1）: 直近 pool_days 営業日（T を含む）の、ゲート通過・押し目状態
  （形成中/反発開始/ブレイク）の銘柄×日をプールし、これに対する位置で正規化する。
  pool には pipeline.compute_daily_features() の出力（既にゲート通過・状態で絞り込み
  済み）を pool_days 日ぶん縦に連結したものを渡す想定。プールの日数が pool_days に
  満たない場合は、利用可能な日数だけで計算し、その旨をログに出す
- 特徴量の変換（§6.2）:
  - ↑: プール内百分位（0〜1、自分自身を含む集合内での位置）。↓: 1 − 百分位
  - ∩（帯 [a,b]）: 帯の内側は1、外側は帯幅 w=b-a だけ離れた所で0になる線形減衰
  - 二値: そのまま0/1（d3_bad_news は 1-値。d4_climax は方向未確定のため次元合成に
    入れない＝スコア化しない）
  - 欠損: 0.5
- 次元スコアと総合（§6.3）: 次元スコア=その次元の特徴量スコアの単純平均、
  状態別に使う次元集合、V1（等加重）/V2（D3を2倍）/V3（等加重＋出来高ゲート）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..features import dimensions, pullback

POOL_DAYS = 20  # DESIGN.md §6.1, §12（既定値。呼び出し側は config.Settings.pool_days を渡す）

# DESIGN.md §6.3: 次元スコア・総合の対象は D1〜D7（D8 は採点しない）
SCORED_DIMENSIONS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
DIMENSIONS_BY_STATE = {
    pullback.STATE_FORMING: ["D1", "D2", "D3", "D4", "D5", "D7"],
    pullback.STATE_BOUNCE: list(SCORED_DIMENSIONS),
    pullback.STATE_BREAK: list(SCORED_DIMENSIONS),
}
# §6.2: d4_climax は L1 で方向を決めるまで次元合成に入れない
EXCLUDED_FROM_DIMENSION_SCORE = {"d4_climax"}
# §6.2: d3_bad_news は「1 - 値」（悪材料が無いほど高スコア）
INVERTED_BINARY = {"d3_bad_news"}
# §6.3 V3: 出来高ゲート（d4_pb_ratio > 1.0 の銘柄は候補に出さず監視のみ）
V3_VOLUME_GATE_FEATURE = "d4_pb_ratio"
V3_VOLUME_GATE_MAX = 1.0

FEATURE_META_BY_ID = {m[0]: m for m in dimensions.FEATURE_METADATA}

# dimension -> 採点対象の feature id（D1〜D7、d4_climax を除く）
SCORED_FEATURE_IDS_BY_DIM = {
    dim: [fid for fid, d, direction, _band in dimensions.FEATURE_METADATA
          if d == dim and direction is not None and fid not in EXCLUDED_FROM_DIMENSION_SCORE]
    for dim in SCORED_DIMENSIONS
}


def _percentile_up(value: float, pool_values: np.ndarray) -> float:
    """プール内百分位（自分自身を含む集合内での位置、0〜1）。プールが空なら NaN。"""
    if pool_values.size == 0:
        return np.nan
    return float(np.mean(pool_values <= value))


def _band_score(value: float, lo: float, hi: float) -> float:
    """∩ 帯変換: 帯内は1、帯幅ぶん外で0になる線形減衰（§6.2）。"""
    w = hi - lo
    if w <= 0:
        return np.nan
    if value < lo:
        dist = lo - value
    elif value > hi:
        dist = value - hi
    else:
        dist = 0.0
    return float(np.clip(1.0 - dist / w, 0.0, 1.0))


def feature_score(fid: str, value, pool_values: np.ndarray) -> float:
    """特徴量1件のスコア（0〜1）。§6.2 の変換規則どおり。欠損は0.5。"""
    meta = FEATURE_META_BY_ID.get(fid)
    if meta is None:
        raise KeyError(f"unknown feature id: {fid}")
    _, _dim, direction, band = meta
    if direction is None or fid in EXCLUDED_FROM_DIMENSION_SCORE:
        return np.nan  # D8（採点しない）/ d4_climax（次元合成に入れない）
    if pd.isna(value):
        return 0.5

    if direction == "binary":
        v = 1.0 if bool(value) else 0.0
        return (1.0 - v) if fid in INVERTED_BINARY else v

    if direction == "band":
        lo, hi = band
        score = _band_score(float(value), lo, hi)
        return 0.5 if np.isnan(score) else score

    # up / down: プール内百分位
    valid_pool = pool_values[~pd.isna(pool_values)].astype(float)
    pct = _percentile_up(float(value), valid_pool)
    if np.isnan(pct):
        return 0.5
    return pct if direction == "up" else (1.0 - pct)


def compute_composite_scores(pool: pd.DataFrame, asof: pd.Timestamp,
                             pool_days: int = POOL_DAYS, log=print) -> pd.DataFrame:
    """T（asof）時点の特徴量スコア・次元スコア・V1/V2/V3 総合スコアを計算する。

    pool は pipeline.compute_daily_features() の出力（pipeline.DAILY_FEATURES_COLS の
    列を持つ、ゲート通過・押し目状態で絞り込み済みの日次特徴量）を、直近 pool_days
    営業日ぶん（asof を含む）縦に連結したもの。

    戻り値: pool のうち date==asof の行だけを抜き出し、dim_D1_score..dim_D7_score /
    score_v1 / score_v2 / score_v3 / v3_volume_gate_pass を埋めて返す（他の列はそのまま）。
    プールの日数が pool_days に満たない場合は、利用可能な日数だけで計算しログに出す。
    """
    asof = pd.Timestamp(asof)
    if len(pool):
        # 先読み防止: asof より後の日付が混入していても使わない（基準集合は T を含む
        # 直近 pool_days 営業日、DESIGN.md §6.1）。呼び出し側の取り違えを内部でも守る
        pool = pool.loc[pd.to_datetime(pool["date"]) <= asof]
    unique_dates = pd.to_datetime(pool["date"]).unique() if len(pool) else []
    n_pool_days = len(unique_dates)
    if n_pool_days < pool_days:
        log(f"[composite] プール日数不足: {n_pool_days}/{pool_days} 日で計算します")

    today = pool[pd.to_datetime(pool["date"]) == asof].copy()
    if len(today) == 0:
        return today

    score_cols: dict[str, pd.Series] = {}
    for dim, fids in SCORED_FEATURE_IDS_BY_DIM.items():
        for fid in fids:
            _, _dim2, direction, _band = FEATURE_META_BY_ID[fid]
            # プール内百分位が要るのは up/down のみ。binary/band 列は bool/NA 混在があり得る
            # ため to_numpy(dtype=float) を通さない（feature_score も up/down 以外では未使用）
            if direction in ("up", "down") and fid in pool.columns:
                pool_values = pool[fid].to_numpy(dtype=float)
            else:
                pool_values = np.array([])
            score_cols[f"__score_{fid}"] = today[fid].map(
                lambda v, _fid=fid, _pv=pool_values: feature_score(_fid, v, _pv))
    scores_df = pd.DataFrame(score_cols, index=today.index)

    for dim, fids in SCORED_FEATURE_IDS_BY_DIM.items():
        cols = [f"__score_{fid}" for fid in fids]
        today[f"dim_{dim}_score"] = scores_df[cols].mean(axis=1) if cols else np.nan

    def _used_dims(state: str) -> list[str]:
        return DIMENSIONS_BY_STATE.get(state, SCORED_DIMENSIONS)

    def _composite(row: pd.Series, weighted: bool) -> float:
        dims = _used_dims(row["state"])
        vals = [row[f"dim_{d}_score"] for d in dims]
        if not weighted:
            return float(np.mean(vals)) * 100
        weights = [2.0 if d == "D3" else 1.0 for d in dims]
        return float(np.average(vals, weights=weights)) * 100

    today["score_v1"] = today.apply(lambda r: _composite(r, weighted=False), axis=1)
    today["score_v2"] = today.apply(lambda r: _composite(r, weighted=True), axis=1)
    today["score_v3"] = today["score_v1"]  # DESIGN.md §6.3: V3 の重みは V1 と同じ等加重

    if V3_VOLUME_GATE_FEATURE in today.columns:
        pb_ratio = today[V3_VOLUME_GATE_FEATURE]
    else:
        pb_ratio = pd.Series(np.nan, index=today.index)
    # d4_pb_ratio > 1.0 の銘柄は V3 の候補から除く（監視のみ）。欠損は除外しない
    # （出来高ゲート条件が確認できないことを不利に倒さない）
    today["v3_volume_gate_pass"] = ~(pb_ratio > 1.0)

    return today
