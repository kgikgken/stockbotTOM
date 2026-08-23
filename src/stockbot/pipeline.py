"""日次パイプライン: 指標→ゲート→スイング→押し目→特徴量→地合い→プール正規化と合成を
通し、日次スナップショットに保存する（DESIGN.md §1 / TASKS.md T-206, T-208, T-301）。

DESIGN.md §1 の手順のうち、d3_template（§7, T-302）は未実装（常に NaN。§6.2 の欠損
規則により次元合成では 0.5 として扱われる）。次元スコア（dim_D1_score..dim_D7_score）・
総合スコア（score_v1/v2/v3）は T-301（scoring/composite.py）で計算する。history_pool
（直近 pool_days-1 営業日ぶんの過去の日次特徴量。load_recent_daily_features 参照）を
渡すと当日ぶんと合わせてプールを組む。渡さない場合は当日ぶんだけがプールになる
（DESIGN.md §6.1 の「プールの日数が足りない初期は利用可能な日数で計算」に該当する
一形態であり、1 日だけの自己正規化として composite 側がログに出す）。

採点対象（T-208）は「gate_pass（G0〜G3 全通過）かつ 状態が形成中/反発開始/ブレイク」。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .features import dimensions, gates, indicators, pullback, regime, swings
from .scoring import composite

# DESIGN.md §6.3: 次元スコアの対象は D1〜D7（D8 は採点しない）
SCORED_DIMENSIONS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
SCORE_VARIANTS = ["v1", "v2", "v3"]  # DESIGN.md §12: 変種は V1/V2/V3 のみ

STRUCT_COLS = [
    "ticker", "date", "state",
    "h0_date", "l0_date", "lp_date",
    "h0_high", "l0_low", "lp_value", "r",
    "leg", "leg_bars", "d",
    "depth_pct", "depth_atr", "retrace", "position", "dev5",
    "is_shallow", "is_deep",
]
DIMENSION_SCORE_COLS = [f"dim_{d}_score" for d in SCORED_DIMENSIONS]
SCORE_COLS = [f"score_{v}" for v in SCORE_VARIANTS]
DAILY_FEATURES_COLS = (STRUCT_COLS + gates.GATE_COLS + dimensions.FEATURE_IDS
                       + DIMENSION_SCORE_COLS + SCORE_COLS)

SCORABLE_STATES = (pullback.STATE_FORMING, pullback.STATE_BOUNCE, pullback.STATE_BREAK)
MIN_HISTORY_BARS = 60  # 指標計算に必要な最低限（SMA200 等はこれ未満だと自然に NaN になる）

# CSV 往復でブール列が壊れないよう明示的に変換する対象（T-301）
BOOL_COLS = (list(gates.GATE_COLS) + ["is_shallow", "is_deep"]
            + [fid for fid, _dim, direction, _band in dimensions.FEATURE_METADATA
               if direction == "binary"])
DATE_COLS = ["date", "h0_date", "l0_date", "lp_date"]


def compute_daily_features(
    ohlcv: Dict[str, pd.DataFrame], universe_tickers: Iterable[str],
    idx_close: pd.Series, k: int, label_n: int,
    earnings_schedule: Optional[pd.DataFrame] = None,
    history_pool: Optional[pd.DataFrame] = None,
    pool_days: int = composite.POOL_DAYS,
    log=print,
) -> pd.DataFrame:
    """全採点銘柄（gate_pass かつ 状態が形成中/反発開始/ブレイク）の特徴量・状態・地合い・
    プール正規化スコア（DESIGN.md §6、T-301）を1銘柄1行でまとめる。各銘柄は自身の系列の
    最終日（＝当日）だけを評価する。

    universe_tickers は既にユニバース通過済み（G0 の「ユニバース通過」条件）の銘柄一覧
    を渡す想定。地合いゲージ・ブレスは日次で1回だけ計算し、全銘柄で共有する。

    history_pool を渡すと、当日ぶんの行と合わせて直近 pool_days 営業日ぶんのプールを
    組んでスコアを計算する（load_recent_daily_features で読み込んだものを渡す想定）。
    渡さない場合は当日ぶんだけをプールとして計算する。
    """
    universe_tickers = list(universe_tickers)
    breadth_universe = {t: ohlcv[t] for t in universe_tickers
                        if t in ohlcv and ohlcv[t] is not None and len(ohlcv[t])}
    asof = idx_close.index[-1]
    breadth_75, breadth_200, n_counted = regime.compute_breadth(breadth_universe, asof)
    gauge = regime.regime_gauge(idx_close, len(idx_close) - 1, breadth_75, breadth_200)
    log(f"[features] 地合い={gauge['level']}({gauge['score']}/6) "
        f"breadth75={breadth_75:.2f} breadth200={breadth_200:.2f} (n={n_counted})"
        if not np.isnan(breadth_75) and not np.isnan(breadth_200) else
        f"[features] 地合い={gauge['level']}({gauge['score']}/6) breadth 未定義")

    rows = []
    gate_fail_counts = {"g1": 0, "g2": 0, "g3": 0, "earnings_near": 0}
    n_evaluated = n_state_scorable = n_gate_pass = 0
    for ticker in universe_tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_HISTORY_BARS:
            continue
        open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
        volume, dividends = df["Volume"], df["Dividends"]
        t_pos = len(df) - 1
        n_evaluated += 1

        sma5 = indicators.sma(close, 5)
        sma75 = indicators.sma(close, 75)
        sma200 = indicators.sma(close, 200)
        atr14 = indicators.atr_wilder(high, low, close, 14)
        raw = swings.detect_raw_swings(high, low, k)
        alternated = swings.alternate_swings(raw)
        pb = pullback.pullback_state(high, low, close, sma5, sma200, atr14, alternated, t_pos, k)
        gate = gates.evaluate_gates(close, high, sma75, sma200, t_pos, True, label_n,
                                    earnings_schedule=earnings_schedule, ticker=ticker)

        if not gate["g1"]:
            gate_fail_counts["g1"] += 1
        if not gate["g2"]:
            gate_fail_counts["g2"] += 1
        if not gate["g3"]:
            gate_fail_counts["g3"] += 1
        if not gate["g0"] and not gate["g0_earnings_unknown"]:
            gate_fail_counts["earnings_near"] += 1

        if pb["state"] in SCORABLE_STATES:
            n_state_scorable += 1
        if not gate["gate_pass"] or pb["state"] not in SCORABLE_STATES:
            continue
        n_gate_pass += 1

        feats, _extra = dimensions.compute_dimensions(
            open_, high, low, close, volume, dividends, alternated, pb, t_pos, k,
            idx_close=idx_close, earnings_schedule=earnings_schedule, ticker=ticker,
            regime=gauge["level"], breadth_75=breadth_75, breadth_200=breadth_200,
        )

        idx = close.index
        row = {
            "ticker": ticker, "date": idx[t_pos], "state": pb["state"],
            "h0_date": idx[pb["h0"]] if pb["h0"] is not None else pd.NaT,
            "l0_date": idx[pb["l0"]] if pb["l0"] is not None else pd.NaT,
            "lp_date": idx[pb["lp"]] if pb["lp"] is not None else pd.NaT,
            "h0_high": pb["h0_high"], "l0_low": pb["l0_low"], "lp_value": pb["lp_value"],
            "r": pb["r"], "leg": pb["leg"], "leg_bars": pb["leg_bars"], "d": pb["d"],
            "depth_pct": pb["depth_pct"], "depth_atr": pb["depth_atr"], "retrace": pb["retrace"],
            "position": pb["position"], "dev5": pb["dev5"],
            "is_shallow": pb["is_shallow"], "is_deep": pb["is_deep"],
        }
        for col in gates.GATE_COLS:
            row[col] = gate[col]
        for fid, value in zip(feats["id"], feats["value"]):
            row[fid] = value
        for col in DIMENSION_SCORE_COLS + SCORE_COLS:
            row[col] = np.nan  # 下でプール正規化スコアにより上書きする（採点対象が無ければ NaN のまま）
        rows.append(row)

    log(f"[features] 評価 {n_evaluated} 銘柄 / 状態該当 {n_state_scorable} / "
        f"採点対象(状態該当かつ全ゲート通過) {n_gate_pass} / "
        f"ゲート落ちの内訳(評価{n_evaluated}銘柄中、複数ゲートに同時該当しうる): "
        f"G1落ち {gate_fail_counts['g1']}, G2落ち {gate_fail_counts['g2']}, "
        f"G3落ち {gate_fail_counts['g3']}, 決算接近 {gate_fail_counts['earnings_near']}")

    if not rows:
        return pd.DataFrame(columns=DAILY_FEATURES_COLS)
    out = pd.DataFrame(rows)[DAILY_FEATURES_COLS]

    pool_parts = [p for p in (history_pool, out) if p is not None and len(p)]
    full_pool = pd.concat(pool_parts, ignore_index=True) if pool_parts else out
    scored = composite.compute_composite_scores(full_pool, asof, pool_days=pool_days, log=log)
    if len(scored):
        scored = scored.drop_duplicates(subset="ticker", keep="last").set_index("ticker")
        for col in DIMENSION_SCORE_COLS + SCORE_COLS:
            out[col] = out["ticker"].map(scored[col]).to_numpy()

    return out


def load_recent_daily_features(daily_dir: Path, asof: pd.Timestamp, pool_days: int) -> pd.DataFrame:
    """直近 pool_days-1 営業日ぶん（asof より前）の日次特徴量スナップショットを読み込み
    連結する（DESIGN.md §6.1 / T-301）。当日ぶんは compute_daily_features 側で作るので
    ここでは含めない。daily_dir に save_daily_features が書いた features_YYYY-MM-DD.csv.gz
    が並んでいる前提。

    ブール列（gate 列・is_shallow/is_deep・二値特徴量）は CSV 往復で "True"/"False" の
    文字列や、欠損混在時は object dtype になり得るため、明示的に bool/NA へ変換する
    （素朴に bool(value) すると NaN や "False" 文字列が真になる不具合を避けるため）。
    """
    daily_dir = Path(daily_dir)
    asof = pd.Timestamp(asof)
    if pool_days <= 1 or not daily_dir.exists():
        return pd.DataFrame(columns=DAILY_FEATURES_COLS)

    dated: List[tuple] = []
    for f in sorted(daily_dir.glob("features_*.csv.gz")):
        name = f.name
        if not (name.startswith("features_") and name.endswith(".csv.gz")):
            continue
        date_str = name[len("features_"):-len(".csv.gz")]
        try:
            d = pd.Timestamp(date_str)
        except ValueError:
            continue
        if d < asof:
            dated.append((d, f))
    dated.sort(key=lambda x: x[0])
    recent = dated[-(pool_days - 1):]

    frames = []
    for _d, f in recent:
        df = pd.read_csv(f, parse_dates=DATE_COLS)
        for col in BOOL_COLS:
            if col in df.columns:
                df[col] = df[col].map(_coerce_bool).astype("boolean")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=DAILY_FEATURES_COLS)
    return pd.concat(frames, ignore_index=True)


def _coerce_bool(v) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v == "True":
            return True
        if v == "False":
            return False
    return pd.NA


def save_daily_features(df: pd.DataFrame, daily_dir: Path, asof: pd.Timestamp) -> Path:
    """daily/features_YYYY-MM-DD.csv.gz に保存する（T-206）。"""
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = daily_dir / f"features_{pd.Timestamp(asof).strftime('%Y-%m-%d')}.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    return path
