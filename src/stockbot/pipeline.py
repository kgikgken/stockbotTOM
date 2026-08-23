"""日次パイプライン: 指標→スイング→押し目→特徴量→地合いを通し、日次スナップショットに
保存する（DESIGN.md §1 / TASKS.md T-206）。

DESIGN.md §1 の手順のうち、以下は未実装のため、このモジュールでは扱わない:
- ゲート G0〜G3（§2）… 未着手（TASKS.md に専用タスクが無い）。暫定として、ゲートを
  通さず「状態が形成中/反発開始/ブレイクの銘柄」を全て「採点銘柄」として扱う
- 正規化と合成・次元スコア・総合スコア（§6、T-301/T-302）… 未実装

受け入れ条件（T-206: 列名が安定し、後日 resolver（T-504）が読める）を満たすため、
次元スコア（dim_D1_score..dim_D7_score）・総合スコア（score_v1/v2/v3）の列はスキーマ
上ここで確保し、値は NaN のまま保存する。T-301/T-302 実装後は、この関数の中で値を
埋めるだけで済み、列名・列順は変えない。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .features import dimensions, indicators, pullback, regime, swings

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
DAILY_FEATURES_COLS = STRUCT_COLS + dimensions.FEATURE_IDS + DIMENSION_SCORE_COLS + SCORE_COLS

SCORABLE_STATES = (pullback.STATE_FORMING, pullback.STATE_BOUNCE, pullback.STATE_BREAK)
MIN_HISTORY_BARS = 60  # 指標計算に必要な最低限（SMA200 等はこれ未満だと自然に NaN になる）


def compute_daily_features(
    ohlcv: Dict[str, pd.DataFrame], universe_tickers: Iterable[str],
    idx_close: pd.Series, k: int,
    earnings_schedule: Optional[pd.DataFrame] = None,
    log=print,
) -> pd.DataFrame:
    """全採点銘柄（状態が形成中/反発開始/ブレイク）の特徴量・状態・地合いを1銘柄1行で
    まとめる。各銘柄は自身の系列の最終日（＝当日）だけを評価する。

    地合いゲージ・ブレスは日次で1回だけ計算し、全銘柄で共有する。
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
    for ticker in universe_tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_HISTORY_BARS:
            continue
        open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
        volume, dividends = df["Volume"], df["Dividends"]
        t_pos = len(df) - 1

        sma5 = indicators.sma(close, 5)
        sma200 = indicators.sma(close, 200)
        atr14 = indicators.atr_wilder(high, low, close, 14)
        raw = swings.detect_raw_swings(high, low, k)
        alternated = swings.alternate_swings(raw)
        pb = pullback.pullback_state(high, low, close, sma5, sma200, atr14, alternated, t_pos, k)

        if pb["state"] not in SCORABLE_STATES:
            continue

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
        for fid, value in zip(feats["id"], feats["value"]):
            row[fid] = value
        for col in DIMENSION_SCORE_COLS + SCORE_COLS:
            row[col] = np.nan  # T-301/T-302 実装後にここを埋める
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=DAILY_FEATURES_COLS)
    out = pd.DataFrame(rows)
    return out[DAILY_FEATURES_COLS]


def save_daily_features(df: pd.DataFrame, daily_dir: Path, asof: pd.Timestamp) -> Path:
    """daily/features_YYYY-MM-DD.csv.gz に保存する（T-206）。"""
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = daily_dir / f"features_{pd.Timestamp(asof).strftime('%Y-%m-%d')}.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    return path
