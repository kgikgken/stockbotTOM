"""momentum系のバックテスト用ラッパー。本番のmomentum/classify.py・indicators.pyの関数を
そのまま呼び出す(バックテスト専用の判定ロジックを別途書かない)。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from momentum.indicators import compute_momentum_features, compute_pool_scores, chandelier_exit_long, atr_wilder
from momentum.classify import classify_state, build_candidate
from .engine import Trade


def momentum_signal_fn(ohlcv_full: Dict[str, pd.DataFrame], t_idx: int,
                       trading_dates: pd.DatetimeIndex, cfg,
                       bench_df_full: pd.DataFrame | None = None) -> List[dict]:
    """T日時点(ohlcv_full各銘柄をt_idxまでにスライスした状態)で状態A/Bが新規に点灯した銘柄を返す。
    本番のcompute_momentum_features・compute_pool_scores・classify_state・build_candidateを
    そのまま使う(ロジックの二重実装を避ける)。

    ★bench_df_full(TOPIX代理の全期間データ)を渡すこと。渡さないと対TOPIX相対強度が全銘柄で
    NaNになり、それに依存するセクター強度(compute_sector_strengthはrel_strengthを使う)も
    連鎖的に無効化され、本番の5要素スコアのうち2つが欠けた簡易版になってしまう(重大な精度劣化)。
    """
    today = trading_dates[t_idx]
    bench_logclose = None
    if bench_df_full is not None and today in bench_df_full.index:
        bpos = bench_df_full.index.get_loc(today)
        bench_slice = bench_df_full.iloc[:bpos + 1]
        if len(bench_slice) >= 130:
            bench_logclose = np.log(bench_slice["Close"])

    items = []
    for tkr, df_full in ohlcv_full.items():
        if today not in df_full.index:
            continue
        pos = df_full.index.get_loc(today)
        df = df_full.iloc[:pos + 1]  # ★T日までのみ(ルックアヘッド防止)
        if len(df) < 260:
            continue
        feat = compute_momentum_features(df, bench_logclose, cfg)
        if feat is None:
            continue
        if feat["close"] < cfg.min_price or feat["adv20_jpy"] < cfg.min_adv_jpy:
            continue
        row = {"ticker": tkr, "sector": df_full.attrs.get("sector", "不明"),
              "name": df_full.attrs.get("name", tkr)}
        items.append({"row": row, "feat": feat})

    if not items:
        return []
    items = compute_pool_scores(items, cfg)
    items.sort(key=lambda x: -x["score"])
    pool = items[: cfg.pool_size]

    signals = []
    for item in pool:
        state = classify_state(item["feat"], cfg)
        if state not in ("A", "B"):
            continue
        c = build_candidate(item["row"], item["feat"], state, item["score"], cfg)
        if c is None:
            continue
        signals.append({"ticker": c.ticker, "engine": state, "entry": c.entry, "stop": c.stop,
                        "target": None, "sector": c.sector, "name": c.name})
    return signals


def momentum_exit_fn(trade: Trade, df_upto: pd.DataFrame, t_idx: int) -> Optional[tuple]:
    """保有中ポジションの手仕舞い判定(本番のシャンデリア・エグジット + 初期ストップをそのまま使用)。
    固定利確なし・シャンデリア水準を割ったら手仕舞い、というmomentum系の設計をそのまま再現する。"""
    if len(df_upto) < 25:
        return None
    h, l, c = df_upto["High"], df_upto["Low"], df_upto["Close"]
    today_low = float(l.iloc[-1])
    today_close = float(c.iloc[-1])

    # 初期ストップ(トレード時に固定した水準)を日中安値が下回ったら手仕舞い
    if today_low <= trade.stop:
        return trade.stop, "stop"

    # シャンデリア水準(直近22日高値-3×ATR)を終値が下回ったら手仕舞い
    atr_n = atr_wilder(h, l, c, 22)
    chand = chandelier_exit_long(h, atr_n, 22, 3.0)
    if len(chand) and not pd.isna(chand.iloc[-1]) and today_close < float(chand.iloc[-1]):
        return today_close, "chandelier"
    return None
