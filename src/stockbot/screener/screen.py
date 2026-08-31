"""日次のスクリーニング（docs/SCREENER.md §2.4・§2.5）。

1. ユニバース通過銘柄について、E1 以外の 18 条件を銘柄ごとに判定する（conditions.py）
2. 18 条件を全て満たした集合（＝ E1 の母集団）に対して E1 を付ける
3. 通過した銘柄を売買代金の降順に並べ、同一 33 業種は 3 件までに絞る

順位は付けない。並び順は売買代金であって優劣ではない（§2.5）。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ..features import pullback, swings
from ..features.indicators import atr_wilder, sma
from .conditions import (
    CONDITION_IDS,
    DIAGNOSTIC_COLS,
    SELF_CONTAINED_IDS,
    evaluate_conditions,
    passes_self_contained,
)

E1_TOP_PCTL = 0.90      # 上位 10% を落とす
E1_MIN_POOL = 10        # 母集団がこれ未満の日は E1 をスキップする
SECTOR_CAP = 3          # 同一 33 業種の上限
MIN_HISTORY_BARS = 60   # 指標計算に必要な最低限（これ未満は評価しない）

SCREEN_COLS = ["ticker", "date"] + CONDITION_IDS + ["passes"] + DIAGNOSTIC_COLS + ["state"]


def evaluate_universe(
    ohlcv: Dict[str, pd.DataFrame], tickers: Iterable[str], idx_close: pd.Series,
    k: int, earnings_schedule: Optional[pd.DataFrame] = None, log=print,
) -> pd.DataFrame:
    """ユニバース通過銘柄それぞれの最終足（＝ T）について 18 条件を判定する。

    各銘柄は自身の系列の最終日だけを評価する（pipeline.compute_daily_features と同じ規約）。
    idx_close は指数終値。銘柄ごとに日付で整列し直し、欠損は最大 3 営業日まで過去方向に
    のみ前方補完する（pipeline.py と同じ扱い。未来は一切参照しない）。
    """
    rows = []
    n_evaluated = 0
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_HISTORY_BARS:
            continue
        n_evaluated += 1
        high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
        t_pos = len(df) - 1
        alt = swings.alternate_swings(swings.detect_raw_swings(high, low, k))
        pb = pullback.pullback_state(high, low, close, sma(close, 5), sma(close, 200),
                                     atr_wilder(high, low, close, 14), alt, t_pos, k)
        row = evaluate_conditions(
            ticker, high, low, close, volume, pb, t_pos,
            idx_close=idx_close.reindex(close.index).ffill(limit=3),
            earnings_schedule=earnings_schedule,
        )
        row["date"] = close.index[t_pos]
        row["state"] = pb["state"]
        rows.append(row)

    if not rows:
        log(f"[screen] 評価 {n_evaluated} 銘柄 / 条件判定 0 件")
        return pd.DataFrame(columns=SCREEN_COLS)
    out = pd.DataFrame(rows)
    out["passes"] = False
    out = out[SCREEN_COLS]
    for cid in SELF_CONTAINED_IDS:
        out[cid] = out[cid].astype("boolean")
    out["E1"] = pd.array([pd.NA] * len(out), dtype="boolean")
    n_pool = int(out.apply(passes_self_contained, axis=1).sum())
    # 候補が 0 件の日に「どの条件で落ちたか」が分からないと運用で困る。
    # pipeline.py の「ゲート落ちの内訳」と同じ趣旨（複数条件に同時該当しうる）
    breakdown = " ".join(f"{cid}:{int((~out[cid].fillna(False)).sum())}"
                         for cid in SELF_CONTAINED_IDS)
    log(f"[screen] 評価 {n_evaluated} 銘柄 / 18条件通過 {n_pool} 件 / "
        f"条件別の不成立件数: {breakdown}")
    return out


def apply_e1(df: pd.DataFrame, top_pctl: float = E1_TOP_PCTL,
             min_pool: int = E1_MIN_POOL, log=print) -> tuple[pd.DataFrame, dict]:
    """E1（当日候補内で d2_rs60 の上位 10% を落とす）を付ける（docs/SCREENER.md §2.4）。

    母集団は A〜D・E2・E3 を全て満たした集合。母集団が min_pool 件未満の日は E1 を
    スキップし（全て成立扱い）、`e1_skipped` を立てて記録に残す。

    戻り値: (E1 と passes を埋めた df, {"e1_pool_n", "e1_skipped", "e1_threshold"})
    """
    out = df.copy()
    if len(out) == 0:
        return out, {"e1_pool_n": 0, "e1_skipped": True, "e1_threshold": np.nan}

    pool_mask = out.apply(passes_self_contained, axis=1).to_numpy(dtype=bool)
    pool_n = int(pool_mask.sum())
    skipped = pool_n < min_pool

    e1 = pd.array([pd.NA] * len(out), dtype="boolean")
    threshold = np.nan
    if skipped:
        e1[pool_mask] = True
    else:
        rs = out.loc[pool_mask, "rs60"]
        threshold = float(rs.quantile(top_pctl))
        # 欠損は不成立（§2.3）。rs60 が取れない銘柄は E1 を通さない
        e1[pool_mask] = (rs.notna() & (rs <= threshold)).to_numpy(dtype=bool)
    out["E1"] = e1
    out["passes"] = pool_mask & (out["E1"].fillna(False).to_numpy(dtype=bool))

    meta = {"e1_pool_n": pool_n, "e1_skipped": bool(skipped), "e1_threshold": threshold}
    if skipped:
        log(f"[screen] E1: 母集団 {pool_n} 件 (< {min_pool}) のためスキップ")
    else:
        log(f"[screen] E1: 母集団 {pool_n} 件 / rs60 の上位{int((1 - top_pctl) * 100)}%"
            f"（> {threshold:.4f}）を除外 → 通過 {int(out['passes'].sum())} 件")
    return out, meta


def select_candidates(df: pd.DataFrame, sector_by_ticker: Optional[Dict[str, str]] = None,
                      sector_cap: int = SECTOR_CAP) -> pd.DataFrame:
    """通過銘柄を売買代金の降順に並べ、同一 33 業種を sector_cap 件までに絞る。

    並び順は売買代金であって優劣ではない（順位を付けない、docs/SCREENER.md §2.5）。
    売買代金が同じ場合は銘柄コード順（実行のたびに並びが変わらないようにするため）。
    業種が取れない銘柄は上限の対象外（互いに別業種として扱う）。
    """
    if len(df) == 0:
        out = df.copy()
        out["sector33"] = pd.Series(dtype=str)
        return out
    passed = df[df["passes"].astype(bool)].copy()
    sector_by_ticker = sector_by_ticker or {}
    passed["sector33"] = passed["ticker"].map(lambda t: str(sector_by_ticker.get(t, "") or ""))
    passed = passed.sort_values(["adv_jpy", "ticker"], ascending=[False, True])

    counts: Dict[str, int] = {}
    keep = []
    for _i, row in passed.iterrows():
        sector = row["sector33"]
        if sector:
            if counts.get(sector, 0) >= sector_cap:
                continue
            counts[sector] = counts.get(sector, 0) + 1
        keep.append(row)
    if not keep:
        return passed.iloc[0:0]
    return pd.DataFrame(keep).reset_index(drop=True)
