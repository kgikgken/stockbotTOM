"""33 業種の強弱（docs/SCREENER.md §2.9）。

ユニバース通過銘柄の等加重リターンを業種ごとに 5 日・20 日の 2 本で計算し、
5 日リターンの降順に 1 位から順位を付ける。

**この順位は候補の並び順にだけ使う。条件にも除外にも使わない**（§2.9）。19 条件は
ブール判定のままで、候補の集合は業種順位の有無で変わらない（§2.2 の「スコアを作らない」
に抵触しないのはこのため —— 銘柄に点数を付けているのではなく、業種を並べているだけ）。

**未来参照はしない。** リターンは Close[T] / Close[T−look] − 1 で、T の引けまでの
値しか読まない（CLAUDE.md）。

**等加重にする理由**: 時価総額も浮動株数も取得していない。加重の根拠が無いので、
根拠のある加重を装うより等加重で揃える。

**構成銘柄はユニバース通過分のみ**（§2.9 の測定結果を参照）。全上場に広げると
流動性の低い銘柄の終値が飛び、別種のノイズが入る。候補はそもそも通過集合から出るので、
関係する母集団はこちらである。
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

# 順位付けに使う窓と、併記する窓。docs/SCREENER.md §2.9 で指定された値であり、
# 探索して決めたものではない。増やすときは §7 の質問ログを通す
RANK_LOOKBACK = 5      # この窓のリターンで 1〜N 位を付ける
LONG_LOOKBACK = 20     # 併記するだけ。順位付けには使わない

SECTOR_COLS = ["sector33", "n", "ret_5d", "ret_20d", "rank_5d", "rank_20d"]

UNCLASSIFIED = "（未分類）"


def _window_return(close: pd.Series, t_pos: int, look: int) -> float:
    """Close[T] / Close[T−look] − 1。読むのは T までで、T より後は見ない。"""
    if t_pos < look or t_pos >= len(close):
        return np.nan
    a = float(close.iloc[t_pos - look])
    b = float(close.iloc[t_pos])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return np.nan
    return b / a - 1.0


def sector_strength(ohlcv: Mapping[str, pd.DataFrame], tickers: Iterable[str],
                    sector_by_ticker: Mapping[str, str], asof,
                    rank_lookback: int = RANK_LOOKBACK,
                    long_lookback: int = LONG_LOOKBACK) -> pd.DataFrame:
    """業種ごとの等加重リターンと順位を返す（docs/SCREENER.md §2.9）。

    ohlcv は銘柄ごとの OHLCV（DatetimeIndex・昇順）。tickers はユニバース通過銘柄。
    asof は判定日 T —— 各銘柄でこの日の位置を探し、そこまでの値だけを使う。
    T の足が無い銘柄はその業種の計算から外す（前日の値で代用しない）。

    戻り値は SECTOR_COLS の列を持ち、`rank_5d` の昇順（1 位が先頭）。
    `n` は実際にリターンを計算できた銘柄数で、**小さい業種ほど順位が振れる**
    （§2.9 の測定）。読み手が割り引けるように必ず一緒に返す。
    """
    asof = pd.Timestamp(asof).normalize()
    rows: Dict[str, list] = {}
    for ticker in tickers:
        df = ohlcv.get(str(ticker))
        if df is None or len(df) == 0:
            continue
        idx = pd.DatetimeIndex(df.index).normalize()
        pos_arr = np.flatnonzero(idx == asof)
        if pos_arr.size != 1:
            continue   # T の足が無い / 重複している。代用しない
        close = df["Close"]
        short = _window_return(close, int(pos_arr[0]), rank_lookback)
        long_ = _window_return(close, int(pos_arr[0]), long_lookback)
        if not np.isfinite(short):
            continue   # 順位付けの窓が取れない銘柄は業種の計算に入れない
        sector = str(sector_by_ticker.get(str(ticker), "") or "") or UNCLASSIFIED
        rows.setdefault(sector, []).append((short, long_))

    if not rows:
        return pd.DataFrame(columns=SECTOR_COLS)

    records = []
    for sector, values in rows.items():
        shorts = np.array([v[0] for v in values], dtype=float)
        longs = np.array([v[1] for v in values], dtype=float)
        records.append({
            "sector33": sector,
            "n": int(len(shorts)),
            "ret_5d": float(np.mean(shorts)),
            # 20 日は履歴が足りない銘柄が混じるので、取れた分だけの平均にする
            "ret_20d": (float(np.nanmean(longs)) if np.isfinite(longs).any() else np.nan),
        })
    out = pd.DataFrame(records)
    # 同率は業種名の昇順で固定する（実行のたびに並びが変わらないようにするため）
    out = out.sort_values(["ret_5d", "sector33"], ascending=[False, True])
    out["rank_5d"] = np.arange(1, len(out) + 1)
    long_order = out.sort_values(["ret_20d", "sector33"], ascending=[False, True])
    out["rank_20d"] = pd.Series(np.arange(1, len(long_order) + 1),
                                index=long_order.index).astype("Int64")
    out.loc[out["ret_20d"].isna(), "rank_20d"] = pd.NA
    return out[SECTOR_COLS].reset_index(drop=True)


def rank_lookup(strength: Optional[pd.DataFrame]) -> Dict[str, dict]:
    """業種名 → {rank_5d, rank_20d, ret_5d, ret_20d, n} の辞書。

    配信記録とカードに載せる値をここから引く。順位表に無い業種は呼び出し側で欠損にする。
    """
    if strength is None or len(strength) == 0:
        return {}
    out: Dict[str, dict] = {}
    for _i, row in strength.iterrows():
        out[str(row["sector33"])] = {
            "rank_5d": int(row["rank_5d"]),
            "rank_20d": (None if pd.isna(row["rank_20d"]) else int(row["rank_20d"])),
            "ret_5d": float(row["ret_5d"]),
            "ret_20d": (None if pd.isna(row["ret_20d"]) else float(row["ret_20d"])),
            "n": int(row["n"]),
        }
    return out


def ranking_table(strength: Optional[pd.DataFrame]) -> list[dict]:
    """要約に残す順位表（docs/SCREENER.md §3.6）。5 日順位の昇順。"""
    if strength is None or len(strength) == 0:
        return []
    rows = []
    for _i, row in strength.iterrows():
        rows.append({
            "sector33": str(row["sector33"]),
            "n": int(row["n"]),
            "rank_5d": int(row["rank_5d"]),
            "rank_20d": (None if pd.isna(row["rank_20d"]) else int(row["rank_20d"])),
            "ret_5d": round(float(row["ret_5d"]), 6),
            "ret_20d": (None if pd.isna(row["ret_20d"]) else round(float(row["ret_20d"]), 6)),
        })
    return rows
