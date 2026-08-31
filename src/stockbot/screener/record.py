"""配信記録（docs/SCREENER.md §3.2）。

配信した銘柄を 1 行 1 銘柄で `daily/delivered_YYYY-MM-DD.csv` に保存する。
ファイル名の日付は配信日（LINE に流した日）で、判定日 T は列 `asof` に持つ。

**このモジュールは T の引けまでのデータしか参照しない。** 結果（T+1 以降）を付ける
のは screener/resolver.py の役割で、書き込むファイルも分ける。配信記録を後から
書き換えないので、その日に何を出したかの記録は結果付けの成否に影響されない。

「止まった線」は features.dimensions.landing_ma()（d3_ma_dist と同じ判定）で決める。
記録には線名だけでなく距離（ATR 単位）も入れる。最も近い線が遠い（例えば 3 ATR 先）
場合でも線名は付くので、線ごとの集計をするときは距離で絞れるようにしておく。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..features.dimensions import landing_ma, ma_values_at
from ..features.indicators import atr_wilder

DELIVERED_PREFIX = "delivered_"
DELIVERED_SUFFIX = ".csv"

DELIVERED_COLS = [
    "delivered_on",       # 配信日（LINE に流した日）
    "asof",               # 判定日 T（この日の引けまでの情報だけで決めた）
    "ticker",
    "name",               # 銘柄名（取れなければ空）
    "landing_ma",         # 止まった線（SMA5/SMA25/SMA75/SMA200。無ければ空）
    "landing_ma_value",   # 押し安値の日のその線の値
    "landing_dist_atr",   # |Lp − 線| / ATR[T]
    "lp",                 # 押し安値 Lp
    "lp_date",
    "h0_high",            # 直近高値 H0.high
    "h0_date",
    "close_t",            # T の終値
    "atr_t",              # ATR14[T]
    "state",              # 押し目状態（形成中/反発開始/ブレイク）
    "depth_pct",          # 押しの深さ
    "pullback_days",      # 押し目日数 d
]

DATE_COLS = ["delivered_on", "asof", "lp_date", "h0_date"]


def build_record(ticker: str, high: pd.Series, low: pd.Series, close: pd.Series,
                 pullback_result: dict, t_pos: int, delivered_on,
                 name: str = "") -> dict:
    """1 銘柄ぶんの配信記録を作る（docs/SCREENER.md §3.2）。

    high/low/close は 0 始まりの位置で扱う整列済み pandas.Series（features 内の
    他モジュールと同じ規約）。pullback_result は features.pullback.pullback_state()
    の戻り値をそのまま渡す。t_pos は判定日 T の位置。

    T より後の行が系列に付いていても結果は変わらない（[t_pos] までしか読まない）。
    """
    idx = close.index
    atr14 = atr_wilder(high, low, close, 14)
    atr_t = float(atr14.iloc[t_pos])

    lp = pullback_result.get("lp")
    lp_value = float(pullback_result.get("lp_value", np.nan))
    landing = {"landing_ma": None, "landing_ma_value": np.nan, "dist_atr": np.nan}
    if lp is not None:
        landing = landing_ma(lp_value, ma_values_at(close, int(lp)), atr_t)

    h0 = pullback_result.get("h0")
    return {
        "delivered_on": pd.Timestamp(delivered_on).normalize(),
        "asof": pd.Timestamp(idx[t_pos]).normalize(),
        "ticker": ticker,
        "name": name,
        "landing_ma": landing["landing_ma"] or "",
        "landing_ma_value": landing["landing_ma_value"],
        "landing_dist_atr": landing["dist_atr"],
        "lp": lp_value,
        "lp_date": pd.Timestamp(idx[lp]).normalize() if lp is not None else pd.NaT,
        "h0_high": float(pullback_result.get("h0_high", np.nan)),
        "h0_date": pd.Timestamp(idx[h0]).normalize() if h0 is not None else pd.NaT,
        "close_t": float(close.iloc[t_pos]),
        "atr_t": atr_t,
        "state": pullback_result.get("state", ""),
        "depth_pct": float(pullback_result.get("depth_pct", np.nan)),
        "pullback_days": pullback_result.get("d"),
    }


def records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """build_record の出力を DELIVERED_COLS の順に並べた DataFrame にする。"""
    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=DELIVERED_COLS)
    return pd.DataFrame(rows)[DELIVERED_COLS]


def delivered_path(daily_dir: Path, delivered_on) -> Path:
    d = pd.Timestamp(delivered_on).strftime("%Y-%m-%d")
    return Path(daily_dir) / f"{DELIVERED_PREFIX}{d}{DELIVERED_SUFFIX}"


def save_delivered(df: pd.DataFrame, daily_dir: Path, delivered_on) -> Path:
    """daily/delivered_YYYY-MM-DD.csv に保存する（既存ファイルは上書きしない）。

    同じ日に 2 回実行しても最初の記録が正なので、既にあるファイルはそのまま残す
    （配信記録は「その日に何を出したか」の台帳であり、後から作り直さない）。
    件数が少ないので圧縮しない（人が git の差分で読めるようにする）。
    """
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = delivered_path(daily_dir, delivered_on)
    if path.exists():
        return path
    out = df.copy()
    for col in DATE_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def load_delivered(path: Path) -> pd.DataFrame:
    """配信記録を読む。日付列は Timestamp、landing_ma などの文字列列は空文字を保つ。"""
    df = pd.read_csv(path, dtype={"ticker": str, "name": str, "landing_ma": str,
                                  "state": str}, keep_default_na=False, na_values=[""])
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("ticker", "name", "landing_ma", "state"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def list_delivered(daily_dir: Path) -> list[tuple[pd.Timestamp, Path]]:
    """daily/ にある配信記録を (配信日, パス) の昇順で返す。"""
    daily_dir = Path(daily_dir)
    if not daily_dir.exists():
        return []
    out: list[tuple[pd.Timestamp, Path]] = []
    for f in sorted(daily_dir.glob(f"{DELIVERED_PREFIX}*{DELIVERED_SUFFIX}")):
        stem = f.name[len(DELIVERED_PREFIX):-len(DELIVERED_SUFFIX)]
        try:
            out.append((pd.Timestamp(stem), f))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0])
    return out


def latest_delivered(daily_dir: Path) -> Optional[pd.DataFrame]:
    """最新の配信記録（無ければ None）。連続点灯日数などの表示に使う。"""
    files = list_delivered(daily_dir)
    if not files:
        return None
    return load_delivered(files[-1][1])
