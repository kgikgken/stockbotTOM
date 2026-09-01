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

CORE_COLS = [
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

# スクリーナー側が渡す列（docs/SCREENER.md §3.2）。build_record の extra で受け取る。
# 判定に使った値ではなく、あとから「なぜこの並びか」「どの注記が要るか」を引くためのもの
EXTRA_DEFAULTS: dict = {
    "adv_jpy": np.nan,             # 20日平均売買代金（配信の並び順の根拠）
    "sector33": "",                # 33業種（同一業種3件までの根拠）
    "a4_earnings_unknown": False,  # True ならカードに「決算日未取得」と出す（§2.6）
    "e1_skipped": False,           # True ならその日は母集団不足で E1 を適用していない（§2.5）
    "earnings_days": np.nan,       # 決算発表までの営業日数（取れなければ NaN）。カードに出す
    "streak": 1,                   # 連続点灯日数。今日が初日なら 1（§3.2）
    "prev_delivered_on": pd.NaT,   # 前回この銘柄が候補になった配信日（無ければ空）
}
EXTRA_COLS = list(EXTRA_DEFAULTS)

DELIVERED_COLS = CORE_COLS + EXTRA_COLS

DATE_COLS = ["delivered_on", "asof", "lp_date", "h0_date", "prev_delivered_on"]


def build_record(ticker: str, high: pd.Series, low: pd.Series, close: pd.Series,
                 pullback_result: dict, t_pos: int, delivered_on,
                 name: str = "", extra: Optional[dict] = None) -> dict:
    """1 銘柄ぶんの配信記録を作る（docs/SCREENER.md §3.2）。

    high/low/close は 0 始まりの位置で扱う整列済み pandas.Series（features 内の
    他モジュールと同じ規約）。pullback_result は features.pullback.pullback_state()
    の戻り値をそのまま渡す。t_pos は判定日 T の位置。extra は EXTRA_DEFAULTS の
    キーだけを受け取る（未知のキーは取り違えなので例外にする）。

    T より後の行が系列に付いていても結果は変わらない（[t_pos] までしか読まない）。
    """
    extra = dict(extra or {})
    unknown = set(extra) - set(EXTRA_DEFAULTS)
    if unknown:
        raise ValueError(f"配信記録に無い列: {sorted(unknown)}")
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
        **{key: extra.get(key, default) for key, default in EXTRA_DEFAULTS.items()},
    }


def records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """build_record の出力を DELIVERED_COLS の順に並べた DataFrame にする。

    EXTRA_COLS（スクリーナー側が渡す列）が無い行は既定値で埋める。CORE_COLS は
    埋めない —— 押し安値や止まった線が欠けている記録は作りたくないので、
    足りなければ KeyError にする。
    """
    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=DELIVERED_COLS)
    df = pd.DataFrame(rows)
    for col, default in EXTRA_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df[DELIVERED_COLS]


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
                                  "state": str, "sector33": str},
                     keep_default_na=False, na_values=[""])
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("ticker", "name", "landing_ma", "state", "sector33"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    for col in ("a4_earnings_unknown", "e1_skipped"):
        if col in df.columns:
            df[col] = df[col].map(_coerce_bool).astype("boolean")
    return df


def _coerce_bool(v) -> object:
    """CSV 往復でブール列が "True"/"False" の文字列になるため明示的に戻す。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v in ("True", "true"):
            return True
        if v in ("False", "false"):
            return False
    return pd.NA


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


MAX_STREAK_LOOKBACK = 60  # 連続点灯を遡る上限（C3 で押し目は 12 日までなのでこれで足りる）


def lookback_stats(daily_dir: Path, tickers, delivered_on,
                   max_files: int = MAX_STREAK_LOOKBACK) -> tuple[dict, dict]:
    """過去の配信記録から、銘柄ごとの連続点灯日数と前回点灯日を求める（docs/SCREENER.md §3.2）。

    - streak: 今日を 1 日目として、直前の配信日から連続で候補に出ている日数
    - prev_delivered_on: 今日より前で最後に候補になった配信日（無ければ None）

    delivered_on より前の配信記録だけを見る（今日の分は数えない）。

    **配信記録のファイルが欠けている日（ワークフローが失敗した日など）は「その日は
    出なかった」と同じ扱いになるため、streak は実際より短くなりうる。** 記録した時点で
    分かることをそのまま書く方針で、あとから遡って直さない。同じ押し目かどうかを厳密に
    見たい場合は `h0_date`（押し目の起点）が一致するかで判定する。
    """
    tickers = list(tickers)
    streak = {t: 1 for t in tickers}
    prev_seen: dict = {t: None for t in tickers}
    if not tickers:
        return streak, prev_seen

    delivered_on = pd.Timestamp(delivered_on).normalize()
    past = [(d, f) for d, f in list_delivered(daily_dir) if d < delivered_on]
    past.sort(key=lambda x: x[0], reverse=True)

    unbroken = set(tickers)   # まだ連続が途切れていない銘柄
    for d, f in past[:max_files]:
        try:
            present = set(load_delivered(f)["ticker"].astype(str))
        except Exception:
            present = set()   # 読めないファイルは「出なかった」扱い（例外にしない）
        for t in tickers:
            if prev_seen[t] is None and t in present:
                prev_seen[t] = d
        for t in list(unbroken):
            if t in present:
                streak[t] += 1
            else:
                unbroken.discard(t)
        if not unbroken and all(v is not None for v in prev_seen.values()):
            break
    return streak, prev_seen


def latest_delivered(daily_dir: Path) -> Optional[pd.DataFrame]:
    """最新の配信記録（無ければ None）。連続点灯日数などの表示に使う。"""
    files = list_delivered(daily_dir)
    if not files:
        return None
    return load_delivered(files[-1][1])
