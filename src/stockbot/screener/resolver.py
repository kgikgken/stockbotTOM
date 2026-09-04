"""結果付け（docs/SCREENER.md §3.3）。

配信記録（screener/record.py）に、配信の 5 営業日後の結果を付けて
`daily/outcome_<配信日>_asof<判定日>.csv` に保存する。**配信記録と同じ命名規則**なので、
同じ配信日に判定が 2 つある日でも結果ファイルが 1 対 1 で対応する。

**CLAUDE.md の未来参照禁止について**: このモジュールと validation/labels.py だけが
T+1 以降のデータを見てよい。ここで見るのは「配信したあとに実際に何が起きたか」で
あって、判定に使う量ではない。逆にこのモジュールは T 以前の値を計算し直さない
（押し安値・直近高値・止まった線は配信記録に書いてある値をそのまま使う）。
これで「記録した時点の判断」と「その後の結果」が混ざらない。

評価窓は T+1..T+5（固定 5 本）。5 本目が市場（営業日軸）に出るまでは結果を付けない。
出たあとで銘柄側の足が 5 本に満たない場合（売買停止・上場廃止）は、取れた分だけで
確定させて `censored` を立てる（生存バイアスを避けるため、記録ごと落とさない）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.store import IDX_TICKER
from ..features.dimensions import LANDING_MA_NAMES
from ..features.indicators import sma
from .record import (
    DELIVERED_COLS,
    as_calendar_date,
    list_delivered,
    load_delivered,
    stamped_name,
)

HORIZON_DAYS = 5  # docs/SCREENER.md §3.3。運用の記録単位であり、探索するパラメータではない

OUTCOME_PREFIX = "outcome_"
OUTCOME_SUFFIX = ".csv"

OUTCOME_COLS = [
    "delivered_on",         # 配信記録と同じキー
    "asof",
    "ticker",
    "resolved_on",          # 結果を付けた日
    "horizon_days",         # 評価窓の本数（既定 5）
    "n_bars",               # 実際に使えた本数（< horizon なら打ち切り）
    "censored",             # n_bars < horizon_days
    "entry_open",           # Open[T+1]（翌営業日の寄付き。SPEC §1 のエントリー基準）
    "close_h",              # Close[T+5]（打ち切り時は取れた最後の終値）
    "max_high",             # max(High[T+1..T+5])
    "min_low",              # min(Low[T+1..T+5])
    "ret_h",                # close_h / entry_open − 1
    "broke_lp",             # 押し安値 Lp を割ったか（Low < Lp）
    "broke_lp_day",         # 割った最初の日（T からの本数。割らなければ空）
    "reached_h0",           # 直近高値 H0.high を更新したか（High > H0.high）
    "reached_h0_day",
    "recovered_sma5",       # 5 日線を回復したか（Close > SMA5 の日があるか）
    "recovered_sma5_day",
    "success",              # 押し安値を割る前に直近高値を更新したか（SPEC §1 成功の定義）
]

DATE_COLS = ["delivered_on", "asof", "resolved_on"]


# ------------------------------------------------------------------ 営業日軸
def trading_calendar(ohlcv: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """市場の営業日軸。指数（store の IDX_TICKER）があればそれを使い、無ければ
    全銘柄の日付の和集合を使う（データ駆動の非取引日判定。祝日表を持たない）。
    """
    idx_df = ohlcv.get(IDX_TICKER)
    if idx_df is not None and len(idx_df):
        return pd.DatetimeIndex(idx_df.index).normalize().sort_values()
    dates: set = set()
    for ticker, df in ohlcv.items():
        if ticker == IDX_TICKER or df is None or len(df) == 0:
            continue
        dates.update(pd.DatetimeIndex(df.index).normalize())
    return pd.DatetimeIndex(sorted(dates))


def due_date(calendar: pd.DatetimeIndex, asof, horizon: int = HORIZON_DAYS) -> Optional[pd.Timestamp]:
    """asof の horizon 営業日後（市場の営業日軸で数える）。まだ来ていなければ None。"""
    if calendar is None or len(calendar) == 0:
        return None
    asof = pd.Timestamp(asof).normalize()
    pos = calendar.searchsorted(asof, side="left")
    if pos >= len(calendar) or calendar[pos] != asof:
        return None  # 判定日そのものが軸に無い（データ未取得）。まだ数えられない
    target = pos + horizon
    if target >= len(calendar):
        return None
    return calendar[target]


# ------------------------------------------------------------------ 1 銘柄ぶん
def _first_day(mask: np.ndarray) -> Optional[int]:
    """T+1 を 1 日目としたときに、条件が最初に成立した日（成立しなければ None）。"""
    hits = np.flatnonzero(mask)
    return int(hits[0]) + 1 if hits.size else None


def resolve_row(row: pd.Series, df: Optional[pd.DataFrame],
                horizon: int = HORIZON_DAYS, resolved_on=None) -> dict:
    """配信記録 1 行に結果を付ける（docs/SCREENER.md §3.3）。

    df は当該銘柄の OHLCV（DatetimeIndex・昇順）。無い / 判定日が入っていない場合は
    n_bars=0・censored=True で全ての結果を欠損にする（例外にしない）。
    """
    out = {
        "delivered_on": pd.Timestamp(row["delivered_on"]).normalize(),
        "asof": pd.Timestamp(row["asof"]).normalize(),
        "ticker": str(row["ticker"]),
        "resolved_on": pd.Timestamp(resolved_on).normalize() if resolved_on is not None else pd.NaT,
        "horizon_days": int(horizon),
        "n_bars": 0,
        "censored": True,
        "entry_open": np.nan, "close_h": np.nan, "max_high": np.nan, "min_low": np.nan,
        "ret_h": np.nan,
        "broke_lp": pd.NA, "broke_lp_day": pd.NA,
        "reached_h0": pd.NA, "reached_h0_day": pd.NA,
        "recovered_sma5": pd.NA, "recovered_sma5_day": pd.NA,
        "success": pd.NA,
    }
    if df is None or len(df) == 0:
        return out

    idx = pd.DatetimeIndex(df.index).normalize()
    asof = out["asof"]
    pos_arr = np.flatnonzero(idx == asof)
    if pos_arr.size != 1:
        return out  # 判定日が無い / 重複している。数えられないので打ち切り扱い
    t_pos = int(pos_arr[0])

    start, end = t_pos + 1, min(t_pos + horizon, len(df) - 1)
    n_bars = max(0, end - start + 1)
    out["n_bars"] = int(n_bars)
    out["censored"] = bool(n_bars < horizon)
    if n_bars == 0:
        return out

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    # SMA5 は各行がその行までの終値だけで決まるので、窓の外の未来には依存しない
    sma5 = sma(df["Close"], 5).to_numpy(dtype=float)

    win = slice(start, end + 1)
    entry_open = float(o[start])
    close_h = float(c[end])
    out["entry_open"] = entry_open
    out["close_h"] = close_h
    out["max_high"] = float(np.nanmax(h[win]))
    out["min_low"] = float(np.nanmin(lo[win]))
    if np.isfinite(entry_open) and entry_open > 0 and np.isfinite(close_h):
        out["ret_h"] = close_h / entry_open - 1.0

    lp = float(row["lp"]) if pd.notna(row["lp"]) else np.nan
    h0_high = float(row["h0_high"]) if pd.notna(row["h0_high"]) else np.nan

    if np.isfinite(lp):
        day = _first_day(lo[win] < lp)
        out["broke_lp"] = day is not None
        out["broke_lp_day"] = day if day is not None else pd.NA
    if np.isfinite(h0_high):
        day = _first_day(h[win] > h0_high)
        out["reached_h0"] = day is not None
        out["reached_h0_day"] = day if day is not None else pd.NA

    day = _first_day(c[win] > sma5[win])
    out["recovered_sma5"] = day is not None
    out["recovered_sma5_day"] = day if day is not None else pd.NA

    # SPEC §1 成功の定義: 押し安値を割らずに直近高値を更新。同じ日に両方起きたら
    # 順序が分からないので、labels.py と同じく保守的に失敗側に倒す
    if out["reached_h0"] is pd.NA or out["broke_lp"] is pd.NA:
        out["success"] = pd.NA
    elif not out["reached_h0"]:
        out["success"] = False
    elif not out["broke_lp"]:
        out["success"] = True
    else:
        out["success"] = bool(out["reached_h0_day"] < out["broke_lp_day"])
    return out


def resolve_delivered(delivered: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
                      horizon: int = HORIZON_DAYS, resolved_on=None) -> pd.DataFrame:
    """配信記録 1 ファイルぶんに結果を付ける。"""
    rows = [resolve_row(row, ohlcv.get(str(row["ticker"])), horizon, resolved_on)
            for _i, row in delivered.iterrows()]
    if not rows:
        return pd.DataFrame(columns=OUTCOME_COLS)
    return pd.DataFrame(rows)[OUTCOME_COLS]


# ------------------------------------------------------------------ ファイル操作
def outcome_path(daily_dir: Path, delivered_on, asof) -> Path:
    return Path(daily_dir) / stamped_name(OUTCOME_PREFIX, delivered_on, asof, OUTCOME_SUFFIX)


def save_outcome(df: pd.DataFrame, daily_dir: Path, delivered_on, asof) -> Path:
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = outcome_path(daily_dir, delivered_on, asof)
    out = df.copy()
    for col in DATE_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def load_outcome(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str})
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("broke_lp", "reached_h0", "recovered_sma5", "success", "censored"):
        if col in df.columns:
            df[col] = df[col].map(_coerce_bool).astype("boolean")
    return df


def _coerce_bool(v) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v in ("True", "true"):
            return True
        if v in ("False", "false"):
            return False
    return pd.NA


def resolve_pending(daily_dir: Path, ohlcv: Dict[str, pd.DataFrame],
                    horizon: int = HORIZON_DAYS, resolved_on=None,
                    calendar: Optional[pd.DatetimeIndex] = None, log=print) -> list[Path]:
    """結果が未付与で、かつ 5 営業日が経過した配信記録に結果を付ける。

    既に outcome ファイルがあるものは触らない（確定した記録を作り直さない）。
    5 営業日が市場側にまだ出ていないものは次回に持ち越す。
    """
    daily_dir = Path(daily_dir)
    if calendar is None:
        calendar = trading_calendar(ohlcv)
    written: list[Path] = []
    n_pending = 0
    for f in list_delivered(daily_dir):
        delivered = load_delivered(f.path)
        if len(delivered) == 0:
            continue   # 候補0件の日。結果を付ける対象が無い
        asof_values = pd.to_datetime(delivered["asof"]).dropna().unique()
        # 判定日はファイル名から採るのが正だが、旧名のファイルは中身から補う
        asof = f.asof if f.asof is not None else as_calendar_date(asof_values[0])
        if outcome_path(daily_dir, f.delivered_on, asof).exists():
            continue
        dues = [due_date(calendar, a, horizon) for a in asof_values]
        if not dues or any(d is None for d in dues):
            n_pending += 1
            continue
        stamp = resolved_on if resolved_on is not None else max(dues)
        outcome = resolve_delivered(delivered, ohlcv, horizon, stamp)
        written.append(save_outcome(outcome, daily_dir, f.delivered_on, asof))
        n_success = int(outcome["success"].fillna(False).astype(bool).sum())
        log(f"[resolve] {f.path.name} → {written[-1].name} "
            f"{len(outcome)}件 / 成功 {n_success} / "
            f"打ち切り {int(outcome['censored'].astype(bool).sum())}")
    if n_pending:
        log(f"[resolve] {horizon}営業日が未経過のため持ち越し: {n_pending}件")
    return written


# ------------------------------------------------------------------ 集計
def load_journal(daily_dir: Path) -> pd.DataFrame:
    """配信記録と結果を結合した運用の台帳を返す（結果が未付与の行は結果列が欠損）。"""
    daily_dir = Path(daily_dir)
    frames = []
    for f in list_delivered(daily_dir):
        delivered = load_delivered(f.path)
        if len(delivered) == 0:
            continue
        asof = f.asof
        if asof is None:
            values = pd.to_datetime(delivered["asof"], errors="coerce").dropna()
            asof = as_calendar_date(values.iloc[0]) if len(values) else None
        op = outcome_path(daily_dir, f.delivered_on, asof) if asof is not None else Path("")
        if op.name and op.exists():
            outcome = load_outcome(op).drop(columns=["delivered_on", "asof"], errors="ignore")
            delivered = delivered.merge(outcome, on="ticker", how="left")
        frames.append(delivered)
    if not frames:
        return pd.DataFrame(columns=list(dict.fromkeys(DELIVERED_COLS + OUTCOME_COLS)))
    return pd.concat(frames, ignore_index=True)


SUMMARY_COLS = ["landing_ma", "n", "n_resolved", "success_rate",
                "broke_lp_rate", "recovered_sma5_rate", "mean_ret_h"]
NO_LINE = "（線なし）"   # 止まった線が取れなかった記録。4本のいずれでもない


def _summary_row(name: str, g: pd.DataFrame) -> dict:
    resolved = g[g["success"].notna()] if "success" in g.columns and len(g) else g.iloc[0:0]
    return {
        "landing_ma": name,
        "n": int(len(g)),
        "n_resolved": int(len(resolved)),
        "success_rate": _rate(resolved, "success"),
        "broke_lp_rate": _rate(resolved, "broke_lp"),
        "recovered_sma5_rate": _rate(resolved, "recovered_sma5"),
        "mean_ret_h": (float(resolved["ret_h"].mean())
                       if len(resolved) and "ret_h" in resolved else np.nan),
    }


def summarize_by_landing_ma(journal: pd.DataFrame, max_dist_atr: Optional[float] = None) -> pd.DataFrame:
    """止まった線ごとの件数と実績（docs/SCREENER.md §3.4）。

    **SMA5 / SMA25 / SMA75 / SMA200 の 4 行を常に出す。** 候補に一度も出ていない線は
    n=0 の行になる。groupby は存在する値しかグループにしないため、素朴に集計すると
    その線の行ごと表から消え、「まだ集計されていない」のか「候補に出ていない」のかが
    読み手に区別できなくなる。

    **記録がある状態での n=0 は「D4（ma_dist ≤ 1.0 かつ 5/25/75日線）がその線を通して
    いない」という情報であって、データ不足ではない。** 判断材料になるのはこの対比なので
    行を落とさない（§3.4）。journal 自体が空なら全て 0 になるが、それは「まだ記録が無い」
    だけで D4 の情報ではない。

    max_dist_atr を与えると、押し安値がその距離より遠い記録を除いて集計する
    （最も近い線が遠い場合、その線で止まったとは言いにくいため）。

    表を出すところまでがこの関数の仕事で、採否の判断はしない（CLAUDE.md）。
    """
    empty = pd.DataFrame(columns=journal.columns if journal is not None else [])
    df = empty if journal is None or "landing_ma" not in getattr(journal, "columns", []) \
        else journal.copy()
    if len(df) and max_dist_atr is not None and "landing_dist_atr" in df.columns:
        df = df[df["landing_dist_atr"] <= max_dist_atr]
    if len(df):
        df["landing_ma"] = df["landing_ma"].fillna("").replace("", NO_LINE)

    groups = {name: g for name, g in df.groupby("landing_ma", sort=False)} if len(df) else {}
    # 4 本は常に出す。線が取れなかった記録がある日だけ NO_LINE を末尾に足す
    names = list(LANDING_MA_NAMES) + ([NO_LINE] if NO_LINE in groups else [])
    rows = [_summary_row(name, groups.get(name, df.iloc[0:0])) for name in names]
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def _rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return np.nan
    vals = df[col].dropna()
    return float(vals.astype(bool).mean()) if len(vals) else np.nan
