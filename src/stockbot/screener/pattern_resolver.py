"""パターンの結果付け（docs/PATTERN.md §3.4 / §4 Q-3 の回答）。

**押し目型の結果付け（`resolver.py`）とは別物である。** あちらは押し安値 `lp` と
直近高値 `h0_high` を参照していて、パターンにはその列が無い。保全してある押し目型
15 件の記録はあちらで付け続ける（列を混ぜない）。

決まっていること（2026-09-11・設計責任者）:

| 項目 | 決め方 |
| --- | --- |
| 評価窓 | **T+1..T+20**（営業日）。**裁量値**。文献の根拠は無い |
| `broke_stop` | 撤退の目安（パターンの最安値）を割ったか（`Low < pattern_low`） |
| `reached_target` | 測定目標に届いたか（`High > target`） |
| `success` | **撤退を割る前に目標に届いたか。** 同じ日に両方なら失敗（保守的） |
| MFE / MAE | 窓内の最大上昇幅・最大下落幅。**ATR 単位でも出す** |

**20 本にした理由は「パターンの形成期間と同程度なら妥当だろう」という程度である。**
5 本では足りない —— span の中央が 28 本で目標まで数週間かかる形なので、5 本で測ると
ほぼ全件が「届かなかった」になり、記録として意味を持たない（§4 Q-3）。

**`recovered_sma5` は持たない。** 押し目型の「5日線回復で手仕舞い」に対応する概念が
パターンには無い（§4 Q-2 で矛盾が指摘された出口ルールでもある）。

**MFE / MAE を必ず出す。** 目標に届かなくても、どこまで伸びたかが分かる。前の
プロジェクトで主指標に指定しながらレポートに載っていなかったのが判定を誤らせた
原因だった（設計責任者の指摘）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .pattern_record import load_pattern_delivered
from .record import as_calendar_date, stamped_name

# **裁量値**（docs/PATTERN.md §1 の出典列でいう裁量値）。文献の根拠は無い。
# 押し目型の 5 本（SCREENER.md §3.3）とは別の数字である
PATTERN_HORIZON_DAYS = 20

PATTERN_OUTCOME_PREFIX = "pattern_outcome_"
PATTERN_OUTCOME_SUFFIX = ".csv"

PATTERN_OUTCOME_COLS = [
    "delivered_on",          # 配信記録と同じキー
    "asof",
    "ticker",
    "pattern",               # 1 銘柄が複数パターンで成立しうるのでキーに要る
    "resolved_on",           # 結果を付けた日
    "horizon_days",          # 評価窓の本数（20）
    "n_bars",                # 実際に使えた本数（< horizon なら打ち切り）
    "censored",              # n_bars < horizon_days
    "entry_open",            # Open[T+1]（SPEC §1 のエントリー基準）
    "close_h",               # Close[T+20]（打ち切り時は取れた最後の終値）
    "max_high",              # max(High[T+1..T+20])
    "min_low",               # min(Low[T+1..T+20])
    "ret_h",                 # close_h / entry_open − 1
    "broke_stop",            # 撤退の目安を割ったか（Low < pattern_low）
    "broke_stop_day",        # 割った最初の日（T からの本数。割らなければ空）
    "reached_target",        # 測定目標に届いたか（High > target）
    "reached_target_day",
    "success",               # **撤退を割る前に目標到達。同日なら失敗（保守的）**
    # MFE / MAE。**エントリー（Open[T+1]）からの最大上昇幅・最大下落幅**
    "mfe",                   # max_high − entry_open（円）
    "mae",                   # min_low − entry_open（円・負）
    "mfe_atr",               # mfe / ATR14[T]
    "mae_atr",               # mae / ATR14[T]（負）
    # 配信時点の距離（記録に書いてある T の引けの値。結果ではない）
    "up_pct", "down_pct", "rr", "breakeven_win_rate",
]

DATE_COLS = ["delivered_on", "asof", "resolved_on"]
BOOL_COLS = ["censored", "broke_stop", "reached_target", "success"]


def _first_day(mask: np.ndarray) -> Optional[int]:
    """T+1 を 1 日目としたときに、条件が最初に成立した日（成立しなければ None）。"""
    hits = np.flatnonzero(mask)
    return int(hits[0]) + 1 if hits.size else None


def _f(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def resolve_row(row: pd.Series, df: Optional[pd.DataFrame],
                horizon: int = PATTERN_HORIZON_DAYS, resolved_on=None) -> dict:
    """成立 1 行に結果を付ける。

    df は当該銘柄の OHLCV（DatetimeIndex・昇順）。無い / 判定日が入っていない場合は
    n_bars=0・censored=True で全ての結果を欠損にする（例外にしない）。

    **判定に使った値は記録から読む**（`pattern_low` / `target` / `atr_t`）。窓の中の
    値動きだけを OHLCV から読む。結果付けは条件を一切参照しない。
    """
    out: dict = {c: np.nan for c in PATTERN_OUTCOME_COLS}
    out.update({
        "delivered_on": pd.Timestamp(row["delivered_on"]).normalize(),
        "asof": pd.Timestamp(row["asof"]).normalize(),
        "ticker": str(row["ticker"]),
        "pattern": str(row.get("pattern") or ""),
        "resolved_on": (pd.Timestamp(resolved_on).normalize()
                        if resolved_on is not None else pd.NaT),
        "horizon_days": int(horizon),
        "n_bars": 0,
        "censored": True,
        "broke_stop": pd.NA, "broke_stop_day": pd.NA,
        "reached_target": pd.NA, "reached_target_day": pd.NA,
        "success": pd.NA,
    })
    for col in ("up_pct", "down_pct", "rr", "breakeven_win_rate"):
        out[col] = _f(row.get(col))

    if df is None or len(df) == 0:
        return out
    idx = pd.DatetimeIndex(df.index).normalize()
    pos_arr = np.flatnonzero(idx == out["asof"])
    if pos_arr.size != 1:
        return out       # 判定日が無い / 重複。数えられないので打ち切り扱い
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
    win = slice(start, end + 1)

    entry_open = float(o[start])
    close_h = float(c[end])
    max_high = float(np.nanmax(h[win]))
    min_low = float(np.nanmin(lo[win]))
    out["entry_open"] = entry_open
    out["close_h"] = close_h
    out["max_high"] = max_high
    out["min_low"] = min_low
    if np.isfinite(entry_open) and entry_open > 0 and np.isfinite(close_h):
        out["ret_h"] = close_h / entry_open - 1.0

    # MFE / MAE は**エントリー（Open[T+1]）から**測る。ret_h と同じ基準にする
    if np.isfinite(entry_open):
        out["mfe"] = max_high - entry_open
        out["mae"] = min_low - entry_open
        atr = _f(row.get("atr_t"))
        if np.isfinite(atr) and atr > 0:
            out["mfe_atr"] = out["mfe"] / atr
            out["mae_atr"] = out["mae"] / atr

    stop, target = _f(row.get("pattern_low")), _f(row.get("target"))
    if np.isfinite(stop):
        day = _first_day(lo[win] < stop)
        out["broke_stop"] = day is not None
        out["broke_stop_day"] = day if day is not None else pd.NA
    if np.isfinite(target):
        day = _first_day(h[win] > target)
        out["reached_target"] = day is not None
        out["reached_target_day"] = day if day is not None else pd.NA

    # **撤退を割る前に目標到達。** 同じ日に両方起きたら順序が分からないので、
    # 押し目型（labels.py）と同じく保守的に失敗側へ倒す
    if out["reached_target"] is pd.NA or out["broke_stop"] is pd.NA:
        out["success"] = pd.NA
    elif not out["reached_target"]:
        out["success"] = False
    elif not out["broke_stop"]:
        out["success"] = True
    else:
        out["success"] = bool(out["reached_target_day"] < out["broke_stop_day"])
    return out


def resolve_delivered(delivered: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
                      horizon: int = PATTERN_HORIZON_DAYS,
                      resolved_on=None) -> pd.DataFrame:
    rows = [resolve_row(row, ohlcv.get(str(row["ticker"])), horizon, resolved_on)
            for _i, row in delivered.iterrows()]
    if not rows:
        return pd.DataFrame(columns=PATTERN_OUTCOME_COLS)
    return pd.DataFrame(rows)[PATTERN_OUTCOME_COLS]


def outcome_path(daily_dir: Path, delivered_on, asof) -> Path:
    """パターンの結果ファイル。**押し目型の `outcome_*` とは別の prefix。**

    列が違うので同じ名前空間に置くと、読む側がどちらか分からなくなる。
    """
    return Path(daily_dir) / stamped_name(PATTERN_OUTCOME_PREFIX, delivered_on, asof,
                                          PATTERN_OUTCOME_SUFFIX)


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


def _coerce_bool(v) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v in ("True", "true"):
            return True
        if v in ("False", "false"):
            return False
    return pd.NA


def load_outcome(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str, "pattern": str})
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map(_coerce_bool).astype("boolean")
    return df


def resolve_file(path: Path, daily_dir: Path, delivered_on, asof,
                 ohlcv: Dict[str, pd.DataFrame],
                 horizon: int = PATTERN_HORIZON_DAYS, resolved_on=None) -> pd.DataFrame:
    """配信記録 1 ファイルぶんを解決して保存する（呼び出し側は `resolver.resolve_pending`）。"""
    delivered = load_pattern_delivered(path)
    outcome = resolve_delivered(delivered, ohlcv, horizon, resolved_on)
    save_outcome(outcome, daily_dir, as_calendar_date(delivered_on), as_calendar_date(asof))
    return outcome
