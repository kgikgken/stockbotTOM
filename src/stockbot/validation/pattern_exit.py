"""ATR 基準の出口の検証（docs/BACKTEST.md §10）。

**2 つを 1 回の実行でまとめる。**

- **A. 記述統計**（検定なし・判定に使わない）: 成立時点の `atr_t ÷ 終値[T]` の分布と、
  ATR×2 が株価の何 % に相当するか
- **B. ATR×2 を出口とした場合の成績**（**検定 1 件**）: 倍率は **2.0 のみ**。
  1.5・2.5 は試さない（§10 の禁止事項）

**検定は既存 10 件 + 今回 1 件 = 11 件。** これ以上増やさない。

**判定式・閾値は変更しない。** ここは出口の当て方を変えて成績を見るだけで、
検出（`features/pattern.py`）には一切触らない。

**未来参照はしない。** `atr_t`・撤退ライン・測定目標はすべて T の引けまでの値で、
再生のときに保存済みである。ここで新しく読むのは **T+1..T+20 の四本値だけ**
（ラベルと同じ範囲）。

**数値の解釈も採否の判断もしない。表を作るまで**（CLAUDE.md）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .layer1 import newey_west_mean_t
from .pattern_replay import HORIZON
from .pattern_report import N_TESTS as N_TESTS_BEFORE
from .replay import _date_position, _truncate_before_holdout

# 事前登録（docs/BACKTEST.md §10）。**倍率は 2.0 のみ。振って比較しない。**
# 根拠は探索窓の MFE 実測平均 2.27（**データ由来**であることを §10 に明記してある）
ATR_MULT = 2.0

# 検定数（docs/BACKTEST.md §10）。**既存 10 件 + 今回 1 件 = 11 件。増やさない。**
# A の記述統計は検定に数えない（判定に使わないため）
N_TESTS_HERE = 1
N_TESTS_TOTAL = N_TESTS_BEFORE + N_TESTS_HERE

# 出口の種類
OUTCOME_TARGET = "target"    # 利確に到達
OUTCOME_STOP = "stop"        # 撤退ラインを割った
OUTCOME_TIMEOUT = "timeout"  # どちらも起きず T+20 の終値で手仕舞い
OUTCOMES = (OUTCOME_TARGET, OUTCOME_STOP, OUTCOME_TIMEOUT)

EXIT_COLS = [
    "date", "ticker", "pattern",
    "close_t", "atr_t", "atr_ratio", "atr2_pct",
    "entry_open", "pattern_low", "target", "atr2_price",
    "n_bars", "censored", "already_at_target",
    "entry_below_stop", "entry_above_target",
    "atr2_outcome", "atr2_exit_day", "atr2_exit_price", "atr2_pnl_pct",
    "tgt_outcome", "tgt_exit_day", "tgt_exit_price", "tgt_pnl_pct",
]


def truncate_before_holdout(ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """四本値をホールドアウト開始日より前で物理的に打ち切る（CLAUDE.md の絶対規則）。

    日付で T を絞るだけでは足りない —— **探索窓の末尾の T は T+20 でホールドアウト
    側のバーに届く**。`replay.py` と同じ関数をそのまま使う。
    """
    return _truncate_before_holdout(ohlcv)


def _first_day(mask: np.ndarray) -> Optional[int]:
    """最初に True になった位置（1 始まり＝ T+何日目か）。無ければ None。"""
    hits = np.flatnonzero(mask)
    return int(hits[0]) + 1 if hits.size else None


def simulate_exit(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  start: int, end: int, entry: float, stop: float, target: float,
                  target_strict: bool) -> dict:
    """1 行ぶんの出口（docs/BACKTEST.md §10 の事前登録）。

    - 利確: 高値が `target` に届いた日に、**その価格で約定したとみなす**
    - 撤退: 安値が `stop` を割った日に、**その価格で約定したとみなす**
    - **同日に両方なら撤退**（保守的。`label_one` の `success` と同じ流儀）
    - どちらも起きなければ **T+20 の終値**で手仕舞い

    `target_strict=True` は `High > target`（既存の `reached_target` と同じ定義）、
    `False` は `High >= target`（§10 B の「到達した日」）。**定義を混ぜないため引数に
    出してある。**

    **約定は指定価格ちょうど。** 手数料・スリッページ・ギャップは考慮しない
    （§10 の「理想約定」）。ギャップで不利に寄った行は `entry_below_stop` /
    `entry_above_target` で数えられるようにしてある。
    """
    out = {"outcome": "", "exit_day": pd.NA, "exit_price": np.nan, "pnl_pct": np.nan}
    n_bars = max(0, end - start + 1)
    if n_bars <= 0 or not np.isfinite(entry) or entry <= 0:
        return out
    # **利確の値が無い行は「時間切れ」にしない。** 出口そのものが定義できていない
    # ので空で返し、平均からも外す（0 や終値で埋めると成績が動く）
    if not np.isfinite(target):
        return out
    win = slice(start, end + 1)
    hit = (_first_day(high[win] > target) if target_strict
           else _first_day(high[win] >= target))
    brk = _first_day(low[win] < stop) if np.isfinite(stop) else None

    if hit is None and brk is None:
        outcome, day, price = OUTCOME_TIMEOUT, n_bars, float(close[end])
    elif hit is None:
        outcome, day, price = OUTCOME_STOP, brk, float(stop)
    elif brk is None:
        outcome, day, price = OUTCOME_TARGET, hit, float(target)
    elif brk <= hit:   # **同日なら撤退**
        outcome, day, price = OUTCOME_STOP, brk, float(stop)
    else:
        outcome, day, price = OUTCOME_TARGET, hit, float(target)

    if not np.isfinite(price):
        return out
    out.update({"outcome": outcome, "exit_day": int(day), "exit_price": price,
                "pnl_pct": (price / entry - 1.0) * 100.0})
    return out


def ohlc_arrays(df: pd.DataFrame) -> tuple:
    """四本値を numpy にする。**銘柄ごとに 1 回で済ませるため外に出してある**
    （同じ銘柄で何千行も当てるので、行ごとに変換すると効かない）。"""
    return tuple(df[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))


def exit_one(df: pd.DataFrame, t_pos: int, row: dict, mult: float = ATR_MULT,
             horizon: int = HORIZON, arrays: Optional[tuple] = None) -> dict:
    """1 行ぶん。**ここだけが T+1 以降を見る**（`label_one` と同じ範囲）。

    `arrays` は `ohlc_arrays(df)` の結果。渡さなければここで作る（結果は同じ）。
    """
    out: dict = {c: np.nan for c in EXIT_COLS if c not in ("date", "ticker", "pattern")}
    out.update({"atr2_outcome": "", "tgt_outcome": "",
                "atr2_exit_day": pd.NA, "tgt_exit_day": pd.NA,
                "censored": True, "n_bars": 0,
                "entry_below_stop": pd.NA, "entry_above_target": pd.NA})

    close_t = float(row.get("close_t", np.nan))
    atr_t = float(row.get("atr_t", np.nan))
    stop = float(row.get("pattern_low", np.nan))
    target = float(row.get("target", np.nan))
    out.update({"close_t": close_t, "atr_t": atr_t, "pattern_low": stop,
                "target": target, "already_at_target": row.get("already_at_target")})
    if np.isfinite(atr_t) and np.isfinite(close_t) and close_t > 0:
        out["atr_ratio"] = atr_t / close_t
        out["atr2_pct"] = mult * atr_t / close_t * 100.0

    start, end = t_pos + 1, min(t_pos + horizon, len(df) - 1)
    n_bars = max(0, end - start + 1)
    out["n_bars"] = int(n_bars)
    out["censored"] = bool(n_bars < horizon)
    if n_bars == 0:
        return out

    o, h, lo, c = arrays if arrays is not None else ohlc_arrays(df)
    entry = float(o[start])
    out["entry_open"] = entry
    if not np.isfinite(entry) or entry <= 0:
        return out
    if np.isfinite(atr_t) and atr_t > 0:
        out["atr2_price"] = entry + mult * atr_t
    out["entry_below_stop"] = bool(np.isfinite(stop) and entry < stop)
    out["entry_above_target"] = bool(np.isfinite(target) and entry > target)

    # B: ATR×2 を出口とする。「到達した日」なので **>=**
    atr2 = simulate_exit(h, lo, c, start, end, entry, stop,
                         out["atr2_price"], target_strict=False)
    # 参考: 測定目標を出口とする。**既存の reached_target と同じ定義（>）** にそろえる
    tgt = simulate_exit(h, lo, c, start, end, entry, stop, target, target_strict=True)
    for prefix, res in (("atr2", atr2), ("tgt", tgt)):
        out[f"{prefix}_outcome"] = res["outcome"]
        out[f"{prefix}_exit_day"] = res["exit_day"]
        out[f"{prefix}_exit_price"] = res["exit_price"]
        out[f"{prefix}_pnl_pct"] = res["pnl_pct"]
    return out


def run(replay: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
        mult: float = ATR_MULT, horizon: int = HORIZON, log=print) -> pd.DataFrame:
    """保存済みの再生結果に出口を当てる。**検出はやり直さない。**

    `replay` は `pattern_replay.load_table` の出力。`ohlcv` は **ホールドアウトを
    打ち切った後**のものを渡すこと（呼び出し側の責任。CLAUDE.md の絶対規則）。
    """
    if len(replay) == 0:
        return pd.DataFrame(columns=EXIT_COLS)
    rows = []
    n_missing = 0
    cache: dict = {}
    for _i, r in replay.iterrows():
        ticker = str(r["ticker"])
        df = ohlcv.get(ticker)
        date_t = pd.Timestamp(r["date"])
        t_pos = _date_position(df.index, date_t) if df is not None else None
        if t_pos is None:
            n_missing += 1
            continue
        if ticker not in cache:
            cache[ticker] = ohlc_arrays(df)
        rows.append({"date": date_t, "ticker": ticker, "pattern": str(r["pattern"]),
                     **exit_one(df, t_pos, r.to_dict(), mult, horizon, cache[ticker])})
    if n_missing:
        # **黙って落とさない。** 落ちた行数はそのまま報告する
        log(f"[pattern-exit] 四本値が引けずに落ちた行: {n_missing}")
    if not rows:
        return pd.DataFrame(columns=EXIT_COLS)
    return pd.DataFrame(rows)[EXIT_COLS]


# ------------------------------------------------------------------ A. 記述統計
def atr_ratio_stats(df: pd.DataFrame, mult: float = ATR_MULT) -> dict:
    """A（検定なし）。`atr_t ÷ 終値[T]` の分布と ATR×倍率 が株価の何 % か。"""
    x = (df["atr_ratio"].dropna().to_numpy(dtype=float)
         if "atr_ratio" in df.columns else np.asarray([], dtype=float))
    if not len(x):
        return {"n": 0}
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)),
            "min": float(x.min()), "q1": float(q[0]), "median": float(q[1]),
            "q3": float(q[2]), "max": float(x.max()), "mean": float(x.mean()),
            "mult": float(mult)}


def format_ratio_stats(name: str, s: dict) -> list[str]:
    """A の表（`atr_t/終値` と ATR×倍率 の % を併記）。"""
    head = (f"{'窓':<10}{'件数':>8}{'最小':>9}{'Q1':>9}{'中央':>9}{'Q3':>9}"
            f"{'最大':>9}{'平均':>9}")
    if not s.get("n"):
        return [head, f"{name:<10}{0:>8}" + "—".rjust(9) * 6]
    m = s["mult"]

    def row(label, scale):
        return (f"{label:<10}{s['n']:>8}"
                + "".join(f"{s[k] * scale:>8.2f}%" for k in
                          ("min", "q1", "median", "q3", "max", "mean")))
    return [head, row(name, 100.0), row(f"　×{m:.1f}", 100.0 * m)]


# ------------------------------------------------------------------ B. 成績
def summarize_exit(name: str, df: pd.DataFrame, prefix: str = "atr2",
                   lag: int = HORIZON) -> dict:
    """B の 1 行（docs/BACKTEST.md §10）。**件数（母数）を必ず持つ。**

    NW t は**日次平均系列**に当てる（D-4 と同じ理由。評価窓が 20 本で重なる）。
    """
    pnl = f"{prefix}_pnl_pct"
    part = df[["date", pnl]].dropna() if len(df) and pnl in df.columns else \
        pd.DataFrame(columns=["date", pnl])
    daily = (part.groupby("date")[pnl].mean().sort_index() if len(part)
             else pd.Series(dtype=float))
    nw = newey_west_mean_t(daily.to_numpy(dtype=float), lag=lag)
    oc = df[f"{prefix}_outcome"] if len(df) and f"{prefix}_outcome" in df.columns \
        else pd.Series(dtype=object)
    n = int(len(df))

    def rate(kind):
        return float((oc == kind).sum()) / n if n else float("nan")

    v = part[pnl] if len(part) else pd.Series(dtype=float)
    return {"group": name, "n": n, "n_days": int(len(daily)),
            "mean_pnl": float(v.mean()) if len(v) else float("nan"),
            "nw_t": nw["t"], "nw_p": nw["p"],
            "target_rate": rate(OUTCOME_TARGET),
            "stop_rate": rate(OUTCOME_STOP),
            "timeout_rate": rate(OUTCOME_TIMEOUT),
            "win_rate": float((v > 0).mean()) if len(v) else float("nan"),
            "median_pnl": float(v.median()) if len(v) else float("nan"),
            # **出口を定義できなかった行**（利確の値が無い）。母数から黙って消さない
            "n_no_exit": int((oc == "").sum()) if n else 0,
            "exit_day_median": (float(df[f"{prefix}_exit_day"].dropna().median())
                                if len(df) and f"{prefix}_exit_day" in df.columns
                                and df[f"{prefix}_exit_day"].notna().any()
                                else float("nan"))}


EXIT_GROUP_COLS = ["group", "n", "n_days", "mean_pnl", "nw_t", "nw_p",
                   "target_rate", "stop_rate", "timeout_rate", "win_rate",
                   "median_pnl", "exit_day_median", "n_no_exit"]


def format_exit_table(table: pd.DataFrame) -> list[str]:
    """B の表（表のみ。解釈はしない）。"""
    out = [f"{'出口':<30}{'件数':>8}{'日数':>7}{'平均損益':>10}{'NW t':>8}"
           f"{'到達率':>8}{'撤退率':>8}{'時間切れ':>9}{'勝率':>8}"
           f"{'中央損益':>10}{'日数中央':>9}{'出口なし':>9}"]
    for _i, r in table.iterrows():
        def pct(v):
            return "—" if pd.isna(v) else f"{float(v) * 100:.1f}%"

        def num(v, d=2, sign="+"):
            return "—" if pd.isna(v) else f"{float(v):{sign}.{d}f}"

        out.append(f"{str(r['group']):<30}{int(r['n']):>8}{int(r['n_days']):>7}"
                   f"{num(r['mean_pnl']) + '%':>10}{num(r['nw_t']):>8}"
                   f"{pct(r['target_rate']):>8}{pct(r['stop_rate']):>8}"
                   f"{pct(r['timeout_rate']):>9}{pct(r['win_rate']):>8}"
                   f"{num(r['median_pnl']) + '%':>10}{num(r['exit_day_median'], 1, ''):>9}"
                   f"{int(r['n_no_exit']):>9}")
    return out


def exit_path(base: Path, window: str) -> Path:
    return Path(base) / f"pattern_exit_{window}.csv.gz"
