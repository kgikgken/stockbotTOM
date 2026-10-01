"""T+1 終値による早期撤退の記述統計（docs/BACKTEST.md §16）。

**記述統計であって判定ではない。** 判定基準を設けない。採用も不採用も決めない。

- **検定は 15 件のまま増やさない**（§12.6・§13.1・§15.1）
- **事前登録（`docs/PREREGISTRATION_H123.md`）に触れない。** H1 の母集団も判定期限も
  動かさない
- **この結果を根拠に設計・判定式・閾値・並び順・配信を変更しない**
- **既存節（§9〜§15）の数字を変更しない。** 出力は `data/pattern_early/` に書き、
  `data/pattern_exit/` を上書きしない

**検出はやり直さない。** 保存済みの再生結果の行に、出口の当て方だけを変えて適用する。

**未来参照はしない。** 建値・撤退ライン・測定目標はすべて T の引けまでの値で、再生の
ときに保存済みである。ここで新しく読むのは **T+1..T+20 の四本値だけ**（ラベルと同じ
範囲）。

**数値の解釈も採否の判断もしない。表を作るまで**（CLAUDE.md）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .pattern_exit import (OUTCOME_STOP, OUTCOME_TARGET, OUTCOME_TIMEOUT,
                           ohlc_arrays, simulate_exit)
from .pattern_replay import HORIZON
from .pattern_strata import N_TESTS_TOTAL as N_TESTS_BEFORE
from .replay import _date_position

# **検定を 1 件も足さない**（§16.1）。記述統計なので判定に使わない
N_TESTS_HERE = 0
N_TESTS_TOTAL = N_TESTS_BEFORE

# この案だけの出口。T+1 の終値が建値を割ったので翌日の始値で降りた行
OUTCOME_EARLY = "early"

# 抜け幅 5 分位のうち集計から外す群（§16.3）。**b >= 0.957 の Q5 は勝率 8% で、
# この案の効果が見込めないため除外する**（設計責任者の指示）。
# **行は落とさない。表から外すだけ** —— 除外した件数を数えられるようにしておく
EXCLUDED_QUANTILE = "Q5"

EARLY_COLS = [
    "date", "ticker", "pattern", "q", "breakout_h", "close_t",
    "entry_open", "pattern_low", "target", "t1_close", "t1_below_entry",
    "n_bars", "censored", "already_at_target",
    "cur_outcome", "cur_exit_day", "cur_exit_price", "cur_pnl_pct",
    "early_outcome", "early_exit_day", "early_exit_price", "early_pnl_pct",
    "early_exit_at_open",
]

SUMMARY_COLS = ["window", "q", "n", "n_cur_only", "n_early_only",
                "cur_mean_pnl", "early_mean_pnl", "diff",
                "cur_win_rate", "early_win_rate", "early_breakeven_rate",
                "cur_exit_day_mean", "early_exit_day_mean", "t1_below_rate",
                "early_rate", "target_rate", "stop_rate", "timeout_rate"]


def simulate_early(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray, start: int, end: int, entry: float,
                   stop: float, target: float) -> dict:
    """1 行ぶんの出口（§16.2 の案）。

    **T+1 の中は現行どおり**（利確＝`High > target`、撤退＝`Low < pattern_low`、
    同日に両方なら撤退）。T+1 がそのどちらでも終わらなかった行だけが分岐する:

    - `Close[T+1] < 建値` → **T+2 の始値で手仕舞い**（`OUTCOME_EARLY`）
    - `Close[T+1] >= 建値` → **撤退ラインを建値に引き上げ**、T+2 以降は現行どおり
      （利確・建値割れ・T+20 の終値のいずれか）

    **T+1 の中では撤退ラインを上げない。** 引き上げは T+1 の引けを見てからなので、
    T+1 の安値が建値を割っていても降りない（§16.2 の実装上の読み）。

    **約定は指定価格ちょうど。** 手数料・スリッページ・ギャップは考慮しない
    （§10 の「理想約定」と同じ）。

    返すのは `outcome` / `exit_day` / `exit_price` / `pnl_pct` /
    `exit_at_open` / `t1_below_entry`。当てられない行は `outcome` が空で、
    **母数から外す**（§11.3・§12.2 と同じ扱い。0 や終値で埋めると成績が動く）。
    """
    out: dict = {"outcome": "", "exit_day": pd.NA, "exit_price": np.nan,
                 "pnl_pct": np.nan, "exit_at_open": False,
                 "t1_below_entry": pd.NA}
    n_bars = max(0, end - start + 1)
    if n_bars <= 0 or not np.isfinite(entry) or entry <= 0:
        return out
    # **利確の値が無い行は「時間切れ」にしない**（§10 と同じ）
    if not np.isfinite(target):
        return out

    def done(outcome: str, day: int, price: float, at_open: bool,
             below) -> dict:
        if not np.isfinite(price):
            return out
        return {"outcome": outcome, "exit_day": int(day), "exit_price": float(price),
                "pnl_pct": (float(price) / entry - 1.0) * 100.0,
                "exit_at_open": bool(at_open), "t1_below_entry": below}

    # --- T+1 の中は現行どおり。**同日に両方なら撤退**
    brk1 = (np.isfinite(stop) and np.isfinite(low[start]) and low[start] < stop)
    hit1 = (np.isfinite(high[start]) and high[start] > target)
    if brk1:
        return done(OUTCOME_STOP, 1, stop, False, pd.NA)
    if hit1:
        return done(OUTCOME_TARGET, 1, target, False, pd.NA)

    t1_close = float(close[start])
    if not np.isfinite(t1_close):
        return out
    below = bool(t1_close < entry)

    if below:
        # **T+2 の始値で手仕舞い。** 翌日が無ければこの案は当てられない
        nxt = start + 1
        if nxt > end or not np.isfinite(open_[nxt]):
            return dict(out, t1_below_entry=True)
        return done(OUTCOME_EARLY, 2, open_[nxt], True, True)

    if start + 1 > end:
        # T+1 が評価窓の最終日。現行と同じく終値で手仕舞い
        return done(OUTCOME_TIMEOUT, 1, t1_close, False, False)

    # **撤退ラインを建値に引き上げ、T+2 以降は現行どおり。**
    # `simulate_exit` の `exit_day` は渡した start からの 1 始まりなので +1 する
    res = simulate_exit(high, low, close, start + 1, end, entry, entry, target,
                        target_strict=True)
    if not res["outcome"]:
        return dict(out, t1_below_entry=False)
    return done(res["outcome"], int(res["exit_day"]) + 1, res["exit_price"],
                False, False)


def early_one(df: pd.DataFrame, t_pos: int, row: dict, horizon: int = HORIZON,
              arrays: Optional[tuple] = None) -> dict:
    """1 行ぶん。**ここだけが T+1 以降を見る**（`label_one` と同じ範囲）。

    現行基準（`cur_*`）も同じ関数（§10 の `simulate_exit`）で同時に出す。
    **同じ行で両方を出さないと差が取れない。**
    """
    out: dict = {c: np.nan for c in EARLY_COLS
                 if c not in ("date", "ticker", "pattern", "q")}
    out.update({"cur_outcome": "", "early_outcome": "",
                "cur_exit_day": pd.NA, "early_exit_day": pd.NA,
                "t1_below_entry": pd.NA, "early_exit_at_open": False,
                "censored": True, "n_bars": 0})

    stop = float(row.get("pattern_low", np.nan))
    target = float(row.get("target", np.nan))
    out.update({"close_t": float(row.get("close_t", np.nan)),
                "breakout_h": float(row.get("breakout_h", np.nan)),
                "pattern_low": stop, "target": target,
                "already_at_target": row.get("already_at_target")})

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
    out["t1_close"] = float(c[start])

    # 現行基準（§10 の tgt と同じ定義。利確は `High > target` の厳密版）
    cur = simulate_exit(h, lo, c, start, end, entry, stop, target,
                        target_strict=True)
    out.update({"cur_outcome": cur["outcome"], "cur_exit_day": cur["exit_day"],
                "cur_exit_price": cur["exit_price"], "cur_pnl_pct": cur["pnl_pct"]})

    early = simulate_early(o, h, lo, c, start, end, entry, stop, target)
    out.update({"early_outcome": early["outcome"],
                "early_exit_day": early["exit_day"],
                "early_exit_price": early["exit_price"],
                "early_pnl_pct": early["pnl_pct"],
                "early_exit_at_open": early["exit_at_open"],
                "t1_below_entry": early["t1_below_entry"]})
    return out


def assign_quantile(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """抜け幅を固定境界で Q1〜Q5 に振る（§4.2 の確定値をそのまま使う）。

    `edges` は両端が ±inf の 6 要素（`pattern_report.load_frozen_edges`）。
    **窓ごとに切り直さない。**
    """
    out = np.full(len(values), "", dtype=object)
    x = np.asarray(values, dtype=float)
    last = len(edges) - 2
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & ((x <= hi) if i == last else (x < hi))
        out[mask] = f"Q{i + 1}"
    return out


def run(replay: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
        edges: Optional[np.ndarray] = None, horizon: int = HORIZON,
        log=print) -> pd.DataFrame:
    """保存済みの再生結果に §16 の出口を当てる。**検出はやり直さない。**

    `ohlcv` は **ホールドアウトを打ち切った後**のものを渡すこと（呼び出し側の責任。
    CLAUDE.md の絶対規則）。
    """
    if len(replay) == 0:
        return pd.DataFrame(columns=EARLY_COLS)
    rows: List[dict] = []
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
                     "q": "",
                     **early_one(df, t_pos, r.to_dict(), horizon, cache[ticker])})
    if n_missing:
        # **黙って落とさない。** 落ちた行数はそのまま報告する
        log(f"[pattern-early] 四本値が引けずに落ちた行: {n_missing}")
    if not rows:
        return pd.DataFrame(columns=EARLY_COLS)
    out = pd.DataFrame(rows)
    if edges is not None:
        out["q"] = assign_quantile(out["breakout_h"].to_numpy(dtype=float), edges)
    return out[EARLY_COLS]


# ------------------------------------------------------------------ 集計
def _rate(series: pd.Series, kind: str, n: int) -> float:
    return float((series == kind).sum()) / n if n else float("nan")


def summarize(window: str, q: str, df: pd.DataFrame) -> dict:
    """1 群ぶんの行（§16.4）。

    **両方の案に出口がある行だけを母数にする。** 片方だけで平均を取ると、差の列が
    別々の行集合の平均の引き算になる（§14.4・§15.1 と同じ理由）。
    片方しか当てられなかった行は `n_cur_only` / `n_early_only` に数えて残す。
    """
    cur_ok = df["cur_outcome"].fillna("").astype(str) != "" if len(df) else df
    early_ok = df["early_outcome"].fillna("").astype(str) != "" if len(df) else df
    both = df[cur_ok & early_ok] if len(df) else df
    n = int(len(both))
    row = {"window": window, "q": q, "n": n,
           "n_cur_only": int((cur_ok & ~early_ok).sum()) if len(df) else 0,
           "n_early_only": int((~cur_ok & early_ok).sum()) if len(df) else 0}
    if not n:
        return {**row, **{c: float("nan") for c in SUMMARY_COLS
                          if c not in row}}
    cur, early = both["cur_pnl_pct"], both["early_pnl_pct"]
    eo = both["early_outcome"]
    row.update({
        "cur_mean_pnl": float(cur.mean()), "early_mean_pnl": float(early.mean()),
        "diff": float(early.mean() - cur.mean()),
        "cur_win_rate": float((cur > 0).mean()),
        "early_win_rate": float((early > 0).mean()),
        # **建値まで引き上げた撤退は損益ちょうど 0 で、勝率に入らない。**
        # 勝率だけ見ると負けに見えるので、同じ行で割合を出しておく（§16.4）
        "early_breakeven_rate": float(np.isclose(
            early.to_numpy(dtype=float), 0.0, atol=1e-9).mean()),
        "cur_exit_day_mean": float(both["cur_exit_day"].astype(float).mean()),
        "early_exit_day_mean": float(both["early_exit_day"].astype(float).mean()),
        "t1_below_rate": float(both["t1_below_entry"].fillna(False).astype(bool).mean()),
        "early_rate": _rate(eo, OUTCOME_EARLY, n),
        "target_rate": _rate(eo, OUTCOME_TARGET, n),
        "stop_rate": _rate(eo, OUTCOME_STOP, n),
        "timeout_rate": _rate(eo, OUTCOME_TIMEOUT, n),
    })
    return row


def by_quantile(table: pd.DataFrame, window: str,
                excluded: str = EXCLUDED_QUANTILE) -> pd.DataFrame:
    """窓別・抜け幅分位別（§16.4）。**Q5 は出さない**（§16.3）。

    群は固定で、その窓に 0 件でも行を落とさない。最後に「Q1〜Q4 計」を足す。
    """
    labels = [f"Q{i}" for i in range(1, 6) if f"Q{i}" != excluded]
    rows = [summarize(window, q, table[table["q"] == q] if len(table) else table)
            for q in labels]
    kept = table[table["q"].isin(labels)] if len(table) else table
    rows.append(summarize(window, "Q1〜Q4 計", kept))
    return pd.DataFrame(rows)[SUMMARY_COLS]


def excluded_count(table: pd.DataFrame, excluded: str = EXCLUDED_QUANTILE) -> dict:
    """集計から外した行を数える（§16.3）。**黙って外さない。**"""
    if not len(table):
        return {"n_excluded": 0, "n_no_quantile": 0}
    q = table["q"].fillna("").astype(str)
    return {"n_excluded": int((q == excluded).sum()),
            "n_no_quantile": int((q == "").sum())}


def early_path(out_dir: Path, window: str) -> Path:
    """行ごとの表。**`data/pattern_exit/` には書かない**（§16.1）。"""
    return Path(out_dir) / f"pattern_early_{window}.csv.gz"


def summary_path(out_dir: Path, window: str) -> Path:
    return Path(out_dir) / f"pattern_early_summary_{window}.csv"


def format_table(summary: pd.DataFrame) -> List[str]:
    """ログに出す行（表のみ。解釈はしない）。"""
    head = (f"{'窓':<10}{'区分':<12}{'件数':>8}{'現行':>9}{'本案':>9}{'差':>8}"
            f"{'現行勝率':>9}{'本案勝率':>9}{'本案建値':>9}{'現行日数':>9}"
            f"{'本案日数':>9}{'T+1マイナス':>12}")
    out = [head, "-" * len(head)]

    def num(v, nd=2, suffix=""):
        return "—" if v is None or pd.isna(v) else f"{float(v):+.{nd}f}{suffix}"

    def pct(v):
        return "—" if v is None or pd.isna(v) else f"{float(v) * 100:.1f}%"

    for _, r in summary.iterrows():
        out.append(f"{str(r['window']):<10}{str(r['q']):<12}{int(r['n']):>8,d}"
                   f"{num(r['cur_mean_pnl'], 2, '%'):>9}"
                   f"{num(r['early_mean_pnl'], 2, '%'):>9}"
                   f"{num(r['diff']):>8}"
                   f"{pct(r['cur_win_rate']):>9}{pct(r['early_win_rate']):>9}"
                   f"{pct(r['early_breakeven_rate']):>9}"
                   f"{num(r['cur_exit_day_mean'], 1, ''):>9}"
                   f"{num(r['early_exit_day_mean'], 1, ''):>9}"
                   f"{pct(r['t1_below_rate']):>12}")
    return out
