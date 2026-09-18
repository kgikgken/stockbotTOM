"""測定目標の到達率係数補正（docs/BACKTEST.md §12）。**ホールドアウトで 1 回だけ。**

**探索ではない。文献値の 1 回検証である**（設計責任者・2026-09-18）。§11.7 の
「これを最後の探索とする」と矛盾しない —— **探索窓を使わない**ため。

出典は Bulkowski の measure rule: **高さ×1.0 は機能せず、到達率の係数を掛けるべき**と
明言されている。**係数は文献値をそのまま使う。当てはめも最適化もしない。**

    利確値 = ネックライン + 高さ × 係数

**固定するもの**（§12 の事前登録）。

- 撤退は**現行（パターン内最安値）**
- 評価窓 **T+1..T+20**、エントリー **Open[T+1]**
- **同日に撤退と目標なら撤退**

**検定は既存 14 件 + 1 件 = 15 件。** これ以上増やさない。

**判定式・閾値は変更しない。** 利確値の作り方を変えて成績を見るだけで、検出
（`features/pattern.py`）には触らない。利確値は**再生のときに保存済みの
`neckline` と `height`** から引き直す —— **T の引けまでの値だけ**でできる。

**未来参照はしない。** 新しく読むのは T+1..T+20 の四本値だけ（§10・§11 と同じ範囲）。

**数値の解釈も採否の判断もしない。表を作るまで**（CLAUDE.md）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..features.pattern import (
    ASCENDING_BOX, ASCENDING_TRIANGLE, DOUBLE_BOTTOM, INVERSE_HS, TRIPLE_BOTTOM,
)
from .layer1 import newey_west_mean_t
from .pattern_exit import (
    OUTCOME_STOP, OUTCOME_TARGET, OUTCOME_TIMEOUT, ohlc_arrays, simulate_exit,
)
from .pattern_replay import HORIZON
from .replay import _date_position

# 検定数（docs/BACKTEST.md §12）。**既存 14 件 + 1 件 = 15 件。増やさない。**
N_TESTS_HERE = 1
N_TESTS_TOTAL = 14 + N_TESTS_HERE

# **到達率の係数（文献値・変更禁止）。** 出典は Bulkowski の measure rule。
# **振って比較しない**（§12 の禁止事項）。C3・C4 は係数が与えられていないので対象外
# （日足かつ k=3 では検出 0 件。PATTERN.md D-10）
TARGET_RATIO: Dict[str, float] = {
    DOUBLE_BOTTOM: 0.73,
    TRIPLE_BOTTOM: 0.74,
    INVERSE_HS: 0.71,
    ASCENDING_TRIANGLE: 0.70,
    ASCENDING_BOX: 0.78,
}

BASE = "base"     # 現行（高さ×1.0）
RATIO = "ratio"   # 係数補正
VARIANTS = (BASE, RATIO)
VARIANT_LABELS = {
    BASE: "現行（高さ×1.0）",
    RATIO: "係数補正（検定15）",
}

TARGET_COLS = (["date", "ticker", "pattern", "close_t", "neckline", "height",
                "pattern_low", "entry_open", "n_bars", "censored", "ratio",
                "already_at_target"]
               + [f"{v}_{s}" for v in VARIANTS
                  for s in ("target", "gap_pct", "outcome", "exit_day", "exit_price",
                            "pnl_pct")])


def _f(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def ratio_for(pattern: str) -> float:
    """そのパターンの係数（文献値）。与えられていないパターンは NaN。"""
    return TARGET_RATIO.get(str(pattern), float("nan"))


def targets(row: dict) -> dict:
    """2 つの利確値。**どちらも保存済みの `neckline` と `height` から引く。**

    - `base`: 現行の測定目標（`neckline + height`）。保存済みの `target` と一致する
    - `ratio`: `neckline + height × 係数`（文献値）
    """
    neck, height = _f(row.get("neckline")), _f(row.get("height"))
    if neck is None or height is None or height <= 0:
        return {BASE: float("nan"), RATIO: float("nan")}
    r = ratio_for(row.get("pattern"))
    return {BASE: neck + height,
            RATIO: (neck + height * r) if np.isfinite(r) else float("nan")}


def target_one(df: pd.DataFrame, t_pos: int, row: dict, horizon: int = HORIZON,
               arrays: Optional[tuple] = None) -> dict:
    """1 行ぶん。**ここだけが T+1 以降を見る**（§10・§11 と同じ範囲）。

    利確の到達は `High > 利確値`（**既存の `reached_target` と同じ定義**）、撤退は
    `Low < パターン最安値`（**既存の `broke_stop` と同じ**）。同日なら撤退。
    """
    out: dict = {c: np.nan for c in TARGET_COLS
                 if c not in ("date", "ticker", "pattern")}
    for v in VARIANTS:
        out[f"{v}_outcome"] = ""
        out[f"{v}_exit_day"] = pd.NA
    out.update({"censored": True, "n_bars": 0,
                "already_at_target": row.get("already_at_target")})

    close_t = _f(row.get("close_t"))
    stop = _f(row.get("pattern_low"))
    out.update({"close_t": close_t if close_t is not None else np.nan,
                "neckline": _f(row.get("neckline")) or np.nan,
                "height": _f(row.get("height")) or np.nan,
                "pattern_low": stop if stop is not None else np.nan,
                "ratio": ratio_for(row.get("pattern"))})

    levels = targets(row)
    for v, tgt in levels.items():
        out[f"{v}_target"] = tgt
        # 利確までの距離は**終値[T] 基準**（§11 の撤退距離と同じ基準）
        out[f"{v}_gap_pct"] = (np.nan if (close_t is None or not np.isfinite(tgt)
                                          or close_t <= 0)
                               else (tgt / close_t - 1.0) * 100.0)

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

    for v, tgt in levels.items():
        # **利確値を引けない案は空のまま。** 係数が与えられていないパターン（C3・C4）は
        # 「利確が無い」のではなく「この案が当てはまらない」
        if not np.isfinite(tgt):
            continue
        res = simulate_exit(h, lo, c, start, end, entry,
                            stop if stop is not None else np.nan, tgt,
                            target_strict=True)
        out[f"{v}_outcome"] = res["outcome"]
        out[f"{v}_exit_day"] = res["exit_day"]
        out[f"{v}_exit_price"] = res["exit_price"]
        out[f"{v}_pnl_pct"] = res["pnl_pct"]
    return out


def run(replay: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
        horizon: int = HORIZON, log=print) -> pd.DataFrame:
    """保存済みの再生結果に 2 つの利確値を当てる。**検出はやり直さない。**

    `ohlcv` は**ホールドアウトを含めたもの**を渡すこと（§12 はホールドアウトで
    1 回だけ走る。呼び出し側が `--include-holdout` を明示する）。
    """
    if len(replay) == 0:
        return pd.DataFrame(columns=TARGET_COLS)
    rows, n_missing, cache = [], 0, {}
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
                     **target_one(df, t_pos, r.to_dict(), horizon, cache[ticker])})
    if n_missing:
        log(f"[pattern-target] 四本値が引けずに落ちた行: {n_missing}")
    if not rows:
        return pd.DataFrame(columns=TARGET_COLS)
    return pd.DataFrame(rows)[TARGET_COLS]


def check_base_matches_saved(replay: pd.DataFrame, table: pd.DataFrame,
                             tol: float = 1e-6) -> dict:
    """`base` が保存済みの `target` と一致することを確かめる（§12.3）。

    一致しなければ、利確値の引き直しがどこかで間違っている。**件数をそのまま返す。**
    """
    if not len(table) or "target" not in replay.columns:
        return {"n": 0, "n_mismatch": 0}
    key = ["date", "ticker", "pattern"]
    a = replay[key + ["target"]].copy()
    a["date"] = pd.to_datetime(a["date"])
    m = table[key + ["base_target"]].merge(a, on=key, how="left")
    both = m[m["base_target"].notna() & m["target"].notna()]
    diff = (both["base_target"] - both["target"]).abs()
    return {"n": int(len(both)), "n_mismatch": int((diff > tol).sum())}


GROUP_COLS = ["group", "n", "n_days", "mean_pnl", "nw_t", "nw_p",
              "target_rate", "stop_rate", "timeout_rate", "win_rate",
              "gap_median", "exit_day_median", "n_no_target"]


def summarize_target(name: str, df: pd.DataFrame, variant: str,
                     lag: int = HORIZON) -> dict:
    """1 案ぶんの行（docs/BACKTEST.md §12）。**件数（母数）を必ず持つ。**

    NW t は**日次平均系列**に当てる（D-4 と同じ理由）。**ただし §12 では t に基準を
    設けない** —— ホールドアウトは 122 営業日で有意性検定に足りないため（§12.4）。
    """
    pnl, oc = f"{variant}_pnl_pct", f"{variant}_outcome"
    have = (df[df[f"{variant}_target"].notna()]
            if len(df) and f"{variant}_target" in df else df)
    part = (have[["date", pnl]].dropna() if len(have) and pnl in have.columns
            else pd.DataFrame(columns=["date", pnl]))
    daily = (part.groupby("date")[pnl].mean().sort_index() if len(part)
             else pd.Series(dtype=float))
    nw = newey_west_mean_t(daily.to_numpy(dtype=float), lag=lag)
    n = int(len(have))
    outcomes = have[oc] if n and oc in have.columns else pd.Series(dtype=object)
    v = part[pnl] if len(part) else pd.Series(dtype=float)
    gap = have[f"{variant}_gap_pct"].dropna() if n else pd.Series(dtype=float)
    day = have[f"{variant}_exit_day"].dropna() if n else pd.Series(dtype=float)

    def rate(kind):
        return float((outcomes == kind).sum()) / n if n else float("nan")

    return {"group": name, "n": n, "n_days": int(len(daily)),
            "mean_pnl": float(v.mean()) if len(v) else float("nan"),
            "nw_t": nw["t"], "nw_p": nw["p"],
            "target_rate": rate(OUTCOME_TARGET),
            "stop_rate": rate(OUTCOME_STOP),
            "timeout_rate": rate(OUTCOME_TIMEOUT),
            "win_rate": float((v > 0).mean()) if len(v) else float("nan"),
            "gap_median": float(gap.median()) if len(gap) else float("nan"),
            "exit_day_median": float(day.median()) if len(day) else float("nan"),
            "n_no_target": (int(len(df)) - n) if len(df) else 0}


def by_variant(table: pd.DataFrame, lag: int = HORIZON) -> pd.DataFrame:
    """現行と係数補正の 2 行。**群は固定。0 件でも行を落とさない。**"""
    rows = [summarize_target(VARIANT_LABELS[v], table, v, lag) for v in VARIANTS]
    return pd.DataFrame(rows)[GROUP_COLS]


def format_target_table(table: pd.DataFrame) -> list[str]:
    """ログに出す行（表のみ。解釈はしない）。"""
    out = [f"{'利確の作り方':<26}{'件数':>8}{'日数':>7}{'平均損益':>10}{'NW t':>8}"
           f"{'到達率':>8}{'撤退率':>8}{'時間切れ':>9}{'勝率':>8}"
           f"{'利確距離':>10}{'日数中央':>9}{'引けず':>8}"]
    for _i, r in table.iterrows():
        def pct(v):
            return "—" if pd.isna(v) else f"{float(v) * 100:.1f}%"

        def num(v, d=2, sign="+"):
            return "—" if pd.isna(v) else f"{float(v):{sign}.{d}f}"

        out.append(f"{str(r['group']):<26}{int(r['n']):>8}{int(r['n_days']):>7}"
                   f"{num(r['mean_pnl']) + '%':>10}{num(r['nw_t']):>8}"
                   f"{pct(r['target_rate']):>8}{pct(r['stop_rate']):>8}"
                   f"{pct(r['timeout_rate']):>9}{pct(r['win_rate']):>8}"
                   f"{num(r['gap_median']) + '%':>10}"
                   f"{num(r['exit_day_median'], 1, ''):>9}{int(r['n_no_target']):>8}")
    return out
