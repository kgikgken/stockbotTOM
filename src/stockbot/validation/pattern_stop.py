"""撤退基準の検証（docs/BACKTEST.md §11）。**これが最後の探索**（設計責任者・2026-09-13）。

現行の撤退ライン（パターンの最安値）を、次の 3 案と比べる。**検定 3 件。**

- **S1**: 直近の谷（パターンの**最後の**谷）
- **S2**: ネックライン（成立時に抜けた線。保ち合い系は上値抵抗線）
- **S3**: 下値支持線（**C1・C2 のみ**。反転系は該当なしとして除外）

**固定するもの**（§11 の事前登録）。

- **利確は測定目標のみ。** §10 の ATR 基準は使わない
- 評価窓 **T+1..T+20**、エントリー **Open[T+1]**
- **同日に撤退と目標なら撤退**（保守的）

**検定は既存 11 件 + 3 件 = 14 件。** これ以上増やさない。

**判定式・閾値は変更しない。** 撤退の当て方を変えて成績を見るだけで、検出
（`features/pattern.py`）には触らない。撤退ラインの値は**再生のときに保存済みの
極値**（`l1..l3` とその位置）から引き直す —— **T の引けまでの値だけ**でできる。

**未来参照はしない。** 新しく読むのは T+1..T+20 の四本値だけ（§10 と同じ範囲）。

**数値の解釈も採否の判断もしない。表を作るまで**（CLAUDE.md）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..features.pattern import ASCENDING_BOX, ASCENDING_TRIANGLE
from .layer1 import newey_west_mean_t
from .pattern_exit import OUTCOME_STOP, OUTCOME_TARGET, OUTCOME_TIMEOUT, ohlc_arrays, simulate_exit
from .pattern_replay import HORIZON
from .replay import _date_position

# 検定数（docs/BACKTEST.md §11）。**既存 11 件 + 3 件 = 14 件。増やさない。**
N_TESTS_HERE = 3
N_TESTS_TOTAL = 11 + N_TESTS_HERE

CURRENT = "current"
STOP_VARIANTS = (CURRENT, "s1", "s2", "s3")
STOP_LABELS = {
    CURRENT: "参考: 現行（パターン最安値）",
    "s1": "S1 直近の谷（検定12）",
    "s2": "S2 ネックライン（検定13）",
    "s3": "S3 下値支持線・C1C2のみ（検定14）",
}

# S3 を引けるパターン（docs/BACKTEST.md §11）。**反転系は該当なしとして除外する** ——
# 反転系のネックラインは水平で、下辺という概念が判定式に無い
SUPPORT_PATTERNS = (ASCENDING_TRIANGLE, ASCENDING_BOX)

TROUGH_KEYS = (("l1_pos", "l1"), ("l2_pos", "l2"), ("l3_pos", "l3"))

STOP_COLS = (["date", "ticker", "pattern", "close_t", "entry_open", "target",
              "n_bars", "censored", "already_at_target"]
             + [f"{v}_{s}" for v in STOP_VARIANTS
                for s in ("stop", "gap_pct", "outcome", "exit_day", "exit_price",
                          "pnl_pct")])


def _f(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def troughs(row: dict) -> list:
    """保存済みの谷を (位置, 値) の並びにする。**欠損は飛ばす。**

    `_row`（`features/pattern.py`）が位置の昇順で書いているので並べ替えない。
    """
    out = []
    for pos_key, val_key in TROUGH_KEYS:
        pos, val = _f(row.get(pos_key)), _f(row.get(val_key))
        if pos is None or val is None:
            continue
        out.append((pos, val))
    return out


def last_trough(row: dict) -> float:
    """**S1: 直近の谷** —— パターンの最後の谷（位置が最大のもの）。"""
    pts = troughs(row)
    if not pts:
        return float("nan")
    return float(max(pts, key=lambda pv: pv[0])[1])


def support_line(row: dict) -> float:
    """**S3: 下値支持線**（C1・C2 のみ。反転系は NaN）。

    **検出側と同じ引き方にそろえる**（`PATTERN.md` §2.2 実装上の読み 3）。

    - **C1 上昇三角**: 谷のタッチ点に当てた回帰直線を T まで延ばした値
      （上辺と同じ `_line_at` の引き方）
    - **C2 上昇ボックス**: 谷の**平均**。ボックスだけは判定式が傾きではなく
      「平均からの距離（±0.75%）」で水平さを要求しているので、上辺と同じく平均で引く

    **新しいパラメータではない。** 既にある判定式の線を、そのまま下辺に当てただけ。
    """
    pattern = str(row.get("pattern") or "")
    if pattern not in SUPPORT_PATTERNS:
        return float("nan")
    pts = troughs(row)
    if len(pts) < 2:
        return float("nan")
    vals = np.asarray([v for _p, v in pts], dtype=float)
    if pattern == ASCENDING_BOX:
        return float(vals.mean())
    t_pos = _f(row.get("t_pos"))
    if t_pos is None:
        return float("nan")
    xs = np.asarray([p for p, _v in pts], dtype=float)
    slope, intercept = (float(v) for v in np.polyfit(xs, vals, 1))
    return float(intercept + slope * t_pos)


def stop_levels(row: dict) -> dict:
    """4 つの撤退ライン。**現行は保存済みの `pattern_low` をそのまま使う。**"""
    return {CURRENT: _f(row.get("pattern_low")) or float("nan"),
            "s1": last_trough(row),
            "s2": _f(row.get("neckline")) or float("nan"),
            "s3": support_line(row)}


def stop_one(df: pd.DataFrame, t_pos: int, row: dict, horizon: int = HORIZON,
             arrays: Optional[tuple] = None) -> dict:
    """1 行ぶん。**ここだけが T+1 以降を見る**（§10 と同じ範囲）。

    利確は**測定目標のみ**で、定義は既存の `reached_target` と同じ `High > 目標`。
    撤退は `Low < 撤退ライン`（既存の `broke_stop` と同じ）。同日なら撤退。
    """
    out: dict = {c: np.nan for c in STOP_COLS
                 if c not in ("date", "ticker", "pattern")}
    for v in STOP_VARIANTS:
        out[f"{v}_outcome"] = ""
        out[f"{v}_exit_day"] = pd.NA
    out.update({"censored": True, "n_bars": 0,
                "already_at_target": row.get("already_at_target")})

    close_t = _f(row.get("close_t"))
    target = _f(row.get("target"))
    out["close_t"] = close_t if close_t is not None else np.nan
    out["target"] = target if target is not None else np.nan

    levels = stop_levels(row)
    for v, stop in levels.items():
        out[f"{v}_stop"] = stop
        # **撤退までの距離は終値[T] 基準**（カードの `down_pct` と同じ基準。§11）。
        # 寄り付きのギャップを混ぜず、仕掛けの形そのものの深さを表す
        out[f"{v}_gap_pct"] = (np.nan if (close_t is None or not np.isfinite(stop)
                                          or close_t <= 0)
                               else (stop / close_t - 1.0) * 100.0)

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

    for v, stop in levels.items():
        # **撤退ラインを引けない案は空のまま。** 「撤退が無い」ではなく「その案が
        # この行に当てはまらない」なので、時間切れにも到達にも数えない（S3 の反転系）
        if not np.isfinite(stop):
            continue
        res = simulate_exit(h, lo, c, start, end, entry, stop,
                            target if target is not None else np.nan,
                            target_strict=True)
        out[f"{v}_outcome"] = res["outcome"]
        out[f"{v}_exit_day"] = res["exit_day"]
        out[f"{v}_exit_price"] = res["exit_price"]
        out[f"{v}_pnl_pct"] = res["pnl_pct"]
    return out


def run(replay: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
        horizon: int = HORIZON, log=print) -> pd.DataFrame:
    """保存済みの再生結果に 4 つの撤退基準を当てる。**検出はやり直さない。**

    `ohlcv` は**ホールドアウトを打ち切った後**のものを渡すこと（呼び出し側の責任）。
    """
    if len(replay) == 0:
        return pd.DataFrame(columns=STOP_COLS)
    if "l1_pos" not in replay.columns:
        # **黙って NaN にしない。** 古い世代の再生結果では S1・S3 が引けない
        log("[pattern-stop] 再生結果に極値の列が無い（l1_pos）。"
            "cli pattern-replay を新しい世代で回し直す必要がある")
        return pd.DataFrame(columns=STOP_COLS)
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
                     **stop_one(df, t_pos, r.to_dict(), horizon, cache[ticker])})
    if n_missing:
        log(f"[pattern-stop] 四本値が引けずに落ちた行: {n_missing}")
    if not rows:
        return pd.DataFrame(columns=STOP_COLS)
    return pd.DataFrame(rows)[STOP_COLS]


def check_positions(replay: pd.DataFrame, table: pd.DataFrame) -> dict:
    """引き直した撤退ラインの整合性（docs/BACKTEST.md §11.3）。

    - **S1 は必ず現行（最安値）以上**である —— 谷の 1 つと谷の最小値だから
    - **S2 は必ず終値[T] より下**である —— 成立は終値が抜けた行だけだから

    **破れた行数をそのまま報告する**（隠さない）。
    """
    if not len(table):
        return {"s1_below_low": 0, "s2_above_close": 0, "n": 0}
    s1, cur = table["s1_stop"].to_numpy(float), table["current_stop"].to_numpy(float)
    s2, close = table["s2_stop"].to_numpy(float), table["close_t"].to_numpy(float)
    return {"n": int(len(table)),
            "s1_below_low": int(np.sum(np.isfinite(s1) & np.isfinite(cur)
                                       & (s1 < cur - 1e-9))),
            "s2_above_close": int(np.sum(np.isfinite(s2) & np.isfinite(close)
                                         & (s2 > close + 1e-9)))}


STOP_GROUP_COLS = ["group", "n", "n_days", "mean_pnl", "nw_t", "nw_p",
                   "stop_rate", "target_rate", "timeout_rate", "win_rate",
                   "gap_median", "exit_day_median", "n_no_stop"]


def summarize_stop(name: str, df: pd.DataFrame, variant: str,
                   lag: int = HORIZON) -> dict:
    """1 案ぶんの行（docs/BACKTEST.md §11）。**件数（母数）を必ず持つ。**

    NW t は**日次平均系列**に当てる（D-4 と同じ理由）。撤退ラインを引けない行
    （S3 の反転系など）は `n_no_stop` に出して平均から外す —— 0 で埋めない。
    """
    pnl, oc = f"{variant}_pnl_pct", f"{variant}_outcome"
    have = (df[df[f"{variant}_stop"].notna()] if len(df) and f"{variant}_stop" in df
            else df)
    part = (have[["date", pnl]].dropna() if len(have) and pnl in have.columns
            else pd.DataFrame(columns=["date", pnl]))
    daily = (part.groupby("date")[pnl].mean().sort_index() if len(part)
             else pd.Series(dtype=float))
    nw = newey_west_mean_t(daily.to_numpy(dtype=float), lag=lag)
    n = int(len(have))
    outcomes = have[oc] if n and oc in have.columns else pd.Series(dtype=object)
    v = part[pnl] if len(part) else pd.Series(dtype=float)
    gap = (have[f"{variant}_gap_pct"].dropna() if n else pd.Series(dtype=float))
    day = (have[f"{variant}_exit_day"].dropna() if n else pd.Series(dtype=float))

    def rate(kind):
        return float((outcomes == kind).sum()) / n if n else float("nan")

    return {"group": name, "n": n, "n_days": int(len(daily)),
            "mean_pnl": float(v.mean()) if len(v) else float("nan"),
            "nw_t": nw["t"], "nw_p": nw["p"],
            "stop_rate": rate(OUTCOME_STOP),
            "target_rate": rate(OUTCOME_TARGET),
            "timeout_rate": rate(OUTCOME_TIMEOUT),
            "win_rate": float((v > 0).mean()) if len(v) else float("nan"),
            # **撤退までの距離の中央値（%）。終値[T] 基準**（§11）
            "gap_median": float(gap.median()) if len(gap) else float("nan"),
            "exit_day_median": float(day.median()) if len(day) else float("nan"),
            # **撤退ラインを引けなかった行**（母数から黙って消さない）
            "n_no_stop": (int(len(df)) - n) if len(df) else 0}


def by_variant(table: pd.DataFrame, lag: int = HORIZON) -> pd.DataFrame:
    """検定 12〜14 と参考（現行）。**群は固定。0 件でも行を落とさない。**"""
    rows = [summarize_stop(STOP_LABELS[CURRENT], table, CURRENT, lag)]
    rows += [summarize_stop(STOP_LABELS[v], table, v, lag)
             for v in STOP_VARIANTS if v != CURRENT]
    return pd.DataFrame(rows)[STOP_GROUP_COLS]


def format_stop_table(table: pd.DataFrame) -> list[str]:
    """ログに出す行（表のみ。解釈はしない）。"""
    out = [f"{'撤退基準':<32}{'件数':>8}{'日数':>7}{'平均損益':>10}{'NW t':>8}"
           f"{'撤退率':>8}{'到達率':>8}{'時間切れ':>9}{'勝率':>8}"
           f"{'撤退距離':>10}{'日数中央':>9}{'引けず':>8}"]
    for _i, r in table.iterrows():
        def pct(v):
            return "—" if pd.isna(v) else f"{float(v) * 100:.1f}%"

        def num(v, d=2, sign="+"):
            return "—" if pd.isna(v) else f"{float(v):{sign}.{d}f}"

        out.append(f"{str(r['group']):<32}{int(r['n']):>8}{int(r['n_days']):>7}"
                   f"{num(r['mean_pnl']) + '%':>10}{num(r['nw_t']):>8}"
                   f"{pct(r['stop_rate']):>8}{pct(r['target_rate']):>8}"
                   f"{pct(r['timeout_rate']):>9}{pct(r['win_rate']):>8}"
                   f"{num(r['gap_median']) + '%':>10}"
                   f"{num(r['exit_day_median'], 1, ''):>9}{int(r['n_no_stop']):>8}")
    return out
