"""既存 4 案の株価帯別・年別の記述統計（docs/BACKTEST.md §13）。

**記述統計であり判定ではない**（設計責任者・2026-09-18）。判定基準を設けず、採用も
不採用も決めない。**検定数は 15 件のまま増やさない** —— ここで作る表は 1 件も検定に
数えない。見つかった傾向は「仮説」として記録し、実運用の記録で確かめる。

**検出はやり直さない。** 保存済みの再生結果（`pattern_replay.load_table`）に、既存の
3 モジュールの当て方をそのまま適用する。

| 案 | 群 | 出どころ |
| --- | --- | --- |
| 抜け幅 5 分位（§9） | Q1〜Q5 | `pattern_report` の固定境界。出口は現行基準 |
| 利確 ATR×2.0（§10） | ATR×2.0 | `pattern_exit.exit_one` の `atr2` |
| 撤退 3 案（§11） | S1・S2・S3 | `pattern_stop.stop_one` |
| 係数補正（§12） | 係数補正 | `pattern_target.target_one` の `ratio` |

**現行基準は 3 経路から引ける**（§10 の `tgt`・§11 の `current`・§12 の `base`）。
**3 つが一致することを数えて出す** —— ずれていれば引き直しがどこかで間違っている。

**未来参照はしない。** 新しく読むのは T+1..T+20 の四本値だけ（§10・§11・§12 と同じ
範囲）。株価帯は成立時の終値 `close_t`、年は判定日 `date` で、どちらも T の値である。

**数値の解釈も採否の判断もしない。表を作るまで**（CLAUDE.md）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .layer1 import newey_west_mean_t
from .pattern_exit import (ATR_MULT, OUTCOME_STOP, OUTCOME_TARGET, OUTCOME_TIMEOUT,
                           exit_one, ohlc_arrays)
from .pattern_replay import HORIZON
from .pattern_report import N_QUANTILES
from .pattern_stop import stop_one
from .pattern_target import target_one
from .replay import _date_position

# **検定は増やさない**（docs/BACKTEST.md §13.1）。ここは記述統計で、判定に使わない。
# §12 までで 15 件。この数字はこのモジュールでは動かさない
N_TESTS_TOTAL = 15
N_TESTS_HERE = 0

# 株価帯の分位数（§13.2）。**3 分位。振らない**
N_PRICE_BANDS = 3

# 現行基準と 5 つの案。**現行は比較の基準**であって案ではない（§13.3）
CUR = "cur"
VARIANTS = (CUR, "atr2", "s1", "s2", "s3", "ratio")
VARIANT_LABELS = {
    CUR: "現行（測定目標・最安値）",
    "atr2": f"利確 ATR×{ATR_MULT:.1f}",
    "s1": "S1 直近の谷",
    "s2": "S2 ネックライン",
    "s3": "S3 下値支持線（C1C2）",
    "ratio": "係数補正",
}

# 案（§13.3）。**群の並びは固定。0 件でも行を落とさない**
PLAN_BREAKOUT = "breakout"
PLAN_ATR = "atr"
PLAN_STOP = "stop"
PLAN_TARGET = "target"
PLAN_LABELS = {
    PLAN_BREAKOUT: "抜け幅 5 分位（§9）",
    PLAN_ATR: "利確 ATR×2.0（§10）",
    PLAN_STOP: "撤退 3 案（§11）",
    PLAN_TARGET: "係数補正（§12）",
}
PLAN_VARIANTS = {
    PLAN_ATR: (CUR, "atr2"),
    PLAN_STOP: (CUR, "s1", "s2", "s3"),
    PLAN_TARGET: (CUR, "ratio"),
}

AXIS_BAND = "band"
AXIS_YEAR = "year"

STRATA_COLS = (["date", "ticker", "pattern", "close_t", "year", "band",
                "breakout_h", "r_20", "n_bars", "censored", "already_at_target",
                "cur_sources_agree"]
               + [f"{v}_{s}" for v in VARIANTS
                  for s in ("outcome", "exit_day", "pnl_pct")])

SUMMARY_COLS = ["window", "axis", "bucket", "plan", "group", "n", "n_days",
                "mean_pnl", "nw_t", "nw_p", "target_rate", "stop_rate",
                "timeout_rate", "win_rate", "mean_r20", "n_na"]

# 探索窓で作った株価帯の境界を固定したファイル（§13.2）。**確認窓とホールドアウトは
# これをそのまま使う。** 窓ごとに切り直すと「低位」が窓ごとに別の値段を指す。
# Actions の cache は窓ごとに分かれていて期限もあるので、**リポジトリに置いて
# 確定値にする**（§4.2 の抜け幅 5 分位と同じ扱い）
PRICE_BAND_EDGES_PATH = Path("data/reference/pattern_price_band_edges.json")


def _f(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ------------------------------------------------------------------ 株価帯の境界
def price_band_edges(df: pd.DataFrame, value: str = "close_t",
                     q: int = N_PRICE_BANDS) -> Optional[np.ndarray]:
    """株価帯の境界を作る（**探索窓で作り、他の窓にはこれをそのまま当てる**・§13.2）。"""
    if value not in df.columns:
        return None
    x = df[value].dropna().to_numpy(dtype=float)
    if len(x) < q:
        return None
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def load_frozen_bands(path: Optional[Path] = None) -> Optional[np.ndarray]:
    """固定した株価帯の境界を読む。無ければ None。

    JSON の `edges` は両端が null（±inf）の `N_PRICE_BANDS + 1` 要素。
    """
    path = Path(path) if path is not None else PRICE_BAND_EDGES_PATH
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["edges"]
    except (OSError, ValueError, KeyError):
        return None
    if not isinstance(raw, list) or len(raw) != N_PRICE_BANDS + 1:
        return None
    out = []
    for i, v in enumerate(raw):
        if v is None:
            out.append(-np.inf if i == 0 else np.inf)
        else:
            out.append(float(v))
    return np.asarray(out, dtype=float)


def save_frozen_bands(edges: np.ndarray, meta: Optional[dict] = None,
                      path: Optional[Path] = None) -> Path:
    """境界をリポジトリに固定する（**探索窓で 1 回だけ作る**）。"""
    path = Path(path) if path is not None else PRICE_BAND_EDGES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(meta or {})
    body["value"] = "close_t"
    body["edges"] = [None if not np.isfinite(e) else float(e) for e in edges]
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def band_labels(edges: np.ndarray) -> List[str]:
    """帯の名前。**境界の値をそのまま名前に入れる**（あとで読み直せるように）。"""
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if not np.isfinite(lo):
            out.append(f"低位（〜{hi:,.0f}円）")
        elif not np.isfinite(hi):
            out.append(f"高位（{lo:,.0f}円〜）")
        else:
            out.append(f"中位（{lo:,.0f}〜{hi:,.0f}円）")
    return out


def assign_bucket(x: np.ndarray, edges: np.ndarray, labels: List[str]) -> np.ndarray:
    """値を帯に割り当てる。**右端の帯だけ上端を含める**（`by_breakout_quantile` と同じ）。"""
    out = np.full(len(x), "", dtype=object)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & ((x < hi) if i < len(edges) - 2 else (x <= hi))
        out[mask] = labels[i]
    return out


# ------------------------------------------------------------------ 1 行ぶん
def strata_one(df: pd.DataFrame, t_pos: int, row: dict, horizon: int = HORIZON,
               arrays: Optional[tuple] = None) -> dict:
    """1 行ぶん。**既存 3 モジュールの当て方をそのまま呼ぶ**（式を書き直さない）。

    現行基準は 3 経路から出る。**一致しなければ `cur_sources_agree` を False にする**
    —— 黙って片方を採らない（§13.5）。
    """
    arrays = arrays if arrays is not None else ohlc_arrays(df)
    e = exit_one(df, t_pos, row, ATR_MULT, horizon, arrays)
    s = stop_one(df, t_pos, row, horizon, arrays)
    t = target_one(df, t_pos, row, horizon, arrays)

    out: dict = {"close_t": e.get("close_t", np.nan),
                 "n_bars": e.get("n_bars", 0), "censored": e.get("censored", True),
                 "already_at_target": row.get("already_at_target"),
                 "breakout_h": _f(row.get("breakout_h")) or np.nan,
                 "r_20": _f(row.get("r_20")) or np.nan}

    src = {CUR: ("tgt", e), "atr2": ("atr2", e), "s1": ("s1", s), "s2": ("s2", s),
           "s3": ("s3", s), "ratio": ("ratio", t)}
    for v, (prefix, res) in src.items():
        out[f"{v}_outcome"] = res.get(f"{prefix}_outcome", "")
        out[f"{v}_exit_day"] = res.get(f"{prefix}_exit_day", pd.NA)
        out[f"{v}_pnl_pct"] = res.get(f"{prefix}_pnl_pct", np.nan)

    # 現行基準の 3 経路（§10 tgt / §11 current / §12 base）がそろっているか
    trio = [e.get("tgt_pnl_pct"), s.get("current_pnl_pct"), t.get("base_pnl_pct")]
    vals = [float(x) for x in trio if _f(x) is not None]
    out["cur_sources_agree"] = bool(
        len(vals) == len(trio) and max(vals) - min(vals) <= 1e-6) if any(
        _f(x) is not None for x in trio) else True
    return out


def run(replay: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame], edges: np.ndarray,
        horizon: int = HORIZON, log=print) -> pd.DataFrame:
    """保存済みの再生結果に 5 案 + 現行を当てる。**検出はやり直さない。**

    `ohlcv` は**ホールドアウトを打ち切った後**のものを渡すこと（呼び出し側の責任。
    CLAUDE.md の絶対規則）。`edges` は固定した株価帯の境界。
    """
    if len(replay) == 0:
        return pd.DataFrame(columns=STRATA_COLS)
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
                     **strata_one(df, t_pos, r.to_dict(), horizon, cache[ticker])})
    if n_missing:
        # **黙って落とさない。** 落ちた行数はそのまま報告する
        log(f"[pattern-strata] 四本値が引けずに落ちた行: {n_missing}")
    if not rows:
        return pd.DataFrame(columns=STRATA_COLS)
    out = pd.DataFrame(rows)
    out["year"] = pd.to_datetime(out["date"]).dt.year
    labels = band_labels(edges)
    out["band"] = assign_bucket(out["close_t"].to_numpy(dtype=float), edges, labels)
    return out[STRATA_COLS]


def check_current_sources(table: pd.DataFrame) -> dict:
    """現行基準の 3 経路が一致しない行を数える（§13.5）。**0 が正常。**"""
    if not len(table) or "cur_sources_agree" not in table.columns:
        return {"n": 0, "n_mismatch": 0}
    ok = table["cur_sources_agree"].astype(bool)
    return {"n": int(len(table)), "n_mismatch": int((~ok).sum())}


# ------------------------------------------------------------------ 集計
def summarize(name: str, df: pd.DataFrame, variant: str,
              lag: int = HORIZON) -> dict:
    """1 群ぶんの行（§13.4）。**件数（母数）を必ず持つ。**

    その案が当てはまらない行（出口が空）は母数から外す —— 「撤退が無い」ではなく
    「その案がこの行に当てはまらない」ため（§11.3・§12.2 と同じ扱い）。

    NW t は**日次平均系列**に当てる（D-4 と同じ理由）。**ただし §13 では判定に
    使わない。参考として出すだけ**（§13.1・§13.4）。
    """
    oc, pnl = f"{variant}_outcome", f"{variant}_pnl_pct"
    have = (df[df[oc].fillna("").astype(str) != ""]
            if len(df) and oc in df.columns else df.iloc[:0] if len(df) else df)
    part = (have[["date", pnl]].dropna() if len(have) and pnl in have.columns
            else pd.DataFrame(columns=["date", pnl]))
    daily = (part.groupby("date")[pnl].mean().sort_index() if len(part)
             else pd.Series(dtype=float))
    nw = newey_west_mean_t(daily.to_numpy(dtype=float), lag=lag)
    n = int(len(have))
    outcomes = have[oc] if n else pd.Series(dtype=object)
    v = part[pnl] if len(part) else pd.Series(dtype=float)
    r20 = (have["r_20"].dropna() if n and "r_20" in have.columns
           else pd.Series(dtype=float))

    def rate(kind):
        return float((outcomes == kind).sum()) / n if n else float("nan")

    return {"group": name, "n": n, "n_days": int(len(daily)),
            "mean_pnl": float(v.mean()) if len(v) else float("nan"),
            "nw_t": nw["t"], "nw_p": nw["p"],
            "target_rate": rate(OUTCOME_TARGET),
            "stop_rate": rate(OUTCOME_STOP),
            "timeout_rate": rate(OUTCOME_TIMEOUT),
            "win_rate": float((v > 0).mean()) if len(v) else float("nan"),
            "mean_r20": float(r20.mean()) if len(r20) else float("nan"),
            "n_na": (int(len(df)) - n) if len(df) else 0}


def _buckets(table: pd.DataFrame, axis: str, edges: np.ndarray) -> List:
    """軸の値の並び。**群は固定**（その窓に 0 件でも落とさない）。"""
    if axis == AXIS_BAND:
        return band_labels(edges)
    years = (sorted(int(y) for y in pd.Series(table["year"]).dropna().unique())
             if len(table) else [])
    return years


def by_strata(table: pd.DataFrame, window: str, axis: str, edges: np.ndarray,
              quantile_edges: Optional[np.ndarray] = None,
              lag: int = HORIZON) -> pd.DataFrame:
    """4 案 × 軸の長い表（§13.3・§13.4）。**窓は各行に明記する。**"""
    key = "band" if axis == AXIS_BAND else "year"
    rows = []
    for bucket in _buckets(table, axis, edges):
        part = (table[table[key] == bucket] if len(table)
                else table)
        rows.extend(_plan_rows(part, window, axis, bucket, quantile_edges, lag))
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLS)
    return pd.DataFrame(rows)[SUMMARY_COLS]


def _plan_rows(part: pd.DataFrame, window: str, axis: str, bucket,
               quantile_edges: Optional[np.ndarray], lag: int) -> List[dict]:
    """1 バケツぶんの 4 案の行。"""
    out = []

    def add(plan, group, sub, variant):
        out.append({"window": window, "axis": axis, "bucket": bucket,
                    "plan": PLAN_LABELS[plan],
                    **summarize(group, sub, variant, lag)})

    # 抜け幅 5 分位（§9）。**出口は現行基準**、群だけを分位で切る
    add(PLAN_BREAKOUT, "現行基準・全体", part, CUR)
    if quantile_edges is not None:
        x = (part["breakout_h"].to_numpy(dtype=float) if len(part)
             else np.asarray([], dtype=float))
        for i in range(len(quantile_edges) - 1):
            lo, hi = quantile_edges[i], quantile_edges[i + 1]
            if len(part):
                mask = (x >= lo) & ((x < hi) if i < len(quantile_edges) - 2
                                    else (x <= hi))
                sub = part[mask]
            else:
                sub = part
            add(PLAN_BREAKOUT, f"Q{i + 1}", sub, CUR)

    for plan in (PLAN_ATR, PLAN_STOP, PLAN_TARGET):
        for v in PLAN_VARIANTS[plan]:
            add(plan, VARIANT_LABELS[v], part, v)
    return out


def format_strata_table(table: pd.DataFrame, plan: str,
                        with_r20: bool = False) -> List[str]:
    """ログに出す行（表のみ。解釈はしない）。**NW t は参考**（§13.4）。"""
    head = (f"{'窓':<8}{'区分':<22}{'群':<24}{'件数':>8}{'日数':>7}{'平均損益':>10}"
            f"{'NW t(参考)':>11}{'到達率':>8}{'撤退率':>8}{'時間切れ':>9}{'勝率':>8}")
    if with_r20:
        head += f"{'平均r20':>10}"
    head += f"{'対象外':>8}"
    out = [head]
    part = table[table["plan"] == PLAN_LABELS[plan]] if len(table) else table
    for _i, r in part.iterrows():
        def pct(v):
            return "—" if pd.isna(v) else f"{float(v) * 100:.1f}%"

        def num(v, d=2, sign="+"):
            return "—" if pd.isna(v) else f"{float(v):{sign}.{d}f}"

        line = (f"{str(r['window']):<8}{str(r['bucket']):<22}{str(r['group']):<24}"
                f"{int(r['n']):>8}{int(r['n_days']):>7}"
                f"{num(r['mean_pnl']) + '%':>10}{num(r['nw_t']):>11}"
                f"{pct(r['target_rate']):>8}{pct(r['stop_rate']):>8}"
                f"{pct(r['timeout_rate']):>9}{pct(r['win_rate']):>8}")
        if with_r20:
            line += f"{num(r['mean_r20'], 4):>10}"
        line += f"{int(r['n_na']):>8}"
        out.append(line)
    return out


def strata_path(base: Path, window: str) -> Path:
    return Path(base) / f"pattern_strata_{window}.csv.gz"


def summary_path(base: Path, window: str) -> Path:
    return Path(base) / f"pattern_strata_summary_{window}.csv"
