"""L1 集計（DESIGN.md §10.2 の 0〜6 / TASKS.md T-403）。

CLAUDE.md「検証結果を解釈しない」に従い、このモジュールは表（csv・markdown）を
出すだけで、閾値の解釈・特徴量の採否・撤退判断は一切行わない。DESIGN.md §10.2 の
記述に「〜する」「〜を落とす」とあるものも、実際に採否を決めているのは設計責任者
（Fable）であり、このモジュールはその判断材料になる値（相関係数、t 値、成功率、
BH 法の q 値、DESIGN.md §10.2 の分類基準そのままの帯ラベル等）を列として出すだけ
で、行を削除したり特徴量を除外したりはしない。

入力は主に validation.replay.load_replay_table() の出力（銘柄×日、
replay.REPLAY_COLS）。ベースライン (a)(b) は打ち切らない全期間 OHLCV
（validation.replay.run_replay に渡したものと同じ）も必要。

CLAUDE.md 絶対規則（ホールドアウトを明示フラグ無しで見ない）: ベースライン(a)
（baseline_a_random_sma200）と(b)（nishimura_trades）は T+h 形式で ohlcv を
先読みするため、呼び出し側が誤って全期間（ホールドアウト含む）の ohlcv を渡しても
内部でホールドアウト開始日より前に打ち切ってから使う（validation.replay の
_truncate_before_holdout と同じ防御。import 方向の都合で小さく複製している
―― calibration.py が本モジュールをインポートするため、逆方向の import は循環になる）。

統計量は numpy/pandas だけで計算する（scipy 等の新規依存を増やさない）。
有意性検定の p 値は正規近似（大標本近似。日次サンプル数が数百〜のため十分実用的）。
相関は Spearman（§6.2 のプール正規化がランクベースであることと整合させる）。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..features import dimensions, indicators
from .replay import HOLDOUT_WINDOW, MIN_HISTORY_BARS, _date_position, replay_universe_tickers
from . import labels as labels_mod

# DESIGN.md §5: D8 は採点しない（direction=None）ので単変量・分布の対象からも外す
SCORED_FEATURE_IDS = [fid for fid, _dim, direction, _band in dimensions.FEATURE_METADATA
                      if direction is not None]
ALL_FEATURE_IDS = dimensions.FEATURE_IDS
DIMENSIONS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
VARIANTS = ["v1", "v2", "v3"]
DEFAULT_H = 10  # DESIGN.md §10.2-1: 単変量の基準horizon。他の段も既定で揃える
N_DRAWS_DEFAULT = 100  # DESIGN.md §10.2-4(a)

# ic_summary() の戻り値のキー（列が1つも無い空のDataFrameを書き出すと0バイトのcsvになり
# 読み込み側でエラーになるため、行が0件でも列だけは必ず持たせるのに使う）
IC_SUMMARY_COLS = ["score", "h", "n_days", "ic_mean", "ic_t", "ic_p", "t_classification",
                   "q1_mean_r", "q5_mean_r", "q5_minus_q1", "q1_success_rate", "q5_success_rate"]
DISTRIBUTION_COLS = ["feature_id", "count", "mean", "std", "min", "p25", "p50", "p75", "max"]
MISSING_RATE_COLS = ["feature_id", "n", "n_missing", "missing_rate"]


# ------------------------------------------------------------------ 検定数ログ
class TestCounter:
    """DESIGN.md §10.2-5「検定数を記録」用のカウンタ。record() を呼ぶたびに1件増える。"""

    def __init__(self) -> None:
        self.records: List[dict] = []

    def record(self, kind: str, name: str) -> None:
        self.records.append({"kind": kind, "name": name})

    @property
    def count(self) -> int:
        return len(self.records)

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["kind", "name"])
        return pd.DataFrame(self.records)


# ------------------------------------------------------------------ 出力ヘルパー
def _df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate 等の追加依存無しで GitHub 風 markdown 表を作る。"""
    if len(df.columns) == 0:
        return ""
    cols = [str(c) for c in df.columns]

    def _fmt(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return f"{v:.6g}"
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, output_dir: Path, name: str) -> Dict[str, Path]:
    """1つの表を csv と markdown の両方で書き出す（受け入れ: すべての表がcsvと
    markdownで出る）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    md_path = output_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_df_to_markdown(df), encoding="utf-8")
    return {"csv": csv_path, "md": md_path}


# ------------------------------------------------------------------ 統計ヘルパー
def newey_west_mean_t(x: np.ndarray, lag: int) -> dict:
    """時系列 x の平均と、Newey-West（Bartlett核、ラグ lag）で自己相関を補正した
    標準誤差・t値（DESIGN.md §10.1「t値は保有日数分のラグのNewey–West」）。

    p値は正規近似（大標本近似、scipy 非依存）。
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"mean": float(np.nan), "se": float(np.nan), "t": float(np.nan),
                "p": float(np.nan), "n": n}
    mean = float(x.mean())
    d = x - mean
    gamma0 = float(np.dot(d, d) / n)
    var = gamma0
    lag = max(0, min(lag, n - 1))
    for l in range(1, lag + 1):
        gamma_l = float(np.dot(d[l:], d[:-l]) / n)
        w = 1.0 - l / (lag + 1)
        var += 2 * w * gamma_l
    var = max(var, 0.0)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float(np.nan)
    p = _normal_two_sided_pvalue(t) if np.isfinite(t) else float(np.nan)
    return {"mean": mean, "se": se, "t": t, "p": p, "n": n}


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman順位相関。pandas の method="spearman" は内部で scipy を要求するため
    （このリポジトリは scipy に依存しない）、順位化してから Pearson で計算する
    （定義上 Spearman はこれと同じ）。"""
    return float(a.rank().corr(b.rank()))


def _spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank().corr()


def _normal_two_sided_pvalue(t: float) -> float:
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))


def bh_fdr_qvalues(pvalues: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg 法の q 値（DESIGN.md §10.2-5「単変量曲線はBH法FDR10%」）。
    NaN の p 値は q も NaN のまま返す。
    """
    p = np.asarray(list(pvalues), dtype=float)
    q = np.full_like(p, np.nan)
    valid_idx = np.where(~np.isnan(p))[0]
    if valid_idx.size == 0:
        return q
    order = valid_idx[np.argsort(p[valid_idx])]
    m = order.size
    prev = 1.0
    q_sorted = np.empty(m)
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, p[i] * m / (rank + 1))
        q_sorted[rank] = val
        prev = val
    for pos, i in enumerate(order):
        q[i] = q_sorted[pos]
    return q


def classify_t(t: float) -> str:
    """DESIGN.md §10.2-5 の分類そのまま（採否は設計責任者。ここはラベル付けのみ）。"""
    if np.isnan(t):
        return "判定不能"
    if t > 3:
        return "採用"
    if t >= 2:
        return "保留"
    return "棄却"


def _truncate_before_holdout(ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """validation.replay._truncate_before_holdout と同じ防御（循環importを避けるため
    小さく複製）。CLAUDE.md 絶対規則によりホールドアウトを明示フラグ無しで見ない。"""
    out = {}
    for ticker, df in ohlcv.items():
        if df is None or len(df) == 0:
            out[ticker] = df
            continue
        out[ticker] = df[df.index < HOLDOUT_WINDOW[0]]
    return out


def depth_band(depth_pct: pd.Series) -> pd.Series:
    """SPEC.md/DESIGN.md の深さ帯（<5% / 5〜10% / >10%）。"""
    bins = [-np.inf, 0.05, 0.10, np.inf]
    labels = ["<5%", "5-10%", ">10%"]
    return pd.cut(depth_pct, bins=bins, labels=labels, right=False)


def _success_rate(label_series: pd.Series) -> float:
    """成功率 = success件数 / (success+failure件数)。未決は分母から除く。"""
    decided = label_series[label_series.isin([labels_mod.LABEL_SUCCESS, labels_mod.LABEL_FAILURE])]
    if len(decided) == 0:
        return float(np.nan)
    return float((decided == labels_mod.LABEL_SUCCESS).mean())


# ------------------------------------------------------------------ 0. 健全性
def sanity_tables(pool: pd.DataFrame, feature_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """DESIGN.md §10.2-0: 分布、欠損率、相関行列、|ρ|>0.7 の組（除外はしない、列挙のみ）。"""
    feature_ids = feature_ids if feature_ids is not None else SCORED_FEATURE_IDS
    numeric_ids = [f for f in feature_ids if f in pool.columns
                  and pd.api.types.is_numeric_dtype(pool[f])]

    dist_rows = []
    missing_rows = []
    for fid in feature_ids:
        if fid not in pool.columns:
            continue
        s = pool[fid]
        n = len(s)
        n_missing = int(s.isna().sum())
        missing_rows.append({"feature_id": fid, "n": n, "n_missing": n_missing,
                             "missing_rate": n_missing / n if n else np.nan})
        if fid in numeric_ids:
            desc = s.astype(float).describe()
            dist_rows.append({"feature_id": fid, "count": desc.get("count", np.nan),
                              "mean": desc.get("mean", np.nan), "std": desc.get("std", np.nan),
                              "min": desc.get("min", np.nan), "p25": desc.get("25%", np.nan),
                              "p50": desc.get("50%", np.nan), "p75": desc.get("75%", np.nan),
                              "max": desc.get("max", np.nan)})
    distribution = pd.DataFrame(dist_rows, columns=DISTRIBUTION_COLS)
    missing_rate = pd.DataFrame(missing_rows, columns=MISSING_RATE_COLS)

    if len(numeric_ids) >= 2:
        corr = _spearman_matrix(pool[numeric_ids].astype(float))
    else:
        corr = pd.DataFrame(index=numeric_ids, columns=numeric_ids, dtype=float)
    corr_out = corr.reset_index().rename(columns={"index": "feature_id"})

    pairs = []
    for i, a in enumerate(numeric_ids):
        for b in numeric_ids[i + 1:]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(rho) > 0.7:
                pairs.append({"feature_a": a, "feature_b": b, "rho": float(rho)})
    high_corr = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "rho"])
    if len(high_corr):
        high_corr = high_corr.reindex(high_corr["rho"].abs().sort_values(ascending=False).index)

    return {"distribution": distribution, "missing_rate": missing_rate,
           "correlation": corr_out, "high_correlation_pairs": high_corr}


# ------------------------------------------------------------------ 1. 単変量
def univariate_decile_curve(pool: pd.DataFrame, feature_id: str, h: int = DEFAULT_H) -> pd.DataFrame:
    """1特徴量ぶんの10分位（連続量）または値ごと（二値・D8）の r_h 平均・成功率曲線。"""
    if feature_id not in pool.columns:
        return pd.DataFrame(columns=["feature_id", "bucket", "n", f"mean_r_{h}", "success_rate"])
    meta = {m[0]: m for m in dimensions.FEATURE_METADATA}
    direction = meta[feature_id][2] if feature_id in meta else None
    col = pool[feature_id]
    r_col = f"r_{h}"
    valid = pool[col.notna()]

    if direction == "binary" or valid[feature_id].nunique(dropna=True) <= 2:
        bucket = valid[feature_id].astype(str)
    else:
        try:
            bucket = pd.qcut(valid[feature_id].astype(float), 10, duplicates="drop")
        except (ValueError, IndexError):
            bucket = valid[feature_id].astype(str)

    rows = []
    for name, idx in valid.groupby(bucket, observed=True).groups.items():
        sub = valid.loc[idx]
        rows.append({
            "feature_id": feature_id, "bucket": str(name), "n": len(sub),
            f"mean_r_{h}": float(sub[r_col].mean()) if r_col in sub else np.nan,
            "success_rate": _success_rate(sub["label"]),
        })
    out = pd.DataFrame(rows, columns=["feature_id", "bucket", "n", f"mean_r_{h}", "success_rate"])
    return out.sort_values("bucket").reset_index(drop=True)


def univariate_all(pool: pd.DataFrame, feature_ids: Optional[List[str]] = None,
                   h: int = DEFAULT_H, test_counter: Optional[TestCounter] = None) -> pd.DataFrame:
    """全特徴量ぶんの単変量分位曲線を1つの縦持ち表にまとめる（DESIGN.md §10.2-1）。"""
    feature_ids = feature_ids if feature_ids is not None else SCORED_FEATURE_IDS
    frames = []
    for fid in feature_ids:
        curve = univariate_decile_curve(pool, fid, h=h)
        if len(curve):
            frames.append(curve)
        if test_counter is not None:
            test_counter.record("univariate", fid)
    if not frames:
        return pd.DataFrame(columns=["feature_id", "bucket", "n", f"mean_r_{h}", "success_rate"])
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ 2/3. IC・分位曲線
def _daily_ic(pool: pd.DataFrame, score_col: str, h: int) -> pd.Series:
    """score_col と r_h の日次 Spearman IC（銘柄横断、各営業日ごと）。"""
    r_col = f"r_{h}"

    def _ic(day_df: pd.DataFrame) -> float:
        sub = day_df[[score_col, r_col]].dropna()
        if len(sub) < 5:
            return np.nan
        return _spearman(sub[score_col], sub[r_col])

    return pool.groupby("date", observed=True).apply(_ic, include_groups=False)


def ic_summary(pool: pd.DataFrame, score_col: str, h: int = DEFAULT_H) -> dict:
    """日次IC平均・Newey-West t（保有日数=hをラグに使う）・分位（Q5-Q1）スプレッド。"""
    daily_ic = _daily_ic(pool, score_col, h)
    nw = newey_west_mean_t(daily_ic.to_numpy(dtype=float), lag=h)

    r_col = f"r_{h}"
    sub = pool[[score_col, r_col, "label"]].dropna(subset=[score_col])
    result = {"score": score_col, "h": h, "n_days": int(daily_ic.notna().sum()),
             "ic_mean": nw["mean"], "ic_t": nw["t"], "ic_p": nw["p"],
             "t_classification": classify_t(nw["t"]),
             "q1_mean_r": np.nan, "q5_mean_r": np.nan, "q5_minus_q1": np.nan,
             "q1_success_rate": np.nan, "q5_success_rate": np.nan}
    if len(sub) >= 10:
        try:
            q = pd.qcut(sub[score_col].astype(float), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                       duplicates="drop")
        except (ValueError, IndexError):
            q = None
        if q is not None:
            sub = sub.assign(_q=q)
            q1 = sub[sub["_q"] == "Q1"]
            q5 = sub[sub["_q"] == "Q5"]
            result["q1_mean_r"] = float(q1[r_col].mean()) if len(q1) else np.nan
            result["q5_mean_r"] = float(q5[r_col].mean()) if len(q5) else np.nan
            if np.isfinite(result["q1_mean_r"]) and np.isfinite(result["q5_mean_r"]):
                result["q5_minus_q1"] = result["q5_mean_r"] - result["q1_mean_r"]
            result["q1_success_rate"] = _success_rate(q1["label"])
            result["q5_success_rate"] = _success_rate(q5["label"])
    return result


def dimension_summary_table(pool: pd.DataFrame, h: int = DEFAULT_H,
                            test_counter: Optional[TestCounter] = None) -> pd.DataFrame:
    """DESIGN.md §10.2-2: D1〜D7 の日次IC平均・t・Q5-Q1。"""
    rows = []
    for dim in DIMENSIONS:
        col = f"dim_{dim}_score"
        if col not in pool.columns:
            continue
        rows.append(ic_summary(pool, col, h=h))
        if test_counter is not None:
            test_counter.record("dimension_ic", dim)
    return pd.DataFrame(rows, columns=IC_SUMMARY_COLS)


def composite_summary_table(pool: pd.DataFrame, h: int = DEFAULT_H,
                            test_counter: Optional[TestCounter] = None) -> pd.DataFrame:
    """DESIGN.md §10.2-3: V1/V2/V3 の日次IC平均・t・Q5-Q1・分位別成功率。"""
    rows = []
    for v in VARIANTS:
        col = f"score_{v}"
        if col not in pool.columns:
            continue
        rows.append(ic_summary(pool, col, h=h))
        if test_counter is not None:
            test_counter.record("composite_ic", v)
    return pd.DataFrame(rows, columns=IC_SUMMARY_COLS)


def composite_decile_curve(pool: pd.DataFrame, variant: str, h: int = DEFAULT_H) -> pd.DataFrame:
    """V1/V2/V3 の10分位曲線（r_h平均・成功率）。"""
    col = f"score_{variant}"
    return univariate_decile_curve(pool.assign(**{col: pool[col]}), col, h=h) if col in pool.columns \
        else pd.DataFrame(columns=["feature_id", "bucket", "n", f"mean_r_{h}", "success_rate"])


CUT_DEFINITIONS = ("state", "regime", "depth_band", "year", "market", "month")


def _cut_series(pool: pd.DataFrame, cut: str, listed: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
    if cut == "state":
        return pool["state"] if "state" in pool.columns else None
    if cut == "regime":
        return pool["regime"] if "regime" in pool.columns else None
    if cut == "depth_band":
        return depth_band(pool["depth_pct"]) if "depth_pct" in pool.columns else None
    if cut == "year":
        return pd.to_datetime(pool["date"]).dt.year if "date" in pool.columns else None
    if cut == "month":
        return pd.to_datetime(pool["date"]).dt.month if "date" in pool.columns else None
    if cut == "market":
        if listed is None or "ticker" not in pool.columns or "market" not in listed.columns:
            return None
        market_map = listed.drop_duplicates("ticker").set_index("ticker")["market"]
        return pool["ticker"].map(market_map)
    raise ValueError(f"unknown cut: {cut}")


def composite_by_cut(pool: pd.DataFrame, variant: str, cut: str, h: int = DEFAULT_H,
                     listed: Optional[pd.DataFrame] = None,
                     test_counter: Optional[TestCounter] = None) -> pd.DataFrame:
    """DESIGN.md §10.2-3 の切り口別（状態・地合い・深さ帯・年・市場区分・月別）。
    cut に該当する情報が無ければ（例: listed 未指定で market）空の表を返す。
    """
    col = f"score_{variant}"
    by_cut_cols = ["variant", "cut", "group", "n_rows"] + IC_SUMMARY_COLS
    cut_series = _cut_series(pool, cut, listed=listed)
    if col not in pool.columns or cut_series is None:
        return pd.DataFrame(columns=by_cut_cols)

    rows = []
    grouped = pool.assign(_cut=cut_series).groupby("_cut", observed=True)
    for group_name, idx in grouped.groups.items():
        sub = pool.loc[idx]
        if len(sub) < 10:
            continue
        summary = ic_summary(sub, col, h=h)
        summary_row = {"variant": variant, "cut": cut, "group": str(group_name),
                       "n_rows": len(sub), **summary}
        rows.append(summary_row)
        if test_counter is not None:
            test_counter.record("composite_by_cut", f"{variant}:{cut}:{group_name}")
    return pd.DataFrame(rows, columns=by_cut_cols)


def all_cuts_tables(pool: pd.DataFrame, h: int = DEFAULT_H, listed: Optional[pd.DataFrame] = None,
                    test_counter: Optional[TestCounter] = None) -> Dict[str, pd.DataFrame]:
    out = {}
    for variant in VARIANTS:
        for cut in CUT_DEFINITIONS:
            table = composite_by_cut(pool, variant, cut, h=h, listed=listed, test_counter=test_counter)
            out[f"{variant}_{cut}"] = table
    return out


# ------------------------------------------------------------------ 4. ベースライン
def baseline_c_equal_weight(pool: pd.DataFrame, h: int = DEFAULT_H) -> dict:
    """DESIGN.md §10.2-4(c): 押し目状態の全銘柄の等加重。"""
    r_col = f"r_{h}"
    return {"baseline": "c_equal_weight", "n": len(pool),
           "mean_r": float(pool[r_col].mean()) if r_col in pool.columns and len(pool) else np.nan,
           "success_rate": _success_rate(pool["label"]) if "label" in pool.columns else np.nan}


def baseline_a_random_sma200(
    ohlcv: Dict[str, pd.DataFrame], listed: pd.DataFrame, pool: pd.DataFrame,
    h: int = DEFAULT_H, n_draws: int = N_DRAWS_DEFAULT, seed: int = 0,
    min_history_bars: int = MIN_HISTORY_BARS,
) -> dict:
    """DESIGN.md §10.2-4(a): SMA200上の銘柄から同数をランダム抽出（乱数100回の平均）。

    候補集合はその日のユニバース（履歴本数のみで決める簡易定義、DESIGN.md §10.1）の
    うち Close[T]>=SMA200[T] の銘柄。抽出数はその日の pool 内の件数に合わせる。
    """
    ohlcv = _truncate_before_holdout(ohlcv)
    rng = np.random.default_rng(seed)
    dates = sorted(pd.to_datetime(pool["date"]).unique()) if len(pool) else []
    per_day_means: List[float] = []
    n_days_used = 0
    for date_t in dates:
        date_t = pd.Timestamp(date_t)
        n_needed = int((pd.to_datetime(pool["date"]) == date_t).sum())
        if n_needed == 0:
            continue
        universe = replay_universe_tickers(listed, ohlcv, date_t, min_history_bars)
        eligible = []
        for ticker in universe:
            df = ohlcv[ticker]
            pos = _date_position(df.index, date_t)
            if pos is None:
                continue
            close_trunc = df["Close"].iloc[: pos + 1]
            sma200 = indicators.sma(close_trunc, 200)
            if len(sma200) == 0 or np.isnan(sma200.iloc[-1]):
                continue
            if close_trunc.iloc[-1] >= sma200.iloc[-1]:
                eligible.append(ticker)
        if not eligible:
            continue
        benchmarks_raw = labels_mod.universe_benchmark_returns(
            {t: ohlcv[t] for t in universe}, date_t, (h,))
        benchmark = benchmarks_raw[h]["mean"]

        draw_means = []
        replace = len(eligible) < n_needed
        for _ in range(n_draws):
            sample = rng.choice(eligible, size=n_needed, replace=replace)
            rs = []
            for ticker in sample:
                df = ohlcv[ticker]
                pos = _date_position(df.index, date_t)
                r = labels_mod.excess_return(df["Close"].to_numpy(dtype=float),
                                             df["Open"].to_numpy(dtype=float), pos, h, benchmark)
                if not np.isnan(r):
                    rs.append(r)
            if rs:
                draw_means.append(float(np.mean(rs)))
        if draw_means:
            per_day_means.append(float(np.mean(draw_means)))
            n_days_used += 1

    return {"baseline": "a_random_sma200", "n_days": n_days_used, "n_draws": n_draws,
           "mean_r": float(np.mean(per_day_means)) if per_day_means else np.nan}


def nishimura_trades(ohlcv: Dict[str, pd.DataFrame], tickers: Iterable[str],
                     start: pd.Timestamp, end: pd.Timestamp, max_hold: int = 15) -> pd.DataFrame:
    """DESIGN.md §10.2-4(b) 西村ルールの再現。エントリー: Close[T]<=0.95*SMA5[T] かつ
    Close[T]>SMA75[T] で T+1 寄付き買い。手仕舞い: Close[t]>SMA5[t] となった最初の日の
    翌日寄付き。max_hold 日以内に成立しなければ max_hold 日目の終値で強制手仕舞い
    （「最長15日」の解釈。翌日寄付きが定義されないため終値を使う）。
    """
    ohlcv = _truncate_before_holdout(ohlcv)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < 80:
            continue
        close, open_ = df["Close"], df["Open"]
        sma5 = indicators.sma(close, 5)
        sma75 = indicators.sma(close, 75)
        c = close.to_numpy(dtype=float)
        o = open_.to_numpy(dtype=float)
        s5 = sma5.to_numpy(dtype=float)
        s75 = sma75.to_numpy(dtype=float)
        idx = df.index
        in_window = (idx >= start) & (idx <= end)
        for t in np.where(in_window)[0]:
            if t + 1 >= len(df) or np.isnan(s5[t]) or np.isnan(s75[t]):
                continue
            if not (c[t] <= 0.95 * s5[t] and c[t] > s75[t]):
                continue
            entry_pos = t + 1
            entry_price = o[entry_pos]
            if not (np.isfinite(entry_price) and entry_price > 0):
                continue
            exit_pos, exit_price, reason = None, None, None
            for h in range(1, max_hold + 1):
                chk = entry_pos + h - 1
                if chk >= len(df) or np.isnan(s5[chk]):
                    break
                if c[chk] > s5[chk]:
                    nxt = chk + 1
                    if nxt < len(df) and np.isfinite(o[nxt]) and o[nxt] > 0:
                        exit_pos, exit_price, reason = nxt, o[nxt], "sma5_recover"
                    break
            if exit_price is None:
                timeout_pos = min(entry_pos + max_hold - 1, len(df) - 1)
                if timeout_pos > entry_pos and np.isfinite(c[timeout_pos]) and c[timeout_pos] > 0:
                    exit_pos, exit_price, reason = timeout_pos, c[timeout_pos], "timeout"
            if exit_price is None:
                continue
            ret = exit_price / entry_price - 1.0
            rows.append({"ticker": ticker, "entry_date": idx[entry_pos], "exit_date": idx[exit_pos],
                        "entry_price": float(entry_price), "exit_price": float(exit_price),
                        "return": float(ret), "hold_days": int(exit_pos - entry_pos),
                        "exit_reason": reason})
    return pd.DataFrame(rows, columns=["ticker", "entry_date", "exit_date", "entry_price",
                                       "exit_price", "return", "hold_days", "exit_reason"])


def nishimura_summary(trades: pd.DataFrame) -> dict:
    """勝率・PF（DESIGN.md §10.2-4(b)の較正値: 2016〜2020で勝率60%前後・PF1.5前後）。"""
    if len(trades) == 0:
        return {"baseline": "b_nishimura", "n_trades": 0, "win_rate": np.nan,
               "profit_factor": np.nan, "avg_hold_days": np.nan}
    wins = trades[trades["return"] > 0]["return"]
    losses = trades[trades["return"] <= 0]["return"]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float(np.inf) if gross_win > 0 else np.nan
    return {"baseline": "b_nishimura", "n_trades": len(trades),
           "win_rate": float((trades["return"] > 0).mean()),
           "profit_factor": pf, "avg_hold_days": float(trades["hold_days"].mean())}


def baseline_comparison_table(
    pool: pd.DataFrame, ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
    listed: Optional[pd.DataFrame] = None, h: int = DEFAULT_H,
    n_draws: int = N_DRAWS_DEFAULT, seed: int = 0,
    nishimura_start: Optional[pd.Timestamp] = None, nishimura_end: Optional[pd.Timestamp] = None,
    min_history_bars: int = MIN_HISTORY_BARS,
) -> pd.DataFrame:
    """(a)(b)(c) を1つの表にまとめる（DESIGN.md §10.2-4）。ohlcv/listed が無ければ
    (a) は計算できないため行を欠く（例外にはしない）。
    """
    rows = [baseline_c_equal_weight(pool, h=h)]
    if ohlcv is not None and listed is not None and len(pool):
        rows.append(baseline_a_random_sma200(ohlcv, listed, pool, h=h, n_draws=n_draws,
                                             seed=seed, min_history_bars=min_history_bars))
        if nishimura_start is not None and nishimura_end is not None:
            tickers = sorted(ohlcv.keys())
            trades = nishimura_trades(ohlcv, tickers, nishimura_start, nishimura_end)
            rows.append(nishimura_summary(trades))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6. 生存バイアス
def survivorship_note(delistings: Optional[pd.DataFrame], listed: Optional[pd.DataFrame]) -> dict:
    """生存バイアスの上限評価（SPEC §4）の診断メモ。実際のIC・リターン数値を補正する
    のではなく、「どれだけの銘柄が生存バイアスで見えていないか」を件数・比率で示す
    （解釈・補正は設計責任者が行う）。delistings/listed が無ければ全て NaN。
    """
    if delistings is None or listed is None or len(listed) == 0:
        return {"n_delisted": np.nan, "n_current_universe": np.nan, "delisted_ratio": np.nan}
    n_delisted = int(len(delistings)) if delistings is not None else 0
    n_current = int(listed["is_equity"].astype(bool).sum()) if "is_equity" in listed.columns else len(listed)
    ratio = n_delisted / (n_delisted + n_current) if (n_delisted + n_current) > 0 else np.nan
    return {"n_delisted": n_delisted, "n_current_universe": n_current, "delisted_ratio": ratio}


# ------------------------------------------------------------------ オーケストレーター
def run_layer1(
    pool: pd.DataFrame, output_dir: Path,
    ohlcv: Optional[Dict[str, pd.DataFrame]] = None, listed: Optional[pd.DataFrame] = None,
    delistings: Optional[pd.DataFrame] = None,
    h: int = DEFAULT_H, n_draws: int = N_DRAWS_DEFAULT, seed: int = 0,
    nishimura_start: Optional[pd.Timestamp] = None, nishimura_end: Optional[pd.Timestamp] = None,
    log=print,
) -> TestCounter:
    """DESIGN.md §10.2 の0〜6をすべて計算し、output_dir 以下に csv/markdown で書き出す。
    受け入れ: すべての表がcsvとmarkdownで出る。検定数はTestCounterに自動で記録される。
    """
    output_dir = Path(output_dir)
    counter = TestCounter()

    sanity = sanity_tables(pool)
    for name, df in sanity.items():
        write_table(df, output_dir, f"0_sanity_{name}")
    log(f"[layer1] 0. 健全性: 特徴量{len(sanity['missing_rate'])}件、"
       f"|ρ|>0.7 の組 {len(sanity['high_correlation_pairs'])}件")

    univariate = univariate_all(pool, h=h, test_counter=counter)
    p_values = []
    for fid in univariate["feature_id"].unique():
        curve = univariate[univariate["feature_id"] == fid]
        rc = f"mean_r_{h}"
        nw = newey_west_mean_t(curve[rc].to_numpy(dtype=float), lag=1)
        p_values.append(nw["p"])
    q = bh_fdr_qvalues(p_values) if p_values else np.array([])
    q_map = dict(zip(univariate["feature_id"].unique(), q)) if len(q) else {}
    univariate = univariate.assign(bh_q=univariate["feature_id"].map(q_map))
    write_table(univariate, output_dir, "1_univariate_deciles")
    log(f"[layer1] 1. 単変量: {univariate['feature_id'].nunique()}特徴量、検定数 {counter.count}")

    dim_table = dimension_summary_table(pool, h=h, test_counter=counter)
    write_table(dim_table, output_dir, "2_dimension_ic")
    log(f"[layer1] 2. 次元スコアIC: {len(dim_table)}件")

    comp_table = composite_summary_table(pool, h=h, test_counter=counter)
    write_table(comp_table, output_dir, "3_composite_ic")
    for v in VARIANTS:
        curve = composite_decile_curve(pool, v, h=h)
        write_table(curve, output_dir, f"3_composite_{v}_deciles")
    cuts = all_cuts_tables(pool, h=h, listed=listed, test_counter=counter)
    for name, df in cuts.items():
        write_table(df, output_dir, f"3_composite_by_{name}")
    log(f"[layer1] 3. 総合V1/V2/V3: {len(comp_table)}件、切り口 {len(cuts)}表、検定数 {counter.count}")

    baseline_table = baseline_comparison_table(pool, ohlcv=ohlcv, listed=listed, h=h,
                                               n_draws=n_draws, seed=seed,
                                               nishimura_start=nishimura_start,
                                               nishimura_end=nishimura_end)
    write_table(baseline_table, output_dir, "4_baseline_comparison")
    log(f"[layer1] 4. ベースライン: {len(baseline_table)}件")

    counter_df = counter.to_frame()
    write_table(counter_df, output_dir, "5_test_count_log")
    log(f"[layer1] 5. 多重検定: 検定数 {counter.count}")

    note = survivorship_note(delistings, listed)
    write_table(pd.DataFrame([note]), output_dir, "6_survivorship_note")
    log(f"[layer1] 6. 生存バイアス上限評価: {note}")

    return counter


def main(argv: Optional[list] = None) -> int:
    """python -m stockbot.validation.layer1 --replay-dir ... --output-dir ... で実行する。

    validation.replay.run_replay が書いた replay_*.csv.gz を --replay-dir から読み、
    store/reference から ohlcv・listed・delistings を読んでベースライン・生存バイアス
    メモも合わせて計算する。
    """
    import argparse

    from ..config import Settings
    from ..data.jpx_lists import load_delistings, normalize_listed
    from ..data.store import IDX_TICKER, OhlcvStore, from_long
    from .replay import load_replay_table

    ap = argparse.ArgumentParser(prog="python -m stockbot.validation.layer1")
    ap.add_argument("--replay-dir", required=True, help="validation.replay.run_replay の出力先")
    ap.add_argument("--output-dir", default=None, help="既定: <DATA_DIR>/layer1")
    ap.add_argument("--h", type=int, default=DEFAULT_H)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nishimura-start", default=None, help="YYYY-MM-DD（既定: replay範囲の最初）")
    ap.add_argument("--nishimura-end", default=None, help="YYYY-MM-DD（既定: replay範囲の最後）")
    args = ap.parse_args(argv)

    cfg = Settings.from_env()
    pool = load_replay_table(args.replay_dir)
    if len(pool) == 0:
        print(f"[layer1] {args.replay_dir} に再生結果が無いため中止")
        return 1

    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)

    listed = None
    listed_path = cfg.reference_dir / "listed_latest.csv"
    if listed_path.exists():
        listed = normalize_listed(pd.read_csv(listed_path, dtype=str).fillna(""))

    delistings = None
    delistings_path = cfg.reference_dir / "delistings.csv"
    if delistings_path.exists():
        delistings = load_delistings(delistings_path)

    dates = sorted(pd.to_datetime(pool["date"]).unique())
    nishimura_start = pd.Timestamp(args.nishimura_start) if args.nishimura_start else pd.Timestamp(dates[0])
    nishimura_end = pd.Timestamp(args.nishimura_end) if args.nishimura_end else pd.Timestamp(dates[-1])

    output_dir = Path(args.output_dir) if args.output_dir else cfg.data_dir / "layer1"
    run_layer1(pool, output_dir, ohlcv=ohlcv, listed=listed, delistings=delistings,
              h=args.h, n_draws=args.n_draws, seed=args.seed,
              nishimura_start=nishimura_start, nishimura_end=nishimura_end, log=print)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
