"""探索の集計（docs/BACKTEST.md §4・§6）。

**検定は 10 件だけ。** パターン別（5 種）と抜け幅 `b` の 5 分位。
**それ以上に広げない**（§5 の禁止事項）。業種・売買代金・span・散らばり・
連続日数・地合いは記録するが検定しない。

**主指標は超過リターン `r_20`。** 勝率・成功率・目標到達率は表に併記するが、
**判定には使わない**（D-2）。

**Newey-West は日次平均系列に当てる**（D-4）。評価窓が 20 本で重なるので、
イベントを独立と扱うと t 値が過大になる。群の行が 0 件の日は系列に入れない
（0 で埋めない）。ラグは保有日数と同じ 20。

**数値の解釈も採否の判断もしない。表を作るまで**（CLAUDE.md）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .layer1 import newey_west_mean_t
from .pattern_replay import HORIZON

# docs/BACKTEST.md §4.1。**C3・C4 を除いた 5 種**（日足かつ k=3 では構造的に
# 検出されないため。PATTERN.md D-10）。§7 Q-1 で確認待ち
TESTED_PATTERNS = ["double_bottom", "triple_bottom", "inverse_hs",
                   "ascending_triangle", "ascending_box"]
N_QUANTILES = 5
N_TESTS = len(TESTED_PATTERNS) + N_QUANTILES   # 10 件。**増やさない**

PATTERN_LABELS = {
    "double_bottom": "ダブルボトム", "triple_bottom": "トリプルボトム",
    "inverse_hs": "逆三尊", "ascending_triangle": "上昇三角",
    "ascending_box": "上昇ボックス",
}

GROUP_COLS = ["group", "n", "n_days", "mean_r20", "nw_t", "nw_p",
              "win_rate", "success_rate", "target_rate", "already_rate",
              "mfe_atr_median", "mae_atr_median"]


def daily_mean_series(df: pd.DataFrame, value: str = "r_20") -> pd.Series:
    """評価日ごとの平均（**行が 0 件の日は入れない。0 で埋めない**）。D-4。"""
    if len(df) == 0:
        return pd.Series(dtype=float)
    part = df[["date", value]].dropna()
    if len(part) == 0:
        return pd.Series(dtype=float)
    return part.groupby("date")[value].mean().sort_index()


def _rate(series: pd.Series) -> float:
    """True の比率。欠損（判定不能）は分母から外す。"""
    s = series.dropna()
    if not len(s):
        return float("nan")
    return float(s.astype(bool).mean())


def summarize_group(name: str, df: pd.DataFrame, lag: int = HORIZON) -> dict:
    """1 群ぶんの行（docs/BACKTEST.md §6）。**件数（母数）を必ず持つ。**"""
    daily = daily_mean_series(df)
    nw = newey_west_mean_t(daily.to_numpy(dtype=float), lag=lag)
    r = df["r_20"].dropna() if "r_20" in df.columns else pd.Series(dtype=float)
    return {
        "group": name,
        "n": int(len(df)),
        "n_days": int(len(daily)),
        # **平均はイベント平均**（日次平均の平均ではない）。t 値だけが日次系列から出る
        "mean_r20": float(r.mean()) if len(r) else float("nan"),
        "nw_t": nw["t"],
        "nw_p": nw["p"],
        # ここから下は診断列。**判定には使わない**（D-2）
        "win_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "success_rate": _rate(df["success"]) if "success" in df.columns else float("nan"),
        "target_rate": (_rate(df["reached_target"]) if "reached_target" in df.columns
                        else float("nan")),
        # **T の時点で既に目標を超えていた行の比率**（§3.1）。この比率が高い群では
        # 上の `target_rate` は「届いた」ではなく「最初から届いていた」を数えている
        "already_rate": (_rate(df["already_at_target"])
                         if "already_at_target" in df.columns else float("nan")),
        "mfe_atr_median": (float(df["mfe_atr"].median())
                           if "mfe_atr" in df.columns else float("nan")),
        "mae_atr_median": (float(df["mae_atr"].median())
                           if "mae_atr" in df.columns else float("nan")),
    }


def overall(df: pd.DataFrame, lag: int = HORIZON) -> dict:
    """全体（比較の基準になる「全体平均」。§5 の判定基準はこれを上回るか）。"""
    return summarize_group("全体", df, lag)


def by_pattern(df: pd.DataFrame, lag: int = HORIZON) -> pd.DataFrame:
    """検定 1〜5: パターン別（docs/BACKTEST.md §4.1）。

    **群は固定**。その窓に 0 件でも行を落とさない —— 件数 0 を表に残すほうが、
    群が消えるより読み手に正確である。
    """
    rows = [summarize_group(PATTERN_LABELS.get(p, p),
                            df[df["pattern"] == p] if len(df) else df, lag)
            for p in TESTED_PATTERNS]
    return pd.DataFrame(rows)[GROUP_COLS]


# 探索窓で作った分位の境界を固定したファイル（docs/BACKTEST.md §4.2・D-3）。
# **確認窓とホールドアウトはこれをそのまま使う。** 窓ごとに切り直すと分位の意味が
# 変わって再現を見たことにならない。Actions の cache は窓ごとに分かれていて期限も
# あるので、**リポジトリに置いて確定値にしてある**
FROZEN_EDGES_PATH = Path("data/reference/pattern_quantile_edges.json")


def load_frozen_edges(path: Optional[Path] = None) -> Optional[np.ndarray]:
    """固定した分位の境界を読む。無ければ None。

    JSON の `edges` は両端が null（±inf）の 6 要素。
    """
    path = Path(path) if path is not None else FROZEN_EDGES_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data["edges"]
    except (OSError, ValueError, KeyError):
        return None
    if not isinstance(raw, list) or len(raw) != N_QUANTILES + 1:
        return None
    out = []
    for i, v in enumerate(raw):
        if v is None:
            out.append(-np.inf if i == 0 else np.inf)
        else:
            out.append(float(v))
    return np.asarray(out, dtype=float)


def quantile_edges(df: pd.DataFrame, value: str = "breakout_h",
                   q: int = N_QUANTILES) -> Optional[np.ndarray]:
    """分位の境界を作る（**探索窓で作り、確認窓にはこれをそのまま当てる**。D-3）。

    確認窓で切り直すと分位の意味が窓ごとに変わり、再現を見たことにならない。
    """
    if value not in df.columns:
        return None
    x = df[value].dropna().to_numpy(dtype=float)
    if len(x) < q:
        return None
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def by_breakout_quantile(df: pd.DataFrame, edges: np.ndarray,
                         value: str = "breakout_h",
                         lag: int = HORIZON) -> pd.DataFrame:
    """検定 6〜10: 抜け幅 `b` の 5 分位別（docs/BACKTEST.md §4.2）。

    `b` = 抜け幅 ÷ 高さ（`PATTERN.md` §2.3 の `breakout_h`）。
    """
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f"Q{i + 1}（{lo:+.3f}〜{hi:+.3f}）".replace("-inf", "下限").replace("+inf", "上限")
        if len(df) and value in df.columns:
            x = df[value].to_numpy(dtype=float)
            # 右端の群だけ上端を含める（それ以外は [lo, hi)）
            mask = (x >= lo) & ((x < hi) if i < len(edges) - 2 else (x <= hi))
            part = df[mask]
        else:
            part = df
        rows.append(summarize_group(label, part, lag))
    return pd.DataFrame(rows)[GROUP_COLS]


def coverage(df: pd.DataFrame, dates: Optional[pd.DatetimeIndex] = None) -> dict:
    """実際に評価できた期間（docs/BACKTEST.md §6）。

    **窓の定義を黙って縮めない。** store の履歴が確認窓の開始日まで届いていない
    可能性があるので、実測をそのまま出す。
    """
    if len(df) == 0:
        return {"first": None, "last": None, "n_days": 0, "n_rows": 0}
    d = pd.to_datetime(df["date"]).dropna()
    return {"first": d.min(), "last": d.max(),
            "n_days": int(d.dt.normalize().nunique()), "n_rows": int(len(df))}


def format_table(table: pd.DataFrame) -> list[str]:
    """ログに出す行（表のみ。解釈はしない）。"""
    out = [f"{'群':<26}{'件数':>7}{'日数':>7}{'平均r20':>10}{'NW t':>8}"
           f"{'勝率':>8}{'成功率':>8}{'目標到達':>9}{'既に到達':>9}"
           f"{'MFE/ATR':>9}{'MAE/ATR':>9}"]
    for _i, r in table.iterrows():
        def pct(v):
            return "—" if pd.isna(v) else f"{float(v) * 100:.1f}%"

        def num(v, d=3):
            return "—" if pd.isna(v) else f"{float(v):+.{d}f}"

        out.append(f"{str(r['group']):<26}{int(r['n']):>7}{int(r['n_days']):>7}"
                   f"{num(r['mean_r20'], 4):>10}{num(r['nw_t'], 2):>8}"
                   f"{pct(r['win_rate']):>8}{pct(r['success_rate']):>8}"
                   f"{pct(r['target_rate']):>9}{pct(r['already_rate']):>9}"
                   f"{num(r['mfe_atr_median'], 2):>9}{num(r['mae_atr_median'], 2):>9}")
    return out
