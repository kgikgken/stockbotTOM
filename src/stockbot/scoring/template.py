"""テンプレート適合度 d3_template（DESIGN.md §7 / TASKS.md T-302）。

現状の実装範囲: 価格テンプレート（格子化・滞在比率 P・理想経路 W・適合度）のみ。
出来高テンプレート（Vol/mean(Vol[l0..h0]) の格子で「上昇脚で高く、押し目で減衰」の
理想経路）は DESIGN.md §7 に理想経路の具体的な数値アンカー（どの比率域を狙うか）が
書かれておらず、CLAUDE.md「パラメータを増やさない・新しい閾値が欲しければ質問ログへ」
に従い実装していない（docs/TASKS.md 質問ログ T-302 参照、設計責任者の回答待ち）。
d3_template（価格と出来高の平均、DESIGN.md §7）はこの回答が出るまで確定できないため、
features/dimensions.py にはまだ配線していない（引き続き NaN のまま）。

窓は l0..T（両端含む）。横軸（時間）は l0→0、H0→a、T→1 となる二区間線形写像
（a = leg_bars/(leg_bars+d)）で 0〜1 に正規化し、10 等分した列に落とす。
縦軸（価格位置）は y=(Close-L0.low)/leg（L0が0、H0が1）を -0.2〜1.2 の範囲で
10 等分（幅0.14）した行に落とす。
"""
from __future__ import annotations

import numpy as np

GRID_N = 10
Y_LO, Y_HI = -0.2, 1.2
Y_BIN_WIDTH = (Y_HI - Y_LO) / GRID_N  # 0.14

# DESIGN.md §7: W の初期値（固定の重み定数。これ以外の数値を増やさない）
W_PATH = 1.0
W_ADJACENT = 0.5
W_BELOW_L0 = -1.0
W_ABOVE_H0_EARLY_PULLBACK = -0.5

# DESIGN.md §7: 押し目の理想経路が下がっていく先（y=0.6〜0.67）。行の丸め込み先として
# 区間の下端寄りである行5（y=0.50〜0.64）を使う（0.6 はこの行に、0.67 はわずかに行6に
# はみ出るが、行の刻み幅0.14に対して丸め誤差の範囲として行5に統一する）
PB_DECLINE_TARGET_ROW = 5


def time_column(t_norm: float) -> int:
    """正規化時間（0〜1）を列インデックス（0〜9）に落とす。"""
    return int(np.clip(np.floor(t_norm * GRID_N), 0, GRID_N - 1))


def y_row(y: float) -> int:
    """正規化価格位置 y を行インデックス（0〜9）に落とす。"""
    return int(np.clip(np.floor((y - Y_LO) / Y_BIN_WIDTH), 0, GRID_N - 1))


def normalized_time(i: int, l0: int, h0: int, t_pos: int, leg_bars: int, d: int) -> float:
    """DESIGN.md §7 の二区間線形写像。l0→0, h0→a, T→1（a=leg_bars/(leg_bars+d)）。"""
    a = leg_bars / (leg_bars + d)
    if i < h0:
        return a * (i - l0) / leg_bars
    return a + (1 - a) * (i - h0) / d


def _pullback_column_split(leg_bars: int, d: int) -> tuple[list[int], list[int]]:
    """境界 a を含む列から後ろを押し目の列とする（H0 自身の足は必ず a ちょうどに
    写像されるため）。戻り値は (上昇脚の列, 押し目の列)。"""
    a = leg_bars / (leg_bars + d)
    col_start = int(min(GRID_N - 1, np.floor(a * GRID_N)))
    return list(range(0, col_start)), list(range(col_start, GRID_N))


def build_price_grid(close: np.ndarray, l0: int, h0: int, t_pos: int,
                     l0_low: float, leg: float, leg_bars: int, d: int) -> np.ndarray:
    """P[row][col] を作る（列ごとに合計1。データが無い列は全0のまま）。

    leg<=0（H0<=L0、構造として不正）の場合は全て NaN の格子を返す。
    """
    if leg <= 0 or leg_bars < 1 or d < 1:
        return np.full((GRID_N, GRID_N), np.nan)
    grid = np.zeros((GRID_N, GRID_N))
    counts = np.zeros(GRID_N)
    for i in range(l0, t_pos + 1):
        t_norm = normalized_time(i, l0, h0, t_pos, leg_bars, d)
        col = time_column(t_norm)
        y = (float(close[i]) - l0_low) / leg
        row = y_row(y)
        grid[row, col] += 1
        counts[col] += 1
    for col in range(GRID_N):
        if counts[col] > 0:
            grid[:, col] /= counts[col]
    return grid


def build_price_weight(leg_bars: int, d: int) -> np.ndarray:
    """DESIGN.md §7 の理想経路 W（価格）。この案件の a（leg_bars, d 由来）で列を
    上昇脚/押し目に分けたうえで組み立てる。"""
    up_cols, pb_cols = _pullback_column_split(leg_bars, d)
    W = np.zeros((GRID_N, GRID_N))

    def _place(row: float, col: int) -> None:
        row_i = int(np.clip(round(row), 0, GRID_N - 1))
        W[row_i, col] += W_PATH
        if row_i - 1 >= 0:
            W[row_i - 1, col] += W_ADJACENT
        if row_i + 1 < GRID_N:
            W[row_i + 1, col] += W_ADJACENT

    # 上昇脚: 左下（row=1, y≈0付近）から右上（row=8, y≈1付近）への対角経路
    n_up = len(up_cols)
    for j, col in enumerate(up_cols):
        row = 1 + (8 - 1) * j / (n_up - 1) if n_up > 1 else 8
        _place(row, col)

    # 押し目: row=8(y≈1)から row=5(y≈0.6〜0.67)へ緩やかに下降し、最後の2列は水平
    n_pb = len(pb_cols)
    flat_cols = pb_cols[-2:] if n_pb >= 2 else list(pb_cols)
    decline_cols = pb_cols[:-2] if n_pb > 2 else []
    n_decline = len(decline_cols)
    for j, col in enumerate(decline_cols):
        row = 8 - (8 - PB_DECLINE_TARGET_ROW) * j / (n_decline - 1) if n_decline > 1 else 8
        _place(row, col)
    for col in flat_cols:
        _place(PB_DECLINE_TARGET_ROW, col)

    # y<0（L0割れ）の全セル: 行0（-0.20〜-0.06、全域が0未満）は列によらず -1
    W[0, :] = W_BELOW_L0
    # 押し目前半の列で y>1.0 のセル: 行9（1.06〜1.20、全域が1.0超）
    first_half_pb = pb_cols[: max(1, n_pb // 2)]
    for col in first_half_pb:
        W[9, col] = W_ABOVE_H0_EARLY_PULLBACK

    return W


def price_template_score(close: np.ndarray, l0: int, h0: int, t_pos: int,
                         l0_low: float, leg: float, leg_bars: int, d: int) -> float:
    """価格テンプレート適合度（0〜1）。窓が定義できない場合は NaN。"""
    if leg <= 0 or leg_bars < 1 or d < 1:
        return np.nan
    P = build_price_grid(close, l0, h0, t_pos, l0_low, leg, leg_bars, d)
    if np.isnan(P).all():
        return np.nan
    W = build_price_weight(leg_bars, d)
    positive_sum = W[W > 0].sum()
    if positive_sum <= 0:
        return np.nan
    score = float((W * P).sum() / positive_sum)
    return float(np.clip(score, 0.0, 1.0))
