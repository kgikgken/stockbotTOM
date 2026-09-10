"""反転系のチャートパターン検出（docs/PATTERN.md §2.1）。

確定済みの交互スイング（`swings.py`）から、ダブルボトム・トリプルボトム・逆三尊を
検出する。**成立するのは終値がネックラインを上抜けた日 T** で、極値が揃っただけでは
成立しない（§2.1 共通）。

**未来参照はしない。** 使うのは `swings_as_of(alternated, t_pos)` が返す確定済みの
極値と、`Close[t_pos]` だけ。T より後の足を足しても結果は変わらない（CLAUDE.md）。

**確定ラグは `swings.py` の k だけ。** LMW の確定ラグ 3 本は重ねない —— 重ねると
6 本遅れる（§4 Q-1 の回答）。

**ネックラインは水平線のみ。** 2 点を結ぶ斜めの線は使わない。傾きを許すと「許容傾き」
という新規パラメータが要り、ε を流用すると意味が変わる（§4 Q-1 の回答）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .swings import SWING_HIGH, SWING_LOW, alternate_swings, detect_raw_swings, swings_as_of

# ------------------------------------------------------------------ 事前登録の値
# docs/PATTERN.md §1。**検出数を見てから動かさない**（D-3・D-5）
# 出典のある値（文献値）
EQUAL_TOL = 0.015           # 等値とみなす幅 ±1.5%（LMW）。ATR 連動にしない
SEARCH_WINDOW = 63          # 探索ウィンドウ（営業日・Savin-Weller-Zvingelis 2007）
DOUBLE_BOTTOM_GAP = 22      # ダブルボトムの 2 安値の最小間隔（営業日・LMW）
# 裁量値（文献に数値が無いことを確認済み。docs/PATTERN.md §1）
ADJACENT_TROUGH_GAP = 10    # トリプル／逆三尊の隣接する谷の最小間隔（営業日）

DOUBLE_BOTTOM = "double_bottom"
TRIPLE_BOTTOM = "triple_bottom"
INVERSE_HS = "inverse_hs"

PATTERN_COLS = [
    "pattern",      # double_bottom / triple_bottom / inverse_hs
    "t_pos",        # 成立日 T（終値がネックラインを上抜けた日）
    "neckline",     # 水平なネックラインの値
    "close_t",      # Close[T]
    "l1_pos", "l1",     # 左の谷
    "l2_pos", "l2",     # 中央の谷（ダブルボトムでは右の谷）
    "l3_pos", "l3",     # 右の谷（ダブルボトムでは欠損）
    "h1_pos", "h1",     # 左のネックライン点
    "h2_pos", "h2",     # 右のネックライン点（ダブルボトムでは欠損）
    "span",         # 使った極値の最初から T までの本数
    # 撤退・目標・比率（docs/PATTERN.md §2.3）。**測定目標は文献上の目安であって、
    # 統計的裏付けは調べていない。** 比率を併記して読み手が自分で判断できるようにする
    "pattern_low",  # 撤退の目安。パターンの最安値
    "height",       # ネックライン − 最安値。測定目標の投影幅
    "target",       # ネックライン + 高さ（測定目標・文献上の目安）
    "breakout_pct",   # (終値 / ネックライン − 1) × 100。負なら未抜け
    "breakout_h",     # 抜け幅 ÷ 高さ。比率 = (1 − b) / (1 + b) の b
    "up_pct",         # (目標 / 終値 − 1) × 100
    "down_pct",       # (最安値 / 終値 − 1) × 100（負）
    "rr",             # up ÷ |down|。**1 未満なら目標のほうが撤退より近い**
    "breakeven_win_rate",   # 1 / (1 + rr)。これを上回る勝率が無いと期待値が負になる
    # 形は揃っているか、ネックラインも抜けたか。**形だけ揃った行を数えるための列。**
    # 検出 0 件だったときに「条件が厳しい」のか「実装が間違っている」のかを
    # 1 回で切り分けるために置いた（設計責任者の指示、docs/PATTERN.md §2.1）
    "breakout",
]


def _within(values, tol: float = EQUAL_TOL) -> bool:
    """各値が平均から ±tol 以内か（§2.1 の「等値」）。平均が 0 以下なら False。"""
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        return False
    mean = float(arr.mean())
    if mean <= 0:
        return False
    return bool(np.max(np.abs(arr - mean)) / mean <= tol)


def _tail_extremes(swings: pd.DataFrame, n: int) -> Optional[list]:
    """末尾 n 個の極値を古い順に返す。足りなければ None。

    交互化済みなので安値と高値は必ず交互に並ぶ。**末尾が安値であること**だけ確かめる
    （反転系はどれも右端が谷で終わる形）。
    """
    if len(swings) < n:
        return None
    tail = swings.iloc[-n:]
    if tail["kind"].iloc[-1] != SWING_LOW:
        return None
    kinds = tail["kind"].tolist()
    # 交互になっていること（生の取り違えを弾く）
    if any(a == b for a, b in zip(kinds, kinds[1:])):
        return None
    return [(int(r["index"]), float(r["value"]), str(r["kind"]))
            for _i, r in tail.iterrows()]


def _row(pattern: str, t_pos: int, neckline: float, close_t: float,
         lows: list, highs: list) -> dict:
    """PATTERN_COLS の 1 行。使わない列は欠損にする。

    `breakout` はここで決める —— 形の判定（極値の並び・等値・間隔・ウィンドウ）を
    通ったあと、終値がネックラインを上抜けたかだけを見る。**形が揃っただけの行も
    作る**ので、呼び出し側が数え分けられる。
    """
    def at(seq, i):
        return seq[i] if i < len(seq) else (np.nan, np.nan)

    (l1p, l1), (l2p, l2), (l3p, l3) = at(lows, 0), at(lows, 1), at(lows, 2)
    (h1p, h1), (h2p, h2) = at(highs, 0), at(highs, 1)
    first = min(p for p, _v in lows + highs)
    return {
        "pattern": pattern, "t_pos": int(t_pos),
        "neckline": float(neckline), "close_t": float(close_t),
        "l1_pos": l1p, "l1": l1, "l2_pos": l2p, "l2": l2, "l3_pos": l3p, "l3": l3,
        "h1_pos": h1p, "h1": h1, "h2_pos": h2p, "h2": h2,
        "span": int(t_pos - first),
        "breakout": bool(close_t > neckline),
        **measured_move(neckline, close_t, [v for _p, v in lows]),
    }


MEASURED_MOVE_COLS = ["pattern_low", "height", "target", "breakout_pct", "breakout_h",
                      "up_pct", "down_pct", "rr", "breakeven_win_rate"]


def measured_move(neckline: float, close_t: float, lows) -> dict:
    """撤退・測定目標・比率（docs/PATTERN.md §2.3）。

    **測定目標は文献上の目安であって、統計的裏付けは調べていない。** ネックラインから
    パターンの最安値までの値幅を、ネックラインの上に同じだけ投影したもの。

    - 撤退の目安 = パターンの最安値
    - 高さ `h` = ネックライン − 最安値
    - 測定目標 = ネックライン + h
    - 抜け幅を高さで割った `b` に対し、**比率 = (1 − b) / (1 + b)**

    **抜けた直後（b ≈ 0）なら比率 ≈ 1.0** で、大きく抜けてから買うほど下がる
    （b=0.05 で 0.90、b=0.20 で 0.67）。抜け幅の下限を入れると、シグナルの確からしさと
    引き換えに比率を削ることになる（§5 D-7）。

    T の引けまでの値だけで決まる。結果ではない。
    """
    nan = {c: np.nan for c in MEASURED_MOVE_COLS}
    finite = [float(v) for v in lows if v is not None and np.isfinite(v)]
    if not finite or not (np.isfinite(neckline) and np.isfinite(close_t) and close_t > 0):
        return nan
    low = min(finite)
    height = neckline - low
    out = dict(nan)
    out["pattern_low"] = low
    out["height"] = float(height)
    out["breakout_pct"] = float((close_t / neckline - 1.0) * 100) if neckline > 0 else np.nan
    if height <= 0:
        return out          # ネックラインが最安値以下。高さが定義できない
    target = neckline + height
    up = target / close_t - 1.0
    down = low / close_t - 1.0
    out["target"] = float(target)
    out["breakout_h"] = float((close_t - neckline) / height)
    out["up_pct"] = float(up * 100)
    out["down_pct"] = float(down * 100)
    if down < 0 and up > 0:
        rr = up / -down
        out["rr"] = float(rr)
        out["breakeven_win_rate"] = float(1.0 / (1.0 + rr))
    return out


def _double_bottom(swings: pd.DataFrame, t_pos: int, close_t: float) -> Optional[dict]:
    """安値 L1 → 高値 H → 安値 L2（docs/PATTERN.md §2.1 R1）。"""
    ext = _tail_extremes(swings, 3)
    if ext is None or ext[0][2] != SWING_LOW:
        return None
    (l1p, l1, _), (hp, h, _), (l2p, l2, _) = ext
    if l2p - l1p < DOUBLE_BOTTOM_GAP:
        return None
    if not _within([l1, l2]):
        return None
    if t_pos - l1p > SEARCH_WINDOW:
        return None
    return _row(DOUBLE_BOTTOM, t_pos, h, close_t, [(l1p, l1), (l2p, l2)], [(hp, h)])


def _five(swings: pd.DataFrame, t_pos: int):
    """反転系の 5 極値（安・高・安・高・安）を古い順に。使えなければ None。"""
    ext = _tail_extremes(swings, 5)
    if ext is None or ext[0][2] != SWING_LOW:
        return None
    (e1p, e1, _), (e2p, e2, _), (e3p, e3, _), (e4p, e4, _), (e5p, e5, _) = ext
    if e3p - e1p < ADJACENT_TROUGH_GAP or e5p - e3p < ADJACENT_TROUGH_GAP:
        return None
    if t_pos - e1p > SEARCH_WINDOW:
        return None
    return (e1p, e1), (e2p, e2), (e3p, e3), (e4p, e4), (e5p, e5)


def _triple_bottom(swings: pd.DataFrame, t_pos: int, close_t: float) -> Optional[dict]:
    """3 谷が平均の ±1.5% 以内（docs/PATTERN.md §2.1 R2）。"""
    five = _five(swings, t_pos)
    if five is None:
        return None
    (e1p, e1), (e2p, e2), (e3p, e3), (e4p, e4), (e5p, e5) = five
    if not _within([e1, e3, e5]):
        return None
    neck = max(e2, e4)
    return _row(TRIPLE_BOTTOM, t_pos, neck, close_t,
                [(e1p, e1), (e3p, e3), (e5p, e5)], [(e2p, e2), (e4p, e4)])


def _inverse_hs(swings: pd.DataFrame, t_pos: int, close_t: float) -> Optional[dict]:
    """両肩が等値・頭が両肩より低い（docs/PATTERN.md §2.1 R3）。深さの下限は無い。"""
    five = _five(swings, t_pos)
    if five is None:
        return None
    (e1p, e1), (e2p, e2), (e3p, e3), (e4p, e4), (e5p, e5) = five
    if not _within([e1, e5]):
        return None          # 両肩
    if not _within([e2, e4]):
        return None          # ネックライン点
    if not (e3 < e1 and e3 < e5):
        return None          # 頭。深さの下限は設けない（§2.1 R3）
    neck = max(e2, e4)
    return _row(INVERSE_HS, t_pos, neck, close_t,
                [(e1p, e1), (e3p, e3), (e5p, e5)], [(e2p, e2), (e4p, e4)])


def detect_patterns(high: pd.Series, low: pd.Series, close: pd.Series, t_pos: int,
                    k: int = 3, alternated: Optional[pd.DataFrame] = None,
                    include_pending: bool = False) -> pd.DataFrame:
    """T 時点で成立している反転系パターンを返す（docs/PATTERN.md §2.1）。

    high/low/close は 0 始まりの位置で扱う整列済み Series（features 内の他モジュールと
    同じ規約）。t_pos は判定日 T の位置。**読むのは T までのデータだけ。**

    alternated を渡すと `alternate_swings` を再実行しない（多数の T を回すとき用）。

    **既定では成立した行だけを返す**（`breakout` が True）。`include_pending=True` に
    すると、形は揃っているがネックラインをまだ抜けていない行も返す。検出 0 件だった
    ときに「条件が厳しい」のか「実装が間違っている」のかを切り分けるためのもので、
    **配信や記録には使わない**（成立の定義は §2.1 共通のまま変えていない）。

    **同じ 5 極値が R2 と R3 の両方に該当したら両方返す**（§2.1）。どちらかに寄せない。
    戻り値は PATTERN_COLS の順で、pattern 名の昇順。
    """
    if alternated is None:
        alternated = alternate_swings(detect_raw_swings(high, low, k))
    if t_pos < 0 or t_pos >= len(close):
        return pd.DataFrame(columns=PATTERN_COLS)
    swings = swings_as_of(alternated, t_pos)
    if len(swings) == 0:
        return pd.DataFrame(columns=PATTERN_COLS)

    close_t = float(close.iloc[t_pos])
    if not np.isfinite(close_t):
        return pd.DataFrame(columns=PATTERN_COLS)

    rows = [f(swings, t_pos, close_t)
            for f in (_double_bottom, _triple_bottom, _inverse_hs)]
    hits = [r for r in rows if r is not None]
    if not include_pending:
        hits = [r for r in hits if r["breakout"]]
    if not hits:
        return pd.DataFrame(columns=PATTERN_COLS)
    out = pd.DataFrame(hits)[PATTERN_COLS]
    return out.sort_values("pattern").reset_index(drop=True)
