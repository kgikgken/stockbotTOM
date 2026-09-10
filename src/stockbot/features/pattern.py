"""チャートパターン検出（docs/PATTERN.md §2.1 反転系・§2.2 保ち合い系）。

確定済みの交互スイング（`swings.py`）から 7 パターンを検出する。反転系はダブルボトム・
トリプルボトム・逆三尊、保ち合い系は上昇三角・上昇ボックス・上昇フラッグ・上昇ペナント。
**成立するのは終値が上値抵抗線（反転系ではネックライン）を上抜けた日 T** で、
極値が揃っただけでは成立しない（§2.1 共通）。

**未来参照はしない。** 使うのは `swings_as_of(alternated, t_pos)` が返す確定済みの
極値と、`Close[t_pos]` だけ。T より後の足を足しても結果は変わらない（CLAUDE.md）。

**確定ラグは `swings.py` の k だけ。** LMW の確定ラグ 3 本は重ねない —— 重ねると
6 本遅れる（§4 Q-1 の回答）。

**ネックラインは水平線のみ。** 2 点を結ぶ斜めの線は使わない。傾きを許すと「許容傾き」
という新規パラメータが要り、ε を流用すると意味が変わる（§4 Q-1 の回答）。
保ち合い系の上辺・下辺は回帰直線で、これは §2.2 の判定式が傾きを要求しているため
（ε で水平かを判定する）。反転系のネックラインとは別物である。

**保ち合い系で仕様に書かれていなかった読み方**（実装上の判断。新しい閾値は作っていない。
docs/PATTERN.md §2.2 の「実装上の読み」に同じものを書いてある）:

1. タッチ点は反転系と同じ確定済み交互スイングの**末尾 5 極値**。交互なので必ず
   3 + 2 に割れる（§1 の「3 + 2（最小 5）」）。どちらの辺が 3 点かは形による
2. 傾きは最小二乗の回帰直線の傾きを、その辺の平均値で割った**日次の比率**。
   ε（日次 0.1%）が比率で与えられているため
3. `終値 > 上辺` の上辺は、回帰直線を T まで延ばした値（ボックスだけは上辺の平均）
4. フラッグ／ペナントの営業日数は `span`（最初の極値から T まで）で数える
5. 旗竿は「最初の極値の終値 ÷ その 5 本前までの各終値 − 1」の最大値
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
BOX_TOL = 0.0075            # ボックスの水平許容 ±0.75%（LMW）。EQUAL_TOL とは別の定数
SEARCH_WINDOW = 63          # 探索ウィンドウ（営業日・Savin-Weller-Zvingelis 2007）
DOUBLE_BOTTOM_GAP = 22      # ダブルボトムの 2 安値の最小間隔（営業日・LMW）
TOUCH_POINTS = 5            # 2 本の線を引くタッチ点数 3 + 2（最小 5）
FLAG_MAX_SPAN = 15          # フラッグは「未満」・ペナントは「以下」（営業日）
# 裁量値（文献に数値が無いことを確認済み。docs/PATTERN.md §1）
ADJACENT_TROUGH_GAP = 10    # トリプル／逆三尊の隣接する谷の最小間隔（営業日）
EPSILON_SLOPE = 0.001       # 水平とみなす日次の傾き（比率）。ε＝日次 0.1% 未満
POLE_LOOKBACK = 5           # 旗竿を探す本数（営業日）
POLE_RISE = 0.10            # 旗竿の上昇率 10%

DOUBLE_BOTTOM = "double_bottom"
TRIPLE_BOTTOM = "triple_bottom"
INVERSE_HS = "inverse_hs"
ASCENDING_TRIANGLE = "ascending_triangle"
ASCENDING_BOX = "ascending_box"
BULL_FLAG = "bull_flag"
BULL_PENNANT = "bull_pennant"

REVERSAL_PATTERNS = (DOUBLE_BOTTOM, TRIPLE_BOTTOM, INVERSE_HS)
CONSOLIDATION_PATTERNS = (ASCENDING_TRIANGLE, ASCENDING_BOX, BULL_FLAG, BULL_PENNANT)

PATTERN_COLS = [
    "pattern",      # REVERSAL_PATTERNS + CONSOLIDATION_PATTERNS のいずれか
    "t_pos",        # 成立日 T（終値が上値抵抗線を上抜けた日）
    "neckline",     # 反転系は水平なネックライン。**保ち合い系は上辺を T まで延ばした値**
    "close_t",      # Close[T]
    "l1_pos", "l1",     # 左の谷（保ち合い系では下辺のタッチ点）
    "l2_pos", "l2",     # 中央の谷（ダブルボトムでは右の谷）
    "l3_pos", "l3",     # 右の谷（谷が 2 つなら欠損）
    "h1_pos", "h1",     # 左のネックライン点（保ち合い系では上辺のタッチ点）
    "h2_pos", "h2",     # 右のネックライン点（ダブルボトムでは欠損）
    "h3_pos", "h3",     # 3 つ目の山（上辺が 3 タッチの保ち合い系だけ）
    "span",         # 使った極値の最初から T までの本数
    # 保ち合い系の線と旗竿（docs/PATTERN.md §2.2）。反転系では欠損
    "upper_slope",  # 上辺の日次傾き（比率）。ε と比べる量
    "lower_slope",  # 下辺の日次傾き（比率）
    "pole_pct",     # 旗竿。5 営業日以内の最大上昇率（%）。フラッグ／ペナントだけ
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


LINE_COLS = ["upper_slope", "lower_slope", "pole_pct"]


def _row(pattern: str, t_pos: int, neckline: float, close_t: float,
         lows: list, highs: list, lines: Optional[dict] = None) -> dict:
    """PATTERN_COLS の 1 行。使わない列は欠損にする。

    `breakout` はここで決める —— 形の判定（極値の並び・等値・間隔・ウィンドウ、
    保ち合い系では傾きと旗竿）を通ったあと、終値が上値抵抗線を上抜けたかだけを見る。
    **形が揃っただけの行も作る**ので、呼び出し側が数え分けられる。

    `lines` は保ち合い系の上辺・下辺の傾きと旗竿（反転系では None＝欠損）。
    """
    def at(seq, i):
        return seq[i] if i < len(seq) else (np.nan, np.nan)

    (l1p, l1), (l2p, l2), (l3p, l3) = at(lows, 0), at(lows, 1), at(lows, 2)
    (h1p, h1), (h2p, h2), (h3p, h3) = at(highs, 0), at(highs, 1), at(highs, 2)
    first = min(p for p, _v in lows + highs)
    return {
        "pattern": pattern, "t_pos": int(t_pos),
        "neckline": float(neckline), "close_t": float(close_t),
        "l1_pos": l1p, "l1": l1, "l2_pos": l2p, "l2": l2, "l3_pos": l3p, "l3": l3,
        "h1_pos": h1p, "h1": h1, "h2_pos": h2p, "h2": h2, "h3_pos": h3p, "h3": h3,
        "span": int(t_pos - first),
        **{c: np.nan for c in LINE_COLS}, **(lines or {}),
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


# ------------------------------------------------------------------ 保ち合い系
# docs/PATTERN.md §2.2。上辺・下辺の 2 本を回帰で引き、傾きで 4 つを区別する。
# **反転系と違って線に傾きがある。** ε（日次 0.1%）が「水平とみなす」判定を担う


def _slope(points: list) -> float:
    """タッチ点に最小二乗直線を当て、**日次の傾きを比率で**返す（§2.2 実装上の読み 2）。

    ε が「日次 0.1%」という比率で与えられているので、価格/本の傾きをその辺の平均値で
    割って比率に直す。平均が 0 以下なら NaN（比較は必ず False になる）。
    """
    xs = np.asarray([p for p, _v in points], dtype=float)
    ys = np.asarray([v for _p, v in points], dtype=float)
    mean = float(ys.mean())
    if not np.isfinite(ys).all() or mean <= 0:
        return float("nan")
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope / mean


def _line_at(points: list, t_pos: int) -> float:
    """タッチ点の回帰直線を T まで延ばした値（§2.2 実装上の読み 3）。"""
    xs = np.asarray([p for p, _v in points], dtype=float)
    ys = np.asarray([v for _p, v in points], dtype=float)
    if not np.isfinite(ys).all():
        return float("nan")
    slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
    return intercept + slope * float(t_pos)


def _touches(swings: pd.DataFrame, t_pos: int):
    """保ち合い系のタッチ点。**末尾 5 極値**を上辺・下辺に分ける（§2.2 実装上の読み 1）。

    交互化済みなので 5 極値は必ず 3 + 2 に割れる（§1 の「3 + 2（最小 5）」）。
    どちらの辺が 3 点になるかは形による —— 反転系と違って**末尾が安値／高値のどちらで
    終わるかは問わない**（保ち合いは右端が谷とは限らない）。

    戻り値は (最初の極値の位置, 上辺のタッチ点, 下辺のタッチ点)。使えなければ None。
    """
    if len(swings) < TOUCH_POINTS:
        return None
    tail = swings.iloc[-TOUCH_POINTS:]
    kinds = tail["kind"].tolist()
    if any(a == b for a, b in zip(kinds, kinds[1:])):
        return None          # 交互になっていない（生の取り違えを弾く）
    pts = [(int(r["index"]), float(r["value"]), str(r["kind"]))
           for _i, r in tail.iterrows()]
    first = pts[0][0]
    if t_pos - first > SEARCH_WINDOW:
        return None
    highs = [(pos, v) for pos, v, kind in pts if kind == SWING_HIGH]
    lows = [(pos, v) for pos, v, kind in pts if kind == SWING_LOW]
    if len(highs) < 2 or len(lows) < 2:
        return None          # 3 + 2 に割れていない
    return first, highs, lows


def _pole(close: pd.Series, start_pos: int) -> float:
    """旗竿。**最初の極値までの 5 営業日以内の最大上昇率（%）**（§2.2 実装上の読み 5）。

    `close[start] / close[start − j] − 1`（j = 1..5）の最大値。無ければ NaN。
    読むのは start_pos までの終値だけで、T より後は見ない。
    """
    if start_pos <= 0:
        return float("nan")
    base = float(close.iloc[start_pos])
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    best = float("nan")
    for j in range(1, POLE_LOOKBACK + 1):
        pos = start_pos - j
        if pos < 0:
            break
        prev = float(close.iloc[pos])
        if not np.isfinite(prev) or prev <= 0:
            continue
        rise = base / prev - 1.0
        if not np.isfinite(best) or rise > best:
            best = rise
    return best * 100 if np.isfinite(best) else float("nan")


def _ascending_triangle(swings: pd.DataFrame, t_pos: int, close_t: float,
                        close: pd.Series) -> Optional[dict]:
    """上辺が水平・下辺が上向き（docs/PATTERN.md §2.2 C1）。

    `|上辺傾き| < ε` かつ `下辺傾き > 0`。下辺に ε の下限は置かない（判定式のまま）。
    """
    got = _touches(swings, t_pos)
    if got is None:
        return None
    _first, highs, lows = got
    upper, lower = _slope(highs), _slope(lows)
    if not (abs(upper) < EPSILON_SLOPE and lower > 0):
        return None
    return _row(ASCENDING_TRIANGLE, t_pos, _line_at(highs, t_pos), close_t, lows, highs,
                {"upper_slope": upper, "lower_slope": lower})


def _ascending_box(swings: pd.DataFrame, t_pos: int, close_t: float,
                   close: pd.Series) -> Optional[dict]:
    """上辺・下辺とも水平で重ならない（docs/PATTERN.md §2.2 C2）。

    上辺の各点が上辺平均の **±0.75% 以内**、下辺も同様、`最低の山 > 最高の谷`。
    **ここだけ上辺は回帰直線ではなく水平（上辺の平均）**である —— 判定式が傾きでは
    なく「平均からの距離」で水平さを要求しているため（§2.2 実装上の読み 3）。
    """
    got = _touches(swings, t_pos)
    if got is None:
        return None
    _first, highs, lows = got
    hv = [v for _p, v in highs]
    lv = [v for _p, v in lows]
    if not _within(hv, BOX_TOL) or not _within(lv, BOX_TOL):
        return None
    if not min(hv) > max(lv):
        return None
    return _row(ASCENDING_BOX, t_pos, float(np.mean(hv)), close_t, lows, highs,
                {"upper_slope": _slope(highs), "lower_slope": _slope(lows)})


def _bull_flag(swings: pd.DataFrame, t_pos: int, close_t: float,
               close: pd.Series) -> Optional[dict]:
    """旗竿のあとの平行な下降チャネル（docs/PATTERN.md §2.2 C3）。

    上辺・下辺とも `傾き < 0`、傾き差が ε 未満、**15 営業日未満**。
    ペナント（C4）は「15 営業日以下」で、境界の扱いが違う。
    """
    got = _touches(swings, t_pos)
    if got is None:
        return None
    first, highs, lows = got
    upper, lower = _slope(highs), _slope(lows)
    if not (upper < 0 and lower < 0):
        return None
    if not abs(upper - lower) < EPSILON_SLOPE:
        return None
    if not t_pos - first < FLAG_MAX_SPAN:      # 未満
        return None
    pole = _pole(close, first)
    if not pole >= POLE_RISE * 100:            # NaN なら False
        return None
    return _row(BULL_FLAG, t_pos, _line_at(highs, t_pos), close_t, lows, highs,
                {"upper_slope": upper, "lower_slope": lower, "pole_pct": pole})


def _bull_pennant(swings: pd.DataFrame, t_pos: int, close_t: float,
                  close: pd.Series) -> Optional[dict]:
    """旗竿のあとの収束（docs/PATTERN.md §2.2 C4）。

    `上辺傾き < 0` かつ `下辺傾き > 0`、**15 営業日以下**。
    フラッグ（C3）は「15 営業日未満」で、境界の扱いが違う。
    """
    got = _touches(swings, t_pos)
    if got is None:
        return None
    first, highs, lows = got
    upper, lower = _slope(highs), _slope(lows)
    if not (upper < 0 and lower > 0):
        return None
    if not t_pos - first <= FLAG_MAX_SPAN:     # 以下
        return None
    pole = _pole(close, first)
    if not pole >= POLE_RISE * 100:            # NaN なら False
        return None
    return _row(BULL_PENNANT, t_pos, _line_at(highs, t_pos), close_t, lows, highs,
                {"upper_slope": upper, "lower_slope": lower, "pole_pct": pole})


def detect_patterns(high: pd.Series, low: pd.Series, close: pd.Series, t_pos: int,
                    k: int = 3, alternated: Optional[pd.DataFrame] = None,
                    include_pending: bool = False) -> pd.DataFrame:
    """T 時点で成立しているパターンを返す（docs/PATTERN.md §2.1・§2.2）。

    high/low/close は 0 始まりの位置で扱う整列済み Series（features 内の他モジュールと
    同じ規約）。t_pos は判定日 T の位置。**読むのは T までのデータだけ。**

    alternated を渡すと `alternate_swings` を再実行しない（多数の T を回すとき用）。

    **既定では成立した行だけを返す**（`breakout` が True）。`include_pending=True` に
    すると、形は揃っているがネックラインをまだ抜けていない行も返す。検出 0 件だった
    ときに「条件が厳しい」のか「実装が間違っている」のかを切り分けるためのもので、
    **配信や記録には使わない**（成立の定義は §2.1 共通のまま変えていない）。

    **同じ極値が複数のパターンに該当したら全部返す**（§2.1）。どちらかに寄せない。
    R2 と R3 に限らず、保ち合い系どうし（水平な上辺は C1 と C2 の両方に該当しうる）でも、
    反転系と保ち合い系の間でも同じ扱いにする。**どの形として出やすいかは、検出数を
    見てから設計責任者が判断する。**

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
    rows += [f(swings, t_pos, close_t, close)
             for f in (_ascending_triangle, _ascending_box, _bull_flag, _bull_pennant)]
    hits = [r for r in rows if r is not None]
    if not include_pending:
        hits = [r for r in hits if r["breakout"]]
    if not hits:
        return pd.DataFrame(columns=PATTERN_COLS)
    out = pd.DataFrame(hits)[PATTERN_COLS]
    return out.sort_values("pattern").reset_index(drop=True)
