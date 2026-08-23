"""スイング検出（DESIGN.md §3, §11 / TASKS.md T-202）。

- 生スイング検出（フラクタル）: 全期間に対して一度だけ行う。各スイングに
  confirm_index = i+k を付ける。フラクタル条件は i−k..i+k のバーだけで決まる
  純粋に因果的（未来を必要とするのは確定までの k 本だけ）な判定なので、
  confirm_index ≤ T で絞れば「T 時点で使える生スイング集合」になる。
- 交互化: 生スイングを confirm_index（＝時間）順に走査する逐次更新アルゴリズム。
  末尾と同種の新しいスイングが来たら、より極端な方を残す（新しい方が極端なら
  末尾を置換し、そうでなければ新しい方を捨てる。同値なら先に来た方＝末尾を残す）。
  異種なら末尾に追加する。

  T が進んで末尾が置き換わることがあるが、これは新たに確定した情報による更新
  であり未来参照ではない（DESIGN.md §3 追記、質問ログ 2026-08-22 T-202 回答）。
  置換は常に末尾の1要素にのみ起こる: i1 と i2 (同種、i1 < i2、i2 の方が極端) の
  間に反対側の生スイング j があれば、j+k < i2+k なので j は i2 より先に確定して
  末尾に追加され、i1 は置換対象から外れる。

  1 回の全走査（O(n)）で「いつ採用され（kept）、いつ置換されて外れたか
  （superseded_at）」を記録したテーブルを作る。任意の T の交互化後スイング集合は
      kept & confirm_index <= T & (superseded_at is NA or superseded_at > T)
  という O(1) フィルタで復元できる（前向き検証・再生 T-402 用）。
"""
from __future__ import annotations

import pandas as pd

SWING_HIGH = "high"
SWING_LOW = "low"

RAW_COLS = ["index", "kind", "value", "confirm_index"]
ALTERNATED_COLS = RAW_COLS + ["kept", "superseded_at"]


def detect_raw_swings(high: pd.Series, low: pd.Series, k: int = 3) -> pd.DataFrame:
    """全期間の生フラクタルスイングを検出する（交互化前、DESIGN.md §3）。

    スイング高値: High[i] >= max(High[i-k..i-1]) かつ High[i] > max(High[i+1..i+k])
    スイング安値: Low[i]  <= min(Low[i-k..i-1])  かつ Low[i]  < min(Low[i+1..i+k])

    引数の high/low は 0 始まりの連番に相当する位置で扱う（DESIGN.md の i, k, T は
    すべて営業日の本数＝バー位置の整数）。戻り値の index / confirm_index はその
    位置（0-indexed）。列: index, kind, value, confirm_index。
    """
    n = len(high)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    rows = []
    for i in range(k, n - k):
        left_h, right_h = h[i - k:i], h[i + 1:i + 1 + k]
        if h[i] >= left_h.max() and h[i] > right_h.max():
            rows.append((i, SWING_HIGH, h[i], i + k))
        left_l, right_l = l[i - k:i], l[i + 1:i + 1 + k]
        if l[i] <= left_l.min() and l[i] < right_l.min():
            rows.append((i, SWING_LOW, l[i], i + k))
    return pd.DataFrame(rows, columns=RAW_COLS)


def alternate_swings(raw: pd.DataFrame) -> pd.DataFrame:
    """交互化（DESIGN.md §3）。confirm_index 順に 1 回走査し、採用/置換を記録する。

    戻り値は raw と同じ行を confirm_index, index, kind の順で並べ替えたもの＋
    kept（採用されたか）, superseded_at（置換されて外れた confirm_index。まだ
    有効、または最後まで不採用のままなら NA）。
    """
    r = raw.sort_values(["confirm_index", "index", "kind"]).reset_index(drop=True)
    kept = [False] * len(r)
    superseded_at = pd.array([pd.NA] * len(r), dtype="Int64")

    tail_pos = None  # 現在の末尾（採用中でまだ置換されていない）の r 上の行位置
    kinds = r["kind"].to_numpy()
    values = r["value"].to_numpy()
    confirm_idx = r["confirm_index"].to_numpy()
    for pos in range(len(r)):
        if tail_pos is not None and kinds[tail_pos] == kinds[pos]:
            tail_value = values[tail_pos]
            more_extreme = (values[pos] > tail_value if kinds[pos] == SWING_HIGH
                            else values[pos] < tail_value)
            if more_extreme:
                superseded_at[tail_pos] = confirm_idx[pos]
                kept[pos] = True
                tail_pos = pos
            # 同値または劣後: 新しい方を不採用のまま捨てる（先に来た方＝末尾が残る）
        else:
            kept[pos] = True
            tail_pos = pos

    out = r.copy()
    out["kept"] = kept
    out["superseded_at"] = superseded_at
    return out


def swings_as_of(alternated: pd.DataFrame, t_pos: int) -> pd.DataFrame:
    """交互化テーブルから T（t_pos）時点の交互化後スイング集合を取り出す（O(1)フィルタ）。

    再生（T-402）で多数の T について問い合わせる際、alternate_swings は 1 回だけ
    実行すればよい。
    """
    m = (
        alternated["kept"]
        & (alternated["confirm_index"] <= t_pos)
        & (alternated["superseded_at"].isna() | (alternated["superseded_at"] > t_pos))
    )
    return alternated.loc[m, RAW_COLS].reset_index(drop=True)


def swings_as_of_fresh(high: pd.Series, low: pd.Series, k: int, t_pos: int) -> pd.DataFrame:
    """便宜関数: 生検出→交互化→T 時点の集合抽出を 1 回分だけ行う（単発呼び出し用）。

    多数の T をまとめて問い合わせる場合は detect_raw_swings + alternate_swings を
    1 回実行し、swings_as_of を繰り返し呼ぶ方が効率的（再生用）。
    """
    raw = detect_raw_swings(high, low, k)
    alternated = alternate_swings(raw)
    return swings_as_of(alternated, t_pos)
