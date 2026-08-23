"""ラベル（検証用、DESIGN.md §9 / TASKS.md T-401）。

CLAUDE.md の絶対規則により、このモジュールだけが T+1 以降のデータを参照してよい
（他の全モジュールは T の引けまでのデータしか使わない）。T 自身やそれ以前のデータは
一切参照しない（超過リターン・成功失敗・MFE/MAE のいずれも Open[T+1] 以降のみを使う）。

- 超過リターン（§9）: r_h = ln(Close[T+h]/Open[T+1]) − ユニバース等加重の同じ量。
  h ∈ {3,5,10,15,20,30}（H_LIST）。ユニバース等加重の同じ量は、ユニバース全銘柄について
  「同じ T+1 始値で買い T+h 終値まで保有した場合の対数リターン」を単純平均したもの
  （universe_benchmark_returns）。生存バイアス対策（T-401 条件付き承認）: T+h の終値が
  無い銘柄（上場廃止・売買停止等）を除外すると生き残った銘柄だけでベンチマークが上振れ
  するため、T+1 以降で入手できる最後の有効な終値までの打ち切りリターンとして平均に含める
  （n_truncated）。T+1 の始値自体が無い銘柄のみ除外する（n_excluded）。銘柄側の r_h
  （excess_return）は T+h が無ければ NaN のまま（打ち切りはユニバース側だけの措置。
  個別銘柄の r_h を打ち切ると「どのhまで定義できたか」が曖昧になるため、代わりに
  compute_labels の戻り値に censored_at を付けて後段で判別できるようにする）
- 成功/失敗/未決（§9）: T+1 から T+N の各日を順に見て、
    - High[t] > H0.high かつ Low[t] < Lp の両方が同じ日に起きたら失敗（保守的、成功より優先）
    - High[t] > H0.high のみなら成功
    - Low[t] < Lp のみなら失敗
    - どちらでもない日はスキップして翌日へ
  最初にどちらかが成立した日で確定し、それ以降の日は見ない。N 日以内にどちらも起きなければ
  未決。N は既定 15（config.Settings.label_n。10/20 は感度分析用）
- MFE/MAE（§9）: 判定が何日目で確定したかに関わらず、常に T+1..T+N の固定窓を使う
  （成功/失敗の判定と違って早期打ち切りしない）。分母の ATR は T 時点（h0 や T+1 ではなく
  atr_t として呼び出し側が渡す、features.pullback 等と同じ ATR14）
  - MFE = (max(High[T+1..T+N]) − Open[T+1]) / ATR[T]
  - MAE = (Open[T+1] − min(Low[T+1..T+N])) / ATR[T]

引数の close/open_/high/low は 0 始まりの位置（swings.py と同じ規約）で扱う整列済み
pandas.Series。h0_high/lp_value/atr_t は features.pullback.pullback_state()・
features.indicators の出力を呼び出し側からそのまま渡す（このモジュールでは再計算しない）。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

H_LIST = (3, 5, 10, 15, 20, 30)  # DESIGN.md §9
DEFAULT_N = 15  # DESIGN.md §12 ラベル N（感度 10/20。config.Settings.label_n と同じ既定値）

LABEL_SUCCESS = "success"
LABEL_FAILURE = "failure"
LABEL_UNDETERMINED = "undetermined"


def universe_benchmark_returns(universe_ohlcv: Dict[str, pd.DataFrame], date_t: pd.Timestamp,
                               h_list: Iterable[int] = H_LIST) -> Dict[int, dict]:
    """T+1 の始値でユニバース全銘柄を等金額買い、T+h の終値（無ければ入手できる最後の
    有効な終値、打ち切り扱い）まで保有した場合の対数リターンを銘柄間で単純平均する
    （h ごと）。除外するのは T+1 の始値自体が無い銘柄だけ（date_t が系列に無い、
    T+1 が系列の外、始値が非有限/非正のいずれか）。

    生存バイアス対策（T-401 条件付き承認）: T+h の終値が無い銘柄（上場廃止・売買停止等）
    を単純に除外すると、T+h まで生き残った銘柄だけでベンチマークを組むことになり平均が
    上振れする。T+1 以降で入手できる最後の有効な終値までの打ち切りリターンとして平均に
    含めることでこれを避ける。

    戻り値: {h: {"mean": 平均対数リターン（該当銘柄が無ければ NaN）,
                "n_used": 平均に使った銘柄数（打ち切り含む）,
                "n_truncated": そのうち T+h の終値が無く打ち切ったリターンを使った銘柄数,
                "n_excluded": T+1 の始値が無く除外した銘柄数（h に依らず同じ値）}}
    T-403 でこれらの件数を集計できるようにするための診断情報。
    """
    date_t = pd.Timestamp(date_t)
    result: Dict[int, dict] = {}
    for h in h_list:
        rs: list[float] = []
        n_truncated = 0
        n_excluded = 0
        for df in universe_ohlcv.values():
            if df is None or len(df) == 0:
                n_excluded += 1
                continue
            idx = df.index
            try:
                pos = idx.get_loc(date_t)
            except KeyError:
                n_excluded += 1
                continue
            if not isinstance(pos, (int, np.integer)):
                n_excluded += 1  # 重複日付など。ユニバースの一銘柄としては扱わない
                continue
            t1 = pos + 1
            if t1 >= len(df):
                n_excluded += 1
                continue
            o = float(df["Open"].iloc[t1])
            if not (np.isfinite(o) and o > 0):
                n_excluded += 1
                continue

            th = pos + h
            last_pos = min(th, len(df) - 1)
            close_arr = df["Close"].to_numpy(dtype=float)
            found_pos = None
            for p in range(last_pos, t1 - 1, -1):
                if np.isfinite(close_arr[p]) and close_arr[p] > 0:
                    found_pos = p
                    break
            if found_pos is None:
                # T+1 以降に有効な終値が一本も無い（始値はあるが直後にデータが尽きる等）
                n_excluded += 1
                continue

            rs.append(np.log(close_arr[found_pos] / o))
            if found_pos < th:
                n_truncated += 1
        result[h] = {
            "mean": float(np.mean(rs)) if rs else np.nan,
            "n_used": len(rs),
            "n_truncated": n_truncated,
            "n_excluded": n_excluded,
        }
    return result


def excess_return(close: np.ndarray, open_: np.ndarray, t_pos: int, h: int,
                  benchmark: float) -> float:
    """r_h = ln(Close[T+h]/Open[T+1]) − benchmark。定義できなければ NaN。"""
    t1, th = t_pos + 1, t_pos + h
    if th >= len(close) or t1 >= len(open_) or np.isnan(benchmark):
        return np.nan
    o, c = float(open_[t1]), float(close[th])
    if not (np.isfinite(o) and o > 0 and np.isfinite(c) and c > 0):
        return np.nan
    return float(np.log(c / o) - benchmark)


def outcome_label(high: np.ndarray, low: np.ndarray, t_pos: int,
                  h0_high: float, lp_value: float, n: int = DEFAULT_N) -> dict:
    """成功/失敗/未決（DESIGN.md §9）。T+1 から T+N（データが無ければそこまで）を順に見て、
    最初に条件が成立した日で確定する。同日に両方成立したら失敗（保守的、成功より優先）。
    """
    end = min(t_pos + n, len(high) - 1)
    for t in range(t_pos + 1, end + 1):
        hit_success = high[t] > h0_high
        hit_failure = low[t] < lp_value
        if hit_failure:  # 同日に両方成立の場合も含め、失敗を優先する
            return {"label": LABEL_FAILURE, "hit_day": t - t_pos}
        if hit_success:
            return {"label": LABEL_SUCCESS, "hit_day": t - t_pos}
    return {"label": LABEL_UNDETERMINED, "hit_day": None}


def mfe_mae(high: np.ndarray, low: np.ndarray, open_: np.ndarray, t_pos: int,
           n: int, atr_t: float) -> dict:
    """MFE/MAE（DESIGN.md §9）。T+1..T+N の固定窓（判定日に関わらず打ち切らない）。"""
    t1 = t_pos + 1
    end = min(t_pos + n, len(high) - 1)
    if t1 > end or t1 >= len(open_):
        return {"mfe": np.nan, "mae": np.nan}
    o = float(open_[t1])
    if not (np.isfinite(o) and np.isfinite(atr_t) and atr_t > 0):
        return {"mfe": np.nan, "mae": np.nan}
    window_high = high[t1:end + 1]
    window_low = low[t1:end + 1]
    mfe = (float(np.nanmax(window_high)) - o) / atr_t
    mae = (o - float(np.nanmin(window_low))) / atr_t
    return {"mfe": mfe, "mae": mae}


def compute_labels(close: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series,
                   t_pos: int, h0_high: float, lp_value: float, atr_t: float,
                   universe_benchmarks: Dict[int, float], n: int = DEFAULT_N,
                   h_list: Iterable[int] = H_LIST) -> dict:
    """T（t_pos）時点の1銘柄ぶんのラベルをまとめて返す（DESIGN.md §9）。

    universe_benchmarks は h → ベンチマーク対数リターン（float）の単純な辞書。
    universe_benchmark_returns() の出力をそのまま渡すのではなく、呼び出し側が
    `{h: v["mean"] for h, v in universe_benchmark_returns(...).items()}` のように
    "mean" を取り出して渡す（このモジュールでは診断用件数は扱わない）。h0_high/lp_value
    は features.pullback.pullback_state() の h0_high/lp_value、atr_t は T 時点の ATR14
    （features.indicators.atr_wilder の t_pos 番目の値）をそのまま渡す。

    戻り値: r_3, r_5, r_10, r_15, r_20, r_30, label, hit_day, mfe, mae, censored_at。
    censored_at は T+1 以降で実際に系列に存在する本数（N で頭打ち。T-401 条件付き承認）。
    censored_at < n なら、r_h（h > censored_at のもの）や未決ラベルが「まだ判定材料が
    出ていないだけ」ではなく「系列がそこで終わっている（上場廃止等）」ことを示す。
    """
    c = close.to_numpy(dtype=float)
    o = open_.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)

    result: dict = {}
    for hz in h_list:
        result[f"r_{hz}"] = excess_return(c, o, t_pos, hz, universe_benchmarks.get(hz, np.nan))

    outcome = outcome_label(h, lo, t_pos, h0_high, lp_value, n)
    result["label"] = outcome["label"]
    result["hit_day"] = outcome["hit_day"]

    mm = mfe_mae(h, lo, o, t_pos, n, atr_t)
    result["mfe"] = mm["mfe"]
    result["mae"] = mm["mae"]

    available = len(c) - (t_pos + 1)
    result["censored_at"] = int(max(0, min(available, n)))
    return result
