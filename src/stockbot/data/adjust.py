"""分割・併合の整合性検査（SPEC §4 ルール）。

前提: Yahoo の履歴価格は分割調整済み（auto_adjust=False でも）。したがって自前で
分割係数を掛け直すと二重調整になる。本モジュールは「調整済みであるべき系列が実際に
連続しているか」を検査し、

  1. 分割イベントが記録されているのに価格が分割比率どおり跳んでいる → 未調整。
     イベント日より前の四本値を比率で割り、出来高を比率で掛けて修正する。
  2. 分割イベントの記録が無いのに、分割比率どおりの跳びと出来高の逆方向の跳びがある
     → 未記録分割の疑い。修正はせず issue として記録し、呼び出し側が銘柄を除外する。

Yahoo の Stock Splits 列は「新株数/旧株数」（1:2 分割なら 2.0、10 株併合なら 0.1）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 日本株で一般的な分割・併合比率。1.5 や 1.2 など端数分割も存在するが、
# 通常の日次変動と区別しにくいため未記録分割の自動検出対象からは外す。
COMMON_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 1 / 3, 0.25, 0.2, 0.1)


def _median_close(df: pd.DataFrame, lo: int, hi: int) -> float:
    seg = df["Close"].iloc[max(lo, 0):max(hi, 0)]
    return float(seg.median()) if len(seg) else float("nan")


def check_splits(df: pd.DataFrame, ticker: str = "", tol: float = 0.08,
                 window: int = 3) -> Tuple[pd.DataFrame, List[Dict]]:
    """1銘柄の日足（COLS 列）を検査し、(修正後 df, issues) を返す。

    issues の要素: {"ticker","date","kind","ratio","observed","action"}
      kind: "unadjusted_split"(修正済み) / "suspected_unrecorded_split"(未修正・要除外)
            / "small_ratio_split"(端数分割。検査対象外として記録のみ)
    """
    issues: List[Dict] = []
    if df is None or len(df) < 2 * window + 2:
        return df, issues
    df = df.copy()
    if "Volume" in df.columns:
        # yfinance の実データは Volume が int64 で返る。分割調整で比率倍すると小数になり、
        # pandas 3.x では int64 列への代入が LossySetitemError になる（未調整分割の
        # 出来高修正で実データに対してのみ発生する。float64 化して回避する）
        df["Volume"] = df["Volume"].astype(float)
    close = df["Close"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    splits = df["Stock Splits"].to_numpy(dtype=float) if "Stock Splits" in df.columns else np.zeros(len(df))
    n = len(df)

    # ---- 1. 記録された分割の調整確認 ----
    for i in np.flatnonzero(splits > 0):
        r = float(splits[i])
        if abs(r - 1.0) < 0.15:
            issues.append({"ticker": ticker, "date": df.index[i], "kind": "small_ratio_split",
                           "ratio": r, "observed": None, "action": "none"})
            continue
        before = _median_close(df, i - window, i)
        after = _median_close(df, i, i + window)
        if not (np.isfinite(before) and np.isfinite(after)) or after <= 0:
            continue
        observed = before / after            # 未調整なら ≈ r、調整済みなら ≈ 1
        if abs(observed - r) / r <= tol:
            # 未調整: i より前を調整する
            df.iloc[:i, df.columns.get_indexer(["Open", "High", "Low", "Close"])] = \
                df.iloc[:i][["Open", "High", "Low", "Close"]].to_numpy() / r
            df.iloc[:i, df.columns.get_indexer(["Volume"])] = \
                df.iloc[:i][["Volume"]].to_numpy() * r
            close = df["Close"].to_numpy(dtype=float)
            vol = df["Volume"].to_numpy(dtype=float)
            issues.append({"ticker": ticker, "date": df.index[i], "kind": "unadjusted_split",
                           "ratio": r, "observed": round(observed, 4), "action": "adjusted_prior_rows"})

    # ---- 2. 未記録分割の疑い ----
    recorded_days = set(np.flatnonzero(splits > 0).tolist())
    for i in range(1, n):
        if close[i] <= 0 or close[i - 1] <= 0:
            continue
        q = close[i - 1] / close[i]          # 分割なら ≈ 比率
        hit = None
        for r in COMMON_RATIOS:
            if abs(q - r) / r <= 0.03:
                hit = r
                break
        if hit is None:
            continue
        if any(abs(i - j) <= window for j in recorded_days):
            continue
        v_prev = float(np.nanmean(vol[max(i - 5, 0):i])) if i > 0 else float("nan")
        v_now = float(np.nanmean(vol[i:i + 5]))
        if not (np.isfinite(v_prev) and np.isfinite(v_now)) or v_prev <= 0:
            continue
        v_ratio = v_now / v_prev
        # 分割(hit>1)なら出来高は増える、併合(hit<1)なら減る
        if (hit > 1 and v_ratio > 1.5) or (hit < 1 and v_ratio < 0.67):
            issues.append({"ticker": ticker, "date": df.index[i], "kind": "suspected_unrecorded_split",
                           "ratio": hit, "observed": round(q, 4), "action": "flag_only"})
    return df, issues


def check_all(ohlcv: Dict[str, pd.DataFrame], **kw) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """全銘柄に check_splits を適用。戻り値: (修正後 dict, issues DataFrame)"""
    out: Dict[str, pd.DataFrame] = {}
    rows: List[Dict] = []
    for t, df in ohlcv.items():
        fixed, issues = check_splits(df, ticker=t, **kw)
        out[t] = fixed
        rows.extend(issues)
    cols = ["ticker", "date", "kind", "ratio", "observed", "action"]
    return out, (pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols))

# ------------------------------------------------------------------ 取得データの健全性
# 2026-09-17 追加。**置換する前に、取ってきた値が壊れていないか見る。**
#
# 1909.T を全履歴再取得したら、yfinance が**全 2,597 行**を時価総額らしき値で返し、
# 出来高を全行 0 にした。`upsert_replace` は取れたものをそのまま入れるので、
# **取得元が壊れていると良い行を壊れた行で置き換えてしまう**（2026-09-17 に実際に起きた）。
# `check_splits` はフレーム内の段差しか見ないので、全体が一様に壊れていると素通りする。
FETCH_ABSURD_CLOSE = 1e7          # 日本株の終値の上限の目安（1,000 万円）
FETCH_ZERO_VOLUME_RATIO = 0.95    # 出来高 0 の行がこの割合を超えたら異常


def fetch_sanity_issues(ohlcv: Dict[str, pd.DataFrame],
                        absurd_close: float = FETCH_ABSURD_CLOSE,
                        zero_volume_ratio: float = FETCH_ZERO_VOLUME_RATIO) -> Dict[str, str]:
    """取得したフレームが**そのまま store に入れてよい形か**を見る（2026-09-17）。

    - 終値に `absurd_close` を超える行がある → **異常**
    - 出来高 0 の行が `zero_volume_ratio` を超える → **異常**

    戻り値は {銘柄: 理由}。**判定はここでは行わない** —— 呼び出し側が「置換しない」を決める。
    **分割や急騰では引っかからない** —— 見ているのは水準であって変化率ではない。
    """
    out: Dict[str, str] = {}
    for ticker, df in (ohlcv or {}).items():
        if df is None or len(df) == 0 or "Close" not in df.columns:
            continue
        c = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
        n_absurd = int(np.sum(np.isfinite(c) & (c > absurd_close)))
        if n_absurd:
            out[ticker] = (f"終値が {absurd_close:,.0f} 円を超える行が {n_absurd}/{len(df)} 行"
                           f"（最大 {np.nanmax(c):,.1f}）")
            continue
        if "Volume" in df.columns:
            v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).to_numpy(dtype=float)
            zero = float(np.mean(v == 0)) if len(v) else 0.0
            if zero > zero_volume_ratio:
                out[ticker] = f"出来高 0 の行が {zero:.1%}（{int(np.sum(v == 0))}/{len(df)} 行）"
    return out
