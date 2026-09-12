"""パターン検出の再生（docs/BACKTEST.md §2・§3）。

各評価日 T で `features/pattern.py` の判定をそのまま走らせ、**成立した行を全件**
保存する。監視は保存しない（成立が事象・監視が状態という `PATTERN.md` §3.1 の
区別はここでも同じ）。

**判定式・閾値は変更しない。** ±1.5%・±0.75%・22 日・63 日・ε・旗竿・15 日は
`PATTERN.md` §1 の事前登録値のまま（BACKTEST.md §5 の禁止事項）。

**未来参照はしない。**

- 交互スイングの表は全期間から 1 回だけ作り、`swings_as_of(alternated, t_pos)` で
  T 時点の集合を取り出す。フラクタルの判定は i−k..i+k のバーだけで決まる純粋に
  因果的な条件で、確定に k 本かかることは `confirm_index <= T` の絞りが表している
  （`swings.py` の設計）。**T ごとに全期間から再計算しても同じ集合になる**ので、
  速度のためだけの最適化である（`tests/test_lookahead.py` のハーネスで固定）
- ATR も同じ理由で全期間から 1 回だけ計算する（各行がその行までのバーで決まる）
- 判定に使う終値・ネックライン・撤退・目標は `Close[T]` までの値だけ
- **ラベルだけが T+1 以降を見る**

**ホールドアウトは明示フラグ無しに生成しない**（CLAUDE.md の絶対規則）。
`replay.py` と同じく OHLCV 自体を物理的に打ち切る —— 日付で T を絞るだけでは、
窓の端の T が T+20 でホールドアウト側のバーを読む。
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..features import pattern as pattern_mod
from ..features.indicators import atr_wilder
from ..features.swings import alternate_swings, detect_raw_swings
from . import labels as labels_mod
from .replay import (
    HOLDOUT_WINDOW,
    MIN_HISTORY_BARS,
    _date_position,
    _filter_holdout,
    _real_trading_days,
    _truncate_before_holdout,
)

# docs/BACKTEST.md §1。確認窓だけが新しい（探索・ホールドアウトは replay.py と同じ）
SEARCH_WINDOW = (pd.Timestamp("2021-08-01"), pd.Timestamp("2026-01-30"))
CONFIRM_WINDOW = (pd.Timestamp("2017-03-15"), pd.Timestamp("2021-07-31"))

HORIZON = 20   # docs/PATTERN.md §3.4 と同じ評価窓。BACKTEST.md §3

# ユニバースのゲート（docs/BACKTEST.md §2.1）。**運用の `universe/build.py` と同じ値**を
# `Settings` から受け取る。ここに数字を書かない —— 書くと運用側と黙ってずれる
# （2026-09-11 の再実行の理由がこれ。§7 Q-2 / §8 D-5）

PREFIX = "pattern_replay_"
SUFFIX = ".csv.gz"

# 保存する列。**配信記録（PATTERN.md §3.3）と同じ形にそろえる** —— あとで実運用の
# 記録と突き合わせられるようにするため
DETECT_COLS = [
    "date", "ticker", "pattern",
    "neckline", "close_t", "atr_t", "breakout_pct", "breakout_h", "span",
    "pattern_low", "height", "target", "up_pct", "down_pct", "rr", "breakeven_win_rate",
    "upper_slope", "lower_slope", "pole_pct", "upper_scatter",
    "adv_jpy", "sector33",
    # **T の時点で既に測定目標を超えているか**（docs/BACKTEST.md §3.1）。
    # `b >= 1` と同値で、この行の `reached_target` は「届いた」ではなく
    # 「最初から届いていた」を数える。読み違えると危ないので列に持つ
    "already_at_target",
]

# ラベル（BACKTEST.md §3）。**主指標は r_20 だけ。** ほかは診断列
LABEL_COLS = [
    "r_20",                 # 超過リターン（ユニバース等加重を引いたもの）
    "raw_r_20",             # ln(Close[T+20] / Open[T+1])（引く前）
    "bench_r_20",           # ユニバース等加重の同じ量
    "entry_open", "close_h", "n_bars", "censored",
    "broke_stop", "broke_stop_day", "reached_target", "reached_target_day", "success",
    "mfe", "mae", "mfe_atr", "mae_atr",
]

REPLAY_COLS = DETECT_COLS + LABEL_COLS


def _first_day(mask: np.ndarray) -> Optional[int]:
    hits = np.flatnonzero(mask)
    return int(hits[0]) + 1 if hits.size else None


def _prepared(ohlcv: Dict[str, pd.DataFrame], tickers, k: int) -> dict:
    """銘柄ごとに「全期間で 1 回だけ作る」ものをまとめる。

    交互スイングの表と ATR。どちらも各行がその行までのバーだけで決まるので、
    T ごとに作り直しても同じ値になる（未来参照ではない）。
    """
    out = {}
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_HISTORY_BARS:
            continue
        alternated = alternate_swings(detect_raw_swings(df["High"], df["Low"], k))
        atr = atr_wilder(df["High"], df["Low"], df["Close"], 14).to_numpy(dtype=float)
        # 20 日平均売買代金。**記録するだけで探索しない**（BACKTEST.md §5）。
        # 各行がその行までの 20 本で決まるので未来参照にならない
        adv = (df["Close"] * df.get("Volume", pd.Series(np.nan, index=df.index))) \
            .rolling(20, min_periods=20).mean().to_numpy(dtype=float)
        out[ticker] = {"df": df, "alternated": alternated, "atr": atr, "adv": adv}
    return out


def label_one(df: pd.DataFrame, t_pos: int, row: dict, bench_r20: float,
              horizon: int = HORIZON) -> dict:
    """1 行ぶんのラベル（docs/BACKTEST.md §3）。**ここだけが T+1 以降を見る。**

    `screener/pattern_resolver.py` と同じ定義にそろえてある（評価窓 20 本、
    撤退＝パターン最安値、目標＝測定目標、同日は失敗）。超過リターンだけが
    バックテスト固有で、運用側の結果付けには無い。
    """
    out: dict = {c: np.nan for c in LABEL_COLS}
    out.update({"broke_stop": pd.NA, "broke_stop_day": pd.NA,
                "reached_target": pd.NA, "reached_target_day": pd.NA,
                "success": pd.NA, "n_bars": 0, "censored": True})

    start, end = t_pos + 1, min(t_pos + horizon, len(df) - 1)
    n_bars = max(0, end - start + 1)
    out["n_bars"] = int(n_bars)
    out["censored"] = bool(n_bars < horizon)
    if n_bars == 0:
        return out

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    win = slice(start, end + 1)

    entry_open = float(o[start])
    close_h = float(c[end])
    out["entry_open"] = entry_open
    out["close_h"] = close_h
    if np.isfinite(entry_open) and entry_open > 0 and np.isfinite(close_h) and close_h > 0:
        raw = float(np.log(close_h / entry_open))
        out["raw_r_20"] = raw
        out["bench_r_20"] = bench_r20
        if np.isfinite(bench_r20):
            out["r_20"] = raw - bench_r20

    max_high = float(np.nanmax(h[win]))
    min_low = float(np.nanmin(lo[win]))
    if np.isfinite(entry_open):
        out["mfe"] = max_high - entry_open
        out["mae"] = min_low - entry_open
        atr = float(row.get("atr_t") or np.nan)
        if np.isfinite(atr) and atr > 0:
            out["mfe_atr"] = out["mfe"] / atr
            out["mae_atr"] = out["mae"] / atr

    stop, target = float(row.get("pattern_low")), float(row.get("target"))
    if np.isfinite(stop):
        day = _first_day(lo[win] < stop)
        out["broke_stop"] = day is not None
        out["broke_stop_day"] = day if day is not None else pd.NA
    if np.isfinite(target):
        day = _first_day(h[win] > target)
        out["reached_target"] = day is not None
        out["reached_target_day"] = day if day is not None else pd.NA

    # 撤退を割る前に目標到達。同日に両方なら失敗（保守的。labels.py と同じ流儀）
    if out["reached_target"] is pd.NA or out["broke_stop"] is pd.NA:
        out["success"] = pd.NA
    elif not out["reached_target"]:
        out["success"] = False
    elif not out["broke_stop"]:
        out["success"] = True
    else:
        out["success"] = bool(out["reached_target_day"] < out["broke_stop_day"])
    return out


def universe_at(prepared: dict, equities: set, date_t: pd.Timestamp,
                min_adv_jpy: float, min_price: float,
                min_history_bars: int = MIN_HISTORY_BARS) -> list:
    """T 時点のユニバース（docs/BACKTEST.md §2.1）。**運用と同じ母集団にする。**

    上場株式のうち、T 時点で

    - 履歴が `min_history_bars` 本以上
    - **20 日平均売買代金が `min_adv_jpy` 以上**
    - **終値が `min_price` 以上**

    を満たす銘柄。どれも T までのバーだけで決まる（未来参照にならない）。

    運用の `universe/build.py` には他に**鮮度**（最終足が 7 日以内）と**分割疑い**の
    ゲートもあるが、ここでは当てない —— どちらも「いま取得できているか」を見る
    データ品質のゲートで、過去の各 T について当時の状態を復元できない。流動性の
    ゲート（売買代金・株価）とは性質が違う（§2.1 に明記）。
    """
    out = []
    for ticker in sorted(equities):
        item = prepared.get(ticker)
        if item is None:
            continue
        df = item["df"]
        pos = _date_position(df.index, date_t)
        if pos is None or pos + 1 < min_history_bars:
            continue
        adv = item["adv"][pos]
        close = float(df["Close"].to_numpy(dtype=float)[pos])
        if not np.isfinite(adv) or adv < min_adv_jpy:
            continue
        if not np.isfinite(close) or close < min_price:
            continue
        out.append(ticker)
    return out


def replay_one_day(date_t: pd.Timestamp, prepared: dict, universe,
                   sectors: Dict[str, str], k: int,
                   horizon: int = HORIZON) -> pd.DataFrame:
    """T 1 日ぶん。**成立した行だけ**を返す（監視は返さない）。

    prepared は `_prepared` の出力、universe はその日のユニバース銘柄の並び。
    """
    rows: List[dict] = []
    positions: dict = {}
    for ticker in universe:
        item = prepared.get(ticker)
        if item is None:
            continue
        df = item["df"]
        t_pos = _date_position(df.index, date_t)
        if t_pos is None or t_pos < MIN_HISTORY_BARS:
            continue
        positions[ticker] = t_pos
        # alternated を渡すと交互化を再実行しない（全期間で 1 回だけ作ってある）
        hits = pattern_mod.detect_patterns(
            df["High"], df["Low"], df["Close"], t_pos, k,
            alternated=item["alternated"])
        if len(hits) == 0:
            continue
        atr_t = float(item["atr"][t_pos]) if np.isfinite(item["atr"][t_pos]) else np.nan
        adv = float(item["adv"][t_pos]) if np.isfinite(item["adv"][t_pos]) else np.nan
        for _i, r in hits.iterrows():
            row = {c: np.nan for c in DETECT_COLS}
            row.update({k2: r[k2] for k2 in r.index if k2 in DETECT_COLS})
            target = float(r["target"]) if pd.notna(r["target"]) else np.nan
            close_t = float(r["close_t"])
            row.update({"date": date_t, "ticker": ticker, "atr_t": atr_t,
                        "adv_jpy": adv, "sector33": sectors.get(ticker, ""),
                        "pattern": str(r["pattern"]),
                        # b >= 1 ⟺ 終値[T] >= 目標（§3.1）
                        "already_at_target": bool(np.isfinite(target)
                                                  and close_t >= target)})
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=REPLAY_COLS)

    # ベンチマークは**ユニバース全銘柄**で計算する（成立した銘柄だけではない）
    bench = labels_mod.universe_benchmark_returns(
        {t: prepared[t]["df"] for t in universe if t in prepared}, date_t, (horizon,))

    bench_r20 = float(bench.get(horizon, {}).get("mean", np.nan))

    out = []
    for row in rows:
        df = prepared[row["ticker"]]["df"]
        out.append({**row,
                    **label_one(df, positions[row["ticker"]], row, bench_r20, horizon)})
    return pd.DataFrame(out)[REPLAY_COLS]


def day_path(output_dir: Path, date_t: pd.Timestamp) -> Path:
    return Path(output_dir) / f"{PREFIX}{pd.Timestamp(date_t):%Y-%m-%d}{SUFFIX}"


def load_table(output_dir: Path) -> pd.DataFrame:
    """保存済みの再生結果をすべて連結して返す（無ければ 0 行）。"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return pd.DataFrame(columns=REPLAY_COLS)
    frames = []
    for f in sorted(output_dir.glob(f"{PREFIX}*{SUFFIX}")):
        df = pd.read_csv(f, dtype={"ticker": str, "pattern": str, "sector33": str})
        if len(df):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=REPLAY_COLS)
    return pd.concat(frames, ignore_index=True)


def run(ohlcv: Dict[str, pd.DataFrame], idx_ohlcv: pd.DataFrame, listed: pd.DataFrame,
        output_dir: Path, start: pd.Timestamp, end: pd.Timestamp, k: int,
        min_adv_jpy: float, min_price: float,
        sectors: Optional[Dict[str, str]] = None,
        min_history_bars: int = MIN_HISTORY_BARS,
        include_holdout: bool = False, log=print) -> None:
    """start〜end の営業日ごとに再生し、日別ファイルに保存する。

    中断再開: 既にその日のファイルがあればスキップする（`replay.py` と同じ）。

    **include_holdout=False（既定）ではホールドアウトを生成しない。** 日付で絞る
    だけでなく OHLCV 自体を打ち切る（CLAUDE.md の絶対規則）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sectors = sectors or {}

    if not include_holdout:
        ohlcv = _truncate_before_holdout(ohlcv)
        idx_ohlcv = idx_ohlcv[idx_ohlcv.index < HOLDOUT_WINDOW[0]]

    dates = pd.bdate_range(start, end)
    dates = _filter_holdout(dates, include_holdout, log=log)
    dates = _real_trading_days(dates, ohlcv, log=log)
    if len(dates) == 0:
        log("[pattern-replay] 再生対象の営業日が無い")
        return

    log(f"[pattern-replay] {dates[0].date()}〜{dates[-1].date()} {len(dates)}営業日")

    # **全日ぶん揃っていれば何もしない。** スイング表の作成（全銘柄）だけで数分かかる
    # ので、集計や出口の検証（§10）だけを回したいときに毎回それを払わないで済む
    todo = [d for d in dates if not day_path(output_dir, d).exists()]
    if not todo:
        log(f"[pattern-replay] {len(dates)}日ぶんすべて保存済み。再生をスキップする")
        return

    all_tickers = sorted(t for t in ohlcv if t != "__IDX__")
    prepared = _prepared(ohlcv, all_tickers, k)
    log(f"[pattern-replay] スイング表と ATR を用意: {len(prepared)}/{len(all_tickers)} 銘柄")

    equities = set(listed[listed["is_equity"].astype(bool)]["ticker"].astype(str))
    log(f"[pattern-replay] ユニバースのゲート: 履歴 {min_history_bars}本以上 / "
        f"20日平均売買代金 {min_adv_jpy / 1e8:.1f}億円以上 / 株価 {min_price:.0f}円以上"
        "（運用の universe/build.py と同じ値）")

    t0 = time.monotonic()
    n_written = 0
    n_univ: list = []
    for i, date_t in enumerate(dates, start=1):
        path = day_path(output_dir, date_t)
        if path.exists():
            continue
        tickers = universe_at(prepared, equities, date_t, min_adv_jpy, min_price,
                              min_history_bars)
        n_univ.append(len(tickers))
        day = replay_one_day(date_t, prepared, tickers, sectors, k)
        day.to_csv(path, index=False, encoding="utf-8", compression="gzip")
        n_written += 1
        if i % 50 == 0 or i == len(dates):
            elapsed = time.monotonic() - t0
            log(f"[pattern-replay] {i}/{len(dates)} {date_t.date()} "
                f"ユニバース {len(tickers)}銘柄 / 成立 {len(day)}件 / 経過 {elapsed:.0f}秒")
    if n_univ:
        arr = np.asarray(n_univ, dtype=float)
        log(f"[pattern-replay] ユニバース: 中央 {np.median(arr):.0f}銘柄 / "
            f"最小 {arr.min():.0f} / 最大 {arr.max():.0f}")
    log(f"[pattern-replay] 書き出し {n_written}日ぶん")
