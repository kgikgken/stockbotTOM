"""除外した銘柄が、除外前に各窓のプールとベンチマークにどれだけ入っていたかを測る
（2026-09-17・設計責任者の指示「8303.T が各窓のプールに入った行数とベンチマークへの
影響を報告」）。

**除外の効果を確かめるための診断であって、検定ではない。** 数えるだけで解釈しない。

見るのは 3 つ。

1. 各窓に store が持っている行数
2. そのうち**ユニバースのゲートを通っていた日数**（履歴 250 本以上・20 日平均売買代金
   `MIN_ADV_JPY` 以上・終値 `MIN_PRICE` 以上。`pattern_replay.universe_at` と同じ条件）
3. ベンチマーク（等加重）への影響 —— プールに入っていた日の `ln(Close[T+20]/Open[T+1])`
   の大きさと、それがその日の平均をどれだけ動かしうるか（`|r| / N`）

**除外後はどの日もプールに入らない**ので、3 つとも 0 になるのが期待値である。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockbot.config import Settings                                   # noqa: E402
from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long      # noqa: E402
from stockbot.validation.layer1 import DATA_QUALITY_EXCLUDED_2026_09   # noqa: E402
from stockbot.validation.pattern_replay import (                       # noqa: E402
    CONFIRM_WINDOW, HORIZON, SEARCH_WINDOW,
)
from stockbot.validation.replay import HOLDOUT_WINDOW, MIN_HISTORY_BARS  # noqa: E402

WINDOWS = (("探索窓", SEARCH_WINDOW), ("確認窓", CONFIRM_WINDOW),
           ("ホールドアウト", HOLDOUT_WINDOW))


def gate_days(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp, cfg: Settings):
    """窓の中で**ユニバースのゲートを通っていた日**（`universe_at` と同じ条件）。"""
    if df is None or len(df) == 0:
        return pd.DatetimeIndex([]), 0
    adv = (df["Close"] * df.get("Volume", pd.Series(np.nan, index=df.index))) \
        .rolling(20, min_periods=20).mean()
    close = df["Close"]
    bars = pd.Series(np.arange(len(df)) + 1, index=df.index)
    ok = ((bars >= MIN_HISTORY_BARS) & adv.notna() & (adv >= cfg.min_adv_jpy)
          & close.notna() & (close >= cfg.min_price))
    win = (df.index >= lo) & (df.index <= hi)
    n_rows = int(win.sum())
    return df.index[ok.to_numpy() & win], n_rows


def horizon_return(df: pd.DataFrame, t: pd.Timestamp, horizon: int = HORIZON) -> float:
    """`ln(Close[T+20] / Open[T+1])`。ベンチマークに入る量（`labels.py` と同じ）。"""
    pos = df.index.get_indexer([t])[0]
    if pos < 0 or pos + 1 >= len(df):
        return float("nan")
    end = min(pos + horizon, len(df) - 1)
    o, c = float(df["Open"].iloc[pos + 1]), float(df["Close"].iloc[end])
    if not (np.isfinite(o) and np.isfinite(c) and o > 0 and c > 0):
        return float("nan")
    return float(np.log(c / o))


def main() -> int:
    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    long = store.load()
    if not len(long):
        print("[excluded-impact] store が空")
        return 0
    ohlcv = from_long(long)
    ohlcv.pop(IDX_TICKER, None)

    # プールの大きさ（等加重ベンチマークの分母）。窓ごとの中央値を目安に使う
    print(f"[excluded-impact] ゲート: 履歴 {MIN_HISTORY_BARS}本以上 / "
          f"20日平均売買代金 {cfg.min_adv_jpy / 1e8:.1f}億円以上 / "
          f"株価 {cfg.min_price:.0f}円以上")
    print(f"[excluded-impact] 対象（2026-09 に除外）: "
          f"{sorted(DATA_QUALITY_EXCLUDED_2026_09)}")

    for ticker in sorted(DATA_QUALITY_EXCLUDED_2026_09):
        df = ohlcv.get(ticker)
        if df is None or len(df) == 0:
            print(f"[excluded-impact] {ticker}: store に行が無い")
            continue
        print(f"[excluded-impact] --- {ticker} "
              f"({df.index.min():%Y-%m-%d}〜{df.index.max():%Y-%m-%d} {len(df)}行) ---")
        for name, (lo, hi) in WINDOWS:
            days, n_rows = gate_days(df, lo, hi, cfg)
            if not len(days):
                print(f"[excluded-impact]   {name}: 行 {n_rows} / "
                      f"**ゲート通過 0 日**（プールに入っていない）")
                continue
            rs = np.asarray([horizon_return(df, t) for t in days], dtype=float)
            rs = rs[np.isfinite(rs)]
            worst = float(np.nanmax(np.abs(rs))) if len(rs) else float("nan")
            print(f"[excluded-impact]   {name}: 行 {n_rows} / "
                  f"**ゲート通過 {len(days)} 日**（{days.min():%Y-%m-%d}〜{days.max():%Y-%m-%d}）"
                  f" / |r20| 最大 {worst:.3f}")
            for n_pool in (900, 1000, 1100):
                print(f"[excluded-impact]     等加重 {n_pool} 銘柄なら、その日の平均を"
                      f"最大 {worst / n_pool:+.5f} 動かしうる")
    print("[excluded-impact] **除外後はどの窓でもプールに入らない** —— "
          "universe_at と build_universe が除外リストを先に外すため（0 日になる）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
