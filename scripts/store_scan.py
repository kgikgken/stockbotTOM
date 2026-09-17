"""store 全体を走査して、1909.T と同種の破損が他銘柄に無いか調べる（2026-09-17）。

**1909.T の形**（2026-09-13 に発生）:

- 終値が**一定倍率**（4,399,472 倍）で跳ね上がった —— 時価総額らしき値が価格欄に入った
- **出来高が全行 0** になった
- `check_splits` は素通りした（取得ウィンドウ全体が一様に再スケールされると
  フレーム内に段差が出ない）

ここでは **store に既に入っている値**を見る。取得時の検査（`store.seam_issues`）は
これから入る行を止めるもので、**既に入ってしまった破損は捕まえられない**ので別に要る。

出すのは表だけ。**修復はしない**（`cli refetch-tickers` が別にある）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockbot.config import Settings                      # noqa: E402
from stockbot.data.store import IDX_TICKER, OhlcvStore    # noqa: E402
from stockbot.validation.layer1 import DATA_QUALITY_EXCLUDED_TICKERS  # noqa: E402
from stockbot.validation.pattern_replay import CONFIRM_WINDOW, SEARCH_WINDOW  # noqa: E402
from stockbot.validation.replay import HOLDOUT_WINDOW     # noqa: E402

# 隣り合う足の終値比がこれを超えたら「分割では説明できない」（store.SEAM_RATIO_MAX と同じ理由）
JUMP_RATIO_MAX = 100.0
# 日本株の終値の上限の目安。**これ自体は判定に使わない** —— 桁違いの行を数えるだけ
ABSURD_CLOSE = 1e7
# 出来高 0 がこれ以上続いたら異常（売買停止は数日で戻る）
ZERO_VOLUME_MIN_RUN = 20

WINDOWS = (("探索窓", SEARCH_WINDOW), ("確認窓", CONFIRM_WINDOW),
           ("ホールドアウト", HOLDOUT_WINDOW))


def longest_zero_run(v: np.ndarray) -> int:
    best = run = 0
    for x in v:
        run = run + 1 if x == 0 else 0
        best = max(best, run)
    return best


def scan(df: pd.DataFrame) -> pd.DataFrame:
    """銘柄ごとに 3 つを数える（**判定はしない。数えるだけ**）。"""
    rows = []
    for ticker, g in df.groupby("ticker", sort=True):
        if ticker == IDX_TICKER or len(g) < 2:
            continue
        g = g.sort_values("date")
        c = g["close"].to_numpy(dtype=float)
        v = g["volume"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = c[1:] / c[:-1]
        ok = np.isfinite(r) & (c[:-1] > 0) & (c[1:] > 0)
        jump = ok & ((r > JUMP_RATIO_MAX) | (r < 1.0 / JUMP_RATIO_MAX))
        n_absurd = int(np.sum(np.isfinite(c) & (c > ABSURD_CLOSE)))
        zero_run = longest_zero_run(v)
        # **出来高 0 の連続は単独では発火させない**（2026-09-17 修正）。単独だと
        # 非流動銘柄と ETF を 43 件拾ってしまい、破損の signal にならなかった。
        # **桁違い行か段差と同時のときだけ**「破損の兆候」として出す
        broken = bool(jump.any()) or n_absurd > 0
        if not broken:
            continue
        idx = np.flatnonzero(jump)
        rows.append({
            "ticker": ticker, "n_rows": int(len(g)),
            "n_jump": int(jump.sum()),
            "max_ratio": float(np.nanmax(r[jump])) if jump.any() else float("nan"),
            "jump_date": (g["date"].to_numpy()[idx[0] + 1] if len(idx) else pd.NaT),
            "n_absurd_close": n_absurd,
            "max_close": float(np.nanmax(c)),
            "longest_zero_volume_run": int(zero_run),
        })
    return pd.DataFrame(rows)


def window_rows(df: pd.DataFrame, tickers) -> list:
    """疑いのある銘柄について、各窓に何行入っているかを数える。"""
    out = []
    for name, (lo, hi) in WINDOWS:
        part = df[df["ticker"].isin(tickers) & (df["date"] >= lo) & (df["date"] <= hi)]
        out.append((name, lo, hi, int(len(part))))
    return out


def main() -> int:
    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    df = store.load()
    if not len(df):
        print("[store-scan] store が空")
        return 0
    n_t = df["ticker"].nunique()
    print(f"[store-scan] 銘柄 {n_t} / 行 {len(df):,} / "
          f"{df['date'].min():%Y-%m-%d}〜{df['date'].max():%Y-%m-%d}")
    print(f"[store-scan] 基準: **隣接終値比 >{JUMP_RATIO_MAX:.0f}倍 または "
          f"終値 >{ABSURD_CLOSE:,.0f}円**。出来高0の連続（{ZERO_VOLUME_MIN_RUN}行以上）は"
          "**単独では出さない** —— 非流動銘柄と ETF を拾うだけだった（2026-09-17 修正）")

    bad = scan(df)
    if not len(bad):
        print("[store-scan] **該当なし。** 同種の破損は他銘柄に無い")
        return 0

    bad = bad.sort_values(["n_absurd_close", "n_jump", "longest_zero_volume_run"],
                          ascending=False)
    print(f"[store-scan] **該当 {len(bad)}銘柄**")
    print(f"[store-scan] {'銘柄':<10}{'行数':>8}{'段差':>6}{'最大比':>14}"
          f"{'桁違い行':>9}{'最大終値':>16}{'出来高0連続':>11}  最初の段差")
    for _i, r in bad.iterrows():
        d = "—" if pd.isna(r["jump_date"]) else f"{pd.Timestamp(r['jump_date']):%Y-%m-%d}"
        mr = "—" if pd.isna(r["max_ratio"]) else f"{r['max_ratio']:,.0f}"
        print(f"[store-scan] {r['ticker']:<10}{r['n_rows']:>8}{r['n_jump']:>6}{mr:>14}"
              f"{r['n_absurd_close']:>9}{r['max_close']:>16,.1f}"
              f"{r['longest_zero_volume_run']:>11}  {d}")

    for ticker in bad["ticker"]:
        for name, lo, hi, n in window_rows(df, {ticker}):
            print(f"[store-scan]   {ticker} の行数 {name}"
                  f"（{lo:%Y-%m-%d}〜{hi:%Y-%m-%d}）: {n}行")
    print("[store-scan] 修復は `python -m stockbot.cli refetch-tickers --tickers <銘柄>`"
          "（取得元が壊れていれば置換されない）")
    print(f"[store-scan] 除外中の銘柄: {sorted(DATA_QUALITY_EXCLUDED_TICKERS)}")
    still = sorted(set(bad["ticker"]) - set(DATA_QUALITY_EXCLUDED_TICKERS))
    if still:
        print(f"[store-scan] **除外リストに入っていない該当銘柄: {still}**")
    else:
        print("[store-scan] 該当銘柄はすべて除外リストに入っている"
              "（どの計算にも入らない）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
