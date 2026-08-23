"""歴史的再生（リプレイ、DESIGN.md §10.1・§10.5 / TASKS.md T-402）。

主評価窓（2021-08〜2026-02）・頑健性窓（2016-08〜2026-02）の各営業日 T について、
T で切った系列から日次パイプライン（M2: features.*、M3a: scoring.composite。
pipeline.compute_daily_features に統合済み）を走らせ、T+1 以降のデータを使う
ラベル（validation.labels、T-401）を付けて、銘柄×日を1つの表に蓄積する。

CLAUDE.md の絶対規則: ホールドアウト（2026-02〜2026-08、HOLDOUT_WINDOW）は
include_holdout=True を明示的に渡さない限り生成しない（既定 False）。

保存形式は日付ごとのファイル（daily/features_YYYY-MM-DD.csv.gz と同じ流儀で
replay_YYYY-MM-DD.csv.gz）。中断再開は「既にファイルがある日はスキップする」で
実現する（CLAUDE.md の「日次で肥大するファイルを毎回丸ごとコミットするとリポジトリが
膨らむ→日付別ファイル」の教訓に合わせた設計）。load_replay_table() で1つの表として
まとめて読み込める。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..features import indicators
from ..pipeline import BOOL_COLS, DATE_COLS, DAILY_FEATURES_COLS, _coerce_bool, compute_daily_features
from . import labels as labels_mod

# DESIGN.md §10.1
MAIN_WINDOW = (pd.Timestamp("2021-08-01"), pd.Timestamp("2026-02-01"))
ROBUSTNESS_WINDOW = (pd.Timestamp("2016-08-01"), pd.Timestamp("2026-02-01"))
HOLDOUT_WINDOW = (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-08-01"))
MIN_HISTORY_BARS = 250  # DESIGN.md §10.1（config.Settings.min_history_bars と同じ既定値）

LABEL_COLS = [f"r_{h}" for h in labels_mod.H_LIST] + ["label", "hit_day", "mfe", "mae", "censored_at"]
REPLAY_COLS = DAILY_FEATURES_COLS + LABEL_COLS


def _date_position(idx: pd.DatetimeIndex, date_t: pd.Timestamp) -> Optional[int]:
    """idx 内での date_t の位置。無い（または重複日付で一意に決まらない）場合は None。"""
    try:
        pos = idx.get_loc(date_t)
    except KeyError:
        return None
    if not isinstance(pos, (int, np.integer)):
        return None
    return int(pos)


def replay_universe_tickers(listed: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
                            date_t: pd.Timestamp, min_history_bars: int = MIN_HISTORY_BARS) -> List[str]:
    """T 時点のユニバース: 現在の上場銘柄のうち、T 時点で履歴 min_history_bars 本以上
    ある銘柄（DESIGN.md §10.1）。本番の universe.build（売買代金・株価下限等）とは別の、
    再生専用の簡易な定義（DESIGN.md に明記された通りの定義であり、本番のゲート G0 の
    「ユニバース通過」判定はこの一覧を渡すことで満たされる）。
    """
    eq = listed[listed["is_equity"].astype(bool)]["ticker"]
    out = []
    for ticker in sorted(eq):
        df = ohlcv.get(ticker)
        if df is None or len(df) == 0:
            continue
        pos = _date_position(df.index, date_t)
        if pos is None:
            continue
        if pos + 1 < min_history_bars:
            continue
        out.append(ticker)
    return out


def replay_one_day(
    date_t: pd.Timestamp,
    ohlcv_full: Dict[str, pd.DataFrame],
    idx_ohlcv_full: pd.DataFrame,
    listed: pd.DataFrame,
    k: int, label_n: int, pool_days: int,
    min_history_bars: int = MIN_HISTORY_BARS,
    earnings_schedule: Optional[pd.DataFrame] = None,
    history_pool: Optional[pd.DataFrame] = None,
    log=print,
) -> pd.DataFrame:
    """T（date_t）1日ぶんの特徴量・状態・スコア・ラベルをまとめる。

    ohlcv_full/idx_ohlcv_full は打ち切らない全期間データ（DatetimeIndex 付き）。
    特徴量・スコア（M2/M3a）は T までに切ったコピーを作って計算し（先読み無し）、
    ラベル（T-401、T+1 以降を見る）だけは打ち切らない全期間データをそのまま使う。

    戻り値は REPLAY_COLS の列を持つ DataFrame（features/gates/dimensions/scores +
    r_3..r_30/label/hit_day/mfe/mae/censored_at）。この日にユニバースが空、または
    採点対象が1件も無ければ0行で返す。
    """
    universe_tickers = replay_universe_tickers(listed, ohlcv_full, date_t, min_history_bars)

    idx_pos = _date_position(idx_ohlcv_full.index, date_t)
    if idx_pos is None or not universe_tickers:
        log(f"[replay] {date_t.date()}: 指数データまたはユニバースが無いためスキップ")
        return pd.DataFrame(columns=REPLAY_COLS)
    idx_close_trunc = idx_ohlcv_full["Close"].iloc[: idx_pos + 1]

    ohlcv_trunc: Dict[str, pd.DataFrame] = {}
    for ticker in universe_tickers:
        pos = _date_position(ohlcv_full[ticker].index, date_t)
        ohlcv_trunc[ticker] = ohlcv_full[ticker].iloc[: pos + 1]

    features_df = compute_daily_features(
        ohlcv_trunc, universe_tickers, idx_close_trunc, k, label_n,
        earnings_schedule=earnings_schedule, history_pool=history_pool,
        pool_days=pool_days, log=log,
    )
    if len(features_df) == 0:
        return features_df

    # ラベル（T-401）: ユニバース等加重ベンチマークは打ち切らない全期間データで計算する
    benchmarks_raw = labels_mod.universe_benchmark_returns(
        {t: ohlcv_full[t] for t in universe_tickers}, date_t, labels_mod.H_LIST)
    benchmarks = {h: v["mean"] for h, v in benchmarks_raw.items()}

    label_rows = []
    for _, row in features_df.iterrows():
        ticker = row["ticker"]
        df_full = ohlcv_full[ticker]
        df_trunc = ohlcv_trunc[ticker]
        t_pos_full = len(df_trunc) - 1  # df_full 内での T の位置は df_trunc の長さ-1 と一致
        atr_trunc = indicators.atr_wilder(df_trunc["High"], df_trunc["Low"], df_trunc["Close"], 14)
        atr_t = float(atr_trunc.iloc[-1])
        out = labels_mod.compute_labels(
            df_full["Close"], df_full["Open"], df_full["High"], df_full["Low"],
            t_pos_full, row["h0_high"], row["lp_value"], atr_t, benchmarks, n=label_n,
        )
        label_rows.append(out)
    label_df = pd.DataFrame(label_rows, index=features_df.index)
    return pd.concat([features_df, label_df], axis=1)[REPLAY_COLS]


def _replay_path(output_dir: Path, date_t: pd.Timestamp) -> Path:
    return Path(output_dir) / f"replay_{pd.Timestamp(date_t).strftime('%Y-%m-%d')}.csv.gz"


def _read_replay_day(path: Path) -> pd.DataFrame:
    """1日ぶんの保存済み再生結果を読み込む。ブール列は CSV 往復で壊れやすいため
    pipeline.py と同じ変換をかける（T-301 と同じ落とし穴を避ける）。"""
    df = pd.read_csv(path, parse_dates=DATE_COLS)
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map(_coerce_bool).astype("boolean")
    return df


def load_replay_table(output_dir: Path) -> pd.DataFrame:
    """output_dir に保存済みの replay_*.csv.gz を全て読み込み1つの表にまとめる。"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return pd.DataFrame(columns=REPLAY_COLS)
    frames = [_read_replay_day(f) for f in sorted(output_dir.glob("replay_*.csv.gz"))]
    if not frames:
        return pd.DataFrame(columns=REPLAY_COLS)
    return pd.concat(frames, ignore_index=True)


def _filter_holdout(dates: pd.DatetimeIndex, include_holdout: bool, log=print) -> pd.DatetimeIndex:
    if include_holdout:
        return dates
    kept = dates[(dates < HOLDOUT_WINDOW[0]) | (dates >= HOLDOUT_WINDOW[1])]
    skipped = len(dates) - len(kept)
    if skipped:
        log(f"[replay] ホールドアウト期間（{HOLDOUT_WINDOW[0].date()}〜{HOLDOUT_WINDOW[1].date()}）"
            f"の {skipped} 営業日をスキップ（include_holdout=False）")
    return kept


def run_replay(
    ohlcv: Dict[str, pd.DataFrame],
    idx_ohlcv: pd.DataFrame,
    listed: pd.DataFrame,
    output_dir: Path,
    start: pd.Timestamp, end: pd.Timestamp,
    k: int, label_n: int, pool_days: int,
    min_history_bars: int = MIN_HISTORY_BARS,
    earnings_schedule: Optional[pd.DataFrame] = None,
    include_holdout: bool = False,
    log=print,
) -> None:
    """start〜end の営業日ごとに1日ぶんの再生を行い、output_dir に
    replay_YYYY-MM-DD.csv.gz として保存する。

    中断再開: output_dir に既にその日のファイルがあればスキップする。再開時は、
    スキップした日の内容をプール（history_pool、DESIGN.md §6.1）の再構築に使う
    （直近 pool_days-1 日ぶんが必要なため）。

    include_holdout=False（既定）のとき、start〜end がホールドアウト
    （HOLDOUT_WINDOW）に一部でもかかっていれば、その部分の営業日を除いて再生する
    （CLAUDE.md の絶対規則: ホールドアウトは明示フラグ無しで生成しない）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(start, end)
    dates = _filter_holdout(dates, include_holdout, log=log)
    if len(dates) == 0:
        log("[replay] 再生対象の営業日が無い")
        return

    pool_history_days: List[pd.DataFrame] = []  # 直近 pool_days-1 日ぶん（新しい順ではなく古い順）
    t_start = time.monotonic()
    n_done = n_skipped = 0

    for i, date_t in enumerate(dates):
        out_path = _replay_path(output_dir, date_t)
        if out_path.exists():
            n_skipped += 1
            day_df = _read_replay_day(out_path)
            pool_history_days.append(day_df[DAILY_FEATURES_COLS])
            pool_history_days = pool_history_days[-(pool_days - 1):] if pool_days > 1 else []
            continue

        history_pool = (pd.concat(pool_history_days, ignore_index=True)
                        if pool_history_days else None)
        day_df = replay_one_day(date_t, ohlcv, idx_ohlcv, listed, k, label_n, pool_days,
                                min_history_bars, earnings_schedule, history_pool, log=log)
        day_df.to_csv(out_path, index=False, compression="gzip")
        n_done += 1

        pool_history_days.append(day_df[DAILY_FEATURES_COLS])
        pool_history_days = pool_history_days[-(pool_days - 1):] if pool_days > 1 else []

        elapsed = time.monotonic() - t_start
        if n_done % 20 == 0 or i == len(dates) - 1:
            log(f"[replay] {date_t.date()} 完了 ({i + 1}/{len(dates)} 営業日, "
                f"新規 {n_done} / 再開スキップ {n_skipped}, 経過 {elapsed:.0f}秒)")

    log(f"[replay] 完了。新規 {n_done} 日 / 再開スキップ {n_skipped} 日 "
        f"/ 合計経過 {time.monotonic() - t_start:.0f}秒")


def main(argv: Optional[list] = None) -> int:
    """python -m stockbot.validation.replay --start ... --end ... で実行する。

    store（backfill 済みの長期履歴が必要。cli.py backfill 参照）と reference の
    listed_latest.csv / earnings_schedule.csv を読み、run_replay を呼ぶ。
    """
    import argparse

    from ..config import Settings
    from ..data.jpx_lists import load_earnings_schedule, normalize_listed
    from ..data.store import IDX_TICKER, OhlcvStore, from_long

    ap = argparse.ArgumentParser(prog="python -m stockbot.validation.replay")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--include-holdout", action="store_true",
                    help="ホールドアウト期間も生成する（通常は指定しない。CLAUDE.md 絶対規則）")
    ap.add_argument("--output-dir", default=None, help="既定: <DATA_DIR>/replay")
    args = ap.parse_args(argv)

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    idx_ohlcv = ohlcv.pop(IDX_TICKER, None)
    if idx_ohlcv is None or len(idx_ohlcv) == 0:
        print("[replay] 指数データが無いため中止（先に cli.py index/daily/backfill を実行）")
        return 1

    listed_path = cfg.reference_dir / "listed_latest.csv"
    if not listed_path.exists():
        print("[replay] listed_latest.csv が無いため中止（先に cli.py listed を実行）")
        return 1
    listed = normalize_listed(pd.read_csv(listed_path, dtype=str).fillna(""))

    earnings_schedule = None
    p = cfg.reference_dir / "earnings_schedule.csv"
    if p.exists():
        earnings_schedule = load_earnings_schedule(p)

    output_dir = Path(args.output_dir) if args.output_dir else cfg.data_dir / "replay"
    run_replay(ohlcv, idx_ohlcv, listed, output_dir,
              pd.Timestamp(args.start), pd.Timestamp(args.end),
              cfg.k, cfg.label_n, cfg.pool_days, cfg.min_history_bars,
              earnings_schedule=earnings_schedule, include_holdout=args.include_holdout,
              log=print)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
