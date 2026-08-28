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
MAX_EMPTY_DAY_RATE = 0.5  # 運用上の安全弁（スコアリングのパラメータではない）。指数データや
# ユニバースが無くて空のまま処理された日の割合がこれを超えたら run_replay を異常終了させる。
# 2026-08-24 に、指数の長期履歴が backfill されておらず対象期間の全日が空のまま
# 「成功」終了していた事故があった（何も処理していないのに緑になるのが最も危険）
#
# 空の日には2種類の全く異なる原因があり、ガードは前者だけで判定する（2026-08-26 追加）。
# - EMPTY_REASON_NO_DATA: 指数データまたはユニバース（履歴 min_history_bars 本以上の
#   銘柄）が1件も無い。データ取得の不備を示す（本来のガードの対象）
# - EMPTY_REASON_NO_CANDIDATES: 指数・ユニバースは揃っているが、ゲート（特に G2:
#   Close>=SMA200）や状態条件で全銘柄が落ちた。地合い急落時（例: 2020-02〜06）に
#   正しく起こりうる市場の実態であり、データの不備ではない。ガードの対象にしない
EMPTY_REASON_NO_DATA = "no_data"
EMPTY_REASON_NO_CANDIDATES = "no_candidates"

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
        empty = pd.DataFrame(columns=REPLAY_COLS)
        empty.attrs["empty_reason"] = EMPTY_REASON_NO_DATA
        return empty
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
        # 指数・ユニバースは揃っているが、ゲートや状態条件で全銘柄が落ちた
        # （地合い急落時に正しく起こりうる。データの不備ではない）
        empty = pd.DataFrame(columns=REPLAY_COLS)
        empty.attrs["empty_reason"] = EMPTY_REASON_NO_CANDIDATES
        return empty

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


def _empty_reason_meta_path(output_dir: Path, date_t: pd.Timestamp) -> Path:
    return Path(output_dir) / f"replay_meta_{pd.Timestamp(date_t).strftime('%Y-%m-%d')}.json"


def _write_empty_reason(output_dir: Path, date_t: pd.Timestamp, reason: str) -> None:
    """空(0行)の日の原因をサイドカー JSON に残す（中断再開後もガードの判定基準を
    再現できるようにするため。.attrs は CSV に永続化されないので別ファイルに書く）。"""
    import json
    _empty_reason_meta_path(output_dir, date_t).write_text(
        json.dumps({"empty_reason": reason}), encoding="utf-8")


def _read_empty_reason(output_dir: Path, date_t: pd.Timestamp) -> str:
    """保存済みのサイドカーが無い場合は EMPTY_REASON_NO_DATA を返す（安全側のデフォルト。
    このガードが導入される前に生成された空ファイルを再開したケースを含む）。"""
    import json
    p = _empty_reason_meta_path(output_dir, date_t)
    if not p.exists():
        return EMPTY_REASON_NO_DATA
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("empty_reason", EMPTY_REASON_NO_DATA)
    except (ValueError, OSError):
        return EMPTY_REASON_NO_DATA


def _read_replay_day(path: Path) -> pd.DataFrame:
    """1日ぶんの保存済み再生結果を読み込む。ブール列は CSV 往復で壊れやすいため
    pipeline.py と同じ変換をかける（T-301 と同じ落とし穴を避ける）。"""
    df = pd.read_csv(path, parse_dates=DATE_COLS)
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map(_coerce_bool).astype("boolean")
    return df


def _load_recent_replay_days(output_dir: Path, before_date: pd.Timestamp,
                             pool_days: int) -> List[pd.DataFrame]:
    """output_dir に既に保存済みの replay_*.csv.gz のうち、before_date より前の直近
    pool_days-1 日ぶんを読み込む（DESIGN.md §6.1 のプールを区切り実行の境界でも
    途切れさせないため。pipeline.load_recent_daily_features と同じ考え方）。

    年ごとなど期間を区切って run_replay を複数回に分けて実行する場合、前の区切りが
    書いた日次ファイルは同じ output_dir に残っている前提（validation-replay ワーク
    フローの Actions cache 等）。無ければ空リスト（従来通り、プールが無い状態から
    始まる）。
    """
    output_dir = Path(output_dir)
    if pool_days <= 1 or not output_dir.exists():
        return []
    before_date = pd.Timestamp(before_date)
    dated: List[tuple] = []
    for f in output_dir.glob("replay_*.csv.gz"):
        name = f.name
        date_str = name[len("replay_"):-len(".csv.gz")]
        try:
            d = pd.Timestamp(date_str)
        except ValueError:
            continue
        if d < before_date:
            dated.append((d, f))
    dated.sort(key=lambda x: x[0])
    recent = dated[-(pool_days - 1):]
    return [_read_replay_day(f)[DAILY_FEATURES_COLS] for _d, f in recent]


def load_replay_table(output_dir: Path) -> pd.DataFrame:
    """output_dir に保存済みの replay_*.csv.gz を全て読み込み1つの表にまとめる。"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return pd.DataFrame(columns=REPLAY_COLS)
    frames = [_read_replay_day(f) for f in sorted(output_dir.glob("replay_*.csv.gz"))]
    if not frames:
        return pd.DataFrame(columns=REPLAY_COLS)
    return pd.concat(frames, ignore_index=True)


def _truncate_before_holdout(ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """CLAUDE.md 絶対規則: ホールドアウト（2026-02〜2026-08）を明示フラグ無しで見ない。

    _filter_holdout は「どの T を評価するか」を絞るだけで、T+h 形式のラベル
    （validation.labels、T-401）が近傍日（例: T=2026-01-28 で h=30 なら T+30 は
    2026-03 頃）を評価する際にホールドアウト側のバーを読んでしまうのは防げない。
    ohlcv 自体を物理的にホールドアウト開始日より前で打ち切ることで、日付範囲の
    フィルタに関わらず読めないようにする（境界に近い T はラベルが NaN/未決になる
    だけで、これは正しい挙動 —— データが無いのと同じ扱いになる）。
    """
    out = {}
    for ticker, df in ohlcv.items():
        if df is None or len(df) == 0:
            out[ticker] = df
            continue
        out[ticker] = df[df.index < HOLDOUT_WINDOW[0]]
    return out


def _filter_holdout(dates: pd.DatetimeIndex, include_holdout: bool, log=print) -> pd.DatetimeIndex:
    if include_holdout:
        return dates
    kept = dates[(dates < HOLDOUT_WINDOW[0]) | (dates >= HOLDOUT_WINDOW[1])]
    skipped = len(dates) - len(kept)
    if skipped:
        log(f"[replay] ホールドアウト期間（{HOLDOUT_WINDOW[0].date()}〜{HOLDOUT_WINDOW[1].date()}）"
            f"の {skipped} 営業日をスキップ（include_holdout=False）")
    return kept


MIN_TRADING_DAY_COVERAGE = 0.5  # T-402: 非取引日判定のしきい値（運用上の安全弁。§12の対象外）


def _real_trading_days(dates: pd.DatetimeIndex, ohlcv: Dict[str, pd.DataFrame],
                       min_frac: float = MIN_TRADING_DAY_COVERAGE, log=print) -> pd.DatetimeIndex:
    """pd.bdate_range が生成する Mon-Fri のカレンダー日から、実際の取引日でない日を
    除く（T-402: 2018-07-16=海の日でも、ごく少数の銘柄にデータの乱れがあると
    「幻の候補」が1〜2件だけ立つ日が生じることが判明。外部の祝日カレンダーは
    持ち込まず、その日にデータを持つ銘柄の割合が全体の min_frac 未満の日を
    非取引日とみなす、データ駆動の判定）。

    実際の取引日は通常ユニバースの大半（実測で約8割）にデータがあり、非取引日に
    紛れ込む幻のデータは数銘柄に限られるため、0.5 という閾値には両者を明確に
    分ける十分な余裕がある。
    """
    if len(ohlcv) == 0 or len(dates) == 0:
        return dates
    counts = pd.Series(0, index=dates, dtype=int)
    for df in ohlcv.values():
        if df is None or len(df) == 0:
            continue
        hits = df.index.intersection(dates)
        if len(hits):
            counts.loc[hits] += 1
    frac = counts / len(ohlcv)
    kept = dates[(frac >= min_frac).to_numpy()]
    dropped = len(dates) - len(kept)
    if dropped:
        dropped_dates = dates[(frac < min_frac).to_numpy()]
        log(f"[replay] 非取引日と判定して {dropped} 日を日付軸から除外: "
            f"{[d.date().isoformat() for d in dropped_dates]}")
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
    stats_start: Optional[pd.Timestamp] = None,
    log=print,
) -> None:
    """start〜end の営業日ごとに1日ぶんの再生を行い、output_dir に
    replay_YYYY-MM-DD.csv.gz として保存する。

    中断再開: output_dir に既にその日のファイルがあればスキップする。再開時は、
    スキップした日の内容をプール（history_pool、DESIGN.md §6.1）の再構築に使う
    （直近 pool_days-1 日ぶんが必要なため）。

    年ごとなど期間を区切って複数回に分けて実行する場合も、プールが区切りの境界で
    途切れないよう、start より前に output_dir に既にある直近 pool_days-1 日ぶんを
    起動時に読み込んでおく（_load_recent_replay_days。pipeline.load_recent_daily_features
    と同じ考え方）。前の区切りの出力が同じ output_dir に残っていることが前提
    （validation-replay ワークフローでは Actions cache で引き継ぐ）。出力するのは
    あくまで start〜end のぶんだけで、この事前読み込み分を書き直すことはない。

    include_holdout=False（既定）のとき、start〜end がホールドアウト
    （HOLDOUT_WINDOW）に一部でもかかっていれば、その部分の営業日を除いて再生する
    （CLAUDE.md の絶対規則: ホールドアウトは明示フラグ無しで生成しない）。
    さらに ohlcv/idx_ohlcv 自体をホールドアウト開始日より前で打ち切ってから使う
    （_truncate_before_holdout）。日付範囲のフィルタだけでは、ホールドアウト直前の
    T のラベル（T+h が h 次第でホールドアウトに入り込む）を防げないため。

    stats_start（DESIGN.md §10.1 開始日の規則）を渡すと、start〜stats_start 未満の
    日は「warmup」列に True を付けて出力する（G3 の 252 本ルックバックと §6.3 の
    プール 20 日ウォームアップのために start を stats_start より前から回す運用を
    想定。プール継続には使うが、集計からは除外する日という意味）。stats_start 以降
    は warmup=False。None（既定）なら warmup 列は付けない（主評価窓など、この区別が
    不要な既存の呼び出しに影響しない）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not include_holdout:
        ohlcv = _truncate_before_holdout(ohlcv)
        idx_ohlcv = idx_ohlcv[idx_ohlcv.index < HOLDOUT_WINDOW[0]]

    dates = pd.bdate_range(start, end)
    dates = _filter_holdout(dates, include_holdout, log=log)
    dates = _real_trading_days(dates, ohlcv, log=log)
    if len(dates) == 0:
        log("[replay] 再生対象の営業日が無い")
        return

    # 直近 pool_days-1 日ぶん（古い順）。区切り実行の境界を越えてプールが続くよう、
    # start より前に既に output_dir にある日を先に読み込んでおく
    pool_history_days: List[pd.DataFrame] = _load_recent_replay_days(output_dir, dates[0], pool_days)
    t_start = time.monotonic()
    n_done = n_skipped = 0
    n_empty_no_data = n_empty_no_candidates = 0

    for i, date_t in enumerate(dates):
        out_path = _replay_path(output_dir, date_t)
        if out_path.exists():
            n_skipped += 1
            day_df = _read_replay_day(out_path)
            if len(day_df) == 0:
                reason = _read_empty_reason(output_dir, date_t)
                if reason == EMPTY_REASON_NO_CANDIDATES:
                    n_empty_no_candidates += 1
                else:
                    n_empty_no_data += 1
            pool_history_days.append(day_df[DAILY_FEATURES_COLS])
            pool_history_days = pool_history_days[-(pool_days - 1):] if pool_days > 1 else []
            continue

        history_pool = (pd.concat(pool_history_days, ignore_index=True)
                        if pool_history_days else None)
        day_df = replay_one_day(date_t, ohlcv, idx_ohlcv, listed, k, label_n, pool_days,
                                min_history_bars, earnings_schedule, history_pool, log=log)
        if stats_start is not None:
            day_df = day_df.copy()
            day_df["warmup"] = bool(date_t < pd.Timestamp(stats_start))
        day_df.to_csv(out_path, index=False, compression="gzip")
        n_done += 1
        if len(day_df) == 0:
            reason = day_df.attrs.get("empty_reason", EMPTY_REASON_NO_DATA)
            _write_empty_reason(output_dir, date_t, reason)
            if reason == EMPTY_REASON_NO_CANDIDATES:
                n_empty_no_candidates += 1
            else:
                n_empty_no_data += 1

        pool_history_days.append(day_df[DAILY_FEATURES_COLS])
        pool_history_days = pool_history_days[-(pool_days - 1):] if pool_days > 1 else []

        elapsed = time.monotonic() - t_start
        if n_done % 20 == 0 or i == len(dates) - 1:
            log(f"[replay] {date_t.date()} 完了 ({i + 1}/{len(dates)} 営業日, "
                f"新規 {n_done} / 再開スキップ {n_skipped} / "
                f"空(データ無し) {n_empty_no_data} / 空(候補0件) {n_empty_no_candidates}, "
                f"経過 {elapsed:.0f}秒)")

    log(f"[replay] 完了。新規 {n_done} 日 / 再開スキップ {n_skipped} 日 / "
        f"空(データ無し) {n_empty_no_data} 日 / 空(候補0件・ゲート等) {n_empty_no_candidates} 日 "
        f"/ 合計経過 {time.monotonic() - t_start:.0f}秒")

    empty_rate = n_empty_no_data / len(dates) if len(dates) else 0.0
    if empty_rate > MAX_EMPTY_DAY_RATE:
        raise RuntimeError(
            f"[replay] 異常終了: 対象 {len(dates)} 営業日のうち {n_empty_no_data} 日 "
            f"({empty_rate:.0%}) が空(指数データまたはユニバースが無い等で採点対象0件)。"
            f"閾値 {MAX_EMPTY_DAY_RATE:.0%} を超えたため失敗として終了する。"
            "store の指数データ範囲や listed_latest.csv、ユニバースの本数基準を確認すること。"
            f"（別途、ゲート等で候補0件だった日が {n_empty_no_candidates} 日あるが、"
            "これはデータの不備ではないためこの判定には含めていない）"
        )


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
    ap.add_argument("--stats-start", default=None,
                    help="YYYY-MM-DD。DESIGN.md §10.1 開始日の規則。start はこれより前"
                         "（G3 の 252 本ルックバック＋§6.3 プール 20 日ウォームアップの"
                         "ため）から回し、stats_start 未満の日は warmup=True で出力する"
                         "（集計除外用。省略時は warmup 列を付けない）")
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
    try:
        run_replay(ohlcv, idx_ohlcv, listed, output_dir,
                  pd.Timestamp(args.start), pd.Timestamp(args.end),
                  cfg.k, cfg.label_n, cfg.pool_days, cfg.min_history_bars,
                  earnings_schedule=earnings_schedule, include_holdout=args.include_holdout,
                  stats_start=pd.Timestamp(args.stats_start) if args.stats_start else None,
                  log=print)
    except RuntimeError as e:
        print(str(e))
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
