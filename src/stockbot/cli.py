"""CLI。GitHub Actions とローカルの両方から同じ手順で呼ぶ。

  python -m stockbot.cli daily       # 取得 → 整合性検査 → 保存・スナップショット → ユニバース
  python -m stockbot.cli listed      # JPX 上場銘柄一覧の更新のみ
  python -m stockbot.cli fetch       # 取得と保存のみ
  python -m stockbot.cli index       # 指数（TOPIX/日経225）の取得と保存のみ
  python -m stockbot.cli backfill    # 検証用の長期履歴取得（HISTORY_DAYS=2600 等）。中断再開可
  python -m stockbot.cli references  # 決算発表予定日・上場廃止銘柄一覧の更新のみ
  python -m stockbot.cli universe    # 保存済みデータからユニバースを再計算
  python -m stockbot.cli features    # 保存済みデータから日次特徴量を再計算・保存
  python -m stockbot.cli pattern     # 反転系パターンの検出数を数える（docs/PATTERN.md §2.1）
  python -m stockbot.cli resolve     # 配信記録に5営業日後の結果を付ける（docs/SCREENER.md §3.3）
  python -m stockbot.cli notify      # その日の配信記録を LINE に流す（docs/SCREENER.md §4）

環境変数: SPEC/README 参照。SCREEN_DRYRUN=1 で合成データ・ネットワーク不要。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .data.adjust import check_all
from .data.jpx_lists import (
    fetch_delistings,
    fetch_earnings_schedule,
    load_earnings_schedule,
    load_listed_with_fallback,
    load_manual_exclusions,
    normalize_listed,
)
from .data.store import IDX_TICKER, OhlcvStore, from_long, to_long
from .data.synthetic import make_synthetic, make_synthetic_index, synthetic_listed
from .data.yf_fetch import fetch_index, fetch_ohlcv
from .features.dimensions import next_earnings_business_days
from .features.indicators import atr_wilder
from .features import (  # noqa: F401
    indicators,
    pattern as pattern_mod,
    pullback,
    regime,
    sector as sector_mod,
    swings,
)
from .pipeline import (
    DAILY_FEATURES_COLS,
    compute_daily_features,
    load_recent_daily_features,
    save_daily_features,
)
from .notify import line_send, message
from .render import context as render_context
from .render import render as render_images_mod
from .screener import pattern_record, record, resolver
from .validation import pattern_exit, pattern_replay, pattern_report
from .universe.build import build_universe, liquidity_stats, load_latest_universe, save_universe, summarize

JST = "Asia/Tokyo"


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=JST)


# ------------------------------------------------------------------ steps
def step_listed(cfg: Settings, log=print) -> pd.DataFrame:
    cfg.ensure_dirs()
    if cfg.dryrun:
        listed = synthetic_listed(60)
        src = "synthetic"
    else:
        listed, src = load_listed_with_fallback(cfg.jpx_listed_url, cfg.universe_seed_csv, log=log)
    asof = _now().strftime("%Y-%m-%d")
    listed.to_csv(cfg.reference_dir / "listed_latest.csv", index=False, encoding="utf-8-sig")
    listed.to_csv(cfg.reference_dir / f"listed_{asof}.csv", index=False, encoding="utf-8-sig")
    n_eq = int(listed["is_equity"].sum())
    log(f"[listed] source={src} 全{len(listed)} 株式{n_eq}")
    return listed


def _load_listed_cached(cfg: Settings, log=print) -> pd.DataFrame:
    p = cfg.reference_dir / "listed_latest.csv"
    if p.exists():
        return normalize_listed(pd.read_csv(p, dtype=str).fillna(""))
    return step_listed(cfg, log=log)


def _target_tickers(cfg: Settings, listed: pd.DataFrame, log=print) -> list[str]:
    """取得対象。auto は「前回ユニバースがあれば通過銘柄＋境界付近、月曜は全件」。"""
    eq = listed[listed["is_equity"].astype(bool)]["ticker"].tolist()
    scope = cfg.fetch_scope
    if scope == "all":
        return eq
    prev = load_latest_universe(cfg.universe_dir)
    if scope == "universe" or (scope == "auto" and prev is not None and _now().weekday() != 0):
        if prev is None:
            return eq
        # 通過銘柄 + 売買代金が下限の 1/2 以上の不合格銘柄（再資格化の取りこぼし防止）
        near = prev[(prev["passes"]) | (prev["adv_jpy"] >= cfg.min_adv_jpy * 0.5)]["ticker"].tolist()
        # 新規上場（前回リストに無い銘柄）も拾う
        new = [t for t in eq if t not in set(prev["ticker"])]
        sel = sorted(set(near) | set(new))
        log(f"[fetch] scope=universe: {len(sel)}/{len(eq)} 銘柄")
        return sel
    return eq


def _refetch_new_splits_full_history(
    ohlcv: dict, issues: pd.DataFrame, cfg: Settings, now: pd.Timestamp, log=print,
    fetch_fn=fetch_ohlcv,
) -> tuple[dict, pd.DataFrame, list[str]]:
    """T-402: 日次取得窓（history_days、既定400本）の中で新規に記録された分割
    （check_splits の kind=="unadjusted_split"）は、その窓の中だけを調整するため、
    窓より過去のstore側の値は未調整のまま残り、窓の境界（400本前後）に段差が立つ
    （9900.Tで判明）。該当銘柄だけ history_full_days（既定2600本）で全履歴を
    再取得し、check_splits を掛け直して ohlcv/issues を差し替える。

    戻り値の3番目（refetched_tickers）は呼び出し側が store.upsert_replace に渡す
    ためのもの。全履歴再取得はマージだと取得ウィンドウの外の古い行が取り残され
    段差が再発するため、対象銘柄は置換で store に反映する（T-402、2026-08-29）。
    """
    if len(issues) == 0:
        return ohlcv, issues, []
    newly_split = sorted(issues.loc[issues["kind"] == "unadjusted_split", "ticker"].unique())
    if not newly_split:
        return ohlcv, issues, []
    log(f"[fetch] 新規Splitsイベント検出（{len(newly_split)}銘柄）: "
        f"全履歴({cfg.history_full_days}本)を再取得: {newly_split}")
    full_ohlcv, _full_meta = fetch_fn(newly_split, cfg.history_full_days, cfg.fetch_deadline_sec,
                                      now_jst=now, close_hhmm=cfg.market_close_hhmm, log=log)
    full_ohlcv, full_issues = check_all(full_ohlcv)
    ohlcv = dict(ohlcv)
    ohlcv.update(full_ohlcv)
    issues = pd.concat([issues[~issues["ticker"].isin(newly_split)], full_issues], ignore_index=True)
    return ohlcv, issues, newly_split


def step_fetch(cfg: Settings, listed: pd.DataFrame, log=print) -> tuple[dict, dict, pd.DataFrame]:
    cfg.ensure_dirs()
    now = _now()
    tickers = _target_tickers(cfg, listed, log=log)
    if cfg.dryrun:
        tickers = tickers[:60]
        ohlcv = make_synthetic(tickers, n_bars=cfg.history_days, end=now.tz_localize(None))
        meta = {"data_total": len(tickers), "data_ok": len(ohlcv), "short": [], "failed": [],
                "elapsed_sec": 0.0, "rounds": [], "period": "synthetic", "asof": str(now)}
    else:
        ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, cfg.fetch_deadline_sec,
                                  now_jst=now, close_hhmm=cfg.market_close_hhmm, log=log)
    ohlcv, issues = check_all(ohlcv)
    refetched_tickers: list[str] = []
    if not cfg.dryrun:
        ohlcv, issues, refetched_tickers = _refetch_new_splits_full_history(ohlcv, issues, cfg, now, log=log)
    log(f"[fetch] ok {meta['data_ok']}/{meta['data_total']} / 履歴不足 {len(meta['short'])} / "
        f"失敗 {len(meta['failed'])} / 分割issue {len(issues)}")

    store = OhlcvStore(cfg.store_dir, cfg.daily_dir, cfg.rev_close_tol, cfg.rev_volume_tol)
    merged, added, revisions = store.upsert_replace(to_long(ohlcv), refetched_tickers)
    store.save(merged)
    files = store.write_daily_increments(added)
    store.append_revisions(revisions, now)
    if len(issues):
        p = cfg.store_dir / "split_issues.csv"
        prev = pd.read_csv(p) if p.exists() else None
        issues["observed_on"] = now.strftime("%Y-%m-%d")
        allx = pd.concat([prev, issues], ignore_index=True) if prev is not None else issues
        allx.drop_duplicates(subset=["ticker", "date", "kind"], keep="last").to_csv(p, index=False)
    log(f"[store] rows {len(merged)} / 新規 {len(added)} / 改訂 {len(revisions)} / 日次ファイル {len(files)}")
    meta.update({"store_rows": int(len(merged)), "added": int(len(added)),
                 "revisions": int(len(revisions)), "split_issues": int(len(issues))})
    (cfg.store_dir / "fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    # ユニバース計算には保存データ全体（今回取得しなかった銘柄の直近データを含む）を使う
    return from_long(merged), meta, issues


INDEX_LABELS = {"^TPX": "TOPIX", "1306.T": "TOPIX(ETF代替: 1306.T)", "^N225": "日経225"}
# 1306.T はTOPIXを追跡するETFで別物の指数ではないため、フォールバック扱いにしない
# （D2相対力・地合いゲージの基準はTOPIXのまま）。日経225だけが本当の代替
INDEX_TRUE_FALLBACKS = {"^N225"}


def step_index(cfg: Settings, log=print) -> pd.DataFrame:
    """指数（TOPIX、失敗時は日経225）を取得し、store に ticker=IDX_TICKER として保存する（T-102）。

    D2（相対力）と地合いゲージ（DESIGN.md §8.1）が後で参照する。DRYRUN は合成。
    実際にどちらのティッカーが使われたかは store/index_meta.json に記録する
    （TOPIXが取れず日経225にフォールバックした場合、validation.report がL1レポートに
    明記するために参照する）。
    """
    cfg.ensure_dirs()
    now = _now()
    if cfg.dryrun:
        df = make_synthetic_index(n_bars=cfg.history_days, end=now.tz_localize(None))
        used = IDX_TICKER
    else:
        df, used = fetch_index(cfg.history_days, now_jst=now, close_hhmm=cfg.market_close_hhmm, log=log)
    if len(df) == 0:
        log("[index] 取得できず。store は前回値のまま")
        return df
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir, cfg.rev_close_tol, cfg.rev_volume_tol)
    merged, added, revisions = store.upsert(to_long({IDX_TICKER: df}))
    store.save(merged)
    store.write_daily_increments(added)
    store.append_revisions(revisions, now)
    # 前回成功時の記録を、今回が失敗（len(df)==0、上のreturnで既に抜けている）で
    # 上書きしないよう、成功時のみ書く（storeを「前回値のまま」にするのと同じ考え方）
    label = "合成(DRYRUN)" if cfg.dryrun else INDEX_LABELS.get(used, used or "不明")
    meta = {"ticker": used, "label": label,
           "is_fallback": (not cfg.dryrun) and used in INDEX_TRUE_FALLBACKS}
    (cfg.store_dir / "index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"[index] source={used or 'synthetic'} 本数={len(df)} 新規={len(added)} 改訂={len(revisions)}")
    return df


def _bar_counts(long_df: pd.DataFrame) -> pd.Series:
    """縦持ち OHLCV から銘柄ごとの本数。"""
    if long_df is None or len(long_df) == 0:
        return pd.Series(dtype=int)
    return long_df.groupby("ticker")["date"].count()


def step_backfill(cfg: Settings, log=print) -> dict:
    """検証用の長期履歴を取得する（T-103）。`daily` とは別コマンド。

    HISTORY_DAYS（例: 2600）で全上場株式を取得する。「取得済み」の判定は本数基準
    ―― 銘柄ごとの store 本数が int(HISTORY_DAYS * 0.9) 未満なら、store に存在して
    いても再取得対象に含める（本数不足のまま取りこぼさない）。締切
    （FETCH_DEADLINE_SEC）に到達して中断した場合、次回実行時は基準を満たす銘柄
    をスキップして残りだけを取得する。
    """
    cfg.ensure_dirs()
    now = _now()
    listed = _load_listed_cached(cfg, log)
    eq = listed[listed["is_equity"].astype(bool)]["ticker"].tolist()

    store = OhlcvStore(cfg.store_dir, cfg.daily_dir, cfg.rev_close_tol, cfg.rev_volume_tol)
    threshold = int(cfg.history_days * 0.9)
    existing_counts = _bar_counts(store.load())
    already_done = [t for t in eq if existing_counts.get(t, 0) >= threshold]
    target = [t for t in eq if existing_counts.get(t, 0) < threshold]
    log(f"[backfill] ユニバース {len(eq)} 銘柄 / 本数基準(>= {threshold}) 済み {len(already_done)} 件をスキップ / "
        f"残り {len(target)} 件を取得")

    if cfg.dryrun:
        ohlcv = make_synthetic(target, n_bars=cfg.history_days, end=now.tz_localize(None))
        meta = {"data_total": len(target), "data_ok": len(ohlcv), "short": [], "failed": [],
                "elapsed_sec": 0.0, "rounds": [], "period": "synthetic", "asof": str(now)}
    elif not target:
        meta = {"data_total": 0, "data_ok": 0, "short": [], "failed": [],
                "elapsed_sec": 0.0, "rounds": [], "period": "", "asof": str(now)}
        ohlcv = {}
    else:
        ohlcv, meta = fetch_ohlcv(target, cfg.history_days, cfg.fetch_deadline_sec,
                                  now_jst=now, close_hhmm=cfg.market_close_hhmm, log=log)
    ohlcv, issues = check_all(ohlcv)

    merged, added, revisions = store.upsert(to_long(ohlcv))
    store.save(merged)
    store.write_daily_increments(added)
    store.append_revisions(revisions, now)
    if len(issues):
        p = cfg.store_dir / "split_issues.csv"
        prev = pd.read_csv(p) if p.exists() else None
        issues["observed_on"] = now.strftime("%Y-%m-%d")
        allx = pd.concat([prev, issues], ignore_index=True) if prev is not None else issues
        allx.drop_duplicates(subset=["ticker", "date", "kind"], keep="last").to_csv(p, index=False)

    merged_counts = _bar_counts(merged)
    cumulative_done = int(sum(1 for t in eq if merged_counts.get(t, 0) >= threshold))
    completion_rate = (cumulative_done / len(eq)) if eq else 0.0
    meta.update({
        "universe_total": len(eq),
        "bar_threshold": threshold,
        "already_done": len(already_done),
        "newly_done": meta["data_ok"],
        "cumulative_done": cumulative_done,
        "completion_rate": round(completion_rate, 4),
        "store_rows": int(len(merged)), "added": int(len(added)),
        "revisions": int(len(revisions)), "split_issues": int(len(issues)),
    })
    (cfg.store_dir / "backfill_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"[backfill] 累計 {cumulative_done}/{len(eq)} ({completion_rate:.1%}) / "
        f"今回取得 {meta['data_ok']} / 新規行 {len(added)}")
    return meta


def step_refetch_recent_splits(cfg: Settings, log=print, fetch_fn=fetch_ohlcv) -> dict:
    """T-402の恒久的な保守作業: storeの直近history_days本以内にStock Splitsイベント
    が記録されている全銘柄を、history_full_daysぶん全履歴再取得してstoreを整合
    させる（`daily`/`backfill`とは別コマンド、定期的または手動で実行する想定）。

    日次fetch（history_days、既定400本）の窓内で分割が起きると、窓の中だけが
    調整され窓より過去のstore側は未調整のまま残り段差が生じる（9900.Tほか9銘柄
    で確認、TASKS.md T-402）。`_refetch_new_splits_full_history`は新規検出時に
    その場で対応するが、それより前に発生した分割は対象外。本関数はstoreの現在
    状態を直接スキャンして能動的に洗い出す。
    """
    cfg.ensure_dirs()
    now = _now()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir, cfg.rev_close_tol, cfg.rev_volume_tol)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)

    targets = []
    for ticker, df in ohlcv.items():
        if df is None or len(df) == 0 or "Stock Splits" not in df.columns:
            continue
        recent = df.tail(cfg.history_days)
        if (recent["Stock Splits"].fillna(0) != 0).any():
            targets.append(ticker)
    targets = sorted(targets)
    log(f"[refetch-recent-splits] 直近{cfg.history_days}本以内にSplitsイベントがある銘柄: "
        f"{len(targets)}件 {targets}")

    if not targets:
        return {"tickers": [], "n_issues": 0}

    if cfg.dryrun:
        full_ohlcv = make_synthetic(targets, n_bars=cfg.history_full_days, end=now.tz_localize(None))
    else:
        full_ohlcv, _meta = fetch_fn(targets, cfg.history_full_days, cfg.fetch_deadline_sec,
                                     now_jst=now, close_hhmm=cfg.market_close_hhmm, log=log)
    full_ohlcv, issues = check_all(full_ohlcv)

    # マージではなく置換（T-402、2026-08-29）: fetchできた範囲だけがstoreに残る
    # ようにし、取得ウィンドウの外に古い行が取り残されて段差が再発するのを防ぐ
    merged, added, revisions = store.upsert_replace(to_long(full_ohlcv), targets)
    store.save(merged)
    store.write_daily_increments(added)
    store.append_revisions(revisions, now)
    log(f"[refetch-recent-splits] 完了: {len(targets)}銘柄 / 新規行 {len(added)} / "
        f"改訂 {len(revisions)} / issue {len(issues)}件")
    return {"tickers": targets, "n_issues": int(len(issues)), "added": int(len(added)),
           "revisions": int(len(revisions))}


def step_references(cfg: Settings, log=print) -> None:
    """決算発表予定日・上場廃止銘柄一覧を JPX から取得し reference/ を更新する（T-104）。

    決算発表予定日はローリング更新（直近1〜2ヶ月分のみ）で完全網羅ではない。
    詳細・制約は docs/DATA_SOURCES.md 参照。取得失敗や0件時は既存ファイルを維持する。
    DRYRUN はネットワークを叩かず何もしない。
    """
    cfg.ensure_dirs()
    if cfg.dryrun:
        log("[references] DRYRUN のためスキップ（既存ファイルを維持）")
        return

    earnings = fetch_earnings_schedule(cfg.jpx_earnings_url, log=log)
    if len(earnings):
        earnings.to_csv(cfg.reference_dir / "earnings_schedule.csv", index=False, encoding="utf-8-sig")
        log(f"[references] 決算発表予定日 {len(earnings)} 件を保存")
    else:
        log("[references] 決算発表予定日 取得0件 → 既存ファイルを維持")

    delistings = fetch_delistings(cfg.jpx_delistings_url, log=log)
    if len(delistings):
        delistings.to_csv(cfg.reference_dir / "delistings.csv", index=False, encoding="utf-8-sig")
        log(f"[references] 上場廃止 {len(delistings)} 件を保存")
    else:
        log("[references] 上場廃止 取得0件 → 既存ファイルを維持")


def step_universe(cfg: Settings, listed: pd.DataFrame, ohlcv: dict | None = None,
                  issues: pd.DataFrame | None = None, log=print) -> pd.DataFrame:
    cfg.ensure_dirs()
    if ohlcv is None:
        store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
        ohlcv = from_long(store.load())
    exclude: list[str] = []
    if issues is not None and len(issues):
        exclude = issues[issues["kind"] == "suspected_unrecorded_split"]["ticker"].unique().tolist()
    manual = load_manual_exclusions(cfg.reference_dir / "manual_exclusions.csv", _now())
    if manual:
        log(f"[universe] 手動除外リスト（データ品質）: {manual}")
        exclude = sorted(set(exclude) | set(manual))
    stats = liquidity_stats(ohlcv, cfg.adv_window)
    u = build_universe(listed, stats, cfg.min_adv_jpy, cfg.min_price, cfg.min_history_bars,
                       exclude, asof=_now(), max_staleness_days=cfg.max_staleness_days)
    dated, latest = save_universe(u, cfg.universe_dir, _now())
    s = summarize(u)
    (cfg.universe_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    log(f"[universe] 株式 {s['equities']} / データあり {s['with_data']} / 通過 {s['passes']} "
        f"(売買代金不足 {s['fail_adv']}, 低位株 {s['fail_price']}, 履歴不足 {s['fail_history']}, "
        f"分割疑い {s['fail_split']}, 鮮度不足 {s['fail_fresh']})")
    return u


def step_features(cfg: Settings, universe: pd.DataFrame, ohlcv: dict, log=print) -> pd.DataFrame:
    """全採点銘柄（状態が形成中/反発開始/ブレイク）の特徴量・状態・地合い・プール正規化
    スコア（次元スコア・総合スコア V1/V2/V3、DESIGN.md §6、T-301）を計算し
    daily/features_YYYY-MM-DD.csv.gz に保存する（T-206/T-301）。

    d3_template（T-302）は未実装のため常に NaN（次元合成では欠損 0.5 として扱われる）。
    """
    cfg.ensure_dirs()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    stored = from_long(store.load())
    idx_df = stored.get(IDX_TICKER)
    if idx_df is None or len(idx_df) == 0:
        log("[features] 指数データが無いためスキップ")
        return pd.DataFrame(columns=DAILY_FEATURES_COLS)

    earnings_schedule = None
    p = cfg.reference_dir / "earnings_schedule.csv"
    if p.exists():
        earnings_schedule = load_earnings_schedule(p)

    asof = idx_df["Close"].index[-1]
    history_pool = load_recent_daily_features(cfg.daily_dir, asof, cfg.pool_days)
    tickers = universe[universe["passes"]]["ticker"].tolist()
    df = compute_daily_features(ohlcv, tickers, idx_df["Close"], cfg.k, cfg.label_n,
                                earnings_schedule=earnings_schedule,
                                history_pool=history_pool, pool_days=cfg.pool_days, log=log)
    path = save_daily_features(df, cfg.daily_dir, _now())
    log(f"[features] ユニバース通過 {len(tickers)} 銘柄中 {len(df)} 件を {path.name} に保存")
    return df


PATTERN_LIST_MAX = 30   # 目視用に列挙する上限。多すぎるとログが読めない


def _num(value, digits: int = 1) -> str:
    """欠損を「—」にする。高さが定義できない行では目標も比率も出ない。"""
    return "—" if pd.isna(value) else f"{float(value):,.{digits}f}"


def _signed(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):+.1f}%"


def _dates(row, keys) -> str:
    """極値を「日付 値」の並びにする。欠損（ダブルボトムの l3 など）は飛ばす。"""
    parts = []
    for key in keys:
        d, v = row.get(f"{key}_date"), row.get(key)
        if pd.isna(d) or pd.isna(v):
            continue
        parts.append(f"{pd.Timestamp(d):%m-%d} {float(v):.1f}")
    return " / ".join(parts) if parts else "—"


def _log_slopes(label: str, part: pd.DataFrame, log) -> None:
    """保ち合い系の上辺・下辺の傾きの分布（docs/PATTERN.md §5 D-11）。

    **観測であって判定ではない。** 上昇三角（C1）の `下辺傾き > 0` に下限が無いので、
    わずかでも正なら通る —— 実質「上辺が水平」だけで通っていないかを切り分けるため
    の表示である。**下辺傾きが ε 未満の件数**を併記する（ε 未満なら「上向き」では
    なく「水平」で、その行は実質ボックスに近い）。

    件数を見て閾値を足すためのものではない（§1 の感度分析はしない方針はそのまま）。
    """
    eps_pct = pattern_mod.EPSILON_SLOPE * 100
    for name in pattern_mod.CONSOLIDATION_PATTERNS:
        sub = part[part["pattern"] == name]
        if not len(sub):
            continue
        for side, col in (("上辺", "upper_slope"), ("下辺", "lower_slope")):
            v = sub[col].dropna() * 100
            if not len(v):
                continue
            line = (f"[pattern] {label}の{name} {side}傾き: 中央 {v.median():+.4f}%/日 / "
                    f"最小 {v.min():+.4f} / 最大 {v.max():+.4f}")
            if col == "lower_slope":
                # ε 未満 ＝「上向き」ではなく「水平」。C1 が実質ボックスになっていないか
                line += f" / |傾き| < ε({eps_pct:.1f}%) が {int((v.abs() < eps_pct).sum())} 件"
            log(line)
        _log_scatter(label, name, sub, log)


def _log_scatter(label: str, name: str, sub: pd.DataFrame, log) -> None:
    """上辺の散らばり（docs/PATTERN.md §2.2・§5 D-13）。**判定には使わない。**

    C1 は上辺の**傾き**だけを制約していて散らばりを制約していないので、山が 3 点
    あると中央が凹んだ V 字でも ε を通る。**傾きを見ているだけでは V 字と水平線を
    区別できない**ので、平均からの最大乖離を別に数える。

    ボックスの水平許容 ±0.75%（`BOX_TOL`）と同じ測り方なので、「この C1 はボックスの
    基準なら落ちる」が読める。**C1 に散らばり条件は足さない**（設計責任者の判断）。
    """
    tol_pct = pattern_mod.BOX_TOL * 100
    v = sub["upper_scatter"].dropna()      # 山 2 点の行は NaN（散らばりが定義できない）
    if not len(v):
        return
    log(f"[pattern] {label}の{name} 山3点以上: {len(v)}/{len(sub)} 件 / "
        f"上辺の散らばり（平均からの最大乖離）中央 {v.median():.2f}% / "
        f"最大 {v.max():.2f}% / "
        f"±{tol_pct:.2f}%（BOX 基準）超 {int((v > tol_pct).sum())} 件")


def _display_columns(cfg: Settings, universe: pd.DataFrame, out: pd.DataFrame,
                     log=print) -> pd.DataFrame:
    """カードに載せる列を足す（docs/PATTERN.md §6.2）。**判定には一切使わない。**

    銘柄名・33業種・20日平均売買代金はユニバースから、決算までの日数は
    `reference/earnings_schedule.csv` から引く。業種は表示のみで並び順に使わない（§6.5）。

    **決算日は 9 割方「未取得」になる**（JPX のカバー率が 2.5〜10.4% で振れる）。
    それでも載せる —— 載せないと取れている 1 割の情報まで消える。
    """
    if not len(out):
        for col in ("name", "sector33", "adv_jpy", "earnings_days", "earnings_unknown"):
            out[col] = pd.Series(dtype="object")
        return out
    have = [c for c in ("ticker", "name", "sector33", "adv_jpy") if c in universe.columns]
    out = out.merge(universe[have].drop_duplicates("ticker"), on="ticker", how="left")
    # **列は必ず作る。** ユニバースに無くても記録とカードの形を変えない
    for col in ("name", "sector33"):
        out[col] = out[col].fillna("") if col in out.columns else ""
    if "adv_jpy" not in out.columns:
        out["adv_jpy"] = np.nan

    schedule = None
    path = cfg.reference_dir / "earnings_schedule.csv"
    if path.exists():
        schedule = load_earnings_schedule(path)
    days = []
    for _i, r in out.iterrows():
        value = (next_earnings_business_days(schedule, r["asof"], str(r["ticker"]))
                 if schedule is not None else None)
        days.append(np.nan if value is None else float(value))
    out["earnings_days"] = days
    out["earnings_unknown"] = out["earnings_days"].isna()
    known = int((~out["earnings_unknown"]).sum())
    log(f"[pattern] 決算日が取れた行 {known}/{len(out)}")
    return out


def step_pattern(cfg: Settings, universe: pd.DataFrame, ohlcv: dict,
                 log=print) -> pd.DataFrame:
    """7 パターンを検出し、成立を記録して監視をスナップショットに残す
    （docs/PATTERN.md §2.1 反転系・§2.2 保ち合い系・§3 記録）。

    判定日 T は各銘柄の最終足。**T の引けまでのデータしか読まない。**

    書くものは 2 つ。

    - `delivered_<配信日>_asof<判定日>.csv` —— **成立した行だけ**（§3.1）。
      既にあれば上書きしない（台帳は後から作り直さない）
    - `pattern_summary_<配信日>_asof<判定日>.json` —— 要約と**その日の監視**（§3.2）。
      台帳ではなく観測値なので上書きしてよい

    **監視は記録しない。** 成立は事象（抜けた日 1 日）、監視は状態（同じ形が何日も
    続く）で単位が違う。監視を台帳に書くと分母が「形 × 滞留日数」になる。

    **「形は揃ったが未抜け」の件数も出す。** 成立が 0 件だったときに、条件が厳しいのか
    実装が間違っているのかを切り分けるため（設計責任者の指示）。
    """
    tickers = universe[universe["passes"]]["ticker"].tolist()
    rows = []
    n_eval = 0
    n_swings_short = 0
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < pattern_mod.SEARCH_WINDOW + cfg.k * 2:
            continue
        n_eval += 1
        t_pos = len(df) - 1
        # **形だけ揃った行も取る**（include_pending）。検出 0 件だったときに
        # 「条件が厳しい」のか「実装が間違っている」のかを 1 回で切り分けるため
        hits = pattern_mod.detect_patterns(df["High"], df["Low"], df["Close"],
                                           t_pos, cfg.k, include_pending=True)
        if len(hits) == 0:
            n_swings_short += 1
        atr = atr_wilder(df["High"], df["Low"], df["Close"]).to_numpy(dtype=float)
        atr_t = float(atr[t_pos]) if np.isfinite(atr[t_pos]) else np.nan
        for _i, r in hits.iterrows():
            # 極値の位置を日付に直す。**チャートで探せるようにするため**（目視確認用、
            # docs/PATTERN.md §5 D-4 の「形を目視確認してから」に対応）
            dates = {}
            for col in ("l1_pos", "l2_pos", "l3_pos", "h1_pos", "h2_pos", "h3_pos"):
                pos = r[col]
                dates[col.replace("_pos", "_date")] = (
                    df.index[int(pos)] if pd.notna(pos) else pd.NaT)
            rows.append({"ticker": ticker, "asof": df.index[t_pos], "atr_t": atr_t,
                         **r.to_dict(), **dates})

    date_cols = ["l1_date", "l2_date", "l3_date", "h1_date", "h2_date", "h3_date"]
    out = pd.DataFrame(
        rows, columns=["ticker", "asof", "atr_t"] + pattern_mod.PATTERN_COLS + date_cols)
    if len(out):
        # ネックラインまでの距離。**監視の並び順（明日抜けるかもしれない順）**に使う。
        # 優劣ではないし、絞り込みの閾値でもない（§6.1）
        out["to_neck_pct"] = (out["neckline"] / out["close_t"] - 1.0) * 100
    out = _display_columns(cfg, universe, out, log)
    done = out[out["breakout"].astype(bool)] if len(out) else out
    pending = out[~out["breakout"].astype(bool)] if len(out) else out
    if len(pending):
        pending = pending.sort_values("to_neck_pct").reset_index(drop=True)
    log(f"[pattern] 評価 {n_eval} 銘柄")
    log(f"[pattern] 成立（ネックライン上抜け済み）: {len(done)} 件")
    log(f"[pattern] 形は揃ったが未抜け: {len(pending)} 件")
    log(f"[pattern] 形も揃わなかった銘柄: {n_swings_short} 銘柄")

    for label, part in (("成立", done), ("未抜け", pending)):
        if not len(part):
            continue
        counts = part["pattern"].value_counts()
        log(f"[pattern] {label}の内訳: "
            + " ".join(f"{k}:{int(v)}" for k, v in counts.items()))
        log(f"[pattern] {label}の span（最初の極値から T まで）: 中央 "
            f"{part['span'].median():.0f}本 / 最小 {part['span'].min():.0f} / "
            f"最大 {part['span'].max():.0f}")
        # 抜け幅の分布（§5 D-7）。**下限は入れていない。** 分布を見るための記録
        bo = part["breakout_pct"].dropna()
        if len(bo):
            log(f"[pattern] {label}の抜け幅: 中央 {bo.median():+.2f}% / "
                f"最小 {bo.min():+.2f}% / 最大 {bo.max():+.2f}%")
        bh = part["breakout_h"].dropna()
        if len(bh):
            log(f"[pattern] {label}の抜け幅÷高さ: 中央 {bh.median():+.3f} / "
                f"最小 {bh.min():+.3f} / 最大 {bh.max():+.3f}")
        rr = part["rr"].dropna()
        if len(rr):
            log(f"[pattern] {label}の比率（測定目標÷撤退・文献上の目安）: 中央 "
                f"{rr.median():.2f} / 最小 {rr.min():.2f} / 最大 {rr.max():.2f}")
            be = part["breakeven_win_rate"].dropna()
            log(f"[pattern] {label}の損益分岐勝率: 中央 {be.median():.1%} / "
                f"最小 {be.min():.1%} / 最大 {be.max():.1%}")
        _log_slopes(label, part, log)

    if len(out):
        multi = out.groupby("ticker")["pattern"].nunique()
        log(f"[pattern] 同じ銘柄で複数パターンに該当: {int((multi > 1).sum())} 銘柄")

    # **成立と未抜けの両方を銘柄つきで出す。** 未抜けを出すのは、成立 0 件が続く間も
    # 「その形をダブルボトムと呼んでよいか」をチャートで確かめられるようにするため
    # （docs/PATTERN.md §5 D-4）。**閾値の判断には使わない**
    for label, part in (("成立", done), ("未抜け", pending)):
        if not len(part):
            continue
        # ネックラインに近い順。目視する順番を決めるだけで、絞り込みではない
        part = part.sort_values("to_neck_pct")
        log(f"[pattern] --- {label} {len(part)}件（ネックラインに近い順）---")
        for _i, r in part.head(PATTERN_LIST_MAX).iterrows():
            log(f"[pattern]   {r['ticker']:<9} {r['pattern']:<14} "
                f"終値 {r['close_t']:.1f} → ネックライン {r['neckline']:.1f} "
                f"({r['to_neck_pct']:+.1f}%) / span {int(r['span'])}本")
            log(f"[pattern]     谷 {_dates(r, ('l1', 'l2', 'l3'))}")
            log(f"[pattern]     山 {_dates(r, ('h1', 'h2', 'h3'))}")
            if pd.notna(r["upper_slope"]):
                # 保ち合い系だけ。ε（日次 0.1%）と比べられるよう %/日 で出す
                log(f"[pattern]     上辺 {r['upper_slope'] * 100:+.3f}%/日 / "
                    f"下辺 {r['lower_slope'] * 100:+.3f}%/日"
                    + ("" if pd.isna(r["pole_pct"]) else
                       f" / 旗竿 {r['pole_pct']:+.1f}%")
                    # 山 3 点以上のときだけ。V 字を水平線と取り違えないための表示
                    + ("" if pd.isna(r["upper_scatter"]) else
                       f" / 上辺の散らばり {r['upper_scatter']:.2f}%"))
            log(f"[pattern]     撤退 {r['pattern_low']:.1f} ({r['down_pct']:+.1f}%) / "
                f"目標 {_num(r['target'])} ({_signed(r['up_pct'])}) / "
                f"比率 {_num(r['rr'], 2)}")
        if len(part) > PATTERN_LIST_MAX:
            log(f"[pattern]   ... 他 {len(part) - PATTERN_LIST_MAX} 件")

    _save_pattern_records(cfg, out, done, pending, n_eval, log)
    return out


def _pattern_counts(done: pd.DataFrame, pending: pd.DataFrame) -> dict:
    """パターン別の (成立, 未抜け) 件数。**2枚目の内訳と同じ表を作る値。**

    表示側と検出側で別々に数えると食い違いうるので、**同じフレームから 1 回だけ数える**。
    """
    counts: dict = {}
    for label, part in (("done", done), ("watch", pending)):
        if not len(part):
            continue
        for name, n in part["pattern"].value_counts().items():
            counts.setdefault(str(name), {"done": 0, "watch": 0})[label] = int(n)
    return counts


def _json_row(row, cols) -> dict:
    """1 行を JSON に入る型へ。日付は ISO 文字列（読む側は文字列でも Timestamp でも可）。"""
    out = {}
    for col in cols:
        if col not in row.index:
            continue
        v = row[col]
        if isinstance(v, pd.Timestamp):
            out[col] = None if pd.isna(v) else f"{v:%Y-%m-%d}"
        # bool は int の派生なので**先に**見る（True が 1 になってしまう）
        elif isinstance(v, (np.bool_, bool)):
            out[col] = bool(v)
        elif pd.isna(v):
            out[col] = None
        # iterrows() の行は object dtype になりうるので、素の int / float も受ける。
        # ここを numpy 型だけにすると span や watch_streak が文字列で書かれる
        elif isinstance(v, (np.integer, int)):
            out[col] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[col] = float(v)
        else:
            out[col] = str(v)
    return out


# 推移を追うのに要る最小限（docs/PATTERN.md §3.2）。**全件ぶん残す**
WATCH_TRACE_COLS = ["ticker", "pattern", "breakout_pct", "watch_streak"]


def _watch_payload(pending: pd.DataFrame) -> tuple[list, list]:
    """スナップショットに書く監視（docs/PATTERN.md §3.2）。**2 つに分ける。**

    - `watch`: **全件**を最小限の列で（銘柄・パターン・抜け幅・連続日数）。
      §3.2 が言う「その日の監視銘柄と breakout_pct」で、連続日数もここから数える
    - `watch_cards`: **カードに載せる上位 `WATCH_MAX` 件だけ**を全列で。
      配信は記録に書いてある値だけで描く（§4.3）ので、カードに出す値はここに要る

    分ける理由は**リポジトリを膨らませないため**。全件を全列で書くと 196 件で
    約 235KB/日、年 60MB 近くになる（日次でコミットするファイルは日付別にしてあるが、
    それでも積み上がる。CLAUDE.md の落とし穴）。カードに要るのは上位 10 件だけである。
    """
    if not len(pending):
        return [], []
    full = [c for c in pattern_record.PATTERN_DELIVERED_COLS
            if c not in ("delivered_on", "asof")] + ["watch_streak", "to_neck_pct"]
    trace = [_json_row(r, WATCH_TRACE_COLS) for _i, r in pending.iterrows()]
    cards = [_json_row(r, full)
             for _i, r in pending.head(render_context.WATCH_MAX).iterrows()]
    return trace, cards


def _save_pattern_records(cfg: Settings, out: pd.DataFrame, done: pd.DataFrame,
                          pending: pd.DataFrame, n_eval: int, log=print) -> None:
    """成立を台帳に、監視をスナップショットに書く（docs/PATTERN.md §3）。

    判定日 T はファイル名に入れる。**同じ配信日に引け前と引け後の 2 回走っても
    両方残る**（2026-09-03 の記録消失と同じ形を作らない）。
    """
    cfg.ensure_dirs()
    delivered_on = record.as_calendar_date(_now())
    asof = (record.as_calendar_date(out["asof"].max()) if len(out)
            else delivered_on)

    # 監視の連続日数（§6.2）。**過去のスナップショットから数える** ——
    # 監視は台帳に無いので、台帳からは数えられない
    keys = list(zip(pending["ticker"].astype(str), pending["pattern"].astype(str)))         if len(pending) else []
    streaks = pattern_record.watch_streaks(cfg.daily_dir, keys, delivered_on)
    if len(pending):
        pending = pending.assign(watch_streak=[
            streaks.get((str(t), str(p)), 1)
            for t, p in zip(pending["ticker"], pending["pattern"])])

    watch_trace, watch_cards = _watch_payload(pending)
    ledger = pattern_record.build_delivered(done, delivered_on, asof)
    path, written = record.save_delivered(ledger, cfg.daily_dir, delivered_on, asof)
    log(f"[pattern] 成立 {len(ledger)}件を {path.name} に"
        + ("保存" if written else "保存しなかった（既存ファイルを残した）"))

    summary = {
        "delivered_on": f"{delivered_on:%Y-%m-%d}",
        "asof": f"{asof:%Y-%m-%d}",
        "n_evaluated": int(n_eval),
        "n_done": int(len(done)),
        "n_watch": int(len(pending)),
        "counts": _pattern_counts(done, pending),
        "delivered_written": bool(written),
        # **監視の全件**を残す（1枚目に載せるのは上位 10 件だけだが、推移は全件で追う）。
        # 記録用の列をそのまま入れる —— 配信は「記録に書いてある値だけ」で描くので
        # （§4.3）、ここに無い値はカードに出せない。銘柄と抜け幅だけでは足りない
        "watch": watch_trace,
        "watch_cards": watch_cards,
    }
    spath = pattern_record.save_pattern_summary(summary, cfg.daily_dir, delivered_on, asof)
    log(f"[pattern] 監視 {len(pending)}件を {spath.name} に保存（記録ではなく観測値）")


def step_resolve(cfg: Settings, log=print) -> list[Path]:
    """配信記録（daily/delivered_<配信日>_asof<判定日>.csv）に 5 営業日後の結果を付ける
    （docs/SCREENER.md §3.3）。

    結果が既にあるファイルと、5 営業日がまだ経過していないファイルには触らない。
    配信記録が 1 件も無ければ何もしない（スクリーナー本体が未配信の間はこれが通常）。
    DRYRUN では data-dryrun/ 側の記録だけを見る（config.data_dir が分かれている）。
    """
    cfg.ensure_dirs()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    written = resolver.resolve_pending(cfg.daily_dir, ohlcv, log=log)
    log(f"[resolve] 結果を付けたファイル {len(written)} 件")
    return written


def step_notify(cfg: Settings, log=print) -> dict:
    """その日のパターン配信を LINE に流す（docs/PATTERN.md §6）。

    読むのは `daily/delivered_*.csv`（成立）と `daily/pattern_summary_*.json`
    （要約と監視）だけで、**株価も指標も計算し直さない**。配信内容と台帳が食い違わない
    ようにするため（SCREENER.md §4.3 と同じ方針）。

    **成立 0 件の日も配信する**（§6.1）。成立は 1 日 0〜13 件で振れるので、0 件の日に
    何も送らないと「動いているのか壊れているのか」が分からない。監視のほうが実用的
    でもある —— 抜けた日に買うなら前日に形を知っておくほうが早い。

    **画像 2 枚だけを送る。テキストは送らない**（§4.5）。Worker の `/upload` は
    caption を付けると画像とは別にテキストを 1 通 push する（`src/worker.js`）ので、
    caption は付けない。描画または送信に失敗した日だけテキストに落とし、本文の先頭に
    失敗した旨を入れる。

    WORKER_URL が無い環境（ローカル・DRYRUN）では本文を作って log に出すだけで、
    送信はしない。
    """
    cfg.ensure_dirs()
    delivered_on = record.as_calendar_date(_now())
    found = pattern_record.latest_pattern_summary(cfg.daily_dir, delivered_on)
    if found is None:
        # その日の検出がまだ走っていない。空の配信を送らない
        log(f"[notify] {delivered_on:%Y-%m-%d} のパターン要約が無いため配信しない"
            "（先に cli pattern を実行）")
        return {"sent": False, "status": None, "reason": "要約が無い"}

    summary = json.loads(found.path.read_text(encoding="utf-8"))
    log(f"[notify] {found.path.name} を読む")
    delivered = None
    if found.asof is not None:
        path = record.delivered_path(cfg.daily_dir, delivered_on, found.asof)
        if path.exists():
            delivered = pattern_record.load_pattern_delivered(path)
    # カードに載せるのは上位 `WATCH_MAX` 件（全件は "watch" に最小限の列で入っている）
    watch = pd.DataFrame(summary.get("watch_cards") or [])
    n_done = 0 if delivered is None else len(delivered)
    log(f"[notify] 成立 {n_done}件 / 監視 {len(watch)}件")

    images = []
    failure = ""
    try:
        images = render_images_mod.render_images(
            delivered, watch, summary, cfg.data_dir / "render",
            stem=f"pattern_{delivered_on:%Y-%m-%d}")
        log(f"[notify] 画像 {len(images)}枚を作成: {[p.name for p in images]}")
    except Exception as e:   # Chromium 無し・フォント無し・起動失敗のいずれでも落とさない
        failure = f"{type(e).__name__}: {e}"
        log(f"[notify] 画像の作成に失敗（テキストに切り替える）: {failure}")

    if images:
        # caption は付けない。付けると Worker がテキストも 1 通 push する（worker.js）
        results = []
        for i, img in enumerate(images):
            res = line_send.push_image(img)
            log(f"[notify] 画像{i + 1}: {res['reason']}")
            results.append(res)
        if all(r["sent"] for r in results):
            return {"sent": True, "status": 200,
                    "reason": f"画像{len(results)}枚を送信", "mode": "image"}
        if any(r["status"] is None for r in results):
            # WORKER_URL 未設定（ローカル・DRYRUN）。失敗ではないのでテキストに落とさない
            return {"sent": False, "status": None,
                    "reason": "WORKER_URL 未設定のため送信しない", "mode": "image"}
        failure = "画像の送信に失敗"
        log(f"[notify] {failure}したためテキストに切り替える")

    text = message.build_message(delivered, watch, summary, fallback=bool(failure))
    log(f"[notify] テキスト {len(text)}文字")
    for line in text.splitlines():
        log(f"[notify]   {line}")
    result = line_send.push_text(text)
    log(f"[notify] {result['reason']}")
    return {**result, "mode": "text"}


# ------------------------------------------------------ バックテスト（BACKTEST.md）
PATTERN_WINDOWS = {
    "search": pattern_replay.SEARCH_WINDOW,      # 2021-08-01〜2026-01-30
    "confirm": pattern_replay.CONFIRM_WINDOW,    # 2017-03-15〜2021-07-31
    "holdout": pattern_replay.HOLDOUT_WINDOW,    # 2026-02-01〜2026-08-01
}


def step_pattern_replay(cfg: Settings, window: str, include_holdout: bool,
                        log=print) -> None:
    """パターン検出を過去に再生する（docs/BACKTEST.md §2）。

    **ホールドアウトは `--include-holdout` を明示しない限り走らない**
    （CLAUDE.md の絶対規則）。確認窓を通過するまで触らない（BACKTEST.md §5）。

    出力は `data/pattern_replay/<窓>/` に日別。中断再開できる。
    """
    if window == "holdout" and not include_holdout:
        log("[pattern-replay] ホールドアウトは --include-holdout を明示したときだけ走る"
            "（CLAUDE.md の絶対規則。確認窓を通過するまで触らない）")
        return
    start, end = PATTERN_WINDOWS[window]
    out_dir = cfg.data_dir / "pattern_replay" / window
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    idx_df = ohlcv.get(IDX_TICKER)
    if idx_df is None or len(idx_df) == 0:
        log("[pattern-replay] 指数データが無いためスキップ")
        return
    listed = _load_listed_cached(cfg, log)
    sectors = dict(zip(listed["ticker"].astype(str),
                       listed.get("sector33", pd.Series("", index=listed.index))))
    log(f"[pattern-replay] 窓={window} {start.date()}〜{end.date()} → {out_dir}")
    pattern_replay.run(ohlcv, idx_df, listed, out_dir, start, end, cfg.k,
                       min_adv_jpy=cfg.min_adv_jpy, min_price=cfg.min_price,
                       sectors=sectors, min_history_bars=cfg.min_history_bars,
                       include_holdout=include_holdout, log=log)


def step_pattern_exit(cfg: Settings, window: str, include_holdout: bool,
                      log=print) -> None:
    """ATR 基準の出口を当てて集計する（docs/BACKTEST.md §10）。

    **検出はやり直さない。** 保存済みの再生結果（`cli pattern-replay` の出力）に
    T+1..T+20 の四本値を当てるだけ。

    出すものは 2 つ。**A は記述統計（検定なし）、B が検定 1 件**（倍率 2.0 のみ）。

    **ホールドアウトは `--include-holdout` を明示したときだけ**（CLAUDE.md の絶対
    規則）。フラグが無ければ四本値を物理的に打ち切る —— 探索窓の末尾の T は
    T+20 でホールドアウト側のバーに届く。
    """
    if window == "holdout" and not include_holdout:
        log("[pattern-exit] ホールドアウトは --include-holdout を明示したときだけ走る"
            "（CLAUDE.md の絶対規則）")
        return
    base = cfg.data_dir / "pattern_replay"
    replay = pattern_replay.load_table(base / window)
    cov = pattern_report.coverage(replay)
    log(f"[pattern-exit] 窓={window} 成立 {cov['n_rows']}件 / 評価日 {cov['n_days']}日")
    if cov["n_rows"] == 0:
        log("[pattern-exit] 行が無い（先に cli pattern-replay を実行）")
        return
    log(f"[pattern-exit] カバーした期間: {cov['first'].date()}〜{cov['last'].date()}")

    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    if not include_holdout:
        ohlcv = pattern_exit.truncate_before_holdout(ohlcv)
    table = pattern_exit.run(replay, ohlcv, log=log)
    if len(table) == 0:
        log("[pattern-exit] 出口を当てられる行が無い")
        return
    out_dir = cfg.data_dir / "pattern_exit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = pattern_exit.exit_path(out_dir, window)
    table.to_csv(path, index=False, compression="gzip")
    log(f"[pattern-exit] 書き出し {len(table)}行 → {path}")

    log(f"[pattern-exit] 検定 {pattern_exit.N_TESTS_TOTAL} 件"
        f"（既存 {pattern_report.N_TESTS} 件 + ATR×{pattern_exit.ATR_MULT:.1f} の 1 件）。"
        "A の記述統計は検定に数えない。これ以上増やさない")
    log("[pattern-exit] --- A. atr_t ÷ 終値[T] の分布（検定なし・判定に使わない）---")
    for line in pattern_exit.format_ratio_stats(
            window, pattern_exit.atr_ratio_stats(table)):
        log(f"[pattern-exit] {line}")

    n_cens = int(table["censored"].astype(bool).sum())
    n_below = int(table["entry_below_stop"].fillna(False).astype(bool).sum())
    n_above = int(table["entry_above_target"].fillna(False).astype(bool).sum())
    n_already = int(table["already_at_target"].fillna(False).astype(bool).sum())
    log(f"[pattern-exit] 母数 {len(table)}件 / 評価窓が 20 本に満たない行 {n_cens}件 "
        f"/ 寄り付きが既に撤退ライン割れ {n_below}件 "
        f"/ 寄り付きが既に測定目標超え {n_above}件（うち T 時点で既に到達 {n_already}件）")

    rows = [pattern_exit.summarize_exit(
        f"ATR×{pattern_exit.ATR_MULT:.1f}（検定11）", table, "atr2")]
    rows.append(pattern_exit.summarize_exit("参考: 測定目標（全行）", table, "tgt"))
    not_already = table[~table["already_at_target"].fillna(False).astype(bool)]
    rows.append(pattern_exit.summarize_exit(
        "参考: 測定目標（既に到達を除く）", not_already, "tgt"))
    log("[pattern-exit] --- B. 出口別の成績 ---")
    for line in pattern_exit.format_exit_table(
            pd.DataFrame(rows)[pattern_exit.EXIT_GROUP_COLS]):
        log(f"[pattern-exit] {line}")
    log("[pattern-exit] **判定対象は ATR×2.0 の行だけ（検定1件）。** 参考の 2 行は"
        "診断で、判定には使わない（BACKTEST.md §10）")
    log("[pattern-exit] 平均損益は**ベンチマークを引いていない素のリターン**である"
        "（r20 とは別物）。約定は指定価格ちょうどで、手数料もスリッページも見ていない")


def step_pattern_report(cfg: Settings, window: str, log=print) -> None:
    """探索の集計を出す（docs/BACKTEST.md §4・§6）。**表のみ。解釈はしない。**

    分位の境界は**探索窓から作り、他の窓にはそれをそのまま当てる**（D-3）。
    """
    base = cfg.data_dir / "pattern_replay"
    df = pattern_replay.load_table(base / window)
    search = pattern_replay.load_table(base / "search")
    cov = pattern_report.coverage(df)
    log(f"[pattern-report] 窓={window} 成立 {cov['n_rows']}件 / 評価日 {cov['n_days']}日")
    if cov["first"] is not None:
        # **窓の定義を黙って縮めない。** 実際に評価できた期間をそのまま出す
        log(f"[pattern-report] カバーした期間: {cov['first'].date()}〜{cov['last'].date()}")
    if cov["n_rows"] == 0:
        log("[pattern-report] 行が無い（先に cli pattern-replay を実行）")
        return

    log(f"[pattern-report] 検定 {pattern_report.N_TESTS} 件"
        "（パターン別5 + 抜け幅5分位）。これ以上増やさない")
    log("[pattern-report] --- 全体 ---")
    for line in pattern_report.format_table(
            pd.DataFrame([pattern_report.overall(df)])[pattern_report.GROUP_COLS]):
        log(f"[pattern-report] {line}")
    log("[pattern-report] --- パターン別（検定1〜5）---")
    for line in pattern_report.format_table(pattern_report.by_pattern(df)):
        log(f"[pattern-report] {line}")

    # **境界は探索窓のものを固定して使う**（BACKTEST.md D-3）。窓ごとに切り直すと
    # 分位の意味が変わって再現を見たことにならない。確定値をリポジトリに置いてある
    edges = pattern_report.load_frozen_edges()
    src = f"固定値 {pattern_report.FROZEN_EDGES_PATH}"
    if edges is None:
        if window != "search":
            # **黙って切り直さない。** 切り直すと D-3 違反になる
            log(f"[pattern-report] 分位の境界（{pattern_report.FROZEN_EDGES_PATH}）が"
                "無いので抜け幅の表は出さない。確認窓・ホールドアウトは探索窓の境界を"
                "使う決まり（BACKTEST.md D-3）")
            return
        edges = pattern_report.quantile_edges(search if len(search) else df)
        src = "この窓（固定値が無いので新規に作成）"
    if edges is None:
        log("[pattern-report] 抜け幅の分位を作れない（件数不足）")
        return
    log(f"[pattern-report] --- 抜け幅 b の5分位（検定6〜10・境界は{src}）---")
    for line in pattern_report.format_table(
            pattern_report.by_breakout_quantile(df, edges)):
        log(f"[pattern-report] {line}")
    if window != "search":
        # **判定対象は Q1 だけ**（設計責任者・2026-09-12）。探索で落ちた群を確認窓で
        # 再度判定すると検定が増える。他の 9 群は集計するが判定には使わない
        log("[pattern-report] **判定対象は Q1（b <= 0.100）のみ。** 他の9群は集計だけで"
            "判定に使わない（探索で落ちた群を再判定すると検定が増える）")
    log("[pattern-report] 主指標は平均r20とNW t。勝率・成功率・目標到達は診断で、"
        "判定には使わない（BACKTEST.md D-2）")


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stockbot")
    ap.add_argument("command", choices=["daily", "listed", "fetch", "index", "backfill",
                                       "references", "universe", "features", "pattern",
                                       "resolve", "notify", "refetch-recent-splits",
                                       "pattern-replay", "pattern-report", "pattern-exit"])
    ap.add_argument("--window", choices=list(PATTERN_WINDOWS), default="search",
                    help="バックテストの窓（docs/BACKTEST.md §1）")
    # **ホールドアウトはこれを明示したときだけ**（CLAUDE.md の絶対規則）
    ap.add_argument("--include-holdout", action="store_true",
                    help="ホールドアウトを生成する（確認窓を通過してから・1回のみ）")
    args = ap.parse_args(argv)
    cfg = Settings.from_env()
    log = print
    log(f"[cfg] data_dir={cfg.data_dir} dryrun={cfg.dryrun} scope={cfg.fetch_scope} "
        f"history_days={cfg.history_days} min_adv={cfg.min_adv_jpy:.0f}")
    try:
        if args.command == "listed":
            step_listed(cfg, log)
        elif args.command == "fetch":
            listed = _load_listed_cached(cfg, log)
            step_fetch(cfg, listed, log)
        elif args.command == "index":
            step_index(cfg, log)
        elif args.command == "backfill":
            step_backfill(cfg, log)
        elif args.command == "refetch-recent-splits":
            step_refetch_recent_splits(cfg, log)
        elif args.command == "references":
            step_references(cfg, log)
        elif args.command == "pattern":
            store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
            ohlcv = from_long(store.load())
            u = load_latest_universe(cfg.universe_dir)
            if u is None:
                log("[pattern] ユニバースが無いためスキップ（先に universe を実行）")
            else:
                step_pattern(cfg, u, ohlcv, log)
        elif args.command == "pattern-replay":
            step_pattern_replay(cfg, args.window, args.include_holdout, log)
        elif args.command == "pattern-report":
            step_pattern_report(cfg, args.window, log)
        elif args.command == "pattern-exit":
            step_pattern_exit(cfg, args.window, args.include_holdout, log)
        elif args.command == "resolve":
            step_resolve(cfg, log)
        elif args.command == "notify":
            step_notify(cfg, log)
        elif args.command == "universe":
            listed = _load_listed_cached(cfg, log)
            step_universe(cfg, listed, log=log)
        elif args.command == "features":
            store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
            ohlcv = from_long(store.load())
            u = load_latest_universe(cfg.universe_dir)
            if u is None:
                log("[features] ユニバースが無いためスキップ（先に universe を実行）")
            else:
                step_features(cfg, u, ohlcv, log)
        else:  # daily
            refresh = cfg.dryrun or _now().weekday() == 0 or not (cfg.reference_dir / "listed_latest.csv").exists()
            listed = step_listed(cfg, log) if refresh else _load_listed_cached(cfg, log)
            ohlcv, meta, issues = step_fetch(cfg, listed, log)
            step_index(cfg, log)
            u = step_universe(cfg, listed, ohlcv, issues, log)
            step_features(cfg, u, ohlcv, log)
        return 0
    except Exception as e:  # 失敗は赤にする（握り潰さない）
        import traceback
        traceback.print_exc()
        log(f"[error] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
