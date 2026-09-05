"""CLI。GitHub Actions とローカルの両方から同じ手順で呼ぶ。

  python -m stockbot.cli daily       # 取得 → 整合性検査 → 保存・スナップショット → ユニバース
  python -m stockbot.cli listed      # JPX 上場銘柄一覧の更新のみ
  python -m stockbot.cli fetch       # 取得と保存のみ
  python -m stockbot.cli index       # 指数（TOPIX/日経225）の取得と保存のみ
  python -m stockbot.cli backfill    # 検証用の長期履歴取得（HISTORY_DAYS=2600 等）。中断再開可
  python -m stockbot.cli references  # 決算発表予定日・上場廃止銘柄一覧の更新のみ
  python -m stockbot.cli universe    # 保存済みデータからユニバースを再計算
  python -m stockbot.cli features    # 保存済みデータから日次特徴量を再計算・保存
  python -m stockbot.cli screen      # 19条件で候補を選び、配信記録に保存（docs/SCREENER.md §2）
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
from .features import indicators, pullback, regime, sector as sector_mod, swings
from .pipeline import (
    DAILY_FEATURES_COLS,
    compute_daily_features,
    load_recent_daily_features,
    save_daily_features,
)
from .notify import line_send, message
from .render import render as render_images_mod
from .screener import record, resolver, screen
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


def _sector_extra(info: dict | None) -> dict:
    """業種の強弱を配信記録の列に移す（§2.9）。順位表に無い業種は欠損のまま。"""
    info = info or {}
    return {
        "sector_rank_5d": info.get("rank_5d") if info.get("rank_5d") is not None else pd.NA,
        "sector_rank_20d": info.get("rank_20d") if info.get("rank_20d") is not None else pd.NA,
        "sector_ret_5d": info.get("ret_5d", np.nan) if info.get("ret_5d") is not None else np.nan,
        "sector_ret_20d": (info.get("ret_20d", np.nan)
                           if info.get("ret_20d") is not None else np.nan),
    }


def step_screen(cfg: Settings, universe: pd.DataFrame, ohlcv: dict, log=print) -> pd.DataFrame:
    """19 条件で候補を選び、配信記録に保存する（docs/SCREENER.md §2）。

    ユニバース通過銘柄を母集団として A〜D・E2・E3 を銘柄ごとに判定し、その通過集合に
    E1（当日候補内で rs60 の上位10%を落とす）を掛け、売買代金の降順・同一33業種3件までに
    絞る。順位は付けない。結果を daily/delivered_<配信日>_asof<判定日>.csv に書く（§3.2）。

    配信記録は「その日に何を出したか」の台帳なので、同じ配信日・同じ判定日で 2 回
    実行しても最初の記録が正（save_delivered が上書きしない）。判定日が違えば
    ファイル名が違うので、引け前と引け後の実行は別の記録として両方残る（§3.2）。
    """
    cfg.ensure_dirs()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    idx_df = from_long(store.load()).get(IDX_TICKER)
    if idx_df is None or len(idx_df) == 0:
        log("[screen] 指数データが無いためスキップ（E1 の rs60 が計算できない）")
        return pd.DataFrame(columns=record.DELIVERED_COLS)

    earnings_schedule = None
    p = cfg.reference_dir / "earnings_schedule.csv"
    if p.exists():
        earnings_schedule = load_earnings_schedule(p)
    else:
        log("[screen] 決算発表予定日が無い → A4 は全銘柄で「決算日未取得」扱い")

    tickers = universe[universe["passes"]]["ticker"].tolist()
    evaluated = screen.evaluate_universe(ohlcv, tickers, idx_df["Close"], cfg.k,
                                         earnings_schedule=earnings_schedule, log=log)
    evaluated, meta = screen.apply_e1(evaluated, log=log)

    # 判定日 T。ファイル名にも入るので、記録を書く前に確定させる（§3.2）。
    # 評価が 0 件の日は銘柄側から取れないので、営業日軸（指数）の最終日で代用する
    asof = record.as_calendar_date(evaluated["date"].max()) if len(evaluated) \
        else record.as_calendar_date(idx_df.index[-1])
    log(f"[screen] 判定日 {asof:%Y-%m-%d}")

    # 地合いゲージ（DESIGN.md §8.1）。条件にも順位にも使わない。配信の見出しに出すだけ
    idx_close = idx_df["Close"]
    asof_idx = idx_close.index[-1]
    breadth_75, breadth_200, _n = regime.compute_breadth(
        {t: ohlcv[t] for t in tickers if t in ohlcv and ohlcv[t] is not None}, asof_idx)
    gauge = regime.regime_gauge(idx_close, len(idx_close) - 1, breadth_75, breadth_200)
    log(f"[screen] 地合い={gauge['level']}({gauge['score']}/6)")
    sector_by_ticker = dict(zip(universe["ticker"], universe["sector33"].fillna("")))
    # 33業種の強弱（§2.9）。並び順にだけ使う。条件にも除外にも使わない
    strength = sector_mod.sector_strength(ohlcv, tickers, sector_by_ticker, asof)
    sector_rank = sector_mod.rank_lookup(strength)
    if len(strength):
        top = " ".join(f"{r['sector33']}:{r['ret_5d'] * 100:+.1f}%"
                       for _i, r in strength.head(3).iterrows())
        log(f"[screen] 業種強弱 {len(strength)}業種（5日・等加重）。上位: {top}")
    else:
        log("[screen] 業種強弱が計算できない（5日ぶんの履歴が無い）→ 並びは売買代金のみ")
    candidates = screen.select_candidates(evaluated, sector_by_ticker,
                                          sector_rank=sector_rank)
    log(f"[screen] 候補 {len(candidates)} 件（業種の5日順位→売買代金降順・"
        f"同一33業種は{screen.SECTOR_CAP}件まで。優劣ではない）")
    log(f"[screen] 候補の止まった線: "
        f"{screen.format_counts(screen.landing_ma_breakdown(candidates), sort=False)}")

    name_by_ticker = dict(zip(universe["ticker"], universe["name"].fillna("")))
    # 配信日は「JST のその日」というカレンダー日であって時刻ではない。tz を落として
    # 過去の配信記録（ファイル名由来で tz なし）と比較できるようにする
    delivered_on = record.as_calendar_date(_now())
    # 連続点灯日数と前回点灯日は、過去の配信記録から記録時点で確定させる（§3.2）
    streaks, prev_seen = record.lookback_stats(
        cfg.daily_dir, [str(t) for t in candidates["ticker"]], delivered_on)
    rows = []
    for _i, cand in candidates.iterrows():
        ticker = str(cand["ticker"])
        df = ohlcv[ticker]
        high, low, close = df["High"], df["Low"], df["Close"]
        t_pos = len(df) - 1
        alt = swings.alternate_swings(swings.detect_raw_swings(high, low, cfg.k))
        pb = pullback.pullback_state(high, low, close, indicators.sma(close, 5),
                                     indicators.sma(close, 200),
                                     indicators.atr_wilder(high, low, close, 14),
                                     alt, t_pos, cfg.k)
        rows.append(record.build_record(
            ticker, high, low, close, pb, t_pos, delivered_on,
            name=str(name_by_ticker.get(ticker, "")),
            extra={"adv_jpy": float(cand["adv_jpy"]), "sector33": str(cand["sector33"]),
                   "a4_earnings_unknown": bool(cand["a4_earnings_unknown"]),
                   "e1_skipped": meta["e1_skipped"],
                   "earnings_days": float(cand["earnings_days"]),
                   "streak": int(streaks.get(ticker, 1)),
                   "prev_delivered_on": prev_seen.get(ticker) or pd.NaT,
                   **_sector_extra(sector_rank.get(str(cand["sector33"])))},
        ))
    delivered = record.records_to_frame(rows)
    path, written = record.save_delivered(delivered, cfg.daily_dir, delivered_on, asof)
    if written:
        delivered_n = len(delivered)
        log(f"[screen] 配信記録 {delivered_n} 件を {path.name} に保存"
            + ("（E1 スキップ日）" if meta["e1_skipped"] else ""))
    else:
        # 台帳は上書きしない。**「保存した」と書かない** —— 以前はメモリ上の件数を
        # そのままログに出していたため、書けていないことが誰にも見えなかった（§3.2）
        existing = record.load_delivered(path)
        delivered_n = len(existing)
        log(f"[screen] 警告: {path.name} は既にある。今回の判定 {len(delivered)}件は"
            f"保存していない（ファイルにある {delivered_n}件が正）")
    repeats = [f"{r['ticker']}:{int(r['streak'])}日目" for _i, r in delivered.iterrows()
               if int(r["streak"]) > 1]
    if repeats:
        log(f"[screen] 連続点灯: {' '.join(repeats)}")

    # Actions のログは 90 日で消える。E1 のスキップ率や条件別の不成立件数は
    # 数週間かけて見るものなので、リポジトリ側にも残す（docs/SCREENER.md §3.6）
    # 取得成功率は fetch_meta.json（step_fetch が書く）から。描画側は screen_summary と
    # delivered しか読まないので、必要な値はここで要約に畳んでおく（docs/SCREENER.md §4.5）
    fetch_meta = {}
    fm = cfg.store_dir / "fetch_meta.json"
    if fm.exists():
        try:
            fetch_meta = json.loads(fm.read_text())
        except (OSError, ValueError):
            fetch_meta = {}
    summary = screen.build_summary(evaluated, candidates, meta, asof, delivered_on, gauge,
                                   fetch_meta=fetch_meta, delivered_written=written,
                                   delivered_n=delivered_n,
                                   sector_ranking=sector_mod.ranking_table(strength))
    screen.save_summary(summary, cfg.daily_dir, delivered_on, asof)
    # D-1 の観測日数。判定日 1 つにつき 1 日で数える（§8 D-3）。判定はまだしない
    days = screen.observation_days(cfg.daily_dir)
    log(f"[screen] 観測 {len(days)}日目（判定日ユニーク） / "
        f"E1 スキップ {sum(1 for d in days if d.e1_skipped)}日")
    # 候補の偏りは日次で残す（§3.6）。集計するだけで条件にも並び順にも使わない
    if summary["sector_candidates"]:
        adv = summary["adv_candidates"]
        log(f"[screen] 候補の業種内訳: "
            f"{screen.format_counts(summary['sector_candidates'], sort=False)}")
        log(f"[screen] 候補の売買代金: 最大 {adv['max'] / 1e8:,.1f}億 / "
            f"中央 {adv['median'] / 1e8:,.1f}億 / 最小 {adv['min'] / 1e8:,.1f}億")
    return delivered


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
    """その日の配信を LINE に流す（docs/SCREENER.md §4）。

    読むのは `daily/delivered_*.csv` と `daily/screen_summary_*.json` だけで、株価も
    指標も計算し直さない。配信内容と台帳が食い違わないようにするため（§4.3）。
    候補0件の日も配信する（§4.2）。同じ配信日に判定が 2 つある日は、判定日が新しい方
    （引け後）を流す。

    **通常は画像カード2枚だけを送る。テキストは送らない**（§4.5）。Worker の
    `/upload` は caption を付けると画像とは別にテキストを 1 通 push する
    （`src/worker.js`）ので、caption は付けない。描画または送信に失敗した日だけ
    テキストに落とし、本文の先頭に失敗した旨を入れる（§4.4）。

    WORKER_URL が無い環境（ローカル・DRYRUN）では本文を作って log に出すだけで、
    送信はしない。
    """
    cfg.ensure_dirs()
    delivered_on = record.as_calendar_date(_now())
    found = screen.latest_summary(cfg.daily_dir, delivered_on)
    if found is None:
        log(f"[notify] {delivered_on:%Y-%m-%d} の要約が無いためスキップ（先に screen を実行）")
        return {"sent": False, "status": None, "reason": "要約が無い"}

    summary = json.loads(found.path.read_text(encoding="utf-8"))
    log(f"[notify] {found.path.name} を読む")
    delivered = None
    if found.asof is not None:
        path = record.delivered_path(cfg.daily_dir, delivered_on, found.asof)
        delivered = record.load_delivered(path) if path.exists() else None
    n = 0 if delivered is None else len(delivered)

    images = []
    failure = ""
    try:
        images = render_images_mod.render_images(
            delivered, summary, cfg.data_dir / "render", stem=f"screen_{delivered_on:%Y-%m-%d}")
        log(f"[notify] 候補 {n}件 / 画像 {len(images)}枚を作成: {[p.name for p in images]}")
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

    text = message.build_message(delivered, summary, fallback=bool(failure))
    log(f"[notify] テキスト {len(text)}文字")
    for line in text.splitlines():
        log(f"[notify]   {line}")
    result = line_send.push_text(text)
    log(f"[notify] {result['reason']}")
    return {**result, "mode": "text"}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stockbot")
    ap.add_argument("command", choices=["daily", "listed", "fetch", "index", "backfill",
                                       "references", "universe", "features", "screen",
                                       "resolve", "notify", "refetch-recent-splits"])
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
        elif args.command == "screen":
            store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
            ohlcv = from_long(store.load())
            u = load_latest_universe(cfg.universe_dir)
            if u is None:
                log("[screen] ユニバースが無いためスキップ（先に universe を実行）")
            else:
                step_screen(cfg, u, ohlcv, log)
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
