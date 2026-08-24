"""CLI。GitHub Actions とローカルの両方から同じ手順で呼ぶ。

  python -m stockbot.cli daily       # 取得 → 整合性検査 → 保存・スナップショット → ユニバース
  python -m stockbot.cli listed      # JPX 上場銘柄一覧の更新のみ
  python -m stockbot.cli fetch       # 取得と保存のみ
  python -m stockbot.cli index       # 指数（TOPIX/日経225）の取得と保存のみ
  python -m stockbot.cli backfill    # 検証用の長期履歴取得（HISTORY_DAYS=2600 等）。中断再開可
  python -m stockbot.cli references  # 決算発表予定日・上場廃止銘柄一覧の更新のみ
  python -m stockbot.cli universe    # 保存済みデータからユニバースを再計算
  python -m stockbot.cli features    # 保存済みデータから日次特徴量を再計算・保存

環境変数: SPEC/README 参照。SCREEN_DRYRUN=1 で合成データ・ネットワーク不要。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import Settings
from .data.adjust import check_all
from .data.jpx_lists import (
    fetch_delistings,
    fetch_earnings_schedule,
    load_earnings_schedule,
    load_listed_with_fallback,
    normalize_listed,
)
from .data.store import IDX_TICKER, OhlcvStore, from_long, to_long
from .data.synthetic import make_synthetic, make_synthetic_index, synthetic_listed
from .data.yf_fetch import fetch_index, fetch_ohlcv
from .pipeline import (
    DAILY_FEATURES_COLS,
    compute_daily_features,
    load_recent_daily_features,
    save_daily_features,
)
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
    log(f"[fetch] ok {meta['data_ok']}/{meta['data_total']} / 履歴不足 {len(meta['short'])} / "
        f"失敗 {len(meta['failed'])} / 分割issue {len(issues)}")

    store = OhlcvStore(cfg.store_dir, cfg.daily_dir, cfg.rev_close_tol, cfg.rev_volume_tol)
    merged, added, revisions = store.upsert(to_long(ohlcv))
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


INDEX_LABELS = {"^TPX": "TOPIX", "^N225": "日経225"}


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
    meta = {"ticker": used, "label": label, "is_fallback": (not cfg.dryrun) and used != "^TPX"}
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


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stockbot")
    ap.add_argument("command", choices=["daily", "listed", "fetch", "index", "backfill",
                                       "references", "universe", "features"])
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
        elif args.command == "references":
            step_references(cfg, log)
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
