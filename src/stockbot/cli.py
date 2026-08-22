"""CLI。GitHub Actions とローカルの両方から同じ手順で呼ぶ。

  python -m stockbot.cli daily       # 取得 → 整合性検査 → 保存・スナップショット → ユニバース
  python -m stockbot.cli listed      # JPX 上場銘柄一覧の更新のみ
  python -m stockbot.cli fetch       # 取得と保存のみ
  python -m stockbot.cli universe    # 保存済みデータからユニバースを再計算

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
from .data.jpx_lists import load_listed_with_fallback, normalize_listed
from .data.store import OhlcvStore, from_long, to_long
from .data.synthetic import make_synthetic, synthetic_listed
from .data.yf_fetch import fetch_ohlcv
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


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stockbot")
    ap.add_argument("command", choices=["daily", "listed", "fetch", "universe"])
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
        elif args.command == "universe":
            listed = _load_listed_cached(cfg, log)
            step_universe(cfg, listed, log=log)
        else:  # daily
            refresh = cfg.dryrun or _now().weekday() == 0 or not (cfg.reference_dir / "listed_latest.csv").exists()
            listed = step_listed(cfg, log) if refresh else _load_listed_cached(cfg, log)
            ohlcv, meta, issues = step_fetch(cfg, listed, log)
            step_universe(cfg, listed, ohlcv, issues, log)
        return 0
    except Exception as e:  # 失敗は赤にする（握り潰さない）
        import traceback
        traceback.print_exc()
        log(f"[error] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
