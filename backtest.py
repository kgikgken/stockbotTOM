"""stockbotTOM v7.1 バックテスト — DCS/カバレッジと将来リターンの関係。

仕様書のLevel 2〜4に相当する、これまで未着手だった検証。DCS(次元スコアの
平均値)やカバレッジ(充足次元の数)が、実際に将来リターンと関係するかを
複数の過去時点を横断して見る。

【方法】
過去数年分のOHLCVを一度だけ取得し、約40の評価時点それぞれで価格系列を
その時点まで切り詰めてrun_pipeline()に通す(=その時点で運用していたら
何が採点されたかを再現する)。エントリーは評価日の翌営業日(t+1)終値
(先読み排除)。保有5/10/20営業日で終値-終値リターンを測り、同日・同母集団
の等ウェイト平均(ベンチマーク)を引いた超過リターンで比較する
(地合いを実力と誤認しないため)。

【限界(必ず読むこと)】
- 生存バイアス未処理: 現在の上場銘柄リストを過去にも適用するため、
  上場廃止銘柄が結果から漏れる。結果は実力より良く出る。
- ポイントインタイム・ユニバース不可: 同じ理由で、過去のその時点で
  実際に存在した銘柄集合を復元できない。
- 単一日付内の銘柄リターンは強く相関するため、40日×N銘柄を独立標本
  として扱えない。Fama-MacBeth方式(日付ごとに横断統計量を出し、その
  時系列を検定する)を使う。実質の観測数は日付の数≈40。
- L0レジームのヒステリシス(prev_state)は連続シミュレーションしていない
  (評価時点が疎なため)。本番の連続運用と評価が完全には一致しない日が
  ありうる。
- 「効いた」は上方バイアスの影響を受けた弱い証拠。「効かなかった」は
  上方バイアスがかかってなお出た結果なので相対的に強い証拠。

使い方:
    python backtest.py                            # 本番: yfinance・5年・約40時点
    python backtest.py --dryrun                    # 合成データでの動作確認
    python backtest.py --n-dates 20 --universe-cap 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]

from v7.config import load_config_v7
from v7.pipeline import run_pipeline
from v7.universe import load_universe
from v7.data import fetch_ohlcv

HOLD_DAYS = [5, 10, 20]
PRIMARY_HOLD = 10
# 評価日より前に必要な最低営業日数。④の要件(SMA200ウォームアップ199+
# 測定上限250=449)に余裕を持たせる。
LOOKBACK_BUFFER = 500
FORWARD_BUFFER = max(HOLD_DAYS) + 1   # +1はt+1エントリーの分
N_DATES_DEFAULT = 40
BACKTEST_HISTORY_DAYS = 1825          # 暦日5年


# ============================================================
# 評価時点の選定・価格系列の切り詰め
# ============================================================

def select_eval_dates(all_dates: pd.DatetimeIndex, n_dates: int) -> list:
    lo = LOOKBACK_BUFFER
    hi = len(all_dates) - FORWARD_BUFFER
    if hi <= lo:
        raise SystemExit(
            f"データが足りない(全{len(all_dates)}営業日、"
            f"前方{LOOKBACK_BUFFER}+後方{FORWARD_BUFFER}が必要)。"
            "--history-days を増やすか取得結果を確認すること。")
    idx = sorted(set(np.linspace(lo, hi - 1, n_dates).round().astype(int)))
    return [all_dates[i] for i in idx]


def truncate_to(ohlcv: dict, index_daily: pd.DataFrame, date: pd.Timestamp):
    ohlcv_t = {k: v.loc[:date] for k, v in ohlcv.items() if len(v.loc[:date]) > 0}
    index_t = index_daily.loc[:date]
    return ohlcv_t, index_t


def entry_position(full_index: pd.DatetimeIndex, date: pd.Timestamp):
    """dateの翌営業日(t+1)の位置。存在しなければNone。"""
    pos = full_index.searchsorted(date, side="right")
    return int(pos) if pos < len(full_index) else None


def forward_return(close: pd.Series, entry_pos: int, hold_days: int):
    exit_pos = entry_pos + hold_days
    if entry_pos < 0 or exit_pos >= len(close):
        return None
    entry = float(close.iloc[entry_pos])
    exitp = float(close.iloc[exit_pos])
    if entry <= 0:
        return None
    return exitp / entry - 1.0


# ============================================================
# 1時点の評価
# ============================================================

def evaluate_date(date: pd.Timestamp, ohlcv_full: dict, index_full: pd.DataFrame,
                  universe: pd.DataFrame, cfg):
    ohlcv_t, index_t = truncate_to(ohlcv_full, index_full, date)
    if len(index_t) == 0:
        return None
    try:
        res = run_pipeline(universe, ohlcv_t, index_t, cfg, str(date.date()))
    except Exception as e:
        print(f"    [{date.date()}] run_pipeline失敗: {type(e).__name__}: {e}")
        return None
    cands = res.get("all_scored") or []
    if not cands:
        return None  # RISK_OFFで停止、またはL1通過ゼロ

    rows = []
    for c in cands:
        full = ohlcv_full.get(c.ticker)
        if full is None or "Close" not in full.columns:
            continue
        epos = entry_position(full.index, date)
        if epos is None:
            continue
        rec = {"code": c.code, "dcs": c.dcs, "coverage": c.coverage,
               "category": c.category}
        ok = True
        for h in HOLD_DAYS:
            r = forward_return(full["Close"], epos, h)
            rec[f"ret_{h}"] = r
            if r is None:
                ok = False
        if ok:
            rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for h in HOLD_DAYS:
        bench = df[f"ret_{h}"].mean()
        df[f"exc_{h}"] = df[f"ret_{h}"] - bench
    df["date"] = date
    df["regime"] = res["regime"]["state"]
    return df


def build_panel(dates: list, ohlcv_full: dict, index_full: pd.DataFrame,
                universe: pd.DataFrame, cfg):
    frames, skipped = [], []
    for d in dates:
        r = evaluate_date(d, ohlcv_full, index_full, universe, cfg)
        print(f"  {d.date()}: {'候補ゼロ(除外)' if r is None else f'{len(r)}銘柄'}")
        if r is None:
            skipped.append(str(d.date()))
        else:
            frames.append(r)
    if not frames:
        raise SystemExit("有効な評価時点が0件。ユニバースや取得結果を確認すること。")
    panel = pd.concat(frames, ignore_index=True)
    meta = {"n_dates_total": len(dates), "n_dates_used": len(frames),
           "n_dates_skipped": len(skipped), "skipped_dates": skipped}
    return panel, meta


# ============================================================
# Fama-MacBeth集計
# ============================================================

def fama_macbeth(values) -> dict:
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": n}
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else float("nan")
    return {"mean": mean, "se": se, "t": t, "n": n}


# ============================================================
# レポート
# ============================================================

def report_category(panel: pd.DataFrame) -> str:
    lines = ["=" * 68, "① 本命 / 監視 / 除外 の超過リターン比較(Fama-MacBeth)", "=" * 68]
    for h in HOLD_DAYS:
        lines.append(f"\n--- 保有{h}営業日 ---")
        lines.append(f"{'区分':8}{'平均超過':>10}{'SE':>9}{'t値':>8}{'日数':>6}")
        stats = {}
        for cat in ["本命", "監視", "除外"]:
            daily = panel[panel["category"] == cat].groupby("date")[f"exc_{h}"].mean()
            st = fama_macbeth(daily.values)
            stats[cat] = st
            lines.append(f"{cat:8}{st['mean']*100:>9.2f}%{st['se']*100:>8.2f}%"
                        f"{st['t']:>8.2f}{st['n']:>6d}")
        if not (np.isnan(stats["本命"]["mean"]) or np.isnan(stats["除外"]["mean"])):
            diff = stats["本命"]["mean"] - stats["除外"]["mean"]
            lines.append(f"  本命−除外 差: {diff*100:+.2f}%")
    return "\n".join(lines)


def report_decile(panel: pd.DataFrame):
    h = PRIMARY_HOLD
    lines = ["\n" + "=" * 68, f"② DCS十分位の単調性(Fama-MacBeth、保有{h}営業日)", "=" * 68]
    decile_daily = {i: [] for i in range(1, 11)}
    for _, g in panel.groupby("date"):
        if len(g) < 10:
            continue
        try:
            dec = pd.qcut(g["dcs"], 10, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        for i in range(1, int(dec.max()) + 1):
            decile_daily.setdefault(i, [])
            decile_daily[i].append(g.loc[dec == i, f"exc_{h}"].mean())
    lines.append(f"{'十分位':8}{'平均超過':>10}{'t値':>8}{'日数':>6}")
    means = {}
    for i in range(1, 11):
        st = fama_macbeth(decile_daily.get(i, []))
        means[i] = st["mean"]
        lines.append(f"D{i:<7}{st['mean']*100:>9.2f}%{st['t']:>8.2f}{st['n']:>6d}")
    valid = [means[i] for i in range(1, 11) if not np.isnan(means[i])]
    monotonic = all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1)) if len(valid) > 1 else False
    lines.append(f"\n単調増加(D1≦D2≦…≦D10): {monotonic}")
    lines.append("(高DCSほど超過リターンが高いはず、という設計の直接検証)")
    return "\n".join(lines), means


def report_coverage_vs_dcs(panel: pd.DataFrame) -> str:
    lines = ["\n" + "=" * 68, "③ カバレッジ vs DCS — 将来リターンへの予測力比較(本丸)", "=" * 68]
    lines.append("「手法数ではなく次元数を数える」という設計思想の直接検証。")
    lines.append("次元の『数』(カバレッジ)と『量の平均』(DCS)、どちらが超過")
    lines.append("リターンとより強く相関するかを、日ごとの断面相関をFama-MacBeth")
    lines.append("集計して比べる。\n")
    for h in HOLD_DAYS:
        rho_dcs, rho_cov = [], []
        for _, g in panel.groupby("date"):
            if len(g) < 10:
                continue
            rho_dcs.append(g["dcs"].corr(g[f"exc_{h}"], method="spearman"))
            rho_cov.append(g["coverage"].corr(g[f"exc_{h}"], method="spearman"))
        st_dcs = fama_macbeth(rho_dcs)
        st_cov = fama_macbeth(rho_cov)
        lines.append(f"--- 保有{h}営業日 ---")
        lines.append(f"{'':10}{'平均ρ':>9}{'SE':>8}{'t値':>8}{'日数':>6}")
        lines.append(f"{'DCS':10}{st_dcs['mean']:>9.3f}{st_dcs['se']:>8.3f}"
                    f"{st_dcs['t']:>8.2f}{st_dcs['n']:>6d}")
        lines.append(f"{'カバレッジ':9}{st_cov['mean']:>9.3f}{st_cov['se']:>8.3f}"
                    f"{st_cov['t']:>8.2f}{st_cov['n']:>6d}")
        if not (np.isnan(st_dcs["mean"]) or np.isnan(st_cov["mean"])):
            winner = "カバレッジ" if abs(st_cov["mean"]) > abs(st_dcs["mean"]) else "DCS"
            lines.append(f"  → 平均|ρ|が大きいのは: {winner}")
    return "\n".join(lines)


def plot_report(panel: pd.DataFrame, decile_means: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    xs = list(range(1, 11))
    ys = [decile_means.get(i, np.nan) * 100 for i in xs]
    ax.bar(xs, ys, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("DCS十分位(1=低い、10=高い)")
    ax.set_ylabel(f"平均超過リターン(%, 保有{PRIMARY_HOLD}日)")
    ax.set_title("DCS十分位別の超過リターン")
    ax.set_xticks(xs)

    ax = axes[1]
    h = PRIMARY_HOLD
    cats = ["本命", "監視", "除外"]
    vals = []
    for cat in cats:
        daily = panel[panel["category"] == cat].groupby("date")[f"exc_{h}"].mean()
        st = fama_macbeth(daily.values)
        vals.append(st["mean"] * 100)
    ax.bar(cats, vals, color=["#55A868", "#DD8452", "#C44E52"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"平均超過リターン(%, 保有{PRIMARY_HOLD}日)")
    ax.set_title("区分別の超過リターン")

    fig.suptitle("stockbotTOM v7.1 バックテスト", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"図を保存: {out_path}")


# ============================================================
# 合成データ(DRYRUN用: 本番main_v7.pyより長期の系列が要る)
# ============================================================

def _bars_needed(history_days: int, n_dates: int) -> int:
    trading = int(history_days * 5 / 7) + 10
    return max(trading, LOOKBACK_BUFFER + n_dates + FORWARD_BUFFER)


def _synthetic_long(tickers: list, n_bars: int):
    from v7.data import _mk_v6
    modes = ["vcp", "pullback", "pivot", "choppy", "downtrend"]
    out = {t: _mk_v6(seed=900 + i, n=n_bars, mode=modes[i % len(modes)])
          for i, t in enumerate(tickers)}
    idx = next(iter(out.values())).index
    p = np.linspace(2000, 2900, len(idx))
    index_full = pd.DataFrame({"Open": p, "High": p * 1.01, "Low": p * 0.99,
                               "Close": p, "Volume": np.full(len(idx), 1e9)},
                              index=idx)
    return out, index_full


def _fetch_index_full(history_days: int):
    import yfinance as yf
    for sym in ("^TPX", "^N225"):
        try:
            h = yf.Ticker(sym).history(period=f"{history_days}d")
            if len(h) >= LOOKBACK_BUFFER:
                if getattr(h.index, "tz", None) is not None:
                    h.index = h.index.tz_localize(None)
                print(f"  指数: {sym} ({len(h)}行)")
                return h[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return None


# ============================================================
# main
# ============================================================

def main(dryrun: bool, n_dates: int, universe_cap, history_days: int, out_dir: str) -> None:
    cfg = load_config_v7()
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    universe = load_universe("universe_jpx.csv")
    if universe_cap:
        universe = universe.head(universe_cap)
    print(f"ユニバース {len(universe)}銘柄 / dryrun={dryrun}")

    if dryrun:
        n_bars = _bars_needed(history_days, n_dates)
        ohlcv_full, index_full = _synthetic_long(universe["ticker"].tolist(), n_bars)
        print(f"  合成データ {n_bars}本")
    else:
        print("OHLCV取得中(1回・全期間)...")
        ohlcv_full, meta = fetch_ohlcv(universe["ticker"].tolist(), history_days,
                                       dryrun=False, deadline_sec=6000)
        print(f"  {meta.get('data_ok', len(ohlcv_full))}/{meta.get('data_total', len(universe))}")
        index_full = _fetch_index_full(history_days)
        if index_full is None:
            raise SystemExit("指数が取得できず、レジーム判定不能。中止。")

    all_dates = index_full.index
    dates = select_eval_dates(all_dates, n_dates)
    print(f"\n評価時点 {len(dates)}件: {dates[0].date()} 〜 {dates[-1].date()}")
    print("各時点を採点中...")
    panel, meta2 = build_panel(dates, ohlcv_full, index_full, universe, cfg)
    print(f"\n有効時点 {meta2['n_dates_used']}/{meta2['n_dates_total']}"
          f"(候補ゼロで除外: {meta2['n_dates_skipped']}件)")

    report = [
        "v7.1 バックテスト — DCS/カバレッジと将来リターン",
        f"評価時点: {meta2['n_dates_used']}件(候補ゼロで除外{meta2['n_dates_skipped']}件)",
        "上方バイアスあり(生存バイアス・ポイントインタイム不可、詳細はスクリプト冒頭の"
        "docstring参照)。「効いた」は弱い証拠、「効かなかった」は強い証拠として読むこと。",
        report_category(panel),
    ]
    dec_text, dec_means = report_decile(panel)
    report.append(dec_text)
    report.append(report_coverage_vs_dcs(panel))
    text = "\n".join(report)
    print("\n" + text)

    (outdir / "backtest_report.txt").write_text(text, encoding="utf-8")
    panel.to_csv(outdir / "backtest_panel.csv", index=False, encoding="utf-8-sig")
    plot_report(panel, dec_means, str(outdir / "backtest_result.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--n-dates", type=int, default=N_DATES_DEFAULT)
    ap.add_argument("--universe-cap", type=int, default=None)
    ap.add_argument("--history-days", type=int, default=BACKTEST_HISTORY_DAYS)
    ap.add_argument("--out-dir", type=str, default="out_v7")
    args = ap.parse_args()
    main(args.dryrun, args.n_dates, args.universe_cap, args.history_days, args.out_dir)
