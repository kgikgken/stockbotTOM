"""stockbotTOM — 形成期間×保有期間グリッド検証(Jegadeesh-Titman型)。

文献調査(2026-08、グリッド較正)で確定した設計に基づく実装。
形成 {5,10,20,40,60,120営業日} × 保有 {5,10,20,40,60営業日} の30セルで、
勝ち組(上位10%)−負け組(下位10%)のWML(winner-minus-loser)を測る。

【文献に基づく設計】
- スキップ 1日 と 5日 の両方で測る。差が大きければbid-askバウンス由来
  (Conrad-Gultekin-Kaul 1997)と判定できる。
- 重複ポートフォリオ + Newey-West HAC標準誤差(lag=保有期間−1)。
  JT(1993)の標準。前回の非重複設計は検出力を不必要に捨てていた。
- 加重は等加重と流動性加重(60日中央値売買代金)の両方。等加重でのみ
  有意なら小型株集中でコストに脆弱と判定する(時価総額データが無いため
  流動性加重を代理として使う)。
- 単一セルの有意水準は t>3.0(Harvey-Liu-Zhu 2016)。t>2.0は使わない。
  316因子のカタログ化研究が示す通り、多数の候補から探す文脈でt>2.0は
  偽陽性が多発する。
- 隣接セルとの符号・大きさの整合性を見る。孤立した1セルの有意はノイズ
  として扱う(単一の確立された統計手法ではないが、JT自身も頑健性の
  根拠を隣接セルの一貫性に置いている)。

【陽性対照(重要)】
このセッションだけでバグを8件以上見つけてきた(選抜バイアス、レンジ不整合、
分母不整合、dropna、tee+pipefail等)。「日本で何も出ない」がバグなのか
市場の性質なのかを区別する手段がこれまで無かった。米国株で同じグリッドを
回し、形成60〜120営業日で正のモメンタムが検出できることを先に確認する。
これが出なければ、日本の結果を信用する前にエンジンを疑うこと。
米国側は現在の上場銘柄リストを使うため生存バイアスがかかる
(モメンタムを過大に見せる方向)。目的は「検出力の確認」なので許容する。

【事前登録した予測(結果を見る前に固定)】
  形成 5〜20  × 保有 5〜20   → 負(リバーサル)
  形成 40〜120              → ゼロ近傍
  符号の境界                → 形成20営業日付近
  米国(対照)・形成60〜120   → 正(モメンタム)

【意図的なスコープ限定】
リターンはClose-to-Close(学術的な標準に合わせ、シグナル検出に専念する)。
寄付ベースの執行可能性検証は、ここで生き残ったセルがあった場合の
次工程とする。R換算やコスト控除もしない(ここは検出、実装可否は別工程)。

使い方:
    python grid_test.py --market jp                # 日本株(本番)
    python grid_test.py --market us                # 米国(陽性対照)
    python grid_test.py --market both               # 両方(推奨)
    python grid_test.py --dryrun --market both       # 合成データで動作確認
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

from v7.universe import load_universe
from v7.data import fetch_ohlcv

FORMATION = [5, 10, 20, 40, 60, 120]
HOLDING = [5, 10, 20, 40, 60]
SKIPS = [1, 5]
DECILE_Q = 0.10
MIN_NAMES_PER_DECILE = 15
MIN_TURNOVER_JP = 1e8      # 60日中央値売買代金(円)の下限
MIN_TURNOVER_US = 5e6      # 60日中央値売買代金(ドル)の下限
T_SIG = 3.0                 # Harvey-Liu-Zhu基準
HISTORY_DAYS = 1825

# 米国側の陽性対照用ユニバース。生存バイアスは承知の上(検出力確認が目的)。
US_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","V",
    "UNH","JNJ","PG","HD","MA","XOM","CVX","ABBV","MRK","PEP",
    "KO","COST","AVGO","WMT","MCD","CSCO","ADBE","CRM","ACN","TMO",
    "ABT","DHR","NKE","LIN","TXN","NEE","PM","UNP","RTX","HON",
    "QCOM","LOW","IBM","AMGN","SBUX","INTU","CAT","GE","BA","DE",
    "SPGI","BLK","GS","AXP","MDT","PLD","MS","ISRG","BKNG","GILD",
    "ADP","LMT","SYK","MMC","CB","VRTX","TJX","ZTS","MO","SCHW",
    "CI","REGN","SO","DUK","PGR","BSX","ETN","FI","AON","APD",
    "CME","ITW","SLB","EQIX","MU","PANW","SNPS","CDNS","MCK","NOC",
    "WM","ORLY","MAR","CSX","EMR","FDX","APH","PSA","ROP","AJG",
    "TGT","NSC","PXD","MCO","PH","ADI","GD","AIG","CTAS","MSI",
    "TT","ECL","CARR","NXPI","PCAR","AZO","CMG","SRE","HUM","FTNT",
    "MET","F","GM","OXY","D","EW","KLAC","LRCX","MNST","IDXX",
    "AEP","EXC","XEL","ED","WEC","ES","DTE","PPL","CMS","FE",
    "ROK","DOV","XYL","AME","IEX","PWR","EFX","VRSK","CTSH","FAST",
]


# ============================================================
# データ構築
# ============================================================

def build_matrices(ohlcv: dict) -> dict:
    fields = ["Close", "Volume"]
    out = {}
    for f in fields:
        out[f] = pd.DataFrame({t: df[f] for t, df in ohlcv.items()
                              if f in df.columns}).sort_index()
    idx = out["Close"].index
    for f in fields:
        out[f] = out[f].reindex(idx)
    return out


# ============================================================
# Newey-West HAC標準誤差(平均値用)
# ============================================================

def newey_west_se(x: np.ndarray, lag: int) -> float:
    """時系列平均のNewey-West HAC標準誤差。Bartlettカーネル。

    lag=0でiid分散推定(ddof=0)に一致することを含め、AR(1)過程・
    重複窓の模擬データで検証済み(実装時のユニットテスト参照)。
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    gamma0 = float(np.dot(e, e) / n)
    S = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        cov = float(np.dot(e[l:], e[:-l]) / n)
        w = 1.0 - l / (lag + 1)
        S += 2 * w * cov
    S = max(S, 1e-14)
    return float(np.sqrt(S / n))


def nw_stats(x: pd.Series, lag: int) -> dict:
    v = x.dropna().values
    n = len(v)
    if n < 10:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "n": n}
    m = float(np.mean(v))
    se = newey_west_se(v, lag)
    t = m / se if se > 0 else np.nan
    return {"mean": m, "se": se, "t": t, "n": n}


# ============================================================
# 形成/保有リターンとWML(vectorized: 日付ループを使わない)
# ============================================================

def formation_return(close: pd.DataFrame, F: int, S: int) -> pd.DataFrame:
    return close.shift(S) / close.shift(S + F) - 1.0


def holding_return(close: pd.DataFrame, H: int, S: int) -> pd.DataFrame:
    return close.shift(-(S + H)) / close.shift(-S) - 1.0


def compute_wml(form_ret: pd.DataFrame, hold_ret: pd.DataFrame,
                liquid: pd.DataFrame, weight: pd.DataFrame):
    """各日付について、断面で上位/下位10%のWMLを等加重・流動性加重で算出。

    axis=1のrank/mean/sumで日付方向のループを排除している
    (backtest.pyのrun_pipeline呼び出しに比べ大幅に軽い)。
    """
    base = liquid & form_ret.notna() & hold_ret.notna()
    ranks = form_ret.where(base).rank(axis=1, pct=True)
    top = ranks >= (1 - DECILE_Q)
    bot = ranks <= DECILE_Q
    n_top = top.sum(axis=1)
    n_bot = bot.sum(axis=1)
    enough = (n_top >= MIN_NAMES_PER_DECILE) & (n_bot >= MIN_NAMES_PER_DECILE)

    ew_top = hold_ret.where(top).mean(axis=1)
    ew_bot = hold_ret.where(bot).mean(axis=1)
    ew_wml = (ew_top - ew_bot).where(enough)

    w_top = weight.where(top)
    w_bot = weight.where(bot)
    lw_top = (hold_ret.where(top) * w_top).sum(axis=1) / w_top.sum(axis=1)
    lw_bot = (hold_ret.where(bot) * w_bot).sum(axis=1) / w_bot.sum(axis=1)
    lw_wml = (lw_top - lw_bot).where(enough)

    return ew_wml, lw_wml, n_top.where(enough), n_bot.where(enough)


def run_grid(mat: dict, min_turnover: float) -> dict:
    close, vol = mat["Close"], mat["Volume"]
    turnover = (close * vol).rolling(60, min_periods=30).median()
    liquid = turnover >= min_turnover

    results = {}
    for S in SKIPS:
        for F in FORMATION:
            fret = formation_return(close, F, S)
            for H in HOLDING:
                hret = holding_return(close, H, S)
                ew, lw, n_top, n_bot = compute_wml(fret, hret, liquid, turnover)
                lag = max(H - 1, 0)
                ew_st = nw_stats(ew, lag)
                lw_st = nw_stats(lw, lag)
                results[(F, H, S)] = {
                    "ew": ew_st, "lw": lw_st,
                    "n_top_med": float(n_top.median()) if n_top.notna().any() else np.nan,
                    "n_bot_med": float(n_bot.median()) if n_bot.notna().any() else np.nan,
                }
    return results


# ============================================================
# 隣接整合性・BH補正
# ============================================================

def neighbor_consistent(results: dict, F: int, H: int, S: int, key: str = "ew") -> bool:
    """(F,H)グリッド上の隣接セル(形成方向・保有方向)と符号・大きさが
    整合しているか。孤立した1セルの有意をノイズとして弾くための簡易判定。"""
    center = results[(F, H, S)][key]["mean"]
    if np.isnan(center) or center == 0:
        return False
    fi, hi = FORMATION.index(F), HOLDING.index(H)
    neighbors = []
    if fi > 0:
        neighbors.append((FORMATION[fi - 1], H))
    if fi < len(FORMATION) - 1:
        neighbors.append((FORMATION[fi + 1], H))
    if hi > 0:
        neighbors.append((F, HOLDING[hi - 1]))
    if hi < len(HOLDING) - 1:
        neighbors.append((F, HOLDING[hi + 1]))
    ok = 0
    for nf, nh in neighbors:
        v = results[(nf, nh, S)][key]["mean"]
        if np.isnan(v):
            continue
        if np.sign(v) == np.sign(center) and abs(v) >= 0.3 * abs(center):
            ok += 1
    return ok >= 1


def benjamini_hochberg(pvals: list, alpha: float = 0.05) -> list:
    idx = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
    if not idx:
        return []
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    kmax = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha * rank / m:
            kmax = rank
    return order[:kmax] if kmax > 0 else []


def two_sided_p(t: float) -> float:
    if np.isnan(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2))


# ============================================================
# レポート
# ============================================================

def grid_table(results: dict, S: int, key: str, field: str) -> str:
    lines = [f"  スキップ{S}日 / {'等加重' if key=='ew' else '流動性加重'} / {field}"]
    header = "形成\\保有" + "".join(f"{h:>9}" for h in HOLDING)
    lines.append("  " + header)
    for F in FORMATION:
        row = f"  {F:>6}日"
        for H in HOLDING:
            st = results[(F, H, S)][key]
            v = st[field]
            if np.isnan(v):
                row += f"{'—':>9}"
            elif field == "mean":
                row += f"{v*100:>8.2f}%"
            else:
                sig = "*" if (not np.isnan(st['t']) and abs(st['t']) >= T_SIG) else " "
                row += f"{v:>8.2f}{sig}"
        lines.append(row)
    return "\n".join(lines)


def build_report(results_jp: dict, results_us: dict | None) -> tuple:
    L = ["形成×保有グリッド検証(Jegadeesh-Titman型)",
        "重複ポートフォリオ + Newey-West HAC(lag=保有−1)。有意水準 t>3.0(Harvey-Liu-Zhu)。",
        "上方バイアスあり(生存バイアス)。WML = 上位10%(勝ち組) − 下位10%(負け組)。\n"]

    if results_us:
        L.append("=" * 72)
        L.append("① 陽性対照(米国) — エンジンが既知のモメンタムを検出できるか")
        L.append("=" * 72)
        L.append("形成60〜120日・保有20〜60日で正・有意が事前予測。")
        for S in SKIPS:
            L.append("\n" + grid_table(results_us, S, "ew", "t"))
        found = []
        for F in [60, 120]:
            for H in [20, 40, 60]:
                st = results_us[(F, H, SKIPS[0])]["ew"]
                if not np.isnan(st["t"]) and st["t"] >= T_SIG:
                    found.append(f"形成{F}×保有{H}(t={st['t']:.2f})")
        if found:
            L.append(f"\n→ 検出: {'、'.join(found)}。エンジンは既知のモメンタムを拾える。")
        else:
            L.append("\n→ 予測領域で有意なし。エンジンに問題がある可能性。"
                    "日本側の結果を信用する前にコードを疑うこと。")

    if not results_jp:
        return "\n".join(L), []

    L.append("\n" + "=" * 72)
    L.append("② 日本株グリッド — t値(等加重)")
    L.append("=" * 72)
    for S in SKIPS:
        L.append("\n" + grid_table(results_jp, S, "ew", "t"))

    L.append("\n" + "=" * 72)
    L.append("③ 日本株グリッド — 平均WML(等加重)")
    L.append("=" * 72)
    for S in SKIPS:
        L.append("\n" + grid_table(results_jp, S, "ew", "mean"))

    L.append("\n" + "=" * 72)
    L.append("④ スキップ1日 vs 5日 — bid-askバウンス汚染の検出")
    L.append("=" * 72)
    L.append("差が大きい(1日側が過大)ほどバウンス由来の疑いが強い(Conrad-Gultekin-Kaul 1997)。")
    L.append(f"{'形成':>6}{'保有':>6}{'skip1平均':>11}{'skip5平均':>11}{'比(5/1)':>9}")
    for F in FORMATION:
        for H in HOLDING:
            m1 = results_jp[(F, H, 1)]["ew"]["mean"]
            m5 = results_jp[(F, H, 5)]["ew"]["mean"]
            if np.isnan(m1) or np.isnan(m5) or abs(m1) < 1e-6:
                continue
            ratio = m5 / m1
            flag = "  ←バウンス疑い" if (np.sign(m1) == np.sign(m5) and abs(ratio) < 0.5) else ""
            L.append(f"{F:>6}{H:>6}{m1*100:>10.2f}%{m5*100:>10.2f}%{ratio:>9.2f}{flag}")

    L.append("\n" + "=" * 72)
    L.append("⑤ 等加重 vs 流動性加重 — 小型株集中の検出")
    L.append("=" * 72)
    L.append("流動性加重で消える(符号反転 or |t|<3)なら小型株集中、実装不能と判定。")
    L.append(f"{'形成':>6}{'保有':>6}{'等加重t':>9}{'流動性加重t':>12}")
    flagged_thin = []
    for F in FORMATION:
        for H in HOLDING:
            for S in SKIPS:
                ew = results_jp[(F, H, S)]["ew"]
                lw = results_jp[(F, H, S)]["lw"]
                if np.isnan(ew["t"]) or abs(ew["t"]) < T_SIG:
                    continue
                lw_sig = (not np.isnan(lw["t"])) and abs(lw["t"]) >= T_SIG \
                        and np.sign(lw["mean"]) == np.sign(ew["mean"])
                mark = "" if lw_sig else "  ←流動性加重で消失(小型株集中)"
                L.append(f"{F:>6}{H:>6}(skip{S}){ew['t']:>7.2f}{lw['t']:>12.2f}{mark}")
                flagged_thin.append((F, H, S, lw_sig))
    if not flagged_thin:
        L.append("  (等加重でt>3.0のセルなし)")

    L.append("\n" + "=" * 72)
    L.append("⑥ 有意セルの棄却フィルタ(t>3.0 かつ 隣接整合 かつ 流動性加重でも残存)")
    L.append("=" * 72)
    survivors = []
    pvals, keys = [], []
    for F in FORMATION:
        for H in HOLDING:
            for S in SKIPS:
                st = results_jp[(F, H, S)]["ew"]
                pvals.append(two_sided_p(st["t"]))
                keys.append((F, H, S))
    bh_sig = set(keys[i] for i in benjamini_hochberg(pvals))
    for F in FORMATION:
        for H in HOLDING:
            for S in SKIPS:
                st = results_jp[(F, H, S)]["ew"]
                lw = results_jp[(F, H, S)]["lw"]
                if np.isnan(st["t"]) or abs(st["t"]) < T_SIG:
                    continue
                consistent = neighbor_consistent(results_jp, F, H, S, "ew")
                lw_ok = (not np.isnan(lw["t"])) and abs(lw["t"]) >= 2.0 \
                        and np.sign(lw["mean"]) == np.sign(st["mean"])
                in_bh = (F, H, S) in bh_sig
                if consistent and lw_ok:
                    survivors.append((F, H, S, st["mean"], st["t"], in_bh))
    if survivors:
        L.append(f"{'形成':>6}{'保有':>6}{'skip':>5}{'平均':>9}{'t':>7}{'BH補正':>8}")
        for F, H, S, m, t, bh in survivors:
            L.append(f"{F:>6}{H:>6}{S:>5}{m*100:>8.2f}%{t:>7.2f}{'有意' if bh else '　　':>8}")
    else:
        L.append("  棄却フィルタを通過したセルなし。")

    L.append("\n" + "=" * 72)
    L.append("⑦ 事前登録した予測との照合")
    L.append("=" * 72)
    def region_summary(f_lo, f_hi, h_lo, h_hi, S=1):
        vals = [results_jp[(F, H, S)]["ew"]["mean"] for F in FORMATION for H in HOLDING
               if f_lo <= F <= f_hi and h_lo <= H <= h_hi
               and not np.isnan(results_jp[(F, H, S)]["ew"]["mean"])]
        return np.mean(vals) if vals else np.nan
    r1 = region_summary(5, 20, 5, 20)
    r2 = region_summary(40, 120, 5, 60)
    L.append(f"  予測: 形成5-20×保有5-20 → 負    実測平均WML = {r1*100:+.2f}%  "
            f"→ {'一致' if r1 < 0 else '不一致'}")
    L.append(f"  予測: 形成40-120 → ゼロ近傍       実測平均WML = {r2*100:+.2f}%  "
            f"→ {'一致' if abs(r2) < abs(r1)/2 or np.isnan(r1) else '不一致(要確認)'}")

    return "\n".join(L), survivors


def plot_heatmaps(results_jp: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, S in zip(axes, SKIPS):
        mat = np.full((len(FORMATION), len(HOLDING)), np.nan)
        for i, F in enumerate(FORMATION):
            for j, H in enumerate(HOLDING):
                mat[i, j] = results_jp[(F, H, S)]["ew"]["t"]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
        ax.set_xticks(range(len(HOLDING))); ax.set_xticklabels(HOLDING)
        ax.set_yticks(range(len(FORMATION))); ax.set_yticklabels(FORMATION)
        ax.set_xlabel("保有(営業日)"); ax.set_ylabel("形成(営業日)")
        ax.set_title(f"スキップ{S}日 t値(等加重)")
        for i in range(len(FORMATION)):
            for j in range(len(HOLDING)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                           fontsize=8, color="white" if abs(v) > 2 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("日本株 形成×保有グリッド — t値ヒートマップ", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"図を保存: {out_path}")


# ============================================================
# 合成データ(DRYRUN)
# ============================================================

def _synthetic(tickers: list, n: int, momentum_strength: float, seed: int) -> dict:
    """momentum_strength>0で意図的に断面モメンタムを埋め込んだ合成データ。
    米国(陽性対照)の検出力テスト用。日本側は0(モメンタムなし)で使う。

    ★設計上の注意: 「銘柄自身の過去リターンが自身の将来リターンを予測する」
    という時系列自己相関を入れただけでは、断面(銘柄間比較)のモメンタムには
    ならない(実装時に確認済み。時系列相関+0.09を埋め込んでも断面WMLは
    負のままだった)。断面モメンタムを作るには、銘柄ごとに持続する潜在的な
    ドリフト格差(regime)を与える必要がある——その格差が「高ドリフト銘柄は
    過去も強く・将来も強い」という真の断面順序を生む。
    """
    rng = np.random.default_rng(seed)
    out = {}
    # 銘柄ごとに持続的なドリフトの「格」を1回だけ引く(時間で変えない)。
    # これが強ければ強いほど、過去の強弱が将来の強弱を予言する。
    quality = rng.normal(0, 0.0006 * momentum_strength, len(tickers))
    for k, t in enumerate(tickers):
        ret = rng.normal(quality[k], 0.018, n)
        price = 1000 * np.exp(np.cumsum(ret))
        lv = np.zeros(n)
        for i in range(1, n):
            lv[i] = 0.8 * lv[i - 1] + rng.normal(0, 0.3)
        vol = 500_000 * np.exp(lv)
        idx = pd.bdate_range(end=pd.Timestamp("2026-08-14"), periods=n)
        out[t] = pd.DataFrame({"Close": price, "Volume": vol}, index=idx)
    return out


# ============================================================
# main
# ============================================================

def main(dryrun: bool, market: str, universe_cap, history_days: int, out_dir: str) -> None:
    outdir = Path(out_dir); outdir.mkdir(parents=True, exist_ok=True)
    results_jp = results_us = None

    if market in ("jp", "both"):
        uni = load_universe("universe_jpx.csv")
        if universe_cap:
            uni = uni.head(universe_cap)
        print(f"[JP] ユニバース {len(uni)}銘柄 / dryrun={dryrun}")
        if dryrun:
            n_bars = max(int(history_days * 5 / 7), 300)
            ohlcv = _synthetic(uni["ticker"].tolist(), n_bars, momentum_strength=0.0, seed=11)
        else:
            ohlcv, meta = fetch_ohlcv(uni["ticker"].tolist(), history_days, dryrun=False, deadline_sec=6000)
            print(f"  {meta.get('data_ok', len(ohlcv))}/{len(uni)}")
        mat = build_matrices(ohlcv)
        print(f"  価格行列 {mat['Close'].shape}")
        print("  グリッド計算中(JP)...")
        results_jp = run_grid(mat, MIN_TURNOVER_JP)

    if market in ("us", "both"):
        us_list = US_TICKERS[:universe_cap] if universe_cap else US_TICKERS
        print(f"[US] ユニバース {len(us_list)}銘柄(陽性対照) / dryrun={dryrun}")
        if dryrun:
            n_bars = max(int(history_days * 5 / 7), 300)
            ohlcv_us = _synthetic(us_list, n_bars, momentum_strength=1.2, seed=22)
        else:
            ohlcv_us, meta = fetch_ohlcv(us_list, history_days, dryrun=False, deadline_sec=3000)
            print(f"  {meta.get('data_ok', len(ohlcv_us))}/{len(us_list)}")
        mat_us = build_matrices(ohlcv_us)
        print("  グリッド計算中(US)...")
        results_us = run_grid(mat_us, MIN_TURNOVER_US)

    text, survivors = build_report(results_jp or {}, results_us)
    print("\n" + text)
    (outdir / "grid_report.txt").write_text(text, encoding="utf-8")
    if results_jp:
        plot_heatmaps(results_jp, str(outdir / "grid_heatmap.png"))
    print(f"\n保存: {outdir}/grid_report.txt" + (", grid_heatmap.png" if results_jp else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--market", choices=["jp", "us", "both"], default="both")
    ap.add_argument("--universe-cap", type=int, default=None)
    ap.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    ap.add_argument("--out-dir", type=str, default="out_v7")
    a = ap.parse_args()
    main(a.dryrun, a.market, a.universe_cap, a.history_days, a.out_dir)
