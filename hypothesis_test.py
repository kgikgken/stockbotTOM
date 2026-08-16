"""stockbotTOM — 仮説検証バックテスト H1 / H2 / H4。

文献調査(2026-08)で事前登録した3仮説を、コスト後・多重検定調整付きで検証する。
v7.1のDCM(5次元)は5年37時点でIC≒0だった。日本株はモメンタム不在・
リバーサル支配という文献の総体と整合しており、検証の方向を反転させる。

【検証する仮説】
  H1 短期リバーサル × 低出来高  (Iihara et al. 2004)
     過去10日リターン最下位分位 かつ 出来高比(10日/60日) < 0.8
  H2 短期リバーサル × 高出来高  (Bremer-Hiraki 1999) — H1の対立仮説
     過去10日リターン最下位分位 かつ 出来高比(10日/60日) > 1.5
     ※H1とH2は出来高条件が排他的なので銘柄の重複は発生しない
  H4 高出来高リターンプレミアム (Gervais-Kaniel-Mingelgrin 2001)
     当日出来高が過去50日で最大

H3(大幅下落後の反発)とH5(収縮×出来高ブレイク)は事前に除外した。
H3は原著者自身が「一般投資家は経済的利益を得られない」と明記しており、
H5は調査が事前確率を低いと判断、かつv7の⑤収縮がIC−0.011で既に外れている。
検定数を減らすことでBH補正の罰を軽くする狙いもある。

【設計上の判断(結果を左右するので明示する)】
1. 非重複期間のみを使う。保有h日なら評価をh日ごとに行う。日次で評価すると
   観測が9/10重なり、t値が大幅に過大評価される。5年で保有10日なら約122期間。
2. エントリー/エグジットを Open-to-Open と Close-to-Close の両方で測る。
   リバーサルは寄付に集中しやすく、Openだけで出る効果は約定可能性が疑わしい。
   両者が食い違ったらそれ自体が重要な情報。
3. 値幅制限で約定できなかった日を除外する。日足しかないので
   High == Low(日中値幅ゼロ)を「ストップ張り付き」の代理指標とする。
4. コストは単一の推定値ではなく往復0/0.2/0.4/0.6%で並べ、どの水準で
   優位性が死ぬかを出す。楽天ゼロコースは手数料0円なので、
   支配的なのはスプレッドとスリッページ。
5. 期待R換算はエントリー時ATR20の2倍を初期リスクと仮定した近似。
   実際のRは損切り幅の定義に依存するので、あくまで代理指標。
6. 多重検定はBenjamini-Hochberg。3仮説×3保有期間=9検定を1家族として扱う。

【残る限界】
- 生存バイアス: 上場廃止銘柄を取得できない。結果は実力より良く出る。
- 配当調整なし: yfinanceをauto_adjust=Falseで取得しているため、権利落ちが
  数%の下落として混入する。3月末・9月末に集中するため、リバーサル系の
  「負け組」に季節的な偏りを生む。月別内訳を併記して影響を確認できるようにした。
- 分割調整はYahoo側で行われている前提。未検証。

使い方:
    python hypothesis_test.py                      # 本番
    python hypothesis_test.py --dryrun             # 合成データで動作確認
    python hypothesis_test.py --universe-cap 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v7.config import load_config_v7
from v7.universe import load_universe
from v7.data import fetch_ohlcv

HOLD_DAYS = [5, 10, 20]
COST_LEVELS = [0.000, 0.002, 0.004, 0.006]   # 往復
LOOKBACK_RET = 10          # H1/H2 のリターン計算期間
VOL_SHORT, VOL_LONG = 10, 60
H1_VOL_MAX = 0.8
H2_VOL_MIN = 1.5
LOSER_QUANTILE = 0.20      # 最下位5分位
H4_VOL_WINDOW = 50
MIN_TURNOVER = 1e8         # 60日中央値売買代金の下限(1億円)
MIN_NAMES = 5              # バスケット最低銘柄数
ATR_WINDOW = 20
R_STOP_MULT = 2.0          # 初期リスク = 2×ATR20 と仮定
HISTORY_DAYS = 1825


# ============================================================
# 価格行列の構築
# ============================================================

def build_matrices(ohlcv: dict) -> dict:
    """銘柄ごとのDataFrame群を、日付×銘柄の行列に変換する。"""
    fields = ["Open", "High", "Low", "Close", "Volume"]
    out = {}
    for f in fields:
        out[f] = pd.DataFrame({t: df[f] for t, df in ohlcv.items()
                              if f in df.columns}).sort_index()
    idx = out["Close"].index
    for f in fields:
        out[f] = out[f].reindex(idx)
    return out


def compute_atr(mat: dict) -> pd.DataFrame:
    """True Rangeの単純移動平均。Wilderではなく単純平均(代理指標なので十分)。"""
    h, l, c = mat["High"], mat["Low"], mat["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).stack(), (h - pc).abs().stack(), (l - pc).abs().stack()],
                   axis=1).max(axis=1).unstack()
    return tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW // 2).mean()


# ============================================================
# シグナル
# ============================================================

def signals_at(mat: dict, i: int, liquid: pd.Series):
    """評価日インデックスiにおける各仮説の選定銘柄(bool Series)を返す。"""
    c = mat["Close"]
    v = mat["Volume"]

    ret = c.iloc[i] / c.iloc[i - LOOKBACK_RET] - 1.0
    vol_s = v.iloc[i - VOL_SHORT + 1:i + 1].mean()
    vol_l = v.iloc[i - VOL_LONG + 1:i + 1].mean()
    vol_ratio = vol_s / vol_l.replace(0, np.nan)

    base = liquid & ret.notna() & vol_ratio.notna()
    if base.sum() < MIN_NAMES * 5:
        return None

    thr = ret[base].quantile(LOSER_QUANTILE)
    loser = base & (ret <= thr)

    h1 = loser & (vol_ratio < H1_VOL_MAX)
    h2 = loser & (vol_ratio > H2_VOL_MIN)

    # H4: 当日出来高が直前50日の最大を厳密に上回る。
    # 当日を含む窓のmaxと ">=" で比較すると、出来高に同値があるだけで成立して
    # しまう(合成データで86.5%が該当した)。直前窓と ">" で比較する。
    vprev = v.iloc[i - H4_VOL_WINDOW:i]
    h4 = base & (v.iloc[i] > vprev.max()) & (v.iloc[i] > 0)

    return {"H1": h1, "H2": h2, "H4": h4}


# ============================================================
# リターン
# ============================================================

def period_returns(mat: dict, atr: pd.DataFrame, i: int, hold: int, price_field: str):
    """i+1でエントリー、i+1+holdでエグジットしたときのリターン。

    価格系列を揃える(Open-to-OpenまたはClose-to-Close)ことで、
    寄引をまたぐバイアスを避ける。
    """
    entry_i, exit_i = i + 1, i + 1 + hold
    if exit_i >= len(mat["Close"]):
        return None, None, None
    p = mat[price_field]
    entry, exitp = p.iloc[entry_i], p.iloc[exit_i]
    ret = exitp / entry.replace(0, np.nan) - 1.0

    # 値幅制限で約定不能だった日を除外(日中値幅ゼロを代理指標とする)
    locked = (mat["High"].iloc[entry_i] == mat["Low"].iloc[entry_i])
    ret = ret.where(~locked)

    risk = (R_STOP_MULT * atr.iloc[i]) / entry.replace(0, np.nan)
    r_mult = ret / risk.replace(0, np.nan)
    return ret, r_mult, entry


# ============================================================
# バックテスト本体
# ============================================================

def run(mat: dict, atr: pd.DataFrame) -> pd.DataFrame:
    c, v = mat["Close"], mat["Volume"]
    turnover = (c * v).rolling(VOL_LONG, min_periods=VOL_LONG // 2).median()
    start = max(VOL_LONG, LOOKBACK_RET, H4_VOL_WINDOW, ATR_WINDOW) + 5
    rows = []

    for hold in HOLD_DAYS:
        # 非重複: hold日ごとに評価
        for i in range(start, len(c) - hold - 2, hold):
            liquid = turnover.iloc[i] >= MIN_TURNOVER
            if liquid.sum() < MIN_NAMES * 5:
                continue
            sig = signals_at(mat, i, liquid)
            if sig is None:
                continue
            for field, tag in [("Open", "OO"), ("Close", "CC")]:
                ret, r_mult, _ = period_returns(mat, atr, i, hold, field)
                if ret is None:
                    continue
                bench_pool = ret[liquid].dropna()
                if len(bench_pool) < MIN_NAMES * 5:
                    continue
                bench = bench_pool.mean()
                for name, sel in sig.items():
                    s_ret = ret[sel].dropna()
                    if len(s_ret) < MIN_NAMES:
                        continue
                    s_r = r_mult[sel].dropna()
                    rows.append({
                        "hypothesis": name, "hold": hold, "price": tag,
                        "date": c.index[i], "n_names": len(s_ret),
                        "raw": s_ret.mean(), "excess": s_ret.mean() - bench,
                        "bench": bench,
                        "r_mult": s_r.mean() if len(s_r) else np.nan,
                    })
    return pd.DataFrame(rows)


# ============================================================
# 統計
# ============================================================

def tstat(x) -> dict:
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    n = len(x)
    if n < 3:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "n": n, "p": np.nan}
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    t = m / se if se > 0 else np.nan
    # 正規近似(nが十分大きい前提。n>=30で実用上問題ない)
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2)) if not np.isnan(t) else np.nan
    return {"mean": m, "se": se, "t": t, "n": n, "p": p}


def benjamini_hochberg(pvals: list, alpha: float = 0.05) -> list:
    """BH法。有意と判定されたインデックスのリストを返す。"""
    idx = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
    if not idx:
        return []
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    sig, kmax = [], -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha * rank / m:
            kmax = rank
    if kmax > 0:
        sig = order[:kmax]
    return sig


# ============================================================
# レポート
# ============================================================

LABEL = {"H1": "H1 リバーサル×低出来高", "H2": "H2 リバーサル×高出来高",
         "H4": "H4 高出来高プレミアム"}
PRICE_LABEL = {"OO": "寄→寄", "CC": "引→引"}


def report(panel: pd.DataFrame) -> str:
    L = ["v7 仮説検証 — H1 / H2 / H4",
        "非重複期間・コスト感応度・BH補正付き",
        "上方バイアス(生存バイアス・配当調整なし)あり。「効かなかった」を強い証拠として読む。"]

    L.append("\n" + "=" * 72)
    L.append("① コストゼロでの超過リターン(ベンチマーク=同日ユニバース等加重平均)")
    L.append("=" * 72)
    pvals, keys = [], []
    for price in ["OO", "CC"]:
        L.append(f"\n--- エントリー/エグジット: {PRICE_LABEL[price]} ---")
        L.append(f"{'仮説':22}{'保有':>5}{'平均超過':>10}{'t値':>8}{'期間数':>7}{'平均銘柄':>8}")
        for hyp in ["H1", "H2", "H4"]:
            for hold in HOLD_DAYS:
                g = panel[(panel.hypothesis == hyp) & (panel.hold == hold)
                         & (panel.price == price)]
                st = tstat(g["excess"].values)
                nm = g["n_names"].mean() if len(g) else np.nan
                L.append(f"{LABEL[hyp]:22}{hold:>5}{st['mean']*100:>9.2f}%"
                        f"{st['t']:>8.2f}{st['n']:>7d}{nm:>8.0f}")
                if price == "CC":     # BH補正は引→引(保守的)を主系列とする
                    pvals.append(st["p"]); keys.append((hyp, hold))

    sig = benjamini_hochberg(pvals)
    L.append("\n" + "-" * 72)
    L.append(f"BH補正(引→引の9検定、α=0.05): "
            f"{'有意 ' + '、'.join(f'{keys[i][0]}/{keys[i][1]}日' for i in sig) if sig else '有意なものなし'}")

    L.append("\n" + "=" * 72)
    L.append("② コスト感応度 — どの水準で優位性が死ぬか(引→引)")
    L.append("=" * 72)
    L.append(f"{'仮説':22}{'保有':>5}" + "".join(f"{c*100:>9.1f}%" for c in COST_LEVELS))
    for hyp in ["H1", "H2", "H4"]:
        for hold in HOLD_DAYS:
            g = panel[(panel.hypothesis == hyp) & (panel.hold == hold)
                     & (panel.price == "CC")]
            if len(g) < 3:
                continue
            cells = []
            for cost in COST_LEVELS:
                st = tstat((g["excess"] - cost).values)
                cells.append(f"{st['mean']*100:>8.2f}%")
            L.append(f"{LABEL[hyp]:22}{hold:>5}" + "".join(cells))
    L.append("\n往復コスト。楽天ゼロコースは手数料0円なので、実質スプレッド+スリッページ。")

    L.append("\n" + "=" * 72)
    L.append("③ 期待R近似(初期リスク=2×ATR20と仮定、コストゼロ、引→引)")
    L.append("=" * 72)
    L.append(f"{'仮説':22}{'保有':>5}{'平均R':>9}{'t値':>8}{'期間数':>7}")
    for hyp in ["H1", "H2", "H4"]:
        for hold in HOLD_DAYS:
            g = panel[(panel.hypothesis == hyp) & (panel.hold == hold)
                     & (panel.price == "CC")]
            st = tstat(g["r_mult"].values)
            L.append(f"{LABEL[hyp]:22}{hold:>5}{st['mean']:>9.3f}{st['t']:>8.2f}{st['n']:>7d}")
    L.append("損切り幅の定義に依存する代理指標。実際のRは執行設計次第。")

    L.append("\n" + "=" * 72)
    L.append("④ H1 vs H2 直接対戦(文献の未解決点、引→引)")
    L.append("=" * 72)
    L.append("Iihara(低出来高で強い) と Bremer-Hiraki(高出来高で強い) は結論が逆。")
    for hold in HOLD_DAYS:
        a = panel[(panel.hypothesis == "H1") & (panel.hold == hold) & (panel.price == "CC")]
        b = panel[(panel.hypothesis == "H2") & (panel.hold == hold) & (panel.price == "CC")]
        m = pd.merge(a[["date", "excess"]], b[["date", "excess"]],
                    on="date", suffixes=("_h1", "_h2"))
        if len(m) < 3:
            L.append(f"  保有{hold}日: 共通期間不足({len(m)})")
            continue
        st = tstat((m["excess_h1"] - m["excess_h2"]).values)
        winner = "H1(低出来高)" if st["mean"] > 0 else "H2(高出来高)"
        verdict = winner if abs(st["t"]) >= 2 else "判定不能(|t|<2)"
        L.append(f"  保有{hold:>2}日: H1−H2 = {st['mean']*100:+.2f}% "
                f"(t={st['t']:+.2f}, n={st['n']}) → {verdict}")

    L.append("\n" + "=" * 72)
    L.append("⑤ 月別内訳 — 配当権利落ち(3月末/9月末)の混入確認(H1、引→引、保有10日)")
    L.append("=" * 72)
    g = panel[(panel.hypothesis == "H1") & (panel.hold == 10) & (panel.price == "CC")].copy()
    if len(g):
        g["month"] = pd.to_datetime(g["date"]).dt.month
        mm = g.groupby("month")["excess"].agg(["mean", "count"])
        L.append(f"{'月':>4}{'平均超過':>10}{'期間数':>7}")
        for m_, row in mm.iterrows():
            mark = "  ←権利落ち集中" if m_ in (3, 9) else ""
            L.append(f"{m_:>4}{row['mean']*100:>9.2f}%{int(row['count']):>7d}{mark}")
        L.append("3月・9月が突出していれば、配当調整なしの影響を疑うこと。")
    return "\n".join(L)


# ============================================================
# main
# ============================================================

def _synthetic(tickers: list, n: int, seed: int = 7) -> dict:
    """DRYRUN用の合成データ。

    v7/data.pyの_mk_v6は出来高がほぼ一定(比率0.71〜1.00)なので、
    H1(出来高比<0.8)・H2(>1.5)・H4(50日最大更新)のどれも発火せず、
    スモークテストとして機能しない。出来高に自己相関とスパイクを持たせた
    系列をここで生成する。値の妥当性は問わない(コード経路の確認が目的)。
    """
    rng = np.random.default_rng(seed)
    out = {}
    for k, t in enumerate(tickers):
        ret = rng.normal(0.0003, 0.018, n)
        # 時々トレンドとショックを入れる
        for _ in range(max(1, n // 200)):
            s = rng.integers(0, max(1, n - 40))
            ret[s:s + 30] += rng.normal(0, 0.004)
        ret[rng.integers(0, n, max(1, n // 100))] += rng.normal(0, 0.07, max(1, n // 100))
        price = 1000 * np.exp(np.cumsum(ret))
        # 出来高: 対数正規 + 自己相関 + スパイク
        lv = np.zeros(n)
        for i in range(1, n):
            lv[i] = 0.85 * lv[i - 1] + rng.normal(0, 0.35)
        vol = 1_000_000 * np.exp(lv)
        vol[rng.integers(0, n, max(1, n // 60))] *= rng.uniform(3, 10, max(1, n // 60))
        idx = pd.bdate_range(end=pd.Timestamp("2026-08-14"), periods=n)
        intr = np.abs(rng.normal(0, 0.008, n))
        out[t] = pd.DataFrame({
            "Open": price * (1 + rng.normal(0, 0.003, n)),
            "High": price * (1 + intr),
            "Low": price * (1 - intr),
            "Close": price,
            "Volume": vol,
        }, index=idx)
    return out


def main(dryrun: bool, universe_cap, history_days: int, out_dir: str) -> None:
    load_config_v7()
    outdir = Path(out_dir); outdir.mkdir(parents=True, exist_ok=True)

    uni = load_universe("universe_jpx.csv")
    if universe_cap:
        uni = uni.head(universe_cap)
    print(f"ユニバース {len(uni)}銘柄 / dryrun={dryrun}")

    if dryrun:
        n_bars = max(int(history_days * 5 / 7), 400)
        ohlcv = _synthetic(uni["ticker"].tolist(), n_bars)
        print(f"  合成データ {n_bars}本 × {len(ohlcv)}銘柄")
    else:
        print("OHLCV取得中...")
        ohlcv, meta = fetch_ohlcv(uni["ticker"].tolist(), history_days,
                                  dryrun=False, deadline_sec=6000)
        print(f"  {meta.get('data_ok', len(ohlcv))}/{len(uni)}")

    mat = build_matrices(ohlcv)
    print(f"  価格行列 {mat['Close'].shape[0]}日 × {mat['Close'].shape[1]}銘柄")
    atr = compute_atr(mat)

    print("検証中...")
    panel = run(mat, atr)
    if panel.empty:
        raise SystemExit("有効な観測が0件。流動性フィルタやデータを確認すること。")
    print(f"  観測 {len(panel)}件")

    text = report(panel)
    print("\n" + text)
    (outdir / "hypothesis_report.txt").write_text(text, encoding="utf-8")
    panel.to_csv(outdir / "hypothesis_panel.csv", index=False, encoding="utf-8-sig")
    print(f"\n保存: {outdir}/hypothesis_report.txt, hypothesis_panel.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--universe-cap", type=int, default=None)
    ap.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    ap.add_argument("--out-dir", type=str, default="out_v7")
    a = ap.parse_args()
    main(a.dryrun, a.universe_cap, a.history_days, a.out_dir)
