"""バックテスト結果の追加分析 — 再実行なしで既存CSVから読み取る(2026-07-26).

backtest_topdown.py が出した backtest_result.csv を読み、レポートには出ていない
以下を調べる。特に「年別の安定性」は、強気相場でだけ成立する設計を見抜くために必須。

  1. Q2のフェア比較  ゾーンが約定した候補だけを取り出し、同じ候補を即時で買った場合と比較。
                    レポートの比較はゾーン側に選別効果(押してきて下端を割らない銘柄だけ)が
                    混ざっているため、これを揃えないと差の解釈を誤る。
  2. 年別の安定性    2021〜2025の各年で結論が変わらないか。2022(下落局面)で崩れるなら、
                    設計は強気相場に依存していることになる。
  3. Rの分布        利益の何割が上位何%のトレードから来ているか(テール依存度)。
                    テール依存が極端なら、部分利確や早い利確は致命的になる。
  4. 保有日数と R   時間ストップを何日にすべきかの実データ。
  5. MFE分析        含み益のピークからどれだけ返したか。固定2R利確なら取れていた分も。

使い方:
    python analyze_backtest.py out_topdown/backtest_result.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _stat(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "件数": len(g),
        "平均R": round(g["r"].mean(), 3),
        "中央R": round(g["r"].median(), 3),
        "勝率": round((g["r"] > 0).mean(), 3),
    })


def _welch(a: pd.Series, b: pd.Series):
    """2群の平均差と95%信頼区間(等分散を仮定しない)。"""
    ma, mb = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = ma - mb
    return d, d - 1.96 * se, d + 1.96 * se, (d / se if se else np.nan)


def section(title: str):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def main(path: str):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    print(f"読込 {len(df):,}件 / 銘柄{df['code'].nunique():,} / "
          f"{df['date'].min().date()}〜{df['date'].max().date()}")

    # ---------- 1. Q2のフェア比較 ----------
    section("1. ゾーン vs 即時 — 同一候補でのフェア比較")
    off = df[df["partial"] == "off"]
    z = off[off["method"] == "zone"].set_index(["date", "code"])
    i = off[off["method"] == "immediate"].set_index(["date", "code"])
    both = z.index.intersection(i.index)
    print(f"ゾーンが約定した候補 {len(z):,}件 / うち即時側も記録がある {len(both):,}件")
    if len(both):
        zr, ir = z.loc[both, "r"], i.loc[both, "r"]
        d, lo, hi, t = _welch(zr, ir)
        print(f"  ゾーン   平均{zr.mean():+.3f}R 中央{zr.median():+.3f}R 勝率{(zr>0).mean():.1%}")
        print(f"  即時     平均{ir.mean():+.3f}R 中央{ir.median():+.3f}R 勝率{(ir>0).mean():.1%}")
        print(f"  差 {d:+.3f}R  t={t:.2f}  95%CI [{lo:+.3f}, {hi:+.3f}]"
              f"  → {'有意' if lo*hi > 0 else '有意差なし'}")
        print()
        print("  ※レポートの比較(全候補)との差:")
        za, ia = off[off["method"]=="zone"]["r"], off[off["method"]=="immediate"]["r"]
        print(f"    全候補での差 {za.mean()-ia.mean():+.3f}R → 同一候補での差 {d:+.3f}R")
        print("    差が縮むなら、ゾーンの優位の一部は『選別効果』だったことになる")

    # ---------- 2. 年別の安定性 ----------
    section("2. 年別の安定性 — 相場局面で結論が変わらないか")
    for method in ("zone", "immediate"):
        sub = off[off["method"] == method]
        r = sub.groupby("year").apply(_stat)
        print(f"[{method}]")
        print(r.to_string())
        print()
    zy = off[off["method"]=="zone"].groupby("year")["r"].mean()
    iy = off[off["method"]=="immediate"].groupby("year")["r"].mean()
    print("年別のゾーン優位(ゾーン − 即時):")
    for y in sorted(set(zy.index) & set(iy.index)):
        d = zy[y] - iy[y]
        print(f"  {y}: {d:+.3f}R  {'█'*max(0,int(d*20))}{'(逆転)' if d<0 else ''}")
    print()
    print("→ 全ての年で正なら頑健。ある年だけ逆転するなら局面依存")

    # ---------- 3. Rの分布とテール依存 ----------
    section("3. Rの分布 — 利益がどれだけテールに依存しているか")
    for method in ("zone", "immediate"):
        r = off[off["method"] == method]["r"].sort_values(ascending=False)
        tot = r.sum()
        print(f"[{method}] 合計{tot:+.0f}R / {len(r):,}件")
        for pct in (0.01, 0.05, 0.10, 0.20):
            n = max(1, int(len(r) * pct))
            share = r.head(n).sum() / tot if tot else np.nan
            print(f"   上位{pct:>4.0%} ({n:>4}件) が合計Rの {share:>6.1%} を生む")
        print(f"   負けトレードの割合 {(r<=0).mean():.1%} / 最大勝ち {r.max():+.1f}R / 最大負け {r.min():+.1f}R")
        print()

    # ---------- 4. 保有日数と実現R ----------
    section("4. 保有日数と実現R — 時間ストップの妥当性")
    zoff = off[off["method"] == "zone"]
    for trig in zoff["trigger"].unique():
        s = zoff[zoff["trigger"] == trig]
        print(f"[{trig}] n={len(s):,}  時間ストップ到達={((s['exit_reason'].str.contains('time')).mean()):.1%}")
        bins = [0, 3, 5, 10, 15, 20, 100]
        s = s.assign(bucket=pd.cut(s["bars"], bins))
        print(s.groupby("bucket", observed=True).apply(_stat).to_string())
        print()

    # ---------- 5. MFE分析 ----------
    section("5. MFE分析 — 含み益をどれだけ返したか / 固定利確なら取れた分")
    for trig in zoff["trigger"].unique():
        s = zoff[zoff["trigger"] == trig]
        giveback = s["mfe"] - s["r"]
        reach2r = (s["mfe"] >= 2.0).mean()
        # 固定2R利確を仮定した場合の実現R(2R到達なら+2R、未到達なら実際のR)
        fixed2 = np.where(s["mfe"] >= 2.0, 2.0, s["r"])
        print(f"[{trig}] n={len(s):,}")
        print(f"   平均MFE {s['mfe'].mean():+.3f}R → 実現 {s['r'].mean():+.3f}R "
              f"(平均{giveback.mean():.3f}R を返している)")
        print(f"   MFEが2R以上に達した割合 {reach2r:.1%}")
        print(f"   もし固定2R利確なら 平均{fixed2.mean():+.3f}R "
              f"({'固定2Rの方が良い' if fixed2.mean() > s['r'].mean() else '現行(トレーリング)の方が良い'})")
        print()

    # ---------- 6. トリガー×手法 ----------
    section("6. トリガー × 手法")
    print(off.groupby(["trigger", "method"]).apply(_stat).to_string())
    print()
    print("=" * 66)
    print("※カタリスト未確認・生存バイアスあり・手数料/スリッページ未考慮。")
    print("  絶対値ではなく比較にのみ使うこと。")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "out_topdown/backtest_result.csv"
    if not Path(p).exists():
        print(f"ファイルが見つからない: {p}")
        sys.exit(1)
    main(p)
