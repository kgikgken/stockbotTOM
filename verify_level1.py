"""v7.1 Level 1 — 独立性の検証（仕様書 §11.2）。

**このシステムは「②〜⑤は①と独立である」という仮説の上に建っている。**
仮説が偽なら「5次元ある」という主張は「1次元を5通りに書き換えただけ」になり、
本設計の中核的な主張が崩れる（§15.1 仮説1）。

仕様書は Level 1 について次のように定めている。
  - Level 1 が失敗すれば以降は無意味である。順序を守ること（§11.1）
  - Phase 4 を飛ばさない。最も安価に反証できる時点で反証を試みる（§13.1）
  - ④⑤ペアは同一のスイングポイント系列を共有するため構造的相関のリスクが最も高い（§11.2.1）

使い方:
    python verify_level1.py out_v7/dimension_scores.csv

判定基準（§11.2）:
    |ρ| > 0.7          → 独立でない。一方を削除する（統合しない）
    第1主成分 > 70%    → 次元分解が機能していない。設計を根本から再検討
    ρ(④,⑤) > 0.7      → ⑤を削除（統合ではなく削除。§3-⑤の方針）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = ["dim1", "dim2", "dim3", "dim4", "dim5"]
LABEL = {"dim1": "①トレンド", "dim2": "②相対力", "dim3": "③需給",
         "dim4": "④時間", "dim5": "⑤収縮"}

RHO_THRESHOLD = 0.7      # §11.2 手順2
PC1_THRESHOLD = 0.70     # §11.2 手順3
MIN_SAMPLES = 30         # これ未満は判定を出さない（n<30は信頼性警告）


def _section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    missing = [d for d in DIMS if d not in df.columns]
    if missing:
        raise SystemExit(f"必要な列がない: {missing}")
    return df


def check_sample_size(df: pd.DataFrame) -> bool:
    """サンプル数が判定に足りるか。不足なら警告して False。"""
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else "?"
    print(f"読込 {n:,}件 / 判定日数 {days} / "
          f"銘柄 {df['code'].nunique() if 'code' in df.columns else '?'}")
    if n < MIN_SAMPLES:
        print()
        print(f"⚠ サンプルが {n} 件しかない（判定には最低 {MIN_SAMPLES} 件必要）。")
        print("  以下の数値は参考値であり、独立性の判定には使えない。")
        print("  毎日の実行で dimension_scores.csv に蓄積してから再実行すること。")
        return False
    return True


def pairwise_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """手順2: 次元間のペアワイズ順位相関（スピアマン）。"""
    _section("手順2 — ペアワイズ順位相関（|ρ| > 0.7 なら独立でない）")
    sub = df[DIMS].dropna()
    if len(sub) < 3:
        print("データ不足で算出不可")
        return pd.DataFrame()
    rho = sub.corr(method="spearman")

    print(f"{'':12}", end="")
    for d in DIMS:
        print(f"{LABEL[d]:>10}", end="")
    print()
    for a in DIMS:
        print(f"{LABEL[a]:12}", end="")
        for b in DIMS:
            v = rho.loc[a, b]
            mark = "*" if (a != b and abs(v) > RHO_THRESHOLD) else " "
            print(f"{v:>9.3f}{mark}", end="")
        print()

    print()
    viol = []
    for i, a in enumerate(DIMS):
        for b in DIMS[i + 1:]:
            v = rho.loc[a, b]
            if abs(v) > RHO_THRESHOLD:
                viol.append((a, b, v))
    if viol:
        print("★閾値超過（独立でないペア）:")
        for a, b, v in sorted(viol, key=lambda x: -abs(x[2])):
            print(f"   {LABEL[a]} × {LABEL[b]}  ρ={v:+.3f}")
            print(f"      → 一方を削除する。統合して次元数を水増ししない（§11.2.2）")
    else:
        print("○ 全ペアで |ρ| < 0.7。設計前提は支持される")
    return rho


def priority_check_45(rho: pd.DataFrame, df: pd.DataFrame) -> None:
    """§11.2.1 ④⑤ペアの優先検証。構造的相関のリスクが最も高い。"""
    _section("§11.2.1 ④⑤ペアの優先検証（同一スイング系列を共有）")
    if rho.empty:
        print("算出不可")
        return
    v = rho.loc["dim4", "dim5"]
    print(f"ρ(④,⑤) = {v:+.3f}   （閾値 {RHO_THRESHOLD}）")
    if abs(v) > RHO_THRESHOLD:
        print()
        print("★判定: ⑤を削除する（§3-⑤の方針）")
        print("  理由: ④は階層D（実証空白）、⑤も階層Dだが、⑤については")
        print("  この領域で最も強い日本株エビデンス（Marshall et al. 2008）が否定的。")
        print("  証拠の非対称性から、統合ではなく⑤の削除を優先する。")
        print()
        print("  対応: 環境変数 V7_DIMS='1,2,3,4' で⑤を無効化して再実行")
    else:
        print("○ 閾値内。④⑤は独立とみなせる")


def pca_check(df: pd.DataFrame) -> None:
    """手順3: 主成分分析。第1主成分の寄与率 > 70% なら次元分解が機能していない。"""
    _section("手順3 — 主成分分析（第1主成分 > 70% なら次元分解が機能していない）")
    sub = df[DIMS].dropna()
    if len(sub) < len(DIMS) + 1:
        print("データ不足で算出不可")
        return
    X = sub.values.astype(float)
    # 標準化（分散の大きい次元に引きずられないように）
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    try:
        cov = np.cov(Z, rowvar=False)
        vals = np.linalg.eigvalsh(cov)[::-1]
        total = vals.sum()
        if total <= 0:
            print("分散がゼロ。算出不可")
            return
        ratio = vals / total
        cum = 0.0
        for i, r in enumerate(ratio, 1):
            cum += r
            print(f"   第{i}主成分  寄与率 {r*100:5.1f}%   累積 {cum*100:5.1f}%")
        print()
        pc1 = ratio[0]
        if pc1 > PC1_THRESHOLD:
            print(f"★第1主成分が {pc1*100:.1f}% と支配的（閾値 {PC1_THRESHOLD*100:.0f}%）")
            print("  → 次元分解が機能していない。設計を根本から再検討（§11.2.2）")
            print("  → 反証条件#5 に該当（§14）")
        else:
            print(f"○ 第1主成分 {pc1*100:.1f}% < {PC1_THRESHOLD*100:.0f}%。次元分解は機能している")
    except Exception as e:
        print(f"算出エラー: {type(e).__name__}: {e}")


def exclusion_impact(df: pd.DataFrame) -> None:
    """手順4: 各次元を除外した場合のDCS順位変化。不変なら冗長。"""
    _section("手順4 — 各次元を除外したときのDCS順位変化（不変なら冗長）")
    sub = df[DIMS].dropna()
    if len(sub) < 5:
        print("データ不足で算出不可")
        return
    base_rank = sub.mean(axis=1).rank(ascending=False)
    print(f"{'除外した次元':16}{'順位相関':>10}   判定")
    for d in DIMS:
        rest = [x for x in DIMS if x != d]
        r = sub[rest].mean(axis=1).rank(ascending=False)
        tau = base_rank.corr(r, method="spearman")
        verdict = "★冗長の疑い" if tau > 0.99 else ("要注意" if tau > 0.95 else "情報あり")
        print(f"{LABEL[d]:16}{tau:>10.4f}   {verdict}")
    print()
    print("順位相関が1.00に近いほど、その次元を外しても並びが変わらない＝冗長。")


def by_regime(df: pd.DataFrame) -> None:
    """手順5: 局面別に再実行。相関が局面依存ならレジーム別の扱いが必要。"""
    _section("手順5 — 局面別の相関（局面依存ならレジーム別の扱いが必要）")
    if "date" not in df.columns:
        print("date列がないため実施不可")
        return
    days = df["date"].nunique()
    if days < 60:
        print(f"判定日数が {days} 日しかない。局面別に分けるには不足（60日以上必要）。")
        print("蓄積が進んでから再実行すること。")
        return
    # 日付を3分割して簡易的に局面を区切る（本来はレジームタグで分けるべき）
    dates = sorted(df["date"].unique())
    thirds = np.array_split(np.array(dates), 3)
    for i, block in enumerate(thirds, 1):
        seg = df[df["date"].isin(block)][DIMS].dropna()
        if len(seg) < MIN_SAMPLES:
            print(f"期間{i}: n={len(seg)} 不足")
            continue
        rho = seg.corr(method="spearman")
        mx = 0.0; pair = None
        for a_i, a in enumerate(DIMS):
            for b in DIMS[a_i + 1:]:
                if abs(rho.loc[a, b]) > abs(mx):
                    mx = rho.loc[a, b]; pair = (a, b)
        print(f"期間{i}（{block[0]}〜{block[-1]}, n={len(seg)}）"
              f" 最大相関 {LABEL[pair[0]]}×{LABEL[pair[1]]} ρ={mx:+.3f}")


def main(path: str) -> None:
    print("v7.1 Level 1 — 独立性の検証（仕様書 §11.2）")
    print()
    df = load_scores(path)
    enough = check_sample_size(df)

    rho = pairwise_correlation(df)
    priority_check_45(rho, df)
    pca_check(df)
    exclusion_impact(df)
    by_regime(df)

    _section("総括")
    if not enough:
        print("⚠ サンプル不足のため、上記はいずれも参考値。")
        print("  仕様書§13.1「最も安価に反証できる時点で反証を試みる」に従い、")
        print("  データが貯まり次第この検証を再実行し、次元数を確定させること。")
    else:
        print("判定基準（§11.2.2）:")
        print("  全ペアで |ρ| < 0.7        → 設計前提は支持される")
        print("  特定ペアで |ρ| > 0.7      → 一方を削除（統合しない）")
        print("  第1主成分が支配的         → 設計を根本から再検討")
        print("  ①以外がすべて①と高相関   → 本設計の前提が崩壊")
    print()
    print("※Level 1 が失敗すれば Level 2 以降は無意味である（§11.1）。")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "out_v7/dimension_scores.csv"
    if not Path(p).exists():
        raise SystemExit(f"ファイルが見つからない: {p}\n"
                         "先に main_v7.py を実行して dimension_scores.csv を生成すること。")
    main(p)
