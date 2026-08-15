"""stockbotTOM v7.1 Phase 2 — 次元別スコア分布の確認とθ較正（仕様書§13）。

Level 1（独立性）が通った後、Phase 2ではθ=0.60という切り方そのものの
妥当性を見る。dimension_scores.csvは全採点銘柄（本命/監視/除外を問わない）
を保存しているので、選抜バイアスを受けない母集団で分布を確認できる。

本スクリプトは判定を下さない。θの変更や次元の存廃は設計判断
（Fable相当・§13.1「最も安価に反証できる時点で反証を試みる」）であり、
ここで出すのはその判断のための材料（分布・θ近傍の密度）のみ。

使い方:
    python histogram.py out_v7/dimension_scores.csv
    python histogram.py out_v7/dimension_scores.csv --theta 0.55
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Actions側は「Install Japanese font」ステップでfonts-noto-cjkを導入済み前提。
# インストールされていても明示指定しないとDejaVu Sansのまま文字化けするため必須。
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]

DIMS = ["dim1", "dim2", "dim3", "dim4", "dim5"]
LABEL = {"dim1": "①トレンド", "dim2": "②相対力", "dim3": "③需給",
         "dim4": "④時間", "dim5": "⑤収縮"}

THETA_DEFAULT = 0.60   # v7/config.py の V7_THETA 既定値と同じ。変更時はそちらと揃える。
NEAR_BAND = 0.05       # θ近傍とみなす片側の幅


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    missing = [d for d in DIMS if d not in df.columns]
    if missing:
        raise SystemExit(f"必要な列がない: {missing}")
    return df


def _section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def summarize(df: pd.DataFrame, theta: float) -> dict:
    out = {}
    for d in DIMS:
        s = df[d].dropna()
        if len(s) == 0:
            out[d] = None
            continue
        out[d] = {
            "n": len(s),
            "mean": s.mean(), "median": s.median(),
            "below": (s < theta).mean(),
            "near": ((s >= theta - NEAR_BAND) & (s <= theta + NEAR_BAND)).mean(),
            "p10": s.quantile(0.10), "p90": s.quantile(0.90),
        }
    return out


def print_summary(stats: dict, theta: float, n_total: int) -> None:
    _section(f"次元別スコア分布（母集団 全{n_total:,}件・θ={theta:.2f}）")
    print(f"{'次元':10}{'n':>7}{'平均':>7}{'中央値':>7}{'θ未満':>8}"
          f"{'θ近傍±.05':>10}{'P10':>7}{'P90':>7}")
    for d in DIMS:
        st = stats[d]
        if st is None:
            print(f"{LABEL[d]:10}  データなし")
            continue
        print(f"{LABEL[d]:10}{st['n']:>7,d}{st['mean']:>7.3f}{st['median']:>7.3f}"
              f"{st['below']*100:>7.1f}%{st['near']*100:>9.1f}%"
              f"{st['p10']:>7.3f}{st['p90']:>7.3f}")
    print()
    print("θ近傍(±0.05)の比率が高い次元ほど、僅かな誤差で本命/監視の判定が")
    print("反転しやすい＝その次元にとってθが鋭敏すぎる可能性がある。")
    print("平均・中央値がθから離れて低い次元は、θ自体がその次元の分布と")
    print("噛み合っていない可能性がある。")
    print()
    print("本スクリプトはここまで。θ変更・次元の存廃の判断はこの先（§13.1）。")


def plot(df: pd.DataFrame, stats: dict, theta: float, out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    bins = np.linspace(0.0, 1.0, 41)
    for i, d in enumerate(DIMS):
        ax = axes[i]
        s = df[d].dropna()
        if len(s) == 0:
            ax.set_title(f"{LABEL[d]}（データなし）")
            ax.axis("off")
            continue
        ax.hist(s, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(theta, color="#C44E52", linestyle="--", linewidth=2)
        st = stats[d]
        ax.set_title(f"{LABEL[d]}   n={st['n']:,}\n"
                     f"θ未満{st['below']*100:.0f}%  θ近傍{st['near']*100:.0f}%",
                     fontsize=10)
        ax.set_xlim(0, 1)
    axes[5].axis("off")
    axes[5].text(0.0, 0.5,
                 f"θ = {theta:.2f}(赤破線)\n近傍帯 = ±{NEAR_BAND:.2f}\n\n"
                 "本図はθ較正の材料。\n判定はしない(§13.1)。",
                 fontsize=11, va="center")
    fig.suptitle("stockbotTOM v7.1 — Phase 2 次元別スコア分布", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n図を保存: {out_path}")


def main(path: str, theta: float, out_path: str) -> None:
    print("v7.1 Phase 2 — 次元別スコア分布とθ較正（仕様書§13）")
    df = load_scores(path)
    stats = summarize(df, theta)
    print_summary(stats, theta, len(df))
    plot(df, stats, theta, out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    theta = THETA_DEFAULT
    if "--theta" in args:
        idx = args.index("--theta")
        theta = float(args[idx + 1])
        del args[idx:idx + 2]

    p = args[0] if args else "out_v7/dimension_scores.csv"
    if not Path(p).exists():
        raise SystemExit(f"ファイルが見つからない: {p}\n"
                         "先に main_v7.py を実行して dimension_scores.csv を生成すること。")
    out_default = str(Path(p).with_name("phase2_histogram.png"))
    main(p, theta, out_default)
