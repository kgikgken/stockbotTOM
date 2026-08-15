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

# ④時間のサブ指標(すべてconfig.pyのd4_*で定義されるpeak_score/山形)。
# ④=4項目の単純平均。どれが足を引っ張っているかは合成値だけでは分からない。
DIM4_PARTS = ["part_4-a", "part_4-b", "part_4-c", "part_4-d"]
DIM4_PART_LABEL = {
    "part_4-a": "4-a ベース継続週数",
    "part_4-b": "4-b 200日線超え日数",
    "part_4-c": "4-c ステージ滞在日数",
    "part_4-d": "4-d ベース形成回数",
}


def analyze_dim4_parts(df: pd.DataFrame, theta: float) -> dict | None:
    """④の内訳(4-a〜4-d)を分析する。列が無い場合はNoneを返す(旧CSV互換)。"""
    if not all(c in df.columns for c in DIM4_PARTS):
        return None
    sub = df[DIM4_PARTS].copy()
    stats = {}
    for c in DIM4_PARTS:
        s = sub[c].dropna()
        stats[c] = {
            "n": len(s), "n_missing": len(sub) - len(s),
            "mean": s.mean() if len(s) else float("nan"),
            "median": s.median() if len(s) else float("nan"),
            "below": (s < theta).mean() if len(s) else float("nan"),
        }
    # ボトルネック分析: 4項目すべて揃っている行で、どれが最小値(足を引っ張る
    # 項目)になっている頻度が高いかを見る。同点は複数カウント。
    complete = sub.dropna()
    bottleneck = {c: 0 for c in DIM4_PARTS}
    if len(complete):
        row_min = complete.min(axis=1)
        for c in DIM4_PARTS:
            bottleneck[c] = int((complete[c] <= row_min + 1e-9).sum())
    stats["_bottleneck_n"] = len(complete)
    stats["_bottleneck"] = bottleneck
    return stats


def print_dim4_parts(stats: dict | None, theta: float) -> None:
    if stats is None:
        print("\n(④のサブ指標列が無いCSVのため、④内訳分析はスキップ)")
        return
    _section(f"④時間のサブ指標内訳（θ={theta:.2f}）")
    print(f"{'サブ指標':24}{'n':>7}{'欠損':>6}{'平均':>7}{'中央値':>7}{'θ未満':>8}")
    for c in DIM4_PARTS:
        st = stats[c]
        print(f"{DIM4_PART_LABEL[c]:24}{st['n']:>7,d}{st['n_missing']:>6,d}"
              f"{st['mean']:>7.3f}{st['median']:>7.3f}{st['below']*100:>7.1f}%")
    n_bn = stats["_bottleneck_n"]
    print(f"\n4項目全て揃う{n_bn:,}件中、各項目が最小値(ボトルネック)だった回数:")
    for c in DIM4_PARTS:
        cnt = stats["_bottleneck"][c]
        pct = cnt / n_bn * 100 if n_bn else float("nan")
        print(f"  {DIM4_PART_LABEL[c]:24}{cnt:>6,d}件  ({pct:5.1f}%)")
    print()
    print("特定の項目に偏っていれば、④が低い主因はその項目のスケール(idealの")
    print("位置・レンジ)にある。均等に近ければ、4項目の同時充足が構造的に")
    print("難しいこと自体が主因(§13.1の判断で言う『スケール修正』の対象範囲)。")


def plot_dim4_parts(df: pd.DataFrame, theta: float, out_path: str) -> bool:
    if not all(c in df.columns for c in DIM4_PARTS):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    bins = np.linspace(0.0, 1.0, 41)
    for ax, c in zip(axes, DIM4_PARTS):
        s = df[c].dropna()
        ax.hist(s, bins=bins, color="#55A868", edgecolor="white", alpha=0.85)
        ax.axvline(theta, color="#C44E52", linestyle="--", linewidth=2)
        below = (s < theta).mean() * 100 if len(s) else float("nan")
        ax.set_title(f"{DIM4_PART_LABEL[c]}\nn={len(s):,}  θ未満{below:.0f}%",
                     fontsize=10)
        ax.set_xlim(0, 1)
    fig.suptitle("stockbotTOM v7.1 — ④時間 サブ指標内訳", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"図を保存: {out_path}")
    return True


def load_scores(path: str) -> pd.DataFrame:
    """CSV単体、またはCSVを含むディレクトリを読む。

    ディレクトリを渡すと配下の*.csvを全て連結する。日付別に蓄積した
    out_v7/history/ をそのまま指定して、複数営業日を横断した検証ができる。
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise SystemExit(f"CSVが1つも無い: {path}")
        df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                       ignore_index=True)
        print(f"({len(files)}ファイルを連結)")
    else:
        df = pd.read_csv(p, encoding="utf-8-sig")
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

    d4_stats = analyze_dim4_parts(df, theta)
    print_dim4_parts(d4_stats, theta)
    d4_out = str(Path(out_path).with_name("dim4_parts.png"))
    if plot_dim4_parts(df, theta, d4_out):
        pass
    else:
        print("\n(④のサブ指標列が無いCSVのため、④内訳の図はスキップ)")


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
