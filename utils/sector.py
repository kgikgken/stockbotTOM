# ============================================
# utils/sector.py
# セクター別の短期モメンタム集計（5日）
# ============================================

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def top_sectors_5d(universe_df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    universe_df に含まれる銘柄の sector 列ごとに、5日リターンの平均を集計して上位を返す。

    想定カラム:
      - sector: セクター名（文字列）
      - ret_5d: 5日リターン（%でも小数でもOK。ここでは「小数(0.05=+5%)」を推奨）
        ※ ret_5d が %表記(5.0=+5%) っぽい場合は自動で /100 して小数に寄せる
    """
    if universe_df is None or len(universe_df) == 0:
        return []

    if "sector" not in universe_df.columns:
        return []

    df = universe_df.copy()

    if "ret_5d" not in df.columns:
        # ret_5d が無い場合は空で返す（理由: 計算元が別モジュールの可能性がある）
        return []

    # 数値化
    df["ret_5d"] = df["ret_5d"].apply(_safe_float)

    # %っぽい値(例: 5.2)が混じってたら小数へ寄せる
    # 例: ret_5d の絶対値が 1 を大きく超える割合が一定以上なら /100
    abs_gt_1_ratio = (df["ret_5d"].abs() > 1.0).mean() if len(df) else 0.0
    if abs_gt_1_ratio > 0.3:
        df["ret_5d"] = df["ret_5d"] / 100.0

    # 欠損除外
    df = df.dropna(subset=["sector", "ret_5d"])
    if len(df) == 0:
        return []

    # 集計（平均）
    g = df.groupby("sector", as_index=False)["ret_5d"].mean()
    g = g.sort_values("ret_5d", ascending=False).head(int(top_k))

    # 出力は %表記で揃える（LINE表示用）
    out: List[Tuple[str, float]] = []
    for _, row in g.iterrows():
        out.append((str(row["sector"]), float(row["ret_5d"]) * 100.0))

    return out