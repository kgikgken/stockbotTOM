from __future__ import annotations

from typing import Tuple

import pandas as pd


def load_positions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "qty", "avg_price"])


def analyze_positions(pos_df: pd.DataFrame, mkt_score: int) -> Tuple[str, float]:
    """ポジション表示用。

    現在は最小実装：
      - position.csv がなければ空
      - total_asset 推定は固定 1.0 (外部で管理)
    """
    if pos_df is None or pos_df.empty:
        return "- なし", 1.0

    lines = []
    for _, row in pos_df.iterrows():
        t = str(row.get("ticker", "")).strip() or "-"
        rr = row.get("RR")
        aev = row.get("AdjEV")
        if rr is None and aev is None:
            lines.append(f"- {t}: 保有中")
        else:
            rr_s = f"{float(rr):.2f}" if rr is not None else "-"
            aev_s = f"{float(aev):.2f}" if aev is not None else "-"
            lines.append(f"- {t}: RR:{rr_s} 期待効率:{aev_s}")

    return "\n".join(lines), 1.0
