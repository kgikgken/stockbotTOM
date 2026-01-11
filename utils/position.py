from __future__ import annotations

import os
import pandas as pd
import yfinance as yf

from utils.screen_logic import detect_setup, score_candidate

def load_positions(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, *, mkt_score: int, macro_on: bool) -> str:
    if df is None or len(df) == 0:
        return "ノーポジション"

    lines = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        try:
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is None or hist.empty or len(hist) < 120:
                lines.append(f"- {ticker}: データ不足")
                continue
        except Exception:
            lines.append(f"- {ticker}: 取得失敗")
            continue

        setup, anchors = detect_setup(hist)
        if setup == "NA":
            lines.append(f"- {ticker}: setup不明")
            continue

        scored = score_candidate(hist, setup, anchors, mkt_score=mkt_score, macro_on=macro_on)
        if not scored:
            lines.append(f"- {ticker}: 評価失敗")
            continue

        adjev = float(scored.get("adjev", 0.0))
        rr = float(scored.get("rr", 0.0))
        note = "（要注意）" if adjev < 0.5 else ""
        lines.append(f"- {ticker}: RR:{rr:.2f} AdjEV:{adjev:.2f}{note}")

    return "\n".join(lines) if lines else "ノーポジション"
