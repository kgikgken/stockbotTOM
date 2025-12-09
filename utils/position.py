from __future__ import annotations

from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 基本設定
# ============================================================
BASE_TOTAL_ASSET = 2_000_000  # 推定ベース資産（ノーポジ時など）


# ============================================================
# Helper
# ============================================================
def _safe_last(series: pd.Series, default: float = np.nan) -> float:
    if series is None or len(series) == 0:
        return float(default)
    v = series.iloc[-1]
    try:
        return float(v)
    except Exception:
        return float(default)


def _guess_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fetch_price(ticker: str) -> Optional[float]:
    """終値（直近）"""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is None or hist.empty:
            return None
        close = hist["Close"].astype(float)
        return float(close.iloc[-1])
    except Exception:
        return None


def _fetch_vola20(ticker: str) -> Optional[float]:
    """20日ボラ"""
    try:
        hist = yf.Ticker(ticker).history(period="60d")
        if hist is None or hist.empty or len(hist) < 21:
            return None
        close = hist["Close"].astype(float)
        ret = close.pct_change(fill_method=None)
        vola20 = ret.rolling(20).std().iloc[-1]
        return float(vola20)
    except Exception:
        return None


def _risk_label_from_vola(vola: Optional[float]) -> str:
    if vola is None or not np.isfinite(vola):
        return "不明"
    if vola < 0.02:
        return "低リスク"
    if vola < 0.05:
        return "中リスク"
    return "高リスク"


# ============================================================
# ポジション読込
# ============================================================
def load_positions(path: str) -> Optional[pd.DataFrame]:
    """
    positions.csv を読み込むだけ。
    想定カラム例:
      ticker, qty, avg_price, ...（多少違っても対応）
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)
    return df


# ============================================================
# ポジション分析
# ============================================================
def analyze_positions(
    df: Optional[pd.DataFrame],
) -> Tuple[str, float, float, float, Dict]:
    """
    戻り値:
      pos_text: LINE用テキスト（「ノーポジション」 or 箇条書き）
      total_asset: 推定総資産
      total_pos: 建玉合計
      lev: レバレッジ（total_pos / total_asset）
      risk_info: 補足情報(dict)
    """
    # ノーポジ
    if df is None or df.empty:
        pos_text = "ノーポジション"
        return pos_text, float(BASE_TOTAL_ASSET), 0.0, 0.0, {}

    qty_col = _guess_col(df, ["qty", "quantity", "shares", "size"])
    price_col = _guess_col(df, ["avg_price", "price", "avg", "entry_price"])

    if qty_col is None or price_col is None:
        pos_text = "ポジション情報の形式が不明（ticker/qty/avg_price が必要）"
        return pos_text, float(BASE_TOTAL_ASSET), 0.0, 0.0, {}

    detail_lines = []
    total_pos = 0.0
    vola_list = []
    hi_risk_count = 0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        try:
            qty = float(row.get(qty_col, 0.0))
        except Exception:
            qty = 0.0

        try:
            avg_price = float(row.get(price_col, 0.0))
        except Exception:
            avg_price = 0.0

        if qty == 0 or avg_price <= 0:
            continue

        cur_price = _fetch_price(ticker) or avg_price
        vola20 = _fetch_vola20(ticker)
        risk_label = _risk_label_from_vola(vola20)

        if vola20 is not None and np.isfinite(vola20):
            vola_list.append(vola20)
            if vola20 >= 0.05:
                hi_risk_count += 1

        notional = cur_price * qty
        total_pos += notional

        pnl_pct = (cur_price / avg_price - 1.0) * 100.0 if avg_price > 0 else 0.0

        # TP/SL目安（シンプル版：ボラで決める）
        if vola20 is None or not np.isfinite(vola20):
            tp_pct = 8.0
            sl_pct = 4.0
        else:
            if vola20 < 0.02:
                tp_pct = 6.0
                sl_pct = 3.0
            elif vola20 < 0.05:
                tp_pct = 8.0
                sl_pct = 4.0
            else:
                tp_pct = 12.0
                sl_pct = 6.0

        tp_price = cur_price * (1.0 + tp_pct / 100.0)
        sl_price = cur_price * (1.0 - sl_pct / 100.0)

        detail_lines.append(
            f"- {ticker}: 現値 {cur_price:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:.2f}%"
        )
        detail_lines.append(
            f"    ・利確目安: +{tp_pct:.1f}%（{tp_price:.1f}）"
        )
        detail_lines.append(
            f"    ・損切り目安: -{sl_pct:.1f}%（{sl_price:.1f}）"
        )
        detail_lines.append(
            f"    ・リスク: {risk_label}"
        )

    if total_pos <= 0:
        pos_text = "ノーポジション"
        return pos_text, float(BASE_TOTAL_ASSET), 0.0, 0.0, {}

    # 推定総資産：最低ベースは BASE_TOTAL_ASSET
    # 建玉が大きいときはレバ1.3〜2.0あたりを想定して逆算
    if total_pos <= BASE_TOTAL_ASSET * 1.3:
        total_asset = float(max(BASE_TOTAL_ASSET, total_pos / 1.3))
    else:
        total_asset = float(max(BASE_TOTAL_ASSET, total_pos / 1.6))

    lev = float(total_pos / total_asset) if total_asset > 0 else 0.0

    avg_vola = float(np.mean(vola_list)) if vola_list else np.nan

    risk_info: Dict = {
        "total_pos": total_pos,
        "lev": lev,
        "avg_vola20": avg_vola,
        "hi_risk_count": hi_risk_count,
        "positions_count": int(len(detail_lines) / 4) if detail_lines else 0,
    }

    pos_text = "\n".join(detail_lines) if detail_lines else "ノーポジション"
    return pos_text, total_asset, total_pos, lev, risk_info