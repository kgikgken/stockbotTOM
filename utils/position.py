from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# 安全価格取得
# ============================================================
def _safe_price(ticker: str):
    """現在値を安全に取る"""
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan
        return float(df["Close"].iloc[-1])
    except:
        return np.nan


# ============================================================
# ボラティリティ分類
# ============================================================
def _classify_vola(vola: float) -> str:
    if not np.isfinite(vola):
        return "mid"
    if vola < 0.02:
        return "low"
    if vola > 0.06:
        return "high"
    return "mid"


def _calc_vola20(ticker: str):
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan
        close = df["Close"].astype(float)
        ret = close.pct_change(fill_method=None)
        vola20 = ret.rolling(20).std().iloc[-1]
        return float(vola20)
    except:
        return np.nan


# ============================================================
# 保有銘柄のTP/SL計算
# ============================================================
def _tp_sl_for_position(cur_price: float, vola20: float):
    """
    Returns:
      tp_pct, sl_pct, tp_price, sl_price
    """

    vc = _classify_vola(vola20)

    # デフォルト
    tp = 8.0
    sl = 4.0

    if vc == "low":
        tp = 6.0
        sl = 3.0
    elif vc == "high":
        tp = 12.0
        sl = 6.0

    tp_price = cur_price * (1 + tp / 100)
    sl_price = cur_price * (1 - sl / 100)

    return tp, sl, tp_price, sl_price


# ============================================================
# positions.csv 読み込み
# ============================================================
def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # 必須カラム保証
        req = {"ticker", "size", "price"}
        if not req.issubset(df.columns):
            raise ValueError("positions.csv に必要な列がありません")
        return df
    except Exception as e:
        print("[WARN] positions.csv 読み込み失敗:", e)
        return pd.DataFrame(columns=["ticker", "size", "price"])


# ============================================================
# 保有銘柄のリスク評価
# ============================================================
def _risk_level(vola: float) -> str:
    if not np.isfinite(vola):
        return "中立"
    if vola < 0.02:
        return "低リスク"
    if vola > 0.06:
        return "高リスク"
    return "中リスク"


# ============================================================
# 保有ポジション分析（メイン）
# ============================================================
def analyze_positions(df: pd.DataFrame):
    """
    Return:
      pos_text(str)
      total_asset(float)
      total_pos(float)
      lev(float)
      risk_info(list[dict])
    """
    if df is None or df.empty:
        return "（現在ポジションなし）", 0, 0, 0, []

    lines = []
    total_pos = 0
    total_asset = 0
    risk_info = []

    for _, r in df.iterrows():
        ticker = str(r["ticker"])
        size = float(r["size"])
        cost = float(r["price"])

        cur = _safe_price(ticker)
        if not np.isfinite(cur):
            continue

        pnl_pct = (cur - cost) / cost * 100
        vola20 = _calc_vola20(ticker)
        risk = _risk_level(vola20)

        # TP/SL
        tp_pct, sl_pct, tp_price, sl_price = _tp_sl_for_position(cur, vola20)

        # 評価額
        pos_val = cur * size
        total_pos += pos_val

        # 推定総資産（含み損益込み）
        total_asset += pos_val

        lines.append(
            f"- {ticker}: 現値 {cur:.1f} / 取得 {cost:.1f} / 損益 {pnl_pct:.2f}%\n"
            f"    ・利確目安: +{tp_pct:.1f}%（{tp_price:.1f}）\n"
            f"    ・損切り目安: -{sl_pct:.1f}%（{sl_price:.1f}）\n"
            f"    ・リスク: {risk}"
        )

        risk_info.append({
            "ticker": ticker,
            "cur": cur,
            "cost": cost,
            "pnl_pct": pnl_pct,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "risk": risk
        })

    # レバレッジ計算（超単純化版）
    lev = total_pos / total_asset if total_asset > 0 else 0

    text = "\n".join(lines)
    return text, total_asset, total_pos, lev, risk_info