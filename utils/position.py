import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# 価格取得（安全版）
# ============================================================
def _safe_fetch_close(ticker: str, period="5d"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        close = df["Close"]
        if len(close) == 0:
            return None
        return float(close.iloc[-1])
    except:
        return None


# ============================================================
# ボラティリティ分類（エラー修正済）
# ============================================================
def _classify_vola(vola: float) -> str:
    """20日ボラティリティ分類：low / mid / high"""

    # -------- 修正点：Series → float に変換 --------
    try:
        if hasattr(vola, "iloc"):      # Series だった場合
            vola = float(vola.iloc[-1])
        else:
            vola = float(vola)
    except Exception:
        vola = np.nan
    # ------------------------------------------------------

    if not np.isfinite(vola):
        return "mid"

    if vola < 0.02:
        return "low"
    if vola > 0.06:
        return "high"

    return "mid"


# ============================================================
# 利確・損切りラインの計算
# ============================================================
def _calc_tp_sl_for_pos(price: float, vola20: float):
    """
    price: 現在値
    vola20: 20日ボラ（標準偏差）
    """

    vc = _classify_vola(vola20)

    if vc == "low":
        tp_pct = 0.03
        sl_pct = 0.015
    elif vc == "high":
        tp_pct = 0.05
        sl_pct = 0.03
    else:  # mid
        tp_pct = 0.04
        sl_pct = 0.02

    tp_price = price * (1 + tp_pct)
    sl_price = price * (1 - sl_pct)

    return tp_pct, sl_pct, tp_price, sl_price


# ============================================================
# ポジションファイル読み込み
# ============================================================
def load_positions(path="positions.csv"):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame(columns=["ticker", "size", "price"])


# ============================================================
# 20日ボラ取得
# ============================================================
def _get_20d_vola(ticker: str):
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan

        close = df["Close"]
        ret = close.pct_change()
        vola20 = ret.rolling(20).std().iloc[-1]
        return vola20
    except:
        return np.nan


# ============================================================
# ポジション評価（利確ラインつき）
# ============================================================
def analyze_positions(df: pd.DataFrame):
    if df is None or df.empty:
        return "（ポジションなし）", 0, 0, 0, []

    lines = []
    total_pos_value = 0

    risk_info_list = []

    for _, r in df.iterrows():
        ticker = str(r["ticker"])
        size = float(r["size"])
        buy_price = float(r["price"])

        cur_price = _safe_fetch_close(ticker)
        if cur_price is None:
            continue

        pnl_pct = (cur_price - buy_price) / buy_price * 100

        # ---- 20日ボラ ----
        vola20 = _get_20d_vola(ticker)

        # ---- 利確・損切り ----
        tp_pct, sl_pct, tp_price, sl_price = _calc_tp_sl_for_pos(cur_price, vola20)

        total_pos_value += cur_price * size

        lines.append(
            f"- {ticker}: 現値 {cur_price:.1f} / 取得 {buy_price:.1f} / 損益 {pnl_pct:.2f}%"
        )

        risk_info_list.append({
            "ticker": ticker,
            "cur": cur_price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        })

    # ---- 推定運用資産（= 現金 + ポジ） ----
    # 現時点ではポジションから計算できる総額だけ
    total_asset = total_pos_value * 1.7  # 仮の資産推定（調整可）

    # ---- レバ ----
    lev = total_pos_value / total_asset if total_asset > 0 else 0

    summary = (
        f"- 推定運用資産: {total_asset:,.0f}円\n"
        f"- 推定ポジション総額: {total_pos_value:,.0f}円（レバ約 {lev:.2f}倍）"
    )

    pos_text = "\n".join(lines) + "\n" + summary

    return pos_text, total_asset, total_pos_value, lev, risk_info_list