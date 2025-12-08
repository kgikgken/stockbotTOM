from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# Load positions
# ============================================================
def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    """
    positions.csv を読む
    （なければ空のDataFrameを返す）

    columns例：
    ticker, shares, avg_price
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        # ファイルなし or 読めない場合
        cols = ["ticker", "shares", "avg_price"]
        return pd.DataFrame(columns=cols)


# ============================================================
# 現値取得（yfinance）
# ============================================================
def _fetch_price(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan
        close = df["Close"].astype(float)
        return float(close.iloc[-1])
    except Exception:
        return np.nan


# ============================================================
# ボラ20（RRなどで使用）
# ============================================================
def _fetch_vola20(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df is None or df.empty or len(df) < 20:
            return np.nan
        close = df["Close"].astype(float)
        ret = close.pct_change(fill_method=None)
        return float(ret.rolling(20).std().iloc[-1])
    except Exception:
        return np.nan


# ============================================================
# Analyze positions
# ============================================================
def analyze_positions(df: pd.DataFrame):
    """
    現在のポジションをまとめて返す

    戻り値：
      pos_text(str)        → 位置情報をLine表示用に整形
      total_asset(float)   → 総資産（推定）
      total_pos(float)     → 建玉合計
      lev(float)           → レバレッジ
      risk_info(dict)      → 拡張用

    ※ ノーポジションでも動作する
    """
    if df is None or df.empty:
        pos_text = "- ノーポジション（休む日）"
        # ※ 資産推定は main 側で決めるので None に返す
        return pos_text, np.nan, 0.0, 1.0, {}

    lines = []
    total_pos = 0.0

    # 詳細情報計算
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        shares = float(row.get("shares", 0))
        avg_price = float(row.get("avg_price", 0))

        if not ticker or shares <= 0:
            continue

        cur = _fetch_price(ticker)
        if not np.isfinite(cur):
            continue

        pnl_pct = ((cur / avg_price) - 1.0) * 100.0
        pnl_pct = round(pnl_pct, 2)

        # 推定RR情報（現状は参考値 → Phase2で本格化）
        vola = _fetch_vola20(ticker)
        # baseline: tp=+8%, sl=-4%
        tp = avg_price * 1.08
        sl = avg_price * 0.96

        value = cur * shares
        total_pos += value

        lines.append(
            f"- {ticker}: 現値 {cur:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:.2f}%\n"
            f"    ・利確目安: +8.0%（{tp:.1f}）\n"
            f"    ・損切り目安: -4.0%（{sl:.1f}）"
        )

    if not lines:
        pos_text = "- ノーポジション（休む日）"
        return pos_text, np.nan, 0.0, 1.0, {}

    pos_text = "\n".join(lines)

    # 総資産推定（現金を含む前提 → 現状は株価でのみ計算）
    # Phase2では 現金残高 / レバ情報を追加可能
    total_asset = total_pos  # conservative
    lev = 1.0  # 現状はレバを固定（Phase2で動的）

    return pos_text, total_asset, total_pos, lev, {}