from __future__ import annotations

import json
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


EQUITY_PATH_DEFAULT = "data/equity.json"


# ============================================================
# yfinance ラッパ
# ============================================================

def _fetch_history(ticker: str, period: str = "60d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _fetch_last_price(ticker: str) -> Optional[float]:
    df = _fetch_history(ticker, period="5d")
    if df is None:
        return None
    try:
        close = df["Close"].astype(float)
        return float(close.iloc[-1])
    except Exception:
        return None


def _calc_20d_vola(ticker: str) -> Optional[float]:
    df = _fetch_history(ticker, period="90d")
    if df is None:
        return None
    try:
        close = df["Close"].astype(float)
        ret = close.pct_change(fill_method=None)
        vola = float(ret.rolling(20).std().iloc[-1])
        if not np.isfinite(vola) or vola <= 0:
            return None
        return vola
    except Exception:
        return None


# ============================================================
# ボラに応じた 利確 / 損切り
# ============================================================

def _classify_vola(vola: Optional[float]) -> Tuple[str, float]:
    """
    vola: 20日ボラ（0.02 = 2%）を想定
    戻り値: (カテゴリ, 実際に使うボラ値)
    """
    if vola is None or not np.isfinite(vola) or vola <= 0:
        return "M", 0.02  # デフォルト中ボラ

    v = float(np.clip(vola, 0.005, 0.08))

    if v < 0.015:
        return "L", v
    elif v < 0.035:
        return "M", v
    else:
        return "H", v


def _calc_tp_sl_for_pos(price: float, vola: Optional[float]) -> Tuple[float, float, float, float]:
    """
    保有ポジション用の TP/SL
    返り値: (tp_pct, sl_pct, tp_price, sl_price)
    """
    cls, v = _classify_vola(vola)

    if cls == "L":
        tp_pct = 0.035  # +3.5%
        sl_pct = 0.015  # -1.5%
    elif cls == "M":
        tp_pct = 0.05   # +5%
        sl_pct = 0.025  # -2.5%
    else:  # H
        tp_pct = 0.07   # +7%
        sl_pct = 0.035  # -3.5%

    tp_price = price * (1.0 + tp_pct)
    sl_price = price * (1.0 - sl_pct)
    return tp_pct, sl_pct, tp_price, sl_price


# ============================================================
# positions.csv 読み込み
# ============================================================

def load_positions(path: str) -> pd.DataFrame:
    """
    positions.csv を読み込む
    想定カラム: ticker,size,price
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "size", "price"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["ticker", "size", "price"])

    # カラム名ゆらぎ対策
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker")
    size_col = cols.get("size") or cols.get("qty") or cols.get("quantity")
    price_col = cols.get("price") or cols.get("avg") or cols.get("average")

    if not (ticker_col and size_col and price_col):
        return pd.DataFrame(columns=["ticker", "size", "price"])

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].astype(str)
    out["size"] = pd.to_numeric(df[size_col], errors="coerce").fillna(0).astype(int)
    out["price"] = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0).astype(float)
    return out


# ============================================================
# equity.json 読み込み
# ============================================================

def _load_equity(equity_path: str, fallback_total_pos: float) -> float:
    """
    Cloudflare Worker から更新される equity.json を読む。
    なければ total_pos をそのまま返す。
    """
    try:
        if os.path.exists(equity_path):
            with open(equity_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            val = float(data.get("equity", fallback_total_pos))
            if np.isfinite(val) and val > 0:
                return val
    except Exception:
        pass
    return fallback_total_pos


# ============================================================
# ポジション分析
# ============================================================

def analyze_positions(
    df: pd.DataFrame,
    equity_path: str = EQUITY_PATH_DEFAULT,
) -> Tuple[str, float, float, float, List[str]]:
    """
    返り値:
      pos_text: 画面に出すテキスト
      total_asset: 推定運用資産
      total_pos:   ポジション総額
      lev:         レバレッジ
      risk_lines:  各ポジションの TP/SL 推奨行
    """
    if df is None or df.empty:
        return "ポジションなし。", 0.0, 0.0, 0.0, []

    lines: List[str] = []
    risk_lines: List[str] = []

    total_pos = 0.0

    for _, r in df.iterrows():
        ticker = str(r["ticker"])
        size = int(r["size"])
        buy_price = float(r["price"])

        if size <= 0 or buy_price <= 0:
            continue

        cur = _fetch_last_price(ticker)
        if cur is None:
            lines.append(f"- {ticker}: データ取得失敗（現値不明）")
            continue

        pos_val = cur * size
        total_pos += pos_val

        pnl_pct = (cur - buy_price) / buy_price * 100.0

        # 利確/損切り
        vola20 = _calc_20d_vola(ticker)
        tp_pct, sl_pct, tp_price, sl_price = _calc_tp_sl_for_pos(cur, vola20)

        lines.append(
            f"- {ticker}: 現値 {cur:.1f} / 取得 {buy_price:.1f} / 損益 {pnl_pct:+.2f}%"
        )

        risk_lines.append(
            f"- {ticker}: 利確:+{tp_pct*100:.1f}%（{tp_price:.0f}円） / "
            f"損切:-{sl_pct*100:.1f}%（{sl_price:.0f}円）"
        )

    if total_pos <= 0:
        return "ポジションなし。", 0.0, 0.0, 0.0, []

    total_asset = _load_equity(equity_path, total_pos)
    lev = float(total_pos / total_asset) if total_asset > 0 else 0.0

    lines.append(f"- 推定運用資産: {total_asset:,.0f}円")
    lines.append(f"- 推定ポジション総額: {total_pos:,.0f}円（レバ約 {lev:.2f}倍）")

    return "\n".join(lines), total_asset, total_pos, lev, risk_lines