import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

# あなたの運用ベース資産（スタート時）
BASE_ASSET = 3_375_662  # 必要ならあとで調整


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    """
    positions.csv を読み込む
    期待フォーマット:
        ticker,qty,avg_price
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "qty", "avg_price"])

    df = pd.read_csv(path)
    for col in ["ticker", "qty", "avg_price"]:
        if col not in df.columns:
            raise ValueError(f"positions.csv に {col} カラムがありません")

    df["ticker"] = df["ticker"].astype(str)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df = df.dropna(subset=["ticker", "qty", "avg_price"])
    return df


def _fetch_hist_for_pos(ticker: str, period: str = "60d") -> pd.DataFrame | None:
    """ポジション用に安全にヒストリカル取得"""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _classify_vola(vola: float) -> str:
    """20日ボラティリティで分類"""
    if not np.isfinite(vola):
        return "mid"
    if vola < 0.02:
        return "low"
    if vola > 0.06:
        return "high"
    return "mid"


def _calc_tp_sl_for_pos(price: float, vola: float) -> Tuple[float, float, float, float]:
    """
    ポジション単位の利確/損切りラインを決める
    tp_pct, sl_pct, tp_price, sl_price を返す
    """
    vc = _classify_vola(vola)

    # ボラ低い銘柄はタイト、ボラ高い銘柄は広め
    if vc == "low":
        tp_pct = 0.06   # +6%
        sl_pct = -0.03  # -3%
    elif vc == "high":
        tp_pct = 0.12   # +12%
        sl_pct = -0.06  # -6%
    else:
        tp_pct = 0.08   # +8%
        sl_pct = -0.045 # -4.5%

    tp_price = price * (1 + tp_pct)
    sl_price = price * (1 + sl_pct)
    return tp_pct, sl_pct, tp_price, sl_price


def analyze_positions(df: pd.DataFrame) -> tuple[str, float, float, float, Dict[str, Any]]:
    """
    ポジションを解析してテキスト＋集計値を返す

    戻り値:
        pos_text: 表示用テキスト
        total_asset: 推定運用資産
        total_pos:   推定ポジション総額
        lev:         レバレッジ (total_pos / total_asset)
        risk_info:   IN/OUT判定で使うリスク概要
    """
    if df is None or df.empty:
        pos_text = "保有ポジションなし。"
        return pos_text, float(BASE_ASSET), 0.0, 0.0, {
            "has_big_loss": False,
            "has_mid_loss": False,
            "has_big_gain": False,
            "max_loss_pct": 0.0,
            "max_gain_pct": 0.0,
        }

    lines: list[str] = []
    total_pos = 0.0
    pnl_list: list[float] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        qty = float(row["qty"])
        avg_price = float(row["avg_price"])

        if qty <= 0 or avg_price <= 0:
            continue

        hist = _fetch_hist_for_pos(ticker)
        if hist is None:
            lines.append(f"- {ticker}: データ取得失敗（現値不明）")
            continue

        close = hist["Close"].astype(float)
        cur_price = float(close.iloc[-1])
        value = cur_price * qty
        total_pos += value

        # 損益
        pnl_pct = (cur_price - avg_price) / avg_price * 100.0
        pnl_list.append(pnl_pct)

        # ボラとTP/SL
        if len(close) >= 20:
            vola20 = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(20)
        else:
            vola20 = np.nan

        tp_pct, sl_pct, tp_price, sl_price = _calc_tp_sl_for_pos(cur_price, vola20)

        # シグナル判定
        signals: list[str] = []
        if pnl_pct <= -5.0:
            signals.append("損切り候補（-5%以上）")
        elif pnl_pct <= -3.0:
            signals.append("警戒ゾーン（-3〜-5%）")
        if pnl_pct >= 10.0:
            signals.append("利確候補（+10%以上）")

        vc = _classify_vola(vola20)
        if vc == "high":
            signals.append("ボラ高め（サイズ注意）")

        line = (
            f"- {ticker}: 現値 {cur_price:.1f} / 取得 {avg_price:.1f} / "
            f"損益 {pnl_pct:+.2f}% / "
            f"利確目安 +{tp_pct*100:.1f}%({tp_price:.1f}) / "
            f"損切目安 {sl_pct*100:.1f}%({sl_price:.1f})"
        )
        if signals:
            line += " / シグナル: " + "・".join(signals)

        lines.append(line)

    total_asset = float(BASE_ASSET)
    lev = total_pos / total_asset if total_asset > 0 else 0.0

    lines.append(f"- 推定運用資産: {total_asset:,.0f}円")
    lines.append(f"- 推定ポジション総額: {total_pos:,.0f}円（レバ約 {lev:.2f}倍）")

    if pnl_list:
        max_loss = float(min(pnl_list))
        max_gain = float(max(pnl_list))
    else:
        max_loss = 0.0
        max_gain = 0.0

    risk_info = {
        "has_big_loss": max_loss <= -5.0,
        "has_mid_loss": max_loss <= -3.0,
        "has_big_gain": max_gain >= 10.0,
        "max_loss_pct": max_loss,
        "max_gain_pct": max_gain,
    }

    return "\n".join(lines), total_asset, total_pos, lev, risk_info