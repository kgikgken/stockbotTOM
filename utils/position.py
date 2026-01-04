from __future__ import annotations

from datetime import date
from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import weekday_monday


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def calc_weekly_new_count(df: pd.DataFrame, today_date: date) -> int:
    """
    positions.csv に entry_date(YYYY-MM-DD) がある場合のみカウント
    無ければ 0（裁量で盛らない）
    """
    if df is None or df.empty or "entry_date" not in df.columns:
        return 0

    mon = weekday_monday(today_date)
    cnt = 0
    for v in df["entry_date"].astype(str).tolist():
        try:
            d = pd.to_datetime(v, errors="coerce").date()
        except Exception:
            continue
        if d is None or pd.isna(d):
            continue
        if mon <= d <= today_date:
            cnt += 1
    return int(cnt)


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """
    positions.csv 最低限:
      ticker, entry_price, quantity
    追加で:
      asset_total があればそれ優先
      risk_per_trade があれば警告計算に使う（無ければ簡易）

    出力は report の “保有ポジション” に貼られるので日本語で
    """
    if df is None or df.empty:
        return "ノーポジション", 2_000_000.0

    # total asset（列があるなら最優先）
    total_asset = None
    for col in ["asset_total", "total_asset", "asset"]:
        if col in df.columns:
            try:
                v = float(df[col].iloc[0])
                if np.isfinite(v) and v > 0:
                    total_asset = v
                    break
            except Exception:
                pass

    lines = []
    total_value = 0.0
    max_loss_est = 0.0  # “想定最大損失”を荒く出す（ロット事故警告）

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)
        if entry_price <= 0 or qty <= 0:
            lines.append(f"- {ticker}: 損益 n/a")
            continue

        # current price
        cur = entry_price
        try:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry_price) / entry_price * 100.0 if entry_price > 0 else np.nan
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        # 簡易“最大損失”見積り：entryの -4% を仮定（本当は銘柄別STOPに合わせたいが positions.csv にstop無い想定）
        # stop_price列があればそれを使う
        stop_price = None
        if "stop_price" in df.columns:
            try:
                sp = float(row.get("stop_price", 0) or 0)
                if np.isfinite(sp) and sp > 0:
                    stop_price = sp
            except Exception:
                stop_price = None

        if stop_price is None:
            stop_price = entry_price * 0.96

        loss = max(0.0, (entry_price - stop_price) * qty)
        max_loss_est += loss

        if np.isfinite(pnl_pct):
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")
        else:
            lines.append(f"- {ticker}: 損益 n/a")

    # asset estimate
    asset_est = float(total_asset) if (total_asset is not None) else (float(total_value) if total_value > 0 else 2_000_000.0)

    # ロット事故警告（資産比 8%超えたら出す）
    warn = ""
    if asset_est > 0 and np.isfinite(max_loss_est) and max_loss_est > 0:
        ratio = max_loss_est / asset_est
        if ratio >= 0.08:
            warn = f"\n\n⚠ ロット事故警告\n想定最大損失: 約{int(round(max_loss_est)):,}円（資産比 {ratio*100:.2f}%）"

    text = "\n".join(lines) if lines else "ノーポジション"
    return text + warn, asset_est