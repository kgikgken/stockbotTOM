from __future__ import annotations
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 基本設定
# ============================================================
DEFAULT_TOTAL_ASSET = 2_000_000.0  # 明示回答 Q2: ベース資産 200万円想定


# ============================================================
# CSV 読み込み
# ============================================================
def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    """
    positions.csv を読み込んで DataFrame を返す。
    無ければ空 DataFrame。
    想定カラム:
      - ticker : 銘柄コード (例: 4971.T)
      - qty    : 保有株数
      - avg_price : 取得単価
      - capital   : 任意。全体資産（あれば優先）
    """
    if not os.path.exists(path):
        print(f"[INFO] positions file not found: {path} (ノーポジ前提)")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read positions: {e}")
        return pd.DataFrame()

    # カラム名を整える（最低限）
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols and "銘柄" in cols:
        df.rename(columns={cols["銘柄"]: "ticker"}, inplace=True)
    if "qty" not in cols and "quantity" in cols:
        df.rename(columns={cols["quantity"]: "qty"}, inplace=True)
    if "qty" not in cols and "shares" in cols:
        df.rename(columns={cols["shares"]: "qty"}, inplace=True)
    if "avg_price" not in cols and "price" in cols:
        df.rename(columns={cols["price"]: "avg_price"}, inplace=True)

    # 最低限のカラムが無ければノーポジ扱い
    if "ticker" not in df.columns or "qty" not in df.columns or "avg_price" not in df.columns:
        print("[WARN] positions.csv columns missing (ticker/qty/avg_price). Treat as no position.")
        return pd.DataFrame()

    # 型を軽く整える
    df["ticker"] = df["ticker"].astype(str)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(float)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce").fillna(0).astype(float)

    # capital があれば数値に
    if "capital" in df.columns:
        df["capital"] = pd.to_numeric(df["capital"], errors="coerce")

    return df


# ============================================================
# 内部: 価格 & ボラ取得
# ============================================================
def _fetch_price_and_vola(ticker: str) -> Tuple[float, float]:
    """
    現値 & 20日ボラを返す。
    取得失敗時は (nan, nan)
    """
    try:
        df = yf.Ticker(ticker).history(period="60d")
        if df is None or df.empty:
            return float("nan"), float("nan")

        close = df["Close"].astype(float)
        price = float(close.iloc[-1])

        if len(close) < 21:
            vola20 = float("nan")
        else:
            ret = close.pct_change(fill_method=None)
            vola20 = float(ret.rolling(20).std().iloc[-1])

        return price, vola20
    except Exception as e:
        print(f"[WARN] price/vola fetch failed for {ticker}: {e}")
        return float("nan"), float("nan")


def _classify_risk(vola20: float) -> str:
    if not np.isfinite(vola20):
        return "中リスク"
    if vola20 < 0.02:
        return "低リスク"
    if vola20 < 0.05:
        return "中リスク"
    return "高リスク"


def _tp_sl_from_vola(vola20: float) -> Tuple[float, float]:
    """
    ボラから TP/SL % を決める（ポジ保有銘柄用）
    戻り値: (tp_pct, sl_pct) 例: 0.08, 0.04  → +8%, -4%
    """
    if not np.isfinite(vola20):
        tp, sl = 0.08, 0.04
    elif vola20 < 0.015:
        tp, sl = 0.06, 0.03
    elif vola20 < 0.03:
        tp, sl = 0.08, 0.04
    else:
        tp, sl = 0.12, 0.06

    tp = float(np.clip(tp, 0.04, 0.18))
    sl = float(np.clip(sl, 0.02, 0.08))
    return tp, sl


# ============================================================
# ポジション分析
# ============================================================
def analyze_positions(df: pd.DataFrame):
    """
    保有ポジションを集計して、
      - 表示用テキスト
      - 推定総資産 total_asset
      - 建玉合計 total_pos
      - レバレッジ lev (= total_pos / total_asset)
      - リスク情報 dict
    を返す。
    """
    # ノーポジ or 不正データ
    if df is None or df.empty:
        text = "- ノーポジション（休む日）"
        total_asset = DEFAULT_TOTAL_ASSET
        total_pos = 0.0
        lev = 0.0
        risk_info = {
            "num_positions": 0,
            "est_total_asset": total_asset,
            "total_position": total_pos,
            "lev": lev,
            "est_daily_risk": 0.0,
        }
        return text, total_asset, total_pos, lev, risk_info

    # 有効ポジのみ
    active = df[(df["qty"] > 0) & (df["avg_price"] > 0)].copy()
    if active.empty:
        text = "- ノーポジション（休む日）"
        # capital があればそれを使う
        if "capital" in df.columns and df["capital"].notna().any():
            total_asset = float(df["capital"].dropna().iloc[0])
        else:
            total_asset = DEFAULT_TOTAL_ASSET
        total_pos = 0.0
        lev = 0.0
        risk_info = {
            "num_positions": 0,
            "est_total_asset": total_asset,
            "total_position": total_pos,
            "lev": lev,
            "est_daily_risk": 0.0,
        }
        return text, total_asset, total_pos, lev, risk_info

    lines = []
    total_pos_value = 0.0
    est_total_asset = None
    total_risk_if_sl = 0.0

    for _, row in active.iterrows():
        ticker = str(row["ticker"]).strip()
        qty = float(row["qty"])
        avg_price = float(row["avg_price"])

        cur_price, vola20 = _fetch_price_and_vola(ticker)
        if not np.isfinite(cur_price):
            cur_price = avg_price

        pos_val = cur_price * qty
        total_pos_value += pos_val

        pnl_pct = 0.0
        if avg_price > 0:
            pnl_pct = (cur_price / avg_price - 1.0) * 100.0

        # TP/SL 推奨
        tp_pct, sl_pct = _tp_sl_from_vola(vola20)
        tp_price = avg_price * (1.0 + tp_pct)
        sl_price = avg_price * (1.0 - sl_pct)

        risk_label = _classify_risk(vola20)
        est_loss_if_sl = pos_val * sl_pct  # sl_pct > 0だが「%」として使うので絶対値
        total_risk_if_sl += abs(est_loss_if_sl)

        lines.append(
            f"- {ticker}: 現値 {cur_price:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:+.2f}%"
        )
        lines.append(
            f"    ・利確目安: +{tp_pct*100:.1f}%（{tp_price:.1f}）"
        )
        lines.append(
            f"    ・損切り目安: -{sl_pct*100:.1f}%（{sl_price:.1f}）"
        )
        lines.append(
            f"    ・リスク: {risk_label}"
        )
        lines.append("")

    # 総資産推定
    if "capital" in df.columns and df["capital"].notna().any():
        est_total_asset = float(df["capital"].dropna().iloc[0])
    else:
        # キャッシュ情報が無いなら、ポジションの 1.2〜1.5倍程度を資産とみなす
        est_total_asset = max(total_pos_value * 1.2, DEFAULT_TOTAL_ASSET)

    lev = 0.0
    if est_total_asset > 0:
        lev = float(total_pos_value / est_total_asset)

    # 1トレードあたり 1.5% リスク想定
    risk_per_trade = est_total_asset * 0.015

    risk_info: Dict[str, float] = {
        "num_positions": int(len(active)),
        "est_total_asset": float(est_total_asset),
        "total_position": float(total_pos_value),
        "lev": lev,
        "est_daily_risk": float(total_risk_if_sl),
        "risk_per_trade": float(risk_per_trade),
    }

    pos_text = "\n".join(lines).strip()
    return pos_text, est_total_asset, total_pos_value, lev, risk_info