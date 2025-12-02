from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str

# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# 日付ユーティリティ（JST）
# ============================================================
def jst_today_date():
    # utils に依存しない「date」だけ
    return (datetime.utcnow() + timedelta(hours=9)).date()


# ============================================================
# yfinance 安全版
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _calc_vola20(close: pd.Series) -> float:
    """
    20日ボラ（リターン標準偏差）
    """
    # FutureWarning 対応で fill_method=None 明示
    ret = close.pct_change(fill_method=None)
    vola20 = ret.rolling(20).std().iloc[-1]
    return float(vola20) if pd.notna(vola20) else np.nan


def _classify_vola(vola: float) -> str:
    """
    ボラを3段階に分類
    """
    if not np.isfinite(vola):
        return "mid"
    if vola < 0.02:
        return "low"
    if vola > 0.06:
        return "high"
    return "mid"


def _tp_sl_from_vola(vola: float, score: float) -> Tuple[float, float]:
    """
    ボラ＋スコアから利確/損切りパーセントを決める
    戻り値は (tp_pct, sl_pct) で、例: 0.08, -0.04
    """
    band = _classify_vola(vola)

    if band == "low":
        tp = 0.06   # 利確 6%
        sl = -0.03  # 損切り -3%
    elif band == "high":
        tp = 0.12   # 利確 12%
        sl = -0.06  # 損切り -6%
    else:
        tp = 0.08   # 利確 8%
        sl = -0.04  # 損切り -4%

    # スコアが極端に高い A+++ は少し利確を伸ばす
    if score >= 90:
        tp += 0.01

    # 安全な範囲にクリップ
    tp = float(np.clip(tp, 0.05, 0.15))
    sl = float(np.clip(sl, -0.07, -0.02))
    return tp, sl


def _leverage_from_market(score: int) -> Tuple[float, str]:
    """
    地合いスコアから推奨レバとラベルを決定
    """
    if score >= 75:
        return 2.0, "攻め寄り（押し目+ブレイク）"
    if score >= 65:
        return 1.7, "やや攻め（押し目中心）"
    if score >= 55:
        return 1.3, "標準（押し目のみ）"
    if score >= 45:
        return 1.1, "守り寄り（厳選のみ）"
    return 1.0, "防御モード（新規INかなり絞る）"


# ============================================================
# スクリーニング
# ============================================================
def run_screening() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    A/Bランク候補を抽出
    - earnings_date があれば ±3日を自動で除外
    - A: score >= 85（最大3銘柄に絞る）
    - B: score >= 72
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    today = jst_today_date()

    A_list: List[Dict[str, Any]] = []
    B_list: List[Dict[str, Any]] = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        # --- 決算 ±3営業日フィルタ（earnings_date が入ってる銘柄だけ）---
        edate = None
        if "earnings_date" in row and pd.notna(row["earnings_date"]):
            try:
                ed = pd.to_datetime(row["earnings_date"]).date()
                edate = ed
            except Exception:
                edate = None

        if edate is not None:
            delta = abs((edate - today).days)
            if delta <= 3:
                # 決算前後3日 → 抽出対象から除外
                continue

        # --- ヒストリカル取得 ---
        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])
        vola20 = _calc_vola20(close)
        tp_pct, sl_pct = _tp_sl_from_vola(vola20, float(sc))

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(sc),
            "price": price,
            "vola20": vola20,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
        }

        if sc >= 85:
            A_list.append(info)
        elif sc >= 72:
            B_list.append(info)

    # スコア順ソート
    A_list.sort(key=lambda x: x["score"], reverse=True)
    B_list.sort(key=lambda x: x["score"], reverse=True)

    # A は最大3銘柄
    if len(A_list) > 3:
        A_list = A_list[:3]

    return A_list, B_list


# ============================================================
# ロット計算（100株単位）
# ============================================================
def _format_yen(v: float) -> str:
    try:
        return f"{int(round(v)):,}円"
    except Exception:
        return "-"


def _lot_plan(capital: float, lev: float, A_list: List[Dict[str, Any]]) -> None:
    """
    推定運用資産 × レバ から、Aランクに均等配分で100株単位ロットを決定
    - capital: 推定運用資産
    - lev: 推奨レバ（例: 1.3）
    - A_list: in-place で info["lot"] を埋める
    """
    if capital <= 0 or not A_list:
        return

    target_notional = capital * lev  # 目標建玉総額
    n = len(A_list)
    if n <= 0:
        return

    per_stock = target_notional / n

    for info in A_list:
        price = info["price"]
        if price <= 0:
            info["lot"] = 0
            continue

        raw_shares = per_stock / price
        lots_100 = int(raw_shares // 100)
        if lots_100 <= 0:
            lots_100 = 1  # 最低でも100株

        info["lot"] = lots_100 * 100


# ============================================================
# イベントカレンダー（任意）
# ============================================================
def load_events_for_today(path: str = "events.csv") -> List[str]:
    """
    events.csv があれば、今日の日付に一致するイベントを返す
    - カラム: date, title, impact(任意)
    - date は jst_today_str() と同じ "YYYY-MM-DD"
    """
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if "date" not in df.columns or "title" not in df.columns:
        return []

    today_str = jst_today_str()
    today_events = df[df["date"] == today_str]

    out: List[str] = []
    for _, r in today_events.iterrows():
        title = str(r["title"])
        impact = str(r.get("impact", "")).strip()
        if impact:
            out.append(f"{title}（{impact}）")
        else:
            out.append(title)
    return out


# ============================================================
# レポート生成
# ============================================================
def build_report() -> str:
    today_str = jst_today_str()

    # --- 地合い ---
    mkt = calc_market_score()
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    # 地合いから推奨レバ
    lev, lev_label = _leverage_from_market(mkt_score)

    # --- セクター ---
    secs = top_sectors_5d()
    if secs:
        sector_lines = [
            f"{i+1}. {name} ({chg:+.2f}%)"
            for i, (name, chg) in enumerate(secs[:3])
        ]
        sector_text = "\n".join(sector_lines)
    else:
        sector_text = "算出不可（データ不足）"

    # --- スクリーニング ---
    A_list, B_list = run_screening()

    # --- ポジション ---
    # load_positions は path 必須仕様に揃える
    pos_df = load_positions("positions.csv")
    # analyze_positions は (text, total_asset, total_pos, lev_now, risk_info) を返す想定
    pos_text, total_asset, total_pos, lev_now, risk_info = analyze_positions(pos_df)

    # 推定運用資産ベース（None/NaN の場合はデフォルト200万）
    if total_asset is None or not np.isfinite(total_asset):
        capital_base = 2_000_000
    else:
        capital_base = float(total_asset)

    # Aランクにロット付与
    _lot_plan(capital_base, lev, A_list)

    # --- イベント ---
    events = load_events_for_today()

    # --- 組み立て ---
    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{lev:.1f}倍（{lev_label}） / 目安: Aランク最大3銘柄")
    lines.append(f"- 推定運用資産ベース: 約{_format_yen(capital_base)}")
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    lines.append("◆ 今日のイベント・警戒情報")
    if events:
        for ev in events:
            lines.append(f"- {ev}")
    else:
        # events.csv 無しでも、地合いスコアから簡易コメント
        if mkt_score <= 45:
            lines.append("- ボラ高め・リスクイベント警戒（ロット控えめ推奨）")
        elif mkt_score >= 75:
            lines.append("- 地合い強いが、イベント前のフルレバは避けること。")
        else:
            lines.append("- 特筆すべきイベントなし（通常モード）")
    lines.append("")

    # --- Aランク ---
    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not A_list:
        lines.append("Aランク条件を満たす銘柄なし。今日は無理に攻めない。")
    else:
        for info in A_list:
            t = info["ticker"]
            name = info["name"]
            sc = info["score"]
            price = info["price"]
            tp_pct = info["tp_pct"]
            sl_pct = info["sl_pct"]
            lot = info.get("lot", 0)

            tp_price = price * (1 + tp_pct)
            sl_price = price * (1 + sl_pct)

            lines.append(f"- {t} {name}  Score:{sc:.1f}  現値:{price:.1f}")
            lines.append(f"    ・IN目安: {price:.1f}")
            lines.append(f"    ・利確目安: {tp_pct*100:.1f}%（{tp_price:.1f}）")
            lines.append(f"    ・損切り目安: {sl_pct*100:.1f}%（{sl_price:.1f}）")
            lines.append(f"    ・推奨ロット: {lot}株")
    lines.append("")

    # --- Bランク ---
    lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ）")
    if A_list and len(A_list) >= 3:
        lines.append("Aランク3銘柄が揃っているため、Bランク表示は省略。")
    else:
        if not B_list:
            lines.append("Bランク候補なし。")
        else:
            for info in B_list[:20]:
                t = info["ticker"]
                name = info["name"]
                sc = info["score"]
                price = info["price"]
                tp_pct = info["tp_pct"]
                sl_pct = info["sl_pct"]
                tp_price = price * (1 + tp_pct)
                sl_price = price * (1 + sl_pct)
                lines.append(f"- {t} {name}  Score:{sc:.1f}  現値:{price:.1f}")
                lines.append(
                    f"    ・利確目安: {tp_pct*100:.1f}%（{tp_price:.1f}）"
                    f" / 損切り目安: {sl_pct*100:.1f}%（{sl_price:.1f}）"
                )
    lines.append("")

    # --- ポジション分析 ---
    lines.append("📊 " + today_str + " ポジション分析")
    lines.append("")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# LINE送信（長文は自動分割）
# ============================================================
def send_line_multi(text: str) -> None:
    """
    LINE メッセージ長対策：
    1通あたり ~3900 文字で分割して Worker に投げる
    Worker 側は今まで通り {"text": "..."} を受け取るだけ
    """
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（printのみ）")
        print(text)
        return

    max_len = 3900
    parts: List[str] = []
    buf = ""

    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > max_len:
            parts.append(buf.rstrip("\n"))
            buf = ""
        buf += line + "\n"

    if buf.strip():
        parts.append(buf.rstrip("\n"))

    for part in parts:
        try:
            r = requests.post(WORKER_URL, json={"text": part}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    text = build_report()
    print(text)
    send_line_multi(text)


if __name__ == "__main__":
    main()