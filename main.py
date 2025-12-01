from __future__ import annotations
import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# 共通ユーティリティ
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> pd.DataFrame | None:
    """安全版ヒストリカル取得"""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print(f"[WARN] fetch_history failed: {ticker} {e}")
        return None


def _classify_vola(vola: float) -> str:
    """ボラティリティ分類: low/mid/high"""
    if vola is None or not np.isfinite(vola):
        return "mid"
    if vola < 0.02:
        return "low"
    if vola > 0.06:
        return "high"
    return "mid"


def _calc_tp_sl_for_candidate(score: float, vola: float) -> Tuple[float, float]:
    """
    スクリーニング銘柄用 TP/SL（現値ベース、％）
    - score が高いほど TP を遠く
    - ボラ高いほど TP/SL を広く
    """
    # --- TP 基本ライン ---
    if score >= 90:
        tp = 0.12
    elif score >= 85:
        tp = 0.10
    elif score >= 80:
        tp = 0.08
    else:
        tp = 0.07

    vola_class = _classify_vola(vola)
    if vola_class == "low":
        tp -= 0.01
        sl = -0.035
    elif vola_class == "high":
        tp += 0.02
        sl = -0.060
    else:
        sl = -0.045

    # 安全な範囲にクリップ
    tp = float(np.clip(tp, 0.06, 0.15))
    sl = float(np.clip(sl, -0.07, -0.03))
    return tp, sl


def _in_label(score: float) -> str:
    """IN の強さラベル（期待値ベース）"""
    if score >= 88:
        return "強IN（自信度◎）"
    if score >= 82:
        return "通常IN（自信度◯）"
    if score >= 78:
        return "軽めIN（打診）"
    return "様子見"


def _fmt_yen(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening() -> Tuple[List[Dict], List[Dict]]:
    """
    ユニバース全体をスクリーニング
    戻り値: (Aリスト, Bリスト)
    各要素は dict:
      {
        ticker, name, sector, score, price,
        vola20, tp_pct, sl_pct, in_label
      }
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception as e:
        print(f"[WARN] universe load failed: {e}")
        return [], []

    if "ticker" not in uni.columns:
        print("[WARN] universe_jpx.csv に ticker カラムがありません")
        return [], []

    A_list: List[Dict] = []
    B_list: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        # スコア算出（utils.scoring）
        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue
        sc = float(sc)

        close = hist["Close"].astype(float)
        if close.isna().all():
            continue
        price = float(close.iloc[-1])

        # ボラティリティ（20日）
        if len(close) >= 21:
            vola20 = close.pct_change(fill_method=None).rolling(20).std().iloc[-1]
            vola20 = float(vola20) if np.isfinite(vola20) else np.nan
        else:
            vola20 = np.nan

        tp_pct, sl_pct = _calc_tp_sl_for_candidate(sc, vola20)
        in_lbl = _in_label(sc)

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": sc,
            "price": price,
            "vola20": vola20,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "in_label": in_lbl,
        }

        # ランク振り分け
        if sc >= 80:
            A_list.append(info)
        elif sc >= 70:
            B_list.append(info)

    A_list = sorted(A_list, key=lambda x: x["score"], reverse=True)
    B_list = sorted(B_list, key=lambda x: x["score"], reverse=True)

    return A_list, B_list


# ============================================================
# レポート生成（3分割）
# ============================================================
def build_report_parts() -> List[str]:
    today = jst_today_str()

    # ---- 地合い ----
    mkt = calc_market_score()
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    # ---- セクターTOP3 ----
    secs = top_sectors_5d()
    sec_lines: List[str] = []
    if isinstance(secs, (list, tuple)) and len(secs) > 0:
        for i, s in enumerate(secs[:3]):
            # s: (sector, pct) を想定
            try:
                sector_name = str(s[0])
                pct = float(s[1])
                sec_lines.append(f"{i+1}. {sector_name} ({pct:+.2f}%)")
            except Exception:
                continue
    if not sec_lines:
        sec_lines.append("算出できるセクターデータがありません。")

    # ---- スクリーニング ----
    A_list, B_list = run_screening()

    # ---- ポジション ----
    try:
        try:
            pos_df = load_positions(POSITIONS_PATH)
        except TypeError:
            # 古いシグネチャ対策
            pos_df = load_positions()
    except Exception as e:
        print(f"[WARN] load_positions failed: {e}")
        pos_df = None

    pos_text = "ポジション情報なし。"
    risk_summary = ""

    try:
        if pos_df is None:
            pos_text = "ポジション情報なし。positions.csv が読めません。"
        else:
            res = analyze_positions(pos_df)
            if isinstance(res, tuple):
                # (text, total_asset, total_pos, lev, risk_info) を想定
                if len(res) >= 1:
                    pos_text = str(res[0])
                if len(res) >= 5 and isinstance(res[4], str):
                    risk_summary = res[4]
            else:
                pos_text = str(res)
    except Exception as e:
        pos_text = f"ポジション分析中にエラー: {e}"

    # --------------------------------------------------------
    # Part1: 結論＋セクター
    # --------------------------------------------------------
    part1_lines: List[str] = []
    part1_lines.append(f"📅 {today} stockbotTOM 日報\n")
    part1_lines.append("◆ 今日の結論")
    part1_lines.append(f"- 地合いスコア: {mkt_score}点")
    part1_lines.append(f"- コメント: {mkt_comment}")
    part1_lines.append("")
    part1_lines.append("◆ 今日のTOPセクター（5日騰落率）")
    part1_lines.extend(sec_lines)

    part1 = "\n".join(part1_lines)

    # --------------------------------------------------------
    # Part2: Core候補（A/B）＋IN目安＋TP/SL
    # --------------------------------------------------------
    part2_lines: List[str] = []
    part2_lines.append("◆ Core候補 Aランク（本命押し目）")
    if not A_list:
        part2_lines.append("本命Aランクなし。")
    else:
        for r in A_list[:30]:
            part2_lines.append(
                f"- {r['ticker']} {r['name']}  "
                f"Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            part2_lines.append(
                f"   IN目安: {r['in_label']} / "
                f"TP:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['price']*(1+r['tp_pct']))}) / "
                f"SL:{r['sl_pct']*100:.1f}%({_fmt_yen(r['price']*(1+r['sl_pct']))})"
            )
    part2_lines.append("")

    part2_lines.append("◆ Core候補 Bランク（押し目候補）")
    if not B_list:
        part2_lines.append("Bランク候補なし。")
    else:
        for r in B_list[:50]:
            part2_lines.append(
                f"- {r['ticker']} {r['name']}  "
                f"Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            part2_lines.append(
                f"   IN目安: {r['in_label']} / "
                f"TP:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['price']*(1+r['tp_pct']))}) / "
                f"SL:{r['sl_pct']*100:.1f}%({_fmt_yen(r['price']*(1+r['sl_pct']))})"
            )

    part2 = "\n".join(part2_lines)

    # --------------------------------------------------------
    # Part3: ポジション分析（TP/SLは utils.position 側で計算済み想定）
    # --------------------------------------------------------
    part3_lines: List[str] = []
    part3_lines.append("◆ ポジション分析")
    part3_lines.append(pos_text)
    if risk_summary:
        part3_lines.append("")
        part3_lines.append(risk_summary)

    part3 = "\n".join(part3_lines)

    return [part1, part2, part3]


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str):
    """1メッセージ送信"""
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定（print のみ）")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=15)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信に失敗:", e)


def send_line_multi(parts: List[str]):
    """複数メッセージ分割送信"""
    for idx, p in enumerate(parts, start=1):
        if not p or not str(p).strip():
            continue
        print(f"\n===== PART {idx} =====\n")
        print(p)
        send_line(p)


# ============================================================
# Entry
# ============================================================
def main():
    parts = build_report_parts()
    send_line_multi(parts)


if __name__ == "__main__":
    main()