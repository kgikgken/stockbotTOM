from __future__ import annotations

import os
from typing import List, Tuple

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
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# データ取得（安全版）
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ============================================================
# スクリーニング
# ============================================================
def run_screening() -> Tuple[list[dict], list[dict]]:
    """
    ユニバースから A/B ランク候補を抽出
    A: score >= 80
    B: 70 <= score < 80
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    A_list: list[dict] = []
    B_list: list[dict] = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        price = float(hist["Close"].iloc[-1])

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(sc),
            "price": price,
        }

        if sc >= 80:
            A_list.append(info)
        elif sc >= 70:
            B_list.append(info)

    A_list = sorted(A_list, key=lambda x: x["score"], reverse=True)
    B_list = sorted(B_list, key=lambda x: x["score"], reverse=True)
    return A_list, B_list


# ============================================================
# IN / OUT アドバイスロジック
# ============================================================
def make_inout_advice(
    mkt_score: int,
    top_picks: List[dict],
    total_pos: float,
    total_asset: float,
    lev: float,
    risk_info: dict,
) -> tuple[str, str, str]:
    """
    今日のIN/OUT判断
    戻り値:
        in_advice: INの方針
        in_size:   ロット感
        out_advice: OUTの方針
    """

    has_big_loss = bool(risk_info.get("has_big_loss", False))
    has_mid_loss = bool(risk_info.get("has_mid_loss", False))
    has_big_gain = bool(risk_info.get("has_big_gain", False))

    # ---- INのベース方針（地合い × 候補）----
    if mkt_score >= 70:
        if len(top_picks) >= 1:
            in_advice = "IN推奨（強地合い×本命候補あり）"
            in_size = "ロット: 中〜大"
        else:
            in_advice = "IN控えめ（候補薄い）"
            in_size = "ロット: 小"
    elif mkt_score >= 60:
        if len(top_picks) >= 1:
            in_advice = "IN可（押し目狙い◯）"
            in_size = "ロット: 小〜中"
        else:
            in_advice = "IN控えめ（候補なし）"
            in_size = "ロット: 極小"
    elif mkt_score >= 50:
        if len(top_picks) >= 1:
            in_advice = "慎重IN（厳選して1〜2銘柄）"
            in_size = "ロット: 小"
        else:
            in_advice = "基本HOLD（新規INは様子見寄り）"
            in_size = "ロット: 0〜極小"
    elif mkt_score >= 40:
        in_advice = "基本HOLD（新規INほぼ見送り）"
        in_size = "ロット: 0〜極小"
    else:
        in_advice = "IN禁止（弱い地合い）"
        in_size = "ロット: 0"

    # ---- レバレッジで微調整 ----
    if lev >= 2.3:
        in_advice += " / 既にレバ過多のため実質IN禁止"
        in_size = "ロット: 0"
    elif lev >= 1.8:
        in_advice += " / 追加INは控えめ"
        in_size += "（最大でも小ロット）"

    # ---- OUT方針 ----
    if lev > 2.3:
        out_advice = "OUT推奨（レバ2.3倍超→縮小必須）"
    elif has_big_loss and mkt_score <= 55:
        out_advice = "損切り優先（-5%以上ポジの整理推奨）"
    elif has_big_gain and mkt_score < 65:
        out_advice = "利確検討（+10%以上利益銘柄の部分利確）"
    else:
        out_advice = "OUT急ぎ不要（通常管理）"

    return in_advice, in_size, out_advice


# ============================================================
# レポート生成
# ============================================================
def build_report() -> str:
    today = jst_today_str()

    # ---- 地合い ----
    mkt = calc_market_score()  # {"score": int, "comment": str}
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    # ---- セクタートップ ----
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join(
            [f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)]
        )
    else:
        sector_text = "算出不可（データ不足）"

    # ---- スクリーニング ----
    A_list, B_list = run_screening()
    top_picks = A_list[:3]  # 今日の主力候補

    # ---- ポジション ----
    pos_df = load_positions()
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    # ---- IN/OUT 判定 ----
    in_adv, in_size, out_adv = make_inout_advice(
        mkt_score, top_picks, total_pos, total_asset, lev, risk_info
    )

    # ---- assemble ----
    lines: list[str] = []
    lines.append(f"📅 {today} stockbotTOM 日報\n")

    # 結論
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append("")

    # IN/OUT
    lines.append("◆ 売買判断（IN/OUT）")
    lines.append(f"- IN: {in_adv} / {in_size}")
    lines.append(f"- OUT: {out_adv}")
    lines.append("")

    # セクター
    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    # Core A
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if not A_list:
        lines.append("本命Aランクなし。")
    else:
        for r in A_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  "
                f"Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
    lines.append("")

    # Core B
    lines.append("◆ Core候補 Bランク（押し目候補）")
    if not B_list:
        lines.append("Bランク候補なし。")
    else:
        for r in B_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  "
                f"Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
    lines.append("")

    # ポジション分析（ここで各銘柄のTP/SLも出ている）
    lines.append("◆ ポジション分析")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# LINE 送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（printのみ）")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信に失敗:", e)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()