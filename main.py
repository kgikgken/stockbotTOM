from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
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
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

# スコアの閾値
A_MIN_SCORE = 85.0   # 本命 A ランク
B_MIN_SCORE = 80.0   # 押し目 B ランク

# 最大採用銘柄数（A＋B 合計）
MAX_PICKS = 5


# ============================================================
# 汎用ヘルパー
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    """安全に株価履歴を取得（失敗時 None）"""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def calc_vola20(hist: pd.DataFrame) -> float:
    """20日ボラ（終値ベースの標準偏差）"""
    close = hist["Close"]
    ret = close.pct_change()
    vola20 = ret.rolling(20).std().iloc[-1]
    if vola20 is None or not np.isfinite(vola20):
        return 0.0
    return float(vola20)


def calc_tp_sl_for_candidate(score: float, vola: float) -> Tuple[float, float]:
    """
    スクリーニング銘柄用の利確/損切り目安（％）
    score と vola からざっくり決める。
    戻り値は (tp_pct, sl_pct) で、例: 0.08, -0.04
    """
    vol_abs = abs(float(vola))

    # ボラ別ベースライン
    if vol_abs < 0.02:
        tp = 0.06
        sl = -0.03
    elif vol_abs < 0.04:
        tp = 0.08
        sl = -0.04
    else:
        tp = 0.12
        sl = -0.06

    # スコアが高いものは利確を少し伸ばす
    if score >= 90:
        tp += 0.02
    elif score < 83:
        tp -= 0.01

    # 変な値はクリップ
    tp = float(np.clip(tp, 0.04, 0.15))
    sl = float(np.clip(sl, -0.08, -0.02))

    return tp, sl


def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    """
    地合いスコアから推奨レバ＆コメントを返す
    """
    if mkt_score >= 70:
        return 1.8, "強め（押し目＋一部ブレイク可）"
    if mkt_score >= 60:
        return 1.5, "やや強め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "標準（押し目のみ）"
    if mkt_score >= 40:
        return 1.1, "やや守り（ロット控えめ）"
    return 1.0, "守り（新規最小ロット）"


def allocate_lots_for_A(
    A_list: List[Dict[str, float]],
    est_asset: float,
    rec_lev: float,
) -> None:
    """
    Aランク銘柄に推奨ロット（100株単位）を埋め込む。
    est_asset: 推定運用資産
    rec_lev: 推奨レバ
    """
    if not A_list or est_asset <= 0 or rec_lev <= 0:
        return

    target_gross = est_asset * rec_lev
    n = len(A_list)
    if n <= 0:
        return

    per_stock_budget = target_gross / n

    for r in A_list:
        price = float(r["price"])
        if price <= 0:
            r["lot"] = 0
            continue

        # 100株単位
        raw = per_stock_budget / (price * 100.0)
        lots = int(raw)
        if lots < 1:
            lots = 1
        r["lot"] = lots * 100


# ============================================================
# スクリーニング
# ============================================================
def run_screening() -> Tuple[List[Dict], List[Dict]]:
    """
    スクリーニング実行
    戻り値: (A_list, B_list)
    ただし A_list, B_list は「候補の全体リスト」（絞り込み前）
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    all_A: List[Dict] = []
    all_B: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue
        sc = float(sc)

        price = float(hist["Close"].iloc[-1])
        vola20 = calc_vola20(hist)
        tp_pct, sl_pct = calc_tp_sl_for_candidate(sc, vola20)

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": sc,
            "price": price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
        }

        if sc >= A_MIN_SCORE:
            all_A.append(info)
        elif sc >= B_MIN_SCORE:
            all_B.append(info)

    # スコア順でソート
    all_A.sort(key=lambda x: x["score"], reverse=True)
    all_B.sort(key=lambda x: x["score"], reverse=True)

    return all_A, all_B


def select_top_candidates(
    all_A: List[Dict],
    all_B: List[Dict],
    max_picks: int = MAX_PICKS,
) -> Tuple[List[Dict], List[Dict]]:
    """
    A優先で最大 max_picks 銘柄までに絞り込む。
    - まず A から max_picks まで
    - A が max_picks 未満なら、残りを B で補充
    """
    # まず A
    A_sel = list(all_A[:max_picks])
    remain = max_picks - len(A_sel)

    if remain > 0:
        B_sel = list(all_B[:remain])
    else:
        B_sel = []

    return A_sel, B_sel


# ============================================================
# レポート生成
# ============================================================
def build_report() -> str:
    today = jst_today_str()

    # ---- 地合い ----
    mkt = calc_market_score()
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = mkt.get("comment", "中立")

    rec_lev, lev_comment = recommend_leverage(mkt_score)

    # ---- セクター ----
    secs = top_sectors_5d()
    if secs:
        sector_lines = [
            f"{i+1}. {s[0]} ({s[1]:+.2f}%)"
            for i, s in enumerate(secs)
        ]
        sector_text = "\n".join(sector_lines)
    else:
        sector_text = "算出できるセクターデータがありません。"

    # ---- スクリーニング ----
    all_A, all_B = run_screening()
    A_list, B_list = select_top_candidates(all_A, all_B, MAX_PICKS)

    # ---- ポジション ----
    pos_df = load_positions(POSITIONS_PATH)
    # pos_text: str, total_asset: float, total_pos: float, lev: float, risk_info: dict
    pos_text, total_asset, total_pos, cur_lev, risk_info = analyze_positions(pos_df)

    # 推定運用資産（なければデフォルト200万）
    est_asset = float(total_asset) if total_asset and np.isfinite(total_asset) else 2_000_000.0

    # Aランクにロット割り当て
    allocate_lots_for_A(A_list, est_asset, rec_lev)

    # ========================================================
    # 本文構築
    # ========================================================
    lines: List[str] = []

    lines.append(f"📅 {today} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}） / 目安: Aランク最大{MAX_PICKS}銘柄")
    lines.append(f"- 推定運用資産ベース: 約{int(est_asset):,}円")
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    # 簡易イベント（あとで本格化するときにここを強化）
    lines.append("◆ 今日のイベント・警戒情報")
    lines.append("- 特筆すべきイベントなし（通常モード）")
    lines.append("")

    # ---- Core A ----
    lines.append(f"◆ Core候補 Aランク（本命押し目・最大{MAX_PICKS}銘柄）")
    if not A_list:
        lines.append("Aランク該当なし。無理な新規INは控える。")
    else:
        for r in A_list:
            ticker = r["ticker"]
            name = r["name"]
            score = r["score"]
            price = r["price"]
            tp_pct = r["tp_pct"]
            sl_pct = r["sl_pct"]
            tp_price = price * (1.0 + tp_pct)
            sl_price = price * (1.0 + sl_pct)
            lot = int(r.get("lot", 0))

            lines.append(
                f"- {ticker} {name}  Score:{score:.1f}  現値:{price:.1f}"
            )
            lines.append(
                f"    ・IN目安: {price:.1f}"
            )
            lines.append(
                f"    ・利確目安: {tp_pct*100:.1f}%（{tp_price:.1f}）"
            )
            lines.append(
                f"    ・損切り目安: {sl_pct*100:.1f}%（{sl_price:.1f}）"
            )
            if lot > 0:
                lines.append(
                    f"    ・推奨ロット: {lot}株"
                )
            lines.append("")

    # ---- Core B ----
    lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ）")
    if not B_list:
        if A_list:
            lines.append("Aランクで枠が埋まっているため、Bランク表示は省略。")
        else:
            lines.append("Bランク候補なし。")
    else:
        for r in B_list:
            ticker = r["ticker"]
            name = r["name"]
            score = r["score"]
            price = r["price"]
            tp_pct = r["tp_pct"]
            sl_pct = r["sl_pct"]
            tp_price = price * (1.0 + tp_pct)
            sl_price = price * (1.0 + sl_pct)

            lines.append(
                f"- {ticker} {name}  Score:{score:.1f}  現値:{price:.1f}"
            )
            lines.append(
                f"    ・利確目安: {tp_pct*100:.1f}%（{tp_price:.1f}） / 損切り目安: {sl_pct*100:.1f}%（{sl_price:.1f}）"
            )
            lines.append("")

    # ---- ポジション分析 ----
    lines.append("")
    lines.append(f"📊 {today} ポジション分析")
    lines.append("")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    """
    Cloudflare Worker 経由で LINE へ送信。
    WORKER_URL には Worker の URL を入れておく。
    Worker 側では { "text": "..."} を受け取って
    あなたの userId 宛に push する実装にしてある前提。
    """
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（print のみ）")
        print(text)
        return

    try:
        res = requests.post(WORKER_URL, json={"text": text}, timeout=20)
        print("[LINE RESULT]", res.status_code, res.text)
    except Exception as e:
        print("[ERROR] LINE送信に失敗:", e)


# ============================================================
# entry point
# ============================================================
def main() -> None:
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()