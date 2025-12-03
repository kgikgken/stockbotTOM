from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


# ======================================
# 基本設定
# ======================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算日フィルタ（±n日を除外）
EARNINGS_EXCLUDE_DAYS = 3

# A/B スコア閾値
A_MIN_SCORE = 85.0
B_MIN_SCORE = 80.0

# 1日に見る銘柄数（A優先で最大3銘柄）
MAX_NAMES = 3

# 1通あたりの最大文字数（LINE 制限対策）
LINE_CHUNK_SIZE = 3500


# ======================================
# 日付関係
# ======================================
def jst_today_date() -> datetime.date:
    """JST の今日の日付を返す"""
    return datetime.now(timezone(timedelta(hours=9))).date()


# ======================================
# Universe 読み込み & 決算フィルタ
# ======================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] universe file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read universe: {e}")
        return None

    if "ticker" not in df.columns:
        print("[WARN] universe has no 'ticker' column")
        return None

    df["ticker"] = df["ticker"].astype(str)

    # earnings_date を date にパースしておく
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    """決算日 ±EARNINGS_EXCLUDE_DAYS に入っていれば True"""
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


# ======================================
# 株価データ取得
# ======================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        print(f"[WARN] fetch history failed {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None
    return df


# ======================================
# 市場スコア → 推奨レバ
# ======================================
def recommend_leverage(mkt_score: int) -> float:
    """
    市場スコアから推奨レバを決める。
    少し守備寄りにチューニング。
    """
    if mkt_score >= 75:
        return 1.5
    if mkt_score >= 60:
        return 1.3
    if mkt_score >= 50:
        return 1.1
    return 1.0


# ======================================
# IN 価格ロジック（世界最高トレーダー仕様）
# ======================================
def calc_entry_price(close: pd.Series) -> float:
    """
    「今日 IN できて、かつ勝ちやすい」最強 IN 価格。
    - 上昇トレンド：5日線〜20日線の中腹
    - それ以外：20日線〜50日線の間
    - ただし、現値より“上”にはしない（今日入れない値段はNG）
    """
    close = close.astype(float)
    last = float(close.iloc[-1])

    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])

    if not np.isfinite(ma5) or not np.isfinite(ma20):
        return round(last, 1)

    trend_up = last > ma20 > ma50 if np.isfinite(ma50) else last > ma20

    if trend_up:
        # 強い上昇トレンド → 5日線と20日線の真ん中あたりが理想
        base = (ma5 + ma20) / 2.0
        lower = ma20 * 0.97   # 深くても20日線の3%下まで
        upper = ma5 * 1.02    # 追いかけ過ぎない
        in_price = max(lower, min(base, upper))
    else:
        # トレンド微妙 or 調整深め → 20日線〜50日線ゾーン
        if np.isfinite(ma50):
            base = (ma20 + ma50) / 2.0
            lower = ma50 * 0.98
            upper = ma20 * 1.02
        else:
            base = ma20
            lower = ma20 * 0.95
            upper = ma20 * 1.02
        in_price = max(lower, min(base, upper))

    # 今日入れない price は意味が無いので現値より上にはしない
    in_price = min(in_price, last)

    # 念のため、極端な値をクリップ
    in_price = float(np.clip(in_price, last * 0.85, last * 1.01))

    return round(in_price, 1)


# ======================================
# TP / SL ロジック（ボラ × スコア）
# ======================================
def calc_tp_sl(
    price: float,
    vola20: Optional[float],
    score: float,
) -> Tuple[float, float, float, float]:
    """
    ボラとスコアに応じて利確・損切りを決定。
    戻り値: (tp_pct, sl_pct, tp_price, sl_price)
    """
    if not np.isfinite(price) or price <= 0:
        return 0.0, 0.0, price, price

    v = float(vola20) if vola20 is not None and np.isfinite(vola20) else 0.03

    # ボラ別ベースライン
    if v < 0.015:
        tp = 0.06
        sl = -0.03
    elif v < 0.03:
        tp = 0.08
        sl = -0.04
    elif v < 0.05:
        tp = 0.10
        sl = -0.05
    else:
        tp = 0.12
        sl = -0.06

    # スコアが高い銘柄ほど利確を伸ばす
    if score >= 90:
        tp += 0.02
    elif score < 83:
        tp -= 0.01

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    tp_price = price * (1.0 + tp)
    sl_price = price * (1.0 + sl)

    return tp, sl, tp_price, sl_price


# ======================================
# スクリーニング本体
# ======================================
def run_screening(
    today: datetime.date,
    mkt_score: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Universe 全銘柄に対してスクリーニング。
    戻り値: (A_list, B_list) ※まだ 3銘柄に絞る前
    """
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return [], []

    A_list: List[Dict] = []
    B_list: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算日前後は安全のためスキップ
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 80:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue
        sc = float(sc)

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])

        # ボラ（20日標準偏差）
        ret = close.pct_change()
        vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 20 else None

        # IN 価格
        in_price = calc_entry_price(close)

        # TP / SL
        tp_pct, sl_pct, tp_price, sl_price = calc_tp_sl(price, vola20, sc)

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": sc,
            "price": price,
            "in_price": in_price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

        if sc >= A_MIN_SCORE:
            A_list.append(info)
        elif sc >= B_MIN_SCORE:
            B_list.append(info)

    # スコア順に並べる
    A_list.sort(key=lambda x: x["score"], reverse=True)
    B_list.sort(key=lambda x: x["score"], reverse=True)

    return A_list, B_list


def select_primary(
    A_list: List[Dict],
    B_list: List[Dict],
    max_names: int = MAX_NAMES,
) -> Tuple[List[Dict], List[Dict]]:
    """
    表示用候補を決定。
    - Aが3つ以上 → A上位3だけ表示、Bは表示しない
    - Aが1〜2 → A全て + B で欠け分を補う
    - Aが0 → Bだけから最大3銘柄
    """
    if len(A_list) >= max_names:
        return A_list[:max_names], []

    if len(A_list) > 0:
        need = max_names - len(A_list)
        primary = A_list + B_list[:need]
        rest_B = B_list[need:]
        return primary, rest_B

    # Aゼロ → Bからのみ
    return B_list[:max_names], B_list[max_names:]


# ======================================
# レポート構築
# ======================================
def build_reports() -> Tuple[List[str], float, float]:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # ---- 地合い ----
    mkt = calc_market_score()
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))
    rec_lev = recommend_leverage(mkt_score)

    # ---- ポジション ----
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, cur_lev, risk_info = analyze_positions(pos_df)

    # total_asset が取れなかったときの保険
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    # ---- セクター ----
    secs = top_sectors_5d()
    if secs:
        sec_lines = [
            f"{i+1}. {name} ({chg:+.2f}%)" for i, (name, chg) in enumerate(secs)
        ]
        sector_text = "\n".join(sec_lines)
    else:
        sector_text = "算出できるセクターデータがありません。"

    # ---- スクリーニング ----
    A_all, B_all = run_screening(today=today_date, mkt_score=mkt_score)
    primary, rest_B = select_primary(A_all, B_all, MAX_NAMES)

    # ======================================
    # Part 1: 結論 + セクター + イベント
    # ======================================
    part1_lines: List[str] = []
    part1_lines.append(f"📅 {today_str} stockbotTOM 日報")
    part1_lines.append("")
    part1_lines.append("◆ 今日の結論")
    part1_lines.append(f"- 地合いスコア: {mkt_score}点")
    part1_lines.append(f"- コメント: {mkt_comment}")
    part1_lines.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円")
    part1_lines.append("")
    part1_lines.append("◆ 今日のTOPセクター（5日騰落率）")
    part1_lines.append(sector_text)
    part1_lines.append("")
    part1_lines.append("◆ 今日のイベント・警戒情報")
    part1_lines.append("- 特筆すべきイベントなし（通常モード）")

    part1 = "\n".join(part1_lines)

    # ======================================
    # Part 2: Core候補 A/B
    # ======================================
    part2_lines: List[str] = []
    part2_lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not primary:
        part2_lines.append("本命Aランク候補なし（今日は無理IN禁止寄り）。")
    else:
        for r in primary:
            part2_lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f} 現値:{r['price']:.1f}"
            )
            part2_lines.append(f"    ・IN目安: {r['in_price']:.1f}")
            part2_lines.append(
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）"
            )
            part2_lines.append(
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）"
            )
            part2_lines.append("")

    part2_lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ）")
    if len(A_all) >= MAX_NAMES:
        part2_lines.append("Aランク3銘柄が揃っているため、Bランク表示は省略。")
    else:
        if not B_all:
            part2_lines.append("Bランク候補なし。")
        else:
            # 表示しすぎるとノイズなので上位5だけ
            for r in B_all[:5]:
                part2_lines.append(
                    f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f} 現値:{r['price']:.1f}"
                )

    part2 = "\n".join(part2_lines)

    # ======================================
    # Part 3: 建て玉最大金額
    # ======================================
    max_pos = int(total_asset * rec_lev)

    part3_lines: List[str] = []
    part3_lines.append("◆ 本日の建て玉最大金額")
    part3_lines.append(f"- 推奨レバ: {rec_lev:.1f}倍")
    part3_lines.append(f"- 今日のMAX建て玉: 約{max_pos:,}円")

    part3 = "\n".join(part3_lines)

    # ======================================
    # Part 4: ポジション分析
    # ======================================
    part4_lines: List[str] = []
    part4_lines.append(f"📊 {today_str} ポジション分析")
    part4_lines.append("")
    part4_lines.append("◆ ポジションサマリ")
    part4_lines.append(pos_text.strip())

    part4 = "\n".join(part4_lines)

    return [part1, part2, part3, part4], total_asset, max_pos


# ======================================
# LINE 送信
# ======================================
def send_line_once(text: str) -> None:
    """
    1通分を送信（長文なら分割して複数回 POST）
    """
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定 → print のみ")
        print(text)
        return

    chunks = [
        text[i : i + LINE_CHUNK_SIZE] for i in range(0, len(text), LINE_CHUNK_SIZE)
    ]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)


def send_line_multi(parts: List[str]) -> None:
    for p in parts:
        if p.strip():
            send_line_once(p)


# ======================================
# Entry Point
# ======================================
def main() -> None:
    parts, total_asset, max_pos = build_reports()

    # コンソールにも一応出しておく
    print("\n\n===== PART 1 =====\n")
    print(parts[0])
    print("\n\n===== PART 2 =====\n")
    print(parts[1])
    print("\n\n===== PART 3 =====\n")
    print(parts[2])
    print("\n\n===== PART 4 =====\n")
    print(parts[3])

    # LINE送信
    send_line_multi(parts)


if __name__ == "__main__":
    main()