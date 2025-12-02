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


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算日フィルタ：±N日を除外
EARNINGS_EXCLUDE_DAYS = 3


# ================================
# 日付・イベントまわり
# ================================
def jst_today_date() -> datetime.date:
    """JST の「今日」の date を返す"""
    return datetime.now(timezone(timedelta(hours=9))).date()


# 今後、自動取得を拡張するためのイベントカレンダー
EVENT_CALENDAR: List[Dict[str, str]] = []


def build_event_warnings(today: datetime.date) -> List[str]:
    """重要イベント接近時の警告メッセージ"""
    warns: List[str] = []
    if not EVENT_CALENDAR:
        return warns

    for ev in EVENT_CALENDAR:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        delta = (d - today).days
        # イベントの2日前〜翌日は警告を出す
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")
    return warns


# ================================
# Universe & データ取得
# ================================
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

    # earnings_date を一度だけパースしておく
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


def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        print(f"[WARN] fetch history failed {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None
    return df


# ================================
# レバレッジ & MAX建て玉
# ================================
def calc_target_leverage(mkt_score: int) -> Tuple[float, str]:
    """
    地合いに応じたレバ設定（今は可変）
    """
    if mkt_score >= 70:
        return 2.0, "攻め（Aランク3銘柄フル）"
    if mkt_score >= 50:
        return 1.3, "標準（押し目のみ）"
    if mkt_score >= 40:
        return 1.0, "守り寄り（ロット控えめ）"
    return 0.8, "守り優先（基本は縮小〜様子見）"


def calc_max_position(total_asset: float, lev: float) -> int:
    """
    その日の推定資産と推奨レバから
    “建て玉最大金額” を計算するだけ。
    """
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(total_asset * lev)


# ================================
# 候補銘柄の TP / SL
# ================================
def calc_candidate_tp_sl(
    price: float,
    vola20: Optional[float],
    mkt_score: int,
) -> Tuple[float, float, float, float]:
    """
    スクリーニング候補の利確・損切り
    戻り値: (tp_pct, sl_pct, tp_price, sl_price)
    """
    if not np.isfinite(price) or price <= 0:
        return 0.0, 0.0, price, price

    v = float(vola20) if vola20 is not None and np.isfinite(vola20) else 0.04

    # ボラ別ベースライン
    if v < 0.02:
        tp = 0.08
        sl = -0.03
    elif v > 0.06:
        tp = 0.12
        sl = -0.06
    else:
        tp = 0.10
        sl = -0.04

    # 地合いで微調整
    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    tp_price = price * (1.0 + tp)
    sl_price = price * (1.0 + sl)
    return tp, sl, tp_price, sl_price


# ================================
# スクリーニング本体
# ================================
def run_screening(
    today: datetime.date,
    mkt_score: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    A / B 候補リストを返す
    A: 本命
    B: 押し目候補（Aが足りないときの補欠）
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

        # 決算日前後 ±EARNINGS_EXCLUDE_DAYS は除外
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])

        # ボラ
        ret = close.pct_change(fill_method=None)
        vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 20 else None

        tp_pct, sl_pct, tp_price, sl_price = calc_candidate_tp_sl(
            price, vola20, mkt_score
        )

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(sc),
            "price": price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

        if sc >= 85:
            A_list.append(info)
        elif sc >= 75:
            B_list.append(info)

    A_list.sort(key=lambda x: x["score"], reverse=True)
    B_list.sort(key=lambda x: x["score"], reverse=True)
    return A_list, B_list


def select_primary_targets(
    A_list: List[Dict],
    B_list: List[Dict],
    max_names: int = 3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    表示用の “推奨3銘柄” を決める
    - Aが3つ以上 → A上位3のみ表示、Bは表示しない
    - Aが1〜2 → A全部 + Bから不足分
    - Aが0 → Bから最大 max_names
    """
    if len(A_list) >= max_names:
        return A_list[:max_names], []

    if len(A_list) > 0:
        need = max_names - len(A_list)
        return A_list + B_list[:need], B_list[need:]

    # Aゼロ → Bからだけ
    return B_list[:max_names], B_list[max_names:]


# ================================
# レポート構築
# ================================
def build_core_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
) -> str:
    # 推定運用資産（None / 0 の場合は 200万で代用）
    if total_asset and np.isfinite(total_asset) and total_asset > 0:
        est_asset = float(total_asset)
    else:
        est_asset = 2_000_000.0

    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    target_lev, lev_label = calc_target_leverage(mkt_score)
    max_pos = calc_max_position(est_asset, target_lev)

    # セクター
    secs = top_sectors_5d()
    if secs:
        sec_lines = [
            f"{i + 1}. {name} ({chg:+.2f}%)" for i, (name, chg) in enumerate(secs)
        ]
        sec_text = "\n".join(sec_lines)
    else:
        sec_text = "算出不可（データ不足）"

    # スクリーニング
    A_list, B_list = run_screening(
        today=today_date,
        mkt_score=mkt_score,
    )
    primary, rest_B = select_primary_targets(A_list, B_list, max_names=3)

    # イベント警告
    events = build_event_warnings(today_date)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{target_lev:.1f}倍（{lev_label}）")
    lines.append(f"- 推定運用資産ベース: 約{int(est_asset):,}円")
    if events:
        for ev in events:
            lines.append(ev)
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sec_text)
    lines.append("")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not primary:
        lines.append("本命Aランク候補なし（今日は無理IN禁止寄り）。")
    else:
        for r in primary:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  "
                f"現値:{r['price']:.1f}"
            )
            lines.append(
                f"    ・IN目安: {r['price']:.1f}"
                f"\n    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）"
                f"\n    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）"
            )
    lines.append("")

    lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ）")
    if len(A_list) >= 3:
        lines.append("Aランク3銘柄が揃っているため、Bランク表示は省略。")
    else:
        if not B_list:
            lines.append("Bランク候補なし。")
        else:
            for r in B_list[:10]:
                lines.append(
                    f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  "
                    f"現値:{r['price']:.1f}"
                )
    lines.append("")

    # 本日の建て玉最大金額
    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {target_lev:.1f}倍")
    lines.append(f"- 運用資産ベース: 約{int(est_asset):,}円")
    lines.append(f"- 今日のMAX建て玉: 約{int(max_pos):,}円")

    return "\n".join(lines)


def build_position_report(
    today_str: str,
    pos_text: str,
) -> str:
    lines: List[str] = []
    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip())
    return "\n".join(lines)


# ================================
# LINE送信
# ================================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（printのみ）")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信に失敗:", e)
        print(text)


# ================================
# Entry
# ================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合いスコア
    mkt = calc_market_score()

    # ポジション（推定資産・レバなど含む）
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    # Core & スクリーニング
    core_report = build_core_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
    )

    # ポジションレポート
    pos_report = build_position_report(today_str=today_str, pos_text=pos_text)

    print(core_report)
    print("\n" + "=" * 40 + "\n")
    print(pos_report)

    # LINE 2通で送信
    send_line(core_report)
    send_line(pos_report)


if __name__ == "__main__":
    main()