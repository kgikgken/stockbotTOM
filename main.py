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

# 一次スクリーニングで見る最大候補数
FIRST_STAGE_MAX = 50  # 全銘柄からまず 50 まで
SECOND_STAGE_MAX = 10  # その中から 10 銘柄まで詳細評価
FINAL_MAX_NAMES = 3    # 最終的に LINE に出すのは最大 3 銘柄


# ============================================================
# 日付・イベント系
# ============================================================
def jst_today_date() -> datetime.date:
    """JST の「今日」の date を返す"""
    return datetime.now(timezone(timedelta(hours=9))).date()


# 重要イベント（必要になったらここに追加）
EVENT_CALENDAR: List[Dict[str, str]] = [
    # 例:
    # {"date": "2025-12-10", "label": "米CPI", "kind": "macro"},
    # {"date": "2025-12-13", "label": "FOMC", "kind": "macro"},
    # {"date": "2025-12-18", "label": "NVDA 決算", "kind": "mega-tech"},
]


def build_event_warnings(today: datetime.date) -> List[str]:
    """イベント接近時の警告メッセージ"""
    warns: List[str] = []
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


# ============================================================
# Universe & データ取得
# ============================================================
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

    # earnings_date があればパース
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


# ============================================================
# レバレッジ & 建て玉
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    """
    地合いに応じたレバ設定
    「1年後の資産最大化」基準でバランス調整
    """
    if mkt_score >= 70:
        return 2.0, "強め（押し目＋一部ブレイク可）"
    if 60 <= mkt_score < 70:
        return 1.5, "やや強め（押し目メイン）"
    if 50 <= mkt_score < 60:
        return 1.3, "標準（押し目のみ）"
    if 40 <= mkt_score < 50:
        return 1.1, "やや守り（ロット控えめ）"
    return 1.0, "守り（新規最小ロット）"


def calc_max_gross(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(total_asset * lev)


# ============================================================
# セクター強度マップ
# ============================================================
def build_sector_strength_map() -> Tuple[str, str, Dict[str, float], str]:
    """
    セクター情報を取得し、
    - 表示用テキスト
    - セクター→強度スコア のマップ
    を返す
    """
    secs = top_sectors_5d()
    if not secs:
        return "算出不可（データ不足）", "", {}, ""

    lines = []
    sector_score_map: Dict[str, float] = {}
    n = len(secs)
    for i, (name, chg) in enumerate(secs):
        rank = i + 1
        lines.append(f"{rank}. {name} ({chg:+.2f}%)")
        # rank が高いほど +、上位ほどスコア高く
        base = (n - rank + 1)
        # 騰落率も少し加味
        sc = base * 1.5 + max(chg, 0.0) * 0.3
        sector_score_map[str(name)] = float(sc)

    sector_text = "\n".join(lines)
    return sector_text, "ok", sector_score_map, ""


# ============================================================
# IN目安・TP/SL・二次スコア用ヘルパ
# ============================================================
def calc_in_price_for_swing(close: pd.Series) -> float:
    """
    3〜10日スイング用の「入りやすくて勝ちやすい押し目」を計算
    - 直近安値からあまり離れすぎない
    - MA10 / MA20 も考慮
    """
    close = close.astype(float)
    price = float(close.iloc[-1])

    # 直近 5〜10 日の安値
    recent_low_5 = float(close.iloc[-5:].min()) if len(close) >= 5 else price
    recent_low_10 = float(close.iloc[-10:].min()) if len(close) >= 10 else recent_low_5
    base_low = max(recent_low_5, recent_low_10 * 0.98)

    ma10 = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else price
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else ma10

    # 「押し目」は MA 帯と直近安値の間あたり
    raw_in = max(base_low * 1.01, min(ma10, ma20))

    # 深すぎる押し目は 3〜10日スイングには不向きなのでクリップ
    # 現値からの乖離を -6%〜-0.5% に制限
    in_price = float(np.clip(raw_in, price * 0.94, price * 0.995))
    return in_price


def calc_tp_sl_for_candidate(
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

    # ベースはボラティリティで決定
    if v < 0.02:
        tp = 0.08
        sl = -0.03
    elif v > 0.06:
        tp = 0.12
        sl = -0.06
    else:
        tp = 0.10
        sl = -0.04

    # 地合いで微調整（強いときは少し伸ばす、弱いときは守り）
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


def calc_trend_score(close: pd.Series) -> float:
    """
    トレンド強度スコア（-15〜+15 くらいを想定）
    - MA20/50 の傾き
    - 現値が MA の上か下か
    """
    close = close.astype(float)

    if len(close) < 25:
        return 0.0

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    ma20_last = float(ma20.iloc[-1])
    ma20_prev = float(ma20.iloc[-5]) if len(ma20.dropna()) >= 5 else ma20_last
    slope20 = (ma20_last / ma20_prev - 1.0) if ma20_prev > 0 else 0.0

    if len(ma50.dropna()) >= 10:
        ma50_last = float(ma50.iloc[-1])
        ma50_prev = float(ma50.iloc[-10])
        slope50 = (ma50_last / ma50_prev - 1.0) if ma50_prev > 0 else 0.0
    else:
        slope50 = 0.0

    price = float(close.iloc[-1])

    score = 0.0

    # MA20 の傾き：1% 上昇で +4 点くらい
    score += np.clip(slope20 / 0.01 * 4.0, -12.0, 12.0)
    # MA50 はもう少し弱めに
    score += np.clip(slope50 / 0.01 * 2.0, -6.0, 6.0)

    # 現値が MA の上か下か
    if ma20_last > 0:
        if price > ma20_last * 1.01:
            score += 2.0
        elif price < ma20_last * 0.99:
            score -= 2.0

    if len(ma50.dropna()) > 0 and ma50.iloc[-1] > 0:
        ma50_last = float(ma50.iloc[-1])
        if price > ma50_last * 1.01:
            score += 2.0
        elif price < ma50_last * 0.99:
            score -= 2.0

    return float(np.clip(score, -15.0, 15.0))


def calc_volume_score(df: pd.DataFrame) -> float:
    """
    出来高スコア（0〜10 目安）
    直近5日 vs 直近20日 の出来高を比較
    """
    if "Volume" not in df.columns:
        return 0.0

    vol = df["Volume"].astype(float).dropna()
    if len(vol) < 5:
        return 0.0

    v5 = float(vol.iloc[-5:].mean())
    v20 = float(vol.iloc[-20:].mean()) if len(vol) >= 20 else float(vol.mean())
    if v20 <= 0:
        return 0.0

    ratio = v5 / v20
    if ratio >= 1.8:
        return 10.0
    if ratio >= 1.4:
        return 7.0
    if ratio >= 1.1:
        return 4.0
    if ratio >= 0.8:
        return 2.0
    return 0.0


def calc_in_distance_score(price: float, in_price: float) -> Tuple[float, float]:
    """
    現値と IN 目安の距離
    - 近いほどスコア高い
    - 遠すぎるものは候補から落とすための gap も返す
    """
    if not (np.isfinite(price) and price > 0 and np.isfinite(in_price) and in_price > 0):
        return 0.0, 1.0

    gap = (price - in_price) / price  # 0〜正 で「どれだけ下に待つか」
    if gap < 0:
        # IN目安が現値より上にある場合は基本ナシ扱い
        return 0.0, float(gap)

    if gap <= 0.01:
        score = 10.0
    elif gap <= 0.03:
        score = 6.0
    elif gap <= 0.05:
        score = 3.0
    else:
        score = 0.0

    return float(score), float(gap)


def calc_ai_final_score(
    base_score: float,
    trend_score: float,
    sector_score: float,
    volume_score: float,
    in_score: float,
) -> float:
    """
    最終 AI スコア（0〜100）
    「1年後の資産成長」を最大化するための
    ・ベーススコア（テクニカル総合）
    ・トレンド
    ・セクター
    ・需給
    ・INしやすさ
    の加重平均
    """
    if not np.isfinite(base_score):
        base_score = 0.0

    # base_score は 0〜100 想定
    s = (
        base_score * 0.6
        + trend_score * 1.0
        + sector_score * 1.0
        + volume_score * 0.8
        + in_score * 0.8
    )
    return float(np.clip(s, 0.0, 100.0))


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening(
    today: datetime.date,
    mkt_score: int,
    total_asset: float,
    sector_score_map: Dict[str, float],
) -> Tuple[List[Dict], List[Dict]]:
    """
    一次（10銘柄まで）＋二次 AI スコアで 3銘柄に絞る前の A/B リストを返す
    戻り値: (A_list, B_list)
    """
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return [], []

    candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算日前後 ±EARNINGS_EXCLUDE_DAYS は除外
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        # sector / industry_big のどちらか
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_sc = score_stock(hist)
        if base_sc is None or not np.isfinite(base_sc):
            continue
        base_sc = float(base_sc)

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])

        # ボラ
        ret = close.pct_change(fill_method=None)
        vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 20 else np.nan

        # IN 目安
        in_price = calc_in_price_for_swing(close)
        in_score, gap = calc_in_distance_score(price, in_price)

        # 「今日から 3〜10日スイングで IN 不可能」なやつは除外
        # 例えば gap > 6% とかは “今日は様子見” 扱い
        if gap > 0.06:
            continue

        # トレンドスコア
        trend_sc = calc_trend_score(close)

        # 出来高スコア
        vol_sc = calc_volume_score(hist)

        # セクタースコア
        sec_sc = float(sector_score_map.get(sector, 0.0))

        # TP / SL
        tp_pct, sl_pct, tp_price, sl_price = calc_tp_sl_for_candidate(
            price=price,
            vola20=vola20,
            mkt_score=mkt_score,
        )

        # 最終 AI スコア
        ai_sc = calc_ai_final_score(
            base_score=base_sc,
            trend_score=trend_sc,
            sector_score=sec_sc,
            volume_score=vol_sc,
            in_score=in_score,
        )

        candidates.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": base_sc,   # ベーススコア
                "ai_score": ai_sc,  # 最終 AI スコア
                "price": price,
                "in_price": in_price,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
            }
        )

    if not candidates:
        return [], []

    # 一次：ベーススコアで 50 銘柄まで絞る
    candidates.sort(key=lambda x: x["score"], reverse=True)
    first_stage = candidates[:FIRST_STAGE_MAX]

    # 二次：AIスコアで 10 銘柄に絞る
    first_stage.sort(key=lambda x: x["ai_score"], reverse=True)
    second_stage = first_stage[:SECOND_STAGE_MAX]

    # A/B 分類（ベーススコア基準）
    A_list: List[Dict] = []
    B_list: List[Dict] = []
    for c in second_stage:
        if c["score"] >= 85.0:
            A_list.append(c)
        elif c["score"] >= 80.0:
            B_list.append(c)

    # AIスコアで再ソート（表示時は「最終スコア順」）
    A_list.sort(key=lambda x: x["ai_score"], reverse=True)
    B_list.sort(key=lambda x: x["ai_score"], reverse=True)

    return A_list, B_list


def select_primary_targets(
    A_list: List[Dict],
    B_list: List[Dict],
    max_names: int = FINAL_MAX_NAMES,
) -> Tuple[List[Dict], List[Dict]]:
    """
    表示用の “推奨3銘柄” を決める
    - Aが3つ以上 → A上位3のみ表示、Bは表示しない（内部候補としては保持可）
    - Aが1〜2 → A全部 + Bから不足分
    - Aが0 → Bから最大 max_names
    """
    if len(A_list) >= max_names:
        return A_list[:max_names], []

    if len(A_list) > 0:
        need = max_names - len(A_list)
        primary = A_list + B_list[:need]
        rest_B = B_list[need:]
        return primary, rest_B

    # Aゼロ → Bからだけ
    return B_list[:max_names], B_list[max_names:]


# ============================================================
# レポート構築
# ============================================================
def build_core_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    # レバ & MAX建て玉
    rec_lev, lev_label = recommend_leverage(mkt_score)
    est_asset = float(total_asset) if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    max_gross = calc_max_gross(est_asset, rec_lev)

    # セクター
    sector_text, _, sector_score_map, _ = build_sector_strength_map()

    # スクリーニング（一次10 → AI最終3）
    A_list, B_list = run_screening(
        today=today_date,
        mkt_score=mkt_score,
        total_asset=est_asset,
        sector_score_map=sector_score_map,
    )

    primary, rest_B = select_primary_targets(A_list, B_list, max_names=FINAL_MAX_NAMES)

    # イベント警告
    events = build_event_warnings(today_date)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_label}）")
    lines.append(f"- 推定運用資産ベース: 約{int(est_asset):,}円")
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    lines.append("◆ 今日のイベント・警戒情報")
    if events:
        for ev in events:
            lines.append(f"- {ev}")
    else:
        lines.append("- 特筆すべきイベントなし（通常モード）")
    lines.append("")

    # Core A
    lines.append(f"◆ Core候補 Aランク（本命押し目・最大{FINAL_MAX_NAMES}銘柄）")
    if not primary:
        lines.append("本命Aランク候補なし（今日は無理IN禁止寄り）。")
    else:
        for r in primary:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f} 現値:{r['price']:.1f}"
            )
            lines.append(
                f"    ・IN目安: {r['in_price']:.1f}\n"
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）\n"
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）"
            )
            lines.append("")

    # Core B
    lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ）")
    if len(A_list) >= FINAL_MAX_NAMES:
        lines.append(f"Aランク{FINAL_MAX_NAMES}銘柄が揃っているため、Bランク表示は省略。")
    else:
        used_B = rest_B if primary else B_list
        if not used_B:
            lines.append("Bランク候補なし。")
        else:
            for r in used_B[:10]:
                lines.append(
                    f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f} 現値:{r['price']:.1f}"
                )

    lines.append("")
    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍")
    lines.append(f"- 今日のMAX建て玉: 約{int(max_gross):,}円")

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


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（printのみ）")
        print(text)
        return

    try:
        res = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", res.status_code, res.text)
    except Exception as e:
        print("[ERROR] LINE送信に失敗:", e)
        print(text)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い
    mkt = calc_market_score()

    # ポジション
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

    # ログ出力
    print(core_report)
    print("\n" + "=" * 40 + "\n")
    print(pos_report)

    # LINE 2通に分割して送信（Worker は {"text": "..."} をそのまま push）
    send_line(core_report)
    send_line(pos_report)


if __name__ == "__main__":
    main()