from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Tuple
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

from utils import (
    load_universe,
    add_indicators,
    extract_metrics,
    calc_market_score,
    calc_sector_strength,
    calc_core_score,
    calc_shortterm_score,
    build_line_message,
    fetch_fundamentals,
    calc_fundamental_score,
)


# ==========================================
# ユーティリティ
# ==========================================


def jst_today() -> date:
    """JSTの今日の日付（dateオブジェクト）を返す。"""
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).date()


def jst_today_str() -> str:
    """JSTの今日の日付を YYYY-MM-DD 文字列で返す。"""
    return jst_today().strftime("%Y-%m-%d")


def fetch_ohlcv(ticker: str, period: str = "80d") -> pd.DataFrame | None:
    """
    yfinanceから日足データを取得する。
    必須カラム: Open, High, Low, Close, Volume
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    return df.tail(80)


def is_in_earnings_window(ticker: str, base_date: date, window: int = 3) -> bool:
    """
    決算日前後の「危険ゾーン」かどうかを判定する。
    - window 日以内に決算日がある場合 True
    - 決算日が取得できない場合は False（＝通常どおりスクリーニング）
    """
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
    except Exception:
        return False

    if cal is None or len(cal) == 0:
        return False

    earnings_date = None

    # yfinance の calendar は銘柄によって形式が違うので、なるべく汎用的に拾う
    try:
        if "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"][0]
            earnings_date = pd.to_datetime(val).date()
        else:
            # 先頭行を日付として解釈するフォールバック
            first_val = cal.iloc[0, 0]
            earnings_date = pd.to_datetime(first_val).date()
    except Exception:
        earnings_date = None

    if earnings_date is None:
        return False

    diff = abs((earnings_date - base_date).days)
    return diff <= window


def load_event_risk(today: date, csv_path: str = "events.csv") -> tuple[int, List[str]]:
    """
    イベントリスクをCSVから読み込み、地合いスコアへの調整値とラベルリストを返す。
    CSV形式: date,label,impact
      - date: YYYY-MM-DD
      - label: 例) FOMC, SQ, 日銀 など
      - impact: -20〜0 程度の整数（地合いスコアの調整値）
    ファイルが無い場合は (0, []) を返す。
    """
    if not os.path.exists(csv_path):
        return 0, []

    adj = 0
    labels: List[str] = []

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return 0, []

    for _, row in df.iterrows():
        d_str = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        impact = row.get("impact", 0)

        if not d_str:
            continue

        try:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
        except Exception:
            continue

        if d != today:
            continue

        try:
            impact_val = int(impact)
        except Exception:
            impact_val = 0

        adj += impact_val
        if label:
            labels.append(label)

    return adj, labels


def load_trade_history(path: str = "data/trade_history.json") -> List[dict]:
    """
    Cloudflare Worker 側で更新している trade_history.json を読み込む。
    無い場合・壊れている場合は空リストを返す。
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        return data
    return []


def _format_yen(v: float) -> str:
    sign = "+" if v > 0 else "-" if v < 0 else "±"
    return f"{sign}{int(round(abs(v)))}円"


def calc_pnl_stats(history: List[dict], today_str: str) -> dict:
    """
    trade_history.json から
    - 本日の損益
    - 今月の成績
    - 最大DD
    などを計算する。
    """
    if not history:
        return {
            "today_profit": 0,
            "month_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "total_profit": 0,
            "avg_profit": 0,
            "max_dd": 0,
        }

    month_prefix = today_str[:7]

    today_profit = 0
    month_trades: List[dict] = []

    for t in history:
        ts = str(t.get("timestamp", ""))
        if len(ts) < 10:
            continue
        d_str = ts[:10]

        try:
            profit = float(t.get("profit", 0))
        except Exception:
            profit = 0.0

        if d_str == today_str:
            today_profit += profit

        if d_str.startswith(month_prefix):
            month_trades.append(t)

    trades = len(month_trades)
    if trades == 0:
        return {
            "today_profit": today_profit,
            "month_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "total_profit": 0,
            "avg_profit": 0,
            "max_dd": 0,
        }

    wins = sum(1 for t in month_trades if t.get("profit", 0) > 0)
    losses = sum(1 for t in month_trades if t.get("profit", 0) < 0)
    total_profit = sum(float(t.get("profit", 0)) for t in month_trades)
    avg_profit = total_profit / trades if trades > 0 else 0.0
    win_rate = wins / trades * 100.0 if trades > 0 else None

    # 最大DDは new_equity のピークからの落ち幅で計算
    peak = None
    max_dd = 0.0
    # timestamp 順にソートして計算
    sorted_trades = sorted(
        month_trades,
        key=lambda t: str(t.get("timestamp", "")),
    )
    for t in sorted_trades:
        eq = t.get("new_equity", None)
        try:
            eq = float(eq)
        except Exception:
            continue
        if peak is None or eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    return {
        "today_profit": today_profit,
        "month_trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "avg_profit": avg_profit,
        "max_dd": max_dd,
    }


def build_pnl_block(stats: dict) -> str:
    """
    損益統計から表示用ブロック文字列を生成。
    """
    today_profit = stats.get("today_profit", 0)
    trades = stats.get("month_trades", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    win_rate = stats.get("win_rate", None)
    total_profit = stats.get("total_profit", 0)
    avg_profit = stats.get("avg_profit", 0)
    max_dd = stats.get("max_dd", 0)

    lines: List[str] = []
    lines.append("◆ トレード成績（自動集計）")
    lines.append(f"本日の損益：{_format_yen(today_profit)}")
    lines.append("")
    lines.append("【今月の成績】")
    lines.append(f"取引回数：{trades} 回")
    if win_rate is None:
        lines.append("勝率：-")
    else:
        lines.append(f"勝率：{win_rate:.1f}%")
    lines.append(f"平均損益：{_format_yen(avg_profit)}")
    lines.append(f"月間総損益：{_format_yen(total_profit)}")
    if max_dd == 0:
        dd_text = "0円"
    else:
        dd_text = f"-{int(round(max_dd))}円"
    lines.append(f"最大DD：{dd_text}")

    return "\n".join(lines)


# ==========================================
# メインロジック
# ==========================================


def screen_universe() -> str:
    """
    ユニバース全体をスクリーニングして、
    LINE通知用の1本のテキストを返す。
    """
    today_d = jst_today()
    today = today_d.strftime("%Y-%m-%d")

    # イベントリスク読み込み
    event_adj, event_labels = load_event_risk(today_d)

    # 地合いスコア（マクロ）
    base_market_score = calc_market_score()
    market_score = int(max(0, min(100, base_market_score + event_adj)))

    universe = load_universe()
    if universe.empty:
        return f"📅 {today} stockbotTOM 日報\n\nユニバースが空です。universe_jpx.csv を確認してください。"

    core_candidates: List[Dict] = []
    short_candidates: List[Dict] = []
    sector_strength_map: Dict[str, int] = {}

    for _, row in universe.iterrows():
        ticker: str = str(row["ticker"])
        name: str = str(row.get("name", ticker))
        sector: str = str(row.get("sector", "その他"))

        # 0. 決算日前後の危険ゾーンはスクリーニングから除外（デフォルト±3日）
        try:
            if is_in_earnings_window(ticker, base_date=today_d, window=3):
                continue
        except Exception:
            # 取得に失敗しても落ちないようにする
            pass

        # 1. 日足取得
        ohlcv = fetch_ohlcv(ticker)
        if ohlcv is None or len(ohlcv) < 40:
            continue

        # 2. インジケータ付与
        ohlcv = add_indicators(ohlcv)

        # 3. 指標抽出
        metrics = extract_metrics(ohlcv)
        if metrics is None:
            continue

        # 4. ハードフィルタ
        turnover_avg20 = metrics.get("turnover_avg20", float("nan"))
        if (not np.isfinite(turnover_avg20)) or turnover_avg20 < 1e8:
            continue

        # 5. ファンダ取得（時価総額などもここから）
        fdict = fetch_fundamentals(ticker)
        mcap = fdict.get("market_cap", float("nan"))
        if (not np.isfinite(mcap)) or mcap < 2e10:
            continue

        fscore = calc_fundamental_score(fdict)

        # 6. スコア計算
        if sector not in sector_strength_map:
            sector_strength_map[sector] = calc_sector_strength(sector)
        sec_strength = sector_strength_map[sector]

        core_score, core_comment = calc_core_score(
            market_score=market_score,
            sector_strength=sec_strength,
            metrics=metrics,
            fundamental_score=fscore,
        )
        short_score, short_comment = calc_shortterm_score(
            market_score=market_score,
            sector_strength=sec_strength,
            metrics=metrics,
        )

        price = metrics.get("close", float("nan"))
        price_int = int(round(price)) if np.isfinite(price) else 0

        # 7. Core / ShortTerm 候補に追加（スコア75以上）
        if core_score >= 75:
            core_candidates.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "score": core_score,
                    "comment": core_comment,
                    "price": price_int,
                }
            )

        if short_score >= 75:
            short_candidates.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "score": short_score,
                    "comment": short_comment,
                    "price": price_int,
                }
            )

    # スコア順にソート
    core_candidates.sort(key=lambda x: x["score"], reverse=True)
    short_candidates.sort(key=lambda x: x["score"], reverse=True)

    # sector_strength_map が空でも build_line_message は動くが、
    # フォールバックとして候補から埋める
    if not sector_strength_map:
        sectors = sorted({r["sector"] for r in core_candidates + short_candidates})
        for s in sectors:
            sector_strength_map[s] = calc_sector_strength(s)

    # メインメッセージ生成
    base_message = build_line_message(
        today=today,
        market_score=market_score,
        core_list=core_candidates,
        short_list=short_candidates,
        sector_strength_map=sector_strength_map,
    )

    # イベント情報を先頭に付ける
    if event_labels:
        ev_lines = ["◆ 今日のイベントリスク"]
        for lbl in event_labels:
            ev_lines.append(f"- {lbl}")
        ev_lines.append(f"- イベント補正: {event_adj}点（地合いスコアに反映済み）")
        ev_block = "\n".join(ev_lines)
        message = ev_block + "\n\n" + base_message
    else:
        message = base_message

    # トレード成績ブロックを末尾に追加
    history = load_trade_history()
    stats = calc_pnl_stats(history, today_str=today)
    pnl_block = build_pnl_block(stats)
    message = message + "\n\n" + pnl_block

    return message


def main() -> None:
    text = screen_universe()
    print(text)


if __name__ == "__main__":
    main()
