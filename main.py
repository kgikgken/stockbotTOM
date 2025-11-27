import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List

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
)


# ==========================================
# ユーティリティ
# ==========================================


def jst_today_str() -> str:
    """JSTの今日の日付を YYYY-MM-DD 文字列で返す。"""
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y-%m-%d")


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

    # 念のため直近だけに絞る
    return df.tail(80)


def fetch_market_cap(ticker: str) -> float:
    """
    時価総額を yfinance から取得。
    取れない場合は NaN を返す。
    """
    try:
        info = yf.Ticker(ticker).fast_info
        mc = getattr(info, "market_cap", None)
        if mc is None:
            # fast_info で取れなければ info も試す
            info2 = yf.Ticker(ticker).info
            mc = info2.get("marketCap")
    except Exception:
        mc = None

    try:
        return float(mc) if mc is not None else float("nan")
    except Exception:
        return float("nan")


# ==========================================
# メインロジック
# ==========================================


def screen_universe() -> str:
    """
    ユニバース全体をスクリーニングして、
    LINE通知 Ver2.0 フォーマットの1本のテキストを返す。
    """
    today = jst_today_str()

    # ユニバース読み込み（ticker, name, sector を持つ DataFrame）
    universe = load_universe()
    if universe.empty:
        return f"📅 {today} stockbotTOM 日報\n\nユニバースが空です。universe_jpx.csv を確認してください。"

    # 地合い（今は仮で固定値。あとで差し替え可）
    market_score = calc_market_score()

    core_candidates: List[Dict] = []
    short_candidates: List[Dict] = []

    # 全セクターの強度をキャッシュしておく（今は全部50）
    sector_strength_map: Dict[str, int] = {}

    for _, row in universe.iterrows():
        ticker: str = str(row["ticker"])
        name: str = str(row.get("name", ticker))
        sector: str = str(row.get("sector", "その他"))

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
        # 4-1. 売買代金 1億円未満は除外
        turnover_avg20 = metrics.get("turnover_avg20", float("nan"))
        if (not np.isfinite(turnover_avg20)) or turnover_avg20 < 1e8:
            continue

        # 4-2. 時価総額 200億未満は除外
        mcap = fetch_market_cap(ticker)
        if (not np.isfinite(mcap)) or mcap < 2e10:
            continue

        # 5. スコア計算
        if sector not in sector_strength_map:
            sector_strength_map[sector] = calc_sector_strength(sector)
        sec_strength = sector_strength_map[sector]

        core_score, core_comment = calc_core_score(
            market_score=market_score,
            sector_strength=sec_strength,
            metrics=metrics,
        )
        short_score, short_comment = calc_shortterm_score(
            market_score=market_score,
            sector_strength=sec_strength,
            metrics=metrics,
        )

        price = metrics.get("close", float("nan"))
        price_int = int(round(price)) if np.isfinite(price) else 0

        # 6. Core / ShortTerm 候補に追加（スコア75以上）
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

    # 7. スコア順にソート
    core_candidates.sort(key=lambda x: x["score"], reverse=True)
    short_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 8. セクター強度マップが空の場合も対応（全て50に）
    if not sector_strength_map:
        # 候補がない場合でも build_line_message は動くが、
        # map が全くの空だと寂しいのでフォールバック
        sectors = sorted({r["sector"] for r in core_candidates + short_candidates})
        for s in sectors:
            sector_strength_map[s] = calc_sector_strength(s)

    # 9. LINE本文生成（Ver2.0）
    message = build_line_message(
        today=today,
        market_score=market_score,
        core_list=core_candidates,
        short_list=short_candidates,
        sector_strength_map=sector_strength_map,
    )

    return message


def main() -> None:
    text = screen_universe()
    print(text)


if __name__ == "__main__":
    main()
