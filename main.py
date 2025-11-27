
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict

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
# 時刻ユーティリティ（JST）
# ==========================================


def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


# ==========================================
# データ取得
# ==========================================


def fetch_history(ticker: str, days: int = 80):
    """
    yfinanceで日足を取得して80本に整形。
    """
    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.tail(days)
    if len(df) < 40:  # あまりにも短いものは除外
        return None

    # 必須列の存在確認
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            return None

    return df


def fetch_market_cap(ticker: str) -> float:
    """
    時価総額を取得（失敗時は np.nan）。
    """
    try:
        info = yf.Ticker(ticker).fast_info
        mc = getattr(info, "market_cap", None)
        if mc is None:
            # 古いyfinanceの場合はこちら
            info2 = yf.Ticker(ticker).info
            mc = info2.get("marketCap")
        return float(mc) if mc is not None else np.nan
    except Exception:
        return float("nan")


# ==========================================
# フィルタリング
# ==========================================


def passes_basic_filters(ind_df, market_cap: float) -> bool:
    """
    事故回避用の最低フィルター。
      - 時価総額 200億 未満除外
      - 売買代金 1億 未満除外
    """
    if not np.isfinite(market_cap) or market_cap < 2e10:
        return False

    if "turnover" not in ind_df.columns:
        return False

    avg_turnover = float(ind_df["turnover"].tail(20).mean())
    if not np.isfinite(avg_turnover) or avg_turnover < 1e8:
        return False

    return True


# ==========================================
# メインロジック
# ==========================================


def run_screening() -> str:
    """
    スクリーニング → スコアリング → LINEメッセージ生成まで一気通貫。
    戻り値はLINEに投げる本文（1メッセージ分）。
    """
    universe = load_universe("universe_jpx.csv")
    market_score = calc_market_score()

    core_list: List[Dict] = []
    short_list: List[Dict] = []

    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        df = fetch_history(ticker, days=80)
        if df is None:
            continue

        df_ind = add_indicators(df)
        metrics = extract_metrics(df_ind)
        if not metrics:
            continue

        mc = fetch_market_cap(ticker)
        if not passes_basic_filters(df_ind, mc):
            continue

        sector_strength = calc_sector_strength(sector)

        # Core スコア
        core_score, core_comment = calc_core_score(
            metrics,
            market_score=market_score,
            sector_strength=sector_strength,
        )

        # ShortTerm スコア
        short_score, short_comment = calc_shortterm_score(
            metrics,
            market_score=market_score,
            sector_strength=sector_strength,
        )

        rec_base = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
        }

        if core_score >= 75.0:
            rec_core = rec_base.copy()
            rec_core.update(
                {
                    "core_score": core_score,
                    "core_comment": core_comment,
                }
            )
            core_list.append(rec_core)

        if short_score >= 75.0:
            rec_short = rec_base.copy()
            rec_short.update(
                {
                    "short_score": short_score,
                    "short_comment": short_comment,
                }
            )
            short_list.append(rec_short)

    # スコア順に整列
    core_list.sort(key=lambda x: x["core_score"], reverse=True)
    short_list.sort(key=lambda x: x["short_score"], reverse=True)

    # 日付（JST）
    today_str = jst_now().strftime("%Y-%m-%d")

    # LINEメッセージ本文生成（Ver2.0仕様）
    message = build_line_message(
        date_str=today_str,
        market_score=market_score,
        core_list=core_list,
        short_list=short_list,
    )

    return message


def main():
    msg = run_screening()
    print(msg)


if __name__ == "__main__":
    main()
