# ============================================
# main.py
# stockbotTOM (Swing専用 / 1〜7日) - clean version
# ============================================

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

import pandas as pd

from utils.util import jst_today_str, jst_now_str
from utils.setup import (
    load_universe,
    load_events,
    load_positions,
    load_weekly_state,
    save_weekly_state,
)
from utils.market import calc_market_regime
from utils.sector import top_sectors_5d
from utils.screener import run_swing_screening
from utils.report import build_daily_report
from utils.line import send_line_via_worker


# ----------------------------
# Paths (repo root)
# ----------------------------
UNIVERSE_PATH = os.getenv("UNIVERSE_PATH", "universe_jpx.csv")
EVENTS_PATH = os.getenv("EVENTS_PATH", "events.csv")
POSITIONS_PATH = os.getenv("POSITIONS_PATH", "positions.csv")
WEEKLY_STATE_PATH = os.getenv("WEEKLY_STATE_PATH", "weekly_state.json")

# Cloudflare Worker URL (LINE送信用)
WORKER_URL = os.getenv("WORKER_URL", "")

# 送信の冗長化（任意）：WORKER_URL が死んだ時の予備
WORKER_URL_BACKUP = os.getenv("WORKER_URL_BACKUP", "")


def _send_message(message: str) -> None:
    """
    LINE送信（Cloudflare Worker 経由）
    - primary -> backup の順で試す
    """
    if not WORKER_URL and not WORKER_URL_BACKUP:
        print("WORKER_URL is empty. Print only:\n")
        print(message)
        return

    last_err: Optional[Exception] = None

    if WORKER_URL:
        try:
            send_line_via_worker(WORKER_URL, message)
            return
        except Exception as e:
            last_err = e

    if WORKER_URL_BACKUP:
        try:
            send_line_via_worker(WORKER_URL_BACKUP, message)
            return
        except Exception as e:
            last_err = e

    # ここまで来たら両方失敗
    if last_err:
        raise last_err


def main() -> int:
    """
    1) データ読込
    2) 地合い/セクター計算
    3) スクリーニング
    4) レポート生成
    5) LINE送信
    """
    today = jst_today_str()

    universe = load_universe(UNIVERSE_PATH)
    events = load_events(EVENTS_PATH, today=today)
    positions = load_positions(POSITIONS_PATH)
    weekly_state = load_weekly_state(WEEKLY_STATE_PATH, today=today)

    # 地合い（market_score / delta_3d / regime / macro_risk）
    market_ctx = calc_market_regime(today=today)

    # セクター動向（直近5日）
    sector_rows = top_sectors_5d(universe, top_n=5)

    # Swingスクリーニング（週次制限・イベント制限・RR/EV/AdjEV・A1/A2など全てここで確定）
    screen_out = run_swing_screening(
        universe=universe,
        positions=positions,
        events=events,
        market_ctx=market_ctx,
        weekly_state=weekly_state,
    )

    # 週次状態の更新（新規カウントなど）
    save_weekly_state(WEEKLY_STATE_PATH, screen_out.get("weekly_state", weekly_state))

    # レポート生成（日本語・分かりやすい文面）
    report_text = build_daily_report(
        today=today,
        now_str=jst_now_str(),
        market_ctx=market_ctx,
        sector_rows=sector_rows,
        events=events,
        screen_out=screen_out,
        positions=positions,
    )

    _send_message(report_text)
    print("OK: sent report")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # 不達リスク低減：落ちても短文アラートだけは送る
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3500:]
        fallback = (
            f"stockbotTOM: 実行エラー\n"
            f"日付: {jst_today_str()}\n"
            f"時刻: {jst_now_str()}\n"
            f"{err}"
        )
        try:
            _send_message(fallback)
        except Exception:
            pass

        print(fallback, file=sys.stderr)
        raise