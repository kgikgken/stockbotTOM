# ============================================
# main.py
# stockbotTOM（Swing専用 / 1〜7日）
# - NO-TRADE完全機械化
# - EV → 地合い/イベントで補正（AdjEV）
# - 速度（R/day）主導
# - 相場認識をレポート明示
# - ロット事故を事前警告
# - LINE送信（Cloudflare Worker 経由想定）
# ============================================

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any

from utils.util import jst_today_str
from utils.setup import load_universe, load_events, load_weekly_state, save_weekly_state
from utils.market import calc_market_regime
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.screener import run_swing_screening
from utils.report import build_daily_report
from utils.line import send_line_message


# -----------------------------
# 設定（必要ならここだけ触る）
# -----------------------------
@dataclass(frozen=True)
class Config:
    # 入力
    universe_path: str = "universe_jpx.csv"
    positions_path: str = "positions.csv"
    events_path: str = "events.csv"
    weekly_state_path: str = "weekly_state.json"

    # 指数（地合い計算のベンチマーク）
    index_ticker: str = "^TOPX"  # yfinanceのTOPIX。環境により取得不可なら screener 側でフォールバック

    # Swing基本（最終採用上限）
    max_final: int = 5

    # 週次制限（新規回数）
    weekly_new_limit: int = 3

    # 決算回避（±営業日）
    earnings_exclude_days: int = 3

    # セクター上限（同一セクター最大数）
    per_sector_limit: int = 2

    # 相関制限（同時採用禁止ライン）
    corr_block_threshold: float = 0.75

    # 候補の“監視”を出すか（cで話していたやつ）
    include_watchlist: bool = True
    watchlist_max: int = 10

    # LINE（worker URL は utils/line.py が env から拾う）
    line_dry_run: bool = False  # Trueなら送らずに標準出力のみ

    # ロット事故警告（資産比 % を超えたら警告）
    lot_accident_warn_pct: float = 8.0


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    cfg = Config(
        line_dry_run=_env_bool("LINE_DRY_RUN", False),
    )

    today = jst_today_str()

    try:
        # 1) 入力読み込み
        universe = load_universe(cfg.universe_path)
        events = load_events(cfg.events_path)
        weekly_state = load_weekly_state(cfg.weekly_state_path)

        positions = load_positions(cfg.positions_path)
        pos_summary = analyze_positions(positions)

        # 2) 地合い（MarketScore / Δ3d / 相場判断 / マクロ警戒）
        regime = calc_market_regime(index_ticker=cfg.index_ticker, events=events)

        # 3) セクター（補助情報）
        sectors_5d = top_sectors_5d(universe)

        # 4) スクリーニング（Swing）
        screening = run_swing_screening(
            universe=universe,
            events=events,
            positions=positions,
            regime=regime,
            today=today,
            weekly_state=weekly_state,
            config={
                "max_final": cfg.max_final,
                "earnings_exclude_days": cfg.earnings_exclude_days,
                "per_sector_limit": cfg.per_sector_limit,
                "corr_block_threshold": cfg.corr_block_threshold,
                "weekly_new_limit": cfg.weekly_new_limit,
                "include_watchlist": cfg.include_watchlist,
                "watchlist_max": cfg.watchlist_max,
            },
        )

        # 5) レポート生成（LINE文）
        message = build_daily_report(
            today=today,
            regime=regime,
            sectors_5d=sectors_5d,
            events=events,
            screening=screening,
            pos_summary=pos_summary,
            weekly_state=weekly_state,
            config={
                "lot_accident_warn_pct": cfg.lot_accident_warn_pct,
            },
        )

        # 6) 送信（LINE）
        if cfg.line_dry_run:
            print(message)
        else:
            send_line_message(message)

        # 7) 週次状態保存（新規回数など）
        # run_swing_screening 側で weekly_state を更新して返す想定
        if isinstance(screening, dict) and "weekly_state" in screening:
            save_weekly_state(cfg.weekly_state_path, screening["weekly_state"])
        else:
            # フォールバック：読み込んだものをそのまま保存（最低限）
            save_weekly_state(cfg.weekly_state_path, weekly_state)

        return 0

    except Exception as e:
        # 失敗した時も “何が起きたか” を出す（Actions で追える）
        err = f"[ERROR] stockbotTOM failed on {today}: {e}"
        print(err, file=sys.stderr)
        traceback.print_exc()

        # 送信できるならエラーもLINEに流す（落ちた理由が即分かる）
        try:
            if not cfg.line_dry_run:
                send_line_message(err)
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())