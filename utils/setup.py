# ============================================
# utils/setup.py
# 初期データ読み込み・週次状態管理
# ============================================

import os
import json
import pandas as pd
from datetime import datetime
from utils.util import jst_today_str

# --------------------------------------------
# パス設定
# --------------------------------------------
UNIVERSE_PATH = "universe_jpx.csv"
EVENTS_PATH = "events.csv"
WEEKLY_STATE_PATH = "weekly_state.json"


# --------------------------------------------
# ユニバース読み込み
# --------------------------------------------
def load_universe() -> pd.DataFrame:
    """
    全銘柄ユニバースを読み込む
    """
    if not os.path.exists(UNIVERSE_PATH):
        raise FileNotFoundError(f"Universe file not found: {UNIVERSE_PATH}")

    df = pd.read_csv(UNIVERSE_PATH)
    return df


# --------------------------------------------
# イベント読み込み
# --------------------------------------------
def load_events() -> pd.DataFrame:
    """
    マクロ・決算イベント読み込み
    """
    if not os.path.exists(EVENTS_PATH):
        return pd.DataFrame()

    df = pd.read_csv(EVENTS_PATH)
    return df


# --------------------------------------------
# 週次状態読み込み
# --------------------------------------------
def load_weekly_state() -> dict:
    """
    週次新規回数などの状態を取得
    """
    today = jst_today_str()

    if not os.path.exists(WEEKLY_STATE_PATH):
        return {
            "week_start": today,
            "new_entries": 0,
        }

    with open(WEEKLY_STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    return state


# --------------------------------------------
# 週次状態保存
# --------------------------------------------
def save_weekly_state(state: dict):
    """
    週次状態を保存
    """
    with open(WEEKLY_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)