# 最強スイング戦略 main.py（プロ仕様）
# by stockbotTOM + 世界最高トレーダー視点

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===============================
# 設定
# ===============================
UNIVERSE_CSV_PATH = "universe_jpx.csv"
MAX_CORE = 3
MAX_WATCH = 2
MAX_SHORT = 2
TP_LOOKBACK_CORE = 10
TP_LOOKBACK_SHORT = 5

# ===============================
# データ読み込み
# ===============================
df = pd.read_csv(UNIVERSE_CSV_PATH)
today = datetime.today().strftime("%Y-%m-%d")

# ===============================
# テクニカル指標計算
# ===============================
def compute_indicators(df):
    df["MA25"] = df["Close"].rolling(window=25).mean()
    df["MA25_slope"] = df["MA25"].diff()
    df["Volume_Avg20"] = df["Volume"].rolling(window=20).mean()
    df["RSI"] = 100 - 100 / (1 + df["Close"].pct_change().rolling(window=14).mean())
    df["High10"] = df["High"].rolling(window=10).max()
    df["High5"] = df["High"].rolling(window=5).max()
    df["TP_core"] = (df["High10"] * 0.6 + df["MA25"] * 0.4).round()
    df["TP_short"] = (df["High5"] * 0.5 + df["MA25"] * 0.5).round()
    return df

# ===============================
# スクリーニング条件
# ===============================
def is_core(row):
    return (
        row["MA25_slope"] > 0 and
        row["Close"] > row["MA25"] and
        row["Volume"] > row["Volume_Avg20"] * 1.5 and
        row["RSI"] > 45 and row["RSI"] < 65 and
        row["Close"] > row["High10"] * 0.95
    )

def is_watch(row):
    return (
        row["MA25_slope"] >= 0 and
        row["Close"] > row["MA25"] * 0.97 and
        row["RSI"] > 40 and row["Volume"] > row["Volume_Avg20"]
    )

def is_short(row):
    return (
        row["RSI"] < 35 and
        row["Volume"] > row["Volume_Avg20"] * 1.2 and
        row["Close"] < row["MA25"]
    )

# ===============================
# メッセージ構築
# ===============================
def make_line_message(core, watch, short):
    lines = [f"📅 {today} stockbotTOM 日報", ""]

    lines.append("◆ 今日の結論")
    lines.append("- レバ目安: 最大 約2.0倍 / ポジ数目安: 3〜4銘柄")
    lines.append("- コメント: 初動＋押し目＋出来高が条件。妥協エントリー禁止。")
    lines.append("")

    lines.append("◆ Core（本命候補）")
    if core.empty:
        lines.append("本命候補なし。焦らず、待つ。")
    else:
        for _, row in core.iterrows():
            lines.append(f"{row['Code']} {row['Name']}")
            lines.append(f"現{int(row['Close'])}円 / TP目安: {int(row['TP_core'])}円")
            lines.append("")

    lines.append("◆ Watch（注目候補）")
    if watch.empty:
        lines.append("該当なし。相場全体に注視。")
    else:
        for _, row in watch.iterrows():
            lines.append(f"{row['Code']} {row['Name']}")
            lines.append(f"現{int(row['Close'])}円（仕上がり中）")
            lines.append("")

    lines.append("◆ ShortTerm（短期リバ候補）")
    if short.empty:
        lines.append("反発初動銘柄なし。待機。")
    else:
        for _, row in short.iterrows():
            lines.append(f"{row['Code']} {row['Name']}")
            lines.append(f"現{int(row['Close'])}円 / TP目安: {int(row['TP_short'])}円")
            lines.append("")

    return "\n".join(lines)

def make_x_post(core, watch, short):
    def core_line(row):
        return (
            f"{row['Code']} {row['Name']}\n"
            f"Edge強 / 現{int(row['Close'])}円 / TP{int(row['TP_core'])}円\n"
            "気づけるやつだけ見ればいい。"
        )
    def watch_line(row):
        return (
            f"{row['Code']} {row['Name']}\n"
            f"仕上がり中 / 現{int(row['Close'])}円\n"
            "理解できるやつだけ来い。"
        )
    def short_line(row):
        return (
            f"{row['Code']} {row['Name']}\n"
            f"短期リバ / 現{int(row['Close'])}円 / TP{int(row['TP_short'])}円\n"
            "判断できるやつだけ残ればいい。"
        )
    texts = []
    if not core.empty:
        texts.append("[Core]")
        for _, row in core.iterrows():
            texts.append(core_line(row))
    if not watch.empty:
        texts.append("[Watch]")
        for _, row in watch.iterrows():
            texts.append(watch_line(row))
    if not short.empty:
        texts.append("[Short]")
        for _, row in short.iterrows():
            texts.append(short_line(row))
    return "\n\n".join(texts)

# ===============================
# 実行ブロック
# ===============================
df = compute_indicators(df)
core_df = df[df.apply(is_core, axis=1)].head(MAX_CORE)
watch_df = df[df.apply(is_watch, axis=1)].head(MAX_WATCH)
short_df = df[df.apply(is_short, axis=1)].head(MAX_SHORT)

line_msg = make_line_message(core_df, watch_df, short_df)
x_msg = make_x_post(core_df, watch_df, short_df)

with open("line_message.txt", "w") as f:
    f.write(line_msg)

with open("x_post.txt", "w") as f:
    f.write(x_msg)
