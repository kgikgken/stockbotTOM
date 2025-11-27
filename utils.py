import pandas as pd
import os

def load_universe(path="universe_jpx.csv"):
    """
    ユニバースCSVを読み込む。
    無い場合はテンプレtickersを返す。
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            return df["ticker"].astype(str).tolist()

    return [
        "6920.T", "8035.T", "4502.T", "9984.T", "8316.T",
        "7203.T", "6861.T", "4063.T", "7735.T", "9433.T"
    ]
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =====================================================
# インジケータ計算
# =====================================================

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    日足 OHLCV DataFrame に各種インジケータ列を追加して返す。
    必須カラム: Open, High, Low, Close, Volume
    """
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    df["close"] = close

    # MA5 / MA20
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 20日平均出来高比
    vol20 = vol.rolling(20).mean()
    df["vol_ratio20"] = vol / vol20

    # 売買代金と平均
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # 60日高値とそこからの乖離率
    win = close.tail(60)
    if len(win) > 0:
        highest = win.max()
        df["off_high_pct"] = (close - highest) / highest * 100.0
    else:
        df["off_high_pct"] = np.nan

    # 20日ボラティリティ（年率換算っぽく）
    ret = close.pct_change()
    df["vola20"] = ret.rolling(20).std() * np.sqrt(20)

    # 20MAの傾き（1日あたりの変化率）
    ma20 = df["ma20"]
    df["trend_slope20"] = ma20.pct_change()

    # 下ヒゲ比率（当日）
    body = (close - df["Open"].astype(float)).abs()
    rng = high - low
    lower_shadow = np.where(close >= df["Open"], close - low, df["Open"] - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rng > 0, lower_shadow / rng, 0.0)
    df["lower_shadow_ratio"] = ratio

    # 直近60営業日での「高値からの日数」
    if len(df) >= 2:
        tail = df.tail(60).copy()
        c_tail = tail["close"]
        if len(c_tail) > 0:
            idx_max = int(np.argmax(c_tail.values))
            days_since_high = (len(c_tail) - 1) - idx_max
        else:
            days_since_high = np.nan
    else:
        days_since_high = np.nan

    df["days_since_high60"] = days_since_high

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float] | None:
    """
    スコア計算で使う指標を1銘柄分まとめて返す。
    """
    if df is None or df.empty:
        return None

    last = df.iloc[-1]

    metrics = {
        "close": _safe_float(last.get("close", np.nan)),
        "ma5": _safe_float(last.get("ma5", np.nan)),
        "ma20": _safe_float(last.get("ma20", np.nan)),
        "rsi": _safe_float(last.get("rsi", np.nan)),
        "vol_ratio20": _safe_float(last.get("vol_ratio20", np.nan)),
        "turnover_avg20": _safe_float(last.get("turnover_avg20", np.nan)),
        "off_high_pct": _safe_float(last.get("off_high_pct", np.nan)),
        "vola20": _safe_float(last.get("vola20", np.nan)),
        "trend_slope20": _safe_float(last.get("trend_slope20", np.nan)),
        "lower_shadow_ratio": _safe_float(last.get("lower_shadow_ratio", np.nan)),
        "days_since_high60": _safe_float(last.get("days_since_high60", np.nan)),
    }

    return metrics


# =====================================================
# 地合い・セクター強度（仮実装）
# =====================================================

def calc_market_score() -> int:
    """
    とりあえず 60 固定。
    あとでユーザーが本物ロジックに差し替える前提。
    """
    return 60


def calc_sector_strength(sector: str) -> int:
    """
    セクター強度（仮で全部50固定）。
    あとでTOPIX-17などから差し替え予定。
    """
    return 50


# =====================================================
# スコア内部ロジック
# =====================================================

def _trend_metric(metrics: Dict[str, float]) -> float:
    """
    トレンド強度を 0〜100 に正規化。
    - 20MA の傾き (slope)
    - 高値からの位置 (off_high_pct)
    """
    slope = metrics.get("trend_slope20", np.nan)
    off_high = metrics.get("off_high_pct", np.nan)

    score = 0.0

    if np.isfinite(slope):
        if slope <= 0:
            slope_score = 0.0
        elif slope >= 0.01:  # 1%/日 以上ならMAX
            slope_score = 60.0
        else:
            slope_score = 60.0 * (slope / 0.01)
        score += slope_score

    if np.isfinite(off_high):
        # 高値から -0%〜-20% の範囲を 0〜40 点にマッピング（浅めの押しが高得点）
        if off_high >= 0:
            pos_score = 30.0  # 高値更新圏
        elif off_high <= -25:
            pos_score = 5.0
        else:
            pos_score = 30.0 - (abs(off_high) / 25.0) * 25.0
        score += max(0.0, min(40.0, pos_score))

    return max(0.0, min(100.0, score))


def _pullback_metric(metrics: Dict[str, float]) -> float:
    """
    押し目の質を 0〜100 に正規化。
    - RSI（30〜40前後が理想）
    - 高値からの下落率
    - 日柄（高値からの日数）
    - 下ヒゲ比率
    """
    rsi = metrics.get("rsi", np.nan)
    off_high = metrics.get("off_high_pct", np.nan)
    days = metrics.get("days_since_high60", np.nan)
    shadow = metrics.get("lower_shadow_ratio", np.nan)

    score = 0.0

    # RSI 部分：30〜40 を高評価にする三角関数
    if np.isfinite(rsi):
        if rsi < 20 or rsi > 60:
            rsi_score = 10.0
        elif 20 <= rsi <= 50:
            # 20→0点, 35→満点, 50→0点（山型）
            center = 35.0
            width = 15.0
            rsi_score = 40.0 * max(0.0, 1.0 - abs(rsi - center) / width)
        else:
            rsi_score = 10.0
        score += rsi_score

    # 高値からの下落率
    if np.isfinite(off_high):
        # -5%〜-25% の押しを高評価
        if off_high >= -3:
            drop_score = 10.0
        elif off_high <= -30:
            drop_score = 5.0
        else:
            drop_score = 30.0 * min(1.0, abs(off_high - 3) / 25.0)
        score += max(0.0, min(30.0, drop_score))

    # 日柄（2〜15日くらいの調整を高評価）
    if np.isfinite(days):
        if days < 1:
            day_score = 5.0
        elif days > 30:
            day_score = 10.0
        else:
            center = 8.0
            width = 8.0
            day_score = 20.0 * max(0.0, 1.0 - abs(days - center) / width)
        score += day_score

    # 下ヒゲ比率（0.4以上を高評価）
    if np.isfinite(shadow):
        if shadow >= 0.5:
            shadow_score = 10.0
        elif shadow >= 0.3:
            shadow_score = 7.0
        else:
            shadow_score = 0.0
        score += shadow_score

    return max(0.0, min(100.0, score))


def _liquidity_score(metrics: Dict[str, float]) -> float:
    """
    流動性＆安定度を 0〜20 にスコア化。
    - 直近20日平均売買代金
    - ボラティリティ（高すぎると減点）
    """
    turnover = metrics.get("turnover_avg20", np.nan)
    vola = metrics.get("vola20", np.nan)

    # 売買代金: 1億〜10億で 0〜16点
    if not np.isfinite(turnover) or turnover < 1e8:
        liq = 0.0
    elif turnover >= 10e8:
        liq = 16.0
    else:
        liq = 16.0 * (turnover - 1e8) / (9e8)

    # ボラティリティ: 低め〜適度を優遇
    if np.isfinite(vola):
        if vola < 0.02:       # 2%
            vola_score = 4.0
        elif vola > 0.06:     # 6%以上は減点
            vola_score = 0.0
        else:
            vola_score = 4.0 * (0.06 - vola) / (0.04)
    else:
        vola_score = 0.0

    return max(0.0, min(20.0, liq + vola_score))


# =====================================================
# Core / ShortTerm スコア
# =====================================================

def calc_core_score(
    market_score: int,
    sector_strength: int,
    metrics: Dict[str, float],
) -> Tuple[int, str]:
    """
    Core用（中期押し目）スコア 0〜100 ＋ コメント1行
    構造:
    - 地合い 20
    - セクター強度 20
    - トレンド強度 20
    - 押し目の質 20
    - 流動性 & 安定度 20
    """
    trend_raw = _trend_metric(metrics)      # 0〜100
    pullback_raw = _pullback_metric(metrics)  # 0〜100
    liq_raw = _liquidity_score(metrics)     # 0〜20

    # 各項目をウェイトに合わせてスケーリング
    m_component = max(0.0, min(20.0, market_score / 100.0 * 20.0))
    s_component = max(0.0, min(20.0, sector_strength / 100.0 * 20.0))
    t_component = max(0.0, min(20.0, trend_raw / 100.0 * 20.0))
    p_component = max(0.0, min(20.0, pullback_raw / 100.0 * 20.0))
    l_component = max(0.0, min(20.0, liq_raw))

    total = int(round(m_component + s_component + t_component + p_component + l_component))

    # コメント生成
    comments: List[str] = []

    if trend_raw >= 70:
        comments.append("トレンド◎")
    elif trend_raw >= 50:
        comments.append("トレンド◯")

    rsi = metrics.get("rsi", np.nan)
    if np.isfinite(rsi):
        if 30 <= rsi <= 40:
            comments.append(f"RSI押し目({int(round(rsi))})")
        elif rsi < 30:
            comments.append(f"RSIやや売られ({int(round(rsi))})")

    close = metrics.get("close", np.nan)
    ma20 = metrics.get("ma20", np.nan)
    if np.isfinite(close) and np.isfinite(ma20) and ma20 > 0:
        dist = abs(close - ma20) / ma20
        if dist <= 0.01:
            comments.append("20MA近辺")
        elif dist <= 0.02:
            comments.append("20MA圏")

    if liq_raw >= 15:
        comments.append("流動性◎")
    elif liq_raw >= 10:
        comments.append("流動性◯")

    if not comments:
        comments.append("押し目良好")

    comment_str = " / ".join(comments)
    return total, comment_str


def calc_shortterm_score(
    market_score: int,
    sector_strength: int,
    metrics: Dict[str, float],
) -> Tuple[int, str]:
    """
    ShortTerm用（1〜3日リバ狙い）スコア 0〜100 ＋ コメント1行
    構造:
    - 地合い 20
    - セクター強度 20
    - トレンド 15
    - 押し目の質 30
    - 流動性 15
    """
    trend_raw = _trend_metric(metrics)      # 0〜100
    pullback_raw = _pullback_metric(metrics)  # 0〜100
    liq_raw = _liquidity_score(metrics)     # 0〜20

    m_component = max(0.0, min(20.0, market_score / 100.0 * 20.0))
    s_component = max(0.0, min(20.0, sector_strength / 100.0 * 20.0))
    t_component = max(0.0, min(15.0, trend_raw / 100.0 * 15.0))
    p_component = max(0.0, min(30.0, pullback_raw / 100.0 * 30.0))
    l_component = max(0.0, min(15.0, liq_raw / 20.0 * 15.0))

    total = int(round(m_component + s_component + t_component + p_component + l_component))

    # コメント生成
    comments: List[str] = []

    rsi = metrics.get("rsi", np.nan)
    if np.isfinite(rsi):
        if rsi <= 35:
            comments.append(f"RSI短期押し目({int(round(rsi))})")

    days = metrics.get("days_since_high60", np.nan)
    if np.isfinite(days):
        if 2 <= days <= 15:
            comments.append(f"調整{int(days)}日")

    shadow = metrics.get("lower_shadow_ratio", np.nan)
    if np.isfinite(shadow) and shadow >= 0.4:
        comments.append("下ヒゲ気味")

    close = metrics.get("close", np.nan)
    ma20 = metrics.get("ma20", np.nan)
    if np.isfinite(close) and np.isfinite(ma20) and ma20 > 0:
        dist = abs(close - ma20) / ma20
        if dist <= 0.015:
            comments.append("20MAタッチ圏")

    if liq_raw >= 12:
        comments.append("流動性◎")
    elif liq_raw >= 8:
        comments.append("流動性◯")

    if not comments:
        comments.append("短期リバ候補")

    comment_str = " / ".join(comments)
    return total, comment_str


# =====================================================
# メッセージ整形（Ver2.0）
# =====================================================

def _market_label_and_risk(market_score: int) -> tuple[str, str, float, int, str]:
    """
    地合いスコアからラベル・レバ目安などを決定。
    戻り値:
      (label, regime_label, max_leverage, max_positions, comment)
    """
    if market_score >= 70:
        label = "やや強め"
        regime_label = "攻め寄り"
        max_leverage = 2.0
        max_positions = 4
        comment = "押し目狙い自体は追い風。ただしルール外のINはしない。"
    elif market_score <= 40:
        label = "弱め"
        regime_label = "守り優先"
        max_leverage = 1.2
        max_positions = 2
        comment = "新規はかなり厳選。サイズ小さめを基本に。"
    else:
        label = "中立"
        regime_label = "中立"
        max_leverage = 1.5
        max_positions = 3
        comment = "軽めのスイングは可。イベント前に無理なフルベットは不要。"
    return label, regime_label, max_leverage, max_positions, comment


def _build_three_line_summary(market_score: int, top_sector_name: str) -> list[str]:
    lines: list[str] = []

    if market_score >= 70:
        lines.append("・地合いはやや強め。押し目狙いは前向きに検討。")
    elif market_score <= 40:
        lines.append("・地合いは弱め。新規は慎重に、サイズ控えめ。")
    else:
        lines.append("・地合いは中立〜レンジ。無理なフルベットは不要。")

    if top_sector_name != "なし":
        lines.append(f"・セクターでは「{top_sector_name}」が相対的に優勢。")
    else:
        lines.append("・セクターは全面的に重く、方向感は出にくい。")

    lines.append("・中期はCore中心、短期はShortTerm候補を必要に応じて確認。")
    return lines


def _rank_sectors_from_candidates(
    core_list: list[dict],
    short_list: list[dict],
    sector_strength_map: dict[str, int],
) -> list[tuple[str, int]]:
    """
    Core / ShortTerm に出ているセクターから「優先セクター」をランク付け。
    単純に銘柄数と強度でスコア化。
    """
    counts: dict[str, int] = {}
    for r in core_list + short_list:
        sec = r["sector"]
        counts[sec] = counts.get(sec, 0) + 1

    scored: list[tuple[str, int]] = []
    for sec, cnt in counts.items():
        strength = sector_strength_map.get(sec, 50)
        score = cnt * 10 + strength  # 出現数をやや重視
        scored.append((sec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _build_priority_names(core_list: list[dict], short_list: list[dict]) -> list[str]:
    """
    最優先銘柄TOP3（Core優先・足りなければShortTerm補填）
    """
    top: list[str] = []

    # Core優先
    for r in core_list:
        code = r["ticker"].replace(".T", "")
        top.append(f"{code} {r['name']}（Core {r['score']}）")
        if len(top) >= 3:
            return top

    # ShortTerm で補填
    for r in short_list:
        code = r["ticker"].replace(".T", "")
        top.append(f"{code} {r['name']}（Short {r['score']}）")
        if len(top) >= 3:
            break

    return top


def build_line_message(
    today: str,
    market_score: int,
    core_list: list[dict],
    short_list: list[dict],
    sector_strength_map: dict[str, int],
) -> str:
    """
    LINE通知 Ver2.0 の本文を 1 本のテキストとして返す。
    フォーマット:
    - 今日の結論（最優先セクター、最優先銘柄TOP3含む）
    - TOPセクター
    - 相場3行まとめ
    - Core
    - ShortTerm
    """
    label, regime_label, max_lev, max_pos, comment = _market_label_and_risk(market_score)

    # セクターランキング
    ranked_sectors = _rank_sectors_from_candidates(core_list, short_list, sector_strength_map)
    if ranked_sectors:
        top_sector_name = ranked_sectors[0][0]
        top_sector_strength = sector_strength_map.get(top_sector_name, 50)
    else:
        top_sector_name = "なし"
        top_sector_strength = 0

    # 最優先銘柄TOP3
    priority_names = _build_priority_names(core_list, short_list)

    lines: list[str] = []
    lines.append(f"📅 {today} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(
        f"- 地合いスコア: {market_score}点（{label} / {regime_label}）"
    )
    lines.append(
        f"- レバ目安: 最大 約{max_lev:.1f}倍 / ポジ数目安: {max_pos}銘柄"
    )
    lines.append(f"- コメント: {comment}")
    lines.append(
        f"- 最優先セクター: {top_sector_name}（強度 {top_sector_strength}）"
    )
    if priority_names:
        lines.append("- 最優先銘柄TOP3:")
        for p in priority_names:
            lines.append(f"   ・{p}")
    else:
        lines.append("- 最優先銘柄TOP3: 対象なし")
    lines.append("")

    # TOPセクター表示
    lines.append("◆ 今日のTOPセクター")
    if ranked_sectors:
        for i, (sec, score) in enumerate(ranked_sectors[:3], 1):
            strength = sector_strength_map.get(sec, 50)
            lines.append(f"{i}位: {sec}（強度 {strength}）")
    else:
        lines.append("セクター候補なし（Core/ShortTerm該当なし）")
    lines.append("")

    # 相場3行まとめ
    lines.append("◆ 今日の相場3行まとめ")
    three_lines = _build_three_line_summary(market_score, top_sector_name)
    lines.extend(three_lines)

    # Coreセクション
    lines.append("◆ Core（本命候補）")
    if core_list:
        for i, r in enumerate(core_list, 1):
            code = r["ticker"]
            lines.append(f"{i}. {code} {r['name']} / {r['sector']}")
            lines.append(
                f"   Score {r['score']} / {r['comment']} / 現値: {r['price']}円"
            )
    else:
        lines.append("本命条件を満たす銘柄なし。今日は無理に攻めない選択もあり.")

    # ShortTermセクション
    lines.append("◆ ShortTerm（短期1〜3日候補）")
    if short_list:
        for i, r in enumerate(short_list, 1):
            code = r["ticker"]
            lines.append(f"{i}. {code} {r['name']} / {r['sector']}")
            lines.append(
                f"   Score {r['score']} / {r['comment']} / 現値: {r['price']}円"
            )
    else:
        lines.append("現在、条件を満たす短期パターン候補はなし。")

    return "\n".join(lines)
