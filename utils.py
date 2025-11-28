from __future__ import annotations

from typing import Dict, List, Tuple

import os
import numpy as np
import pandas as pd
import yfinance as yf


# =====================================================
# ユニバース読み込み
# =====================================================


def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    """
    ユニバースCSVを読み込んで DataFrame を返す。
    必須: ticker カラム
    任意: name, sector カラム
    無い場合はデフォルトの銘柄リストを返す。
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str)
            if "name" not in df.columns:
                df["name"] = df["ticker"]
            else:
                df["name"] = df["name"].astype(str)
            if "sector" not in df.columns:
                df["sector"] = "その他"
            else:
                df["sector"] = df["sector"].astype(str)
            return df[["ticker", "name", "sector"]]

    # フォールバック（テンプレユニバース）
    data = {
        "ticker": [
            "6920.T",  # レーザーテック
            "8035.T",  # 東エレク
            "4502.T",  # 武田薬品
            "9984.T",  # ソフトバンクG
            "8316.T",  # 三井住友FG
            "7203.T",  # トヨタ
            "6861.T",  # キーエンス
            "4063.T",  # 信越化学
            "7735.T",  # スクリーン
            "9433.T",  # KDDI
        ],
    }
    df = pd.DataFrame(data)
    df["name"] = df["ticker"]
    df["sector"] = "その他"
    return df[["ticker", "name", "sector"]]


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
    if len(close) >= 1:
        highest = close.tail(60).max()
        df["off_high_pct"] = (close - highest) / highest * 100.0
    else:
        df["off_high_pct"] = np.nan

    # 20日ボラティリティ（簡易版）
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
# マクロ地合い・セクター強度
# =====================================================

_MARKET_TICKERS = {
    "NK": "^N225",      # 日経平均
    "TOPIX": "^TOPX",   # TOPIX
    "NDX": "^NDX",      # NASDAQ100
    "VIX": "^VIX",      # VIX
    "USDJPY": "JPY=X",  # ドル円
}


def _safe_last_pct(ticker: str, period: str = "10d") -> float:
    """
    指定ティッカーの直近の日足から「前日比(%)」を返す。
    取れなければ 0.0。
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return 0.0
    if df is None or len(df) < 2:
        return 0.0
    close = df["Close"].astype(float)
    ret = close.pct_change().iloc[-1]
    if not np.isfinite(ret):
        return 0.0
    return float(ret * 100.0)  # %


def _safe_n_day_return(ticker: str, days: int = 5, period: str = "60d") -> float:
    """
    指定ティッカーの n 日リターン(%) を返す。
    足りなければ 0.0。
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return 0.0
    if df is None or len(df) <= days:
        return 0.0
    close = df["Close"].astype(float)
    ret = close.iloc[-1] / close.iloc[-(days + 1)] - 1.0
    if not np.isfinite(ret):
        return 0.0
    return float(ret * 100.0)


def calc_market_score() -> int:
    """
    地合いスコア（0〜100）を算出。
    - 日本株（N225 / TOPIX）の強さ
    - 米ハイテク（NDX）の雰囲気
    - VIX（恐怖指数、低いほど良い）
    - ドル円（急変はマイナス）
    """
    # 日本株：日経・TOPIXの 1日 & 5日リターン
    nk_1 = _safe_last_pct(_MARKET_TICKERS["NK"])
    nk_5 = _safe_n_day_return(_MARKET_TICKERS["NK"], days=5)

    tp_1 = _safe_last_pct(_MARKET_TICKERS["TOPIX"])
    tp_5 = _safe_n_day_return(_MARKET_TICKERS["TOPIX"], days=5)

    # 米ハイテク
    ndx_1 = _safe_last_pct(_MARKET_TICKERS["NDX"])
    ndx_5 = _safe_n_day_return(_MARKET_TICKERS["NDX"], days=5)

    # VIX：上昇はリスクオフ
    vix_1 = _safe_last_pct(_MARKET_TICKERS["VIX"])

    # ドル円の1日変動（急変ならマイナス）
    fx_1 = _safe_last_pct(_MARKET_TICKERS["USDJPY"])

    score = 50.0  # ベースは中立50

    # 日本株（重視）
    jp_today = (nk_1 + tp_1) / 2.0
    jp_5d = (nk_5 + tp_5) / 2.0

    score += max(-15.0, min(15.0, jp_today * 10.0))
    score += max(-10.0, min(10.0, jp_5d * (10.0 / 3.0)))

    # 米ハイテク（NDX）：リスクオンかどうか
    us_mix = (ndx_1 * 0.6 + ndx_5 * 0.4)
    score += max(-8.0, min(8.0, us_mix * 4.0))

    # VIX：上昇＝マイナス
    score -= max(-10.0, min(10.0, vix_1 * 2.0))

    # ドル円：急激な動きはマイナス
    if abs(fx_1) > 0.7:
        score -= 5.0
    elif abs(fx_1) > 0.4:
        score -= 2.0

    score = max(0.0, min(100.0, score))
    return int(round(score))


# セクター → インデックス/ETF 対応（必要に応じてユーザー側で調整）
_SECTOR_TICKER_MAP: Dict[str, str] = {
    "電機・精密": "1615.T",
    "銀行": "1615.T",
    "エネルギー資源": "1662.T",
    "建設": "1619.T",
    "情報通信": "1595.T",
    # 見つからない場合は TOPIX にフォールバック
}


def calc_sector_strength(sector: str) -> int:
    """
    セクター強度（0〜100）。
    - 対応するETF/指数があればそれを利用
    - 無ければ TOPIX 全体と同じ扱い
    """
    sector = str(sector)
    ticker = _SECTOR_TICKER_MAP.get(sector, _MARKET_TICKERS["TOPIX"])

    sec_1 = _safe_last_pct(ticker)
    sec_5 = _safe_n_day_return(ticker, days=5)
    sec_20 = _safe_n_day_return(ticker, days=20)
    sec_60 = _safe_n_day_return(ticker, days=60, period="250d")

    mkt_1 = _safe_last_pct(_MARKET_TICKERS["TOPIX"])
    mkt_5 = _safe_n_day_return(_MARKET_TICKERS["TOPIX"], days=5)
    mkt_20 = _safe_n_day_return(_MARKET_TICKERS["TOPIX"], days=20)
    mkt_60 = _safe_n_day_return(_MARKET_TICKERS["TOPIX"], days=60, period="250d")

    score = 50.0

    # 絶対パフォーマンス
    score += max(-15.0, min(15.0, sec_1 * 8.0))
    score += max(-10.0, min(10.0, sec_5 * (10.0 / 3.0)))

    # 中期〜長期
    score += max(-8.0, min(8.0, sec_20 * 2.0))
    score += max(-7.0, min(7.0, sec_60 * 1.5))

    # 相対パフォーマンス（TOPIX比）
    rel_1 = sec_1 - mkt_1
    rel_5 = sec_5 - mkt_5
    rel_20 = sec_20 - mkt_20
    rel_60 = sec_60 - mkt_60

    score += max(-10.0, min(10.0, rel_1 * 10.0))
    score += max(-6.0, min(6.0, rel_5 * 3.0))
    score += max(-6.0, min(6.0, rel_20 * 2.0))
    score += max(-5.0, min(5.0, rel_60 * 1.5))

    score = max(0.0, min(100.0, score))
    return int(round(score))


# =====================================================
# ファンダメンタルスコア
# =====================================================


def fetch_fundamentals(ticker: str) -> Dict[str, float]:
    """
    yfinance から簡易ファンダ情報を取得して dict で返す。
    取れないものは NaN。
    """
    out: Dict[str, float] = {
        "market_cap": np.nan,
        "roe": np.nan,
        "op_margin": np.nan,
        "rev_growth": np.nan,
        "pbr": np.nan,
        "per": np.nan,
        "div_yield": np.nan,
    }

    try:
        tk = yf.Ticker(ticker)
    except Exception:
        return out

    # fast_info から時価総額
    try:
        fi = tk.fast_info
        mc = getattr(fi, "market_cap", None)
        if mc is not None:
            out["market_cap"] = float(mc)
    except Exception:
        pass

    # info から各種指標
    try:
        info = tk.info
    except Exception:
        info = {}

    def _get(key, default=np.nan):
        v = info.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    # ROE / 営業利益率 / 成長率など（存在するなら）
    out["roe"] = _get("returnOnEquity", np.nan)
    out["op_margin"] = _get("operatingMargins", np.nan)
    if np.isnan(out["op_margin"]):
        out["op_margin"] = _get("profitMargins", np.nan)

    out["rev_growth"] = _get("revenueGrowth", np.nan)
    out["pbr"] = _get("priceToBook", np.nan)
    out["per"] = _get("trailingPE", np.nan)
    dy = info.get("dividendYield", np.nan)
    try:
        out["div_yield"] = float(dy) * 100.0 if dy not in (None, np.nan) else np.nan
    except Exception:
        out["div_yield"] = np.nan

    return out


def calc_fundamental_score(f: Dict[str, float]) -> int:
    """
    ファンダメンタルの総合スコア 0〜100。
    - ROE
    - 利益率
    - 売上成長
    - PBR
    - 配当利回り
    """
    score = 0.0

    roe = f.get("roe", np.nan)
    opm = f.get("op_margin", np.nan)
    gr = f.get("rev_growth", np.nan)
    pbr = f.get("pbr", np.nan)
    dy = f.get("div_yield", np.nan)

    # ROE: 0〜20%以上を 0〜25点
    if np.isfinite(roe):
        if roe <= 0:
            s = 0.0
        elif roe >= 0.2:
            s = 25.0
        else:
            s = 25.0 * (roe / 0.2)
        score += s

    # 営業利益率: 0〜20%以上を 0〜20点
    if np.isfinite(opm):
        if opm <= 0:
            s = 0.0
        elif opm >= 0.2:
            s = 20.0
        else:
            s = 20.0 * (opm / 0.2)
        score += s

    # 売上成長率: -10%〜+20% を 0〜25点
    if np.isfinite(gr):
        if gr <= -0.1:
            s = 0.0
        elif gr >= 0.2:
            s = 25.0
        else:
            s = 25.0 * ((gr + 0.1) / 0.3)
        s = max(0.0, min(25.0, s))
        score += s

    # PBR: 1〜2倍を高評価、極端な高PBRは減点
    if np.isfinite(pbr):
        if pbr < 0.5:
            s = 8.0
        elif pbr <= 1.5:
            s = 15.0
        elif pbr <= 3.0:
            s = 10.0
        elif pbr <= 5.0:
            s = 5.0
        else:
            s = 0.0
        score += s

    # 配当利回り: 2〜5% を高評価
    if np.isfinite(dy) and dy > 0:
        if 2.0 <= dy <= 5.0:
            score += 10.0
        elif dy > 0.5:
            score += 5.0

    score = max(0.0, min(100.0, score))
    return int(round(score))


# =====================================================
# スコア内部ロジック
# =====================================================


def _trend_metric(metrics: Dict[str, float]) -> float:
    """
    トレンド強度を 0〜100 に正規化。
    - 20MA の傾き (slope)
    - 高値からの位置 (off_high_pct)
    - MA5 と MA20 の位置関係
    """
    slope = metrics.get("trend_slope20", np.nan)
    off_high = metrics.get("off_high_pct", np.nan)
    ma5 = metrics.get("ma5", np.nan)
    ma20 = metrics.get("ma20", np.nan)

    score = 0.0

    # 傾き：右肩上がりを高評価
    if np.isfinite(slope):
        if slope <= 0:
            slope_score = 0.0
        elif slope >= 0.01:  # 1%/日 以上ならMAX
            slope_score = 50.0
        else:
            slope_score = 50.0 * (slope / 0.01)
        score += slope_score

    # 高値からの位置：高値更新圏〜浅い押しを高評価
    if np.isfinite(off_high):
        if off_high >= 0:
            pos_score = 25.0  # 高値更新圏
        elif off_high <= -25:
            pos_score = 5.0
        else:
            pos_score = 25.0 - (abs(off_high) / 25.0) * 20.0
        score += max(0.0, min(25.0, pos_score))

    # MA5 vs MA20：ゴールデンクロス寄りを加点
    if np.isfinite(ma5) and np.isfinite(ma20) and ma20 > 0:
        if ma5 > ma20:
            score += 15.0  # 短期が長期を上回る＝強い
        elif ma5 > ma20 * 0.98:
            score += 8.0   # ほぼ同水準

    return max(0.0, min(100.0, score))


def _pullback_metric(metrics: Dict[str, float]) -> float:
    """
    押し目の質を 0〜100 に正規化。
    - RSI（30〜40前後が理想）
    - 高値からの下落率
    - 日柄（高値からの日数）
    - 下ヒゲ比率
    - 出来高バースト
    """
    rsi = metrics.get("rsi", np.nan)
    off_high = metrics.get("off_high_pct", np.nan)
    days = metrics.get("days_since_high60", np.nan)
    shadow = metrics.get("lower_shadow_ratio", np.nan)
    vol_ratio = metrics.get("vol_ratio20", np.nan)

    score = 0.0

    # RSI 部分：30〜40 を高評価にする三角関数
    if np.isfinite(rsi):
        if rsi < 20 or rsi > 60:
            rsi_score = 5.0
        elif 20 <= rsi <= 50:
            center = 35.0
            width = 15.0
            rsi_score = 45.0 * max(0.0, 1.0 - abs(rsi - center) / width)
        else:
            rsi_score = 10.0
        score += rsi_score

    # 高値からの下落率
    if np.isfinite(off_high):
        if off_high >= -3:
            drop_score = 10.0
        elif off_high <= -30:
            drop_score = 5.0
        else:
            drop_score = 25.0 * min(1.0, abs(off_high - 3) / 25.0)
        score += max(0.0, min(25.0, drop_score))

    # 日柄（2〜15日くらいの調整を高評価）
    if np.isfinite(days):
        if days < 1:
            day_score = 5.0
        elif days > 30:
            day_score = 8.0
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

    # 出来高バースト（Volume Burst）
    if np.isfinite(vol_ratio):
        if vol_ratio >= 2.5:
            vol_score = 10.0
        elif vol_ratio >= 1.5:
            vol_score = 7.0
        elif vol_ratio >= 1.0:
            vol_score = 3.0
        else:
            vol_score = 0.0
        score += vol_score

    return max(0.0, min(100.0, score))


def _liquidity_score(metrics: Dict[str, float]) -> float:
    """
    流動性＆安定度を 0〜20 にスコア化。
    - 直近20日平均売買代金
    - ボラティリティ（高すぎると減点）
    """
    turnover = metrics.get("turnover_avg20", np.nan)
    vola = metrics.get("vola20", np.nan)

    # 売買代金: 1億〜20億で 0〜16点 に補正
    if not np.isfinite(turnover) or turnover < 1e8:
        liq = 0.0
    elif turnover >= 20e8:
        liq = 16.0
    else:
        liq = 16.0 * (turnover - 1e8) / (19e8)

    # ボラティリティ: 1.5〜3.5% をベストに
    if np.isfinite(vola):
        if vola < 0.015:
            vola_score = 1.0
        elif vola < 0.035:
            vola_score = 4.0
        elif vola < 0.06:
            vola_score = 2.0
        else:
            vola_score = 0.0
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
    fundamental_score: int | None = None,
) -> Tuple[int, str]:
    """
    Core用（中期押し目）スコア 0〜100 ＋ コメント1行
    構造:
    - 地合い 20
    - セクター強度 20
    - トレンド強度 20
    - 押し目の質 20
    - 流動性 & 安定度 10
    - ファンダ 10
    """
    trend_raw = _trend_metric(metrics)        # 0〜100
    pullback_raw = _pullback_metric(metrics)  # 0〜100
    liq_raw = _liquidity_score(metrics)       # 0〜20

    m_component = max(0.0, min(20.0, market_score / 100.0 * 20.0))
    s_component = max(0.0, min(20.0, sector_strength / 100.0 * 20.0))
    t_component = max(0.0, min(20.0, trend_raw / 100.0 * 20.0))
    p_component = max(0.0, min(20.0, pullback_raw / 100.0 * 20.0))
    l_component = max(0.0, min(10.0, liq_raw / 20.0 * 10.0))

    if fundamental_score is None or not np.isfinite(fundamental_score):
        f_component = 5.0  # 情報なしなら中立寄り
    else:
        f_component = max(0.0, min(10.0, fundamental_score / 100.0 * 10.0))

    total = int(round(m_component + s_component + t_component + p_component + l_component + f_component))

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

    vol_ratio = metrics.get("vol_ratio20", np.nan)
    if np.isfinite(vol_ratio):
        if vol_ratio >= 2.0:
            comments.append("需給◎（出来高増）")
        elif vol_ratio >= 1.3:
            comments.append("需給◯")

    if fundamental_score is not None and np.isfinite(fundamental_score):
        if fundamental_score >= 70:
            comments.append("ファンダ◎")
        elif fundamental_score >= 50:
            comments.append("ファンダ◯")

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
    trend_raw = _trend_metric(metrics)        # 0〜100
    pullback_raw = _pullback_metric(metrics)  # 0〜100
    liq_raw = _liquidity_score(metrics)       # 0〜20

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
# メッセージ整形
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
        score = cnt * 10 + strength
        scored.append((sec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _build_priority_names(core_list: list[dict], short_list: list[dict]) -> list[str]:
    """
    最優先銘柄TOP3（Core優先・足りなければShortTerm補填）
    """
    top: list[str] = []

    for r in core_list:
        code = r["ticker"].replace(".T", "")
        top.append(f"{code} {r['name']}（Core {r['score']}）")
        if len(top) >= 3:
            return top

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
    LINE通知本文を 1 本のテキストとして返す。
    """
    label, regime_label, max_lev, max_pos, comment = _market_label_and_risk(market_score)

    ranked_sectors = _rank_sectors_from_candidates(core_list, short_list, sector_strength_map)
    if ranked_sectors:
        top_sector_name = ranked_sectors[0][0]
        top_sector_strength = sector_strength_map.get(top_sector_name, 50)
    else:
        top_sector_name = "なし"
        top_sector_strength = 0

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

    lines.append("◆ 今日のTOPセクター")
    if ranked_sectors:
        for i, (sec, score) in enumerate(ranked_sectors[:3], 1):
            strength = sector_strength_map.get(sec, 50)
            lines.append(f"{i}位: {sec}（強度 {strength}）")
    else:
        lines.append("セクター候補なし（Core/ShortTerm該当なし）")
    lines.append("")

    lines.append("◆ 今日の相場3行まとめ")
    three_lines = _build_three_line_summary(market_score, top_sector_name)
    lines.extend(three_lines)
    lines.append("")

    lines.append("◆ Core（本命候補）")
    if core_list:
        for i, r in enumerate(core_list, 1):
            code = r["ticker"]
            lines.append(f"{i}. {code} {r['name']} / {r['sector']}")
            lines.append(
                f"   Score {r['score']} / {r['comment']} / 現値: {r['price']}円"
            )
    else:
        lines.append("本命条件を満たす銘柄なし。今日は無理に攻めない選択もあり。")
    lines.append("")

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
