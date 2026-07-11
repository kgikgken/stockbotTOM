"""モメンタム系テクニカル指標 — 全て日次OHLCVのみから計算(yfinance単一ソース前提).

歪み系(mispricing/indicators.py)とは独立。ATR/SMAの純粋な数学的ユーティリティのみ
共有元から再利用し、モメンタム固有の指標(ADX・12-1モメンタム・52週高値近接度・
相対強度・VCP収縮・ドンチアンブレイク・シャンデリア水準)はここに実装する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mispricing.indicators import sma, atr_wilder  # 純粋な数学ユーティリティのみ再利用


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    atr_n = tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean() / atr_n.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def chandelier_exit_long(high: pd.Series, atr_series: pd.Series, n: int, mult: float) -> pd.Series:
    """直近n日高値 − mult×ATR(n)。ロング用の可変トレーリングストップ水準。"""
    return high.rolling(n, min_periods=n).max() - mult * atr_series


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    return high.rolling(n, min_periods=n).max()


def vcp_contraction(high: pd.Series, low: pd.Series, lookback: int, ratio: float) -> pd.Series:
    """直近半分の値幅 vs 直近全体の値幅(いずれも前日までで評価し、当日のブレイク足自体を含めない)。"""
    rng = (high - low)
    recent = rng.rolling(lookback // 2, min_periods=lookback // 2).mean().shift(1)
    baseline = rng.rolling(lookback, min_periods=lookback).mean().shift(1)
    return recent / baseline.replace(0, np.nan) <= ratio


def compute_momentum_features(df: pd.DataFrame, bench_logclose: pd.Series | None, cfg) -> dict | None:
    """df: OHLCV daily。bench_logclose: ベンチマーク(TOPIX)のlog(close)系列(日付整列前)。"""
    if df is None or len(df) < max(cfg.regime_mom_days, 260) + 5:
        return None
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 260:
        return None

    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    close_now = float(c.iloc[-1])

    sma20 = sma(c, cfg.pullback_sma_slow)
    sma50, sma150, sma200 = sma(c, 50), sma(c, 150), sma(c, 200)
    atr_n = atr_wilder(h, l, c, cfg.atr_period)
    adx_n = adx(h, l, c, cfg.adx_period)
    chand = chandelier_exit_long(h, atr_n, cfg.atr_period, cfg.chandelier_mult)
    donch_prev = donchian_high(h, cfg.donchian_days).shift(1)  # 前日までのN日高値(当日ブレイク判定用)
    vcp_now = bool(vcp_contraction(h, l, cfg.vcp_lookback, cfg.vcp_contraction_ratio).iloc[-1])

    vmean20 = float(v.iloc[-21:-1].mean())
    vsd20 = float(v.iloc[-21:-1].std(ddof=0))
    vol_ratio_today = float(v.iloc[-1]) / vmean20 if vmean20 > 0 else np.nan

    # --- モメンタム総合スコア構成要素 ---
    logc = np.log(c)
    mom_12_1 = float(logc.shift(21).iloc[-1] - logc.shift(252).iloc[-1]) if len(c) > 252 else np.nan
    high52w = float(c.iloc[-252:].max())
    high52w_proximity = close_now / high52w if high52w > 0 else np.nan

    rel_strength = np.nan
    if bench_logclose is not None:
        s_now, s_now_al = logc.align(bench_logclose, join="inner")
        if len(s_now) > 130:
            stock_ret126 = float(s_now.iloc[-1] - s_now.iloc[-127])
            bench_ret126 = float(s_now_al.iloc[-1] - s_now_al.iloc[-127])
            rel_strength = stock_ret126 - bench_ret126

    trend_align = bool(close_now > sma50.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1]) \
        if not (pd.isna(sma50.iloc[-1]) or pd.isna(sma150.iloc[-1]) or pd.isna(sma200.iloc[-1])) else False
    breakdown = bool(close_now < sma(c, cfg.breakdown_sma).iloc[-1])
    bounce = bounce_confirmed(df, cfg.bounce_lookback_days, cfg.bounce_min_close_position)
    swing_high, days_since_high, depth_atr = swing_high_depth(df, atr_n, cfg.swing_high_lookback_days)

    adv20_jpy = float((c * v).iloc[-21:-1].mean())

    return {
        "close": close_now, "atr": float(atr_n.iloc[-1]), "adx": float(adx_n.iloc[-1]),
        "sma20": float(sma20.iloc[-1]),
        "sma50": float(sma50.iloc[-1]), "sma150": float(sma150.iloc[-1]), "sma200": float(sma200.iloc[-1]),
        "chandelier": float(chand.iloc[-1]), "donchian_prev": float(donch_prev.iloc[-1]) if not pd.isna(donch_prev.iloc[-1]) else np.nan,
        "vcp_now": vcp_now, "vol_ratio_today": vol_ratio_today,
        "mom_12_1": mom_12_1, "high52w_proximity": high52w_proximity, "rel_strength": rel_strength,
        "trend_align": trend_align, "breakdown": breakdown, "bounce_confirmed": bounce,
        "swing_high": swing_high, "days_since_swing_high": days_since_high, "pullback_depth_atr": depth_atr,
        "adv20_jpy": adv20_jpy,
        "last_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
    }


def swing_high_depth(df: pd.DataFrame, atr_series: pd.Series, lookback_days: int = 60) -> tuple[float, int, float]:
    """直近lookback_days日以内の高値(スイング高値)から現在までの下落を測る(押し目の深さ・期間の共有計算)。
    戻り値: (スイング高値, スイング高値からの経過営業日数, ATR倍数での下落幅)。
    調査レポート(2026-07-11)の「深さはATR正規化が望ましい」「継続期間20営業日以内」提案に対応。"""
    h, c = df["High"], df["Close"]
    window_h = h.iloc[-lookback_days:]
    swing_high = float(window_h.max())
    swing_high_idx = window_h.idxmax()
    days_since = int((df.index >= swing_high_idx).sum()) - 1
    close_now = float(c.iloc[-1])
    atr_now = float(atr_series.iloc[-1])
    depth_atr = (swing_high - close_now) / atr_now if atr_now > 0 else 0.0
    return swing_high, days_since, depth_atr


def bounce_confirmed(df: pd.DataFrame, lookback_days: int = 2, min_close_position: float = 0.5) -> bool:
    """押し目が「ゾーンに近いだけ」ではなく「実際に反発し始めている」かを確認する。

    ①当日の終値が高値-安値レンジの上位側(既定50%以上)で引けている(下げ止まりの気配)
    ②直近lookback_days日の終値変化がプラス(短期の向きが上を向いている)
    の両方を満たす場合のみTrueとする。まだ下げている途中の銘柄を「押し目ゾーン内」という
    理由だけで拾ってしまうのを防ぐための追加フィルター(単体の predictive signal としてではなく、
    既存のより実証的な条件—トレンド整列・業種内相対—の上に重ねるタイミング確認として使う)。
    """
    c, h, l = df["Close"], df["High"], df["Low"]
    if len(c) < lookback_days + 1:
        return False
    today_range = float(h.iloc[-1] - l.iloc[-1])
    if today_range <= 1e-9:
        return False
    close_position = float(c.iloc[-1] - l.iloc[-1]) / today_range
    short_term_turn = float(c.iloc[-1]) > float(c.iloc[-1 - lookback_days])
    return close_position >= min_close_position and short_term_turn


def bounce_confirmed_strong(df: pd.DataFrame) -> bool:
    """反転エンジン向けの強化反発確認: 前日高値を上抜けている(ミニブレイク型の反発)。
    調査レポートの「反転エンジンはより厳格な反発確認を推奨(CLV+前日高値上抜け)」に対応。"""
    c, h = df["Close"], df["High"]
    if len(c) < 2:
        return False
    return float(c.iloc[-1]) > float(h.iloc[-2])


def tob_suspect(df: pd.DataFrame, cfg) -> tuple[bool, str]:
    """公開買付(TOB)・MBO・M&A等のコーポレートアクション疑いを検出する簡易ヒューリスティック.

    2つの独立した経路で判定する:
    経路A(単日ジャンプ→ボラ収縮): 発表日に単日で大きく跳ね上がり、その後は成立を織り込んで
      買付価格に張り付き、ボラティリティが急激に低下するという典型的なTOBの値動きを検出。
    経路B(持続的な絶対ボラ圧縮・ジャンプ非依存): MBOはプレミアムが小さく段階的な値上がりで
      単日ジャンプとして検出できない場合や、発表がlookback_days(既定250営業日)より前で
      経路Aの観測窓に収まらない場合でも、「直近の変動率がそれ以前と比べ持続的に極端に低い」
      という状態(=価格が固定されている)を独立に検出する。

    いずれも通常のモメンタム(方向感を伴う継続的な値動き)とは性質が異なり、
    上値・下値とも買付条件に規定されるため、ATRベースの損切り・トレール設計が機能しない。
    価格データのみでの検出のため確定的ではなく、あくまで「疑い」の除外(要人間確認)。
    """
    c = df["Close"].dropna()
    if len(c) < cfg.tob_lookback_days + 40:
        return False, ""
    lr = np.log(c / c.shift(1)).dropna()

    # ---- 経路A: 単日ジャンプ→ボラ収縮(観測窓を250営業日に拡張済み) ----
    window = lr.iloc[-cfg.tob_lookback_days:]
    jump_idx = window.abs().idxmax()
    jump_ret = float(window.loc[jump_idx])
    if jump_ret >= cfg.tob_jump_threshold:
        days_since = int((c.index >= jump_idx).sum()) - 1
        if days_since >= 3:
            post = lr.loc[lr.index >= jump_idx].iloc[1:]
            if len(post) >= 3:
                post_vol = float(post.std(ddof=0))
                pre = lr.loc[lr.index < jump_idx].iloc[-40:]
                if len(pre) >= 15:
                    pre_vol = float(pre.std(ddof=0))
                    if pre_vol > 1e-9 and post_vol < pre_vol * cfg.tob_vol_collapse_ratio:
                        return True, (f"{days_since}営業日前に単日{jump_ret*100:+.0f}%の急騰後、"
                                      f"変動率が平常時の{post_vol/pre_vol*100:.0f}%に急低下"
                                      "(公開買付/M&A等で価格が固定されている疑い・要確認)")

    # ---- 経路B: 持続的な絶対ボラ圧縮(単日ジャンプの検出可否によらない・MBO等の緩やかな値動き対策) ----
    need = cfg.tob_sustained_baseline_days + cfg.tob_sustained_recent_days
    if len(lr) >= need:
        recent = lr.iloc[-cfg.tob_sustained_recent_days:]
        baseline = lr.iloc[-need:-cfg.tob_sustained_recent_days]
        if len(recent) >= 10 and len(baseline) >= 30:
            recent_vol = float(recent.std(ddof=0))
            baseline_vol = float(baseline.std(ddof=0))
            if baseline_vol > 1e-9 and recent_vol < baseline_vol * cfg.tob_sustained_vol_ratio:
                return True, (f"直近{cfg.tob_sustained_recent_days}営業日の変動率が、"
                              f"それ以前{cfg.tob_sustained_baseline_days}営業日の"
                              f"{recent_vol/baseline_vol*100:.0f}%まで持続的に低下"
                              "(単日ジャンプは観測窓内で未検出だが、MBO/TOB等で価格が"
                              "固定されている疑い・要確認)")

    # ---- 経路C: 直近の絶対的な値動きの薄さ(相対比較なしのシンプルな直接判定・新規) ----
    # 経路A/Bはいずれも「過去と比べて」という相対判定。経路Cは「今、単純に動いていない」を
    # 直接見る。相対比較のベースラインが取りにくいケースへの保険。
    recent_flat = lr.iloc[-cfg.tob_flat_recent_days:]
    if len(recent_flat) >= cfg.tob_flat_recent_days:
        flat_vol = float(recent_flat.std(ddof=0))
        if flat_vol < cfg.tob_flat_vol_threshold:
            return True, (f"直近{cfg.tob_flat_recent_days}営業日の変動率が{flat_vol*100:.2f}%と極めて低い"
                          "(絶対水準として値動きがほぼ無い。TOB/MBO等で価格が固定されている疑い・要確認)")

    return False, ""


def zscore_list(values: list) -> list:
    """母集団全体でz-score化。NaN/欠損は個別にNaNのまま返す(呼び出し側でその項を除外)。
    母数が少ない・無分散の場合は中立化(全て0)して数字の捏造を避ける。"""
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(arr) < 5 or float(arr.std(ddof=0)) < 1e-12:
        return [0.0 if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan for v in values]
    mu, sd = float(arr.mean()), float(arr.std(ddof=0))
    return [np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else (v - mu) / sd for v in values]


def compute_sector_strength(items: list, cfg) -> dict:
    """各業種の代表モメンタムを、構成銘柄の対TOPIX相対強度の中央値(価格データのみ)で算出し、
    業種間でz-score化して返す({セクター名: z値})。

    業種指数そのものは持たないため、構成銘柄の等ウェイト代理値として計算する
    (歪み系の資金循環マップ「業種指数=構成銘柄等ウェイト代理」と同じ考え方)。
    ニュース・決算等は一切読まない。「なぜ強いか」ではなく「価格上、強いかどうか」のみ検出する。
    構成銘柄が少なすぎる業種(既定3銘柄未満)は代表値として使わない(少数銘柄のノイズ排除)。
    """
    by_sector: dict = {}
    for it in items:
        sec = it["row"].get("sector") or "不明"
        rs = it["feat"]["rel_strength"]
        if rs is not None and not (isinstance(rs, float) and np.isnan(rs)):
            by_sector.setdefault(sec, []).append(rs)

    sector_medians = {sec: float(np.median(vals)) for sec, vals in by_sector.items()
                      if len(vals) >= cfg.sector_strength_min_members}
    if len(sector_medians) < 3:
        return {}  # 業種数が少なすぎてz化しても意味がない

    secs = list(sector_medians.keys())
    arr = np.array([sector_medians[s] for s in secs])
    sd = float(arr.std(ddof=0))
    if sd < 1e-12:
        return {s: 0.0 for s in secs}
    mu = float(arr.mean())
    return {s: (sector_medians[s] - mu) / sd for s in secs}


def compute_pool_scores(items: list, cfg) -> list:
    """items: [{'row':dict,'feat':dict}, ...] 流動性・TOB除外を通過した全銘柄。
    3つの連続値(12-1モメンタム・52週高値近接度・対TOPIX相対強度)を母集団z-score化してから
    加重合成する(指示①: 素点のまま合成すると単位が揃わず名目上の重みと実際の寄与度がズレるため)。
    トレンド整列ボーナスは二値のためz化対象外、素点のまま維持。
    ★セクター強度(価格データのみの機械的加点)を第5要素として追加。"""
    z_mom = zscore_list([it["feat"]["mom_12_1"] for it in items])
    z_h52 = zscore_list([it["feat"]["high52w_proximity"] for it in items])
    z_rs = zscore_list([it["feat"]["rel_strength"] for it in items])
    sector_z = compute_sector_strength(items, cfg)

    for it, zm, zh, zr in zip(items, z_mom, z_h52, z_rs):
        parts = []
        if not (isinstance(zm, float) and np.isnan(zm)):
            parts.append((zm, cfg.w_mom_12_1))
        if not (isinstance(zh, float) and np.isnan(zh)):
            parts.append((zh, cfg.w_high52w))
        if not (isinstance(zr, float) and np.isnan(zr)):
            parts.append((zr, cfg.w_relstrength))
        parts.append((1.0 if it["feat"]["trend_align"] else 0.0, cfg.w_trend_align))
        sec = it["row"].get("sector") or "不明"
        sz = sector_z.get(sec)
        if sz is not None:
            parts.append((sz, cfg.w_sector_strength))
        it["sector_strength_z"] = sz
        total_w = sum(w for _, w in parts)
        it["score"] = (sum(v * w for v, w in parts) / total_w) if total_w > 0 else float("-inf")
    return items


def momentum_score(feat: dict, cfg) -> float:
    """単銘柄の簡易スコア(母集団z化ができない文脈=保有銘柄チェック等でのフォールバック用)。
    候補プール選定の主経路は compute_pool_scores を使うこと。"""
    parts = []
    if not (feat["mom_12_1"] is None or np.isnan(feat["mom_12_1"])):
        parts.append(("mom_12_1", feat["mom_12_1"] * 100, cfg.w_mom_12_1))
    if not np.isnan(feat["high52w_proximity"]):
        parts.append(("high52w", feat["high52w_proximity"] * 100, cfg.w_high52w))
    if not np.isnan(feat["rel_strength"]):
        parts.append(("relstrength", feat["rel_strength"] * 100, cfg.w_relstrength))
    parts.append(("trend_align", 10.0 if feat["trend_align"] else 0.0, cfg.w_trend_align))
    if not parts:
        return float("-inf")
    total_w = sum(p[2] for p in parts)
    return sum(v * w for _, v, w in parts) / total_w if total_w > 0 else float("-inf")
