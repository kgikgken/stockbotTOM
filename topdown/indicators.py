"""topdown用の特徴量 — 日次OHLCVのみから計算(自立版・旧パッケージ非依存)。

カタリスト代理(ギャップ+出来高)・節目ブレイク・S高急騰検知・セクター騰落を実装。
押し目判定は旧momentum凍結版の5ゲートをtopdown内に直接実装した(2026-07-17自立化):
  ①トレンド整列(終値>50日>150日>200日) ②非対称押し目ゾーン(20日線の上+2.5%〜下-5%)
  ③深さ上限(スイング高値からATR22×3以内・momentum凍結値に忠実にATR22を使用)
  ④継続期間上限(スイング高値から35営業日以内) ⑤反発確認(CLV≥0.5相当+直近2日プラス転換)
※旧実装との差分1点のみ: 旧経路はVCPブレイク(状態B)判定が押し目より優先されたが、topdownでは
  BREAKトリガーが先に評価されるため実質同等。それ以外のゲート値・計算は同一。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ta import sma, atr_wilder, tob_suspect, bounce_confirmed, swing_high_depth  # noqa: F401


def compute_topdown_features(df: pd.DataFrame, cfg) -> dict | None:
    if df is None or len(df) < 270:
        return None
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return None
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 260:
        return None

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    close_now = float(c.iloc[-1])
    atr_n = atr_wilder(h, l, c, cfg.atr_period)          # 損切り・利確設計用(ATR14)
    atr22 = atr_wilder(h, l, c, 22)                       # 押し目深さゲート用(momentum凍結値に忠実)
    sma20, sma50 = sma(c, 20), sma(c, 50)
    sma150, sma200 = sma(c, 150), sma(c, 200)
    lr = np.log(c / c.shift(1)).dropna()

    vmean20 = float(v.iloc[-21:-1].mean())
    vol_ratio_today = float(v.iloc[-1]) / vmean20 if vmean20 > 0 else np.nan

    # --- カタリスト代理: 直近N営業日以内の単日ギャップ+当該日の出来高急増 ---
    window = min(cfg.gap_max_days_since + 1, len(lr))
    recent_lr = lr.iloc[-window:]
    gap_found, gap_days_since, gap_ret, gap_vol_ratio = False, None, None, None
    gap_high = gap_low = None; gap_date = None
    if len(recent_lr):
        idx_max = recent_lr.idxmax()
        cand = float(recent_lr.loc[idx_max])
        pos = v.index.get_loc(idx_max)
        vmean_at = float(v.iloc[max(0, pos - 21):pos - 1].mean()) if pos >= 22 else np.nan
        vr_at = float(v.iloc[pos]) / vmean_at if (vmean_at and vmean_at > 0) else np.nan
        if cand >= cfg.gap_threshold and vr_at and vr_at >= cfg.gap_vol_mult:
            gap_found, gap_days_since, gap_ret, gap_vol_ratio = True, int((lr.index >= idx_max).sum()) - 1, cand, vr_at
            gap_high = float(h.loc[idx_max]); gap_low = float(l.loc[idx_max])
            gap_date = str(idx_max.date()) if hasattr(idx_max, "date") else str(idx_max)

    # --- 節目ブレイク: 前日までの20日高値を終値で上抜け+出来高+上昇トレンド ---
    breakout_found = False
    breakout_level = None; pre_breakout_low = None
    trend_up_simple = (not pd.isna(sma50.iloc[-1]) and not pd.isna(sma200.iloc[-1])
                       and close_now > float(sma50.iloc[-1]) > float(sma200.iloc[-1]))
    if len(h) > cfg.breakout_lookback + 1:
        prev_high = float(h.iloc[-cfg.breakout_lookback - 1:-1].max())
        if close_now > prev_high and vol_ratio_today and vol_ratio_today >= cfg.breakout_vol_mult and trend_up_simple:
            breakout_found = True
            breakout_level = prev_high
            # ブレイク直前の押し安値: 直近10営業日(当日除く)の安値
            pre_breakout_low = float(l.iloc[-11:-1].min())

    # --- S高・急騰済み検知(寄り天リスク→監視格下げ) ---
    chg1d = float(lr.iloc[-1]) if len(lr) else 0.0
    chg3d = float(lr.iloc[-3:].sum()) if len(lr) >= 3 else 0.0
    spiked = chg1d >= np.log(1 + cfg.spike_1d_threshold) or chg3d >= np.log(1 + cfg.spike_3d_threshold)

    # --- 押し目型: 凍結5ゲート(topdown内直接実装・自立版) ---
    pullback_state_a = False
    dip_low = None; prev_day_high = None
    if not any(pd.isna(x.iloc[-1]) for x in (sma20, sma50, sma150, sma200)):
        trend_align = close_now > float(sma50.iloc[-1]) > float(sma150.iloc[-1]) > float(sma200.iloc[-1])
        ratio20 = close_now / float(sma20.iloc[-1]) - 1
        in_zone = (-cfg.pullback_lower_pct / 100) <= ratio20 <= (cfg.pullback_upper_pct / 100)
        _, days_since_high, depth_atr22 = swing_high_depth(df, atr22, cfg.swing_high_lookback_days)
        depth_ok = depth_atr22 <= cfg.pullback_depth_atr_mult
        duration_ok = days_since_high <= cfg.pullback_max_duration_days
        bounce = bounce_confirmed(df, cfg.bounce_lookback_days, cfg.bounce_min_close_position)
        pullback_state_a = bool(trend_align and in_zone and depth_ok and duration_ok and bounce)
        if pullback_state_a:
            # 押し安値 = スイング高値以降の最安値(最低でも直近5営業日から取る)
            look = max(int(days_since_high) + 1, 5)
            dip_low = float(l.iloc[-look:].min())
            prev_day_high = float(h.iloc[-2]) if len(h) >= 2 else None

    adv20_jpy = float((c * v).iloc[-21:-1].mean())
    ret5d = float(np.log(close_now / float(c.iloc[-6]))) if len(c) > 6 else np.nan

    return {
        "close": close_now, "atr": float(atr_n.iloc[-1]),
        "vol_ratio_today": vol_ratio_today, "adv20_jpy": adv20_jpy,
        "gap_found": gap_found, "gap_days_since": gap_days_since, "gap_ret": gap_ret,
        "gap_vol_ratio": gap_vol_ratio,
        "breakout_found": breakout_found, "spiked": spiked,
        "chg1d_pct": (np.exp(chg1d) - 1) * 100, "ret5d": ret5d,
        "pullback_state_a": pullback_state_a,
        "gap_high": gap_high, "gap_low": gap_low, "gap_date": gap_date,
        "breakout_level": breakout_level, "pre_breakout_low": pre_breakout_low,
        "dip_low": dip_low, "prev_day_high": prev_day_high,
        "earnings_est_days": estimate_days_to_earnings(df, cfg),
        "last_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
    }


def compute_sector_rank(items: list, cfg) -> dict:
    """STEP2: 東証33業種の直近N日騰落(構成銘柄等ウェイト代理)。
    戻り値: {"top": [(sector, ret%), ...], "bottom": [...], "by_sector": {sector: ret%}}"""
    by_sector: dict = {}
    for it in items:
        sec = it["row"].get("sector") or "不明"
        r = it["feat"].get("ret5d")
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            by_sector.setdefault(sec, []).append(r)
    med = {s: float(np.median(v)) * 100 for s, v in by_sector.items() if len(v) >= cfg.sector_min_members}
    ranked = sorted(med.items(), key=lambda x: -x[1])
    return {"top": ranked[: cfg.sector_top_n], "bottom": ranked[-cfg.sector_bottom_n:] if len(ranked) >= cfg.sector_bottom_n else [],
            "by_sector": med}


def estimate_days_to_earnings(df: pd.DataFrame, cfg) -> int | None:
    """出来高スパイクの周期から次回決算までの営業日数を推定する(警告用・精度は中程度)。

    日本企業の決算は四半期ごと(約60営業日間隔)でほぼ同じ時期に来る。過去2年の
    「出来高が20日平均の2.5倍以上に跳ねた日」を決算候補日とみなし、直近の候補日+60営業日を
    次回発表の推定日とする。指数リバランス・突発ニュース等の出来高も混入するため、
    あくまで「可能性がある」という警告に留め、除外判定には使わない(最終確認はiSPEED)。
    戻り値: 次回までの推定営業日数。推定できない場合はNone。
    """
    if not getattr(cfg, "earnings_est_enabled", True):
        return None
    try:
        v = df["Volume"].dropna()
        if len(v) < 260:
            return None
        vm = v.rolling(20).mean().shift(1)
        ratio = (v / vm).dropna()
        spikes = ratio[ratio >= cfg.earnings_spike_vol_mult]
        if len(spikes) < 2:
            return None
        pos = [v.index.get_loc(i) for i in spikes.index]
        # 直近2年(約500営業日)以内のスパイクのみ使う
        last = len(v) - 1
        pos = [p for p in pos if last - p <= 500]
        if not pos:
            return None
        # 四半期周期に合致するスパイクだけを決算候補として残す
        cycle = cfg.earnings_cycle_days
        latest = max(pos)
        est = latest + cycle
        # 推定日が既に過ぎている場合は次の周期へ送る
        while est <= last:
            est += cycle
        days = int(est - last)
        return days if 0 < days <= cycle else None
    except Exception:
        return None
