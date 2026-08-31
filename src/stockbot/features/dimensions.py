"""特徴量 D1〜D8（DESIGN.md §5 / TASKS.md T-204）。

- T までのデータのみ。スイング由来の量は確定済みスイングのみ（swings.swings_as_of
  経由）。週足は確定週のみ（indicators.weekly_ohlcv が T を含む週を落とす）
- 欠損は NaN のまま保持する（§5 共通ルール。0.5 に丸めるのは §6 の正規化側）
- D3〜D6 の多く（Lp/L0/leg 等を使うもの）は、押し目構造が無い（pullback["h0"] is None）
  場合は NaN になる。D6 は状態が 反発開始/ブレイク でなければ次元ごと欠損（§5 D6）

このモジュールは自分では指標を再取得しない部分（押し目構造そのもの）を除き、
SMA/ATR/BB幅など §5 の式に必要な指標は features.indicators を使って内部で計算する
（features.pullback.pullback_state の呼び出し結果だけを外から受け取る）。

d3_template（§7 テンプレート適合度、T-302）は scoring.template を使って計算する
（価格・出来高テンプレートの平均。窓が定義できない場合は NaN）。

d3_ma_dist の「着地 MA の種別」（集計用、DESIGN.md 表の注記）は特徴量スコアの対象
ではないため、戻り値のメイン表には入れず、2つ目の戻り値（extra dict）で返す。
種別の判定そのものは公開関数 landing_ma() に切り出してある（docs/SCREENER.md §3 の
配信記録が「止まった線」を同じ判定で記録するため）。

引数の open_/high/low/close/volume/dividends は 0 始まりの位置（swings.py と同じ規約）
で扱う、実日付の DatetimeIndex を持つ整列済み pandas.Series（週足集約・決算日数計算
に実日付が要る）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.jpx_lists import next_earnings_business_days
from ..scoring import template
from .indicators import atr_simple, atr_wilder, bb_width, sma, true_range, weekly_ohlcv
from .pullback import STATE_BOUNCE, STATE_BREAK, trigger_fired
from .swings import SWING_HIGH, swings_as_of

# ------------------------------------------------------------------ メタデータ
# (id, dimension, direction, band)
#   direction: "up"(↑) | "down"(↓) | "band"(∩、band=(lo,hi)) | "binary"(二値) | None(D8, 採点しない)
FEATURE_METADATA: list[tuple[str, str, Optional[str], Optional[tuple]]] = [
    ("d1_ma_stack", "D1", "up", None),
    ("d1_slope200", "D1", "up", None),
    ("d1_slope75", "D1", "up", None),
    ("d1_maturity", "D1", "band", (20, 150)),
    ("d1_leg_strength", "D1", "up", None),
    ("d2_rs60", "D2", "up", None),
    ("d2_rs120", "D2", "up", None),
    ("d2_rsline_pos", "D2", "up", None),
    ("d3_retrace", "D3", "band", (0.33, 0.50)),
    ("d3_depth_atr", "D3", "band", (1.5, 4.0)),
    ("d3_depth_pct", "D3", "band", (0.05, 0.10)),
    ("d3_duration", "D3", "band", (3, 15)),
    ("d3_maxdrop", "D3", "down", None),
    ("d3_slope", "D3", "down", None),
    ("d3_down_ratio", "D3", "band", (0.50, 0.75)),
    ("d3_lower_highs", "D3", "down", None),
    ("d3_hl_dist", "D3", "up", None),
    ("d3_ma_dist", "D3", "down", None),
    ("d3_position", "D3", "band", (0.10, 0.50)),
    ("d3_dev5", "D3", "band", (-0.08, -0.03)),
    ("d3_bad_news", "D3", "binary", None),
    ("d3_template", "D3", "up", None),
    ("d4_pb_ratio", "D4", "down", None),
    ("d4_pb_slope", "D4", "down", None),
    ("d4_bounce_vol", "D4", "up", None),
    ("d4_climax", "D4", "binary", None),
    ("d5_atr_ratio", "D5", "down", None),
    ("d5_range_contr", "D5", "down", None),
    ("d5_bbw_pct", "D5", "down", None),
    ("d5_step_ratio", "D5", "down", None),
    ("d6_trig_a", "D6", "up", None),
    ("d6_trig_b", "D6", "up", None),
    ("d6_break_r", "D6", "up", None),
    ("d6_bounce_str", "D6", "up", None),
    ("d6_wick", "D6", "up", None),
    ("d6_age", "D6", "down", None),
    ("d7_hl_streak", "D7", "up", None),
    ("d7_close_pos", "D7", "up", None),
    ("d7_weeks_above", "D7", "band", (4, 40)),
    ("earnings_days", "D8", None, None),
    ("exdiv_flag", "D8", None, None),
    ("limit_flag", "D8", None, None),
    ("regime", "D8", None, None),       # T-207 で実装。ここでは常に NaN
    ("breadth_75", "D8", None, None),   # T-207 で実装。ここでは常に NaN
    ("breadth_200", "D8", None, None),  # T-207 で実装。ここでは常に NaN
]
FEATURE_IDS = [m[0] for m in FEATURE_METADATA]
RESULT_COLS = ["id", "dimension", "direction", "band_lo", "band_hi", "value"]

D6_IDS = frozenset(m[0] for m in FEATURE_METADATA if m[1] == "D6")


# 「着地した線」の候補。DESIGN.md §0 が使うと決めた 4 本だけ（増やさない）
LANDING_MA_NAMES = ("SMA5", "SMA25", "SMA75", "SMA200")


def ma_values_at(close: pd.Series, pos: int) -> dict:
    """位置 pos における SMA5/25/75/200 の値（landing_ma に渡す形）。

    窓が満たない線は NaN になる（indicators.sma と同じ）。pos までの終値しか
    使わないので、系列を pos で切って計算しても同じ値になる。
    """
    out: dict = {}
    for name in LANDING_MA_NAMES:
        n = int(name[3:])
        series = sma(close, n)
        out[name] = float(series.iloc[pos]) if 0 <= pos < len(series) else np.nan
    return out


def landing_ma(lp_value: float, ma_values: dict, atr_t: float) -> dict:
    """押し安値 Lp が「どの線で止まったか」と、その線までの距離（DESIGN.md §5 D3 d3_ma_dist）。

    d3_ma_dist は「Lp に最も近い SMA との距離 /ATR」という 1 つの数値だが、運用では
    どの線で止まったか（線の名前そのもの）を記録して線ごとに集計したい
    （docs/SCREENER.md §3）。距離と線名を同時に返すのはこの関数だけにして、
    compute_dimensions（特徴量表）も配信記録も同じ判定を通す。

    ma_values は {"SMA5": 値, ...}（ma_values_at の出力）。NaN の線は候補から外す。
    距離が同じ線が複数ある場合は LANDING_MA_NAMES の並び（短い線が先）で最初のものを選ぶ。

    戻り値: {"landing_ma": 線名 or None, "landing_ma_value": 線の値 or NaN,
             "dist_atr": 距離/ATR or NaN}。候補が 1 本も無い、ATR が使えない、
    lp_value が NaN のいずれかなら全て None/NaN（例外にしない）。
    """
    none = {"landing_ma": None, "landing_ma_value": np.nan, "dist_atr": np.nan}
    if not (np.isfinite(lp_value) and np.isfinite(atr_t) and atr_t > 0):
        return none
    finite = [(name, float(ma_values[name])) for name in LANDING_MA_NAMES
              if name in ma_values and np.isfinite(ma_values[name])]
    if not finite:
        return none
    name, value = min(finite, key=lambda nv: abs(lp_value - nv[1]))
    return {"landing_ma": name, "landing_ma_value": value,
            "dist_atr": abs(lp_value - value) / atr_t}


def _regression_slope(y: np.ndarray) -> float:
    """y を等間隔の x=0..n-1 に単回帰したときの傾き（1点以下は NaN）。"""
    n = len(y)
    if n < 2 or np.any(np.isnan(y)):
        return np.nan
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _swing_before(alternated_swings: pd.DataFrame, t_pos: int, before_index: int,
                  kind: str) -> Optional[int]:
    """before_index より前の直近の確定スイング（種別 kind）。無ければ None。"""
    confirmed = swings_as_of(alternated_swings, t_pos)
    cand = confirmed[(confirmed["kind"] == kind) & (confirmed["index"] < before_index)]
    if cand.empty:
        return None
    return int(cand["index"].max())


def _find_crossover_above(close: np.ndarray, sma200: np.ndarray, t_pos: int,
                          cap: int) -> Optional[int]:
    """T 以前で最も新しい「Close が SMA200 を上抜けた日」を探す（cap 本より前は見ない）。"""
    start = max(1, t_pos - cap + 1)
    for t in range(t_pos, start - 1, -1):
        prev = t - 1
        if prev < 0:
            break
        c, cp = close[t], close[prev]
        s, sp = sma200[t], sma200[prev]
        if np.isnan(c) or np.isnan(cp) or np.isnan(s) or np.isnan(sp):
            continue
        if c >= s and cp < sp:
            return t
    return None


def compute_dimensions(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, dividends: pd.Series,
    alternated_swings: pd.DataFrame, pullback: dict, t_pos: int, k: int,
    idx_close: Optional[pd.Series] = None,
    earnings_schedule: Optional[pd.DataFrame] = None, ticker: Optional[str] = None,
    regime: Optional[str] = None, breadth_75: Optional[float] = None,
    breadth_200: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """T（t_pos）時点の D1〜D8 特徴量を計算する（DESIGN.md §5）。

    pullback は features.pullback.pullback_state() の戻り値（同じ t_pos, k で計算した
    もの）をそのまま渡す。idx_close は TOPIX 等の指数終値（ticker と同じ位置規約で
    整列済み、D2 用）。earnings_schedule/ticker は jpx_lists.load_earnings_schedule() の
    出力（D8 の earnings_days 用）。regime/breadth_75/breadth_200 は
    features.regime.regime_gauge()/compute_breadth() の出力（T-207、日次で1回だけ計算
    してすべての銘柄に共通して渡す想定）。どれも省略時は対応する特徴量が NaN になる。

    戻り値: (features_df[id, dimension, direction, band_lo, band_hi, value], extra)。
    features_df の id 集合は FEATURE_IDS と常に一致する（欠損は value=NaN の行として残る）。
    extra は d3_ma_dist の着地 MA 種別など、スコア対象外の補助情報。
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    o = open_.to_numpy(dtype=float)
    v = volume.to_numpy(dtype=float)

    sma5 = sma(close, 5)
    sma25 = sma(close, 25)
    sma75 = sma(close, 75)
    sma200 = sma(close, 200)
    atr14 = atr_wilder(high, low, close, 14)
    atr5 = atr_simple(high, low, close, 5)
    atr20 = atr_simple(high, low, close, 20)
    tr = true_range(high, low, close)
    bbw = bb_width(close, 20, 2.0)
    vol_sma20 = sma(volume, 20)

    sma5_a, sma25_a, sma75_a, sma200_a = (s.to_numpy(dtype=float) for s in (sma5, sma25, sma75, sma200))
    atr14_a = atr14.to_numpy(dtype=float)

    atr_t = atr14_a[t_pos] if t_pos < len(atr14_a) else np.nan

    values: dict[str, object] = {fid: np.nan for fid in FEATURE_IDS}
    extra: dict = {}

    state = pullback.get("state")
    h0, l0, lp = pullback.get("h0"), pullback.get("l0"), pullback.get("lp")
    has_structure = h0 is not None and l0 is not None and lp is not None

    # ---------------------------------------------------------------- D1
    if np.isfinite(atr_t) and atr_t > 0:
        if not (np.isnan(sma25_a[t_pos]) or np.isnan(sma75_a[t_pos]) or np.isnan(sma200_a[t_pos])):
            values["d1_ma_stack"] = ((sma25_a[t_pos] - sma75_a[t_pos]) / atr_t
                                     + (sma75_a[t_pos] - sma200_a[t_pos]) / atr_t)
        if t_pos - 30 >= 0 and not (np.isnan(sma200_a[t_pos]) or np.isnan(sma200_a[t_pos - 30])):
            values["d1_slope200"] = (sma200_a[t_pos] - sma200_a[t_pos - 30]) / atr_t
        if t_pos - 15 >= 0 and not (np.isnan(sma75_a[t_pos]) or np.isnan(sma75_a[t_pos - 15])):
            values["d1_slope75"] = (sma75_a[t_pos] - sma75_a[t_pos - 15]) / atr_t

    cross = _find_crossover_above(c, sma200_a, t_pos, cap=250)
    if cross is not None:
        values["d1_maturity"] = min(t_pos - cross, 250)

    if has_structure and pullback.get("leg_bars"):
        leg, leg_bars = pullback["leg"], pullback["leg_bars"]
        atr_h0 = atr14_a[h0] if h0 < len(atr14_a) else np.nan
        if np.isfinite(atr_h0) and atr_h0 > 0 and leg_bars:
            values["d1_leg_strength"] = (leg / atr_h0) / leg_bars

    # ---------------------------------------------------------------- D2
    if idx_close is not None:
        ic = idx_close.to_numpy(dtype=float)
        for n, fid in ((60, "d2_rs60"), (120, "d2_rs120")):
            if t_pos - n >= 0:
                c0, c1, i0, i1 = c[t_pos - n], c[t_pos], ic[t_pos - n], ic[t_pos]
                if min(c0, c1, i0, i1) > 0 and not np.isnan([c0, c1, i0, i1]).any():
                    values[fid] = float(np.log(c1 / c0) - np.log(i1 / i0))
        window = 60
        if t_pos - window + 1 >= 0:
            rs_line = c[t_pos - window + 1: t_pos + 1] / ic[t_pos - window + 1: t_pos + 1]
            if not np.isnan(rs_line).any():
                rmin, rmax = rs_line.min(), rs_line.max()
                if rmax > rmin:
                    values["d2_rsline_pos"] = float((rs_line[-1] - rmin) / (rmax - rmin))

    # ---------------------------------------------------------------- D3
    if has_structure:
        l0_low, lp_value = pullback["l0_low"], pullback["lp_value"]
        d = pullback["d"]
        values["d3_retrace"] = pullback["retrace"]
        values["d3_depth_atr"] = pullback["depth_atr"]
        values["d3_depth_pct"] = pullback["depth_pct"]
        values["d3_duration"] = d
        values["d3_position"] = pullback["position"]
        values["d3_dev5"] = pullback["dev5"]

        interval = range(h0 + 1, t_pos + 1)
        if len(interval):
            drops = [c[t - 1] - l[t] for t in interval if t - 1 >= 0]
            if drops and np.isfinite(atr_t) and atr_t > 0:
                values["d3_maxdrop"] = max(drops) / atr_t
            closes_win = c[h0 + 1: t_pos + 1]
            if np.isfinite(atr_t) and atr_t > 0:
                slope = _regression_slope(closes_win)
                if not np.isnan(slope):
                    values["d3_slope"] = slope / atr_t
            opens_win = o[h0 + 1: t_pos + 1]
            down_days = (closes_win < opens_win)
            if len(down_days):
                values["d3_down_ratio"] = float(down_days.mean())

            highs_in_interval = swings_as_of(alternated_swings, t_pos)
            highs_in_interval = highs_in_interval[
                (highs_in_interval["kind"] == SWING_HIGH)
                & (highs_in_interval["index"] > h0) & (highs_in_interval["index"] <= t_pos)
            ]
            values["d3_lower_highs"] = int(len(highs_in_interval))

            if np.isfinite(atr_t) and atr_t > 0:
                values["d3_hl_dist"] = (lp_value - l0_low) / atr_t

                landing = landing_ma(
                    lp_value,
                    {"SMA5": sma5_a[lp], "SMA25": sma25_a[lp],
                     "SMA75": sma75_a[lp], "SMA200": sma200_a[lp]},
                    atr_t,
                )
                values["d3_ma_dist"] = landing["dist_atr"]
                if landing["landing_ma"] is not None:
                    extra["d3_ma_dist_landing_ma"] = landing["landing_ma"]
                    extra["d3_ma_dist_landing_value"] = landing["landing_ma_value"]

            bad_news = False
            for t in interval:
                if t - 1 < 0:
                    continue
                atr_here = atr14_a[t] if t < len(atr14_a) else np.nan
                if np.isfinite(atr_here) and atr_here > 0 and (c[t - 1] - l[t]) > 2.5 * atr_here:
                    bad_news = True
                    break
                if o[t] < l[t - 1] * 0.95:
                    bad_news = True
                    break
            values["d3_bad_news"] = bad_news

        price_score = template.price_template_score(
            c, l0, h0, t_pos, l0_low, pullback["leg"], pullback["leg_bars"], d)
        volume_score = template.volume_template_score(v, l0, h0, t_pos, pullback["leg_bars"], d)
        values["d3_template"] = template.d3_template_score(price_score, volume_score)

    # ---------------------------------------------------------------- D4
    if has_structure:
        pre = v[l0: h0 + 1]
        post = v[h0 + 1: t_pos + 1]
        if pre.size and post.size and pre.mean() > 0:
            values["d4_pb_ratio"] = post.mean() / pre.mean()
        if post.size and post.mean() != 0:
            slope = _regression_slope(post)
            if not np.isnan(slope):
                values["d4_pb_slope"] = slope / post.mean()

        if post.size:
            climax_i = int(np.argmax(post)) + (h0 + 1)
            is_down_day = c[climax_i] < o[climax_i]
            vol_sma20_t = vol_sma20.iloc[climax_i]
            atr_climax = atr14_a[climax_i] if climax_i < len(atr14_a) else np.nan
            if (is_down_day and not np.isnan(vol_sma20_t) and v[climax_i] > 2 * vol_sma20_t
                    and np.isfinite(atr_climax) and atr_climax > 0
                    and abs(l[climax_i] - lp_value) <= atr_climax):
                values["d4_climax"] = True
            else:
                values["d4_climax"] = False

    if state in (STATE_BOUNCE, STATE_BREAK):
        vol_sma20_t = vol_sma20.iloc[t_pos]
        if not np.isnan(vol_sma20_t) and vol_sma20_t > 0:
            values["d4_bounce_vol"] = v[t_pos] / vol_sma20_t

    # ---------------------------------------------------------------- D5
    atr5_t, atr20_t = atr5.iloc[t_pos], atr20.iloc[t_pos]
    if not np.isnan(atr5_t) and not np.isnan(atr20_t) and atr20_t > 0:
        values["d5_atr_ratio"] = atr5_t / atr20_t

    if has_structure:
        d = pullback["d"]
        if d >= 8:
            interval_days = list(range(h0 + 1, t_pos + 1))
            first5, last5 = interval_days[:5], interval_days[-5:]
            tr_arr = tr.to_numpy(dtype=float)
            tr_first, tr_last = tr_arr[first5], tr_arr[last5]
            if not np.isnan(tr_first).any() and not np.isnan(tr_last).any() and tr_first.mean() > 0:
                values["d5_range_contr"] = tr_last.mean() / tr_first.mean()

    bbw_a = bbw.to_numpy(dtype=float)
    window = 120
    if t_pos - window + 1 >= 0:
        bbw_win = bbw_a[t_pos - window + 1: t_pos + 1]
        cur = bbw_win[-1]
        valid = bbw_win[~np.isnan(bbw_win)]
        if not np.isnan(cur) and valid.size:
            values["d5_bbw_pct"] = float((valid <= cur).mean())

    if has_structure:
        h0_prev = _swing_before(alternated_swings, t_pos, l0, SWING_HIGH)
        if h0_prev is not None:
            prev_window = l[h0_prev + 1: l0 + 1]
            if prev_window.size:
                prev_lp = float(prev_window.min())
                prev_h0_high = h[h0_prev]
                if prev_h0_high > 0:
                    prev_depth_pct = (prev_h0_high - prev_lp) / prev_h0_high
                    cur_depth_pct = pullback["depth_pct"]
                    if prev_depth_pct:
                        values["d5_step_ratio"] = cur_depth_pct / prev_depth_pct

    # ---------------------------------------------------------------- D6
    if has_structure and state in (STATE_BOUNCE, STATE_BREAK):
        sma5_arr = sma5_a
        values["d6_trig_a"] = bool(c[t_pos] > h[t_pos - 1] and c[t_pos] > sma5_arr[t_pos]) if t_pos >= 1 else False
        # trig_b は §4.3 の定義そのもの（trig_a とは独立に評価する）
        if t_pos >= 2:
            values["d6_trig_b"] = bool(
                sma5_arr[t_pos] > sma5_arr[t_pos - 1]
                and sma5_arr[t_pos - 1] <= sma5_arr[t_pos - 2]
                and c[t_pos] > sma5_arr[t_pos]
            )
        else:
            values["d6_trig_b"] = False
        r_value = pullback.get("r")
        values["d6_break_r"] = bool(r_value is not None and not np.isnan(r_value) and c[t_pos] > r_value)

        if np.isfinite(atr_t) and atr_t > 0:
            values["d6_bounce_str"] = (c[t_pos] - pullback["lp_value"]) / atr_t

        rng = h[lp] - l[lp]
        if rng > 0:
            values["d6_wick"] = (min(o[lp], c[lp]) - l[lp]) / rng

        trig_day = None
        for t in range(lp, t_pos + 1):
            if trigger_fired(c, sma5_a, h, t):
                trig_day = t
                break
        if trig_day is not None:
            values["d6_age"] = t_pos - trig_day

    # ---------------------------------------------------------------- D7
    daily = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                          "Volume": volume})
    asof = close.index[t_pos]
    weekly = weekly_ohlcv(daily, asof)
    if len(weekly):
        w_close = weekly["Close"].to_numpy(dtype=float)
        w_low = weekly["Low"].to_numpy(dtype=float)
        w_high = weekly["High"].to_numpy(dtype=float)

        streak = 0
        for i in range(len(weekly) - 1, 0, -1):
            if w_low[i] > w_low[i - 1]:
                streak += 1
                if streak >= 6:
                    break
            else:
                break
        values["d7_hl_streak"] = min(streak, 6)

        last_range = w_high[-1] - w_low[-1]
        if last_range > 0:
            values["d7_close_pos"] = (w_close[-1] - w_low[-1]) / last_range

        w_sma40 = sma(weekly["Close"], 40).to_numpy(dtype=float)
        above_streak = 0
        for i in range(len(weekly) - 1, -1, -1):
            if np.isnan(w_sma40[i]):
                break
            if w_close[i] > w_sma40[i]:
                above_streak += 1
            else:
                break
        values["d7_weeks_above"] = above_streak if above_streak > 0 else (
            0 if not np.isnan(w_sma40[-1]) else np.nan)

    # ---------------------------------------------------------------- D8
    interval_full = range((h0 + 1) if has_structure else t_pos, t_pos + 1)
    div = dividends.to_numpy(dtype=float)
    values["exdiv_flag"] = bool(any(div[t] > 0 for t in interval_full if t < len(div)))

    limit_flag = False
    for t in range(max(0, t_pos - 4), t_pos + 1):
        if t - 1 < 0:
            continue
        prev_c = c[t - 1]
        if prev_c == 0 or np.isnan(prev_c) or np.isnan(c[t]):
            continue
        if abs(c[t] / prev_c - 1) >= 0.15 and (c[t] == h[t] or c[t] == l[t]):
            limit_flag = True
            break
    values["limit_flag"] = limit_flag

    if earnings_schedule is not None and ticker is not None:
        days = next_earnings_business_days(earnings_schedule, asof, ticker)
        if days is not None:
            values["earnings_days"] = days
    if regime is not None:
        values["regime"] = regime
    if breadth_75 is not None and not np.isnan(breadth_75):
        values["breadth_75"] = breadth_75
    if breadth_200 is not None and not np.isnan(breadth_200):
        values["breadth_200"] = breadth_200

    rows = []
    for fid, dim, direction, band in FEATURE_METADATA:
        band_lo, band_hi = band if band is not None else (np.nan, np.nan)
        rows.append((fid, dim, direction, band_lo, band_hi, values[fid]))
    df = pd.DataFrame(rows, columns=RESULT_COLS)
    return df, extra
