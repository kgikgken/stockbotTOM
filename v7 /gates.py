"""v7.1 第5部 — 前段ゲート(L0 レジーム判定 / L1 ユニバース)。

L0の役割は初版から反転している(§5.1.1)。
  初版: 地合い悪化を検知して停止する(根拠=DD・階層E)
  v7.1: トレンド戦略が報われる「継続局面」でのみ稼働する(根拠=Hanauer 2014・階層A)

「日本のモメンタムが弱いのは転換局面が多いから。継続局面に限定すれば弱くない」
というのがL0の存在理由であり、②の弱さ(階層C)に対する構造的な回答でもある。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import sma, to_weekly, slope_normalized

RISK_ON, NEUTRAL, RISK_OFF = "RISK_ON", "NEUTRAL", "RISK_OFF"


# ============================================================
# L0-A【主】指数トレンド判定(階層A)
#   レジームの切り替えはL0-Aのみが行う(§5.1.2 規則1)
# ============================================================

def l0_regime(index_daily: pd.DataFrame, cfg, prev_state: str | None = None) -> dict:
    """TOPIX(または代理指数)の週足30週線に対する位置と傾きでレジームを決める。

    ヒステリシス(3%バンド／5日確認)は、単なるノイズ除去ではなく
    「転換局面の検出」として理論的正当化を得ている(§5.1.3)。
      - 30週線が明確に上向き＋株価が上 → 継続局面
      - 30週線付近で往復 → 転換局面(＝新規停止)
    """
    try:
        w = to_weekly(index_daily)
    except Exception:
        return {"state": NEUTRAL, "reason": "週足変換失敗", "confidence": "低"}

    n = cfg.l0_weekly_ma
    if len(w) < n + cfg.l0_slope_weeks:
        return {"state": NEUTRAL, "reason": f"週足{len(w)}本(必要{n + cfg.l0_slope_weeks})",
                "confidence": "低"}

    ma = sma(w["Close"], n)
    if pd.isna(ma.iloc[-1]):
        return {"state": NEUTRAL, "reason": "30週線算出不可", "confidence": "低"}

    ma_now = float(ma.iloc[-1])
    close_w = float(w["Close"].iloc[-1])
    slope = slope_normalized(ma, cfg.l0_slope_weeks)
    if slope is None:
        return {"state": NEUTRAL, "reason": "傾き算出不可", "confidence": "低"}

    # ヒステリシス: 30週線から±バンド以内は「転換局面」として中立に倒す
    band = cfg.l0_hysteresis_band
    dev = (close_w - ma_now) / ma_now if ma_now > 0 else 0.0

    if dev > band and slope > 0:
        state = RISK_ON
        reason = f"継続局面(30週線+{dev*100:.1f}% 傾き{slope*100:+.2f}%/週)"
    elif dev < -band and slope < 0:
        state = RISK_OFF
        reason = f"下降継続(30週線{dev*100:.1f}% 傾き{slope*100:+.2f}%/週)"
    else:
        state = NEUTRAL
        reason = f"転換局面(30週線{dev*100:+.1f}%・バンド±{band*100:.0f}%内 or 傾き不一致)"

    # ★asofは週足ラベル(その週の金曜)ではなく、実際にデータが含まれる最終日を返す。
    # resample("W-FRI")は週の途中でもラベルが金曜になるため、そのまま出すと
    # 「レジーム基準日が価格基準日より未来」に見え、先読みを疑わせる表示になる
    # (データ自体は正しく過去のみを使っている)。§6.2規則5の趣旨に沿って実日付を出す。
    try:
        asof = str(index_daily.index[-1].date())
    except Exception:
        asof = str(w.index[-1].date())
    return {"state": state, "reason": reason, "confidence": "中",
            "dev": dev, "slope": slope, "ma": ma_now, "close": close_w,
            "asof": asof}


def l0_distribution_days(index_daily: pd.DataFrame, cfg) -> dict:
    """L0-B【補助】ディストリビューションデー(階層E)。

    §5.1.2 規則3: 日本株で有意性が確認されるまで寄与度ゼロでよい。
    既定で無効。単独でレジームを切り替えることは禁止されている。
    """
    if not cfg.l0_dd_enabled:
        return {"count": None, "tag": None, "note": "無効(階層E・自前検証まで寄与度ゼロ)"}
    try:
        c = index_daily["Close"].dropna()
        v = index_daily["Volume"].dropna()
        w = min(cfg.l0_dd_window, len(c) - 1)
        ret = c.pct_change().iloc[-w:]
        vv = v.iloc[-w:]
        vprev = v.shift(1).iloc[-w:]
        dd = int(((ret <= -cfg.l0_dd_drop_pct) & (vv > vprev)).sum())
        tag = "低" if dd >= cfg.l0_dd_cluster else "中"
        return {"count": dd, "tag": tag,
                "note": f"直近{w}営業日でDD{dd}回(階層E・信頼度タグのみ)"}
    except Exception:
        return {"count": None, "tag": None, "note": "算出不可"}


# ============================================================
# L1 投資可能ユニバース
#   Amihud ILLIQ(階層A)を主指標とする(§5.2.2)
# ============================================================

def amihud_illiq(daily: pd.DataFrame, window: int) -> float | None:
    """Amihud非流動性 = 平均(|日次リターン| ÷ 日次売買代金)。

    「売買代金1円あたりの価格反応」。値が大きいほど非流動的。
    単純な売買代金下限より査読的裏付けがある(§5.2.2)。
    """
    try:
        c = daily["Close"].dropna()
        v = daily["Volume"].dropna()
        idx = c.index.intersection(v.index)
        c, v = c.loc[idx], v.loc[idx]
        if len(c) < window + 1:
            return None
        ret = c.pct_change().abs().iloc[-window:]
        turnover = (c * v).iloc[-window:]
        valid = turnover > 0
        if valid.sum() < window * 0.5:
            return None
        # 単位が極小になるため1e9倍して扱いやすくする(相対比較のみに使う)
        return float((ret[valid] / turnover[valid]).mean() * 1e9)
    except Exception:
        return None


def build_universe(universe: pd.DataFrame, ohlcv: dict, cfg) -> tuple[list, list]:
    """L1: 流動性・価格・上場期間でユニバースを絞る。

    ★制約: yfinanceではポイントインタイム構築ができない(§10.2)。
    現在上場している銘柄しか取得できないため、生存バイアスは除去できない。
    この事実は出力に恒久表示する。
    """
    rows, drops = [], []
    for _, r in universe.iterrows():
        tkr = str(r["ticker"]).strip()
        df = ohlcv.get(tkr)
        if df is None or len(df) < cfg.l1_min_listed_days:
            drops.append({"code": tkr.replace(".T", ""), "reason": "履歴不足/未取得"})
            continue
        c = df["Close"].dropna()
        v = df["Volume"].dropna()
        if len(c) < cfg.l1_min_listed_days or len(v) < 21:
            drops.append({"code": tkr.replace(".T", ""), "reason": "履歴不足"})
            continue
        close_now = float(c.iloc[-1])
        if close_now < cfg.l1_min_price:
            drops.append({"code": tkr.replace(".T", ""), "reason": "価格下限"})
            continue
        adv = float((c * v).iloc[-21:-1].mean())
        if adv < cfg.l1_min_adv_jpy:
            drops.append({"code": tkr.replace(".T", ""), "reason": "売買代金下限"})
            continue
        illiq = amihud_illiq(df, cfg.l1_illiq_window)
        rows.append({"row": r.to_dict(), "df": df, "close": close_now,
                     "adv": adv, "illiq": illiq})

    # Amihud ILLIQ の上位(=非流動的な側)を落とす。パーセンタイルで相対判定する
    vals = [x["illiq"] for x in rows if x["illiq"] is not None]
    if vals:
        cut = float(np.percentile(vals, cfg.l1_illiq_percentile))
        kept = []
        for x in rows:
            if x["illiq"] is not None and x["illiq"] > cut:
                drops.append({"code": x["row"]["ticker"].replace(".T", ""),
                              "reason": f"Amihud非流動(上位{100-cfg.l1_illiq_percentile:.0f}%)"})
                continue
            kept.append(x)
        rows = kept
    return rows, drops


# ============================================================
# 市場ファクター(②の基準)
#   TOPIXは構成銘柄が段階的に縮小し非定常(§10.4)。
#   「その時点の全上場銘柄の加重平均」を自前構築するのが仕様書の推奨。
# ============================================================

def build_market_index(ohlcv: dict, min_names: int = 30) -> pd.Series | None:
    """ユニバース全銘柄の等加重平均リターンから自前の市場指数を構築する。

    §10.4.3: TOPIXは構成銘柄が2,136→約1,700→約1,200と縮小するため、
    長期の市場ファクターとしては非定常。自前構築の方が一貫性が高い。
    """
    series = []
    for df in ohlcv.values():
        try:
            c = df["Close"].dropna()
            if len(c) > 260:
                series.append(c.pct_change())
        except Exception:
            continue
    if len(series) < min_names:
        return None
    mat = pd.concat(series, axis=1)
    avg_ret = mat.mean(axis=1).fillna(0.0)
    return (1.0 + avg_ret).cumprod() * 1000.0
