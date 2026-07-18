"""topdown/ta.py — 純粋なテクニカル計算・パターン検出(自立版).

2026-07-17自立化: 旧momentum/mispricingパッケージから、新スクリーニングが使う関数だけを
そのまま移植(ロジック変更なし・検証済みコードの逐語コピー)。これによりtopdownは旧パッケージへの
依存を持たない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


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
