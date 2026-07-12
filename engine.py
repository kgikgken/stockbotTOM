"""ウォークフォワード・バックテスト・エンジン(共通コア).

設計原則(ルックアヘッド・バイアス防止が最優先):
- 各営業日Tの判定は、必ず「T日までのデータ(T日を含む)」のみを使う。df.iloc[:t+1]のスライスで実現。
- 判定に使う関数(classify_state等)は、本番(main_momentum.py等)と完全に同じものを再利用する。
  バックテスト専用に別ロジックを書くと、本番と違うものを検証してしまう("バックテストしたつもり")リスクがある。
- エントリーの約定は「シグナル発生日の翌営業日の始値」でシミュレートする(本番のレポートが
  前日終値ベースで生成され、寄り付き後に発注する運用と整合)。想定エントリーより大きく上に
  窓が開いた場合は見送る(本番のフラグ「寄り付きで大きく上に窓が開いていたら追撃しない」と整合)。
- 出口(ストップ・利確・時間ストップ・シャンデリア)は、日々の高値・安値・終値を使って判定する。

既知の限界(隠さず明記):
- yfinanceは生存者バイアスを含む(上場廃止銘柄が欠落)。この限界はバックテスト結果を
  楽観方向にバイアスさせる。
- 手数料・スプレッド・市場インパクトは簡易モデル(config経由)でしか近似できない。
- セクター相対指標(セクター強度・業種内相対z)は、その日時点で「取得できた」全銘柄から
  計算する。過去実際にはさらに多くの銘柄が存在した可能性があり、完全な再現ではない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ticker: str
    engine: str              # "A"/"B"(momentum) or "R"/"M"(swing2w)
    entry_date: str
    entry_price: float
    planned_entry: float     # シグナル発生日の終値(参考。実約定はentry_priceの方)
    stop: float
    target: Optional[float]  # momentumは固定利確なしのためNone
    shares: int
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # "stop"/"target"/"time"/"chandelier"/"end_of_data"
    r_multiple: Optional[float] = None
    days_held: Optional[int] = None


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    skipped_gap_chase: int = 0     # 窓開けで見送った件数(参考)
    signals_fired: int = 0         # シグナル自体が発生した回数(採用有無を問わず)


def _slice_upto(df: pd.DataFrame, t_idx: int) -> pd.DataFrame:
    """日付インデックスt_idx(0始まり)までのデータを返す(t_idxを含む、それより先は一切見せない)。"""
    return df.iloc[:t_idx + 1]


def run_walk_forward(
    ohlcv_full: Dict[str, pd.DataFrame],
    trading_dates: pd.DatetimeIndex,
    universe_rows: Dict[str, dict],
    signal_fn: Callable[[Dict[str, pd.DataFrame], int, pd.DatetimeIndex, dict], List[dict]],
    exit_fn: Callable[[Trade, pd.DataFrame, int], Optional[tuple]],
    cfg,
    min_lookback_days: int = 260,
    gap_chase_guard_pct: float = 3.0,
) -> BacktestResult:
    """共通の日次ウォークフォワード・ループ。

    signal_fn(ohlcv_full, t_idx, trading_dates, cfg) -> [{'ticker':..., 'engine':..., 'entry':...,
        'stop':..., 'target':Optional, 'name':..., 'sector':...}, ...]
        T日時点で新規に点灯した候補のリストを返す(本番のclassify関数をラップして呼ぶ)。
        signal_fn内部で必ずohlcv_full[ticker]をt_idxまでにスライスしてから本番関数に渡すこと。

    exit_fn(trade, df_ticker, t_idx) -> (exit_price, exit_reason) or None
        保有中ポジションについて、T日時点で手仕舞い条件に該当するか判定する。
        該当しなければNone。df_tickerはt_idxまでスライド済みのものを渡す。
    """
    result = BacktestResult()
    open_trades: Dict[str, Trade] = {}
    sector_open_count: Dict[str, int] = {}
    pending_entries: List[dict] = []  # 前日にシグナル発生、翌日寄付きで約定待ちのもの

    for t_idx in range(min_lookback_days, len(trading_dates)):
        today = trading_dates[t_idx]
        today_str = str(today.date())

        # ---- ①前日シグナル分の約定処理(寄り付きで約定 or 窓開けで見送り) ----
        still_pending = []
        for sig in pending_entries:
            tkr = sig["ticker"]
            df = ohlcv_full.get(tkr)
            if df is None or today not in df.index:
                continue  # データ欠落日はスキップ(翌日以降に持ち越さない=機会損失として記録しない簡易処理)
            open_px = float(df.loc[today, "Open"])
            planned = sig["entry"]
            if planned <= 0:
                continue
            gap_pct = (open_px / planned - 1) * 100
            if gap_pct > gap_chase_guard_pct:
                result.skipped_gap_chase += 1
                continue
            if tkr in open_trades:
                continue  # 既に保有中なら二重エントリーしない
            if len(open_trades) >= cfg.max_positions:
                continue
            sec = sig.get("sector", "不明")
            if sector_open_count.get(sec, 0) >= cfg.max_per_sector:
                continue
            risk_w = open_px - sig["stop"]
            if risk_w <= 0:
                continue
            risk_amount = cfg.account_equity * cfg.risk_pct_fixed / 100 if cfg.account_equity > 0 else 0
            shares = int(risk_amount / risk_w // 100 * 100) if risk_amount > 0 else 100
            tr = Trade(ticker=tkr, engine=sig["engine"], entry_date=today_str, entry_price=open_px,
                      planned_entry=planned, stop=sig["stop"], target=sig.get("target"),
                      shares=max(shares, 100))
            open_trades[tkr] = tr
            sector_open_count[sec] = sector_open_count.get(sec, 0) + 1

        # ---- ②保有中ポジションの手仕舞い判定 ----
        closed_tickers = []
        for tkr, tr in open_trades.items():
            df = ohlcv_full.get(tkr)
            if df is None or today not in df.index:
                continue
            df_upto = _slice_upto(df, df.index.get_loc(today))
            out = exit_fn(tr, df_upto, len(df_upto) - 1)
            if out is not None:
                exit_price, reason = out
                tr.exit_date = today_str
                tr.exit_price = exit_price
                tr.exit_reason = reason
                risk_w = tr.entry_price - tr.stop
                tr.r_multiple = (exit_price - tr.entry_price) / risk_w if risk_w > 0 else 0.0
                entry_dt = pd.Timestamp(tr.entry_date)
                tr.days_held = int(np.busday_count(entry_dt.date(), today.date()))
                result.trades.append(tr)
                closed_tickers.append(tkr)
                sec = universe_rows.get(tkr, {}).get("sector", "不明")
                sector_open_count[sec] = max(0, sector_open_count.get(sec, 0) - 1)
        for tkr in closed_tickers:
            del open_trades[tkr]

        # ---- ③本日時点の新規シグナル検出(翌営業日に約定させるためpendingに積む) ----
        new_signals = signal_fn(ohlcv_full, t_idx, trading_dates, cfg)
        result.signals_fired += len(new_signals)
        pending_entries = [s for s in new_signals if s["ticker"] not in open_trades]

    # ---- 期間末に残った建玉は「データ終端」として最終日終値で強制クローズ(未確定として区別) ----
    last_idx = len(trading_dates) - 1
    last_date = trading_dates[last_idx]
    for tkr, tr in open_trades.items():
        df = ohlcv_full.get(tkr)
        if df is None or last_date not in df.index:
            continue
        exit_price = float(df.loc[last_date, "Close"])
        tr.exit_date = str(last_date.date())
        tr.exit_price = exit_price
        tr.exit_reason = "end_of_data"
        risk_w = tr.entry_price - tr.stop
        tr.r_multiple = (exit_price - tr.entry_price) / risk_w if risk_w > 0 else 0.0
        entry_dt = pd.Timestamp(tr.entry_date)
        tr.days_held = int(np.busday_count(entry_dt.date(), last_date.date()))
        result.trades.append(tr)

    return result
