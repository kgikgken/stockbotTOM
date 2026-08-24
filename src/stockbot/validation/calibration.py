"""西村ルールの再現・較正（DESIGN.md §10.2(b) / TASKS.md T-404）。

DESIGN.md §10.2(b) のルール（Close[T]<=0.95*SMA5[T] かつ Close[T]>SMA75[T] で
T+1 寄付き買い、Close>SMA5 で翌日寄付き手仕舞い、最長15日）を
validation.layer1.nishimura_trades/nishimura_summary でそのまま売買シミュレーション
し、出典の数値（西村、2000〜2020・全銘柄・上場廃止込み。RESEARCH.md §13の等級付け
参照。査読なしの実務検証）と並べる。

CLAUDE.md「検証結果を解釈しない」に従い、差分を数値で示すだけで良否は判断しない。

CLAUDE.md の絶対規則（ホールドアウトを見ない）を守るため、渡された ohlcv は
HOLDOUT_WINDOW[0]（2026-02-01）以降のデータを内部で必ず切り捨ててから使う
（呼び出し側が誤って渡しても安全なよう防御的に行う）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .layer1 import _df_to_markdown, nishimura_summary, nishimura_trades
from .replay import HOLDOUT_WINDOW

# 出典: 西村ルール（2000〜2020・全銘柄・上場廃止込み。DESIGN.md §10.2(b)/RESEARCH.md）
SOURCE_LABEL = "出典(西村, 2000-2020, 全銘柄・上場廃止込み)"
SOURCE_WIN_RATE = 0.6274
SOURCE_AVG_RETURN = 0.0145
SOURCE_PROFIT_FACTOR = 1.593
SOURCE_AVG_HOLD_DAYS = 6.72

# TASKS.md T-404: 出典の期間と重なる部分（2016-2020）と、そうでない部分（主評価窓側）
DEFAULT_WINDOWS: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = [
    ("2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
    ("2021-2026", pd.Timestamp("2021-01-01"), HOLDOUT_WINDOW[0] - pd.Timedelta(days=1)),
]

RESULT_COLS = ["window", "n_trades", "win_rate", "win_rate_diff", "avg_return", "avg_return_diff",
              "profit_factor", "profit_factor_diff", "avg_hold_days", "avg_hold_days_diff"]


def _truncate_before_holdout(ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """CLAUDE.md 絶対規則: ホールドアウト（2026-02〜2026-08）は明示フラグ無しで見ない。
    このモジュールにホールドアウトを見せる経路自体が無い（フラグは提供しない）。
    """
    out = {}
    for ticker, df in ohlcv.items():
        if df is None or len(df) == 0:
            out[ticker] = df
            continue
        out[ticker] = df[df.index < HOLDOUT_WINDOW[0]]
    return out


def run_calibration(
    ohlcv: Dict[str, pd.DataFrame], tickers: Iterable[str],
    windows: Optional[List[Tuple[str, pd.Timestamp, pd.Timestamp]]] = None,
    max_hold: int = 15,
) -> pd.DataFrame:
    """窓ごとに西村ルールを再現し、出典との差分を並べた表を返す（列は RESULT_COLS）。
    先頭行は出典そのもの（差分列は NaN）。
    """
    windows = windows if windows is not None else DEFAULT_WINDOWS
    tickers = list(tickers)
    safe_ohlcv = _truncate_before_holdout(ohlcv)

    rows = [{
        "window": SOURCE_LABEL, "n_trades": np.nan,
        "win_rate": SOURCE_WIN_RATE, "win_rate_diff": np.nan,
        "avg_return": SOURCE_AVG_RETURN, "avg_return_diff": np.nan,
        "profit_factor": SOURCE_PROFIT_FACTOR, "profit_factor_diff": np.nan,
        "avg_hold_days": SOURCE_AVG_HOLD_DAYS, "avg_hold_days_diff": np.nan,
    }]

    for name, start, end in windows:
        end = min(pd.Timestamp(end), HOLDOUT_WINDOW[0] - pd.Timedelta(days=1))
        trades = nishimura_trades(safe_ohlcv, tickers, pd.Timestamp(start), end, max_hold=max_hold)
        summary = nishimura_summary(trades)
        avg_return = float(trades["return"].mean()) if len(trades) else np.nan
        pf = summary["profit_factor"]
        rows.append({
            "window": name, "n_trades": summary["n_trades"],
            "win_rate": summary["win_rate"],
            "win_rate_diff": (summary["win_rate"] - SOURCE_WIN_RATE
                              if not np.isnan(summary["win_rate"]) else np.nan),
            "avg_return": avg_return,
            "avg_return_diff": avg_return - SOURCE_AVG_RETURN if not np.isnan(avg_return) else np.nan,
            "profit_factor": pf,
            "profit_factor_diff": pf - SOURCE_PROFIT_FACTOR if np.isfinite(pf) else np.nan,
            "avg_hold_days": summary["avg_hold_days"],
            "avg_hold_days_diff": (summary["avg_hold_days"] - SOURCE_AVG_HOLD_DAYS
                                   if not np.isnan(summary["avg_hold_days"]) else np.nan),
        })
    return pd.DataFrame(rows)[RESULT_COLS]


def write_calibration_report(table: pd.DataFrame, output_path: Path) -> Path:
    """TASKS.md T-404: validation/reports/calibration_nishimura.md に出す。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 西村ルール較正（DESIGN.md §10.2(b) / TASKS.md T-404）",
        "",
        f"ルール: Close[T]<=0.95*SMA5[T] かつ Close[T]>SMA75[T] で T+1 寄付き買い、"
        f"Close[t]>SMA5[t] となった最初の日の翌日寄付きで手仕舞い（最長15日、"
        f"成立しなければ15日目の終値で強制手仕舞い）。",
        "",
        f"出典: {SOURCE_LABEL}。勝率 {SOURCE_WIN_RATE:.2%}、平均損益 {SOURCE_AVG_RETURN:+.2%}、"
        f"PF {SOURCE_PROFIT_FACTOR:.3f}、平均保有 {SOURCE_AVG_HOLD_DAYS:.2f}日。",
        "",
        "数値の解釈・良否の判断はしない。出典との差分（*_diff 列 = こちらの値 − 出典）を"
        "数値で示すだけ。ホールドアウト（2026-02〜2026-08）は使用していない。",
        "",
        _df_to_markdown(table),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main(argv: Optional[list] = None) -> int:
    """python -m stockbot.validation.calibration [--output ...] で実行する。
    store の全銘柄に対して DEFAULT_WINDOWS（2016-2020, 2021-2026）で較正する。
    """
    import argparse

    from ..config import Settings
    from ..data.store import IDX_TICKER, OhlcvStore, from_long

    ap = argparse.ArgumentParser(prog="python -m stockbot.validation.calibration")
    ap.add_argument("--output", default=None,
                    help="既定: src/stockbot/validation/reports/calibration_nishimura.md")
    args = ap.parse_args(argv)

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)
    if not ohlcv:
        print("[calibration] store にデータが無いため中止（先に backfill/daily を実行）")
        return 1

    table = run_calibration(ohlcv, sorted(ohlcv.keys()))
    default_path = Path(__file__).resolve().parent / "reports" / "calibration_nishimura.md"
    output_path = Path(args.output) if args.output else default_path
    write_calibration_report(table, output_path)
    print(f"[calibration] {output_path} に出力")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
