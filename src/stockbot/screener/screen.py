"""日次のスクリーニング（docs/SCREENER.md §2.4・§2.5）。

1. ユニバース通過銘柄について、E1 以外の 18 条件を銘柄ごとに判定する（conditions.py）
2. 18 条件を全て満たした集合（＝ E1 の母集団）に対して E1 を付ける
3. 通過した銘柄を売買代金の降順に並べ、同一 33 業種は 3 件までに絞る

順位は付けない。並び順は売買代金であって優劣ではない（§2.5）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ..features import pullback, swings
from ..features.dimensions import LANDING_MA_NAMES
from ..features.indicators import atr_wilder, sma
from .record import as_calendar_date
from .conditions import (
    CONDITION_IDS,
    DIAGNOSTIC_COLS,
    SELF_CONTAINED_IDS,
    evaluate_conditions,
    passes_self_contained,
)

E1_TOP_PCTL = 0.90      # 上位 10% を落とす
E1_MIN_POOL = 10        # 母集団がこれ未満の日は E1 をスキップする
SECTOR_CAP = 3          # 同一 33 業種の上限
MIN_HISTORY_BARS = 60   # 指標計算に必要な最低限（これ未満は評価しない）

SCREEN_COLS = ["ticker", "date"] + CONDITION_IDS + ["passes"] + DIAGNOSTIC_COLS + ["state"]

SUMMARY_PREFIX = "screen_summary_"
SUMMARY_SUFFIX = ".json"


# ------------------------------------------------------------------ 監視用の集計
def fail_counts(df: pd.DataFrame) -> Dict[str, int]:
    """条件ごとの不成立件数（docs/SCREENER.md §3.6）。

    1 銘柄が複数の条件で同時に落ちうるので、合計は評価件数と一致しない
    （pipeline.py の「ゲート落ちの内訳」と同じ趣旨）。欠損は不成立として数える。
    E1 はその日の候補集合に依存し、母集団の外では未判定なのでここには含めない。
    """
    if df is None or len(df) == 0:
        return {cid: 0 for cid in SELF_CONTAINED_IDS}
    return {cid: int((~df[cid].fillna(False).astype(bool)).sum()) for cid in SELF_CONTAINED_IDS}


def landing_ma_breakdown(df: pd.DataFrame) -> Dict[str, int]:
    """止まった線ごとの件数（docs/SCREENER.md §3.6）。

    線が取れなかった行（押し目構造が無い、ATR が使えない）は数えない。
    キーは LANDING_MA_NAMES の並びで固定する（0 件の線も 0 として残す）ので、
    日をまたいで同じ形で比べられる。
    """
    counts = {name: 0 for name in LANDING_MA_NAMES}
    if df is None or len(df) == 0 or "landing_ma" not in df.columns:
        return counts
    for name, n in df["landing_ma"].fillna("").value_counts().items():
        if name in counts:
            counts[name] = int(n)
    return counts


def format_counts(counts: Dict[str, int], top: Optional[int] = None, sort: bool = True) -> str:
    """"A1:3 B2:1 ..." の形にする。sort=False なら渡された順（線の内訳はこちら）。"""
    items = sorted(counts.items(), key=lambda kv: -kv[1]) if sort else list(counts.items())
    return " ".join(f"{k}:{v}" for k, v in (items[:top] if top else items))


def evaluate_universe(
    ohlcv: Dict[str, pd.DataFrame], tickers: Iterable[str], idx_close: pd.Series,
    k: int, earnings_schedule: Optional[pd.DataFrame] = None, log=print,
) -> pd.DataFrame:
    """ユニバース通過銘柄それぞれの最終足（＝ T）について 18 条件を判定する。

    各銘柄は自身の系列の最終日だけを評価する（pipeline.compute_daily_features と同じ規約）。
    idx_close は指数終値。銘柄ごとに日付で整列し直し、欠損は最大 3 営業日まで過去方向に
    のみ前方補完する（pipeline.py と同じ扱い。未来は一切参照しない）。
    """
    rows = []
    n_evaluated = 0
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_HISTORY_BARS:
            continue
        n_evaluated += 1
        high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
        t_pos = len(df) - 1
        alt = swings.alternate_swings(swings.detect_raw_swings(high, low, k))
        pb = pullback.pullback_state(high, low, close, sma(close, 5), sma(close, 200),
                                     atr_wilder(high, low, close, 14), alt, t_pos, k)
        row = evaluate_conditions(
            ticker, high, low, close, volume, pb, t_pos,
            idx_close=idx_close.reindex(close.index).ffill(limit=3),
            earnings_schedule=earnings_schedule,
        )
        row["date"] = close.index[t_pos]
        row["state"] = pb["state"]
        rows.append(row)

    if not rows:
        log(f"[screen] 評価 {n_evaluated} 銘柄 / 条件判定 0 件")
        return pd.DataFrame(columns=SCREEN_COLS)
    out = pd.DataFrame(rows)
    out["passes"] = False
    out = out[SCREEN_COLS]
    for cid in SELF_CONTAINED_IDS:
        out[cid] = out[cid].astype("boolean")
    out["E1"] = pd.array([pd.NA] * len(out), dtype="boolean")
    n_pool = int(out.apply(passes_self_contained, axis=1).sum())
    # 候補が 0 件の日に「どの条件で落ちたか」が分からないと運用で困る。
    # pipeline.py の「ゲート落ちの内訳」と同じ趣旨（複数条件に同時該当しうる）
    fails = fail_counts(out)
    log(f"[screen] 評価 {n_evaluated} 銘柄 / 18条件通過 {n_pool} 件")
    log(f"[screen] 条件別の不成立件数（多い順）: {format_counts(fails)}")
    # D4 を掛ける前の「どの線で止まったか」。SMA200 は D4 で落ちるので、
    # ここに出る SMA200 の件数が D4 が捨てている分になる
    log(f"[screen] 止まった線の内訳（押し目構造がある銘柄・D4適用前）: "
        f"{format_counts(landing_ma_breakdown(out), sort=False)}")
    known = _earnings_known(out)
    log(f"[screen] 決算発表予定日が取れた銘柄: {known}/{len(out)} "
        f"({known / len(out):.1%}) ← A4 が実際に効いた範囲")
    return out


def apply_e1(df: pd.DataFrame, top_pctl: float = E1_TOP_PCTL,
             min_pool: int = E1_MIN_POOL, log=print) -> tuple[pd.DataFrame, dict]:
    """E1（当日候補内で d2_rs60 の上位 10% を落とす）を付ける（docs/SCREENER.md §2.4）。

    母集団は A〜D・E2・E3 を全て満たした集合。母集団が min_pool 件未満の日は E1 を
    スキップし（全て成立扱い）、`e1_skipped` を立てて記録に残す。

    戻り値: (E1 と passes を埋めた df, {"e1_pool_n", "e1_skipped", "e1_threshold"})
    """
    out = df.copy()
    if len(out) == 0:
        return out, {"e1_pool_n": 0, "e1_skipped": True, "e1_threshold": np.nan}

    pool_mask = out.apply(passes_self_contained, axis=1).to_numpy(dtype=bool)
    pool_n = int(pool_mask.sum())
    skipped = pool_n < min_pool

    e1 = pd.array([pd.NA] * len(out), dtype="boolean")
    threshold = np.nan
    if skipped:
        e1[pool_mask] = True
    else:
        rs = out.loc[pool_mask, "rs60"]
        threshold = float(rs.quantile(top_pctl))
        # 欠損は不成立（§2.3）。rs60 が取れない銘柄は E1 を通さない
        e1[pool_mask] = (rs.notna() & (rs <= threshold)).to_numpy(dtype=bool)
    out["E1"] = e1
    out["passes"] = pool_mask & (out["E1"].fillna(False).to_numpy(dtype=bool))

    meta = {"e1_pool_n": pool_n, "e1_skipped": bool(skipped), "e1_threshold": threshold}
    if skipped:
        log(f"[screen] E1: 母集団 {pool_n} 件 (< {min_pool}) のためスキップ")
    else:
        log(f"[screen] E1: 母集団 {pool_n} 件 / rs60 の上位{int((1 - top_pctl) * 100)}%"
            f"（> {threshold:.4f}）を除外 → 通過 {int(out['passes'].sum())} 件")
    return out, meta


def select_candidates(df: pd.DataFrame, sector_by_ticker: Optional[Dict[str, str]] = None,
                      sector_cap: int = SECTOR_CAP) -> pd.DataFrame:
    """通過銘柄を売買代金の降順に並べ、同一 33 業種を sector_cap 件までに絞る。

    並び順は売買代金であって優劣ではない（順位を付けない、docs/SCREENER.md §2.5）。
    売買代金が同じ場合は銘柄コード順（実行のたびに並びが変わらないようにするため）。
    業種が取れない銘柄は上限の対象外（互いに別業種として扱う）。
    """
    if len(df) == 0:
        out = df.copy()
        out["sector33"] = pd.Series(dtype=str)
        return out
    passed = df[df["passes"].astype(bool)].copy()
    sector_by_ticker = sector_by_ticker or {}
    passed["sector33"] = passed["ticker"].map(lambda t: str(sector_by_ticker.get(t, "") or ""))
    passed = passed.sort_values(["adv_jpy", "ticker"], ascending=[False, True])

    counts: Dict[str, int] = {}
    keep = []
    for _i, row in passed.iterrows():
        sector = row["sector33"]
        if sector:
            if counts.get(sector, 0) >= sector_cap:
                continue
            counts[sector] = counts.get(sector, 0) + 1
        keep.append(row)
    if not keep:
        return passed.iloc[0:0]
    return pd.DataFrame(keep).reset_index(drop=True)


def _earnings_known(evaluated: pd.DataFrame) -> int:
    """判定日以降の決算発表予定日が取れた銘柄数（docs/SCREENER.md §2.6）。"""
    if evaluated is None or len(evaluated) == 0 or "a4_earnings_unknown" not in evaluated:
        return 0
    return int((~evaluated["a4_earnings_unknown"].fillna(True).astype(bool)).sum())


def _earnings_coverage(evaluated: pd.DataFrame) -> Optional[float]:
    """決算発表予定日が取れた銘柄の割合。評価0件の日は None。"""
    n = 0 if evaluated is None else len(evaluated)
    return round(_earnings_known(evaluated) / n, 4) if n else None


def build_summary(evaluated: pd.DataFrame, candidates: pd.DataFrame, meta: dict,
                  asof, delivered_on, gauge: Optional[dict] = None,
                  fetch_meta: Optional[dict] = None) -> dict:
    """その日のスクリーニングの要約（docs/SCREENER.md §3.6）。

    Actions のログは 90 日で消えるが、E1 のスキップ率や条件別の不成立件数は
    数週間から数か月かけて見るものなので、リポジトリ側に残す。
    """
    gauge = gauge or {}
    fetch_meta = fetch_meta or {}
    return {
        "delivered_on": as_calendar_date(delivered_on).strftime("%Y-%m-%d"),
        "asof": as_calendar_date(asof).strftime("%Y-%m-%d") if asof is not None else None,
        # 地合いゲージ（DESIGN.md §8.1 の6点）。順位にも条件にも使わない。配信の見出し用
        "regime_level": gauge.get("level"),
        "regime_score": gauge.get("score"),
        "n_evaluated": int(len(evaluated)),
        # 取得成功率（配信の健全性ブロック用、§4.5）。fetch_meta.json から畳む
        "fetch_ok": fetch_meta.get("data_ok"),
        "fetch_total": fetch_meta.get("data_total"),
        "n_pool": int(meta.get("e1_pool_n", 0)),
        "n_candidates": int(len(candidates)),
        # A4（決算）のカバー率。JPX のローリング更新は決算期の谷間でほぼ空になるので、
        # A4 がその日どれだけ効いていたかを日次で残す（docs/SCREENER.md §2.6・§3.6）
        "earnings_known": _earnings_known(evaluated),
        "earnings_coverage": _earnings_coverage(evaluated),
        "e1_skipped": bool(meta.get("e1_skipped", True)),
        "e1_threshold": (None if not np.isfinite(meta.get("e1_threshold", np.nan))
                         else float(meta["e1_threshold"])),
        "fail_counts": fail_counts(evaluated),
        "landing_ma_all": landing_ma_breakdown(evaluated),
        "landing_ma_candidates": landing_ma_breakdown(candidates),
    }


def summary_path(daily_dir: Path, delivered_on) -> Path:
    d = as_calendar_date(delivered_on).strftime("%Y-%m-%d")
    return Path(daily_dir) / f"{SUMMARY_PREFIX}{d}{SUMMARY_SUFFIX}"


def save_summary(summary: dict, daily_dir: Path, delivered_on) -> Path:
    """daily/screen_summary_YYYY-MM-DD.json に保存する（毎日上書きしてよい）。

    配信記録（delivered_*.csv）と違って上書きしてよい —— これは判断の記録ではなく、
    その日のスクリーニングがどう動いたかの観測値だから。
    """
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = summary_path(daily_dir, delivered_on)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
