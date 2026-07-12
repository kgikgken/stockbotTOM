"""保有ポジションの監視 — 状態C判定(指示③)+ スコア劣化早期警告(指示⑥)+ TOB急騰検知(指示⑦).

いずれも自動決済はしない。人間への警告フラグに留める設計を一貫させている。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .data import fetch_ohlcv
from .indicators import compute_momentum_features


def _read_entry_scores(positions_path: str = "positions.csv") -> dict:
    """positions.csv に任意列 entry_score があれば読む(無ければ空dict・グレースフルに縮退)。"""
    p = Path(positions_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, dtype=str)
    except Exception:
        return {}
    if "entry_score" not in df.columns or "ticker" not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            out[str(r["ticker"]).strip()] = float(r["entry_score"])
        except Exception:
            continue
    return out


def _read_entry_dates(positions_path: str = "positions.csv") -> dict:
    """positions.csv に任意列 entry_date があれば読む(無ければ空dict)。"""
    p = Path(positions_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, dtype=str)
    except Exception:
        return {}
    if "entry_date" not in df.columns or "ticker" not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        v = str(r.get("entry_date", "")).strip()
        if v and v != "nan":
            out[str(r["ticker"]).strip()] = v
    return out


def _business_day_gap(entry_date: str, today: str) -> int | None:
    """entry_date から today までの営業日差。パース不可ならNone。"""
    try:
        d0 = pd.Timestamp(entry_date).normalize()
        d1 = pd.Timestamp(today).normalize()
    except Exception:
        return None
    if d1 < d0:
        return None
    return int(np.busday_count(d0.date(), d1.date()))


def backfill_entry_scores(positions_path: str, eligible: List[dict], today: str, dryrun: bool, cfg: Config) -> int:
    """指示⑩: positions.csv の entry_score が空欄の行に、当日のプール計算結果から自動転記する。

    ★指示⑭(信頼性ガード): entry_date が無い、またはentry_dateと検出日(today)の営業日差が
    cfg経由の閾値(既定2営業日)を超える場合は、乖離が大きすぎるとみなし自動転記を見送る。
    この場合、既存の「対象外」フォールバック(プール順位のみでの劣化判定)に合流させる。

    安全策(⑩から変更なし):
    - dryrun時は実行しない
    - ticker/shares/entry_price 等の既存列・既存値は一切変更しない(新規列のみ追記)
    - 書き戻し前に行数とticker列が変化していないことを自己検証してから保存する
    """
    if dryrun:
        return 0
    p = Path(positions_path)
    if not p.exists():
        return 0
    try:
        df = pd.read_csv(p, dtype=str).fillna("")
    except Exception:
        return 0
    if "ticker" not in df.columns:
        return 0

    orig_rows, orig_tickers = len(df), list(df["ticker"])
    for col in ("entry_score", "entry_score_date", "entry_score_note"):
        if col not in df.columns:
            df[col] = ""
    has_entry_date = "entry_date" in df.columns

    score_map = {it["row"]["ticker"]: it["score"] for it in (eligible or [])}
    filled = 0
    changed = False
    for i, row in df.iterrows():
        existing = str(row.get("entry_score", "")).strip()
        if existing not in ("", "nan"):
            continue
        ticker = str(row["ticker"]).strip()

        # ★指示⑭: 乖離ガード
        entry_date = str(row.get("entry_date", "")).strip() if has_entry_date else ""
        if not entry_date or entry_date == "nan":
            note = "entry_date未記録のため自動転記見送り(乖離検証不可)、手動確認推奨"
            if str(row.get("entry_score_note", "")).strip() != note:
                df.at[i, "entry_score_note"] = note
                changed = True
            continue
        gap_days = _business_day_gap(entry_date, today)
        if gap_days is None:
            note = f"entry_date({entry_date})の解析に失敗したため自動転記見送り、手動確認推奨"
            if str(row.get("entry_score_note", "")).strip() != note:
                df.at[i, "entry_score_note"] = note
                changed = True
            continue
        if gap_days > cfg.entry_score_max_gap_bdays:
            note = f"エントリー日から{gap_days}営業日乖離(閾値{cfg.entry_score_max_gap_bdays}日超)のため自動転記見送り、手動確認推奨"
            if str(row.get("entry_score_note", "")).strip() != note:
                df.at[i, "entry_score_note"] = note
                changed = True
            continue

        if ticker in score_map and score_map[ticker] != float("-inf"):
            df.at[i, "entry_score"] = f"{score_map[ticker]:.4f}"
            df.at[i, "entry_score_date"] = today
            df.at[i, "entry_score_note"] = "自動転記(捕捉日=本日実行時点。真のエントリー日と異なる場合あり)"
            filled += 1
            changed = True
        elif str(row.get("entry_score_note", "")).strip() == "":
            df.at[i, "entry_score_note"] = "自動転記対象外(母集団外)"
            changed = True

    if not changed:
        return 0
    if len(df) != orig_rows or list(df["ticker"]) != orig_tickers:
        return 0  # 自己検証失敗 → 保存しない(安全側)
    try:
        df.to_csv(p, index=False)
    except Exception:
        return 0
    return filled


def check_held_positions(pos_df: pd.DataFrame, universe: pd.DataFrame, cfg: Config,
                         eligible: List[dict] | None = None) -> List[dict]:
    if pos_df is None or len(pos_df) == 0:
        return []

    tickers = [str(t).strip() for t in pos_df["ticker"].tolist() if str(t).strip()]
    ohlcv, _ = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    uni_idx = universe.drop_duplicates("ticker").set_index("ticker") if len(universe) else None
    entry_scores = _read_entry_scores("positions.csv")

    # 指示⑥: 今日の母集団(eligible)内での順位・スコア分布を使う。母集団のスコア標準偏差を先に算出。
    pop_scores = [it["score"] for it in (eligible or []) if it["score"] != float("-inf")]
    pop_std = float(np.std(pop_scores, ddof=0)) if len(pop_scores) >= 5 else None
    pool_cut_rank = cfg.pool_size
    top_pool_tickers = {it["row"]["ticker"] for it in (eligible or [])[:pool_cut_rank]}
    eligible_by_ticker = {it["row"]["ticker"]: it for it in (eligible or [])}

    alerts: List[dict] = []
    for _, p in pos_df.iterrows():
        ticker = str(p["ticker"]).strip()
        if not ticker:
            continue
        urow = uni_idx.loc[ticker] if (uni_idx is not None and ticker in uni_idx.index) else None
        name = str(urow["name"]) if urow is not None else ticker
        code = ticker.replace(".T", "")

        try:
            df = ohlcv.get(ticker)
            if df is None:
                alerts.append({"code": code, "name": name, "state_c": None, "score_drop": False,
                               "tob_jump": False, "note": "データ取得不可(コード不一致・上場廃止等、要確認)"})
                continue
            feat = compute_momentum_features(df, None, cfg)
            if feat is None:
                alerts.append({"code": code, "name": name, "state_c": None, "score_drop": False,
                               "tob_jump": False, "note": "データ不足(算出不可)"})
                continue

            notes = []
            is_c = bool(feat["breakdown"])
            if is_c:
                notes.append(f"状態Cに該当(終値{feat['close']:,.0f}円が50日線{feat['sma50']:,.0f}円を下回った)。"
                            f"シャンデリア水準({feat['chandelier']:,.0f}円)到達まで自動手仕舞いはされない点に注意")

            # ---- 指示⑥: スコア劣化早期警告(状態Cの手前) ----
            score_drop = False
            if not is_c:
                item = eligible_by_ticker.get(ticker)
                today_score = item["score"] if item else None
                if today_score is None:
                    notes.append("本日の候補プール母集団に含まれず(流動性未達等)、スコア追跡不可")
                else:
                    fell_out = ticker not in top_pool_tickers
                    sd_drop = None
                    entry_score = entry_scores.get(ticker)
                    if entry_score is not None and pop_std and pop_std > 1e-9:
                        sd_drop = (entry_score - today_score) / pop_std
                    if fell_out or (sd_drop is not None and sd_drop >= cfg.score_drop_sd_threshold):
                        score_drop = True
                        reason_bits = []
                        if fell_out:
                            reason_bits.append(f"候補プール上位{pool_cut_rank}位から脱落")
                        if sd_drop is not None and sd_drop >= cfg.score_drop_sd_threshold:
                            reason_bits.append(f"エントリー時比 標準偏差{sd_drop:.1f}分低下")
                        notes.append("スコア劣化警告(状態Cではないが弱含み): " + " / ".join(reason_bits))
                    elif entry_score is None:
                        notes.append("entry_score未記録のためエントリー比の劣化判定は不可(プール順位のみ判定)")

            # ---- 指示⑦⑪: TOB急騰検知(2段階化・誤検知抑制) ----
            # Day0: 単日+15%以上のジャンプ → 弱い「参考」通知(TOBか通常のブレイク加速か当日は区別不能)
            # Day cfg.position_tob_confirm_days: ジャンプ後のボラが崩れていれば強い警告に格上げ。
            #      崩れていなければ通常のブレイク加速とみなし、追加アラートは出さない(静かに終わる)。
            tob_jump = False
            tob_stage = None
            c = df["Close"].dropna()
            lr_all = np.log(c / c.shift(1)).dropna()
            if len(lr_all) >= cfg.position_tob_baseline_days + cfg.position_tob_confirm_days:
                lookback = lr_all.iloc[-(cfg.position_tob_confirm_days + 5):]
                if len(lookback) and float(lookback.abs().max()) >= cfg.position_tob_jump_threshold:
                    jump_idx = lookback.abs().idxmax()
                    jump_ret = float(lookback.loc[jump_idx])
                    days_since = int((lr_all.index >= jump_idx).sum()) - 1
                    if days_since == 0:
                        tob_jump, tob_stage = True, "day0"
                        notes.append(f"直近1営業日で単日{jump_ret*100:+.0f}%の急騰を検知(参考)。"
                                    "TOB等の可能性と、通常のブレイクアウト加速の可能性の両方があり、"
                                    f"本日時点では判別不可。{cfg.position_tob_confirm_days}営業日後に"
                                    "ボラティリティの状況を再確認する")
                    elif days_since >= cfg.position_tob_confirm_days:
                        post = lr_all.loc[lr_all.index > jump_idx]
                        pre = lr_all.loc[lr_all.index < jump_idx].iloc[-cfg.position_tob_baseline_days:]
                        if len(post) >= 2 and len(pre) >= 10:
                            post_vol, pre_vol = float(post.std(ddof=0)), float(pre.std(ddof=0))
                            if pre_vol > 1e-9 and post_vol < pre_vol * cfg.tob_vol_collapse_ratio:
                                tob_jump, tob_stage = True, "confirmed"
                                notes.append(f"{days_since}営業日前の単日{jump_ret*100:+.0f}%急騰後、"
                                            f"変動率が平常時の{post_vol/pre_vol*100:.0f}%に低下 → "
                                            "TOB疑いアラート(要確認)に格上げ。シャンデリア・エグジットは"
                                            "通常ボラティリティ前提のため機能しない可能性。"
                                            "iSPEEDで適時開示を確認してください")
                            # 崩れていなければ通常のブレイク加速とみなし、何も出さない(意図的な沈黙)

            alerts.append({
                "code": code, "name": name, "state_c": is_c, "score_drop": score_drop,
                "tob_jump": tob_jump, "tob_stage": tob_stage,
                "close": feat["close"], "sma50": feat["sma50"], "chandelier": feat["chandelier"],
                "note": " / ".join(notes) if notes else "状態C非該当・スコア劣化なし・急騰なし(平常)",
            })
        except Exception as e:
            # ★不正なticker表記・想定外のデータ形状などで、1銘柄の処理が異常終了しても
            # 全体をクラッシュさせない(defense-in-depth)。詳細はログのみ、通知は安全な要約のみ。
            print(f"[WARN] 保有銘柄チェックで例外(ticker={ticker}): {type(e).__name__}: {e}")
            alerts.append({"code": code, "name": name, "state_c": None, "score_drop": False,
                           "tob_jump": False, "note": "処理中に例外が発生(コード表記等を要確認・詳細はActionsログ参照)"})
    return alerts
