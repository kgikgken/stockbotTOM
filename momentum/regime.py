"""STEP1: レジームフィルター(TOPIX) — 例外なく厳守.

生の条件(raw_attack): TOPIX終値 > 150日移動平均(既定・指示⑤a) かつ 直近12ヶ月リターン > 0

指示⑧: 非対称ヒステリシスを適用する。
- 防御モードへの移行(raw_attack=False)は即時反映。資産防御を優先。
- 攻撃モードへの移行(raw_attack=True)は、連続cfg.regime_confirm_days営業日(既定2日)
  raw_attackが真であることを確認してから初めて有効化する。新規リスクは一呼吸置いて取る設計。
このため、GitHub Actions実行間で「直近の raw_attack 履歴」を小さなCSVに永続化する
(momentum_regime_history.csv・workflow側でgit commitして翌回に引き継ぐ)。
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .data import fetch_regime_series


def _read_history(path: str, keep_days: int) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    return rows[-keep_days:]


def _last_business_day_before(d: pd.Timestamp) -> pd.Timestamp:
    prev = d - pd.tseries.offsets.BDay(1)
    return prev.normalize()


def _check_history_gap(path: str, today: str) -> str:
    """指示⑫-2: 前営業日のレコードが履歴に無ければ、過去回の永続化失敗を疑い警告文を返す。"""
    p = Path(path)
    if not p.exists():
        return ""  # 初回実行は正常(欠落ではない)
    try:
        with open(p, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return "⚠レジーム履歴ファイルが破損している可能性(読み込み失敗)"
    if not rows:
        return ""
    last_date = rows[-1].get("日付", "")
    expected_prev = _last_business_day_before(pd.Timestamp(today)).strftime("%Y-%m-%d")
    if last_date and last_date < expected_prev:
        return (f"⚠レジーム履歴に前営業日({expected_prev})の記録が無く、最終記録は{last_date}。"
               "過去の実行でCSV永続化(git push)が失敗した可能性があり、"
               "連続日数カウントが途切れている疑いがある")
    return ""


def _git_commit_history(path: str) -> str:
    """指示⑫-1: 履歴ファイルをPythonから直接commit/pushし、当日のレポートに結果を反映できるようにする。
    ワークフロー側の一括永続化ステップより先に実行されるため、失敗検知が当日中に間に合う。"""
    try:
        subprocess.run(["git", "config", "user.name", "stockbot"], capture_output=True, timeout=15)
        subprocess.run(["git", "config", "user.email", "stockbot@users.noreply.github.com"],
                       capture_output=True, timeout=15)
        subprocess.run(["git", "add", path], capture_output=True, timeout=15)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True, timeout=15)
        if diff.returncode == 0:
            return ""  # 差分なし(初回以外の通常ケースでは無風)
        commit = subprocess.run(["git", "commit", "-m", "momentum regime history (auto)"],
                                capture_output=True, timeout=15, text=True)
        if commit.returncode != 0:
            return f"⚠レジーム履歴のgit commitに失敗: {commit.stderr[:150]}"
        push = subprocess.run(["git", "push"], capture_output=True, timeout=30, text=True)
        if push.returncode != 0:
            return f"⚠レジーム履歴のgit pushに失敗: {push.stderr[:150]}(連続日数カウントが引き継がれない可能性)"
        return ""
    except Exception as e:
        return f"⚠レジーム履歴の永続化処理で例外: {str(e)[:150]}"


def _append_history(path: str, today: str, raw_attack: bool, final_attack: bool) -> None:
    p = Path(path)
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["日付", "raw_attack", "final_attack"])
        w.writerow([today, "1" if raw_attack else "0", "1" if final_attack else "0"])


def compute_regime(cfg, today: str | None = None) -> dict:
    persist_warn = _check_history_gap(cfg.regime_history_path, today) if today else ""

    series, source = fetch_regime_series(cfg)
    if series is None or len(series) < cfg.regime_sma_days:
        result = {
            "ok": False, "mode": "防御モード", "attack": False, "raw_attack": False,
            "reason": f"レジーム系列取得不可({source}) — 安全側で防御モード",
            "source": source, "hysteresis_note": "", "persist_warn": persist_warn,
        }
        if today:
            _append_history(cfg.regime_history_path, today, False, False)
            if not cfg.dryrun:
                git_warn = _git_commit_history(cfg.regime_history_path)
                if git_warn:
                    result["persist_warn"] = (persist_warn + " / " + git_warn) if persist_warn else git_warn
        return result

    close_now = float(series.iloc[-1])
    sma_n = float(series.iloc[-cfg.regime_sma_days:].mean())
    mom_days = min(cfg.regime_mom_days, len(series) - 1)
    mom_12m = close_now / float(series.iloc[-1 - mom_days]) - 1.0

    above_sma = close_now > sma_n
    positive_mom = mom_12m > 0
    raw_attack = above_sma and positive_mom

    # ---- 指示⑧: 非対称ヒステリシス ----
    hysteresis_note = ""
    if not raw_attack:
        final_attack = False  # 防御は即時
    else:
        history = _read_history(cfg.regime_history_path, cfg.regime_confirm_days - 1)
        recent_raw = [h.get("raw_attack") == "1" for h in history]
        consecutive_ok = len(recent_raw) >= (cfg.regime_confirm_days - 1) and all(recent_raw[-(cfg.regime_confirm_days - 1):])
        final_attack = consecutive_ok
        if not final_attack:
            n_confirmed = sum(1 for v in reversed(recent_raw) if v)
            hysteresis_note = (f"条件は攻撃モード相当だが確認日数不足のため防御モード継続"
                              f"(連続{n_confirmed + 1}/{cfg.regime_confirm_days}日目)")

    if today:
        _append_history(cfg.regime_history_path, today, raw_attack, final_attack)
        if not cfg.dryrun:
            git_warn = _git_commit_history(cfg.regime_history_path)
            if git_warn:
                persist_warn = (persist_warn + " / " + git_warn) if persist_warn else git_warn

    detail = (f"TOPIX代理(1306) {close_now:,.1f} vs {cfg.regime_sma_days}日線 {sma_n:,.1f}"
             f"({'上' if above_sma else '下'}) / 12ヶ月{mom_12m*100:+.1f}%"
             f"({'プラス' if positive_mom else 'マイナス'})"
             " ※1306はETF価格(分割調整後)であり本物のTOPIX指数値ではない")
    if hysteresis_note:
        detail += f" / {hysteresis_note}"
    if persist_warn:
        detail += f" / {persist_warn}"

    return {
        "ok": True,
        "attack": final_attack, "raw_attack": raw_attack,
        "mode": "攻撃モード" if final_attack else "防御モード",
        "close": close_now, "sma200": sma_n, "mom_12m": mom_12m * 100,
        "above_sma": above_sma, "positive_mom": positive_mom,
        "source": source, "hysteresis_note": hysteresis_note, "persist_warn": persist_warn,
        "detail": detail,
    }
