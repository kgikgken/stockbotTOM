# utils/report.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np


# -------------------------------------------------
# 表示ヘルパ
# -------------------------------------------------
def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.2f}%"


def _fmt_money(yen: float) -> str:
    try:
        v = float(yen)
        if not np.isfinite(v):
            return "n/a"
        return f"{int(round(v)):,}円"
    except Exception:
        return "n/a"


def _yn(flag: bool) -> str:
    return "Y" if bool(flag) else "N"


def _safe(x, default="n/a") -> str:
    try:
        if x is None:
            return default
        s = str(x)
        return s if s.strip() else default
    except Exception:
        return default


# -------------------------------------------------
# レポート組み立て
# -------------------------------------------------
def build_report(
    *,
    today_str: str,
    mode_label: str,  # "Swing専用 / 1〜7日" など
    mkt: Dict,
    delta3d: Optional[int],
    leverage: float,
    lev_comment: str,
    max_position_yen: float,
    sectors_5d: List[Tuple[str, float]],
    event_lines: List[str],
    no_trade: bool,
    no_trade_reasons: List[str],
    swing_selected: List[Dict],
    swing_watch: List[Dict],
    pos_text: str,
    lot_risk_warn: Optional[Dict] = None,
) -> str:
    """
    必要キー（swing_selected 各要素）
      ticker, name, sector, setup, rr, ev, adj_ev, r_per_day,
      entry, in_low, in_high, price_now, atr,
      stop, tp1, tp2, exp_days,
      gu, action
    """

    mkt_score = int(mkt.get("score", 50))
    mkt_comment = _safe(mkt.get("comment", "中立"))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append(f"◆ 今日の結論（{mode_label}）")

    if no_trade:
        rs = " / ".join(no_trade_reasons) if no_trade_reasons else "条件該当"
        lines.append(f"🚫 新規見送り（{rs}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    d3 = "n/a" if delta3d is None else f"{delta3d:+d}"
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {d3}")
    lines.append(f"- レバ: {leverage:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{_fmt_money(max_position_yen)}")
    lines.append("")

    # セクター
    lines.append("📈 セクター（5日）")
    if sectors_5d:
        for i, (sec, pct) in enumerate(sectors_5d[:5], start=1):
            lines.append(f"{i}. {sec} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    # イベント
    lines.append("⚠ イベント")
    if event_lines:
        for s in event_lines:
            lines.append(s)
    else:
        lines.append("- 特になし")
    lines.append("")

    # Swing 本命
    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if swing_selected:
        rr_avg = float(np.mean([c.get("rr", 0.0) for c in swing_selected]))
        ev_avg = float(np.mean([c.get("ev", 0.0) for c in swing_selected]))
        adj_avg = float(np.mean([c.get("adj_ev", 0.0) for c in swing_selected]))
        rpd_avg = float(np.mean([c.get("r_per_day", 0.0) for c in swing_selected]))
        lines.append(
            f"  候補数:{len(swing_selected)}銘柄 / 平均RR:{rr_avg:.2f} / 平均EV:{ev_avg:.2f} / 平均AdjEV:{adj_avg:.2f} / 平均R/day:{rpd_avg:.2f}"
        )
        lines.append("")

        for c in swing_selected:
            ticker = _safe(c.get("ticker"))
            name = _safe(c.get("name"))
            sector = _safe(c.get("sector"))
            setup = _safe(c.get("setup", "A"))
            rr = float(c.get("rr", 0.0) or 0.0)
            adj_ev = float(c.get("adj_ev", 0.0) or 0.0)
            rpd = float(c.get("r_per_day", 0.0) or 0.0)

            entry = float(c.get("entry", np.nan))
            in_low = float(c.get("in_low", np.nan))
            in_high = float(c.get("in_high", np.nan))
            price_now = float(c.get("price_now", np.nan))
            atr = float(c.get("atr", np.nan))

            stop = float(c.get("stop", np.nan))
            tp1 = float(c.get("tp1", np.nan))
            tp2 = float(c.get("tp2", np.nan))
            exp_days = float(c.get("exp_days", np.nan))

            gu = bool(c.get("gu", False))
            action = _safe(c.get("action", "WATCH_ONLY"))
            action_jp = {
                "EXEC_NOW": "即IN可",
                "LIMIT_WAIT": "指値待ち",
                "WATCH_ONLY": "監視のみ",
            }.get(action, action)

            lines.append(f"- {ticker} {name} [{sector}]")
            lines.append(f"  形:{setup}  RR:{rr:.2f}  AdjEV:{adj_ev:.2f}  R/day:{rpd:.2f}")

            if np.isfinite(in_low) and np.isfinite(in_high) and np.isfinite(entry):
                lines.append(
                    f"  IN:{entry:.1f}（帯:{in_low:.1f}〜{in_high:.1f}）"
                    + (f"  現在:{price_now:.1f}" if np.isfinite(price_now) else "")
                    + (f"  ATR:{atr:.1f}" if np.isfinite(atr) else "")
                    + f"  GU:{_yn(gu)}"
                )
            else:
                lines.append(
                    f"  IN:{entry:.1f}"
                    + (f"  現在:{price_now:.1f}" if np.isfinite(price_now) else "")
                    + f"  GU:{_yn(gu)}"
                )

            if np.isfinite(stop) and np.isfinite(tp1) and np.isfinite(tp2):
                ed = f"{exp_days:.1f}" if np.isfinite(exp_days) else "n/a"
                lines.append(
                    f"  STOP:{stop:.1f}  TP1:{tp1:.1f}  TP2:{tp2:.1f}  ExpectedDays:{ed}  行動:{action_jp}"
                )
            else:
                lines.append(f"  行動:{action_jp}")

            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # 監視リスト
    lines.append("🧠 監視リスト（今日は入らない）")
    if swing_watch:
        for c in swing_watch[:10]:
            ticker = _safe(c.get("ticker"))
            name = _safe(c.get("name", ""))
            sector = _safe(c.get("sector", ""))
            setup = _safe(c.get("setup", "-"))
            rr = float(c.get("rr", 0.0) or 0.0)
            rpd = float(c.get("r_per_day", 0.0) or 0.0)
            reason = _safe(c.get("watch_reason", c.get("drop_reason", "理由不明")))
            action = _safe(c.get("action", "WATCH_ONLY"))
            action_jp = {
                "EXEC_NOW": "即IN可",
                "LIMIT_WAIT": "指値待ち",
                "WATCH_ONLY": "監視のみ",
            }.get(action, action)
            gu = bool(c.get("gu", False))

            # 形があるものは短縮表示
            if setup != "-" and rr > 0:
                lines.append(
                    f"- {ticker} {name} [{sector}] 形:{setup} RR:{rr:.2f} R/day:{rpd:.2f} 理由:{reason} 行動:{action_jp} GU:{_yn(gu)}"
                )
            else:
                lines.append(f"- {ticker} {name} [{sector}] 理由:{reason}")
    else:
        lines.append("- なし")

    lines.append("")

    # ロット事故警告
    if lot_risk_warn:
        try:
            loss_yen = float(lot_risk_warn.get("max_loss_yen", np.nan))
            loss_pct = float(lot_risk_warn.get("max_loss_pct", np.nan))
            if np.isfinite(loss_yen) and np.isfinite(loss_pct):
                lines.append(
                    f"⚠ ロット事故警告：想定最大損失 ≈ {int(round(loss_yen)):,}円（資産比 {loss_pct:.2f}%）"
                )
                lines.append("")
        except Exception:
            pass

    # ポジション
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)