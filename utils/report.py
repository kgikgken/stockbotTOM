# utils/report.py
from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any, Dict, List, Tuple

from utils.util import MarketState, EventState, fmt_float, fmt_int

DEFAULT_TZ = timezone(timedelta(hours=9))

# 表示制御
SHOW_WATCHLIST = False  # True にすると「参考：除外理由」も表示


def _market_judge(market: MarketState) -> str:
    if market.no_trade:
        return "弱い（新規禁止）"
    if market.score >= 70:
        return "強い"
    if market.score >= 60:
        return "やや強い"
    if market.score >= 50:
        return "中立"
    return "弱い"


def _regime_text(market: MarketState) -> str:
    if market.regime == "bull":
        return "上昇基調"
    if market.regime == "bear":
        return "下落警戒"
    return "中立"


def build_line_report(
    date_str: str,
    market: MarketState,
    sectors: List[Tuple[str, float]],
    events: EventState,
    result: Dict[str, Any],
) -> str:
    picked = result["picked"]
    rejected = result["rejected"]
    no_trade = bool(result["no_trade"])
    reasons = result.get("reasons", [])
    weekly_new = result.get("weekly_new", 0)
    weekly_limit = result.get("weekly_limit", 3)
    macro_event_near = bool(result.get("macro_event_near", False))
    max_final = int(result.get("max_final", 5))

    avg_rr = result.get("avg_rr", 0.0)
    avg_ev = result.get("avg_ev", 0.0)
    avg_adj_ev = result.get("avg_adj_ev", 0.0)
    avg_rpd = result.get("avg_rpd", 0.0)

    lines: List[str] = []
    lines.append(f"📅 {date_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    if no_trade:
        reason_text = " / ".join(reasons) if reasons else "条件"
        if macro_event_near and max_final == 2:
            lines.append(f"🚫 新規見送り（イベント接近）")
        else:
            lines.append(f"🚫 新規見送り（{reason_text}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {fmt_int(market.score)}点（{_market_judge(market)}）")
    lines.append(f"- 地合い変化: Δ{fmt_int(market.delta_3d)}")
    lines.append(f"- 相場判断: {_regime_text(market)}")
    lines.append(f"- 週次新規回数: {weekly_new} / {weekly_limit}")
    lines.append(f"- マクロ警戒: {'ON' if macro_event_near else 'OFF'}")
    lines.append(f"- 推奨レバレッジ: {fmt_float(market.leverage,1)}倍（{'守り（新規禁止）' if no_trade else '通常'}）")
    lines.append(f"- 最大建玉目安: 約{fmt_int(market.max_gross)}円")
    lines.append("")

    # セクター
    lines.append("📈 セクター動向（直近5日）")
    if sectors:
        for i, (name, r) in enumerate(sectors[:5], 1):
            lines.append(f"{i}. {name} ({fmt_float(r,2)}%)")
    else:
        lines.append("（セクター集計は未設定）")
    lines.append("")

    # イベント
    lines.append("⚠ 重要イベント")
    lines.append(events.macro_event_text if events.macro_event_text else "なし")
    lines.append("")

    # 候補
    lines.append("🏆 Swing候補（順張りのみ / 追いかけ禁止 / 速度重視）")
    lines.append(
        f"  候補数:{len(picked)}銘柄  平均RR:{fmt_float(avg_rr,2)} / 平均EV:{fmt_float(avg_ev,2)} / 平均補正EV:{fmt_float(avg_adj_ev,2)} / 平均R/日:{fmt_float(avg_rpd,2)}"
    )
    if macro_event_near and max_final == 2:
        lines.append("※ イベント接近のため、候補は最大2銘柄までに制限")
    lines.append("")

    for c in picked:
        lines.append(f"- {c.ticker} {c.name}［{c.sector}］")
        lines.append(
            f"  型:{c.setup}  RR:{fmt_float(c.rr,2)}  補正EV:{fmt_float(c.adj_ev,2)}  R/日:{fmt_float(c.rpd,2)}  特性:{c.macro_tag}"
        )
        lines.append(
            f"  エントリー:{fmt_float(c.entry_center,1)}（範囲:{fmt_float(c.entry_low,1)}〜{fmt_float(c.entry_high,1)}） 現在値:{fmt_float(c.close,1)}  ATR:{fmt_float(c.atr,1)}  GU:{'あり' if c.gu else 'なし'}"
        )
        lines.append(
            f"  損切:{fmt_float(c.stop,1)}  利確1:{fmt_float(c.tp1,1)}  利確2:{fmt_float(c.tp2,1)}  想定日数:{fmt_float(c.exp_days,1)}"
        )
        lines.append(f"  行動:{c.action}")
        lines.append("")

    lines.append("🧠 参考：除外理由（今日は入らない）")
    if SHOW_WATCHLIST:
      
      # 上位だけ（多すぎると読めない）
      shown = 0
      for c in rejected[:10]:
          if c.reject_reason:
              nm = f" {c.name}" if c.name else ""
              sec = f"［{c.sector}］" if c.sector else ""
              lines.append(f"- {c.ticker}{nm}{sec} 理由:{c.reject_reason}")
              shown += 1
          if shown >= 10:
              break
      if shown == 0:
          lines.append("（なし）")
      
    lines.append("")
    lines.append("📊 保有ポジション")
    lines.append("- 4971.T: 損益 n/a")

    return "\n".join(lines).strip()