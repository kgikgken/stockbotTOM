# main_with_reduce.py
# Simplified integration of reduction signal logic (additive, not replacing existing)

from utils.market import enhance_market_score
from utils.position import load_positions, analyze_positions
from utils.events import load_events  # user must ensure helper exists
from screening import run_screening
from utils.util import jst_today_str, jst_today_date
import yfinance as yf


def detect_reduce_signals(mkt_info, events, positions, candidates, today_date):
    signals = []

    # Check if big events today (simplified)
    for ev in events:
        if ev.get("date") == str(today_date) and ev.get("kind") == "macro":
            signals.append(f"イベント: {ev.get('label')}")

    # A: Wave collapse (SOX/NVDA)
    try:
        sox = yf.Ticker("^SOX").history(period="6d")
        if len(sox) >= 2:
            sox_chg = (sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1) * 100
            if sox_chg <= -3.0:
                signals.append(f"SOX {sox_chg:.1f}%")
    except Exception:
        pass

    try:
        nvda = yf.Ticker("NVDA").history(period="6d")
        if len(nvda) >= 2:
            nvda_chg = (nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1) * 100
            if nvda_chg <= -4.0:
                signals.append(f"NVDA {nvda_chg:.1f}%")
    except Exception:
        pass

    # B: RR swap (simplified)
    if positions and candidates:
        best_new = max(candidates, key=lambda x: x.get("rr", 0))
        for pos in positions:
            old_rr = pos.get("rr", 0)
            if best_new.get("rr", 0) - old_rr > 1.0:
                signals.append(f"RR乗換候補: {best_new['ticker']}")

    return signals


def build_reduce_message(signals):
    if not signals:
        return ""
    lines = ["\n◆ 縮小アラート"]
    for s in signals:
        lines.append(f"- {s}")
    lines.append("推奨: 寄りで部分縮小 or 乗換")
    return "\n".join(lines)


def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = mkt.get("score", 50)

    pos_df = load_positions("positions.csv")
    pos_text, total_asset, total_pos, lev, pos_list = analyze_positions(pos_df)

    events = load_events("events.csv")
    candidates = run_screening(today_date, mkt_score)

    signals = detect_reduce_signals(mkt, events, pos_list, candidates, today_date)
    reduce_msg = build_reduce_message(signals)

    report = f"📅 {today_str} stockbotTOM 日報\n"
    report += f"◆ 地合いスコア: {mkt_score}\n"
    if reduce_msg:
        report += reduce_msg + "\n"

    print(report)


if __name__ == "__main__":
    main()
