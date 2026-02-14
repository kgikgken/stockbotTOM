from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


from utils.util import safe_float

def _fmt_yen(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"


def _fmt_oku(yen: float) -> str:
    """Format yen value as Japanese "億" unit (1億=1e8円).

    Used as a liquidity proxy (ADV20 / median traded value).
    """
    try:
        y = float(yen)
    except Exception:
        return "-"
    if not (y > 0) or not (y == y):
        return "-"
    oku = y / 1e8
    if oku < 10:
        return f"{oku:.1f}億"
    return f"{oku:.0f}億"

def build_report(
    today_str: str,
    market: Dict,
    delta3: float,
    futures_chg: float,
    risk_on: bool,
    macro_on: bool,
    events_lines: List[str],
    no_trade: bool,
    weekly_used: int,
    weekly_max: int,
    leverage: float,
    policy_lines: List[str],
    cands: List[Dict],
    pos_text: str,
    saucers: Dict[str, List[Dict]] | List[Dict] | None = None,
) -> str:
    mkt_score = int(market.get("score", 50))
    mkt_comment = str(market.get("comment", "中立"))

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    # Macro day preface (keep strict; do not promote market-in on event days)
    if macro_on:
        lines.append("⚠ 本日は重要イベント警戒日")
        if risk_on:
            lines.append("※ 先物Risk-ONにつき、警戒しつつ最大5まで表示")
        lines.append("")
        if events_lines:
            lines.append("対象イベント：")
            for ev in events_lines:
                if ev.startswith("⚠ "):
                    lines.append("・" + ev.replace("⚠ ", "").split("（")[0])
            lines.append("")
    # Header
    if no_trade:
        reason = "重要イベント警戒" if macro_on else "地合い条件"
        lines.append(f"新規：🛑 NO（{reason}）")
    else:
        lines.append("新規：✅ OK（指値 / 現値INは銘柄別）")
    lines.append("")

    fut_txt = f"  先物:{futures_chg:+.2f}%(NKD=F) {'Risk-ON' if risk_on else ''}".rstrip()
    lines.append(f"地合い：{mkt_score}（{mkt_comment}）  ΔMarketScore_3d:{delta3:.1f}{fut_txt}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"週次新規：{weekly_used} / {weekly_max}")
    lines.append(f"推奨レバ：{leverage:.1f}x")
    lines.append("")

    # Policy (explicit; was previously computed but not rendered)
    if policy_lines:
        lines.append("🧭 運用ルール（本日）")
        for p in policy_lines:
            if str(p).strip():
                lines.append("・" + str(p).strip())
        lines.append("")

    # Candidates (beginner-first)
    if cands:
        lines.append("👀 監視リスト（新規は見送り / 最大5）" if no_trade else "🏆 狙える形（ランキング / 最大5）")

        # Beginner-first output: show *what to do* (order type & price) and hide the rest.
        # Avoid confusing "buy limit above market" situations: if price is below the band, we do NOT suggest a limit order.
        band_tol = 0.0005  # 0.05% 表示/判定のズレを吸収

        def _risk_mid(entry_p: float, sl_p: float) -> float:
            if entry_p > 0 and sl_p > 0 and entry_p > sl_p:
                return float((entry_p - sl_p) / entry_p * 100.0)
            return float("nan")

        # Keep original ranking (idx) but renumber within each bucket for readability.
        order_items: List[Tuple[int, str]] = []
        watch_items: List[Tuple[int, str]] = []
        skip_items: List[Tuple[int, str]] = []

        for idx, c in enumerate(cands, 1):
            ticker = str(c.get("ticker", ""))
            name = str(c.get("name", ticker))
            setup = str(c.get("setup", "")).strip()

            entry_low = safe_float(c.get("entry_low"), 0.0)
            entry_high = safe_float(c.get("entry_high"), 0.0)
            entry_price = safe_float(c.get("entry_price"), (entry_low + entry_high) / 2.0)
            sl = safe_float(c.get("sl"), 0.0)
            tp1 = safe_float(c.get("tp1"), 0.0)
            close_last = safe_float(c.get("close_last"), 0.0)

            # Market-in (現値IN)
            entry_mode = str(c.get("entry_mode", "LIMIT_ONLY"))
            in_band = (
                (close_last > 0)
                and (entry_low > 0)
                and (entry_high > 0)
                and (close_last >= entry_low * (1.0 - band_tol))
                and (close_last <= entry_high * (1.0 + band_tol))
            )
            market_in_ok = bool(entry_mode == "MARKET_OK" and in_band and (not macro_on) and (not no_trade))

            # Liquidity summary tags (keep only what matters for execution)
            liq_grade = int(safe_float(c.get("liq_grade"), 0.0)) if c.get("liq_grade") is not None else 0
            adv20 = safe_float(c.get("adv20"), float("nan"))
            impact = safe_float(c.get("amihud_bps100m"), float("nan"))
            weekly_ok = c.get("weekly_ok", None)

            tags: List[str] = []
            if setup:
                tags.append(setup)
            if liq_grade == 2:
                tags.append("板厚◎")
            elif liq_grade == 1:
                tags.append("板厚○")
            if np.isfinite(adv20):
                tags.append(f"ADV{_fmt_oku(adv20)}")
            if np.isfinite(impact):
                tags.append(f"Imp{impact:.0f}")
            if weekly_ok is True:
                tags.append("週足OK")
            elif weekly_ok is False:
                tags.append("週足NG")
            tag_txt = f" [{'/'.join(tags)}]" if tags else ""

            # Compute concise reason
            ns = safe_float(c.get("noise_score"), float("nan"))
            q = safe_float(c.get("quality"), float("nan"))
            vr = safe_float(c.get("vol_ratio"), float("nan"))
            gf = safe_float(c.get("gap_freq"), float("nan"))
            gu = bool(c.get("gu", False))

            # Where is price vs band?
            above_band = bool(close_last > 0 and entry_high > 0 and close_last > entry_high * (1.0 + band_tol))
            below_band = bool(close_last > 0 and entry_low > 0 and close_last < entry_low * (1.0 - band_tol))

            # Beginner action classification
            # - ORDER: either market-in (rare) or safe pullback limit below current
            # - WATCH: below the band (do not suggest a limit order above market)
            # - SKIP: event day / macro / GU / quality-noise issues
            if no_trade:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（停止）"))
                continue
            if macro_on:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（イベント）"))
                continue
            if gu:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（GU）"))
                continue
            if weekly_ok is False and setup in ("A1-Strong", "A1"):
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（週足NG）"))
                continue
            if in_band and np.isfinite(ns) and ns >= 2:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（ノイズ{int(ns)}）"))
                continue
            if in_band and np.isfinite(vr) and vr > 1.35:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（出来高↑）"))
                continue
            if in_band and np.isfinite(gf) and gf > 0.25:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（Gap多）"))
                continue
            if in_band and np.isfinite(q) and q < -0.05:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（品質低）"))
                continue

            # Order suggestion
            if market_in_ok:
                r_mid = _risk_mid(close_last, sl)
                risk_txt = f"{r_mid:.1f}%" if np.isfinite(r_mid) else "-"
                order_items.append(
                    (
                        idx,
                        f"🟢 {ticker} {name}{tag_txt} 今IN（現値 {_fmt_yen(close_last)}） / SL {_fmt_yen(sl)} / TP1 {_fmt_yen(tp1)} / Risk {risk_txt}",
                    )
                )
                continue

            if above_band and entry_price > 0 and close_last > 0 and entry_price < close_last:
                r_mid = _risk_mid(entry_price, sl)
                risk_txt = f"{r_mid:.1f}%" if np.isfinite(r_mid) else "-"
                order_items.append(
                    (
                        idx,
                        f"🟢 {ticker} {name}{tag_txt} 指値（押し待ち）{_fmt_yen(entry_price)} / SL {_fmt_yen(sl)} / TP1 {_fmt_yen(tp1)} / Risk {risk_txt}",
                    )
                )
                continue

            if in_band and entry_price > 0 and close_last > 0:
                if entry_price <= close_last:
                    r_mid = _risk_mid(entry_price, sl)
                    risk_txt = f"{r_mid:.1f}%" if np.isfinite(r_mid) else "-"
                    order_items.append(
                        (
                            idx,
                            f"🟢 {ticker} {name}{tag_txt} 指値（帯内）{_fmt_yen(entry_price)} / SL {_fmt_yen(sl)} / TP1 {_fmt_yen(tp1)} / Risk {risk_txt}",
                        )
                    )
                else:
                    watch_items.append((idx, f"🟡 {ticker} {name}{tag_txt} 監視（帯内だが指値が上：注文は様子見）"))
                continue

            if below_band:
                watch_items.append((idx, f"🟡 {ticker} {name}{tag_txt} 監視（戻り待ち：帯まで距離あり）"))
                continue

            watch_items.append((idx, f"🟡 {ticker} {name}{tag_txt} 監視"))

        if order_items:
            lines.append("✅ 今日やること：注文（上から優先）")
            for n, (_rank, txt) in enumerate(sorted(order_items, key=lambda x: x[0]), 1):
                lines.append(f"{n}. {txt}")
        else:
            lines.append("✅ 今日やること：注文")
            lines.append("・該当なし")

        if watch_items:
            lines.append("")
            lines.append("👀 監視（まだ入らない）")
            for n, (_rank, txt) in enumerate(sorted(watch_items, key=lambda x: x[0]), 1):
                lines.append(f"{n}. {txt}")

        if skip_items:
            lines.append("")
            lines.append("🚫 見送り")
            for n, (_rank, txt) in enumerate(sorted(skip_items, key=lambda x: x[0]), 1):
                lines.append(f"{n}. {txt}")

        lines.append("")
    else:
        lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
        lines.append("・該当なし")
        lines.append("")

    # Positions (beginner-first: compact one line per position)
    if pos_text.strip():
        import re

        def _pick_num(line: str) -> str:
            m = re.search(r"([0-9]{1,3}(?:,[0-9]{3})*)", line)
            return m.group(1) if m else ""

        def _cut_tail(s: str) -> str:
            for sep in ("（", " / "):
                if sep in s:
                    s = s.split(sep, 1)[0]
            return s.strip()

        raw_lines = [ln.strip() for ln in pos_text.splitlines() if ln.strip()]
        blocks: List[List[str]] = []
        cur: List[str] = []
        for ln in raw_lines:
            if ln.startswith("■ ") and cur:
                blocks.append(cur)
                cur = []
            cur.append(ln)
        if cur:
            blocks.append(cur)

        lines.append("📊 ポジション（やること）")
        for b in blocks:
            head = b[0].replace("■", "").strip()
            status = ""
            next_act = ""
            entry = ""
            now = ""
            pnl = ""
            sl = ""
            tp1 = ""
            for ln in b[1:]:
                if "状態：" in ln and not status:
                    status = _cut_tail(ln.split("状態：", 1)[1])
                if "次アクション：" in ln and not next_act:
                    next_act = _cut_tail(ln.split("次アクション：", 1)[1])
                if "取得単価：" in ln and not entry:
                    m1 = re.search(r"取得単価：\s*([0-9,]+)\s*円", ln)
                    if m1:
                        entry = m1.group(1)
                    m2 = re.search(r"現値：\s*([0-9,]+)", ln)
                    if m2:
                        now = m2.group(1)
                if "Entry" in ln and not entry:
                    entry = _pick_num(ln)
                if ("現値：" in ln or "Now" in ln) and not now:
                    now = _pick_num(ln)
                if "損益：" in ln and not pnl:
                    pnl = _cut_tail(ln.split("損益：", 1)[1])
                if ("想定SL：" in ln or "SL：" in ln) and not sl:
                    sl = _pick_num(ln)
                if ("想定TP1：" in ln or "TP1：" in ln) and not tp1:
                    tp1 = _pick_num(ln)

            act = next_act or status or "保有"
            parts: List[str] = [f"■ {head}：{act}"]
            if entry:
                parts.append(f"Entry {entry}")
            if now:
                parts.append(f"Now {now}")
            if pnl:
                parts.append(f"PnL {pnl}")
            if sl:
                parts.append(f"SL {sl}")
            if tp1:
                parts.append(f"TP1 {tp1}")
            lines.append(" / ".join(parts))

        lines.append("")

    # Summary: removed (beginner-first mode). The actionable list above is the summary.

    # Saucer bucket (beginner-first)
    # Expected format: dict {"D":[...], "W":[...], "M":[...]}
    if saucers:
        def _iter_tf(key: str):
            if isinstance(saucers, dict):
                return list(saucers.get(key, []) or [])
            # backward-compat: legacy list with 'timeframe' = 'W'/'M'
            if isinstance(saucers, list):
                if key == "W":
                    return [x for x in saucers if str(x.get("timeframe", "W")) == "W"]
                if key == "M":
                    return [x for x in saucers if str(x.get("timeframe", "W")) == "M"]
                return []
            return []

        def _tf_title(key: str) -> str:
            return {"D": "日足", "W": "週足", "M": "月足"}.get(key, key)

        def _len_label(tf_key: str, n: int) -> str:
            if n <= 0:
                return "-"
            if tf_key == "W":
                return f"{n}週"
            if tf_key == "M":
                return f"{n}ヶ月"
            return f"{n}本"

        for key in ("D", "W", "M"):
            items = _iter_tf(key)[:5]
            lines.append(f"🥣 ソーサー枠（{_tf_title(key)}）ランキング（最大5）")
            if not items:
                lines.append("・該当なし")
                continue

            for idx, s in enumerate(items, 1):
                ticker = str(s.get("ticker", ""))
                name = str(s.get("name", ticker))
                tier = str(s.get("tier", "A") or "A")
                tier_tag = "（準候補）" if tier.upper() == "B" else ""

                rim_f = safe_float(s.get("rim"), 0.0)
                last_f = safe_float(s.get("last"), 0.0)
                atrp_f = safe_float(s.get("atrp"), 0.0)
                cup_len = int(s.get("cup_len", 0) or 0)
                progress = float(s.get("progress", 0.0))
                prog_pct = int(round(min(1.5, max(0.0, progress)) * 100))

                # Zone/SL
                entry_low = safe_float(s.get("entry_low"), float("nan"))
                entry_high = safe_float(s.get("entry_high"), float("nan"))
                sl_s = safe_float(s.get("sl"), float("nan"))
                hvol_ratio = safe_float(s.get("handle_vol_ratio"), float("nan"))
                warn = " ⚠" if (np.isfinite(hvol_ratio) and hvol_ratio >= 1.25) else ""

                # If scan provides an explicit zone, use it; otherwise fallback to a rim-buffer zone.
                if rim_f > 0 and np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low > 0 and entry_high > 0:
                    zone_low = float(min(entry_low, entry_high))
                    zone_high = float(max(entry_low, entry_high))
                else:
                    base_pre = {"D": 0.6, "W": 0.9, "M": 1.2}.get(key, 0.8)  # percent
                    max_pre = {"D": 2.0, "W": 3.0, "M": 4.0}.get(key, 2.5)   # percent
                    atr_pre = (atrp_f * 0.35) if atrp_f > 0 else 0.0
                    pre_buf_pct = max(base_pre, atr_pre)
                    pre_buf_pct = min(pre_buf_pct, max_pre)
                    zone_low = rim_f * (1.0 - pre_buf_pct / 100.0) if rim_f > 0 else 0.0
                    zone_high = rim_f * (1.0 - base_pre / 100.0) if rim_f > 0 else 0.0
                    zone_high = max(zone_low, zone_high)

                if not (zone_low > 0 and zone_high > 0 and np.isfinite(sl_s) and sl_s > 0):
                    lines.append(f"{idx}. 🟡 {ticker} {name}{tier_tag} 監視（ゾーン計算失敗）")
                    continue

                # Risk range inside the zone
                r1 = (zone_low - sl_s) / zone_low * 100.0
                r2 = (zone_high - sl_s) / zone_high * 100.0
                r_lo = min(r1, r2)
                r_hi = max(r1, r2)
                risk_txt = f"{r_lo:.1f}〜{r_hi:.1f}%" if abs(r_hi - r_lo) >= 0.15 else f"{r_hi:.1f}%"

                # Order type hint
                order_tag = "指値"
                tol_zone = 0.0010
                if last_f > 0:
                    if last_f < zone_low * (1.0 - tol_zone):
                        order_tag = "逆指値"
                    elif last_f > zone_high * (1.0 + tol_zone):
                        order_tag = "押し待ち指値"

                # Where is price now? (one short note)
                now_note = ""
                if last_f > 0:
                    if last_f < zone_low * (1.0 - tol_zone):
                        to_zone = (zone_low / last_f - 1.0) * 100.0
                        now_note = f"（今は下：+{to_zone:.1f}%待ち）"
                    elif last_f > zone_high * (1.0 + tol_zone):
                        over = (last_f / zone_high - 1.0) * 100.0
                        risk_last = (last_f - sl_s) / last_f * 100.0
                        chase = f" r_now {risk_last:.1f}%" if np.isfinite(risk_last) else ""
                        if np.isfinite(risk_last) and risk_last > 8.0:
                            chase += "（今買うな）"
                        now_note = f"（今は上：+{over:.1f}%{chase}）"
                    else:
                        now_note = "（ゾーン内）"

                # Print one line per symbol
                if order_tag == "逆指値":
                    if abs(zone_high / zone_low - 1.0) <= 0.001:
                        ord_txt = f"逆指値 Trg {_fmt_yen(zone_low)}"
                    else:
                        ord_txt = f"逆指値 Trg {_fmt_yen(zone_low)} / 上限 {_fmt_yen(zone_high)}"
                else:
                    if abs(zone_high / zone_low - 1.0) <= 0.001:
                        ord_txt = f"指値 {_fmt_yen(zone_low)}"
                    else:
                        ord_txt = f"指値 {_fmt_yen(zone_low)}〜{_fmt_yen(zone_high)}"
                    if order_tag == "押し待ち指値":
                        ord_txt = "指値（押し待ち）" + ord_txt.replace("指値 ", "")

                lines.append(
                    f"{idx}. 🟢 {ticker} {name}{tier_tag}{warn} {ord_txt} / SL {_fmt_yen(sl_s)} / Risk {risk_txt} / 進捗{prog_pct}% / 長さ{_len_label(key, cup_len)} {now_note}"
                )
    return "\n".join(lines).rstrip() + "\n"
