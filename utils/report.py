from __future__ import annotations

from typing import Dict, List
import numpy as np


from utils.util import safe_float

def _fmt_yen(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"

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

    # Candidates
    if cands:
        if no_trade:
            lines.append("👀 監視リスト（新規は見送り / 最大5）")
        else:
            lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
        for c in cands:
            ticker = str(c.get("ticker", ""))
            name = str(c.get("name", ticker))
            sector = str(c.get("sector", ""))
            entry_mode = str(c.get("entry_mode", "LIMIT_ONLY"))
            market_in_ok = bool(entry_mode == "MARKET_OK" and (not macro_on) and (not no_trade))
            suffix = "（現値IN可）" if market_in_ok else ""
            lines.append(f"■ {ticker} {name}（{sector}）{suffix}")
            lines.append("")
            # Entry
            lines.append("【エントリー】")
            entry_low = safe_float(c.get('entry_low'), 0.0)
            entry_high = safe_float(c.get('entry_high'), 0.0)
            entry_price = safe_float(c.get('entry_price'), (entry_low + entry_high) / 2.0)
            sl = safe_float(c.get('sl'), 0.0)
            close_last = safe_float(c.get('close_last'), 0.0)
            risk_pct = safe_float(c.get('risk_pct'), 0.0)
            rr0 = safe_float(c.get('rr'), 0.0)
            p_hit0 = safe_float(c.get('p_hit'), 0.0)
            p_be0 = (1.0 / (rr0 + 1.0)) if rr0 > 0 else 1.0
            band_tol = 0.0005  # 0.05%: 表示丸め/取得誤差の吸収（screener側と合わせる）
            in_band_tol = (
                (close_last > 0)
                and (entry_low > 0)
                and (entry_high > 0)
                and (close_last >= entry_low * (1.0 - band_tol))
                and (close_last <= entry_high * (1.0 + band_tol))
            )
            if risk_pct <= 0.0 and entry_price > 0 and sl > 0:
                risk_pct = (entry_price - sl) / entry_price * 100.0
            
            lines.append(f"・指値目安（中央）：{_fmt_yen(entry_price)} 円")
            if entry_low > 0 and entry_high > 0:
                lines.append(f"・エントリー帯：{_fmt_yen(entry_low)} 〜 {_fmt_yen(entry_high)} 円")
            if close_last > 0 and entry_low > 0 and entry_high > 0:
                # Distance to entry band (readability-first).
                # IMPORTANT: use the same tolerance as the screener's entry_mode (band_tol),
                # so the report and the decision logic never diverge.
                if in_band_tol:
                    dist_txt = "（帯内）"
                elif close_last < entry_low:
                    need = (entry_low / close_last - 1.0) * 100.0
                    dist_txt = f"（帯まで +{need:.1f}%）"
                else:
                    need = (close_last / entry_high - 1.0) * 100.0
                    dist_txt = f"（帯まで -{need:.1f}%）"
                lines.append(f"・現値（終値）：{_fmt_yen(close_last)} 円{dist_txt}")

            if bool(c.get('gu', False)):
                lines.append("・GU：Yes（寄り後再判定）")
            
            lines.append(f"・現値IN：{'OK' if market_in_ok else 'NG'}")
            if not market_in_ok:
                # NG reason (deterministic, aligned with entry_mode logic and global constraints)
                reason = None
                if no_trade:
                    reason = "新規停止中"
                elif macro_on:
                    reason = "重要イベント警戒"
                elif bool(c.get('gu', False)):
                    reason = "GU（寄り後再判定）"
                elif close_last > 0 and entry_low > 0 and close_last < entry_low * (1.0 - band_tol):
                    reason = "現値がエントリー帯より下（待ち）"
                elif close_last > 0 and entry_high > 0 and close_last > entry_high * (1.0 + band_tol):
                    reason = "現値がエントリー帯より上（押し待ち/指値）"
                elif (in_band_tol and risk_pct >= 8.0):
                    reason = f"リスク幅大（{risk_pct:.1f}%）"
                elif (in_band_tol and (p_hit0 < (p_be0 + 0.10))):
                    reason = f"到達確率が損益分岐を十分上回らない（p={p_hit0:.3f} / p_be={p_be0:.3f}）"
                elif mkt_score < 60:
                    reason = f"地合い不足（{mkt_score}<60）"
                else:
                    reason = "条件未達"
                lines.append(f"・NG理由：{reason}")
            
            lines.append(f"・損切り：{_fmt_yen(sl)} 円")
            warn = " ⚠" if risk_pct >= 8.0 else ""
            if risk_pct > 0:
                lines.append(f"・リスク幅：{risk_pct:.1f}%{warn}")
            lines.append("")
            # Targets (single line)
            lines.append("【利確目標】")
            lines.append(f"・利確①：{_fmt_yen(c.get('tp1', 0.0))} 円、②：{_fmt_yen(c.get('tp2', 0.0))} 円")
            lines.append("")
            # Indicators
            lines.append("【指標（参考）】")
            lines.append(f"・CAGR寄与度（/日）：{c.get('cagr', 0.0):.2f}")
            p_hit = p_hit0
            rr = rr0
            exp_r_hit = safe_float(c.get('exp_r_hit'), rr * p_hit)
            p_be = p_be0
            ev_r = (p_hit * rr) - ((1.0 - p_hit) * 1.0)
            lines.append(f"・到達確率（目安）：{p_hit:.3f}（損益分岐 p={p_be:.3f}）")
            lines.append(f"・期待値（R）：{ev_r:.2f}R")
            lines.append(f"・期待R×到達確率（参考）：{exp_r_hit:.2f}")
            lines.append(f"・RR（TP1基準）：{rr:.2f}")
            lines.append(f"・想定日数（中央値）：{safe_float(c.get('expected_days'), 0.0):.1f}日")
            lines.append("")
    else:
        lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
        lines.append("該当なし")
        lines.append("")

    # Positions (as-is; already unified in latest spec for audit, if enabled upstream)
    if pos_text.strip():
        lines.append("📊 ポジション")
        lines.append(pos_text.rstrip())
        lines.append("")

    # Summary (all displayed cands, in order)
    if cands:
        lines.append("まとめ（指値一覧）")
        for i, c in enumerate(cands, 1):
            ticker = str(c.get("ticker", ""))
            name = str(c.get("name", ticker))
            entry = _fmt_yen(c.get("entry_price", (c.get("entry_low",0)+c.get("entry_high",0))/2.0))
            lines.append(f"{i}. {ticker} {name}：{entry} 円")
        lines.append("")

    # Saucer bucket (separate; requested to be at the very end)
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

        for key in ("D", "W", "M"):
            items = _iter_tf(key)[:5]
            lines.append(f"🥣 ソーサー枠（{_tf_title(key)}）最大5")
            if not items:
                lines.append("・該当なし")
                continue
            for s in items:
                ticker = str(s.get("ticker", ""))
                name = str(s.get("name", ticker))
                sector = str(s.get("sector", ""))
                tier = str(s.get("tier", "A") or "A")
                tier_tag = "（準候補）" if tier.upper() == "B" else ""
                rim_f = safe_float(s.get("rim"), 0.0)
                last_f = safe_float(s.get("last"), 0.0)
                atrp_f = safe_float(s.get("atrp"), 0.0)
                cup_len = int(s.get("cup_len", 0) or 0)
                progress = float(s.get("progress", 0.0))
                prog_pct = int(round(min(1.5, max(0.0, progress)) * 100))
                depth = float(s.get("depth", 0.0))

                handle_low = safe_float(s.get("handle_low"), float("nan"))
                handle_high = safe_float(s.get("handle_high"), float("nan"))
                handle_pb = safe_float(s.get("handle_pb"), float("nan"))
                handle_ok = bool(s.get("handle_ok", False))

                def _len_label(tf_key: str, n: int) -> str:
                    if n <= 0:
                        return "-"
                    if tf_key == "W":
                        return f"{n}週"
                    if tf_key == "M":
                        return f"{n}ヶ月"
                    return f"{n}本"

                # Execution zone from saucer scan (preferred).
                entry_low = safe_float(s.get("entry_low"), float("nan"))
                entry_high = safe_float(s.get("entry_high"), float("nan"))
                sl_s = safe_float(s.get("sl"), float("nan"))
                risk_s = safe_float(s.get("risk_pct"), float("nan"))
                handle_len = int(s.get("handle_len", 0) or 0)
                hvol_ratio = safe_float(s.get("handle_vol_ratio"), float("nan"))

                # Your intent: enter *before* a clean breakout, inside the handle, near the rim ("ココ").
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
                lines.append(f"■ {ticker} {name}（{sector}）[{_tf_title(key)}]{tier_tag}")
                extra_parts = []
                if rim_f > 0:
                    extra_parts.append(f"リム {_fmt_yen(rim_f)}")
                if np.isfinite(sl_s) and sl_s > 0:
                    extra_parts.append(f"SL {_fmt_yen(sl_s)}")
                # Risk is a function of where you get filled inside the zone.
                # To avoid misleading "one number", show a range (low->high) when possible.
                risk_txt = ""
                if np.isfinite(sl_s) and sl_s > 0 and zone_low > 0 and zone_high > 0:
                    r1 = (zone_low - sl_s) / zone_low * 100.0
                    r2 = (zone_high - sl_s) / zone_high * 100.0
                    if np.isfinite(r1) and np.isfinite(r2) and r1 > 0 and r2 > 0:
                        r_lo = min(r1, r2)
                        r_hi = max(r1, r2)
                        if abs(r_hi - r_lo) >= 0.15:
                            risk_txt = f"リスク {r_lo:.1f}〜{r_hi:.1f}%"
                        else:
                            risk_txt = f"リスク {r_hi:.1f}%"
                elif np.isfinite(risk_s) and risk_s > 0:
                    risk_txt = f"リスク {risk_s:.1f}%"
                if risk_txt:
                    extra_parts.append(risk_txt)
                extra = " / ".join(extra_parts) if extra_parts else ""

                # Order type hint (important):
                # - If current close is below the zone, a *limit* above the market is misleading.
                #   Treat it as a stop/trigger idea (buy when price rises into the zone).
                # - If above the zone, it's an "押し待ち".
                order_tag = "指値"
                if last_f > 0 and zone_low > 0 and zone_high > 0:
                    tol_zone = 0.0010
                    if last_f < zone_low * (1.0 - tol_zone):
                        order_tag = "逆指値"
                    elif last_f > zone_high * (1.0 + tol_zone):
                        order_tag = "押し待ち指値"

                if rim_f > 0 and zone_low > 0 and zone_high > 0:
                    # Print order guide:
                    # - 逆指値: show trigger/limit to avoid "range means limit" confusion
                    # - 指値/押し待ち指値: show zone as a band
                    if order_tag == "逆指値":
                        if abs(zone_high / zone_low - 1.0) <= 0.001:
                            lines.append(
                                f"・IN（先回り/ハンドル 逆指値）：トリガー {_fmt_yen(zone_low)} 円（{extra}）"
                            )
                        else:
                            lines.append(
                                f"・IN（先回り/ハンドル 逆指値）：トリガー {_fmt_yen(zone_low)} 円 / 上限 {_fmt_yen(zone_high)} 円（{extra}）"
                            )
                    else:
                        if abs(zone_high / zone_low - 1.0) <= 0.001:
                            lines.append(
                                f"・IN（先回り/ハンドル {order_tag}）：{_fmt_yen(zone_low)} 円（{extra}）"
                            )
                        else:
                            lines.append(
                                f"・IN（先回り/ハンドル {order_tag}）：{_fmt_yen(zone_low)} 〜 {_fmt_yen(zone_high)} 円（{extra}）"
                            )
                else:
                    lines.append("・IN（先回り/ハンドル 指値）：-")

                # Show where the current (TF-close) is relative to IN zone and rim.
                if last_f > 0 and rim_f > 0:
                    tol_zone = 0.0010
                    tol_rim = {"D": 0.0020, "W": 0.0030, "M": 0.0050}.get(key, 0.0020)
                    in_zone = bool(zone_low > 0 and zone_high > 0 and (zone_low * (1.0 - tol_zone) <= last_f <= zone_high * (1.0 + tol_zone)))

                    # Distance helpers
                    dist_to_rim = (rim_f / last_f - 1.0) * 100.0 if last_f > 0 else float("nan")
                    risk_last = float("nan")
                    if np.isfinite(sl_s) and sl_s > 0 and last_f > 0:
                        risk_last = (last_f - sl_s) / last_f * 100.0
                    risk_last_note = ""
                    if np.isfinite(risk_last) and risk_last > 8.0:
                        risk_last_note = f" / 現値リスク {risk_last:.1f}%（上限超）"
                    dist_txt = ""
                    if abs(last_f / rim_f - 1.0) <= tol_rim:
                        # around the rim
                        if in_zone:
                            dist_txt = "（INゾーン内 / リム付近）"
                        else:
                            # above zone but still at rim-ish
                            if zone_high > 0 and last_f > zone_high * (1.0 + tol_zone):
                                over = (last_f / zone_high - 1.0) * 100.0
                                dist_txt = f"（INゾーン外（上） / INゾーン上 +{over:.1f}% / リム付近{risk_last_note}）"
                            else:
                                dist_txt = "（リム付近）"
                    elif last_f < rim_f:
                        # below rim
                        if in_zone:
                            dist_txt = f"（INゾーン内 / リムまで +{dist_to_rim:.1f}%）"
                        else:
                            if zone_low > 0 and last_f < zone_low * (1.0 - tol_zone):
                                to_zone = (zone_low / last_f - 1.0) * 100.0
                                dist_txt = f"（INゾーン外（下） / INゾーンまで +{to_zone:.1f}% / リムまで +{dist_to_rim:.1f}%）"
                            elif zone_high > 0 and last_f > zone_high * (1.0 + tol_zone):
                                over = (last_f / zone_high - 1.0) * 100.0
                                dist_txt = f"（INゾーン外（上） / INゾーン上 +{over:.1f}% / リムまで +{dist_to_rim:.1f}%{risk_last_note}）"
                            else:
                                dist_txt = f"（INゾーン外 / リムまで +{dist_to_rim:.1f}%）"
                    else:
                        # above rim (already broke out)
                        up = (last_f / rim_f - 1.0) * 100.0
                        dist_txt = f"（上抜け済 +{up:.1f}%）"

                    htxt = "-"
                    if np.isfinite(handle_pb) and handle_pb > 0:
                        htxt = f"{handle_pb * 100.0:.1f}%"

                    hlen_txt = "-"
                    if handle_len > 0:
                        hlen_txt = _len_label(key, handle_len)
                    vtxt = "-"
                    if np.isfinite(hvol_ratio) and hvol_ratio > 0:
                        vtxt = f"{hvol_ratio:.2f}x"
                        # Handle volume expansion is typically not ideal; mark it when clearly expanding.
                        if hvol_ratio >= 1.25:
                            vtxt = vtxt + " ⚠"

                    lines.append(
                        f"・現値（終値）：{_fmt_yen(last_f)} 円{dist_txt}（進捗 {prog_pct}% / 深さ {depth:.0%} / ハンドル {htxt} / 期間 {hlen_txt} / 出来高 {vtxt} / 長さ {_len_label(key, cup_len)}）"
                    )
                else:
                    lines.append(
                        f"・進捗 {prog_pct}% / 深さ {depth:.0%} / 長さ {_len_label(key, cup_len)}"
                    )
    return "\n".join(lines).rstrip() + "\n"
