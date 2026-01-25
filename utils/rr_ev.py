# === rr_ev.py (contract-fixed) ===
def calc_ev(setup, mkt_score: int, macro_on: bool):
    exp_r = setup.exp_r_tp1
    prob = setup.hit_prob
    days = max(setup.exp_days, 1e-6)
    ev = (exp_r * prob) / days

    passes = (
        setup.rr >= setup.rr_min and
        ev >= setup.ev_min and
        setup.r_per_day >= setup.rday_min
    )

    debug = {
        "exp_r": exp_r,
        "prob": prob,
        "days": days,
        "ev": ev,
    }
    return ev, passes, debug
