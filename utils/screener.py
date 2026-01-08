
from utils.market import calc_market_score
from utils.events import get_events
from utils.sector import top_sectors
from utils.features import extract_features
from utils.setup import classify_setup
from utils.entry import calc_entry
from utils.rr_ev import calc_rr, calc_ev
from utils.util import rr_min_by_market

def run_screening():
    market = calc_market_score()
    rr_min = rr_min_by_market(market["score"])

    candidates = []
    for t in ["4063.T","4182.T"]:
        f = extract_features(t)
        setup = classify_setup(f)
        ent = calc_entry(f)
        rr = calc_rr(ent["entry"], ent["stop"], ent["tp2"])
        ev = calc_ev(rr)

        if rr >= rr_min and ev >= 0.5:
            candidates.append({
                "ticker": t,
                "setup": setup,
                "rr": rr,
                "ev": ev
            })

    return {
        "market": market,
        "candidates": candidates,
        "events": get_events(),
        "sectors": top_sectors()
    }
