
def calc_rr(entry, stop, tp2):
    return abs(tp2-entry)/abs(entry-stop)

def calc_ev(rr):
    return rr * 0.4
