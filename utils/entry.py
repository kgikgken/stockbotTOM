
def calc_entry(features):
    p = features["price"]
    return {
        "entry": p,
        "stop": p * 0.97,
        "tp2": p * 1.06
    }
