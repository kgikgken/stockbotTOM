
def generate_report(data):
    lines = []
    m = data["market"]
    lines.append(f"地合い: {m['score']} 点")
    lines.append("")
    for c in data["candidates"]:
        lines.append(f"{c['ticker']} {c['setup']} RR:{c['rr']:.2f} EV:{c['ev']:.2f}")
    return "\n".join(lines)
