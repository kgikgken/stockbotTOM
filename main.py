
import pandas as pd, yfinance as yf
from stockbotTOM_scoring_vTrendGate import score_stock, in_rank
from stockbotTOM_rr_vTrendGate import compute_tp_sl_rr
from stockbotTOM_qualify_vTrendGate import trend_gate

UNIVERSE_PATH="universe_jpx.csv"

def main():
    uni=pd.read_csv(UNIVERSE_PATH)
    rows=[]
    for t in uni.ticker:
        df=yf.Ticker(t).history(period="260d", auto_adjust=True)
        if df is None or len(df)<80: continue
        if not trend_gate(df): continue
        sc=score_stock(df)
        if sc is None: continue
        rr=compute_tp_sl_rr(df)
        ev=0.45*rr["rr"]-0.55
        rows.append((ev,rr["rr"],t,sc,in_rank(df),rr["entry"]))
    rows=sorted(rows, reverse=True)[:5]
    for r in rows:
        print(r)

if __name__=="__main__":
    main()
