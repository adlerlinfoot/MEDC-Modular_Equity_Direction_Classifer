import numpy as np
import pandas as pd
from src.optimize import run_optimize
from src.data_loader import load_equity

# -----------------------------------------
# Universe definition (100 tickers)
# -----------------------------------------
UNIVERSE = {
    "Tech": [
        "AAPL","MSFT","NVDA","GOOGL","META","CRM","ADBE","ORCL","AMD","INTC",
        "AMAT","QCOM","TXN","MU","IBM"
    ],
    "Consumer": [
        "TSLA","NKE","HD","SBUX","MCD","COST","TGT","LOW","MAR","BKNG"
    ],
    "Healthcare": [
        "JNJ","PFE","MRK","ABBV","LLY","BMY","AMGN","REGN","VRTX","GILD"
    ],
    "Financials": [
        "JPM","BAC","GS","MS","WFC","C","BLK","SCHW","AXP","USB"
    ],
    "Energy": [
        "XOM","CVX","COP","SLB","HAL","PSX","MPC","EOG","PXD","DVN"
    ],
    "Industrials": [
        "CAT","DE","HON","GE","UPS","FDX","LMT","BA","RTX","MMM"
    ],
    "Materials": [
        "LIN","NEM","FCX","CLF","APD","ALB","DD","ECL","MLM","VMC"
    ],
    "Utilities": [
        "NEE","DUK","SO","AEP","EXC","SRE","D","XEL","WEC","PEG"
    ],
    "Small/Mid Cap Mix": [
        "PLTR","NET","CRWD","RBLX","UPST","AFRM","SQ","SHOP","ROKU","FSLR"
    ]
}

# Flatten universe
ALL_TICKERS = [t for sector in UNIVERSE.values() for t in sector]


# -----------------------------------------
# Helper: classify market cap bucket
# -----------------------------------------
def get_market_cap_bucket(ticker):
    try:
        df = load_equity(ticker, start="2010-01-01")
        mc = df["market_cap"].iloc[-1] if "market_cap" in df.columns else None
    except:
        mc = None

    if mc is None:
        return "Unknown"

    if mc > 200e9:
        return "Mega"
    elif mc > 10e9:
        return "Large"
    elif mc > 2e9:
        return "Mid"
    else:
        return "Small"


# -----------------------------------------
# Main evaluation
# -----------------------------------------
def evaluate_universe(start="2010-01-01", end=None):
    results = []

    for ticker in ALL_TICKERS:
        print(f"\n=== Evaluating {ticker} ===")
        try:
            best_params, val_acc = run_optimize(ticker, start=start, end=end)
        except Exception as e:
            print(f"Error on {ticker}: {e}")
            continue

        # sector lookup
        sector = next((s for s, lst in UNIVERSE.items() if ticker in lst), "Unknown")
        cap_bucket = get_market_cap_bucket(ticker)

        results.append({
            "ticker": ticker,
            "sector": sector,
            "cap_bucket": cap_bucket,
            "accuracy": val_acc
        })

    df = pd.DataFrame(results)

    # Sort best → worst
    df_sorted = df.sort_values("accuracy", ascending=False)

    # Average accuracy
    avg_acc = df["accuracy"].mean()

    # Sector averages
    sector_avg = df.groupby("sector")["accuracy"].mean().sort_values(ascending=False)

    # Market cap averages
    cap_avg = df.groupby("cap_bucket")["accuracy"].mean().sort_values(ascending=False)

    return df_sorted, avg_acc, sector_avg, cap_avg


# -----------------------------------------
# CLI
# -----------------------------------------
if __name__ == "__main__":
    df_sorted, avg_acc, sector_avg, cap_avg = evaluate_universe()

    print("\n\n=== OVERALL AVERAGE ACCURACY ===")
    print(f"{avg_acc:.4f}")

    print("\n=== BEST → WORST TICKERS ===")
    print(df_sorted.to_string(index=False))

    print("\n=== SECTOR AVERAGES ===")
    print(sector_avg)

    print("\n=== MARKET CAP AVERAGES ===")
    print(cap_avg)
