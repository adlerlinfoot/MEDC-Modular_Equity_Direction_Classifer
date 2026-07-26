import yfinance as yf
import pandas as pd

def load_equity(ticker, start="2010-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, progress=False)

    # Flatten MultiIndex columns: (Close, NVDA)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]

    # Bring Date out of index
    df = df.reset_index().rename(columns={"Date": "date"})

    # Clean duplicates
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    return df.reset_index(drop=True)
