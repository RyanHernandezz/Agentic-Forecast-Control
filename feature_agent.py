import yfinance as yf
import pandas as pd
import numpy as np

class FeatureAgent:
    """Fetches and engineers struct macro features for Crude Oil forecasting."""
    
    def __init__(self, tickers: list = None):
        self.tickers = tickers if tickers else ["CL=F", "^VIX", "SPY", "^TNX", "DX-Y.NYB"]

    def fetch_data(self, period: str = "1y") -> pd.DataFrame:
        df_list = []
        for ticker in self.tickers:
            t_df = yf.download(ticker, period=period, progress=False)[["Close"]].copy()
            if isinstance(t_df.columns, pd.MultiIndex):
                t_df.columns = t_df.columns.get_level_values(0).astype(str)
            
            # Universally rename whatever level 0 was ('Close' usually) to the ticker name
            t_df.rename(columns={"Close": ticker, "Price": ticker}, inplace=True)
            df_list.append(t_df)
        
        df = pd.concat(df_list, axis=1)
        df = df.resample('B').ffill(limit=2)
        
        macro_tickers = [t for t in self.tickers if t != "CL=F"]
        for ticker in macro_tickers:
            if ticker in df.columns:
                df[f"{ticker}_mom_5"] = df[ticker].pct_change(5)
                df[f"{ticker}_mom_20"] = df[ticker].pct_change(20)
                df[f"{ticker}_corr_CL"] = df[ticker].rolling(20).corr(df["CL=F"])
                
        df.bfill(inplace=True)
        df.ffill(inplace=True) 
        
        # We rename the root 'CL=F' back to 'Close' strictly to satisfy ModelerAgent legacy hardcodings
        if "CL=F" in df.columns:
            df.rename(columns={"CL=F": "Close"}, inplace=True)
            
        return df
