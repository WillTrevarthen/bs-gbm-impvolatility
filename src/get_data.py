import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def get_risk_free_rate(ticker="^IRX"):
    """
    Fetches the 13-week Treasury Bill rate to use as the risk-free rate.
    Divided by 100 to convert from percentage (e.g., 5.0) to decimal (0.05).
    """
    try:
        # ^IRX is the ticker for 13 Week Treasury Bill Yield
        irx = yf.Ticker(ticker)
        hist = irx.history(period="1d")
        if hist.empty:
            return 0.05 # Fallback to 5% if data fails
        return hist['Close'].iloc[-1] / 100
    except Exception as e:
        print(f"Warning: Could not fetch risk-free rate. Defaulting to 5%. {e}")
        return 0.05

def get_market_data(ticker, specific_expiry=None):
    """
    Fetches Spot Price, Risk-Free Rate, and Option Chain.
    
    Args:
        ticker (str): The stock symbol (e.g., "SPY", "AAPL")
        specific_expiry (str): Optional. Date in 'YYYY-MM-DD'. 
                               If None, selects the 4th available expiration (approx 1 month out).
    """
    print(f"--- Fetching Data for {ticker} ---")
    
    # 1. Get Stock Object
    stock = yf.Ticker(ticker)
    
    # 2. Get Spot Price (S)
    try:
        hist = stock.history(period="1d")
        spot_price = hist['Close'].iloc[-1]
        print(f"Current Spot Price: ${spot_price:.2f}")
    except IndexError:
        raise ValueError(f"Could not fetch data for {ticker}. Check ticker symbol.")

    # 3. Get Risk Free Rate (r)
    risk_free_rate = get_risk_free_rate()
    print(f"Risk-Free Rate: {risk_free_rate:.2%}")

    # 4. Handle Expiration Dates
    expirations = stock.options
    if not expirations:
        raise ValueError("No options data found for this ticker.")
    
    # Select Expiry
    if specific_expiry:
        if specific_expiry not in expirations:
            print(f"Available expirations: {expirations[:5]}...")
            raise ValueError(f"Expiration {specific_expiry} not found.")
        target_date = specific_expiry
    else:
        # Default: Pick the 4th expiration (usually ~3-4 weeks out)
        # We skip the first few to avoid erratic gamma/theta near expiry
        target_idx = min(3, len(expirations)-1) 
        target_date = expirations[target_idx]
    
    print(f"Selected Expiration: {target_date}")

    # 5. Get Option Chain
    chain = stock.option_chain(target_date)
    calls = chain.calls
    puts = chain.puts
    
    # 6. Calculate Time to Expiry (T) in Years
    today = datetime.now()
    expiry_dt = datetime.strptime(target_date, "%Y-%m-%d")
    
    # T = (Target Date - Today) / 365
    days_to_expiry = (expiry_dt - today).days
    if days_to_expiry <= 0:
         # Handle case where option expires today/tomorrow
        days_to_expiry = 1 
    
    T = days_to_expiry / 365.25
    print(f"Time to Expiry (T): {T:.4f} years ({days_to_expiry} days)")

    # 7. Data Cleaning
    # Filter for liquidity: Remove strikes with no bids or no open interest
    # We calculate 'mid_price' because 'lastPrice' can be stale
    
    def clean_df(df):
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        # Drop rows where there is essentially no market
        df = df[ (df['bid'] > 0) & (df['openInterest'] > 0) ].copy()
        return df

    calls_clean = clean_df(calls)
    puts_clean = clean_df(puts)

    return {
        "ticker": ticker,
        "spot_price": spot_price,
        "risk_free_rate": risk_free_rate,
        "T": T,
        "expiry_date": target_date,
        "calls": calls_clean,
        "puts": puts_clean
    }

if __name__ == "__main__":
    # Test the function
    try:
        # Using SPY as it simulates S&P 500 (European-ish behavior)
        data = get_market_data("SPY") 
        
        print("\n--- Data Sample ---")
        print(data['calls'][['strike', 'lastPrice', 'mid_price', 'impliedVolatility', 'openInterest']].head())
    except Exception as e:
        print(e)