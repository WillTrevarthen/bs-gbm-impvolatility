import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from datetime import datetime

# Import your custom math modules
from bs_math import implied_volatility
from get_data import get_risk_free_rate

def get_surface_data(ticker, max_expirations=6):
    """
    Loops through option expirations to build a dataset of:
    [Strike, Time_to_Expiry, Implied_Volatility]
    """
    print(f"--- Fetching 3D Surface Data for {ticker} ---")
    stock = yf.Ticker(ticker)
    
    # 1. Get Spot Price
    try:
        spot_price = stock.history(period="1d")['Close'].iloc[-1]
        print(f"Spot Price: ${spot_price:.2f}")
    except:
        print("Error fetching spot price.")
        return None

    r = get_risk_free_rate()
    expirations = stock.options
    
    surface_data = [] # List to store [strike, T, iv]
    
    # 2. Loop through expirations (limit to top N to save time)
    for date_str in expirations[:max_expirations]:
        print(f"Processing Expiry: {date_str}...")
        
        # Calculate T
        expiry_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_to_expiry = (expiry_date - datetime.now()).days
        
        # Skip if expiry is today or passed (avoids math errors)
        if days_to_expiry < 2:
            continue
            
        T = days_to_expiry / 365.25
        
        # Get Chain
        try:
            chain = stock.option_chain(date_str)
            calls = chain.calls
            
            # 3. Filtering to speed up calculation and reduce noise
            # Only take strikes within 20% of spot price (Moneyness 0.8 to 1.2)
            # Only take options with liquidity (Bid > 0)
            mask = (
                (calls['strike'] > spot_price * 0.8) & 
                (calls['strike'] < spot_price * 1.2) & 
                (calls['bid'] > 0)
            )
            filtered_calls = calls[mask]
            
            # 4. Calculate IV using YOUR Newton-Raphson solver
            for index, row in filtered_calls.iterrows():
                K = row['strike']
                market_price = (row['bid'] + row['ask']) / 2
                
                # Use your custom solver
                iv = implied_volatility(market_price, spot_price, K, T, r)
                
                # Filter bad calculations (nan) or extreme outliers
                if not np.isnan(iv) and 0.05 < iv < 3.0:
                    surface_data.append([K, T, iv])
                    
        except Exception as e:
            print(f"Skipping {date_str}: {e}")

    return pd.DataFrame(surface_data, columns=['Strike', 'Time', 'IV'])

def plot_3d_surface(df, ticker):
    """
    Renders the Volatility Surface.
    """
    if df.empty:
        print("No data collected.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Data for plotting
    X = df['Strike']
    Y = df['Time']
    Z = df['IV']

    # Plot Trisurf (Triangular Surface) - good for scattered data
    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, edgecolor='none', alpha=0.8)

    # Labels
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Time to Expiry (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Implied Volatility Surface: {ticker}')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Volatility')
    
    # Adjust view angle for better "3D" effect
    ax.view_init(elev=30, azim=220)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Choose a liquid ticker (SPY, QQQ, IWM, AAPL)
    # SPY is best because it has strikes every $1
    ticker = "SPY" 
    
    # 1. Collect Data
    # Note: This might take 10-20 seconds to solve IVs for all options
    df = get_surface_data(ticker, max_expirations=8)
    
    # 2. Visualize
    plot_3d_surface(df, ticker)