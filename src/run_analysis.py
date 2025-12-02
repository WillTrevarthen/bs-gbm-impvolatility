import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from get_data import get_market_data
from bs_math import implied_volatility, get_breeden_litzenberger_pdf

def main():
    # 1. Fetch Data
    ticker = "SPY"
    data = get_market_data(ticker)
    
    S0 = data['spot_price']
    r = data['risk_free_rate']
    T = data['T']
    
    # Filter for deep ITM and OTM to get a wide range, but focus near spot
    # Cleaning: Ensure strikes are sorted
    calls = data['calls'].sort_values('strike').reset_index(drop=True)
    
    # 2. Calculate Implied Volatility (Newton-Raphson) for each strike
    print("\n--- Calculating Implied Volatility (Newton-Raphson) ---")
    ivs = []
    valid_strikes = []
    valid_prices = []
    
    for index, row in calls.iterrows():
        K = row['strike']
        market_price = row['mid_price'] # Use mid-price for better accuracy
        
        # Calculate IV manually
        iv = implied_volatility(market_price, S0, K, T, r)
        
        # Filter out failed calculations or extreme outliers
        if not np.isnan(iv) and 0.01 < iv < 2.0:
            ivs.append(iv)
            valid_strikes.append(K)
            valid_prices.append(market_price)

    # Convert to numpy arrays for vector math
    strikes_arr = np.array(valid_strikes)
    prices_arr = np.array(valid_prices)
    iv_arr = np.array(ivs)

    # 3. Breeden-Litzenberger: Extract the Market PDF
    # Note: smooth_factor needs tuning based on data noisiness. 
    # For SPY (liquid), 0.05-0.5 is usually okay. For AAPL, might need higher.
    print("--- Extracting Market Probability Density (Breeden-Litzenberger) ---")
    market_pdf, price_spline = get_breeden_litzenberger_pdf(strikes_arr, prices_arr, T, r, smooth_factor=1.5)

    # 4. Generate the Theoretical Log-Normal PDF (Black-Scholes Assumption)
    # We use the ATM Volatility for this "Standard Model"
    # Find index of strike closest to spot
    atm_idx = (np.abs(strikes_arr - S0)).argmin()
    atm_iv = iv_arr[atm_idx]
    
    print(f"ATM Volatility (for comparison): {atm_iv:.2%}")
    
    # Log-Normal PDF Formula
    # Scale parameter is S0 * exp(rT) roughly, shape is sigma*sqrt(T)
    sigma_std = atm_iv * np.sqrt(T)
    mu_std = np.log(S0) + (r - 0.5 * atm_iv**2) * T
    
    # x values for plotting
    x_points = np.linspace(strikes_arr.min(), strikes_arr.max(), 500)
    
    # Scipy Log-Norm pdf
    # s = sigma*sqrt(T), scale = exp(mu)
    theory_pdf = lognorm.pdf(x_points, s=sigma_std, scale=np.exp(mu_std))

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: The Volatility Smile
    axes[0].plot(strikes_arr, iv_arr, 'o-', markersize=4, label='Calculated IV')
    axes[0].axvline(S0, color='gray', linestyle='--', label='Spot Price')
    axes[0].set_title(f"Volatility Smile ({ticker})")
    axes[0].set_xlabel("Strike Price")
    axes[0].set_ylabel("Implied Volatility")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Call Prices & Smoothing
    # Verify the Spline fits the data well
    axes[1].plot(strikes_arr, prices_arr, 'o', alpha=0.5, label='Market Prices')
    axes[1].plot(strikes_arr, price_spline(strikes_arr), 'r-', label='Cubic Spline')
    axes[1].set_title("Price Smoothing (Crucial for 2nd Deriv)")
    axes[1].set_xlabel("Strike Price")
    axes[1].set_ylabel("Call Price")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: The PDF Comparison (The Result)
    # Normalize area for visualization comparison (approximate)
    axes[2].plot(strikes_arr, market_pdf, color='blue', lw=2, label='Market Implied (Breeden-Litzenberger)')
    axes[2].plot(x_points, theory_pdf, color='orange', linestyle='--', lw=2, label=f'Black-Scholes (LogNorm, IV={atm_iv:.0%})')
    
    axes[2].set_title("Future Price Probability Distributions")
    axes[2].set_xlabel("Stock Price at Expiry")
    axes[2].set_ylabel("Probability Density")
    axes[2].axvline(S0, color='gray', linestyle='--', alpha=0.5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()