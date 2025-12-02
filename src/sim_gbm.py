import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from get_data import get_market_data

# --- CONFIGURATION ---
N_SIMS = 5000       # Number of simulations to run
HESTON_KAPPA = 2.0  # Speed of mean reversion for Volatility
HESTON_THETA = 0.04 # Long-term average variance (0.04 = 20% Vol)
HESTON_XI = 0.3     # "Vol of Vol" (How much volatility jiggles)
HESTON_RHO = -0.7   # Correlation between Price and Vol (Negative = Crash Fear)

def find_atm_volatility(calls_df, spot_price):
    """
    Finds the Implied Volatility (IV) of the At-The-Money (ATM) Call option.
    """
    calls_df['abs_diff'] = abs(calls_df['strike'] - spot_price)
    atm_row = calls_df.sort_values('abs_diff').iloc[0]
    
    iv = atm_row['impliedVolatility']
    print(f"--- Calibration ---")
    print(f"ATM Strike: ${atm_row['strike']}")
    print(f"Market ATM IV: {iv:.2%}")
    return iv

def simulate_gbm_paths(S0, mu, sigma, T, n_sims, n_steps):
    """
    Simulates Geometric Brownian Motion (Constant Volatility).
    """
    dt = T / n_steps
    
    # Generate random Brownian Motion
    Z = np.random.normal(0, 1, (n_sims, n_steps))
    
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    daily_returns = np.exp(drift + diffusion)
    
    # Initialize paths
    price_paths = np.zeros((n_sims, n_steps + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    return np.linspace(0, T, n_steps + 1), price_paths

def simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_sims, n_steps):
    """
    Simulates Heston Stochastic Volatility Model.
    
    dS = r*S*dt + sqrt(v)*S*dW1  (Stock Process)
    dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2  (Variance Process)
    """
    dt = T / n_steps
    
    # 1. Generate Correlated Brownian Motions (dW1, dW2)
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]] # Correlation matrix
    
    # Shape: (n_steps, n_sims, 2)
    Z = np.random.multivariate_normal(mean, cov, (n_steps, n_sims))
    
    # 2. Initialize Arrays
    S = np.full(n_sims, S0)
    v = np.full(n_sims, v0) # v0 is variance (sigma^2)
    
    # To store history
    S_paths = np.zeros((n_steps + 1, n_sims))
    v_paths = np.zeros((n_steps + 1, n_sims))
    S_paths[0] = S
    v_paths[0] = v
    
    # 3. Time Loop (Euler-Maruyama Method)
    for t in range(n_steps):
        # Current values
        S_t = S
        v_t = v
        
        # Ensure variance is positive (Full Truncation Scheme)
        v_t_pos = np.maximum(v_t, 0)
        
        # Calculate Random Shocks
        Z_S = Z[t, :, 0] # Stock noise
        Z_v = Z[t, :, 1] # Vol noise
        
        # Update Variance (dv)
        # dv = Speed*(LongTerm - Current)*dt + VolOfVol*sqrt(Current)*Noise
        dv = kappa * (theta - v_t_pos) * dt + xi * np.sqrt(v_t_pos) * np.sqrt(dt) * Z_v
        
        # Update Stock (dS)
        # dS = drift*S*dt + sqrt(CurrentVol)*S*Noise
        dS = r * S_t * dt + np.sqrt(v_t_pos) * S_t * np.sqrt(dt) * Z_S
        
        # Update State
        v = v_t + dv
        S = S_t + dS
        
        # Store
        S_paths[t+1] = S
        v_paths[t+1] = v
        
    return np.linspace(0, T, n_steps + 1), S_paths.T, v_paths.T

def compare_simulations(time_axis, gbm_paths, heston_paths, S0, atm_iv):
    """
    Visualizes GBM vs Heston side-by-side.
    """
    plt.figure(figsize=(16, 8))
    
    # --- Plot 1: The Paths (Visual Chaos check) ---
    plt.subplot(2, 2, 1)
    plt.plot(time_axis, gbm_paths[:50].T, lw=1, alpha=0.5, color='tab:blue')
    plt.title("GBM Paths (Constant Volatility)")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_axis, heston_paths[:50].T, lw=1, alpha=0.5, color='tab:red')
    plt.title("Heston Paths (Stochastic Volatility)")
    plt.grid(True, alpha=0.3)
    
    # --- Plot 2: The Distributions (The "Fat Tail" check) ---
    plt.subplot(2, 1, 2)
    
    # Get final prices
    gbm_final = gbm_paths[:, -1]
    heston_final = heston_paths[:, -1]
    
    sns.kdeplot(gbm_final, fill=True, color='tab:blue', label=f"GBM (Log-Normal, IV={atm_iv:.1%})", alpha=0.3)
    sns.kdeplot(heston_final, fill=True, color='tab:red', label="Heston (Stochastic Vol)", alpha=0.3)
    
    plt.axvline(S0, color='black', linestyle='--', label="Start Price")
    plt.title(f"Distribution Comparison at Expiry (Correlation rho={HESTON_RHO})")
    plt.xlabel("Price at Expiry")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ticker = "SPY"
    try:
        # 1. Get Data
        data = get_market_data(ticker)
        S0 = data['spot_price']
        r = data['risk_free_rate']
        T = data['T']
        
        # 2. Get Starting Params
        atm_iv = find_atm_volatility(data['calls'], S0)
        
        # 3. Simulate GBM (Constant Vol)
        days = int(T * 365)
        print(f"\nSimulating {days} steps for {N_SIMS} paths...")
        t_axis, gbm_paths = simulate_gbm_paths(S0, r, atm_iv, T, N_SIMS, days)
        
        # 4. Simulate Heston (Stochastic Vol)
        # v0 = Initial Variance (IV^2)
        v0 = atm_iv ** 2 
        # Note: We use the globals defined at top of script for Kappa, Theta, Xi
        t_axis, heston_paths, vol_paths = simulate_heston_paths(
            S0, v0, r, HESTON_KAPPA, HESTON_THETA, HESTON_XI, HESTON_RHO, T, N_SIMS, days
        )
        
        # 5. Compare
        compare_simulations(t_axis, gbm_paths, heston_paths, S0, atm_iv)

    except Exception as e:
        print(f"Error: {e}")