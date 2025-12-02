import numpy as np
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline

# --- PART 1: BLACK-SCHOLES FUNDAMENTALS ---

def bs_call_price(S, K, T, r, sigma):
    """Calculate Black-Scholes Call Price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    """Calculate Vega (Derivative of Price w.r.t. Sigma)"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# --- PART 2: NEWTON-RAPHSON SOLVER ---

def implied_volatility(market_price, S, K, T, r, tol=1e-5, max_iter=100):
    """
    Solve for sigma using Newton-Raphson method.
    x_new = x_old - f(x) / f'(x)
    here: sigma_new = sigma_old - (BS_Price - Market_Price) / Vega
    """
    sigma = 0.5 # Initial guess (50% vol)
    
    for i in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        
        diff = market_price - price
        
        if abs(diff) < tol:
            return sigma
        
        # Avoid division by zero if vega is tiny
        if abs(vega) < 1e-8:
            break
            
        sigma = sigma + diff / vega
        
    return np.nan # If convergence fails

# --- PART 3: BREEDEN-LITZENBERGER (PDF EXTRACTION) ---

def get_breeden_litzenberger_pdf(strikes, prices, T, r, smooth_factor=0.5):
    """
    Derives the Risk-Neutral Probability Density Function (PDF)
    using the Breeden-Litzenberger formula:
    
    PDF(K) = e^(rT) * (d^2 C / dK^2)
    
    Args:
        strikes: Array of Strike Prices (X-axis)
        prices: Array of Call Option Prices (Y-axis)
        T: Time to expiry
        r: Risk-free rate
        smooth_factor: 's' parameter for spline. 
                       Higher = smoother curve (less noise), Lower = fits data tighter.
    """
    # 1. Fit a smooth curve (Spline) through the Market Prices
    # k=3 means Cubic Spline
    # s is the smoothing factor (crucial because market data is jagged)
    spline = UnivariateSpline(strikes, prices, k=3, s=smooth_factor)
    
    # 2. Take the Second Derivative of the Price Curve
    # .derivative(n=2) returns a new function representing the 2nd deriv
    pdf_func = spline.derivative(n=2)
    
    # 3. Calculate raw PDF values
    pdf_values = pdf_func(strikes)
    
    # 4. Apply the Breeden-Litzenberger constant: e^(rT)
    pdf_values = pdf_values * np.exp(r * T)
    
    # 5. Handle numerical noise (Negative probabilities are impossible)
    pdf_values = np.maximum(pdf_values, 0)
    
    return pdf_values, spline