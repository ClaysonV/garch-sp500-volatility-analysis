import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

def run_garch_model():
    """
    This function demonstrates a complete workflow for building a GARCH(1,1) model:
    1. Fetches S&P 500 data.
    2. Prepares the data by calculating returns.
    3. Visualizes the returns to show volatility clustering.
    4. Fits a GARCH(1,1) model.
    5. Prints and interprets the model summary.
    6. Plots the resulting conditional volatility.
    7. Forecasts future volatility.
    """
    
    # --- 1. Fetch Data ---
    print("Fetching S&P 500 (^GSPC) data...")
    # Download data from 2010 to the present
    # The yfinance library now defaults to auto_adjust=True, which is great.
    data = yf.download('^GSPC', start='2010-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    
    if data.empty:
        print("Error: No data fetched. Check ticker symbol or network connection.")
        return

    # --- 2. Prepare Data ---
    # GARCH models work on returns, not prices.
    # We calculate percentage returns. Multiplying by 100 is common
    # as it makes the variance numbers in the summary easier to read.
    
    # *** THIS IS THE CORRECTED LINE ***
    # We use 'Close' as yfinance auto_adjust=True now provides the adjusted price in this column.
    returns = data['Close'].pct_change() * 100
    
    # Drop the first NaN value that results from pct_change()
    returns = returns.dropna()
    
    print("Data preparation complete.")

    # --- 3. Visualize Returns (Volatility Clustering) ---
    print("Plotting daily returns...")
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, returns, label='S&P 500 Daily Returns (%)', color='blue', alpha=0.7, linewidth=0.5)
    plt.title('S&P 500 Daily Returns')
    plt.ylabel('Percentage Return (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.figtext(0.5, 0.01, 'Note: Observe the "volatility clustering."\n'
                            'Periods of high volatility (like 2011, 2020) are clumped together.',
                ha='center', fontsize=10, style='italic')
    plt.show()

    # --- 4. Specify and Fit the GARCH Model ---
    print("\nFitting GARCH(1,1) model...")
    # GARCH(1,1) is the most common model.
    # p=1: The 'ARCH' term (yesterday's shock).
    # q=1: The 'GARCH' term (yesterday's volatility).
    # vol='Garch': Specifies the GARCH model type.
    # We assume a normal distribution ('Normal') for the errors for simplicity.
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
    
    # Fit the model. disp='off' hides the fitting iterations.
    results = model.fit(disp='off')

    # --- 5. Print and Interpret the Results ---
    print("\n--- GARCH(1,1) Model Results ---")
    print(results.summary())

    print("\n--- How to Interpret the Summary ---")
    print(f"Mean Model (Constant): {results.params['mu']:.4f}")
    print("  - This is the average daily return. It's usually very close to 0.")
    
    print("\nVolatility Model Coefficients:")
    print(f"  omega (constant variance): {results.params['omega']:.4f}")
    print("    - The baseline, long-run variance. Must be positive.")
    
    print(f"  alpha[1] (ARCH term): {results.params['alpha[1]']:.4f}")
    print("    - Measures reaction to shocks (yesterday's squared error).")
    print("    - A high alpha means volatility is 'spiky' and reacts quickly to news.")
    
    print(f"  beta[1] (GARCH term): {results.params['beta[1]']:.4f}")
    print("    - Measures persistence of volatility (yesterday's variance).")
    print("    - A high beta (e.g., > 0.9) means volatility is 'persistent' and shocks take a long time to die down.")
    
    # Check for persistence
    persistence = results.params['alpha[1]'] + results.params['beta[1]']
    print(f"\nVolatility Persistence (alpha + beta): {persistence:.4f}")
    print("  - If this is close to 1.0, it confirms that volatility is highly persistent (a key feature of financial data).")
    print("  - If it is > 1.0, the model is 'non-stationary' and may be unstable.")

    print("\nP-values (P>|z|):")
    print("  - Look at the p-values for alpha[1] and beta[1]. If they are < 0.05, the coefficients are statistically significant.")


    # --- 6. Plot the Conditional Volatility ---
    print("\nPlotting the conditional volatility from the model...")
    # The 'arch' library has a built-in plot function
    fig = results.plot(annualize=None) # We plot daily volatility, not annualized
    fig.set_size_inches(12, 8)
    plt.suptitle('GARCH(1,1) Model - Conditional Volatility and Standardized Residuals', y=1.02, fontsize=16)
    
    # Customize the top plot (conditional volatility)
    ax0 = fig.get_axes()[0]
    ax0.set_title('Conditional Volatility (Daily %)')
    ax0.set_ylabel('Volatility (%)')
    ax0.plot(results.conditional_volatility, label='Cond. Volatility', color='red', linewidth=1.5)
    ax0.legend()
    
    # Customize the bottom plot (standardized residuals)
    ax1 = fig.get_axes()[1]
    ax1.set_title('Standardized Residuals')

    plt.tight_layout()
    plt.show()

    # --- 7. Forecast Future Volatility ---
    print("\nForecasting future volatility...")
    # Forecast the variance for the next 5 days
    forecast_horizon = 5
    forecast = results.forecast(horizon=forecast_horizon)
    
    # The forecast object contains 'mean', 'variance', and 'residual_variance'
    # We want the variance forecast from the last day of our data
    variance_forecast = forecast.variance.iloc[-1]
    
    # Volatility is the square root of variance
    volatility_forecast = np.sqrt(variance_forecast)
    
    print(f"\n--- {forecast_horizon}-Day Volatility Forecast (Daily %) ---")
    print("This is the model's prediction for the standard deviation of returns for the next 5 days.")
    for i in range(forecast_horizon):
        print(f"  Day T+{i+1}: {volatility_forecast[f'h.{i+1}']:.3f}%")
        
    print("\nNote: This forecasts the *magnitude* of price moves (volatility), not the *direction* (price).")


if __name__ == "__main__":
    run_garch_model()