# S&P 500 Volatility Analysis using GARCH(1,1)

This project demonstrates the use of a GARCH(1,1) model to analyze and forecast the volatility of the S&P 500 index. It includes fetching live financial data, data preparation, model fitting, and interpretation of the results.

This is a common technique used in quantitative finance for risk management and derivatives pricing.

## 📊 Key Features

* **Data Fetching:** Downloads historical S&P 500 (`^GSPC`) data from 2010 to the present using `yfinance`.
* **Data Preparation:** Calculates daily percentage returns, which are the inputs for the GARCH model.
* **Visualization:**
    * Plots the daily returns to visually identify **volatility clustering** (periods of high volatility followed by more high volatility).
    * Plots the **conditional volatility** estimated by the fitted GARCH model.
* **Modeling:** Fits a GARCH(1,1) model using the `arch` library.
* **Interpretation:** Prints a detailed summary of the model's coefficients (`omega`, `alpha`, `beta`) and explains their significance.
* **Forecasting:** Provides a 5-day forecast of future volatility.

## 📈 Sample Output

Running the script will produce two plots:

1.  **S&P 500 Daily Returns:** This plot shows the daily returns, where you can clearly see periods like the 2011 crisis and the 2020 COVID-19 crash, which exhibit high volatility.
2.  **GARCH(1,1) Model Results:** This two-panel plot shows:
    * The **Conditional Volatility** (in red) overlaid on the standardized residuals. This shows the model's estimate of daily volatility (standard deviation).
    * The **Standardized Residuals** themselves, which should ideally look like white noise.

The script will also print the model summary and a 5-day volatility forecast to the console.

## 🚀 How to Run

Follow these steps to run the analysis on your local machine.

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/garch-sp500-volatility-analysis.git](https://github.com/YOUR_USERNAME/garch-sp500-volatility-analysis.git)
cd garch-sp500-volatility-analysis


## Create Virtual Enviroments

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

## Install dependencies
pip install -r requirements.txt

## Run the Script 
python garch_analysis.py