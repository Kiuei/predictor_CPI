import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Get historical stock data (Example: Apple)
ticker = "AAPL"
stock_data = yf.download(ticker, start="2015-01-01", end="2023-01-01", auto_adjust=False)

# Calculate daily log returns
stock_data["Log Returns"] = np.log(stock_data["Adj Close"] / stock_data["Adj Close"].shift(1))
log_returns = stock_data["Log Returns"].dropna()

# Monte Carlo Simulation
simulations = 10000
days = 365*2  # Predicting x days ahead
initial_price = stock_data["Adj Close"].iloc[-1]  # Last known price

# Get mean and standard deviation of log returns
mu = log_returns.mean()  # Keep it in daily scale
sigma = log_returns.std()
drift = mu - (0.5 * sigma**2)

# Simulate price paths
simulated_prices = np.zeros((days, simulations))
simulated_prices[0] = initial_price

for t in range(1, days):
    random_shocks = np.random.normal(mu, sigma, simulations)  # Random changes
    simulated_prices[t] = simulated_prices[t-1] * np.exp(random_shocks)

# Monte Carlo Simulation Plot
plt.figure(figsize=(10, 5))

# Plot all simulated paths (blue, transparent)
plt.plot(simulated_prices, alpha=0.1, color='blue')

# Compute and plot the average simulation path (bold red line)
average_simulation = simulated_prices.mean(axis=1)
plt.plot(average_simulation, color='red', linewidth=2, label="Average Path")

# Add labels to the red line at key points
for day in range(0, days, max(1, days // 5)):  # Label every ~5th interval
    plt.text(day, average_simulation[day], f"{average_simulation[day]:.2f}",
             fontsize=10, color="black", ha="center", bbox=dict(facecolor="white", alpha=0.6))
# Always label the last day
plt.text(days - 1, average_simulation[-1], f"{average_simulation[-1]:.2f}",
         fontsize=10, color="black", ha="center", bbox=dict(facecolor="white", alpha=0.6))


# Labels & Legend
plt.title(f"Monte Carlo Simulation of {ticker} Stock Price (X Days)")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

