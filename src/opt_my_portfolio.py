import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# Read list of stocks available in Revolut
revolut_stocks = pd.read_csv('../data/revolut_stocks.csv')
revolut_symbols = list(revolut_stocks.iloc[:, 2])

# for dev only
revolut_symbols = revolut_symbols[: 5]

# Fetch historical data for these stocks
historical_data_path = '../data/historical_data.pkl'
if os.path.isfile(historical_data_path):
    historical_data = pd.read_pickle(historical_data_path)
else:
    historical_data = yf.download(' '.join(revolut_symbols), '2020-01-01')
    historical_data.to_pickle(historical_data_path)

# Load transactions
transactions = pd.read_csv('../data/transactions - Sheet1.csv')
no_of_stocks = transactions.groupby('symbol')['no_of_stocks'].sum()
portfolio_historical_data = historical_data['Close'][transactions['symbol']]

# Compute weights
current_prices = portfolio_historical_data.iloc[-1, :]
current_value = current_prices * no_of_stocks
weights = np.array(current_value / current_value.sum())

# Annualized return
start_value = portfolio_historical_data.iloc[0, :] * no_of_stocks
total_return = (current_value.sum() - start_value.sum()) / start_value.sum()
all_dates = portfolio_historical_data.index
no_of_days = (all_dates[-1] - all_dates[0]).days
annualized_return = ((1 + total_return) ** (365 / no_of_days)) - 1
print('annualized_return =\t{}' .format(annualized_return))

# Portfolio annualized standard deviation
daily_returns = portfolio_historical_data.pct_change()
cov_matrix = daily_returns.cov()
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
no_of_datapoints_per_year = 250
annualized_std = np.sqrt(portfolio_variance) * np.sqrt(no_of_datapoints_per_year)
print('annualized_std =\t{}'.format(annualized_std))

# Risk free rate
risk_free_rate = 0.02

# Sharpe ratio
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
print('sharpe_ratio =\t{}'.format(sharpe_ratio))

# Test investments in new stocks to maximize Sharpe ratio
invest_money = 30  # in USD
results = []  # each element is (sharpe_ratio, annualized_return, annualized_std, symbol)
for symbol in historical_data['Close'].columns:
    print((1, 1, 1, symbol))

results_sorted = sorted(results, reverse=True)
for res in results_sorted[:5]:
    print(res)
