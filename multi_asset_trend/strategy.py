# Multi-asset trend on 8 ETFs, monthly
# The pass rules below were written on 2026-11-20, after I had seen the final equity value of the full period but before computing any stats or benchmarks, so this run is partly exploratory. The rules were not changed after seeing results.
#
# Prediction: lower return than SPY, smaller drawdowns, better Sharpe than the single-asset SMA, weak in whipsaw years like 2022.
#
# Setup: 8 ETFs (SPY, EFA, EEM, TLT, IEF, GLD, DBC, VNQ), held when month-end close > 10-month SMA, else cash; equal weight, monthly; signal shifted 1 month.
# Costs 0.1% per side; cash earns the actual monthly 3-month T-bill rate (FRED DTB3, monthly average, lagged one month like the G10 project, saved by fred_rates.py).
# The first version used a flat 0.1427%/month (the average over the period); changed on 2026-11-23, after the first run, to match G10. The pass rules were not changed.
# Window 2006-12 to 2026-10. Annualised return = CAGR.
# Sharpe = mean(monthly return - cash rate) / std * sqrt(12).
#
# Split: 70/30 by months. In-sample 2006-12 to 2020-10 (167 months), out-of-sample 2020-11 to 2026-10 (72 months).
# Nothing is fitted, so the split is a reporting discipline, not a safeguard.
#
# PASS requires ALL of:
# 1. Sharpe >= equal-weight buy-and-hold of the same 8 ETFs
# 2. Max drawdown no worse than 2/3 of SPY buy-and-hold's max drawdown
# 3. Fewer than 2 losing sub-periods (2006-12 to 2012, 2013-19, 2020-26)
# AND conditions 1 and 2 must also hold in the out-of-sample period alone.
#
# Known simplifications: prices are dividend-adjusted (yfinance default); data downloaded 2026-11-20 and end date pinned to 2026-10-31 for reproducibility; strategy pays no liquidation cost at the end while the benchmarks pay an exit cost (conservative for the strategy).


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
HERE = Path(__file__).parent

# Import tickers
etf = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
table = yf.download(etf, period = "max").resample("ME")['Close'].last()
# Trim data to be same start and end date to get the main price table
# End date pinned (not "drop the last row") so results don't change when rerun
table = table.loc[:"2026-10-31"]
sidetable = table.notna().all(axis = 1)
end_date = sidetable[sidetable].index[-1]
start_date = sidetable.idxmax()
main_table = table.loc[start_date : end_date]
# Create SMA table
sma_table = main_table.rolling(10).mean()
# Create percentage change table
pct_table = main_table.pct_change()
# Get the signal table (one/zero) and account for missing the first 10 months of SMA
one_zero_table = main_table > sma_table
one_zero_table = one_zero_table.astype(float)
one_zero_table = one_zero_table.where(sma_table.notna())
# Multiply percentage change table by the signal (one/zero) table to get returns for days in the market accounting for lookahead
one_zero_table = one_zero_table.shift(1)
returns_table = pct_table * one_zero_table
# Add in t-bill interest
tbill = pd.read_csv(HERE / "tbill_raw.csv", index_col = 0, parse_dates = True)["rate_annual_pct"]
monthly_tbill = (tbill / 100 / 12).shift(1)
monthly_tbill = monthly_tbill.reindex(main_table.index)
returns_table = returns_table.where(one_zero_table != 0, monthly_tbill, axis = 0)
# Account for costs
shifted_table = one_zero_table.shift(1).isna()
temp_table = one_zero_table == 1
combined_table = shifted_table & temp_table
combined_table = combined_table.astype(float)
trades_table = one_zero_table.diff().abs().fillna(combined_table)
cost_table = trades_table * 0.001
cost_adjusted_returns_table = returns_table - cost_table
# Average the eight ETFs into one portfolio and drop empty rows
final_portfolio = cost_adjusted_returns_table.mean(axis = 1)
final_portfolio = final_portfolio.dropna()
# Show the in-sample and out-of-sample for the main portfolio
in_sample_cut = int(len(final_portfolio) * 0.7)
in_sample_portfolio = final_portfolio.iloc[:in_sample_cut]
out_of_sample_portfolio = final_portfolio.iloc[in_sample_cut:]
# Benchmarks: no signal, no cash. SPY and the equal-weight basket (monthly rebalanced, no rebalancing cost) each pay 0.1% entry (first month) and 0.1% exit (last month), a flat 0.001 off the return. In sample and out of sample cuts included
spy_benchmark_table = pct_table["SPY"]
spy_benchmark_table = spy_benchmark_table.loc[final_portfolio.index[0] : end_date].copy()
spy_benchmark_table.iloc[0] = spy_benchmark_table.iloc[0] - 0.001
spy_benchmark_table.iloc[-1] = spy_benchmark_table.iloc[-1] - 0.001
equal_weight_benchmark_table = pct_table.mean(axis = 1)
equal_weight_benchmark_table = equal_weight_benchmark_table.loc[final_portfolio.index[0] : end_date].copy()
equal_weight_benchmark_table.iloc[[0, -1]] -= 0.001
in_sample_spy = spy_benchmark_table.iloc[:in_sample_cut]
out_of_sample_spy = spy_benchmark_table.iloc[in_sample_cut:]
in_sample_equal_weight = equal_weight_benchmark_table.iloc[:in_sample_cut]
out_of_sample_equal_weight = equal_weight_benchmark_table.iloc[in_sample_cut:]
sub_period_1 = final_portfolio.loc["2006-12" : "2012-12"]
sub_period_2 = final_portfolio.loc["2013-01" : "2019-12"]
sub_period_3 = final_portfolio.loc["2020-01" : "2026-10"]
# Compute all the statistics
def stats(r):
    equity_curve = (1 + r).cumprod()
    total_return = (equity_curve.iloc[-1] - 1) * 100
    cagr = ((equity_curve.iloc[-1] ** (1/(len(r)/12))) - 1) * 100
    annualised_volatility = (r.std() * np.sqrt(12)) * 100
    excess = r - monthly_tbill
    sharpe_ratio = (excess.mean() / excess.std()) * np.sqrt(12)
    running_peaks = equity_curve.cummax()
    dd = ((equity_curve / running_peaks) - 1) * 100
    max_dd = dd.min()
    return total_return, cagr, annualised_volatility, sharpe_ratio, max_dd

series_dict = {
    "Strategy, full": final_portfolio,
    "Strategy, in-sample": in_sample_portfolio,
    "Strategy, out-of-sample": out_of_sample_portfolio,
    "SPY, full": spy_benchmark_table,
    "SPY, in-sample": in_sample_spy,
    "SPY, out-of-sample": out_of_sample_spy,
    "Equal-weight, full": equal_weight_benchmark_table,
    "Equal-weight, in-sample": in_sample_equal_weight,
    "Equal-weight, out-of-sample": out_of_sample_equal_weight,
    "Strategy 2006-12 to 2012-12": sub_period_1,
    "Strategy 2013-01 to 2019-12": sub_period_2,
    "Strategy 2020-01 to 2026-10": sub_period_3,
}

results = pd.DataFrame(index=["Total return %", "Ann. return %", "Ann. vol %", "Sharpe", "Max drawdown %"])
for label, r in series_dict.items():
    results[label] = stats(r)
results = results.T.round(2)

print(results.to_string())

# Pass rules, as written in the header at the top
strat = stats(final_portfolio)
ew = stats(equal_weight_benchmark_table)
spy = stats(spy_benchmark_table)
losing = 0
for r in [sub_period_1, sub_period_2, sub_period_3]:
    if stats(r)[0] < 0:
        losing = losing + 1
strat_o = stats(out_of_sample_portfolio)
ew_o = stats(out_of_sample_equal_weight)
spy_o = stats(out_of_sample_spy)

print("\nFull sample")
if strat[3] >= ew[3]:
    print("1. Sharpe at least equal-weight buy-and-hold's: PASS", round(strat[3], 2), "vs", round(ew[3], 2))
else:
    print("1. Sharpe at least equal-weight buy-and-hold's: FAIL", round(strat[3], 2), "vs", round(ew[3], 2))
if abs(strat[4]) <= abs(spy[4]) * (2 / 3):
    print("2. Max drawdown no worse than 2/3 of SPY's: PASS", round(strat[4], 1), "vs SPY", round(spy[4], 1))
else:
    print("2. Max drawdown no worse than 2/3 of SPY's: FAIL", round(strat[4], 1), "vs SPY", round(spy[4], 1))
if losing < 2:
    print("3. Fewer than 2 losing sub-periods: PASS", losing, "losing")
else:
    print("3. Fewer than 2 losing sub-periods: FAIL", losing, "losing")

print("\nOut-of-sample (rules 1 and 2 only)")
if strat_o[3] >= ew_o[3]:
    print("1. Sharpe at least equal-weight buy-and-hold's: PASS", round(strat_o[3], 2), "vs", round(ew_o[3], 2))
else:
    print("1. Sharpe at least equal-weight buy-and-hold's: FAIL", round(strat_o[3], 2), "vs", round(ew_o[3], 2))
if abs(strat_o[4]) <= abs(spy_o[4]) * (2 / 3):
    print("2. Max drawdown no worse than 2/3 of SPY's: PASS", round(strat_o[4], 1), "vs SPY", round(spy_o[4], 1))
else:
    print("2. Max drawdown no worse than 2/3 of SPY's: FAIL", round(strat_o[4], 1), "vs SPY", round(spy_o[4], 1))

if strat[3] >= ew[3] and abs(strat[4]) <= abs(spy[4]) * (2 / 3) and losing < 2 and strat_o[3] >= ew_o[3] and abs(strat_o[4]) <= abs(spy_o[4]) * (2 / 3):
    print("\nOVERALL: PASS")
else:
    print("\nOVERALL: FAIL")

# Chart
strategy_equity = (1 + final_portfolio).cumprod()
spy_equity = (1 + spy_benchmark_table).cumprod()
ew_equity = (1 + equal_weight_benchmark_table).cumprod()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
ax1.plot(strategy_equity, label="Multi-asset trend (net)")
ax1.plot(spy_equity, label="SPY buy-and-hold")
ax1.plot(ew_equity, label="Equal-weight 8 ETFs, held")
ax1.axvline(out_of_sample_portfolio.index[0], color="grey", linestyle="--")
ax1.set_yscale("log")
ax1.set_title("Multi-asset trend vs buy-and-hold: growth of 1 (dashed line = start of out-of-sample)")
ax1.legend()
for curve in [strategy_equity, spy_equity, ew_equity]:
    ax2.plot((curve / curve.cummax() - 1) * 100)
ax2.set_title("Drawdown (%)")
plt.show()
