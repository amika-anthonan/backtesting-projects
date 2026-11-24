# 200-day SMA on SPY, daily
# Hold SPY when the close is above its 200-day SMA, otherwise cash. The signal uses yesterday's close and the trade happens at today's close. Costs 0.1% per side. SPY from Yahoo (yfinance), 1993-01-29 to 2026-11-20.
# Cash earns the 3-month T-bill rate (FRED DTB3, monthly average, lagged one month; fred_rates.py saves it to tbill_raw.csv).
# Split: first 70% of days in-sample, last 30% out-of-sample. Nothing is fitted, so the split is just for reporting.
# Pass rules: max drawdown no more than 2/3 of buy-and-hold's, Sharpe at least buy-and-hold's.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
HERE = Path(__file__).parent

ticker_symbol = "SPY"
spy_ticker = yf.Ticker(ticker_symbol)
# End date is fixed so the results don't change between runs (end is exclusive, so this includes 20 Nov 2026)
historical_data = spy_ticker.history(period="max", end="2026-11-21")
spy_daily_close = historical_data[["Close"]].copy()

# The SMA uses the full history, so the held-out period starts with a proper average behind it
spy_daily_close["SMA_200"] = spy_daily_close["Close"].rolling(window=200).mean()

# Cash earns the actual 3-month T-bill rate (monthly average from fred_rates.py), shifted one month like the other projects
# The first version used a flat 4% a year; changed on 2026-11-23, after the first run
tbill = pd.read_csv(HERE / "tbill_raw.csv", index_col = 0, parse_dates = True)["rate_annual_pct"]
cash_by_month = (tbill / 100 / 252).shift(1)
cash_by_month.index = cash_by_month.index.to_period("M")
months = spy_daily_close.index.tz_localize(None).to_period("M")
spy_daily_close["cash_daily"] = cash_by_month.reindex(months).values

# First 70% is in-sample; the last 30% is held out and tested separately below
cut = int(len(spy_daily_close) * 0.7)
in_sample = spy_daily_close.iloc[:cut].copy()
held_out = spy_daily_close.iloc[cut:].copy()

cost = 0.001

def run_strategy(period, label):
    days_in_market = 0
    holdingposition = False
    previous_close = None
    previous_sma = None
    double_previous_close = None
    double_previous_sma = None
    trades_list = []
    daily_return_list = []
    cash_list = []

    for date, row in period.iterrows():
        current_close = row["Close"]
        current_sma = row["SMA_200"]
        cash_daily = row["cash_daily"]

        # Need two days of SMA before we can check for a crossover
        if pd.isna(previous_sma) or pd.isna(double_previous_sma):
            double_previous_close = previous_close
            double_previous_sma = previous_sma
            previous_close = current_close
            previous_sma = current_sma
            continue

        if holdingposition == False and previous_close > previous_sma and double_previous_close <= double_previous_sma:
            # Buy at today's close, so today only earns cash
            templist = []
            holdingposition = True
            cost_adjusted_purchase = current_close * (1 + cost)
            templist.append(cost_adjusted_purchase)
            buy_day_multiplier = (1 + cash_daily) * (1 - cost)
            daily_return_list.append(buy_day_multiplier)
            cash_list.append(cash_daily)
            double_previous_close = previous_close
            double_previous_sma = previous_sma
            previous_close = current_close
            previous_sma = current_sma
            continue

        if holdingposition == True and previous_close < previous_sma and double_previous_close >= double_previous_sma:
            # Sell at today's close, so today earns the market return up to the close
            holdingposition = False
            cost_adjusted_sale = current_close * (1 - cost)
            templist.append(cost_adjusted_sale)
            trades_list.append(templist)
            sell_day_multiplier = (current_close / previous_close) * (1 - cost)
            daily_return_list.append(sell_day_multiplier)
            cash_list.append(cash_daily)
            days_in_market += 1
            double_previous_close = previous_close
            double_previous_sma = previous_sma
            previous_close = current_close
            previous_sma = current_sma
            continue

        if holdingposition == True:
            spy_multiplier = current_close / previous_close
            daily_return_list.append(spy_multiplier)
            cash_list.append(cash_daily)
            days_in_market += 1
        else:
            daily_return_list.append(1 + cash_daily)
            cash_list.append(cash_daily)

        double_previous_close = previous_close
        double_previous_sma = previous_sma
        previous_close = current_close
        previous_sma = current_sma

    # Close any open position at the end of the period
    if holdingposition == True:
        cost_adjusted_sale = current_close * (1 - cost)
        templist.append(cost_adjusted_sale)
        trades_list.append(templist)

    number_of_trades = len(trades_list)

    benchmark_table = period.dropna()

    multipliers_arrays = np.array(daily_return_list)
    strategy_equity_curve = np.cumprod(np.insert(multipliers_arrays, 0, 1.0))
    strategy_peaks = np.maximum.accumulate(strategy_equity_curve)
    strategy_drawdowns = (strategy_equity_curve / strategy_peaks) * 100 - 100
    sma_max_drawdown = np.min(strategy_drawdowns)

    # Buy and hold benchmark over the same period
    spy_daily_pct = benchmark_table["Close"].pct_change().dropna().values[1:]
    spy_multipliers = 1 + spy_daily_pct
    spy_multipliers[0] = spy_multipliers[0] * (1 - cost)
    spy_multipliers[-1] = spy_multipliers[-1] * (1 - cost)
    benchmark_equity_curve = np.cumprod(np.insert(spy_multipliers, 0, 1.0))

    benchmark_peaks = np.maximum.accumulate(benchmark_equity_curve)
    benchmark_drawdowns = (benchmark_equity_curve / benchmark_peaks) * 100 - 100
    benchmark_max_dd = np.min(benchmark_drawdowns)

    strategy_excess_returns = (np.array(daily_return_list) - 1) - np.array(cash_list)
    strategy_sharpe = (np.mean(strategy_excess_returns) / np.std(strategy_excess_returns)) * np.sqrt(252)
    benchmark_excess_returns = spy_daily_pct - benchmark_table["cash_daily"].values[2:]
    benchmark_sharpe = (np.mean(benchmark_excess_returns) / np.std(benchmark_excess_returns)) * np.sqrt(252)

    years = len(daily_return_list) / 252
    strategy_total = (strategy_equity_curve[-1] - 1) * 100
    benchmark_total = (benchmark_equity_curve[-1] - 1) * 100
    strategy_cagr = (strategy_equity_curve[-1] ** (1 / years) - 1) * 100
    benchmark_cagr = (benchmark_equity_curve[-1] ** (1 / years) - 1) * 100
    strategy_vol = np.std(multipliers_arrays - 1) * np.sqrt(252) * 100
    benchmark_vol = np.std(spy_daily_pct) * np.sqrt(252) * 100
    time_in_market = days_in_market / len(daily_return_list) * 100

    print(label)
    print("Dates:", period.index[0].date(), "to", period.index[-1].date())
    print("Strategy vs buy and hold")
    print("Total return %:", round(strategy_total, 1), "vs", round(benchmark_total, 1))
    print("Annualised return %:", round(strategy_cagr, 2), "vs", round(benchmark_cagr, 2))
    print("Annualised volatility %:", round(strategy_vol, 2), "vs", round(benchmark_vol, 2))
    print("Sharpe ratio:", round(strategy_sharpe, 2), "vs", round(benchmark_sharpe, 2))
    print("Max drawdown %:", round(sma_max_drawdown, 1), "vs", round(benchmark_max_dd, 1))
    print("Trades:", number_of_trades, "| Time in market %:", round(time_in_market, 1))

    # Pass rules: drawdown no more than 2/3 of buy-and-hold's, Sharpe at least buy-and-hold's
    dd_pass = abs(sma_max_drawdown) <= abs(benchmark_max_dd) * (2 / 3)
    sharpe_pass = strategy_sharpe >= benchmark_sharpe
    print("Drawdown test passed:", dd_pass, "| Sharpe test passed:", sharpe_pass)
    print()

    plt.figure(figsize=(10, 5))
    plt.plot(strategy_equity_curve, label="200 SMA strategy")
    plt.plot(benchmark_equity_curve, label="Buy and hold")
    plt.yscale("log")
    plt.legend()
    plt.title(label + " equity curves (log scale)")
    plt.show()


run_strategy(in_sample, "In-sample (first 70%)")
run_strategy(held_out, "Out-of-sample (last 30%)")
