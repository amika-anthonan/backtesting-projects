# Backtesting projects

These are three simple backtests I wrote in Python to see how some basic trading rules would have done on past data. Each project folder has a strategy file, a script that downloads the interest rates, and the rates saved as a CSV.

The first one holds the SPY index when its price is above its 200-day average, and holds cash otherwise. The second is a currency carry test. Each month it buys the three currencies with the highest interest rates against the US dollar and sells the three with the lowest. The third holds eight ETFs when each one is above its 10-month average, and cash otherwise.

I split each test 70/30. The first 70% of the data is in-sample, and the last 30% is out-of-sample. The SMA passed both rules in-sample and failed both out-of-sample. The carry test failed the full-period Sharpe rule at 0.26, but passed out-of-sample on its own at 0.33. The multi-asset test passed everything on the full period and out-of-sample.

The pass rules are in each project folder, with a note on when each one was set. For the SMA and multi-asset tests I changed the cash rate after the first run, so those folders also keep the first run's numbers.

These are tests of one rule each on past data. They don't show that any of them would make money.

To run one, install the packages with `pip install yfinance pandas numpy matplotlib`, open the project folder and run `strategy.py`. The rates are already saved, so you don't need to run `fred_rates.py` unless you want newer data. Prices come from Yahoo Finance through yfinance. The end dates are fixed in the code, so the results stay the same each time.

The results depend on the rules I picked, and small changes to them change the answer. Costs are simple: 0.1% per trade for SPY and the ETFs, and 2 basis points per unit traded for the currencies. The currency rate data ends in January 2026, so the carry test stops there.
