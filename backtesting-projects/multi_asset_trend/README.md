# Multi-asset trend

This folder tests a trend rule on eight ETFs: SPY, EFA, EEM, TLT, IEF, GLD, DBC and VNQ. The data runs from 2006 to October 2026.

Each month, an ETF is held if its month-end close is above its 10-month average. If it isn't, that part of the portfolio sits in cash. The eight parts are equal weight. Trades cost 0.1% per side. Cash earns the 3-month T-bill rate, lagged one month, from `tbill_raw.csv`.

`strategy.py` runs the test. `fred_rates.py` downloads the T-bill rate from FRED and saves it to `tbill_raw.csv`. You only need to run it if you want newer rates.

The full period passed all three rules. The strategy's Sharpe ratio was 0.64, against 0.47 for holding the same eight ETFs equally weighted. Its drawdown was -8.2%, against -50.8% for SPY. Out-of-sample it also passed both comparisons: a Sharpe of 0.70 against 0.55, and a drawdown of -5.1% against -23.9% for SPY. It returned much less than SPY over the same period, 186% against 674%. The aim was lower risk, not higher return.

The pass rules are at the top of `strategy.py`. I wrote them on 20 November 2026, after I had seen the final value of the full period, but before calculating any statistics or benchmarks. So this run is partly exploratory. I didn't change the rules after seeing the results. The first run used a flat cash rate of 0.1427% a month. I switched to the monthly T-bill rate afterwards, and both runs passed.

To run it, install the packages from the top-level README and run `strategy.py`. It downloads prices from Yahoo Finance. The end date is fixed in the code, so the results stay the same each time.
