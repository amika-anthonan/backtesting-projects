# 200-day SMA on SPY

This folder tests a simple trend rule on the SPY index, using daily prices from 1993 to November 2026.

The rule is to hold SPY when its close is above its 200-day average, and hold cash otherwise. The signal uses yesterday's close, and the trade happens at today's close. Each trade costs 0.1%.

`strategy.py` runs the test and prints the results. `fred_rates.py` downloads the 3-month T-bill rate from FRED, and `tbill_raw.csv` holds the monthly rates it saved. The cash days earn that rate.

The data is split 70/30. In-sample, from 1993 to 2016, the strategy passed both rules. Its drawdown was -23.4% against -55.2% for buy and hold, and its Sharpe ratio was 0.49 against 0.41. Out-of-sample, from 2016 to 2026, it failed both. Its drawdown was -23.6% against -33.7%, and its Sharpe ratio was 0.44 against 0.68. The drawdown missed the limit of two-thirds of buy and hold's, which is about -22.5%, by about one point.

The pass rules are that the drawdown is no more than two-thirds of buy and hold's, and the Sharpe ratio is at least buy and hold's. The first run used a flat 4% cash rate, and on that run the out-of-sample drawdown passed. I changed the cash rate to the real T-bill rate after that run, and the result changed. The rules did not change.

To run it, install the packages from the top-level README and run `strategy.py`. It downloads SPY prices from Yahoo Finance. The end date is fixed in the code, so the results stay the same each time.          
