# G10 currency carry

This folder tests a currency carry rule, using monthly exchange rates and interest rates from 2006 to January 2026.

Each month, the strategy buys the three currencies with the highest interest rates against the US dollar, and sells the three with the lowest. Each side gets equal weight. It uses last month's interest rates, so it never trades on information it didn't have yet. Trading costs 2 basis points per unit traded.

`strategy.py` runs the test and downloads the exchange rates from Yahoo Finance. `fred_rates.py` downloads the interest rates from FRED and saves them to `rates_raw.csv`, which is what `strategy.py` reads. You only need to run `fred_rates.py` if you want newer rates, because it overwrites that file.

The data is split 70/30 by months. Over the full period, the net Sharpe ratio was 0.26, which is below the 0.3 line, so the full test fails. The drawdown was -27.4%, which passes the -35% limit, and none of the three sub-periods lost money. Out-of-sample on its own, the Sharpe ratio was 0.33 and the drawdown was -6.3%, so that part passes. In-sample on its own, the Sharpe ratio was 0.25 and fails.

The pass rules are at the top of `strategy.py`: a net Sharpe of at least 0.3, fewer than two losing sub-periods, and a max drawdown better than -35%. The carry Sharpe doesn't take off a cash rate, because the returns are already interest differences between currencies. The annual return is the average monthly return times 12, not compounded growth.

To run it, install the packages from the top-level README and run `strategy.py`.
