# G10 currency carry vs USD, monthly
# Each month: long the 3 currencies with the highest interest rate vs USD, short the 3 lowest (1/3 weight each), using last month's rates. AUD, JPY, EUR, GBP, NZD, CAD, CHF, SEK, NOK.
# FX from Yahoo (yfinance), 3-month rates from FRED (fred_rates.py saves them to rates_raw.csv).
# Window 2006-06 to 2026-01 (the rates data runs out after that). Costs 2bp per unit traded.
# Split: first 70% of months in-sample, last 30% out-of-sample. Nothing is fitted, so the split is just for reporting.
# Pass rules: net Sharpe at least 0.3, fewer than 2 losing sub-periods (2006-06 to 2012, 2013-19, 2020-26), max drawdown better than -35%.
# The 0.3 Sharpe line was set before the test was run (from my notes at the time); the other two limits are as written here.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
HERE = Path(__file__).parent

series = {}
currencies = {
    "AUD" : ("AUDUSD=X", False),
    "JPY" : ("USDJPY=X", True),
    "EUR" : ("EURUSD=X", False),
    "GBP" : ("GBPUSD=X", False),
    "NZD" : ("NZDUSD=X", False),
    "CAD" : ("USDCAD=X", True),
    "CHF" : ("USDCHF=X", True),
    "SEK" : ("USDSEK=X", True),
    "NOK" : ("USDNOK=X", True)
}

for name, (ticker, invert) in currencies.items():
    close = yf.Ticker(ticker).history(period="max")["Close"]
    close.index = close.index.tz_localize(None)
    if invert:
        close = 1 / close
    series[name] = close.resample("ME").last()

exchange_table = pd.concat(series, axis=1)
exchange_table = exchange_table[exchange_table.index < pd.Timestamp.today().normalize()]

interest_table = pd.read_csv(HERE / "rates_raw.csv", index_col = 0, parse_dates = True).to_period("M")
exchange_table = exchange_table.to_period("M")
# The window is fixed so the results don't change from run to run (rates data runs out in 2026 for some currencies)
start_month = "2006-05"
end_month = "2026-01"
interest_table = interest_table.loc[start_month:end_month]
exchange_table = exchange_table.loc[start_month:end_month]

monthly_returns_exchange = exchange_table.pct_change().iloc[1:]
diff = interest_table.sub(interest_table["USD"], axis=0)
monthly_returns_interest = diff / 100 /12
monthly_returns_interest = monthly_returns_interest.drop('USD', axis = 1)
monthly_returns_interest = monthly_returns_interest.shift(1).iloc[1:]

interest_rank_table = monthly_returns_interest.rank(axis=1, method = "first")
interest_rank_table_top = interest_rank_table >= 7
interest_rank_table_top = interest_rank_table_top.astype(float) / 3
interest_rank_table_bottom = interest_rank_table <= 3
interest_rank_table_bottom = interest_rank_table_bottom.astype(float) / -3
weights = interest_rank_table_top + interest_rank_table_bottom

combined_returns_table = monthly_returns_exchange + monthly_returns_interest
master_returns_table = combined_returns_table * weights

total_monthly_return = master_returns_table.sum(axis = 1)
baseline_monthly_return = combined_returns_table.mean(axis = 1)

# Costs: 2bp per unit traded
turnover = weights.diff().abs().fillna(weights.abs()).sum(axis=1)
net_monthly_return = total_monthly_return - turnover * 0.0002

# 70/30 split, same as the other projects
cut = int(len(net_monthly_return) * 0.7)
in_sample = net_monthly_return.iloc[:cut]
held_out = net_monthly_return.iloc[cut:]

# No cash rate taken off the Sharpe: long/short returns are already interest differentials vs USD, so there is no cash leg to subtract
def stats(r):
    equity = (1 + r).cumprod()
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    dd = equity / equity.cummax() - 1
    return {
        "Total return %": (equity.iloc[-1] - 1) * 100,
        "Ann. return %": ann_ret * 100,
        "Ann. vol %": ann_vol * 100,
        "Sharpe": ann_ret / ann_vol,
        "Max drawdown %": dd.min() * 100,
    }

periods = {
    "2006-06 to 2012-12": net_monthly_return.loc["2006-06":"2012-12"],
    "2013-01 to 2019-12": net_monthly_return.loc["2013-01":"2019-12"],
    "2020-01 to 2026-01": net_monthly_return.loc["2020-01":"2026-01"],
}

rows = {
    "Strategy, gross": stats(total_monthly_return),
    "Strategy, net": stats(net_monthly_return),
    "Baseline (long all 9)": stats(baseline_monthly_return),
    "Net in-sample (first 70%)": stats(in_sample),
    "Baseline in-sample": stats(baseline_monthly_return.iloc[:cut]),
    "Net out-of-sample (last 30%)": stats(held_out),
    "Baseline out-of-sample": stats(baseline_monthly_return.iloc[cut:]),
}
for label, r in periods.items():
    rows["Net " + label] = stats(r)

results = pd.DataFrame(rows).T.round(2)
print(results.to_string())
print("\nIn-sample:", in_sample.index[0], "to", in_sample.index[-1])
print("Out-of-sample:", held_out.index[0], "to", held_out.index[-1])
print("Avg monthly turnover:", round(turnover.mean(), 3))
print("In-sample turnover:", round(turnover.iloc[:cut].mean(), 3))
print("Out-of-sample turnover:", round(turnover.iloc[cut:].mean(), 3))

# Pass rules
# Net Sharpe at least 0.3, fewer than 2 losing sub-periods, max drawdown better than -35%
# The sub-periods rule is only used on the full sample
net = stats(net_monthly_return)
neg_periods = 0
for r in periods.values():
    if stats(r)["Ann. return %"] < 0:
        neg_periods = neg_periods + 1

print("\nFull sample")
if net["Sharpe"] >= 0.3:
    print("1. Net Sharpe at least 0.3: PASS", round(net["Sharpe"], 2))
else:
    print("1. Net Sharpe at least 0.3: FAIL", round(net["Sharpe"], 2))
if neg_periods < 2:
    print("2. Fewer than 2 losing sub-periods: PASS", neg_periods, "losing")
else:
    print("2. Fewer than 2 losing sub-periods: FAIL", neg_periods, "losing")
if net["Max drawdown %"] > -35:
    print("3. Max drawdown better than -35%: PASS", round(net["Max drawdown %"], 1))
else:
    print("3. Max drawdown better than -35%: FAIL", round(net["Max drawdown %"], 1))
if net["Sharpe"] >= 0.3 and neg_periods < 2 and net["Max drawdown %"] > -35:
    print("OVERALL: PASS")
else:
    print("OVERALL: FAIL")

ins = stats(in_sample)
print("\nIn-sample (first 70%)")
if ins["Sharpe"] >= 0.3:
    print("1. Net Sharpe at least 0.3: PASS", round(ins["Sharpe"], 2))
else:
    print("1. Net Sharpe at least 0.3: FAIL", round(ins["Sharpe"], 2))
if ins["Max drawdown %"] > -35:
    print("3. Max drawdown better than -35%: PASS", round(ins["Max drawdown %"], 1))
else:
    print("3. Max drawdown better than -35%: FAIL", round(ins["Max drawdown %"], 1))
if ins["Sharpe"] >= 0.3 and ins["Max drawdown %"] > -35:
    print("OVERALL: PASS")
else:
    print("OVERALL: FAIL")

oos = stats(held_out)
print("\nOut-of-sample (last 30%)")
if oos["Sharpe"] >= 0.3:
    print("1. Net Sharpe at least 0.3: PASS", round(oos["Sharpe"], 2))
else:
    print("1. Net Sharpe at least 0.3: FAIL", round(oos["Sharpe"], 2))
if oos["Max drawdown %"] > -35:
    print("3. Max drawdown better than -35%: PASS", round(oos["Max drawdown %"], 1))
else:
    print("3. Max drawdown better than -35%: FAIL", round(oos["Max drawdown %"], 1))
if oos["Sharpe"] >= 0.3 and oos["Max drawdown %"] > -35:
    print("OVERALL: PASS")
else:
    print("OVERALL: FAIL")

# Chart
dates = net_monthly_return.index.to_timestamp()
equity = (1 + net_monthly_return).cumprod()
baseline_equity = (1 + baseline_monthly_return).cumprod()
drawdown = (equity / equity.cummax() - 1) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
ax1.plot(dates, equity.values, label="Carry strategy (net)")
ax1.plot(dates, baseline_equity.values, label="Equal-weight all 9")
ax1.axvline(dates[cut], color="grey", linestyle="--")
ax1.set_title("G10 carry: growth of 1 (dashed line = start of out-of-sample)")
ax1.legend()
ax2.fill_between(dates, drawdown.values, 0, color="red", alpha=0.4)
ax2.set_title("Strategy drawdown (%)")
plt.show()
