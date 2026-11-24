import pandas as pd
from pathlib import Path
HERE = Path(__file__).parent

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

rates = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
rates.columns = ["rate_annual_pct"]

# Monthly average of the daily 3-month T-bill rate, saved so the strategy always uses the same data
monthly = rates["rate_annual_pct"].resample("ME").mean()
monthly.to_csv(HERE / "tbill_raw.csv")

print(monthly.tail())
