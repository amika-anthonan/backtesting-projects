import pandas as pd
from pathlib import Path
HERE = Path(__file__).parent

# 3-month interbank rate from FRED for each currency (USD uses the 3-month T-bill)
rate_codes = {
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "NOK": "IR3TIB01NOM156N",
    "SEK": "IR3TIB01SEM156N",
    "USD": "TB3MS",
}
# Backup if a download fails: overnight call-money rate (a different rate, so it gets printed if used)
backup_codes = {
    "AUD": "IRSTCI01AUM156N",
    "NZD": "IRSTCI01NZM156N",
    "CAD": "IRSTCI01CAM156N",
    "CHF": "IRSTCI01CHM156N",
    "EUR": "IRSTCI01EZM156N",
    "GBP": "IRSTCI01GBM156N",
    "JPY": "IRSTCI01JPM156N",
    "NOK": "IRSTCI01NOM156N",
    "SEK": "IRSTCI01SEM156N",
}

rates = {}
for currency, code in rate_codes.items():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + code
    try:
        table = pd.read_csv(url)
    except Exception:
        code = backup_codes[currency]
        print(currency, "main series failed, using backup", code)
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + code
        table = pd.read_csv(url)
    # Column names from FRED aren't always the same, so rename them
    table.columns = ["date", "value"]
    table["date"] = pd.to_datetime(table["date"])
    # Missing values come in as dots, so they become NaN
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    series = table.set_index("date")["value"]
    rates[currency] = series
    print(currency, code, series.first_valid_index().date(), "to", series.last_valid_index().date())

pd.concat(rates, axis=1).to_csv(HERE / "rates_raw.csv")
