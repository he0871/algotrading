from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd

# Setup Alpaca credentials
API_KEY = "<>"
API_SECRET = "<>"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Symbols: SPY (S&P 500 ETF), GLD (Gold ETF), TLT (Bond ETF)
symbols = ["SPY", "GLD", "TLT"]

# Get date range: last 10 trading days
end_date = datetime.now()
start_date = end_date - timedelta(days=14)  # extra buffer for weekends

# Download 1-minute data
def fetch_1min_data(symbol):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )
    bars = client.get_stock_bars(request_params).df
    return bars[bars.index.get_level_values("symbol") == symbol]

# Fetch and save to Parquet
if __name__ == '__main__':
    for symbol in symbols:
        df = fetch_1min_data(symbol)
        df.reset_index(inplace=True)
        df.to_parquet(f"/Users/jingyuanhe/code/algotrading/data/{symbol}_{start_date.date()}_{end_date.date()}.parquet")
        print(f"Saved {symbol} data with {len(df)} rows.")
