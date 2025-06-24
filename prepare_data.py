import pandas as pd
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=14)  # extra buffer for weekends

# Load data
spy_df = pd.read_parquet(f"SPY_{start_date.date()}_{end_date.date()}.parquet")
gld_df = pd.read_parquet(f"GLD_{start_date.date()}_{end_date.date()}.parquet")
tlt_df = pd.read_parquet(f"TLT_{start_date.date()}_{end_date.date()}.parquet")

# Use 'timestamp' as datetime index
for df in [spy_df, gld_df, tlt_df]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

# Keep only close prices and rename columns
spy_close = spy_df[['close']].rename(columns={'close': 'spx'})
gld_close = gld_df[['close']].rename(columns={'close': 'gold'})
tlt_close = tlt_df[['close']].rename(columns={'close': 'bond'})

# Join all on timestamp
df_all = spy_close.join(gld_close, how='inner').join(tlt_close, how='inner')

# Create lag features
def add_lags(df, col, lags):
    for i in range(1, lags+1):
        df[f"{col}_t-{i}"] = df[col].shift(i)
    return df

df_all = add_lags(df_all, 'spx', 5)
df_all = add_lags(df_all, 'gold', 5)
df_all = add_lags(df_all, 'bond', 5)

df_all = df_all.dropna()
df_all = df_all.between_time('13:30', '20:00')

print(df_all.shape)
print(df_all)

# Optional: Save to disk
df_all.to_parquet(f"/Users/jingyuanhe/code/algotrading/data/dataset_{start_date.date()}.parquet")
