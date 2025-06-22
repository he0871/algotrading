
import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def bollinger_signal(bars: pd.DataFrame, window: int = 20, num_std: float = 2.0):
    df = bars.copy()
    df["sma"] = df["close"].rolling(window).mean()
    df["std"] = df["close"].rolling(window).std()
    df["upper"] = df["sma"] + num_std * df["std"]
    df["lower"] = df["sma"] - num_std * df["std"]

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Check cross above upper band (Sell signal)
    if previous["close"] <= previous["upper"] and latest["close"] > latest["upper"]:
        return "sell"

    # Check cross below lower band (Buy signal)
    elif previous["close"] >= previous["lower"] and latest["close"] < latest["lower"]:
        return "buy"

    return "hold"

def run_bollinger_trade(client, trading_client, symbol="TSLA", minutes=50, qty=1, end=None):
    # Step 1: Get minute-level history
    bars = get_minute_history(client, symbol, minutes=minutes, end=end)

    # Step 2: Generate signal
    signal = bollinger_signal(bars)
    print(f"[{symbol}] Bollinger Signal: {signal}")

    # Step 3: Check if already holding position
    positions = trading_client.get_all_positions()
    held_qty = 0
    for p in positions:
        if p.symbol == symbol:
            held_qty = int(p.qty)

    # Step 4: Submit order based on signal
    if signal == "buy" and held_qty == 0:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"✅ BUY order placed for {symbol} ({qty} shares)")

    elif signal == "sell" and held_qty > 0:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=held_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"✅ SELL order placed for {symbol} ({held_qty} shares)")

    else:
        print(f"⏸️ No action taken for {symbol}")

if __name__ == "__main__":

    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_API_SECRET")

    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

    run_bollinger_trade(data_client, trade_client, symbol="TSLA", qty=1)
