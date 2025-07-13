import os
import time
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from collections import defaultdict, deque
import logging

# --- LOGGER SETUP --- #
LOG_FILE = "trading_bot.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION --- #
ALPACA_API_KEY = os.environ.get("APCA_API_KEY_ID", "YOUR_KEY")
ALPACA_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "YOUR_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL = "AAPL"
TIMEFRAME = "5Min"
WINDOW = 50  # lookback window for indicators

# Q-Learning Hyperparameters
ACTIONS = ['buy', 'sell', 'hold']
ALPHA = 0.1  # learning rate
GAMMA = 0.95  # discount factor
EPSILON = 0.1  # exploration rate

# Hallucination parameters
HALLUCINATION_STEPS = 10  # Number of simulated steps per real tick
HALLUCINATION_NOISE = 0.005  # Max percent change for simulated prices

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def get_state(self, indicators):
        # Discretize indicators for state representation (simple version)
        return tuple(np.round(indicators, 2))

    def choose_action(self, state):
        if np.random.rand() < EPSILON:
            return np.random.choice(len(ACTIONS))
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += ALPHA * (reward + GAMMA * best_next - self.q_table[state][action])

def get_indicators(df):
    # Compute indicators: SMA, EMA, RSI (as in ML4T)
    df['SMA'] = df['close'].rolling(window=WINDOW).mean()
    df['EMA'] = df['close'].ewm(span=WINDOW).mean()
    delta = df['close'].diff() 
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14, min_periods=1).mean()
    avg_loss = down.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df.fillna(method='bfill', inplace=True)
    return df

def reward_function(position, price, prev_price):
    if prev_price is None:
        return 0
    if position == 1:  # long
        return price - prev_price
    elif position == -1:
        return prev_price - price
    else:
        return 0

def hallucinate_step(df, position):
    # Simulate a next close price with some noise
    last_price = df.iloc[-1]['close']
    pct_change = np.random.uniform(-HALLUCINATION_NOISE, HALLUCINATION_NOISE)
    new_price = last_price * (1 + pct_change)
    new_row = {'close': new_price}
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df = get_indicators(new_df)
    return new_df, new_price

def main():
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, 'v2')
    agent = QLearningAgent()
    position = 0  # 1: long, 0: flat
    prev_price = None
    price_history = deque(maxlen=WINDOW+20)
    logger.info("Trading bot started.")

    while True:
        try:
            barset = api.get_bars(SYMBOL, TIMEFRAME, limit=WINDOW+20).df
            if barset.empty:
                logger.warning("No data returned. Skipping iteration.")
                time.sleep(60)
                continue

            prices = barset[barset['symbol'] == SYMBOL]
            prices = prices.reset_index()
            df = prices[['close']]
            price_history.extend(df['close'].tolist())
            df = pd.DataFrame({'close': list(price_history)})
            df = get_indicators(df)
            indicators = df.iloc[-1][['SMA', 'EMA', 'RSI']].values

            # Q-Learning step (real tick)
            state = agent.get_state(indicators)
            action_idx = agent.choose_action(state)
            action = ACTIONS[action_idx]
            current_price = df.iloc[-1]['close']

            # Execute trade based on action
            if action == 'buy' and position == 0:
                api.submit_order(symbol=SYMBOL, qty=1, side='buy', type='market', time_in_force='gtc')
                position = 1
                logger.info(f"BUY executed at {current_price:.2f}")
            elif action == 'sell' and position == 1:
                api.submit_order(symbol=SYMBOL, qty=1, side='sell', type='market', time_in_force='gtc')
                position = 0
                logger.info(f"SELL executed at {current_price:.2f}")

            reward = reward_function(position, current_price, prev_price)
            next_state = agent.get_state(indicators)
            agent.update(state, action_idx, reward, next_state)
            logger.info(
                f"Step info | Position: {position} | Action: {action} | Price: {current_price:.2f} | Reward: {reward:.4f} | State: {state}"
            )
            prev_price = current_price

            # --- HALLUCINATION: Simulate extra Q-learning updates off-market ---
            sim_df = df.copy()
            sim_position = position
            sim_prev_price = prev_price
            for _ in range(HALLUCINATION_STEPS):
                sim_df, sim_price = hallucinate_step(sim_df, sim_position)
                sim_indicators = sim_df.iloc[-1][['SMA', 'EMA', 'RSI']].values
                sim_state = agent.get_state(sim_indicators)
                sim_action_idx = agent.choose_action(sim_state)
                sim_action = ACTIONS[sim_action_idx]
                sim_reward = reward_function(sim_position, sim_price, sim_prev_price)
                # Simulate position changes in hallucination
                if sim_action == 'buy' and sim_position == 0:
                    sim_position = 1
                elif sim_action == 'sell' and sim_position == 1:
                    sim_position = 0
                sim_next_state = agent.get_state(sim_indicators)
                agent.update(sim_state, sim_action_idx, sim_reward, sim_next_state)
                logger.debug(
                    f"Hallucination | Position: {sim_position} | Action: {sim_action} | Price: {sim_price:.2f} | Reward: {sim_reward:.4f} | State: {sim_state}"
                )
                sim_prev_price = sim_price

            time.sleep(60)  # Wait for 1 minute before next real tick

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()