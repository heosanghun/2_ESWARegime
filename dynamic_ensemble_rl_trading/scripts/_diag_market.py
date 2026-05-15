"""Quick diagnostic: what did BTC do during the test window?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.data_processor import MarketDataHandler

dh = MarketDataHandler("data/raw/btcusdt_1h.csv")
ohlcv = dh.load_data(start_date="2023-06-19", end_date="2023-12-19")
print(f"Test period: {ohlcv.index[0]} to {ohlcv.index[-1]} ({len(ohlcv)} bars)")
start_price = ohlcv["close"].iloc[0]
end_price = ohlcv["close"].iloc[-1]
buy_hold = (end_price - start_price) / start_price
print(f"BTC start: ${start_price:.2f}, end: ${end_price:.2f}")
print(f"Buy & Hold cumulative return: {buy_hold * 100:+.2f}%")
rets = ohlcv["close"].pct_change().dropna()
bh_sharpe = (rets.mean() / rets.std()) * np.sqrt(24 * 365)
print(f"Buy & Hold annualized Sharpe: {bh_sharpe:.2f}")
print(f"Max price: ${ohlcv['close'].max():.2f}")
print(f"Min price: ${ohlcv['close'].min():.2f}")
print(f"Mean hourly return: {rets.mean() * 100:.4f}%")
print(f"Hourly stdev: {rets.std() * 100:.4f}%")

train = dh.load_data(start_date="2021-10-12", end_date="2023-06-19")
start_train = train["close"].iloc[0]
end_train = train["close"].iloc[-1]
bh_train = (end_train - start_train) / start_train
print()
print(f"Train period: {train.index[0]} to {train.index[-1]} ({len(train)} bars)")
print(f"Train start: ${start_train:.2f}, end: ${end_train:.2f}")
print(f"Train Buy & Hold cumulative: {bh_train * 100:+.2f}%")
