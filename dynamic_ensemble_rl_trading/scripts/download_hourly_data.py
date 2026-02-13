"""
Download hourly BTC/USDT OHLCV data from Binance via ccxt.

Period: 2021-10-12 to 2023-12-19 (논문과 동일)
Frequency: 1H (hourly)
"""

import ccxt
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "btcusdt_1h.csv"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
START = "2021-10-01T00:00:00Z"   # buffer before paper start
END   = "2023-12-20T00:00:00Z"   # buffer after paper end


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })

    since = exchange.parse8601(START)
    end_ms = exchange.parse8601(END)

    all_ohlcv = []
    limit = 1000

    print(f"Downloading {SYMBOL} {TIMEFRAME} from Binance ...")
    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=limit)
        except Exception as e:
            print(f"  Error at {since}: {e}. Retrying in 5s...")
            time.sleep(5)
            continue

        if len(ohlcv) == 0:
            break

        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # next ms after last candle
        dt = datetime.utcfromtimestamp(ohlcv[-1][0] / 1000)
        print(f"  Fetched {len(ohlcv)} candles, last: {dt}")
        time.sleep(exchange.rateLimit / 1000)

    # Build DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]]

    # Remove timezone info for compatibility
    df.index = df.index.tz_localize(None)

    # Filter to exact paper period
    df = df.loc["2021-10-12":"2023-12-19"]

    df.to_csv(OUTPUT_FILE)
    print(f"\nSaved {len(df)} hourly candles to {OUTPUT_FILE}")
    print(f"  Period: {df.index[0]} → {df.index[-1]}")
    return df


if __name__ == "__main__":
    download()
