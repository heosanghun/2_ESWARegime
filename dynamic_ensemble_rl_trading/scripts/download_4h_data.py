"""
Download 4-hour BTC/USDT OHLCV data from Binance REST API.
Period: 2017-01-01 to 2026-05-31 (approx. 9 years)
Frequency: 4h (4-hour)
"""

import json
import urllib.request
import urllib.parse
import time
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "btcusdt_4h.csv"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
START = "2017-01-01T00:00:00Z"
END   = "2026-05-31T00:00:00Z"


def to_ms(iso_str):
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_ms = to_ms(START)
    end_ms = to_ms(END)
    current_ms = start_ms

    all_candles = []
    
    print(f"Downloading {SYMBOL} {INTERVAL} from Binance API (2017 → 2026)...")
    
    while current_ms < end_ms:
        params = {
            'symbol': SYMBOL,
            'interval': INTERVAL,
            'startTime': current_ms,
            'limit': 1000
        }
        url = f"https://api.binance.com/api/v3/klines?{urllib.parse.urlencode(params)}"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"  Error at {current_ms}: {e}. Retrying in 5s...")
            time.sleep(5)
            continue

        if not data or len(data) == 0:
            break

        for kline in data:
            open_time = kline[0]
            if open_time >= end_ms:
                break
            all_candles.append({
                'date': datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
            
        last_ms = data[-1][0]
        if last_ms <= current_ms:
            break
        current_ms = last_ms + 1
        
        dt_str = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Fetched {len(data)} candles, last: {dt_str}")
        time.sleep(0.3)

    if not all_candles:
        print("No candles fetched.")
        return

    import csv
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'open', 'high', 'low', 'close', 'volume'])
        writer.writeheader()
        writer.writerows(all_candles)

    print(f"\nSaved {len(all_candles)} 4h candles to {OUTPUT_FILE}")
    print(f"  Period: {all_candles[0]['date']} → {all_candles[-1]['date']}")


if __name__ == "__main__":
    download()
