"""Download BTC/USDT 1h OHLCV for OOS 2024 forward-test window
(2024-03-01 → 2024-08-31) from Binance via ccxt.

Also writes a synthetic neutral-sentiment news CSV in the schema expected
by ``NewsSentimentExtractor`` so the existing backtest pipeline runs
unchanged. The synthetic news is *intentionally neutral* — it serves as a
controlled placeholder (consistent with how the original 2021-2023 dataset
appears to use placeholder articles) and ensures the OOS test isolates the
effect of price dynamics rather than re-introducing a look-ahead-biased
sentiment signal.

Outputs:
  data/raw/btcusdt_1h_oos2024.csv
  data/cryptonews_finbert_2024-03-01_2024-08-31.csv
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

logger = logging.getLogger("oos2024_data")

START_DATE = "2024-03-01"
END_DATE = "2024-08-31"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"

OUT_OHLCV = ROOT / "data" / "raw" / "btcusdt_1h_oos2024.csv"
OUT_NEWS = ROOT / "data" / f"cryptonews_finbert_{START_DATE}_{END_DATE}.csv"


def download_binance_ohlcv() -> pd.DataFrame:
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    since = exchange.parse8601(f"{START_DATE}T00:00:00Z")
    end_ms = exchange.parse8601(f"{END_DATE}T23:59:59Z")
    limit = 1000

    all_ohlcv = []
    logger.info("Downloading %s %s from Binance: %s → %s", SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=limit)
        except Exception as e:  # pragma: no cover - network heuristic
            logger.warning("Binance fetch error at %s: %s. Retrying in 5s ...", since, e)
            time.sleep(5)
            continue
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        last_dt = datetime.utcfromtimestamp(ohlcv[-1][0] / 1000)
        logger.info("  Fetched %d candles, last: %s", len(ohlcv), last_dt)
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.tz_localize(None)
    df = df.loc[START_DATE:END_DATE]
    return df


def write_synthetic_neutral_news(start: str, end: str) -> pd.DataFrame:
    """Mirror the schema of the existing FinBERT CSV but with strictly
    neutral sentiment. This guarantees that any OOS behaviour is *not*
    attributable to a 2025-era LLM having re-scored 2024 headlines.
    """
    timestamps = pd.date_range(start=start, end=end, freq="3h", inclusive="both")
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append({
            "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "sentiment": "neutral",
            "sentiment_polarity": 0.0,
            "sentiment_subjectivity": 0.0,
            "source": "synthetic_neutral_placeholder",
            "subject": "Bitcoin",
            "title": f"OOS-2024 Placeholder Article {i + 1}",
            "text": "Synthetic neutral placeholder article for OOS 2024 forward test.",
            "url": f"https://example.com/oos2024/news/{i + 1}",
            "sentiment_polarity_legacy": 0.0,
            "sentiment_subjectivity_legacy": 0.0,
            "sentiment_class": "neutral",
        })
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    OUT_OHLCV.parent.mkdir(parents=True, exist_ok=True)
    OUT_NEWS.parent.mkdir(parents=True, exist_ok=True)

    if OUT_OHLCV.exists():
        df = pd.read_csv(OUT_OHLCV, parse_dates=["timestamp"]).set_index("timestamp")
        logger.info("OHLCV already exists at %s (%d rows). Skipping download.", OUT_OHLCV, len(df))
    else:
        df = download_binance_ohlcv()
        if df.empty:
            logger.error("Failed to download any candles for %s..%s", START_DATE, END_DATE)
            return 1
        df.to_csv(OUT_OHLCV)
        logger.info("Wrote %d OHLCV rows → %s (%s..%s)", len(df), OUT_OHLCV, df.index[0], df.index[-1])

    if OUT_NEWS.exists():
        logger.info("Synthetic news file already exists: %s", OUT_NEWS)
    else:
        news_df = write_synthetic_neutral_news(START_DATE, END_DATE)
        news_df.to_csv(OUT_NEWS, index=False)
        logger.info("Wrote %d synthetic neutral news rows → %s", len(news_df), OUT_NEWS)

    # Brief OHLCV summary
    if not df.empty:
        cum = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
        logger.info(
            "OHLCV summary: n=%d  close[0]=%.2f  close[-1]=%.2f  cumret=%+.2f%%",
            len(df), df["close"].iloc[0], df["close"].iloc[-1], 100 * cum,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
