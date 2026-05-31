"""
Generate synthetic data matching paper specifications.

Creates synthetic data aligned with the paper data spec to enable
system testing and verification.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
log_dir = Path('results')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'data_generation.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_ohlcv_data(
    start_date: str = '2021-10-12',
    end_date: str = '2023-12-19',
    base_price: float = 50000.0,
    output_path: str = 'data/raw/ohlcv_data.csv'
) -> pd.DataFrame:
    """
    Generate synthetic BTC/USDT OHLCV data matching paper specifications.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    base_price : float
        Initial price (USD).
    output_path : str
        Output file path.
    
    Returns
    -------
    pd.DataFrame
        Generated OHLCV data.
    """
    logger.info(f"Generating OHLCV data from {start_date} to {end_date}")
    
    # Build date range (hourly)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    timestamps = pd.date_range(start=start, end=end, freq='h')
    
    logger.info(f"Total timestamps: {len(timestamps)}")
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # Price generation (random walk + trend)
    # Late 2021: bull, 2022: bear, 2023: recovery
    prices = []
    current_price = base_price
    
    for i, ts in enumerate(timestamps):
        # Different volatility by market regime
        if ts.year == 2021 or (ts.year == 2022 and ts.month <= 5):
            # Bull market: positive trend
            trend = 0.0001  # 0.01% per hour
            volatility = 0.02  # 2% volatility
        elif ts.year == 2022:
            # Bear market: negative trend
            trend = -0.00015  # 0.015% per hour down
            volatility = 0.025  # 2.5% volatility (high)
        else:
            # 2023: recovery phase
            trend = 0.00005  # mild uptrend
            volatility = 0.018  # 1.8% volatility
        
        # Random walk
        change = np.random.normal(trend, volatility)
        current_price = current_price * (1 + change)
        
        # Build OHLC
        intraday_volatility = volatility * 0.3
        open_price = current_price
        close_price = open_price * (1 + np.random.normal(0, intraday_volatility))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intraday_volatility * 0.5)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intraday_volatility * 0.5)))
        volume = np.random.lognormal(15, 0.5)  # log-normal distribution
        
        prices.append({
            'date': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_price = close_price
    
    df = pd.DataFrame(prices)
    df.set_index('date', inplace=True)
    
    # Save file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file)
    
    logger.info(f"Generated OHLCV data: {len(df)} rows")
    logger.info(f"Saved to: {output_file}")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def generate_news_data(
    start_date: str = '2021-10-12',
    end_date: str = '2023-12-19',
    total_articles: int = 31037,
    output_path: str = 'data/cryptonews_2021-10-12_2023-12-19.csv'
) -> pd.DataFrame:
    """
    Generate news data matching paper specifications.
    
    Parameters
    ----------
    start_date : str
        Start date.
    end_date : str
        End date.
    total_articles : int
        Total article count (paper: 31,037).
    output_path : str
        Output file path.
    
    Returns
    -------
    pd.DataFrame
        Generated news data.
    """
    logger.info(f"Generating news data: {total_articles} articles")
    
    np.random.seed(42)
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days
    
    # Articles per day (some days concentrated)
    articles_per_day = np.random.poisson(total_articles / date_range, date_range)
    articles_per_day = articles_per_day.clip(min=1)  # at least 1
    
    # Adjust total article count
    current_total = articles_per_day.sum()
    if current_total != total_articles:
        diff = total_articles - current_total
        # Distribute difference randomly
        indices = np.random.choice(len(articles_per_day), abs(diff), replace=True)
        articles_per_day[indices] += np.sign(diff)
    
    # Generate articles
    articles = []
    sentiment_classes = ['positive', 'neutral', 'negative']
    sentiment_distribution = [0.45, 0.34, 0.21]  # distribution mentioned in paper
    
    sources = ['CoinDesk', 'CoinTelegraph', 'Bloomberg', 'Reuters', 'CryptoNews']
    subjects = ['Bitcoin', 'Cryptocurrency', 'Market', 'Trading', 'Regulation']
    
    article_id = 0
    for day_offset in range(date_range):
        date = start + timedelta(days=day_offset)
        num_articles = articles_per_day[day_offset]
        
        for _ in range(num_articles):
            # Random time
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            timestamp = date.replace(hour=hour, minute=minute)
            
            # Select by sentiment distribution
            sentiment_class = np.random.choice(
                sentiment_classes,
                p=sentiment_distribution
            )
            
            # Generate sentiment scores
            if sentiment_class == 'positive':
                polarity = np.random.uniform(0.1, 1.0)
                subjectivity = np.random.uniform(0.3, 0.8)
            elif sentiment_class == 'negative':
                polarity = np.random.uniform(-1.0, -0.1)
                subjectivity = np.random.uniform(0.3, 0.8)
            else:
                polarity = np.random.uniform(-0.1, 0.1)
                subjectivity = np.random.uniform(0.1, 0.5)
            
            articles.append({
                'date': timestamp,
                'sentiment': sentiment_class,
                'sentiment_polarity': polarity,
                'sentiment_subjectivity': subjectivity,
                'source': np.random.choice(sources),
                'subject': np.random.choice(subjects),
                'title': f'Crypto News Article {article_id}',
                'text': f'Sample news text for article {article_id}',
                'url': f'https://example.com/news/{article_id}'
            })
            article_id += 1
    
    df = pd.DataFrame(articles)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Generated news data: {len(df)} articles")
    logger.info(f"Saved to: {output_file}")
    logger.info(f"Sentiment distribution:")
    logger.info(f"  Positive: {(df['sentiment'] == 'positive').sum()} ({(df['sentiment'] == 'positive').sum()/len(df)*100:.1f}%)")
    logger.info(f"  Neutral: {(df['sentiment'] == 'neutral').sum()} ({(df['sentiment'] == 'neutral').sum()/len(df)*100:.1f}%)")
    logger.info(f"  Negative: {(df['sentiment'] == 'negative').sum()} ({(df['sentiment'] == 'negative').sum()/len(df)*100:.1f}%)")
    
    return df


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic data matching paper specifications')
    parser.add_argument(
        '--ohlcv-output',
        type=str,
        default='data/raw/ohlcv_data.csv',
        help='Output path for OHLCV data'
    )
    parser.add_argument(
        '--news-output',
        type=str,
        default='data/cryptonews_2021-10-12_2023-12-19.csv',
        help='Output path for news data'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2021-10-12',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-19',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--num-articles',
        type=int,
        default=31037,
        help='Number of news articles to generate'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 100)
    logger.info("Starting synthetic data generation")
    logger.info("=" * 100)
    
    # Generate OHLCV data
    ohlcv_data = generate_ohlcv_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.ohlcv_output
    )
    
    # Generate news data
    news_data = generate_news_data(
        start_date=args.start_date,
        end_date=args.end_date,
        total_articles=args.num_articles,
        output_path=args.news_output
    )
    
    logger.info("=" * 100)
    logger.info("Synthetic data generation complete!")
    logger.info("=" * 100)
    logger.info(f"OHLCV data: {len(ohlcv_data)} rows")
    logger.info(f"News data: {len(news_data)} articles")
    logger.info("\nNext step: proceed with model training.")


if __name__ == "__main__":
    main()
