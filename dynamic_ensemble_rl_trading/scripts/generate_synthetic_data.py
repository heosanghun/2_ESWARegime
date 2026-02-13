"""
논문 스펙에 맞는 합성 데이터 생성 스크립트.

논문에서 사용한 데이터 스펙에 맞춰 합성 데이터를 생성하여
시스템 테스트 및 검증을 가능하게 합니다.
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
    논문 스펙에 맞는 BTC/USDT OHLCV 합성 데이터 생성.
    
    Parameters
    ----------
    start_date : str
        시작 날짜 (YYYY-MM-DD).
    end_date : str
        종료 날짜 (YYYY-MM-DD).
    base_price : float
        초기 가격 (USD).
    output_path : str
        출력 파일 경로.
    
    Returns
    -------
    pd.DataFrame
        생성된 OHLCV 데이터.
    """
    logger.info(f"Generating OHLCV data from {start_date} to {end_date}")
    
    # 날짜 범위 생성 (hourly)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    timestamps = pd.date_range(start=start, end=end, freq='H')
    
    logger.info(f"Total timestamps: {len(timestamps)}")
    
    # 시드 고정 (재현 가능성)
    np.random.seed(42)
    
    # 가격 생성 (랜덤 워크 + 트렌드)
    # 2021년 말: 상승장, 2022년: 약세장, 2023년: 회복
    prices = []
    current_price = base_price
    
    for i, ts in enumerate(timestamps):
        # 시장 체제에 따른 다른 변동성
        if ts.year == 2021 or (ts.year == 2022 and ts.month <= 5):
            # 상승장: 양의 트렌드
            trend = 0.0001  # 시간당 0.01% 상승
            volatility = 0.02  # 2% 변동성
        elif ts.year == 2022:
            # 약세장: 음의 트렌드
            trend = -0.00015  # 시간당 0.015% 하락
            volatility = 0.025  # 2.5% 변동성 (높은 변동성)
        else:
            # 2023년: 회복 단계
            trend = 0.00005  # 약한 상승
            volatility = 0.018  # 1.8% 변동성
        
        # 랜덤 워크
        change = np.random.normal(trend, volatility)
        current_price = current_price * (1 + change)
        
        # OHLC 생성
        intraday_volatility = volatility * 0.3
        open_price = current_price
        close_price = open_price * (1 + np.random.normal(0, intraday_volatility))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intraday_volatility * 0.5)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intraday_volatility * 0.5)))
        volume = np.random.lognormal(15, 0.5)  # 로그 정규 분포
        
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
    
    # 파일 저장
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
    논문 스펙에 맞는 뉴스 데이터 생성.
    
    Parameters
    ----------
    start_date : str
        시작 날짜.
    end_date : str
        종료 날짜.
    total_articles : int
        총 기사 수 (논문: 31,037개).
    output_path : str
        출력 파일 경로.
    
    Returns
    -------
    pd.DataFrame
        생성된 뉴스 데이터.
    """
    logger.info(f"Generating news data: {total_articles} articles")
    
    np.random.seed(42)
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days
    
    # 날짜별 기사 수 분포 (일부 날짜에 집중)
    articles_per_day = np.random.poisson(total_articles / date_range, date_range)
    articles_per_day = articles_per_day.clip(min=1)  # 최소 1개
    
    # 총 기사 수 조정
    current_total = articles_per_day.sum()
    if current_total != total_articles:
        diff = total_articles - current_total
        # 차이를 랜덤하게 분배
        indices = np.random.choice(len(articles_per_day), abs(diff), replace=True)
        articles_per_day[indices] += np.sign(diff)
    
    # 기사 생성
    articles = []
    sentiment_classes = ['positive', 'neutral', 'negative']
    sentiment_distribution = [0.45, 0.34, 0.21]  # 논문에서 언급한 분포
    
    sources = ['CoinDesk', 'CoinTelegraph', 'Bloomberg', 'Reuters', 'CryptoNews']
    subjects = ['Bitcoin', 'Cryptocurrency', 'Market', 'Trading', 'Regulation']
    
    article_id = 0
    for day_offset in range(date_range):
        date = start + timedelta(days=day_offset)
        num_articles = articles_per_day[day_offset]
        
        for _ in range(num_articles):
            # 시간 랜덤 생성
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            timestamp = date.replace(hour=hour, minute=minute)
            
            # 감정 분포에 따라 선택
            sentiment_class = np.random.choice(
                sentiment_classes,
                p=sentiment_distribution
            )
            
            # 감정 점수 생성
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
    
    # 파일 저장
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
    logger.info("합성 데이터 생성 시작")
    logger.info("=" * 100)
    
    # OHLCV 데이터 생성
    ohlcv_data = generate_ohlcv_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.ohlcv_output
    )
    
    # 뉴스 데이터 생성
    news_data = generate_news_data(
        start_date=args.start_date,
        end_date=args.end_date,
        total_articles=args.num_articles,
        output_path=args.news_output
    )
    
    logger.info("=" * 100)
    logger.info("합성 데이터 생성 완료!")
    logger.info("=" * 100)
    logger.info(f"OHLCV 데이터: {len(ohlcv_data)} rows")
    logger.info(f"뉴스 데이터: {len(news_data)} articles")
    logger.info("\n다음 단계: 모델 학습을 진행하세요.")


if __name__ == "__main__":
    main()
