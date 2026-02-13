"""
데이터 요구사항 확인 및 다운로드 가이드 스크립트.

논문에서 요구하는 데이터가 있는지 확인하고,
없는 경우 다운로드 방법을 안내합니다.
"""

import sys
from pathlib import Path
import yaml
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
        logging.FileHandler(str(log_dir / 'data_check.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_data_files(config_path: str = 'config/config.yaml') -> dict:
    """
    논문에서 요구하는 데이터 파일 존재 여부 확인.
    
    Parameters
    ----------
    config_path : str
        Config 파일 경로.
    
    Returns
    -------
    dict
        데이터 파일 존재 여부 및 상세 정보.
    """
    base_path = Path(config_path).parent.parent
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results = {
        'ohlcv': {
            'required': True,
            'path': base_path / config['data']['ohlcv_path'],
            'exists': False,
            'size': 0,
            'description': 'BTC/USDT hourly data (2021-10-12 ~ 2023-12-19)'
        },
        'news': {
            'required': True,
            'path': base_path / config['data']['news_path'],
            'exists': False,
            'size': 0,
            'description': 'Cryptocurrency news data (31,037 articles)'
        },
        'charts': {
            'required': False,  # Can be generated from OHLCV
            'path': base_path / config['data']['chart_images_path'],
            'exists': False,
            'size': 0,
            'description': 'Candlestick chart images (224x224)'
        }
    }
    
    # Check OHLCV data
    ohlcv_path = results['ohlcv']['path']
    if ohlcv_path.exists():
        results['ohlcv']['exists'] = True
        results['ohlcv']['size'] = ohlcv_path.stat().st_size
        logger.info(f"OHLCV data found: {ohlcv_path} ({results['ohlcv']['size']:,} bytes)")
    else:
        logger.warning(f"OHLCV data not found: {ohlcv_path}")
    
    # Check news data
    news_path = results['news']['path']
    if news_path.exists():
        results['news']['exists'] = True
        results['news']['size'] = news_path.stat().st_size
        logger.info(f"News data found: {news_path} ({results['news']['size']:,} bytes)")
    else:
        logger.warning(f"News data not found: {news_path}")
    
    # Check chart images
    charts_path = results['charts']['path']
    if charts_path.exists() and charts_path.is_dir():
        # Count images
        image_files = list(charts_path.glob('*.png')) + list(charts_path.glob('*.jpg'))
        results['charts']['exists'] = len(image_files) > 0
        results['charts']['num_images'] = len(image_files)
        if results['charts']['exists']:
            total_size = sum(f.stat().st_size for f in image_files)
            results['charts']['size'] = total_size
            logger.info(f"Chart images found: {charts_path} ({len(image_files):,} images, {total_size / (1024**2):.2f} MB)")
    else:
        logger.warning(f"Chart images not found: {charts_path}")
    
    return results


def print_data_requirements():
    """논문에서 요구하는 데이터 요구사항 출력."""
    print("\n" + "=" * 100)
    print("논문 데이터 요구사항")
    print("=" * 100)
    
    print("\n1. OHLCV 데이터")
    print("   - 자산: BTC/USDT")
    print("   - 거래소: Binance")
    print("   - 기간: 2021-10-12 ~ 2023-12-19 (26개월)")
    print("   - 주기: Hourly (시간별)")
    print("   - 예상 데이터 포인트: 약 18,720개")
    print("   - 형식: CSV (date, open, high, low, close, volume)")
    
    print("\n2. 뉴스 데이터")
    print("   - 파일: cryptonews_2021-10-12_2023-12-19.csv")
    print("   - 총 기사 수: 31,037개")
    print("   - 컬럼: date, sentiment, source, subject, text, title, url")
    print("   - 기간: 2021-10-12 ~ 2023-12-19")
    
    print("\n3. 캔들스틱 이미지 데이터")
    print("   - 파일: chart_(7.42GB).zip (압축)")
    print("   - 또는: OHLCV 데이터로부터 실시간 생성 가능")
    print("   - 이미지 크기: 224x224 픽셀")
    print("   - Lookback: 60시간")
    
    print("\n" + "=" * 100)
    print("데이터 다운로드 방법")
    print("=" * 100)
    
    print("\n1. Google Drive (논문에서 제공)")
    print("   URL: https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB")
    print("   포함 파일:")
    print("   - cryptonews_2021-10-12_2023-12-19.csv (12.6 MB)")
    print("   - chart_(7.42GB).zip (6.8 GB 압축)")
    print("   주의: OHLCV 데이터는 별도로 준비 필요")
    
    print("\n2. Binance API (OHLCV 데이터)")
    print("   python-binance 라이브러리 사용:")
    print("   ```python")
    print("   from binance.client import Client")
    print("   client = Client()")
    print("   klines = client.get_historical_klines(")
    print("       'BTCUSDT', Client.KLINE_INTERVAL_1HOUR,")
    print("       '2021-10-12', '2023-12-19'")
    print("   )")
    print("   ```")
    
    print("\n3. 캔들스틱 이미지 생성")
    print("   OHLCV 데이터가 있으면 코드로 자동 생성:")
    print("   ```python")
    print("   from src.data.candlestick_generator import CandlestickGenerator")
    print("   generator = CandlestickGenerator(image_size=224, lookback_hours=60)")
    print("   # OHLCV 데이터로부터 이미지 생성")
    print("   ```")
    
    print("\n" + "=" * 100)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check data requirements')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Print requirements
    print_data_requirements()
    
    # Check existing files
    logger.info("Checking existing data files...")
    results = check_data_files(args.config)
    
    # Summary
    print("\n" + "=" * 100)
    print("데이터 존재 여부 요약")
    print("=" * 100)
    
    all_required_exist = all(
        results[key]['exists'] for key in results 
        if results[key]['required']
    )
    
    for data_type, info in results.items():
        status = "EXISTS" if info['exists'] else "MISSING"
        required = "(필수)" if info['required'] else "(선택)"
        print(f"\n{data_type.upper()} {required}: {status}")
        print(f"  경로: {info['path']}")
        print(f"  설명: {info['description']}")
        if info['exists']:
            if 'num_images' in info:
                print(f"  이미지 수: {info['num_images']:,}")
            print(f"  크기: {info['size'] / (1024**2):.2f} MB")
    
    print("\n" + "=" * 100)
    
    if all_required_exist:
        print("✅ 모든 필수 데이터가 준비되었습니다!")
    else:
        print("⚠️ 일부 필수 데이터가 없습니다. 위의 다운로드 방법을 참고하세요.")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
