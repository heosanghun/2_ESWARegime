"""
Check data requirements and download guide.

Verifies whether paper-required data exists and provides download guidance when missing.
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
    Check whether paper-required data files exist.
    
    Parameters
    ----------
    config_path : str
        Path to config file.
    
    Returns
    -------
    dict
        Data file existence and details.
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
    """Print paper data requirements."""
    print("\n" + "=" * 100)
    print("Paper Data Requirements")
    print("=" * 100)
    
    print("\n1. OHLCV Data")
    print("   - Asset: BTC/USDT")
    print("   - Exchange: Binance")
    print("   - Period: 2021-10-12 ~ 2023-12-19 (26 months)")
    print("   - Frequency: Hourly")
    print("   - Expected data points: ~18,720")
    print("   - Format: CSV (date, open, high, low, close, volume)")
    
    print("\n2. News Data")
    print("   - File: cryptonews_2021-10-12_2023-12-19.csv")
    print("   - Total articles: 31,037")
    print("   - Columns: date, sentiment, source, subject, text, title, url")
    print("   - Period: 2021-10-12 ~ 2023-12-19")
    
    print("\n3. Candlestick Image Data")
    print("   - File: chart_(7.42GB).zip (compressed)")
    print("   - Or: can be generated on the fly from OHLCV data")
    print("   - Image size: 224x224 pixels")
    print("   - Lookback: 60 hours")
    
    print("\n" + "=" * 100)
    print("Data Download Methods")
    print("=" * 100)
    
    print("\n1. Google Drive (provided in paper)")
    print("   URL: https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB")
    print("   Included files:")
    print("   - cryptonews_2021-10-12_2023-12-19.csv (12.6 MB)")
    print("   - chart_(7.42GB).zip (6.8 GB compressed)")
    print("   Note: OHLCV data must be prepared separately")
    
    print("\n2. Binance API (OHLCV data)")
    print("   Using python-binance library:")
    print("   ```python")
    print("   from binance.client import Client")
    print("   client = Client()")
    print("   klines = client.get_historical_klines(")
    print("       'BTCUSDT', Client.KLINE_INTERVAL_1HOUR,")
    print("       '2021-10-12', '2023-12-19'")
    print("   )")
    print("   ```")
    
    print("\n3. Candlestick Image Generation")
    print("   If OHLCV data exists, generate automatically in code:")
    print("   ```python")
    print("   from src.data.candlestick_generator import CandlestickGenerator")
    print("   generator = CandlestickGenerator(image_size=224, lookback_hours=60)")
    print("   # Generate images from OHLCV data")
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
    print("Data Availability Summary")
    print("=" * 100)
    
    all_required_exist = all(
        results[key]['exists'] for key in results 
        if results[key]['required']
    )
    
    for data_type, info in results.items():
        status = "EXISTS" if info['exists'] else "MISSING"
        required = "(required)" if info['required'] else "(optional)"
        print(f"\n{data_type.upper()} {required}: {status}")
        print(f"  Path: {info['path']}")
        print(f"  Description: {info['description']}")
        if info['exists']:
            if 'num_images' in info:
                print(f"  Image count: {info['num_images']:,}")
            print(f"  Size: {info['size'] / (1024**2):.2f} MB")
    
    print("\n" + "=" * 100)
    
    if all_required_exist:
        print("All required data is ready!")
    else:
        print("Some required data is missing. See download methods above.")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
