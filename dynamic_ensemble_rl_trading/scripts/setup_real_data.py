"""
Integrate real data into the project.

Checks user-provided real data paths and integrates them into the project
via symbolic links or copy when needed.
"""

import sys
from pathlib import Path
import pandas as pd
import shutil
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/data_setup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_and_setup_data():
    """Check real data paths and integrate into the project."""
    
    # Real data paths
    real_data_paths = {
        'news': Path(r'D:\AI\TradingAgents\0_data\crypto_news\cryptonews_2021-10-12_2023-12-19.csv'),
        'ohlcv': Path(r'D:\AI\MCTS\data_\ohlc\BTC-USD.csv'),
        'charts': Path(r'D:\AI\TradingAgents\0_data\candlestick_images\chart_(7.42GB)'),
    }
    
    # Project data paths
    project_data_paths = {
        'news': Path('data/cryptonews_2021-10-12_2023-12-19.csv'),
        'ohlcv': Path('data/raw/ohlcv_data.csv'),
        'charts': Path('data/raw/charts'),
    }
    
    logger.info("=" * 100)
    logger.info("Starting real data check and integration")
    logger.info("=" * 100)
    
    # 1. Check and copy news data
    if real_data_paths['news'].exists():
        logger.info(f"OK News data found: {real_data_paths['news']}")
        project_data_paths['news'].parent.mkdir(parents=True, exist_ok=True)
        
        # Check file size
        size_mb = real_data_paths['news'].stat().st_size / (1024 * 1024)
        logger.info(f"  File size: {size_mb:.2f} MB")
        
        # Sample check
        try:
            df = pd.read_csv(real_data_paths['news'], nrows=5)
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Sample rows: {len(df)}")
        except Exception as e:
            logger.warning(f"  Sample read failed: {e}")
        
        # Symbolic link or copy
        if not project_data_paths['news'].exists():
            try:
                # Windows symbolic links may require admin privileges
                project_data_paths['news'].symlink_to(real_data_paths['news'])
                logger.info(f"  Symbolic link created: {project_data_paths['news']}")
            except Exception:
                # Fall back to copy
                shutil.copy2(real_data_paths['news'], project_data_paths['news'])
                logger.info(f"  File copied: {project_data_paths['news']}")
        else:
            logger.info(f"  Already exists: {project_data_paths['news']}")
    else:
        logger.error(f"FAIL News data missing: {real_data_paths['news']}")
    
    # 2. Check and convert OHLCV data
    if real_data_paths['ohlcv'].exists():
        logger.info(f"OK OHLCV data found: {real_data_paths['ohlcv']}")
        
        # Read and inspect data
        try:
            df = pd.read_csv(real_data_paths['ohlcv'])
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Total rows: {len(df)}")
            logger.info(f"  Date range: {df['Date'].min()} ~ {df['Date'].max()}")
            
            # Convert dates
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter to paper period (2021-10-12 ~ 2023-12-19)
            start_date = pd.to_datetime('2021-10-12')
            end_date = pd.to_datetime('2023-12-19')
            
            df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
            
            if len(df_filtered) > 0:
                logger.info(f"  Rows after filtering: {len(df_filtered)}")
                logger.info(f"  Filtered date range: {df_filtered['Date'].min()} ~ {df_filtered['Date'].max()}")
                
                # Standardize column names
                df_filtered = df_filtered.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Set date as index
                df_filtered.set_index('date', inplace=True)
                
                # Save
                project_data_paths['ohlcv'].parent.mkdir(parents=True, exist_ok=True)
                df_filtered.to_csv(project_data_paths['ohlcv'])
                logger.info(f"  Saved: {project_data_paths['ohlcv']}")
            else:
                logger.warning(f"  WARN No data for paper period!")
                logger.info(f"  Using full dataset.")
                
                # Standardize column names
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                df.set_index('date', inplace=True)
                
                project_data_paths['ohlcv'].parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(project_data_paths['ohlcv'])
                logger.info(f"  Saved: {project_data_paths['ohlcv']}")
                
        except Exception as e:
            logger.error(f"  OHLCV data processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.error(f"FAIL OHLCV data missing: {real_data_paths['ohlcv']}")
    
    # 3. Check chart images
    if real_data_paths['charts'].exists():
        logger.info(f"OK Chart images directory found: {real_data_paths['charts']}")
        
        # Count PNG files
        try:
            png_files = list(real_data_paths['charts'].rglob('*.png'))
            logger.info(f"  PNG file count: {len(png_files)}")
            
            if len(png_files) > 0:
                logger.info(f"  Sample file: {png_files[0].name}")
                
                # Create symbolic link (directory)
                project_data_paths['charts'].parent.mkdir(parents=True, exist_ok=True)
                
                if not project_data_paths['charts'].exists():
                    try:
                        # Directory symbolic link on Windows
                        import os
                        if os.name == 'nt':  # Windows
                            import subprocess
                            subprocess.run([
                                'cmd', '/c', 'mklink', '/D',
                                str(project_data_paths['charts']),
                                str(real_data_paths['charts'])
                            ], check=True)
                            logger.info(f"  Directory symbolic link created: {project_data_paths['charts']}")
                        else:
                            project_data_paths['charts'].symlink_to(real_data_paths['charts'])
                            logger.info(f"  Symbolic link created: {project_data_paths['charts']}")
                    except Exception as e:
                        logger.warning(f"  Symbolic link creation failed: {e}")
                        logger.info(f"  Recommend using the direct path.")
                else:
                    logger.info(f"  Already exists: {project_data_paths['charts']}")
        except Exception as e:
            logger.error(f"  Chart image check failed: {e}")
    else:
        logger.error(f"FAIL Chart images directory missing: {real_data_paths['charts']}")
    
    logger.info("=" * 100)
    logger.info("Data integration complete!")
    logger.info("=" * 100)
    
    # Config update guidance
    logger.info("\nNext steps:")
    logger.info("1. Verify data paths in config/config.yaml")
    logger.info("2. Run model training: python scripts/train.py --component all")


if __name__ == "__main__":
    check_and_setup_data()
