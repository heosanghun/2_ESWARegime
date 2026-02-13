"""
실제 데이터를 프로젝트에 통합하는 스크립트.

사용자가 제공한 실제 데이터 경로를 확인하고,
필요시 심볼릭 링크 또는 복사를 통해 프로젝트에 통합합니다.
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
    """실제 데이터 경로를 확인하고 프로젝트에 통합."""
    
    # 실제 데이터 경로
    real_data_paths = {
        'news': Path(r'D:\AI\TradingAgents\0_data\crypto_news\cryptonews_2021-10-12_2023-12-19.csv'),
        'ohlcv': Path(r'D:\AI\MCTS\data_\ohlc\BTC-USD.csv'),
        'charts': Path(r'D:\AI\TradingAgents\0_data\candlestick_images\chart_(7.42GB)'),
    }
    
    # 프로젝트 데이터 경로
    project_data_paths = {
        'news': Path('data/cryptonews_2021-10-12_2023-12-19.csv'),
        'ohlcv': Path('data/raw/ohlcv_data.csv'),
        'charts': Path('data/raw/charts'),
    }
    
    logger.info("=" * 100)
    logger.info("실제 데이터 확인 및 통합 시작")
    logger.info("=" * 100)
    
    # 1. 뉴스 데이터 확인 및 복사
    if real_data_paths['news'].exists():
        logger.info(f"✓ 뉴스 데이터 발견: {real_data_paths['news']}")
        project_data_paths['news'].parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 크기 확인
        size_mb = real_data_paths['news'].stat().st_size / (1024 * 1024)
        logger.info(f"  파일 크기: {size_mb:.2f} MB")
        
        # 샘플 확인
        try:
            df = pd.read_csv(real_data_paths['news'], nrows=5)
            logger.info(f"  컬럼: {list(df.columns)}")
            logger.info(f"  샘플 행 수: {len(df)}")
        except Exception as e:
            logger.warning(f"  샘플 읽기 실패: {e}")
        
        # 심볼릭 링크 또는 복사
        if not project_data_paths['news'].exists():
            try:
                # Windows에서는 심볼릭 링크가 관리자 권한 필요할 수 있음
                project_data_paths['news'].symlink_to(real_data_paths['news'])
                logger.info(f"  심볼릭 링크 생성: {project_data_paths['news']}")
            except Exception:
                # 복사로 대체
                shutil.copy2(real_data_paths['news'], project_data_paths['news'])
                logger.info(f"  파일 복사 완료: {project_data_paths['news']}")
        else:
            logger.info(f"  이미 존재: {project_data_paths['news']}")
    else:
        logger.error(f"✗ 뉴스 데이터 없음: {real_data_paths['news']}")
    
    # 2. OHLCV 데이터 확인 및 변환
    if real_data_paths['ohlcv'].exists():
        logger.info(f"✓ OHLCV 데이터 발견: {real_data_paths['ohlcv']}")
        
        # 데이터 읽기 및 확인
        try:
            df = pd.read_csv(real_data_paths['ohlcv'])
            logger.info(f"  컬럼: {list(df.columns)}")
            logger.info(f"  총 행 수: {len(df)}")
            logger.info(f"  날짜 범위: {df['Date'].min()} ~ {df['Date'].max()}")
            
            # 날짜 변환
            df['Date'] = pd.to_datetime(df['Date'])
            
            # 논문 기간 필터링 (2021-10-12 ~ 2023-12-19)
            start_date = pd.to_datetime('2021-10-12')
            end_date = pd.to_datetime('2023-12-19')
            
            df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
            
            if len(df_filtered) > 0:
                logger.info(f"  필터링 후 행 수: {len(df_filtered)}")
                logger.info(f"  필터링된 날짜 범위: {df_filtered['Date'].min()} ~ {df_filtered['Date'].max()}")
                
                # 컬럼명 표준화 (대소문자 통일)
                df_filtered = df_filtered.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # date를 인덱스로 설정
                df_filtered.set_index('date', inplace=True)
                
                # 저장
                project_data_paths['ohlcv'].parent.mkdir(parents=True, exist_ok=True)
                df_filtered.to_csv(project_data_paths['ohlcv'])
                logger.info(f"  저장 완료: {project_data_paths['ohlcv']}")
            else:
                logger.warning(f"  ⚠ 논문 기간에 해당하는 데이터가 없습니다!")
                logger.info(f"  전체 데이터를 사용합니다.")
                
                # 컬럼명 표준화
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
                logger.info(f"  저장 완료: {project_data_paths['ohlcv']}")
                
        except Exception as e:
            logger.error(f"  OHLCV 데이터 처리 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.error(f"✗ OHLCV 데이터 없음: {real_data_paths['ohlcv']}")
    
    # 3. 차트 이미지 확인
    if real_data_paths['charts'].exists():
        logger.info(f"✓ 차트 이미지 디렉토리 발견: {real_data_paths['charts']}")
        
        # PNG 파일 개수 확인
        try:
            png_files = list(real_data_paths['charts'].rglob('*.png'))
            logger.info(f"  PNG 파일 개수: {len(png_files)}")
            
            if len(png_files) > 0:
                logger.info(f"  샘플 파일: {png_files[0].name}")
                
                # 심볼릭 링크 생성 (디렉토리)
                project_data_paths['charts'].parent.mkdir(parents=True, exist_ok=True)
                
                if not project_data_paths['charts'].exists():
                    try:
                        # Windows에서는 디렉토리 심볼릭 링크
                        import os
                        if os.name == 'nt':  # Windows
                            import subprocess
                            subprocess.run([
                                'cmd', '/c', 'mklink', '/D',
                                str(project_data_paths['charts']),
                                str(real_data_paths['charts'])
                            ], check=True)
                            logger.info(f"  디렉토리 심볼릭 링크 생성: {project_data_paths['charts']}")
                        else:
                            project_data_paths['charts'].symlink_to(real_data_paths['charts'])
                            logger.info(f"  심볼릭 링크 생성: {project_data_paths['charts']}")
                    except Exception as e:
                        logger.warning(f"  심볼릭 링크 생성 실패: {e}")
                        logger.info(f"  직접 경로 사용을 권장합니다.")
                else:
                    logger.info(f"  이미 존재: {project_data_paths['charts']}")
        except Exception as e:
            logger.error(f"  차트 이미지 확인 실패: {e}")
    else:
        logger.error(f"✗ 차트 이미지 디렉토리 없음: {real_data_paths['charts']}")
    
    logger.info("=" * 100)
    logger.info("데이터 통합 완료!")
    logger.info("=" * 100)
    
    # Config 파일 업데이트 안내
    logger.info("\n다음 단계:")
    logger.info("1. config/config.yaml에서 데이터 경로 확인")
    logger.info("2. 모델 학습 실행: python scripts/train.py --component all")


if __name__ == "__main__":
    check_and_setup_data()
