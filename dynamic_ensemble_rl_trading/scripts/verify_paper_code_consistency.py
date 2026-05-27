"""
Comprehensive paper-code consistency and data existence verification.

This script:
1. Checks whether data mentioned in the paper actually exists
2. Verifies the codebase matches the paper methodology
3. Checks hyperparameter consistency
4. Validates data paths and formats
"""

import sys
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
log_dir = Path('results')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'verification.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Data specs mentioned in the paper
PAPER_DATA_SPEC = {
    'ohlcv': {
        'asset': 'BTC/USDT',
        'period': '26 months',
        'start_date': '2021-10-12',
        'end_date': '2023-12-19',
        'frequency': 'hourly',
        'expected_path': 'data/raw/ohlcv_data.csv'
    },
    'news': {
        'file': 'cryptonews_2021-10-12_2023-12-19.csv',
        'total_articles': 31037,
        'period': '2021-10-12 to 2023-12-19',
        'expected_path': 'data/cryptonews_2021-10-12_2023-12-19.csv',
        'columns': ['date', 'sentiment', 'source', 'subject', 'text', 'title', 'url']
    },
    'charts': {
        'file': 'chart_(7.42GB).zip',
        'compressed_size': '7.42GB',
        'extracted_path': 'data/raw/charts/',
        'image_size': '224x224',
        'lookback_hours': 60
    }
}

# Hyperparameters mentioned in the paper
PAPER_HYPERPARAMETERS = {
    'regime_classifier': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'confidence_threshold': 0.6,  # theta
        'sma_window': 50,
        'bull_threshold': 0.0005,  # 0.05%
        'bear_threshold': -0.0005  # -0.05%
    },
    'ppo': {
        'learning_rate': 3e-4,
        'batch_size': 64,
        'gamma': 0.99,
        'n_steps': 2048,
        'n_epochs': 10
    },
    'ensemble': {
        'temperature': 10.0,  # T
        'performance_window': 30,  # days
        'num_agents_per_pool': 5
    },
    'transaction_costs': {
        'fee': 0.0005,  # 0.05%
        'slippage': 0.0002  # 0.02%
    },
    'features': {
        'visual': {
            'image_size': 224,
            'lookback_hours': 60,
            'model': 'ResNet-18'
        },
        'technical': {
            'num_indicators': 15,
            'normalization_window': 30,
            'indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'BB']
        },
        'sentiment': {
            'aggregation_window': 24,  # hours
            'use_ewma': True
        }
    }
}


class PaperCodeConsistencyVerifier:
    """
    Paper-code consistency and data verifier.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize verifier.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file.
        """
        self.config_path = Path(config_path)
        self.base_path = self.config_path.parent.parent
        
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}")
            self.config = {}
        
        self.verification_results = {
            'data_existence': {},
            'code_consistency': {},
            'hyperparameter_consistency': {},
            'overall_status': 'UNKNOWN'
        }
        
        logger.info("Initialized Paper-Code Consistency Verifier")
    
    def check_data_existence(self) -> Dict:
        """
        Check whether data files mentioned in the paper exist.
        
        Returns
        -------
        dict
            Data existence verification results.
        """
        logger.info("=" * 100)
        logger.info("Data existence verification")
        logger.info("=" * 100)
        
        results = {}
        
        # OHLCV data check
        ohlcv_path = self.base_path / PAPER_DATA_SPEC['ohlcv']['expected_path']
        results['ohlcv'] = {
            'expected_path': str(ohlcv_path),
            'exists': ohlcv_path.exists(),
            'is_file': ohlcv_path.is_file() if ohlcv_path.exists() else False,
            'size': ohlcv_path.stat().st_size if ohlcv_path.exists() and ohlcv_path.is_file() else 0
        }
        
        if results['ohlcv']['exists'] and results['ohlcv']['is_file']:
            try:
                df = pd.read_csv(ohlcv_path, nrows=5)
                results['ohlcv']['has_ohlcv_columns'] = all(
                    col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']
                )
                results['ohlcv']['sample_rows'] = len(df)
            except Exception as e:
                results['ohlcv']['error'] = str(e)
                results['ohlcv']['has_ohlcv_columns'] = False
        else:
            results['ohlcv']['has_ohlcv_columns'] = False
        
        logger.info(f"OHLCV data: {'OK exists' if results['ohlcv']['exists'] else 'FAIL missing'}")
        if results['ohlcv']['exists']:
            logger.info(f"  Path: {results['ohlcv']['expected_path']}")
            logger.info(f"  Size: {results['ohlcv']['size']:,} bytes")
        
        # News data check
        news_path = self.base_path / PAPER_DATA_SPEC['news']['expected_path']
        results['news'] = {
            'expected_path': str(news_path),
            'exists': news_path.exists(),
            'is_file': news_path.is_file() if news_path.exists() else False,
            'size': news_path.stat().st_size if news_path.exists() and news_path.is_file() else 0
        }
        
        if results['news']['exists'] and results['news']['is_file']:
            try:
                df = pd.read_csv(news_path, nrows=5)
                results['news']['has_required_columns'] = all(
                    col in df.columns for col in PAPER_DATA_SPEC['news']['columns']
                )
                results['news']['sample_rows'] = len(df)
                
                # Check total row count (optional)
                try:
                    total_rows = sum(1 for _ in open(news_path, 'r', encoding='utf-8')) - 1
                    results['news']['total_rows'] = total_rows
                    results['news']['matches_paper'] = abs(total_rows - PAPER_DATA_SPEC['news']['total_articles']) < 100
                except:
                    results['news']['total_rows'] = None
                    results['news']['matches_paper'] = None
            except Exception as e:
                results['news']['error'] = str(e)
                results['news']['has_required_columns'] = False
        else:
            results['news']['has_required_columns'] = False
        
        logger.info(f"News data: {'OK exists' if results['news']['exists'] else 'FAIL missing'}")
        if results['news']['exists']:
            logger.info(f"  Path: {results['news']['expected_path']}")
            logger.info(f"  Size: {results['news']['size']:,} bytes")
            if 'total_rows' in results['news'] and results['news']['total_rows']:
                logger.info(f"  Rows: {results['news']['total_rows']:,} (paper: {PAPER_DATA_SPEC['news']['total_articles']:,})")
        
        # Candlestick image data check
        charts_path = self.base_path / PAPER_DATA_SPEC['charts']['extracted_path']
        charts_zip = self.base_path / PAPER_DATA_SPEC['charts']['file']
        
        results['charts'] = {
            'extracted_path': str(charts_path),
            'extracted_exists': charts_path.exists() and charts_path.is_dir(),
            'zip_path': str(charts_zip),
            'zip_exists': charts_zip.exists() and charts_zip.is_file(),
            'zip_size': charts_zip.stat().st_size if charts_zip.exists() and charts_zip.is_file() else 0
        }
        
        if results['charts']['extracted_exists']:
            try:
                image_files = list(charts_path.glob('*.png')) + list(charts_path.glob('*.jpg'))
                results['charts']['num_images'] = len(image_files)
                results['charts']['has_images'] = len(image_files) > 0
            except:
                results['charts']['num_images'] = 0
                results['charts']['has_images'] = False
        else:
            results['charts']['num_images'] = 0
            results['charts']['has_images'] = False
        
        logger.info(f"Candlestick images: {'OK exists' if (results['charts']['extracted_exists'] or results['charts']['zip_exists']) else 'FAIL missing'}")
        if results['charts']['zip_exists']:
            logger.info(f"  ZIP file: {results['charts']['zip_path']}")
            logger.info(f"  Size: {results['charts']['zip_size'] / (1024**3):.2f} GB")
        if results['charts']['extracted_exists']:
            logger.info(f"  Extracted path: {results['charts']['extracted_path']}")
            if 'num_images' in results['charts']:
                logger.info(f"  Image count: {results['charts']['num_images']:,}")
        
        self.verification_results['data_existence'] = results
        return results
    
    def check_code_consistency(self) -> Dict:
        """
        Check whether the codebase matches the paper methodology.
        
        Returns
        -------
        dict
            Code consistency verification results.
        """
        logger.info("=" * 100)
        logger.info("Code-paper consistency verification")
        logger.info("=" * 100)
        
        results = {}
        
        # 1. Four-layer architecture check
        results['architecture'] = {
            'layer1_feature_fusion': Path(self.base_path / 'src/data/feature_fusion.py').exists(),
            'layer2_regime_classification': Path(self.base_path / 'src/regime/regime_classifier.py').exists(),
            'layer3_ppo_rl': Path(self.base_path / 'src/agents/ppo_agent.py').exists(),
            'layer4_ensemble': Path(self.base_path / 'src/ensemble/ensemble_trader.py').exists()
        }
        
        results['architecture']['all_layers_exist'] = all(results['architecture'].values())
        
        logger.info(f"4-layer architecture: {'OK all implemented' if results['architecture']['all_layers_exist'] else 'FAIL some missing'}")
        
        # 2. Main component check
        results['components'] = {
            'multimodal_features': {
                'visual': Path(self.base_path / 'src/data/candlestick_generator.py').exists(),
                'technical': Path(self.base_path / 'src/data/feature_extractor.py').exists(),
                'sentiment': Path(self.base_path / 'src/data/news_sentiment.py').exists()
            },
            'regime_classification': {
                'classifier': Path(self.base_path / 'src/regime/regime_classifier.py').exists(),
                'ground_truth': Path(self.base_path / 'src/regime/ground_truth.py').exists()
            },
            'ppo_agents': {
                'agent': Path(self.base_path / 'src/agents/ppo_agent.py').exists(),
                'pool': Path(self.base_path / 'src/agents/pool.py').exists(),
                'manager': Path(self.base_path / 'src/agents/agent_manager.py').exists()
            },
            'ensemble': {
                'weighting': Path(self.base_path / 'src/ensemble/weighting.py').exists(),
                'trader': Path(self.base_path / 'src/ensemble/ensemble_trader.py').exists()
            },
            'backtest': {
                'backtester': Path(self.base_path / 'src/backtest/backtester.py').exists(),
                'metrics': Path(self.base_path / 'src/backtest/metrics.py').exists()
            }
        }
        
        # 3. Baseline and ablation model check
        results['baselines'] = {
            'single_ppo': Path(self.base_path / 'src/baselines/single_ppo_agent.py').exists(),
            'xgboost': Path(self.base_path / 'src/baselines/xgboost_trader.py').exists(),
            'cnn': Path(self.base_path / 'src/baselines/cnn_trader.py').exists(),
            'simple_ensemble': Path(self.base_path / 'src/baselines/simple_ensemble.py').exists()
        }
        
        results['ablation'] = {
            'model1': Path(self.base_path / 'src/ablation/no_dynamic_weighting.py').exists(),
            'model2': Path(self.base_path / 'src/ablation/no_confidence_selection.py').exists(),
            'model3': Path(self.base_path / 'src/ablation/no_ensemble.py').exists(),
            'model4': Path(self.base_path / 'src/ablation/no_regime_classification.py').exists()
        }
        
        logger.info(f"Baseline methods: {'OK all implemented' if all(results['baselines'].values()) else 'FAIL some missing'}")
        logger.info(f"Ablation models: {'OK all implemented' if all(results['ablation'].values()) else 'FAIL some missing'}")
        
        self.verification_results['code_consistency'] = results
        return results
    
    def check_hyperparameter_consistency(self) -> Dict:
        """
        Check whether hyperparameters match the paper.
        
        Returns
        -------
        dict
            Hyperparameter consistency verification results.
        """
        logger.info("=" * 100)
        logger.info("Hyperparameter consistency verification")
        logger.info("=" * 100)
        
        if not self.config:
            logger.warning("Cannot load config file; skipping hyperparameter verification.")
            return {}
        
        results = {}
        
        # Regime Classifier hyperparameters
        if 'regime' in self.config:
            regime_config = self.config['regime']
            results['regime'] = {
                'confidence_threshold': {
                    'paper': PAPER_HYPERPARAMETERS['regime_classifier']['confidence_threshold'],
                    'config': regime_config.get('confidence_threshold'),
                    'matches': regime_config.get('confidence_threshold') == PAPER_HYPERPARAMETERS['regime_classifier']['confidence_threshold']
                },
                'sma_window': {
                    'paper': PAPER_HYPERPARAMETERS['regime_classifier']['sma_window'],
                    'config': regime_config.get('sma_window'),
                    'matches': regime_config.get('sma_window') == PAPER_HYPERPARAMETERS['regime_classifier']['sma_window']
                }
            }
        
        # Ensemble hyperparameters
        if 'ensemble' in self.config:
            ensemble_config = self.config['ensemble']
            results['ensemble'] = {
                'temperature': {
                    'paper': PAPER_HYPERPARAMETERS['ensemble']['temperature'],
                    'config': ensemble_config.get('temperature'),
                    'matches': ensemble_config.get('temperature') == PAPER_HYPERPARAMETERS['ensemble']['temperature']
                },
                'performance_window': {
                    'paper': PAPER_HYPERPARAMETERS['ensemble']['performance_window'],
                    'config': ensemble_config.get('performance_window'),
                    'matches': ensemble_config.get('performance_window') == PAPER_HYPERPARAMETERS['ensemble']['performance_window']
                },
                'num_agents_per_pool': {
                    'paper': PAPER_HYPERPARAMETERS['ensemble']['num_agents_per_pool'],
                    'config': ensemble_config.get('num_agents_per_pool'),
                    'matches': ensemble_config.get('num_agents_per_pool') == PAPER_HYPERPARAMETERS['ensemble']['num_agents_per_pool']
                }
            }
        
        # Transaction Costs
        if 'training' in self.config:
            training_config = self.config['training']
            results['transaction_costs'] = {
                'fee': {
                    'paper': PAPER_HYPERPARAMETERS['transaction_costs']['fee'],
                    'config': training_config.get('transaction_fee'),
                    'matches': abs(training_config.get('transaction_fee', 0) - PAPER_HYPERPARAMETERS['transaction_costs']['fee']) < 1e-6
                },
                'slippage': {
                    'paper': PAPER_HYPERPARAMETERS['transaction_costs']['slippage'],
                    'config': training_config.get('slippage'),
                    'matches': abs(training_config.get('slippage', 0) - PAPER_HYPERPARAMETERS['transaction_costs']['slippage']) < 1e-6
                }
            }
        
        # Features
        if 'features' in self.config:
            features_config = self.config['features']
            results['features'] = {
                'visual': {
                    'image_size': {
                        'paper': PAPER_HYPERPARAMETERS['features']['visual']['image_size'],
                        'config': features_config.get('visual', {}).get('image_size'),
                        'matches': features_config.get('visual', {}).get('image_size') == PAPER_HYPERPARAMETERS['features']['visual']['image_size']
                    },
                    'lookback_hours': {
                        'paper': PAPER_HYPERPARAMETERS['features']['visual']['lookback_hours'],
                        'config': features_config.get('visual', {}).get('lookback_hours'),
                        'matches': features_config.get('visual', {}).get('lookback_hours') == PAPER_HYPERPARAMETERS['features']['visual']['lookback_hours']
                    }
                },
                'sentiment': {
                    'aggregation_window': {
                        'paper': PAPER_HYPERPARAMETERS['features']['sentiment']['aggregation_window'],
                        'config': features_config.get('sentiment', {}).get('aggregation_window'),
                        'matches': features_config.get('sentiment', {}).get('aggregation_window') == PAPER_HYPERPARAMETERS['features']['sentiment']['aggregation_window']
                    }
                }
            }
        
        # Print results
        for category, params in results.items():
            logger.info(f"\n{category.upper()}:")
            for param_name, param_data in params.items():
                if isinstance(param_data, dict) and 'matches' in param_data:
                    status = "OK" if param_data['matches'] else "FAIL"
                    logger.info(
                        f"  {status} {param_name}: "
                        f"paper={param_data['paper']}, "
                        f"config={param_data['config']}"
                    )
        
        self.verification_results['hyperparameter_consistency'] = results
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive verification report."""
        report = []
        report.append("=" * 100)
        report.append("Paper-Code Consistency and Data Existence Verification Report")
        report.append("=" * 100)
        report.append("")
        
        # Data existence
        report.append("[1. Data Existence]")
        report.append("-" * 100)
        
        data_results = self.verification_results.get('data_existence', {})
        
        if 'ohlcv' in data_results:
            ohlcv = data_results['ohlcv']
            status = "OK" if ohlcv.get('exists') else "FAIL"
            report.append(f"{status} OHLCV data")
            report.append(f"  Path: {ohlcv.get('expected_path', 'N/A')}")
            report.append(f"  Exists: {'yes' if ohlcv.get('exists') else 'no'}")
            if ohlcv.get('exists'):
                report.append(f"  Size: {ohlcv.get('size', 0):,} bytes")
                report.append(f"  OHLCV columns present: {'yes' if ohlcv.get('has_ohlcv_columns') else 'no'}")
            report.append("")
        
        if 'news' in data_results:
            news = data_results['news']
            status = "OK" if news.get('exists') else "FAIL"
            report.append(f"{status} News data")
            report.append(f"  Path: {news.get('expected_path', 'N/A')}")
            report.append(f"  Exists: {'yes' if news.get('exists') else 'no'}")
            if news.get('exists'):
                report.append(f"  Size: {news.get('size', 0):,} bytes")
                if 'total_rows' in news and news['total_rows']:
                    report.append(f"  Rows: {news['total_rows']:,} (paper: {PAPER_DATA_SPEC['news']['total_articles']:,})")
            report.append("")
        
        if 'charts' in data_results:
            charts = data_results['charts']
            status = "OK" if (charts.get('extracted_exists') or charts.get('zip_exists')) else "FAIL"
            report.append(f"{status} Candlestick image data")
            if charts.get('zip_exists'):
                report.append(f"  ZIP file: {charts.get('zip_path', 'N/A')}")
                report.append(f"  Size: {charts.get('zip_size', 0) / (1024**3):.2f} GB")
            if charts.get('extracted_exists'):
                report.append(f"  Extracted path: {charts.get('extracted_path', 'N/A')}")
                if 'num_images' in charts:
                    report.append(f"  Image count: {charts['num_images']:,}")
            report.append("")
        
        # Code consistency
        report.append("[2. Code-Paper Consistency]")
        report.append("-" * 100)
        
        code_results = self.verification_results.get('code_consistency', {})
        
        if 'architecture' in code_results:
            arch = code_results['architecture']
            status = "OK" if arch.get('all_layers_exist') else "FAIL"
            report.append(f"{status} 4-layer architecture implementation")
            report.append(f"  Layer 1 (Feature Fusion): {'OK' if arch.get('layer1_feature_fusion') else 'FAIL'}")
            report.append(f"  Layer 2 (Regime Classification): {'OK' if arch.get('layer2_regime_classification') else 'FAIL'}")
            report.append(f"  Layer 3 (PPO RL): {'OK' if arch.get('layer3_ppo_rl') else 'FAIL'}")
            report.append(f"  Layer 4 (Ensemble): {'OK' if arch.get('layer4_ensemble') else 'FAIL'}")
            report.append("")
        
        if 'baselines' in code_results:
            baselines = code_results['baselines']
            all_exist = all(baselines.values())
            status = "OK" if all_exist else "FAIL"
            report.append(f"{status} Baseline method implementation")
            for name, exists in baselines.items():
                report.append(f"  {name}: {'OK' if exists else 'FAIL'}")
            report.append("")
        
        if 'ablation' in code_results:
            ablation = code_results['ablation']
            all_exist = all(ablation.values())
            status = "OK" if all_exist else "FAIL"
            report.append(f"{status} Ablation study model implementation")
            for name, exists in ablation.items():
                report.append(f"  {name}: {'OK' if exists else 'FAIL'}")
            report.append("")
        
        # Hyperparameter consistency
        report.append("[3. Hyperparameter Consistency]")
        report.append("-" * 100)
        
        hyper_results = self.verification_results.get('hyperparameter_consistency', {})
        
        if hyper_results:
            for category, params in hyper_results.items():
                report.append(f"\n{category.upper()}:")
                for param_name, param_data in params.items():
                    if isinstance(param_data, dict) and 'matches' in param_data:
                        status = "OK" if param_data['matches'] else "FAIL"
                        report.append(
                            f"  {status} {param_name}: "
                            f"paper={param_data['paper']}, "
                            f"config={param_data['config']}"
                        )
        else:
            report.append("Cannot load config file; verification skipped.")
        
        report.append("")
        
        # Overall assessment
        report.append("=" * 100)
        report.append("[Overall Assessment]")
        report.append("=" * 100)
        
        # Data existence summary
        data_exists = (
            data_results.get('ohlcv', {}).get('exists', False) or
            data_results.get('news', {}).get('exists', False) or
            data_results.get('charts', {}).get('extracted_exists', False) or
            data_results.get('charts', {}).get('zip_exists', False)
        )
        
        code_consistent = code_results.get('architecture', {}).get('all_layers_exist', False)
        
        report.append(f"Data exists: {'OK at least some' if data_exists else 'FAIL none'}")
        report.append(f"Code implementation: {'OK complete' if code_consistent else 'FAIL incomplete'}")
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_report(self, output_path: str) -> None:
        """Save verification report."""
        report = self.generate_comprehensive_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Verification report saved to {output_path}")
        print("\n" + report)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify paper-code consistency and data existence')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/verification/comprehensive_verification_report.txt',
        help='Output path for verification report'
    )
    
    args = parser.parse_args()
    
    verifier = PaperCodeConsistencyVerifier(args.config)
    
    # Check data existence
    verifier.check_data_existence()
    
    # Check code consistency
    verifier.check_code_consistency()
    
    # Check hyperparameter consistency
    verifier.check_hyperparameter_consistency()
    
    # Generate and save report
    verifier.save_report(args.output)
    
    logger.info("Comprehensive verification completed!")


if __name__ == "__main__":
    main()
