# Dynamic Ensemble Reinforcement Learning Trading System

A robust hierarchical ensemble framework for responding to market regime changes in financial trading, based on the research paper "A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes".

## Overview

This system implements a four-layer hierarchical architecture that adapts to market regime changes through real-time regime classification and performance-based dynamic weight allocation. The framework consists of specialized agent pools for different market conditions (Bull, Bear, Sideways) and uses ensemble learning to aggregate their decisions.

## System Architecture

The system consists of four main layers:

1. Multimodal Feature Fusion Layer: Combines candlestick chart images (CNN), technical indicators, and news sentiment into a unified state vector.

2. Market Regime Classification Layer: Uses XGBoost to classify the market into Bull, Bear, or Sideways regimes with confidence-based selection mechanism.

3. PPO Reinforcement Learning Layer: Three separate pools of 5 PPO agents each, specialized for different market regimes with regime-specific reward functions.

4. Ensemble Decision Layer: Aggregates policies from active agent pool using dynamic weighting based on rolling 30-day Sharpe Ratio.

## Key Features

- Real-time market regime classification with confidence threshold (theta = 0.6)
- Regime-specific reward functions for optimal behavior in each market condition
- Dynamic ensemble weighting using Softmax with temperature (T = 10)
- Walk-Forward Expanding Window Cross-Validation for robust evaluation
- Comprehensive backtesting with transaction costs (0.05% fee, 0.02% slippage)

## Requirements

- Python 3.9 or higher
- PyTorch
- Stable Baselines3 (or custom PPO implementation)
- XGBoost
- Pandas, NumPy
- TA-Lib (for technical indicators)
- Gymnasium (or Gym)
- Matplotlib, Plotly (for visualization)

## Project Structure

```
dynamic_ensemble_rl_trading/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── regime/           # Regime classification
│   ├── env/               # Trading environment
│   ├── agents/            # RL agents and pools
│   ├── ensemble/          # Ensemble decision making
│   ├── backtest/          # Backtesting engine
│   ├── visualization/     # Plotting and visualization
│   └── utils/             # Utility functions
├── scripts/               # Execution scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit and integration tests
├── config/                # Configuration files
├── data/                  # Data files
├── models/                # Saved models
└── results/               # Results and outputs
```

## Dataset

### News Sentiment Data
- File: data/cryptonews_2021-10-12_2023-12-19.csv
- Period: October 12, 2021 to December 19, 2023
- Total articles: 31,037
- Columns: date, sentiment, source, subject, text, title, url
- Sentiment distribution: Positive (45.0%), Neutral (34.0%), Negative (21.0%)

### Candlestick Image Data
- File: data/chart_(7.42GB).zip (compressed)
- Contains candlestick chart images for feature extraction

## Data Download

All required data files can be downloaded from the following Google Drive link:

https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB

The Google Drive contains:
- cryptonews_2021-10-12_2023-12-19.csv (12.6 MB)
- chart_(7.42GB).zip (6.8 GB compressed)

Extract the chart images zip file to data/raw/charts/ directory after downloading.

## Installation

### Requirements
- Python 3.9+
- PyTorch
- Stable Baselines3 (or custom PPO implementation)
- XGBoost
- Pandas, NumPy
- TA-Lib (for technical indicators)
- Gymnasium (or Gym)
- Matplotlib, Plotly (for visualization)

### Setup
```bash
# Download the code from anonymous repository
# Repository: https://anonymous.4open.science/r/YOUR-ANONYMOUS-LINK-ID

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start (Component Testing)
```bash
python scripts/quick_start.py
```
This script tests basic components with synthetic data without requiring full data setup.

### Training

Train regime classifier:
```bash
python scripts/train.py --component regime --config config/config.yaml
```

Train PPO agents:
```bash
python scripts/train.py --component agents --config config/config.yaml
```

### Main Trading Execution
```bash
python scripts/main.py
```
This executes Algorithm 1 from the paper, running the complete trading loop.

### Evaluation
```bash
python scripts/evaluate.py --config config/config.yaml
```

### Example Usage
```bash
python scripts/example_usage.py
```
See example_usage.py for detailed usage examples of individual components.

## Configuration

Main configuration files are located in the `config/` directory:
- config.yaml: Main system configuration
- hyperparameters.yaml: Model hyperparameters
- paths.yaml: File paths and directories

## Key Hyperparameters

- Regime Classification: XGBoost (n_estimators=100, max_depth=6)
- Confidence Threshold: theta = 0.6
- PPO Agents: Learning rate = 3e-4, Batch size = 64, Gamma = 0.99
- Dynamic Ensemble: Performance window = 30 days, Temperature T = 10
- Transaction Costs: Fee = 0.05%, Slippage = 0.02%

## Performance

Based on 26 months of backtesting on BTC/USDT data:
- Sharpe Ratio: 1.89
- Cumulative Return: 89.3%
- CAGR: 34.2%
- Maximum Drawdown: -16.2%
- Win Rate: 67.8%

During 2022 bear market crisis:
- Return: +7.9% (vs -12.3% for baseline)
- Maximum Drawdown: -8.2%

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- API Documentation: docs/API.md
- Architecture Details: docs/ARCHITECTURE.md
- Reproduction Guide: docs/REPRODUCTION.md
- Implementation Status: IMPLEMENTATION_STATUS.md
- Anonymous Upload Guide: ANONYMOUS_UPLOAD.md

## Citation

If you use this code in your research, please cite the original paper:

```
A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes
```

## License

This project is provided for research and educational purposes.

