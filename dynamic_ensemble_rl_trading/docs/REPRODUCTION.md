# Reproduction Guide

This guide provides step-by-step instructions for reproducing the results from the paper "A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes".

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training, but not required for inference)
- At least 16GB RAM
- Sufficient disk space for data and models

## Installation

1. Download the code from anonymous repository:
```bash
# Repository: https://anonymous.4open.science/r/YOUR-ANONYMOUS-LINK-ID
# Download and extract the code to your local directory
cd dynamic_ensemble_rl_trading
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download data from Google Drive:
   - Link: https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB
   - Download `cryptonews_2021-10-12_2023-12-19.csv` to `data/` directory
   - Download `chart_(7.42GB).zip` and extract to `data/raw/charts/` directory

2. Prepare OHLCV data:
   - Place your OHLCV CSV file in `data/raw/` directory
   - Update `config/config.yaml` with the correct path

## Training

### Step 1: Train Regime Classifier

```bash
python scripts/train.py --component regime --config config/config.yaml
```

This will:
- Load and preprocess data
- Generate ground truth labels
- Extract multimodal features
- Train XGBoost classifier
- Save model to `models/regime_classifier/`

### Step 2: Train PPO Agents

```bash
python scripts/train.py --component agents --config config/config.yaml
```

This will:
- Create trading environments for each regime
- Train 15 PPO agents (5 per regime)
- Save models to `models/ppo_agents/`

Note: Training may take several hours or days depending on hardware.

## Evaluation

### Run Main Trading Loop

```bash
python scripts/main.py
```

This executes Algorithm 1 from the paper:
- Loads trained models
- Runs trading loop on test data
- Saves results to `results/backtest/trading_results.pkl`

### Evaluate Performance

```bash
python scripts/evaluate.py --config config/config.yaml
```

This will:
- Run backtest on trading results
- Calculate performance metrics
- Generate comparison with Buy & Hold
- Create visualization plots

## Expected Results

Based on the paper, you should expect:
- Sharpe Ratio: ~1.89
- Cumulative Return: ~89.3%
- CAGR: ~34.2%
- Max Drawdown: ~-16.2%

During 2022 bear market:
- Return: +7.9% (vs -12.3% for baseline)
- Max Drawdown: -8.2%

## Troubleshooting

### GPU Memory Issues

If you encounter GPU memory errors:
- Reduce batch size in `config/hyperparameters.yaml`
- Train agents sequentially instead of in parallel
- Use CPU mode (set device='cpu' in agent initialization)

### Data Issues

- Ensure all data files are in correct locations
- Check that date formats match in config files
- Verify OHLCV data has required columns

### Model Loading Errors

- Ensure models are trained before running main.py
- Check model file paths in config
- Verify model files exist in specified directories

## Citation

If you use this code, please cite the original paper.

