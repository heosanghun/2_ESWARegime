# API Documentation

## Data Processing Modules

### MarketDataHandler

Main class for loading and preprocessing OHLCV data.

#### Methods

- `load_data(start_date=None, end_date=None)`: Load OHLCV data from CSV
- `get_walk_forward_splits(...)`: Generate Walk-Forward splits
- `get_data()`: Get loaded data
- `get_ohlcv_columns()`: Get OHLCV column mapping

### TechnicalFeatureExtractor

Extracts 15 technical indicators from OHLCV data.

#### Methods

- `extract_features(df)`: Extract all technical features
- `get_feature_names()`: Get list of feature names

### CandlestickGenerator

Generates candlestick images and extracts visual features.

#### Methods

- `generate_candlestick_image(ohlcv_data, timestamp, ohlcv_columns)`: Generate image
- `extract_visual_features(image)`: Extract ResNet-18 features
- `process_timestamp(ohlcv_data, timestamp, ohlcv_columns)`: Process single timestamp
- `batch_process(ohlcv_data, timestamps, ohlcv_columns, batch_size)`: Batch processing

### NewsSentimentExtractor

Processes news sentiment data and aggregates scores.

#### Methods

- `load_news_data(start_date=None, end_date=None)`: Load news data
- `aggregate_sentiment_by_hour(target_timestamps)`: Aggregate sentiment
- `get_sentiment_features(target_timestamps)`: Get sentiment feature vectors

### FeatureFusion

Combines multimodal features into unified state vector.

#### Methods

- `create_unified_state(ohlcv_data, timestamp, ohlcv_columns)`: Create state for timestamp
- `batch_create_unified_states(ohlcv_data, timestamps, ohlcv_columns)`: Batch processing
- `get_state_dimension()`: Get dimension of unified state vector

## Regime Classification

### RegimeGroundTruth

Generates ground truth labels for regime classification.

#### Methods

- `calculate_sma_slope(price_data)`: Calculate normalized SMA slope
- `generate_labels(price_data)`: Generate regime labels
- `get_regime_name(label)`: Get regime name from label

### RegimeClassifier

XGBoost-based market regime classifier.

#### Methods

- `fit(X, y, validation_data=None)`: Train classifier
- `predict_proba(X)`: Get probability distribution
- `predict(X)`: Predict regime labels
- `select_regime_with_confidence(state, previous_regime)`: Select with confidence (Eq. 4)
- `predict_with_confidence(state, previous_regime)`: Get detailed prediction
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load trained model

## Trading Environment

### MultiRegimeTradingEnv

Gymnasium-compatible trading environment.

#### Methods

- `reset(seed=None, options=None)`: Reset environment
- `step(action)`: Execute action and return (obs, reward, terminated, truncated, info)
- `render(mode='human')`: Render environment state

### RegimeRewardCalculator

Calculates regime-specific rewards.

#### Methods

- `calculate_bull_reward(...)`: Bull market reward
- `calculate_bear_reward(...)`: Bear market reward
- `calculate_sideways_reward(...)`: Sideways market reward
- `calculate_reward(regime, ...)`: General reward calculation
- `reset()`: Reset portfolio history

## Agent Management

### PPOAgent

PPO agent wrapper for Stable Baselines3.

#### Methods

- `train(total_timesteps, callback=None, log_interval=10)`: Train agent
- `predict(observation, deterministic=False)`: Predict action
- `get_policy_distribution(observation)`: Get policy probabilities
- `save(filepath)`: Save agent
- `load(filepath)`: Load agent

### PPOAgentPool

Manages pool of PPO agents.

#### Methods

- `train_pool(total_timesteps, callback=None, log_interval=10)`: Train all agents
- `get_pool_actions(observation, return_probs=True)`: Get actions from all agents
- `get_agent(index)`: Get specific agent
- `save_pool(base_path)`: Save all agents
- `load_pool(base_path)`: Load all agents

### HierarchicalAgentManager

Manages three regime-specific agent pools.

#### Methods

- `get_pool(regime)`: Get pool for regime
- `train_all_pools(total_timesteps, ...)`: Train all pools
- `train_pool_by_regime(regime, total_timesteps, ...)`: Train specific pool
- `save_all_pools(base_path)`: Save all pools
- `load_all_pools(base_path)`: Load all pools

## Ensemble

### DynamicWeightCalculator

Calculates dynamic weights based on agent performance.

#### Methods

- `initialize_agents(num_agents)`: Initialize tracking
- `update_returns(agent_index, return_value)`: Update agent returns
- `calculate_weights()`: Calculate weights (Eq. 6)
- `reset()`: Reset all tracking

### EnsembleTrader

Aggregates policies and selects final action.

#### Methods

- `initialize_agents(num_agents)`: Initialize tracking
- `update_agent_performance(agent_index, portfolio_value)`: Update performance
- `get_ensemble_action(state, active_pool)`: Get ensemble action (Eq. 7, 8)
- `reset()`: Reset tracking

## Backtesting

### Backtester

Backtesting engine with Walk-Forward methodology.

#### Methods

- `run_backtest(trading_history, ohlcv_data, ohlcv_columns)`: Run backtest
- `calculate_buy_hold_benchmark(ohlcv_data, start_date, end_date, ohlcv_columns)`: Calculate benchmark

### PerformanceMetrics

Calculates performance metrics.

#### Methods

- `calculate_cumulative_return(portfolio_values)`: Cumulative return
- `calculate_cagr(portfolio_values, num_years)`: CAGR
- `calculate_sharpe_ratio(returns, periods_per_year)`: Sharpe Ratio
- `calculate_max_drawdown(portfolio_values)`: Maximum Drawdown
- `calculate_win_rate(returns)`: Win rate
- `calculate_profit_factor(returns)`: Profit factor
- `calculate_all_metrics(portfolio_values, returns, num_years)`: All metrics
- `compare_with_benchmark(strategy_metrics, benchmark_metrics)`: Compare with benchmark

## Visualization

### TradingPlotter

Creates visualization plots.

#### Methods

- `plot_portfolio_vs_benchmark(...)`: Portfolio vs benchmark
- `plot_regime_detection(...)`: Regime detection over time
- `plot_dynamic_weights(...)`: Dynamic weights visualization
- `plot_performance_metrics(...)`: Performance metrics comparison


