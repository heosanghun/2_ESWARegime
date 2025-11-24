# System Architecture Documentation

## Overview

The Dynamic Ensemble Reinforcement Learning Trading System implements a four-layer hierarchical architecture designed to adapt to market regime changes through specialized agent pools and dynamic ensemble weighting.

## Architecture Layers

### Layer 1: Multimodal Feature Fusion

**Purpose**: Combine heterogeneous data sources into unified state representation.

**Components**:
- Visual Features (F_visual): ResNet-18 CNN processing 224x224 candlestick images
- Technical Features (F_tech): 15 technical indicators with rolling normalization
- Sentiment Features (F_senti): News sentiment aggregation over 24-hour windows

**Output**: Unified state vector S_t = concatenate(F_visual, F_tech, F_senti)

**Key Modules**:
- `src/data/candlestick_generator.py`: Image generation and CNN feature extraction
- `src/data/feature_extractor.py`: Technical indicator calculation
- `src/data/news_sentiment.py`: Sentiment score aggregation
- `src/data/feature_fusion.py`: State vector concatenation

### Layer 2: Market Regime Classification

**Purpose**: Identify current market regime (Bull, Bear, Sideways) with confidence assessment.

**Components**:
- XGBoost classifier trained on multimodal features
- Ground truth generation from SMA-50 slope
- Confidence-based selection mechanism (theta = 0.6)

**Key Modules**:
- `src/regime/ground_truth.py`: Label generation from price trends
- `src/regime/regime_classifier.py`: XGBoost classification and confidence logic

**Decision Logic** (Eq. 4):
- If max(P(R|S_t)) >= 0.6: Switch to predicted regime
- Otherwise: Maintain previous regime

### Layer 3: PPO Reinforcement Learning

**Purpose**: Specialized agents for each market regime with regime-specific rewards.

**Components**:
- Three independent agent pools (Bull, Bear, Sideways)
- Five PPO agents per pool (total 15 agents)
- Regime-specific reward functions

**Reward Functions**:
- Bull: R = (V_{t+1} - V_t) / V_t (profit maximization)
- Bear: R = SortinoRatio - C * TransactionCost (risk minimization)
- Sideways: R = R_bull - 5 * TransactionCost (whipsaw avoidance)

**Key Modules**:
- `src/env/trading_env.py`: Gymnasium trading environment
- `src/env/rewards.py`: Regime-specific reward calculation
- `src/agents/ppo_agent.py`: PPO agent implementation
- `src/agents/pool.py`: Agent pool management
- `src/agents/agent_manager.py`: Hierarchical pool coordination

### Layer 4: Ensemble Decision

**Purpose**: Aggregate policies from active agent pool using dynamic weighting.

**Components**:
- Dynamic weight calculation based on 30-day rolling Sharpe Ratio
- Softmax weighting with temperature T = 10
- Policy aggregation and action selection

**Weighting Mechanism** (Eq. 6):
w_{i,t} = exp(SR_{i,30}/T) / sum(exp(SR_{j,30}/T))

**Policy Aggregation** (Eq. 7):
pi_ensemble(a|s_t) = sum(w_{i,t} * pi_i(a|s_t))

**Action Selection** (Eq. 8):
a_t = argmax(pi_ensemble(a|s_t))

**Key Modules**:
- `src/ensemble/weighting.py`: Dynamic weight calculation
- `src/ensemble/ensemble_trader.py`: Policy aggregation and action selection

## Data Flow

1. **Input**: OHLCV data, news data, candlestick images
2. **Feature Extraction**: Multimodal features extracted in parallel
3. **State Creation**: Features concatenated into unified state vector
4. **Regime Classification**: XGBoost predicts regime with confidence
5. **Pool Selection**: Active agent pool selected based on regime
6. **Weight Calculation**: Dynamic weights computed from agent performance
7. **Policy Aggregation**: Individual policies combined with weights
8. **Action Selection**: Final action selected via argmax
9. **Execution**: Action executed in trading environment
10. **Reward Calculation**: Regime-specific reward computed
11. **Performance Update**: Agent performance tracked for future weighting

## Key Design Decisions

### Why Three Separate Pools?

Separate pools prevent catastrophic forgetting. Agents specialized for Bear markets retain their defensive strategies even during Bull runs, ready to activate instantly when needed.

### Why Confidence Threshold?

Prevents erratic regime switching during ambiguous market conditions. Only switches when prediction confidence is high (>= 0.6), maintaining policy stability.

### Why Dynamic Weighting?

Allows the system to adapt to changing agent performance. Better-performing agents naturally receive higher weights, while maintaining diversity through temperature parameter.

### Why 30-Day Window?

Balances responsiveness to recent changes with stability against short-term noise. Empirically determined to be optimal.

### Why Temperature T = 10?

Provides balance between exploitation (favoring best agents) and exploration (maintaining ensemble diversity). Too low would be winner-take-all, too high would be uniform averaging.

## Module Dependencies

```
data_processor -> feature_extractor
                -> candlestick_generator
                -> news_sentiment -> feature_fusion
                
feature_fusion -> regime_classifier
                -> trading_env (x3 for each regime)
                
trading_env -> agent_manager -> ensemble_trader
                              -> backtester -> metrics
                                            -> plotter
```

## Configuration

All system parameters are configurable via:
- `config/config.yaml`: Main configuration
- `config/hyperparameters.yaml`: Model hyperparameters
- `config/paths.yaml`: File paths

## Extensibility

The architecture is designed for extension:
- Additional regimes can be added by creating new agent pools
- New feature types can be integrated into FeatureFusion
- Alternative RL algorithms can replace PPO
- Different ensemble methods can be implemented


