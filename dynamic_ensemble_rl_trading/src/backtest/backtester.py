"""
Backtesting engine for trading strategies.

This module implements Walk-Forward Expanding Window Cross-Validation
for robust evaluation of trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine with Walk-Forward methodology.
    
    Implements Walk-Forward Expanding Window Cross-Validation to
    simulate realistic operational workflow and avoid look-ahead bias.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_fee: float = 0.0005,
        slippage: float = 0.0002
    ):
        """
        Initialize the backtester.
        
        Parameters
        ----------
        initial_capital : float, default=10000.0
            Initial portfolio capital.
        transaction_fee : float, default=0.0005
            Transaction fee as fraction (0.05%).
        slippage : float, default=0.0002
            Slippage as fraction (0.02%).
        """
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        
        self.metrics_calculator = PerformanceMetrics()
    
    def run_backtest(
        self,
        trading_history: List[Dict],
        ohlcv_data: pd.DataFrame,
        ohlcv_columns: Optional[dict] = None
    ) -> Dict:
        """
        Run backtest on trading history.
        
        Parameters
        ----------
        trading_history : list of dict
            List of trading decisions with timestamps, actions, etc.
        ohlcv_data : pd.DataFrame
            OHLCV data for price execution.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        
        Returns
        -------
        dict
            Backtest results including portfolio values and metrics.
        """
        logger.info(f"Running backtest on {len(trading_history)} trades")
        
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # Initialize portfolio
        portfolio_value = self.initial_capital
        position = 0.0  # Position size (fraction)
        cash = self.initial_capital
        
        portfolio_values = [portfolio_value]
        returns = []
        trades = []
        
        # Process each trading decision
        for i, trade in enumerate(trading_history):
            timestamp = trade['timestamp']
            action = trade['action']
            
            if timestamp not in ohlcv_data.index:
                continue
            
            # Get execution price (next period's open to avoid look-ahead)
            if i < len(trading_history) - 1:
                next_timestamp = trading_history[i + 1]['timestamp']
                if next_timestamp in ohlcv_data.index:
                    execution_price = ohlcv_data.loc[next_timestamp, ohlcv_columns['open']]
                else:
                    execution_price = ohlcv_data.loc[timestamp, ohlcv_columns['close']]
            else:
                execution_price = ohlcv_data.loc[timestamp, ohlcv_columns['close']]
            
            # Apply slippage
            execution_price *= (1 + self.slippage)
            
            # Execute action
            # Action mapping: 0=Strong Sell, 1=Sell, 2=Hold, 3=Buy, 4=Strong Buy
            action_weights = [-1.0, -0.5, 0.0, 0.5, 1.0]
            target_position = action_weights[action]
            
            # Calculate position change
            position_change = target_position - position
            
            if abs(position_change) > 1e-6:
                # Calculate transaction cost
                transaction_amount = abs(position_change) * portfolio_value
                transaction_cost = transaction_amount * self.transaction_fee
                
                # Update position and cash
                position = target_position
                cash -= transaction_cost
                
                trades.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': execution_price,
                    'position': position,
                    'transaction_cost': transaction_cost
                })
            
            # Update portfolio value
            if i > 0:
                prev_timestamp = trading_history[i - 1]['timestamp']
                if prev_timestamp in ohlcv_data.index:
                    prev_price = ohlcv_data.loc[prev_timestamp, ohlcv_columns['close']]
                    price_change = (execution_price - prev_price) / prev_price
                else:
                    price_change = 0.0
            else:
                price_change = 0.0
            
            # Portfolio value = cash + position * value_change
            portfolio_value = cash + (self.initial_capital * position * (1 + price_change))
            portfolio_value = max(0.0, portfolio_value)
            
            # Calculate return
            if len(portfolio_values) > 0:
                return_value = (portfolio_value - portfolio_values[-1]) / portfolio_values[-1]
                returns.append(return_value)
            
            portfolio_values.append(portfolio_value)
        
        # Calculate metrics
        portfolio_array = np.array(portfolio_values)
        returns_array = np.array(returns) if returns else np.array([0.0])
        
        # Estimate number of years
        if len(trading_history) > 1:
            time_span = (trading_history[-1]['timestamp'] - trading_history[0]['timestamp']).days
            num_years = time_span / 365.25
        else:
            num_years = 1.0
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            portfolio_array,
            returns_array,
            num_years
        )
        
        logger.info("Backtest completed")
        logger.info(f"Cumulative Return: {metrics['cumulative_return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        return {
            'portfolio_values': portfolio_array,
            'returns': returns_array,
            'trades': trades,
            'metrics': metrics,
            'num_trades': len(trades)
        }
    
    def calculate_buy_hold_benchmark(
        self,
        ohlcv_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        ohlcv_columns: Optional[dict] = None
    ) -> Dict:
        """
        Calculate Buy & Hold benchmark performance.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data.
        start_date : pd.Timestamp
            Start date.
        end_date : pd.Timestamp
            End date.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        
        Returns
        -------
        dict
            Buy & Hold metrics.
        """
        if ohlcv_columns is None:
            ohlcv_columns = {'close': 'close'}
        
        # Filter data
        mask = (ohlcv_data.index >= start_date) & (ohlcv_data.index <= end_date)
        data = ohlcv_data[mask]
        
        if len(data) == 0:
            return {'metrics': {}}
        
        # Calculate returns
        prices = data[ohlcv_columns['close']].values
        initial_price = prices[0]
        final_price = prices[-1]
        
        portfolio_values = prices / initial_price * self.initial_capital
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate years
        time_span = (end_date - start_date).days
        num_years = time_span / 365.25
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            portfolio_values,
            returns,
            num_years
        )
        
        return {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'metrics': metrics
        }

