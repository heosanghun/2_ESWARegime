"""
Data processing module for loading and preprocessing market data.

This module handles OHLCV data loading, validation, and provides
Walk-Forward methodology for train/validation/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """
    Handler for market OHLCV data loading and preprocessing.
    
    This class manages the loading, validation, and splitting of market data
    using Walk-Forward methodology to avoid look-ahead bias.
    """
    
    def __init__(
        self,
        data_path: str,
        date_column: str = 'date',
        ohlcv_columns: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the MarketDataHandler.
        
        Parameters
        ----------
        data_path : str
            Path to the OHLCV CSV file.
        date_column : str, default='date'
            Name of the date column in the CSV.
        ohlcv_columns : dict, optional
            Mapping of standard column names. If None, uses default mapping:
            {'open': 'open', 'high': 'high', 'low': 'low', 
             'close': 'close', 'volume': 'volume'}
        """
        self.data_path = Path(data_path)
        self.date_column = date_column
        
        if ohlcv_columns is None:
            self.ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        else:
            self.ohlcv_columns = ohlcv_columns
        
        self.data: Optional[pd.DataFrame] = None
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that the data file exists."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}"
            )
    
    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Parameters
        ----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. If None, loads from beginning.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. If None, loads to end.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with OHLCV data, indexed by datetime.
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
        
        # Parse date column
        if self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(
                df[self.date_column],
                format='mixed',
                errors='coerce'
            )
            df.set_index(self.date_column, inplace=True)
        else:
            raise ValueError(
                f"Date column '{self.date_column}' not found in data"
            )
        
        # Validate OHLCV columns
        required_cols = list(self.ohlcv_columns.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Filter by date range if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Data validation and cleaning
        df = self._clean_data(df)
        
        self.data = df
        logger.info(
            f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}"
        )
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data.
        
        Returns
        -------
        pd.DataFrame
            Cleaned data with validated values.
        """
        df = df.copy()
        
        # Remove rows with missing values
        initial_len = len(df)
        df.dropna(subset=list(self.ohlcv_columns.values()), inplace=True)
        if len(df) < initial_len:
            logger.warning(
                f"Removed {initial_len - len(df)} rows with missing values"
            )
        
        # Validate OHLC relationships
        invalid_rows = (
            (df[self.ohlcv_columns['high']] < df[self.ohlcv_columns['low']]) |
            (df[self.ohlcv_columns['high']] < df[self.ohlcv_columns['open']]) |
            (df[self.ohlcv_columns['high']] < df[self.ohlcv_columns['close']]) |
            (df[self.ohlcv_columns['low']] > df[self.ohlcv_columns['open']]) |
            (df[self.ohlcv_columns['low']] > df[self.ohlcv_columns['close']])
        )
        
        if invalid_rows.any():
            num_invalid = invalid_rows.sum()
            logger.warning(
                f"Found {num_invalid} rows with invalid OHLC relationships. "
                "Removing these rows."
            )
            df = df[~invalid_rows]
        
        # Remove negative or zero prices
        price_cols = [
            self.ohlcv_columns['open'],
            self.ohlcv_columns['high'],
            self.ohlcv_columns['low'],
            self.ohlcv_columns['close']
        ]
        for col in price_cols:
            df = df[df[col] > 0]
        
        # Remove negative volumes
        if self.ohlcv_columns['volume'] in df.columns:
            df = df[df[self.ohlcv_columns['volume']] >= 0]
        
        return df
    
    def get_walk_forward_splits(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        validation_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate Walk-Forward Expanding Window splits.
        
        This method implements Walk-Forward methodology where the training
        window expands over time, simulating realistic operational workflow.
        
        Parameters
        ----------
        train_start : str
            Training start date in 'YYYY-MM-DD' format.
        train_end : str
            Training end date in 'YYYY-MM-DD' format.
        test_start : str
            Test start date in 'YYYY-MM-DD' format.
        test_end : str
            Test end date in 'YYYY-MM-DD' format.
        validation_ratio : float, default=0.2
            Ratio of training data to use for validation.
        
        Returns
        -------
        train_df : pd.DataFrame
            Training data.
        validation_df : pd.DataFrame
            Validation data (last portion of training period).
        test_df : pd.DataFrame
            Test data (out-of-sample).
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        
        # Extract training data
        train_mask = (
            (self.data.index >= train_start_dt) &
            (self.data.index <= train_end_dt)
        )
        train_data = self.data[train_mask].copy()
        
        # Split training into train and validation
        val_start_idx = int(len(train_data) * (1 - validation_ratio))
        validation_data = train_data.iloc[val_start_idx:].copy()
        train_data = train_data.iloc[:val_start_idx].copy()
        
        # Extract test data (completely out-of-sample)
        test_mask = (
            (self.data.index >= test_start_dt) &
            (self.data.index <= test_end_dt)
        )
        test_data = self.data[test_mask].copy()
        
        logger.info(
            f"Walk-Forward splits: Train={len(train_data)}, "
            f"Validation={len(validation_data)}, Test={len(test_data)}"
        )
        
        return train_data, validation_data, test_data
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data.
        
        Returns
        -------
        pd.DataFrame
            The loaded OHLCV data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data
    
    def get_ohlcv_columns(self) -> Dict[str, str]:
        """
        Get the OHLCV column mapping.
        
        Returns
        -------
        dict
            Mapping of standard names to actual column names.
        """
        return self.ohlcv_columns.copy()

