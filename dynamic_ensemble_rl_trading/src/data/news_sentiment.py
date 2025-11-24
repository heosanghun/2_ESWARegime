"""
News sentiment feature extraction module.

This module processes news sentiment data and aggregates sentiment scores
over time windows to create sentiment features for the trading system.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import Optional, Dict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class NewsSentimentExtractor:
    """
    Extracts sentiment features from news data.
    
    Processes news CSV files, aggregates sentiment scores over 24-hour
    windows using simple moving averages and exponentially weighted
    moving averages to create sentiment feature vectors.
    """
    
    def __init__(
        self,
        news_path: str,
        aggregation_window_hours: int = 24,
        use_ewma: bool = True
    ):
        """
        Initialize the NewsSentimentExtractor.
        
        Parameters
        ----------
        news_path : str
            Path to the news CSV file.
        aggregation_window_hours : int, default=24
            Number of hours to aggregate sentiment over.
        use_ewma : bool, default=True
            Whether to use EWMA in addition to simple MA.
        """
        self.news_path = Path(news_path)
        self.aggregation_window_hours = aggregation_window_hours
        self.use_ewma = use_ewma
        
        self.news_data: Optional[pd.DataFrame] = None
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that the news file exists."""
        if not self.news_path.exists():
            raise FileNotFoundError(
                f"News file not found: {self.news_path}"
            )
    
    def load_news_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load news data from CSV file.
        
        Parameters
        ----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with news data and parsed sentiment.
        """
        logger.info(f"Loading news data from {self.news_path}")
        
        try:
            df = pd.read_csv(self.news_path)
        except Exception as e:
            logger.error(f"Error loading news CSV: {e}")
            raise
        
        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(
                df['date'],
                format='mixed',
                errors='coerce'
            )
        else:
            raise ValueError("'date' column not found in news data")
        
        # Filter by date range if specified
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # Parse sentiment column
        df = self._parse_sentiment(df)
        
        self.news_data = df
        logger.info(f"Loaded {len(df)} news articles")
        
        return df
    
    def _parse_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse sentiment column from string format to structured data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sentiment column as string.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with parsed sentiment columns.
        """
        if 'sentiment' not in df.columns:
            raise ValueError("'sentiment' column not found")
        
        # Parse sentiment strings
        sentiments = []
        for sent_str in df['sentiment']:
            try:
                if isinstance(sent_str, str):
                    # Parse string representation of dict
                    sent_dict = ast.literal_eval(sent_str)
                else:
                    sent_dict = sent_str
                
                sentiments.append({
                    'class': sent_dict.get('class', 'neutral'),
                    'polarity': sent_dict.get('polarity', 0.0),
                    'subjectivity': sent_dict.get('subjectivity', 0.0)
                })
            except Exception as e:
                logger.warning(f"Error parsing sentiment: {e}")
                sentiments.append({
                    'class': 'neutral',
                    'polarity': 0.0,
                    'subjectivity': 0.0
                })
        
        # Add parsed sentiment columns
        sentiment_df = pd.DataFrame(sentiments)
        df['sentiment_class'] = sentiment_df['class']
        df['sentiment_polarity'] = sentiment_df['polarity']
        df['sentiment_subjectivity'] = sentiment_df['subjectivity']
        
        return df
    
    def aggregate_sentiment_by_hour(
        self,
        target_timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores for each target timestamp.
        
        For each timestamp, aggregates sentiment from news articles
        within the aggregation window (e.g., last 24 hours).
        
        Parameters
        ----------
        target_timestamps : pd.DatetimeIndex
            Timestamps for which to aggregate sentiment.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with aggregated sentiment features, indexed by timestamp.
        """
        if self.news_data is None:
            raise ValueError("News data not loaded. Call load_news_data() first.")
        
        logger.info(
            f"Aggregating sentiment for {len(target_timestamps)} timestamps"
        )
        
        sentiment_features = []
        valid_timestamps = []
        
        for timestamp in target_timestamps:
            # Define time window
            window_start = timestamp - timedelta(hours=self.aggregation_window_hours)
            
            # Get news articles in the window
            window_news = self.news_data[
                (self.news_data['date'] >= window_start) &
                (self.news_data['date'] <= timestamp)
            ]
            
            if len(window_news) == 0:
                # No news in window, use neutral values
                features = {
                    'sentiment_polarity_mean': 0.0,
                    'sentiment_polarity_ewma': 0.0,
                    'sentiment_subjectivity_mean': 0.0,
                    'sentiment_subjectivity_ewma': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'news_count': 0
                }
            else:
                # Calculate aggregated features
                polarity_values = window_news['sentiment_polarity'].values
                subjectivity_values = window_news['sentiment_subjectivity'].values
                
                # Simple moving average
                polarity_mean = np.mean(polarity_values)
                subjectivity_mean = np.mean(subjectivity_values)
                
                # Exponentially weighted moving average
                if self.use_ewma and len(polarity_values) > 0:
                    # Use more recent articles with higher weight
                    weights = np.exp(np.linspace(-1, 0, len(polarity_values)))
                    weights = weights / weights.sum()
                    
                    polarity_ewma = np.average(polarity_values, weights=weights)
                    subjectivity_ewma = np.average(
                        subjectivity_values, weights=weights
                    )
                else:
                    polarity_ewma = polarity_mean
                    subjectivity_ewma = subjectivity_mean
                
                # Count sentiment classes
                positive_count = (window_news['sentiment_class'] == 'positive').sum()
                negative_count = (window_news['sentiment_class'] == 'negative').sum()
                neutral_count = (window_news['sentiment_class'] == 'neutral').sum()
                
                features = {
                    'sentiment_polarity_mean': polarity_mean,
                    'sentiment_polarity_ewma': polarity_ewma,
                    'sentiment_subjectivity_mean': subjectivity_mean,
                    'sentiment_subjectivity_ewma': subjectivity_ewma,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'news_count': len(window_news)
                }
            
            sentiment_features.append(features)
            valid_timestamps.append(timestamp)
        
        # Create DataFrame
        sentiment_df = pd.DataFrame(
            sentiment_features,
            index=valid_timestamps
        )
        
        logger.info(
            f"Created sentiment features for {len(sentiment_df)} timestamps"
        )
        
        return sentiment_df
    
    def get_sentiment_features(
        self,
        target_timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Get sentiment feature vector F_senti for target timestamps.
        
        Parameters
        ----------
        target_timestamps : pd.DatetimeIndex
            Timestamps for which to get sentiment features.
        
        Returns
        -------
        pd.DataFrame
            Sentiment feature vectors F_senti.
        """
        return self.aggregate_sentiment_by_hour(target_timestamps)

