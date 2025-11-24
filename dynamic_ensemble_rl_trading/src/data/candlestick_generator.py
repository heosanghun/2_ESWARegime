"""
Candlestick chart image generation and visual feature extraction.

This module generates candlestick chart images from OHLCV data and
extracts visual features using ResNet-18 CNN.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image, ImageDraw
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning(
        "PyTorch not available. Visual feature extraction will be limited."
    )

logger = logging.getLogger(__name__)


class CandlestickGenerator:
    """
    Generates candlestick chart images and extracts visual features.
    
    Creates 224x224 pixel candlestick images from OHLCV data and
    uses ResNet-18 (pre-trained on ImageNet) to extract visual features.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        lookback_hours: int = 60,
        use_resnet: bool = True
    ):
        """
        Initialize the CandlestickGenerator.
        
        Parameters
        ----------
        image_size : int, default=224
            Size of the generated image (image_size x image_size).
        lookback_hours : int, default=60
            Number of hours of OHLCV data to include in the image.
        use_resnet : bool, default=True
            Whether to use ResNet-18 for feature extraction.
        """
        self.image_size = image_size
        self.lookback_hours = lookback_hours
        self.use_resnet = use_resnet and TORCH_AVAILABLE
        
        self.resnet_model = None
        self.transform = None
        
        if self.use_resnet:
            self._load_resnet_model()
    
    def _load_resnet_model(self) -> None:
        """Load pre-trained ResNet-18 model for feature extraction."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping ResNet loading")
            return
        
        logger.info("Loading ResNet-18 model (ImageNet pre-trained)")
        
        # Load pre-trained ResNet-18
        self.resnet_model = models.resnet18(pretrained=True)
        self.resnet_model.eval()
        
        # Remove the final classification layer to get features
        self.resnet_model = torch.nn.Sequential(
            *list(self.resnet_model.children())[:-1]
        )
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("ResNet-18 model loaded successfully")
    
    def generate_candlestick_image(
        self,
        ohlcv_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        ohlcv_columns: Optional[dict] = None
    ) -> np.ndarray:
        """
        Generate a candlestick chart image for a specific timestamp.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data with datetime index.
        timestamp : pd.Timestamp
            Target timestamp for which to generate the image.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        
        Returns
        -------
        np.ndarray
            Image array of shape (image_size, image_size, 3).
        """
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # Get data up to the timestamp
        data_up_to_timestamp = ohlcv_data[
            ohlcv_data.index <= timestamp
        ].tail(self.lookback_hours)
        
        if len(data_up_to_timestamp) < 2:
            # Return blank image if insufficient data
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Create image
        image = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(image)
        
        # Calculate price range for scaling
        all_highs = data_up_to_timestamp[ohlcv_columns['high']].values
        all_lows = data_up_to_timestamp[ohlcv_columns['low']].values
        price_min = np.min(all_lows)
        price_max = np.max(all_highs)
        price_range = price_max - price_min
        if price_range == 0:
            price_range = 1
        
        # Calculate dimensions
        width = self.image_size
        height = self.image_size
        padding = 20
        chart_width = width - 2 * padding
        chart_height = height - 2 * padding
        
        # Draw candlesticks
        num_candles = len(data_up_to_timestamp)
        candle_width = chart_width / max(num_candles, 1)
        
        for i, (idx, row) in enumerate(data_up_to_timestamp.iterrows()):
            open_price = row[ohlcv_columns['open']]
            high_price = row[ohlcv_columns['high']]
            low_price = row[ohlcv_columns['low']]
            close_price = row[ohlcv_columns['close']]
            
            # Calculate positions
            x = padding + i * candle_width + candle_width / 2
            
            # Scale prices to chart coordinates
            high_y = padding + chart_height - (
                (high_price - price_min) / price_range * chart_height
            )
            low_y = padding + chart_height - (
                (low_price - price_min) / price_range * chart_height
            )
            open_y = padding + chart_height - (
                (open_price - price_min) / price_range * chart_height
            )
            close_y = padding + chart_height - (
                (close_price - price_min) / price_range * chart_height
            )
            
            # Determine color (green for up, red for down)
            is_up = close_price >= open_price
            color = (0, 200, 0) if is_up else (200, 0, 0)
            
            # Draw wick (high-low line)
            draw.line([x, high_y, x, low_y], fill=(100, 100, 100), width=1)
            
            # Draw body (open-close rectangle)
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)
            body_height = max(body_bottom - body_top, 1)
            
            draw.rectangle(
                [x - candle_width/2 + 1, body_top,
                 x + candle_width/2 - 1, body_bottom],
                fill=color,
                outline=color
            )
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    
    def extract_visual_features(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Extract visual features from candlestick image using ResNet-18.
        
        Parameters
        ----------
        image : np.ndarray
            Image array of shape (H, W, 3).
        
        Returns
        -------
        np.ndarray
            Feature vector of shape (512,) from ResNet-18.
        """
        if not self.use_resnet:
            # Fallback: flatten the image
            logger.warning(
                "ResNet not available, using flattened image as features"
            )
            return image.flatten()
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.resnet_model(image_tensor)
            features = features.squeeze().numpy()
        
        return features
    
    def process_timestamp(
        self,
        ohlcv_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        ohlcv_columns: Optional[dict] = None
    ) -> np.ndarray:
        """
        Generate image and extract features for a timestamp.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data with datetime index.
        timestamp : pd.Timestamp
            Target timestamp.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        
        Returns
        -------
        np.ndarray
            Visual feature vector F_visual.
        """
        image = self.generate_candlestick_image(
            ohlcv_data, timestamp, ohlcv_columns
        )
        features = self.extract_visual_features(image)
        
        return features
    
    def batch_process(
        self,
        ohlcv_data: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        ohlcv_columns: Optional[dict] = None,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Process multiple timestamps in batches.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data with datetime index.
        timestamps : pd.DatetimeIndex
            Timestamps to process.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        batch_size : int, default=32
            Batch size for processing.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with visual features, indexed by timestamp.
        """
        logger.info(f"Processing {len(timestamps)} timestamps in batches")
        
        features_list = []
        valid_timestamps = []
        
        for i, timestamp in enumerate(timestamps):
            try:
                features = self.process_timestamp(
                    ohlcv_data, timestamp, ohlcv_columns
                )
                features_list.append(features)
                valid_timestamps.append(timestamp)
            except Exception as e:
                logger.warning(
                    f"Error processing timestamp {timestamp}: {e}"
                )
                continue
        
        if not features_list:
            raise ValueError("No valid features extracted")
        
        # Create DataFrame
        feature_dim = len(features_list[0])
        feature_df = pd.DataFrame(
            features_list,
            index=valid_timestamps,
            columns=[f'visual_feat_{i}' for i in range(feature_dim)]
        )
        
        logger.info(
            f"Extracted visual features for {len(feature_df)} timestamps"
        )
        
        return feature_df

