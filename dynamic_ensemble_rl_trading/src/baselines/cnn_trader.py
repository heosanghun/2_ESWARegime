"""
CNN-based trader baseline implementation.

This baseline uses only CNN visual features (candlestick images)
to predict trading actions, without technical indicators or sentiment.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
import torch
import torch.nn as nn

from ..data.candlestick_generator import CandlestickGenerator

logger = logging.getLogger(__name__)


class CNNActionPredictor(nn.Module):
    """CNN-based action predictor."""
    
    def __init__(self, feature_dim: int = 512, num_actions: int = 5):
        """
        Initialize CNN action predictor.
        
        Parameters
        ----------
        feature_dim : int, default=512
            Dimension of CNN features (from ResNet-18).
        num_actions : int, default=5
            Number of trading actions.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)


class CNNTrader:
    """
    CNN-based trader baseline.
    
    This baseline:
    - Uses only CNN visual features (candlestick images)
    - No technical indicators or sentiment features
    - Direct action prediction from visual features
    """
    
    def __init__(
        self,
        image_size: int = 224,
        num_actions: int = 5,
        learning_rate: float = 1e-4,
        device: str = 'auto'
    ):
        """
        Initialize CNN trader.
        
        Parameters
        ----------
        image_size : int, default=224
            Size of candlestick images.
        num_actions : int, default=5
            Number of trading actions.
        learning_rate : float, default=1e-4
            Learning rate for training.
        device : str, default='auto'
            Device to use ('cpu', 'cuda', or 'auto').
        """
        self.image_size = image_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Initialize candlestick generator for feature extraction
        self.candlestick_generator = CandlestickGenerator(
            image_size=image_size,
            use_resnet=True
        )
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize CNN predictor
        feature_dim = 512  # ResNet-18 feature dimension
        self.model = CNNActionPredictor(feature_dim=feature_dim, num_actions=num_actions)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.is_fitted = False
        logger.info(f"Initialized CNN Trader (device={self.device})")
    
    def extract_features(self, ohlcv_data: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Extract CNN features from OHLCV data.
        
        Parameters
        ----------
        ohlcv_data : np.ndarray
            OHLCV data array.
        timestamps : np.ndarray
            Timestamps for each data point.
        
        Returns
        -------
        np.ndarray
            CNN features (N, 512).
        """
        features = []
        for i in range(len(timestamps)):
            # Generate candlestick image and extract features
            feature = self.candlestick_generator.extract_features(
                ohlcv_data, i
            )
            features.append(feature)
        
        return np.array(features)
    
    def fit(
        self,
        X_features: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 50,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Train the CNN trader.
        
        Parameters
        ----------
        X_features : np.ndarray
            CNN features (N, 512).
        y : np.ndarray
            Action labels (N,).
        batch_size : int, default=32
            Batch size for training.
        epochs : int, default=50
            Number of training epochs.
        validation_data : tuple, optional
            (X_val, y_val) for validation.
        """
        logger.info(f"Training CNN Trader on {len(X_features)} samples")
        
        X_tensor = torch.FloatTensor(X_features).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("CNN Trader training completed")
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict action from CNN features.
        
        Parameters
        ----------
        features : np.ndarray
            CNN features (512,).
        
        Returns
        -------
        int
            Action index (0-4).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(features_tensor)
            prediction = torch.argmax(outputs, dim=1).item()
        
        return int(prediction)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution.
        
        Parameters
        ----------
        features : np.ndarray
            CNN features (512,).
        
        Returns
        -------
        np.ndarray
            Probability distribution over actions.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(features_tensor)
            proba = outputs.cpu().numpy()[0]
        
        return proba
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved CNN Trader to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_fitted = True
        logger.info(f"Loaded CNN Trader from {path}")
