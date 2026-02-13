"""
XGBoost-based trader baseline implementation.

This baseline uses XGBoost classifier to directly predict trading actions
without reinforcement learning. Uses multimodal features.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.error("XGBoost not available. Please install xgboost.")

logger = logging.getLogger(__name__)


class XGBoostTrader:
    """
    XGBoost-based trader baseline.
    
    This baseline:
    - Uses XGBoost classifier to predict actions directly
    - No reinforcement learning
    - Uses multimodal features (visual, technical, sentiment)
    - Can predict either regime or action directly
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        predict_action: bool = True
    ):
        """
        Initialize XGBoost trader.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting rounds.
        max_depth : int, default=6
            Maximum tree depth.
        learning_rate : float, default=0.1
            Learning rate.
        random_state : int, default=42
            Random seed.
        predict_action : bool, default=True
            If True, predict actions directly (5 classes).
            If False, predict regime first then map to action.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for XGBoostTrader")
        
        self.predict_action = predict_action
        num_classes = 5 if predict_action else 3  # 5 actions or 3 regimes
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss'
        )
        
        self.is_fitted = False
        logger.info(f"Initialized XGBoost Trader (predict_action={predict_action})")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Train the XGBoost trader.
        
        Parameters
        ----------
        X : np.ndarray
            Training features (multimodal state vectors).
        y : np.ndarray
            Training labels (actions or regimes).
        validation_data : tuple, optional
            (X_val, y_val) for early stopping.
        """
        logger.info(f"Training XGBoost Trader on {len(X)} samples")
        
        if validation_data is not None:
            X_val, y_val = validation_data
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X, y, verbose=False)
        
        self.is_fitted = True
        logger.info("XGBoost Trader training completed")
    
    def predict(self, state: np.ndarray) -> int:
        """
        Predict action from state.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        
        Returns
        -------
        int
            Action index (0-4) or regime index (0-2).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        state_2d = state.reshape(1, -1)
        prediction = self.model.predict(state_2d)[0]
        return int(prediction)
    
    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        """
        Get prediction probability distribution.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        
        Returns
        -------
        np.ndarray
            Probability distribution over actions or regimes.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        state_2d = state.reshape(1, -1)
        proba = self.model.predict_proba(state_2d)[0]
        return proba
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        self.model.save_model(path)
        logger.info(f"Saved XGBoost Trader to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        self.model.load_model(path)
        self.is_fitted = True
        logger.info(f"Loaded XGBoost Trader from {path}")
