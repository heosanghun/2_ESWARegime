"""
Market regime classification module using XGBoost.

This module implements the Market Regime Classification Layer that
classifies the market into Bull, Bear, or Sideways regimes with
confidence-based selection mechanism.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.error("XGBoost not available. Please install xgboost.")

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    XGBoost-based market regime classifier.
    
    Classifies market into three regimes (Bull, Bear, Sideways) using
    multimodal features. Implements confidence-based selection mechanism
    to prevent erratic regime switching.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        confidence_threshold: float = 0.6,
        random_state: int = 42
    ):
        """
        Initialize the RegimeClassifier.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting rounds.
        max_depth : int, default=6
            Maximum tree depth.
        learning_rate : float, default=0.1
            Learning rate.
        confidence_threshold : float, default=0.6
            Confidence threshold theta for regime switching.
        random_state : int, default=42
            Random seed for reproducibility.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for RegimeClassifier")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss'
        )
        
        self.is_fitted = False
        self.current_regime: Optional[int] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Train the regime classifier.
        
        Parameters
        ----------
        X : np.ndarray
            Training features (unified state vectors).
        y : np.ndarray
            Training labels (0: Bear, 1: Sideways, 2: Bull).
        validation_data : tuple, optional
            (X_val, y_val) for early stopping.
        """
        logger.info(f"Training regime classifier on {len(X)} samples")
        
        if validation_data is not None:
            X_val, y_val = validation_data
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Regime classifier training completed")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability distribution over regimes.
        
        Parameters
        ----------
        X : np.ndarray
            Feature vectors (unified state vectors).
        
        Returns
        -------
        np.ndarray
            Probability distribution P(R|S_t) of shape (n_samples, 3).
            Columns: [Bear, Sideways, Bull]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime labels.
        
        Parameters
        ----------
        X : np.ndarray
            Feature vectors (unified state vectors).
        
        Returns
        -------
        np.ndarray
            Predicted regime labels (0: Bear, 1: Sideways, 2: Bull).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def select_regime_with_confidence(
        self,
        state: np.ndarray,
        previous_regime: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Select regime with confidence-based mechanism (Eq. 4).
        
        Implements the confidence-based selection:
        - If max(P(R|S_t)) >= theta: R_t = argmax(P(R|S_t))
        - Otherwise: R_t = R_{t-1}
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector S_t.
        previous_regime : int, optional
            Previous regime R_{t-1}. If None, uses stored current_regime.
        
        Returns
        -------
        regime : int
            Selected regime (0: Bear, 1: Sideways, 2: Bull).
        confidence : float
            Maximum probability (confidence) of the prediction.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get probability distribution
        proba = self.predict_proba(state.reshape(1, -1))[0]
        
        # Maximum probability (confidence)
        max_prob = np.max(proba)
        predicted_regime = np.argmax(proba)
        
        # Confidence-based selection
        if max_prob >= self.confidence_threshold:
            # High confidence: switch to predicted regime
            selected_regime = predicted_regime
        else:
            # Low confidence: keep previous regime
            if previous_regime is not None:
                selected_regime = previous_regime
            elif self.current_regime is not None:
                selected_regime = self.current_regime
            else:
                # No previous regime: use prediction anyway
                selected_regime = predicted_regime
                logger.warning(
                    "No previous regime available, using prediction despite "
                    f"low confidence ({max_prob:.3f})"
                )
        
        # Update current regime
        self.current_regime = selected_regime
        
        return selected_regime, max_prob
    
    def predict_with_confidence(
        self,
        state: np.ndarray,
        previous_regime: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Predict regime with confidence and return detailed information.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector S_t.
        previous_regime : int, optional
            Previous regime R_{t-1}.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'regime': Selected regime (0, 1, or 2)
            - 'confidence': Maximum probability
            - 'probabilities': Full probability distribution
            - 'regime_name': Name of selected regime
        """
        regime, confidence = self.select_regime_with_confidence(
            state, previous_regime
        )
        
        proba = self.predict_proba(state.reshape(1, -1))[0]
        
        regime_names = ['Bear', 'Sideways', 'Bull']
        
        return {
            'regime': regime,
            'confidence': confidence,
            'probabilities': {
                'Bear': proba[0],
                'Sideways': proba[1],
                'Bull': proba[2]
            },
            'regime_name': regime_names[regime]
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("No model to save. Model not fitted.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model.load_model(str(filepath))
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")

