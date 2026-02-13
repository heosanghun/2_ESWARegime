"""
Model 2: Confidence-based Selection Removed

This ablation model removes the confidence-based selection mechanism.
Always switches to predicted regime immediately, regardless of confidence.
Expected Sharpe Ratio: ~1.41
"""

import numpy as np
from typing import Dict, Optional
import logging

from ..regime.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


class NoConfidenceSelection(RegimeClassifier):
    """
    Ablation Model 2: Confidence-based Selection Removed.
    
    This model:
    - Keeps regime classification
    - Removes confidence threshold check
    - Always switches to predicted regime immediately
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize Model 2.
        
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
        """
        # Initialize parent but set confidence_threshold to 0.0 (always switch)
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            confidence_threshold=0.0,  # Always switch
            random_state=random_state
        )
        logger.info("Initialized Model 2: Confidence-based Selection Removed")
    
    def predict_with_confidence(
        self,
        state: np.ndarray,
        previous_regime: Optional[int] = None
    ) -> Dict:
        """
        Predict regime without confidence check (always switch).
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        previous_regime : int, optional
            Previous regime index.
        
        Returns
        -------
        dict
            Regime prediction result (always switches).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        state_2d = state.reshape(1, -1)
        proba = self.model.predict_proba(state_2d)[0]
        
        # Always switch to predicted regime (no confidence check)
        predicted_regime = int(np.argmax(proba))
        max_prob = float(np.max(proba))
        
        regime_names = ['Bear', 'Sideways', 'Bull']
        
        return {
            'regime': predicted_regime,
            'regime_name': regime_names[predicted_regime],
            'confidence': max_prob,
            'probabilities': proba.tolist()
        }
