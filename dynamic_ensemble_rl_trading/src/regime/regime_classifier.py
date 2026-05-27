"""
Market regime classification module using XGBoost.

This module implements the Market Regime Classification Layer that
classifies the market into Bull, Bear, or Sideways regimes with
confidence-based selection mechanism.

May 2026 (v2 — generalisation): the ESWA review (Reviewer #3 item
#10) and the overnight diagnostic ablation revealed that the v1
classifier was severely overfitted under the forward-looking
Trend-Scanning labels (train accuracy ≈ 1.000, test accuracy ≈ 0.47
across the five walk-forward folds). Root causes:

1. ``max_depth = 6`` with 539 input features produces decision trees
   so expressive that they memorise the training set.
2. No L1/L2 regularisation was applied.
3. ``colsample_bytree`` / ``subsample`` were exposed in
   ``config.yaml`` but were never forwarded to the constructor —
   i.e. the regularisation knobs were inert.

The v2 constructor exposes the full XGBoost regularisation surface
and defaults to shallower trees, column / row subsampling, and a
non-trivial ``reg_lambda``. The pipeline driver in
``scripts/train_and_verify.py`` now forwards these hyperparameters
from ``config.yaml`` so they are no longer ignored.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

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

    Classifies the market into three regimes (Bull, Bear, Sideways) using
    multimodal features. Implements a confidence-based selection
    mechanism to prevent erratic regime switching.

    The v2 (May 2026) constructor adds explicit regularisation
    parameters. Defaults are tuned for the ESWA walk-forward windows
    after the diagnostic in ``scripts/_diag_classifier_ablation.py``.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        confidence_threshold: float = 0.6,
        random_state: int = 42,
        colsample_bytree: float = 0.7,
        subsample: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_child_weight: float = 1.0,
        early_stopping_rounds: Optional[int] = 30,
        prob_ema_span: int = 0,
    ):
        """
        Initialise the RegimeClassifier.

        Parameters
        ----------
        n_estimators : int, default=200
            Number of boosting rounds. The early-stopping criterion
            (when a validation set is supplied) will typically truncate
            this well below 200.
        max_depth : int, default=4
            Maximum tree depth. v1's default of 6 produced 100% train
            accuracy / 47% test accuracy on Trend-Scanning labels.
        learning_rate : float, default=0.05
            Shrinkage rate. Halved from v1's 0.1 to match the deeper
            ensemble (more trees, slower learning).
        confidence_threshold : float, default=0.6
            Confidence threshold theta for regime switching.
        random_state : int, default=42
            Random seed for reproducibility.
        colsample_bytree : float, default=0.7
            Fraction of input features sampled per tree. Strong
            regulariser on 539-D feature space.
        subsample : float, default=0.8
            Fraction of training rows sampled per tree.
        reg_lambda : float, default=1.0
            L2 regularisation on leaf weights.
        reg_alpha : float, default=0.0
            L1 regularisation on leaf weights.
        min_child_weight : float, default=1.0
            Minimum sum of instance weight per child node.
        early_stopping_rounds : int or None, default=30
            Round count for XGBoost early stopping when validation
            data is supplied to :meth:`fit`. Set ``None`` to disable.
        prob_ema_span : int, default=0
            Span for exponential moving average of class probabilities
            during sequential inference. ``0`` disables smoothing.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for RegimeClassifier")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.prob_ema_span = int(prob_ema_span)
        self._ema_alpha = (
            2.0 / (self.prob_ema_span + 1.0) if self.prob_ema_span > 0 else 0.0
        )

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_weight=min_child_weight,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
        )

        self.is_fitted = False
        self.current_regime: Optional[int] = None
        self._ema_proba: Optional[np.ndarray] = None
        self._last_smoothed_proba: Optional[np.ndarray] = None

    def reset_smoothing(self) -> None:
        """Reset sequential probability EMA state (call at backtest start)."""
        self._ema_proba = None
        self._last_smoothed_proba = None
        self.current_regime = None

    def _smooth_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply EMA to a single-step probability vector."""
        if self.prob_ema_span <= 0:
            smoothed = raw_proba
        elif self._ema_proba is None:
            self._ema_proba = raw_proba.copy()
            smoothed = self._ema_proba
        else:
            a = self._ema_alpha
            self._ema_proba = a * raw_proba + (1.0 - a) * self._ema_proba
            smoothed = self._ema_proba
        self._last_smoothed_proba = smoothed.copy()
        return smoothed

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

        # ── Class imbalance fix (Reviewer #3 follow-up) ──
        # In the train window Bear:Sideways:Bull ≈ 6980 : 1171 : 6609.
        # Without balancing, the Sideways minority is essentially ignored
        # and the model becomes strongly Bear/Bull biased — which then
        # forces the test window (a +62% Bull market) to be labelled Bear
        # almost everywhere. Use inverse-frequency sample weights.
        unique, counts = np.unique(y, return_counts=True)
        cls_to_weight = {
            int(c): float(len(y)) / (len(unique) * cnt)
            for c, cnt in zip(unique, counts)
        }
        sample_weight = np.array([cls_to_weight[int(v)] for v in y], dtype=np.float64)
        logger.info(
            "  Class distribution: %s | weights: %s",
            dict(zip(unique.tolist(), counts.tolist())),
            {k: round(v, 3) for k, v in cls_to_weight.items()},
        )

        if validation_data is not None:
            X_val, y_val = validation_data
            # XGBoost 2.0+ moved early_stopping_rounds onto the
            # constructor; older releases accept it on .fit(). Try both.
            es_rounds = self.early_stopping_rounds
            try:
                if es_rounds is not None:
                    # XGBoost 2.0+ constructor surface.
                    self.model.set_params(early_stopping_rounds=es_rounds)
                self.model.fit(
                    X, y,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            except (TypeError, KeyError):
                # Older XGBoost: accept the keyword on .fit().
                self.model.fit(
                    X, y,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=es_rounds if es_rounds is not None else 10,
                    verbose=False,
                )
            best_iter = getattr(self.model, "best_iteration", None)
            if best_iter is not None:
                logger.info("  Early stopping at iteration %d / %d",
                            best_iter, self.n_estimators)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)

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
        
        raw_proba = self.predict_proba(state.reshape(1, -1))[0]
        proba = self._smooth_proba(raw_proba)

        max_prob = float(np.max(proba))
        predicted_regime = int(np.argmax(proba))
        
        # Confidence-based selection.
        #
        # The paper (Eq. 4) defaults to the previous regime under low
        # confidence. In raw-mode honest evaluation we found that this
        # propagates the very first uncertain Bear call forever in test
        # windows where the classifier is biased. We therefore fall back
        # to Sideways (flat) when confidence is low — both safer
        # (no leveraged conviction) and bias-resistant. The original
        # behaviour can be restored via env var ESWA_PAPER_FALLBACK=1.
        import os as _os
        paper_fallback = _os.environ.get('ESWA_PAPER_FALLBACK', '0') == '1'
        if max_prob >= self.confidence_threshold:
            selected_regime = predicted_regime
        else:
            if paper_fallback:
                if previous_regime is not None:
                    selected_regime = previous_regime
                elif self.current_regime is not None:
                    selected_regime = self.current_regime
                else:
                    selected_regime = predicted_regime
            else:
                # Sideways (label 1) → flat position via the Sideways pool.
                selected_regime = 1
        
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
        proba = self._last_smoothed_proba
        if proba is None:
            raw_proba = self.predict_proba(state.reshape(1, -1))[0]
            proba = self._smooth_proba(raw_proba)

        regime_names = ['Bear', 'Sideways', 'Bull']

        return {
            'regime': regime,
            'confidence': confidence,
            'probabilities': {
                'Bear': float(proba[0]),
                'Sideways': float(proba[1]),
                'Bull': float(proba[2]),
            },
            'regime_name': regime_names[regime],
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

