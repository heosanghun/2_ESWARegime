"""
LSTM-based sequential regime classifier.

Uses normalized technical features over a sliding window to predict
p(R|s_t). Keeps the same predict_with_confidence contract as XGBoost.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

REGIME_NAMES = ("Bear", "Sideways", "Bull")


class _LSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class SequentialRegimeClassifier:
    """LSTM regime classifier with confidence routing + prob EMA."""

    MODEL_FILE = "model.pt"
    META_FILE = "model_meta.json"

    def __init__(
        self,
        sequence_window: int = 48,
        n_features: int = 19,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 5,
        confidence_threshold: float = 0.35,
        prob_ema_span: int = 0,
        random_state: int = 42,
        use_visual: bool = True,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SequentialRegimeClassifier")

        self.sequence_window = int(sequence_window)
        self.n_features = int(n_features)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.confidence_threshold = confidence_threshold
        self.prob_ema_span = int(prob_ema_span)
        self.random_state = random_state
        self.use_visual = bool(use_visual)
        self.device = torch.device(device)

        self._tech_start = 512 if self.use_visual else 0
        self._ema_alpha = (
            2.0 / (self.prob_ema_span + 1.0) if self.prob_ema_span > 0 else 0.0
        )

        torch.manual_seed(random_state)
        self.net = _LSTMNet(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.is_fitted = False
        self.current_regime: Optional[int] = None
        self._ema_proba: Optional[np.ndarray] = None
        self._last_smoothed_proba: Optional[np.ndarray] = None
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_window)

    def reset_smoothing(self) -> None:
        self._ema_proba = None
        self._last_smoothed_proba = None
        self.current_regime = None
        self._buffer.clear()

    def _smooth_proba(self, raw_proba: np.ndarray) -> np.ndarray:
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

    def extract_technical(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        return state[self._tech_start : self._tech_start + self.n_features]

    def _class_weights(self, y: np.ndarray) -> torch.Tensor:
        unique, counts = np.unique(y, return_counts=True)
        weights = np.ones(3, dtype=np.float32)
        for c, cnt in zip(unique, counts):
            weights[int(c)] = float(len(y)) / (len(unique) * cnt)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Train on sequence batches. X: (N, window, n_features)."""
        if X.ndim != 3:
            raise ValueError(f"Expected X.ndim==3, got {X.ndim}")

        self.n_features = int(X.shape[2])
        self.sequence_window = int(X.shape[1])
        self.net = _LSTMNet(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        Xtr = torch.tensor(X, dtype=torch.float32)
        ytr = torch.tensor(y, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if validation_data is not None:
            Xv, yv = validation_data
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xv, dtype=torch.float32),
                    torch.tensor(yv, dtype=torch.long),
                ),
                batch_size=self.batch_size,
                shuffle=False,
            )

        criterion = nn.CrossEntropyLoss(weight=self._class_weights(y))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state = None
        stale = 0

        logger.info(
            "Training LSTM classifier: %d samples, window=%d, features=%d",
            len(X),
            self.sequence_window,
            self.n_features,
        )

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            train_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                n_batches += 1

            train_loss /= max(n_batches, 1)
            val_loss = None
            if val_loader is not None:
                self.net.eval()
                vloss = 0.0
                vn = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        logits = self.net(xb)
                        vloss += float(criterion(logits, yb).item())
                        vn += 1
                val_loss = vloss / max(vn, 1)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self.net.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        logger.info(
                            "Early stop at epoch %d (val_loss=%.4f)", epoch, val_loss
                        )
                        break

            if epoch == 1 or epoch % 5 == 0 or val_loss is not None:
                msg = f"  epoch {epoch:02d}  train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                logger.info(msg)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.is_fitted = True

        logger.info("  Train accuracy: %.3f", self._accuracy(X, y))
        if validation_data is not None:
            logger.info("  Val accuracy: %.3f", self._accuracy(*validation_data))

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = np.argmax(self.predict_proba(X), axis=1)
        return float(np.mean(preds == y))

    def _forward_sequences(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.tensor(
                    X[start : start + self.batch_size], dtype=torch.float32
                )
                xb = xb.to(self.device)
                logits = self.net(xb)
                pb = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(pb)
        return np.vstack(probs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            return self._forward_sequences(X)
        if X.ndim == 2 and X.shape[1] == self.n_features:
            return self._forward_sequences(X.reshape(1, 1, -1))
        raise ValueError(f"Unsupported X shape {X.shape} for LSTM predict_proba")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _proba_from_buffer(self) -> np.ndarray:
        if len(self._buffer) == 0:
            return np.ones(3, dtype=np.float64) / 3.0

        seq = np.zeros((1, self.sequence_window, self.n_features), dtype=np.float64)
        buf = list(self._buffer)
        if len(buf) < self.sequence_window:
            pad = self.sequence_window - len(buf)
            seq[0, pad:, :] = 0.0
            seq[0, pad:, :] = np.stack(buf, axis=0)
        else:
            seq[0] = np.stack(buf[-self.sequence_window :], axis=0)
        return self._forward_sequences(seq)[0]

    def observe_state(self, state: np.ndarray) -> None:
        self._buffer.append(self.extract_technical(state))

    def select_regime_with_confidence(
        self,
        state: np.ndarray,
        previous_regime: Optional[int] = None,
    ) -> Tuple[int, float]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self.observe_state(state)
        raw_proba = self._proba_from_buffer()
        proba = self._smooth_proba(raw_proba)

        max_prob = float(np.max(proba))
        predicted_regime = int(np.argmax(proba))

        paper_fallback = os.environ.get("ESWA_PAPER_FALLBACK", "0") == "1"
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
                selected_regime = 1

        self.current_regime = selected_regime
        return selected_regime, max_prob

    def predict_with_confidence(
        self,
        state: np.ndarray,
        previous_regime: Optional[int] = None,
    ) -> Dict[str, any]:
        regime, confidence = self.select_regime_with_confidence(state, previous_regime)
        proba = self._last_smoothed_proba
        if proba is None:
            proba = np.ones(3) / 3.0

        return {
            "regime": regime,
            "confidence": confidence,
            "probabilities": {
                "Bear": float(proba[0]),
                "Sideways": float(proba[1]),
                "Bull": float(proba[2]),
            },
            "regime_name": REGIME_NAMES[regime],
        }

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("No model to save. Model not fitted.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")

        torch.save(self.net.state_dict(), str(path))
        meta = {
            "backend": "lstm",
            "sequence_window": self.sequence_window,
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "confidence_threshold": self.confidence_threshold,
            "prob_ema_span": self.prob_ema_span,
            "use_visual": self.use_visual,
            "random_state": self.random_state,
        }
        meta_path = path.parent / self.META_FILE
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("LSTM model saved → %s", path)

    def load_model(self, filepath: str) -> None:
        path = Path(filepath)
        if path.suffix != ".pt":
            candidate = path.with_suffix(".pt")
            if candidate.exists():
                path = candidate
            elif (path / self.MODEL_FILE).exists():
                path = path / self.MODEL_FILE
            else:
                path = path.with_suffix(".pt")

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        meta_path = path.parent / self.META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.sequence_window = int(meta.get("sequence_window", self.sequence_window))
            self.n_features = int(meta.get("n_features", self.n_features))
            self.hidden_dim = int(meta.get("hidden_dim", self.hidden_dim))
            self.num_layers = int(meta.get("num_layers", self.num_layers))
            self.dropout = float(meta.get("dropout", self.dropout))
            self.confidence_threshold = float(
                meta.get("confidence_threshold", self.confidence_threshold)
            )
            self.prob_ema_span = int(meta.get("prob_ema_span", self.prob_ema_span))
            self.use_visual = bool(meta.get("use_visual", self.use_visual))
            self._tech_start = 512 if self.use_visual else 0
            self._buffer = deque(maxlen=self.sequence_window)

        self.net = _LSTMNet(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.net.load_state_dict(torch.load(str(path), map_location=self.device))
        self.is_fitted = True
        logger.info("LSTM model loaded from %s", path)
