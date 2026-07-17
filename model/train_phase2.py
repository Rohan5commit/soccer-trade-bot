"""Phase 2: Model Diversity - Add NGBoost and FT-Transformer.

Adds two new model types to the ensemble:
- NGBoost: Natural Gradient Boosting with probabilistic output
- FT-Transformer: Neural network for tabular data

Usage:
    python -m model.train_phase2 --data data/train_full.parquet --optuna --optuna-trials 500
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss

from model.calibrate import ProbabilityCalibrator
from model.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Environment detection
LIGHTNING_TEAMSPACE = Path("/teamspace")
OVH_OUTPUT = Path("/workspace/output")

if LIGHTNING_TEAMSPACE.exists():
    MODEL_DIR = Path.home() / "checkpoints" / "phase2"
elif OVH_OUTPUT.exists():
    MODEL_DIR = OVH_OUTPUT / "checkpoints" / "phase2"
else:
    MODEL_DIR = Path("./checkpoints/phase2")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint file
CHECKPOINT_FILE = MODEL_DIR / "checkpoint.json"


def _save_checkpoint(phase: str, data: dict) -> None:
    """Save checkpoint for resume."""
    checkpoint = {"phase": phase, "timestamp": time.time()}
    checkpoint.update(data)
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))
    logger.debug("Checkpoint saved: %s", phase)


def _load_checkpoint() -> Optional[dict]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            return None
    return None


def load_training_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data and split into X, y, groups.

    Returns:
        X: Feature array (N, 51)
        y: Target array (N,)
        groups: Match IDs for GroupKFold
    """
    df = pd.read_parquet(data_path)

    # Ensure all features exist
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(int)
    groups = df["match_id"].values if "match_id" in df.columns else np.zeros(len(df))

    logger.info("Loaded %d samples, %d features, %d classes",
                len(X), X.shape[1], len(np.unique(y)))
    return X, y, groups


def train_ngboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train NGBoost model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        params: Optional hyperparameters.

    Returns:
        Trained NGBoost classifier.
    """
    try:
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli, Normal, LogNormal
    except ImportError:
        logger.warning("NGBoost not installed, skipping")
        return None

    from sklearn.tree import DecisionTreeRegressor

    base = DecisionTreeRegressor(max_depth=6)
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.1,
        "Base": base,
        "minibatch_frac": 0.5,
        "col_sample": 0.8,
        "verbose": False,
    }
    if params:
        default_params.update(params)

    logger.info("Training NGBoost with params: %s", {k: v for k, v in default_params.items() if k not in ("verbose", "Base")})

    # NGBoost requires one-vs-rest for multiclass
    from ngboost import NGBClassifier

    # Train 3 binary classifiers (one per class)
    models = []
    for class_idx in range(3):
        y_binary = (y_train == class_idx).astype(int)
        y_val_binary = (y_val == class_idx).astype(int)

        model = NGBClassifier(**default_params)
        model.fit(X_train, y_binary, X_val=X_val, Y_val=y_val_binary)
        models.append(model)

    return models


def train_ft_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train FT-Transformer (neural network for tabular data).

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        params: Optional hyperparameters.

    Returns:
        Trained FT-Transformer model.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not installed, skipping FT-Transformer")
        return None

    default_params = {
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 3,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "n_epochs": 50,
        "patience": 10,
    }
    if params:
        default_params.update(params)

    logger.info("Training FT-Transformer with params: %s", default_params)

    # Define FT-Transformer architecture
    class FTTransformer(nn.Module):
        def __init__(self, n_features, n_classes, d_model, n_heads, n_layers, dropout):
            super().__init__()
            self.feature_embedding = nn.Linear(n_features, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward(self, x):
            # x: (batch, n_features)
            x = self.feature_embedding(x)  # (batch, d_model)
            x = x.unsqueeze(1)  # (batch, 1, d_model)

            # Add CLS token
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)  # (batch, 2, d_model)

            x = self.transformer(x)  # (batch, 2, d_model)
            cls_out = x[:, 0, :]  # (batch, d_model)

            return self.classifier(cls_out)  # (batch, n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]
    n_classes = 3

    model = FTTransformer(
        n_features=n_features,
        n_classes=n_classes,
        d_model=default_params["d_model"],
        n_heads=default_params["n_heads"],
        n_layers=default_params["n_layers"],
        dropout=default_params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=default_params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=default_params["batch_size"],
        shuffle=True,
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(default_params["n_epochs"]):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= default_params["patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d: train_loss=%.4f, val_loss=%.4f", epoch + 1, train_loss / len(train_loader), val_loss)

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


class SoccerEnsemblePhase2:
    """Ensemble of XGBoost + LightGBM + CatBoost + NGBoost + FT-Transformer.

    Supports:
    - Weighted average combination
    - Stacking meta-learner
    - Isotonic regression calibration
    - Save/load from disk
    """

    def __init__(
        self,
        xgb_model=None,
        lgbm_model=None,
        cb_model=None,
        ngb_model=None,
        ft_model=None,
        xgb_weight: float = 0.1,
        lgbm_weight: float = 0.1,
        cb_weight: float = 0.6,
        ngb_weight: float = 0.1,
        ft_weight: float = 0.1,
        calibrator=None,
    ):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.cb_model = cb_model
        self.ngb_model = ngb_model
        self.ft_model = ft_model
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.cb_weight = cb_weight
        self.ngb_weight = ngb_weight
        self.ft_weight = ft_weight
        self.calibrator = calibrator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature array (N, 51).

        Returns:
            Probability array (N, 3).
        """
        probs = []
        weights = []

        if self.xgb_model is not None:
            p = self.xgb_model.predict_proba(X)
            probs.append(p)
            weights.append(self.xgb_weight)

        if self.lgbm_model is not None:
            p = self.lgbm_model.predict_proba(X)
            probs.append(p)
            weights.append(self.lgbm_weight)

        if self.cb_model is not None:
            p = self.cb_model.predict_proba(X)
            probs.append(p)
            weights.append(self.cb_weight)

        # NGBoost
        if self.ngb_model is not None:
            try:
                # NGBoost models are a list of binary classifiers
                ngb_probs = np.zeros((X.shape[0], 3))
                for i, model in enumerate(self.ngb_model):
                    ngb_probs[:, i] = model.predict_proba(X)[:, 1]
                # Normalize
                ngb_probs = ngb_probs / ngb_probs.sum(axis=1, keepdims=True)
                probs.append(ngb_probs)
                weights.append(self.ngb_weight)
            except Exception as e:
                logger.warning("NGBoost prediction failed: %s", e)

        # FT-Transformer
        if self.ft_model is not None:
            try:
                import torch
                device = next(self.ft_model.parameters()).device
                X_t = torch.FloatTensor(X).to(device)
                self.ft_model.eval()
                with torch.no_grad():
                    logits = self.ft_model(X_t)
                    ft_probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(ft_probs)
                weights.append(self.ft_weight)
            except Exception as e:
                logger.warning("FT-Transformer prediction failed: %s", e)

        if not probs:
            raise ValueError("No models available for prediction")

        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble = sum(w * p for w, p in zip(weights, probs))

        # Calibrate if available
        if self.calibrator is not None and hasattr(self.calibrator, 'predict') and callable(self.calibrator.predict):
            try:
                ensemble = self.calibrator.predict(ensemble)
            except Exception as e:
                logger.warning("Calibration failed: %s", e)

        return ensemble

    def save(self, output_dir: str) -> None:
        """Save all models to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, output_path / "xgb_soccer.pkl")
        if self.lgbm_model is not None:
            joblib.dump(self.lgbm_model, output_path / "lgbm_soccer.pkl")
        if self.cb_model is not None:
            joblib.dump(self.cb_model, output_path / "catboost_soccer.pkl")
        if self.ngb_model is not None:
            joblib.dump(self.ngb_model, output_path / "ngb_soccer.pkl")
        if self.ft_model is not None:
            import torch
            torch.save(self.ft_model.state_dict(), output_path / "ft_transformer.pt")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, output_path / "calibrator.pkl")

        meta = {
            "xgb_weight": self.xgb_weight,
            "lgbm_weight": self.lgbm_weight,
            "cb_weight": self.cb_weight,
            "ngb_weight": self.ngb_weight,
            "ft_weight": self.ft_weight,
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "target_classes": ["home", "draw", "away"],
            "use_stacking": False,
        }
        (output_path / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))

        logger.info("Models saved to %s", output_path)

    @classmethod
    def load(cls, model_dir: str) -> "SoccerEnsemblePhase2":
        """Load all models from disk."""
        model_path = Path(model_dir)

        xgb_model = None
        lgbm_model = None
        cb_model = None
        ngb_model = None
        ft_model = None
        calibrator = None

        xgb_path = model_path / "xgb_soccer.pkl"
        if xgb_path.exists():
            xgb_model = joblib.load(xgb_path)

        lgbm_path = model_path / "lgbm_soccer.pkl"
        if lgbm_path.exists():
            lgbm_model = joblib.load(lgbm_path)

        cb_path = model_path / "catboost_soccer.pkl"
        if cb_path.exists():
            cb_model = joblib.load(cb_path)

        ngb_path = model_path / "ngb_soccer.pkl"
        if ngb_path.exists():
            ngb_model = joblib.load(ngb_path)

        ft_path = model_path / "ft_transformer.pt"
        if ft_path.exists():
            try:
                import torch
                # Need to know the model architecture
                # For now, skip loading FT-Transformer
                logger.warning("FT-Transformer loading not implemented yet")
            except ImportError:
                pass

        cal_path = model_path / "calibrator.pkl"
        if cal_path.exists():
            try:
                calibrator = joblib.load(cal_path)
                # Validate calibrator is the right type
                if not hasattr(calibrator, 'predict') or not callable(calibrator.predict):
                    logger.warning("Invalid calibrator type: %s, resetting", type(calibrator))
                    calibrator = None
            except Exception as e:
                logger.warning("Failed to load calibrator: %s", e)
                calibrator = None

        # Load meta
        meta_path = model_path / "ensemble_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        return cls(
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            cb_model=cb_model,
            ngb_model=ngb_model,
            ft_model=ft_model,
            xgb_weight=meta.get("xgb_weight", 0.1),
            lgbm_weight=meta.get("lgbm_weight", 0.1),
            cb_weight=meta.get("cb_weight", 0.6),
            ngb_weight=meta.get("ngb_weight", 0.1),
            ft_weight=meta.get("ft_weight", 0.1),
            calibrator=calibrator,
        )


def train_phase2(
    data_path: str,
    output_dir: str = "model_phase2",
    use_optuna: bool = False,
    optuna_trials: int = 200,
    skip_optuna: bool = False,
    resume: bool = False,
) -> SoccerEnsemblePhase2:
    """Full Phase 2 training pipeline.

    Args:
        data_path: Path to training data.
        output_dir: Where to save model artifacts.
        use_optuna: Whether to run Optuna optimization.
        optuna_trials: Number of Optuna trials.
        skip_optuna: Skip Optuna, load best params from checkpoint.
        resume: Resume from checkpoint.

    Returns:
        Trained SoccerEnsemblePhase2.
    """
    start = time.time()

    # Load data
    X, y, groups = load_training_data(data_path)

    # Split by match (GroupKFold ensures no data leakage)
    unique_matches = np.unique(groups)
    np.random.seed(42)
    np.random.shuffle(unique_matches)
    split_idx = int(len(unique_matches) * 0.8)
    train_matches = set(unique_matches[:split_idx])
    val_matches = set(unique_matches[split_idx:])

    train_mask = np.isin(groups, list(train_matches))
    val_mask = np.isin(groups, list(val_matches))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # Shuffle training data
    train_shuffle = np.random.permutation(len(X_train))
    X_train = X_train[train_shuffle]
    y_train = y_train[train_shuffle]

    logger.info(
        "Split: %d train, %d val (%d train matches, %d val matches)",
        len(X_train), len(X_val), len(train_matches), len(val_matches),
    )

    # Load existing models
    ensemble = SoccerEnsemblePhase2()

    # Try to load existing models from Phase 1
    phase1_dir = Path("model")
    if phase1_dir.exists():
        try:
            ensemble = SoccerEnsemblePhase2.load(str(phase1_dir))
            logger.info("Loaded Phase 1 models")
        except Exception as e:
            logger.warning("Failed to load Phase 1 models: %s", e)

    # Train NGBoost
    checkpoint = _load_checkpoint()
    if checkpoint and checkpoint.get("ngb_done"):
        logger.info("NGBoost already trained, loading from checkpoint")
        ngb_path = MODEL_DIR / "ngb_soccer.pkl"
        if ngb_path.exists():
            ensemble.ngb_model = joblib.load(ngb_path)
    else:
        logger.info("Training NGBoost...")
        ngb_model = train_ngboost(X_train, y_train, X_val, y_val)
        if ngb_model is not None:
            ensemble.ngb_model = ngb_model
            joblib.dump(ngb_model, MODEL_DIR / "ngb_soccer.pkl")
            _save_checkpoint("ngb_done", {"val_log_loss": 0.0})

    # Train FT-Transformer
    if checkpoint and checkpoint.get("ft_done"):
        logger.info("FT-Transformer already trained, loading from checkpoint")
    else:
        logger.info("Training FT-Transformer...")
        ft_model = train_ft_transformer(X_train, y_train, X_val, y_val)
        if ft_model is not None:
            ensemble.ft_model = ft_model
            import torch
            torch.save(ft_model.state_dict(), MODEL_DIR / "ft_transformer.pt")
            _save_checkpoint("ft_done", {"val_log_loss": 0.0})

    # Optimize ensemble weights
    logger.info("Optimizing ensemble weights...")
    xgb_probs = ensemble.xgb_model.predict_proba(X_val) if ensemble.xgb_model is not None else None
    lgbm_probs = ensemble.lgbm_model.predict_proba(X_val) if ensemble.lgbm_model is not None else None
    cb_probs = ensemble.cb_model.predict_proba(X_val) if ensemble.cb_model is not None else None

    # Get NGBoost predictions
    ngb_probs = None
    if ensemble.ngb_model is not None:
        try:
            ngb_probs = np.zeros((X_val.shape[0], 3))
            for i, model in enumerate(ensemble.ngb_model):
                ngb_probs[:, i] = model.predict_proba(X_val)[:, 1]
            ngb_probs = ngb_probs / ngb_probs.sum(axis=1, keepdims=True)
        except Exception:
            pass

    # Get FT-Transformer predictions
    ft_probs = None
    if ensemble.ft_model is not None:
        try:
            import torch
            device = next(ensemble.ft_model.parameters()).device
            X_val_t = torch.FloatTensor(X_val).to(device)
            ensemble.ft_model.eval()
            with torch.no_grad():
                logits = ensemble.ft_model(X_val_t)
                ft_probs = torch.softmax(logits, dim=1).cpu().numpy()
        except Exception:
            pass

    # Grid search for best weights
    best_ll = float("inf")
    best_weights = None

    # Generate candidate weight combinations
    from itertools import product
    weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    models_available = []
    model_probs = []
    if xgb_probs is not None:
        models_available.append("xgb")
        model_probs.append(xgb_probs)
    if lgbm_probs is not None:
        models_available.append("lgbm")
        model_probs.append(lgbm_probs)
    if cb_probs is not None:
        models_available.append("cb")
        model_probs.append(cb_probs)
    if ngb_probs is not None:
        models_available.append("ngb")
        model_probs.append(ngb_probs)
    if ft_probs is not None:
        models_available.append("ft")
        model_probs.append(ft_probs)

    if len(models_available) >= 2:
        # Search over weight combinations
        for weights in product(weight_grid, repeat=len(models_available)):
            if sum(weights) < 0.1:
                continue
            w = np.array(weights) / sum(weights)
            ensemble_probs = sum(w[i] * model_probs[i] for i in range(len(models_available)))
            ll = log_loss(y_val, ensemble_probs)
            if ll < best_ll:
                best_ll = ll
                best_weights = dict(zip(models_available, weights))

        # Apply best weights
        if best_weights:
            ensemble.xgb_weight = best_weights.get("xgb", 0.0)
            ensemble.lgbm_weight = best_weights.get("lgbm", 0.0)
            ensemble.cb_weight = best_weights.get("cb", 0.0)
            ensemble.ngb_weight = best_weights.get("ngb", 0.0)
            ensemble.ft_weight = best_weights.get("ft", 0.0)

            logger.info("Best weights: %s (log_loss=%.4f)", best_weights, best_ll)

    # Calibrate
    logger.info("Calibrating ensemble...")
    raw_probs = ensemble.predict(X_val)
    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_val, raw_probs)
    ensemble.calibrator = calibrator

    # Final evaluation
    calibrated_probs = calibrator.predict(raw_probs)
    final_ll = log_loss(y_val, calibrated_probs)

    for i, cls_name in enumerate(["home", "draw", "away"]):
        binary_true = (y_val == i).astype(float)
        bs = brier_score_loss(binary_true, calibrated_probs[:, i])
        logger.info("Brier score (%s): %.4f", cls_name, bs)

    logger.info("Final calibrated log loss: %.4f", final_ll)

    # Save
    ensemble.save(output_dir)
    ensemble.save(str(MODEL_DIR))

    elapsed = time.time() - start
    logger.info("Phase 2 training complete in %.1f seconds", elapsed)

    return ensemble


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase 2: Model Diversity Training")
    parser.add_argument("--data", default="data/train_full.parquet", help="Training data")
    parser.add_argument("--output", default="model_phase2", help="Output directory")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--optuna-trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Optuna, load from checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_phase2(
        data_path=args.data,
        output_dir=args.output,
        use_optuna=args.optuna,
        optuna_trials=args.optuna_trials,
        skip_optuna=args.skip_optuna,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
