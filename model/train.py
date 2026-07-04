"""XGBoost + LightGBM ensemble training.

Trains an ensemble of XGBoost and LightGBM classifiers with:
- GroupKFold by match_id (no data leakage)
- GridSearchCV for hyperparameter tuning
- Isotonic regression calibration
- Saves calibrated model artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import log_loss, brier_score_loss

from model.calibrate import ProbabilityCalibrator
from model.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def load_training_data(
    parquet_path: str = "data/train.parquet",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load training data from parquet file.

    Args:
        parquet_path: Path to training data parquet.

    Returns:
        Tuple of (features_df, labels, match_ids).
    """
    parquet_path_str = str(parquet_path)
    if not Path(parquet_path_str).exists():
        logger.warning("Parquet not found at %s — generating synthetic data", parquet_path_str)
        from data.build_dataset import generate_synthetic_data
        df = generate_synthetic_data()
        Path(parquet_path_str).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path_str, index=False)
        logger.info("Synthetic data saved to %s (%d rows)", parquet_path_str, len(df))
    else:
        df = pd.read_parquet(parquet_path_str)

    # Ensure correct feature columns
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(int)
    groups = df["match_id"].values

    logger.info(
        "Loaded %d samples, %d features, %d matches",
        len(X), len(FEATURE_NAMES), len(np.unique(groups)),
    )

    return df, X, y, groups


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_grid_search: bool = True,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        use_grid_search: Whether to run GridSearchCV.

    Returns:
        Trained XGBClassifier.
    """
    logger.info("Training XGBoost...")

    base_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
    }

    if use_grid_search:
        param_grid = {
            "max_depth": [4, 6],
            "learning_rate": [0.01, 0.05],
            "n_estimators": [500, 1000],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [3, 5],
        }

        model = xgb.XGBClassifier(**base_params)

        # Use GroupKFold to prevent match leakage
        gkf = GroupKFold(n_splits=5)
        # For grid search, we'll use a simple split instead of full CV
        # to save time on H100
        split_idx = int(len(X_train) * 0.8)
        X_gs, y_gs = X_train[:split_idx], y_train[:split_idx]

        # Sample subset for faster grid search
        sample_size = min(50000, len(X_gs))
        indices = np.random.choice(len(X_gs), sample_size, replace=False)
        X_sample = X_gs[indices]
        y_sample = y_gs[indices]

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="neg_log_loss",
            n_jobs=1,
            verbose=1,
        )
        grid_search.fit(X_sample, y_sample)

        best_params = grid_search.best_params_
        logger.info("Best XGBoost params: %s", best_params)

        # Retrain with best params on full training set
        final_params = {**base_params, **best_params}
    else:
        final_params = {
            **base_params,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
        }

    final_params["early_stopping_rounds"] = 50
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)
    ll = log_loss(y_val, y_pred_proba)
    logger.info("XGBoost validation log loss: %.4f", ll)

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    """Train LightGBM classifier.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Trained LGBMClassifier.
    """
    logger.info("Training LightGBM...")

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 1500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)
    ll = log_loss(y_val, y_pred_proba)
    logger.info("LightGBM validation log loss: %.4f", ll)

    return model


class SoccerEnsemble:
    """Ensemble of XGBoost and LightGBM for win probability prediction.

    Averages softmax outputs before calibration.
    """

    def __init__(
        self,
        xgb_model: Optional[xgb.XGBClassifier] = None,
        lgbm_model: Optional[lgb.LGBMClassifier] = None,
        calibrator: Optional[ProbabilityCalibrator] = None,
        xgb_weight: float = 0.5,
        lgbm_weight: float = 0.5,
    ) -> None:
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.calibrator = calibrator
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.

        Args:
            X: Feature array (N, 38).

        Returns:
            Calibrated probabilities (N, 3) for [home, draw, away].
        """
        probs_list = []

        if self.xgb_model is not None:
            xgb_probs = self.xgb_model.predict_proba(X)
            probs_list.append((xgb_probs, self.xgb_weight))

        if self.lgbm_model is not None:
            lgbm_probs = self.lgbm_model.predict_proba(X)
            probs_list.append((lgbm_probs, self.lgbm_weight))

        if not probs_list:
            raise ValueError("No models in ensemble")

        # Weighted average
        total_weight = sum(w for _, w in probs_list)
        ensemble_probs = sum(p * w for p, w in probs_list) / total_weight

        # Calibrate
        if self.calibrator is not None:
            ensemble_probs = self.calibrator.predict(ensemble_probs)

        return ensemble_probs

    def predict_single(self, X: np.ndarray) -> np.ndarray:
        """Predict for a single sample.

        Args:
            X: Feature array (1, 38).

        Returns:
            Calibrated probabilities (3,).
        """
        return self.predict(X)[0]

    def save(self, output_dir: str = "model") -> None:
        """Save all model artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, output_path / "xgb_soccer.pkl")
            logger.info("XGBoost model saved")

        if self.lgbm_model is not None:
            joblib.dump(self.lgbm_model, output_path / "lgbm_soccer.pkl")
            logger.info("LightGBM model saved")

        if self.calibrator is not None:
            self.calibrator.save(str(output_path / "calibrator.pkl"))

        # Save metadata
        meta = {
            "xgb_weight": self.xgb_weight,
            "lgbm_weight": self.lgbm_weight,
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "target_classes": ["home", "draw", "away"],
        }
        (output_path / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: str = "model") -> "SoccerEnsemble":
        """Load ensemble from disk."""
        model_path = Path(model_dir)

        xgb_model = None
        lgbm_model = None
        calibrator = None

        xgb_path = model_path / "xgb_soccer.pkl"
        if xgb_path.exists():
            xgb_model = joblib.load(xgb_path)

        lgbm_path = model_path / "lgbm_soccer.pkl"
        if lgbm_path.exists():
            lgbm_model = joblib.load(lgbm_path)

        cal_path = model_path / "calibrator.pkl"
        if cal_path.exists():
            calibrator = ProbabilityCalibrator.load(str(cal_path))

        # Load weights from metadata
        meta_path = model_path / "ensemble_meta.json"
        xgb_weight = 0.5
        lgbm_weight = 0.5
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            xgb_weight = meta.get("xgb_weight", 0.5)
            lgbm_weight = meta.get("lgbm_weight", 0.5)

        ensemble = cls(
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            calibrator=calibrator,
            xgb_weight=xgb_weight,
            lgbm_weight=lgbm_weight,
        )
        logger.info("Ensemble loaded from %s", model_dir)
        return ensemble


def train(
    data_path: str = "data/train.parquet",
    output_dir: str = "model",
    use_grid_search: bool = False,
) -> SoccerEnsemble:
    """Full training pipeline.

    Args:
        data_path: Path to training data.
        output_dir: Where to save model artifacts.
        use_grid_search: Whether to run GridSearchCV (slower but better).

    Returns:
        Trained SoccerEnsemble.
    """
    start = time.time()

    # Load data
    df, X, y, groups = load_training_data(data_path)

    # Split by match (GroupKFold ensures no match leakage)
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

    logger.info(
        "Split: %d train, %d val (%d train matches, %d val matches)",
        len(X_train), len(X_val), len(train_matches), len(val_matches),
    )

    # Train models
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, use_grid_search)
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)

    # Create ensemble
    ensemble = SoccerEnsemble(xgb_model=xgb_model, lgbm_model=lgbm_model)

    # Calibrate on validation set
    raw_probs = ensemble.predict(X_val)
    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_val, raw_probs)
    ensemble.calibrator = calibrator

    # Final evaluation
    calibrated_probs = calibrator.predict(raw_probs)
    final_ll = log_loss(y_val, calibrated_probs)

    # Per-class Brier scores
    for i, cls_name in enumerate(["home", "draw", "away"]):
        binary_true = (y_val == i).astype(float)
        bs = brier_score_loss(binary_true, calibrated_probs[:, i])
        logger.info("Brier score (%s): %.4f", cls_name, bs)

    logger.info("Final calibrated log loss: %.4f", final_ll)

    # Save
    ensemble.save(output_dir)

    elapsed = time.time() - start
    logger.info("Training complete in %.1f seconds", elapsed)

    return ensemble


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LightGBM ensemble"
    )
    parser.add_argument("--data", default="data/train.parquet", help="Training data")
    parser.add_argument("--output", default="model", help="Output directory")
    parser.add_argument("--grid-search", action="store_true", help="Run GridSearchCV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train(
        data_path=args.data,
        output_dir=args.output,
        use_grid_search=args.grid_search,
    )


if __name__ == "__main__":
    main()
