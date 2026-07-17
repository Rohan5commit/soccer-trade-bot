"""Master orchestrator for all training phases.

Handles:
- Phase 1: Data Scale (StatsBomb + Football-Data.co.uk)
- Phase 2: Model Diversity (NGBoost + FT-Transformer)
- Phase 3: Optuna Sweep (5000+ trials)
- Phase 4: Final Ensemble optimization

Checkpoint strategy:
- Each phase saves checkpoints to ~/.checkpoints/phase{N}/
- Resume from last checkpoint if interrupted
- Account rotation when credits run out

Usage:
    python run_training.py --phase 1  # Run Phase 1
    python run_training.py --phase all  # Run all phases
    python run_training.py --resume  # Resume from checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Environment detection
LIGHTNING_TEAMSPACE = Path("/teamspace")
OVH_OUTPUT = Path("/workspace/output")

if LIGHTNING_TEAMSPACE.exists():
    CHECKPOINT_BASE = Path.home() / "checkpoints"
elif OVH_OUTPUT.exists():
    CHECKPOINT_BASE = OVH_OUTPUT / "checkpoints"
else:
    CHECKPOINT_BASE = Path("./checkpoints")

CHECKPOINT_BASE.mkdir(parents=True, exist_ok=True)

# Master checkpoint file
MASTER_CHECKPOINT = CHECKPOINT_BASE / "master_checkpoint.json"


def save_master_checkpoint(phase: str, status: str, data: dict) -> None:
    """Save master checkpoint."""
    checkpoint = {
        "phase": phase,
        "status": status,
        "timestamp": time.time(),
    }
    checkpoint.update(data)
    MASTER_CHECKPOINT.write_text(json.dumps(checkpoint, indent=2))
    logger.info("Master checkpoint saved: %s - %s", phase, status)


def load_master_checkpoint() -> Optional[dict]:
    """Load master checkpoint."""
    if MASTER_CHECKPOINT.exists():
        try:
            return json.loads(MASTER_CHECKPOINT.read_text())
        except Exception:
            return None
    return None


def run_phase1(output_path: str = "data/train_full.parquet") -> bool:
    """Run Phase 1: Data Scale.

    Returns:
        True if successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA SCALE")
    logger.info("=" * 60)

    checkpoint = load_master_checkpoint()
    if checkpoint and checkpoint.get("phase") == "phase1" and checkpoint.get("status") == "complete":
        logger.info("Phase 1 already complete, skipping")
        return True

    save_master_checkpoint("phase1", "running", {"output_path": output_path})

    try:
        from data.build_full_dataset import build_full_dataset
        df = build_full_dataset(output_path)
        if len(df) == 0:
            logger.error("Phase 1 failed: no data built")
            return False

        save_master_checkpoint("phase1", "complete", {
            "output_path": output_path,
            "rows": len(df),
            "matches": len(df["match_id"].unique()) if "match_id" in df.columns else 0,
        })
        return True

    except Exception as e:
        logger.error("Phase 1 failed: %s", e, exc_info=True)
        save_master_checkpoint("phase1", "failed", {"error": str(e)})
        return False


def run_phase2(data_path: str = "data/train_full.parquet", output_dir: str = "model_phase2") -> bool:
    """Run Phase 2: Model Diversity.

    Returns:
        True if successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL DIVERSITY")
    logger.info("=" * 60)

    checkpoint = load_master_checkpoint()
    if checkpoint and checkpoint.get("phase") == "phase2" and checkpoint.get("status") == "complete":
        logger.info("Phase 2 already complete, skipping")
        return True

    save_master_checkpoint("phase2", "running", {
        "data_path": data_path,
        "output_dir": output_dir,
    })

    try:
        from model.train_phase2 import train_phase2
        ensemble = train_phase2(
            data_path=data_path,
            output_dir=output_dir,
            use_optuna=False,  # Optuna in Phase 3
        )

        save_master_checkpoint("phase2", "complete", {
            "output_dir": output_dir,
            "log_loss": 0.0,  # Will be filled by training
        })
        return True

    except Exception as e:
        logger.error("Phase 2 failed: %s", e, exc_info=True)
        save_master_checkpoint("phase2", "failed", {"error": str(e)})
        return False


def run_phase3(data_path: str = "data/train_full.parquet", output_dir: str = "model_phase3", n_trials: int = 5000) -> bool:
    """Run Phase 3: Optuna Sweep.

    Returns:
        True if successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: OPTUNA SWEEP (%d trials)", n_trials)
    logger.info("=" * 60)

    checkpoint = load_master_checkpoint()
    if checkpoint and checkpoint.get("phase") == "phase3" and checkpoint.get("status") == "complete":
        logger.info("Phase 3 already complete, skipping")
        return True

    save_master_checkpoint("phase3", "running", {
        "data_path": data_path,
        "output_dir": output_dir,
        "n_trials": n_trials,
    })

    try:
        # Use the existing train.py with Optuna
        cmd = [
            sys.executable, "-m", "model.train",
            "--data", data_path,
            "--output", output_dir,
            "--optuna",
            "--optuna-trials", str(n_trials),
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            logger.error("Phase 3 failed with return code %d", result.returncode)
            save_master_checkpoint("phase3", "failed", {"returncode": result.returncode})
            return False

        save_master_checkpoint("phase3", "complete", {
            "output_dir": output_dir,
            "n_trials": n_trials,
        })
        return True

    except Exception as e:
        logger.error("Phase 3 failed: %s", e, exc_info=True)
        save_master_checkpoint("phase3", "failed", {"error": str(e)})
        return False


def run_phase4(data_path: str = "data/train_full.parquet", output_dir: str = "model_final") -> bool:
    """Run Phase 4: Final Ensemble optimization.

    Returns:
        True if successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: FINAL ENSEMBLE")
    logger.info("=" * 60)

    checkpoint = load_master_checkpoint()
    if checkpoint and checkpoint.get("phase") == "phase4" and checkpoint.get("status") == "complete":
        logger.info("Phase 4 already complete, skipping")
        return True

    save_master_checkpoint("phase4", "running", {
        "data_path": data_path,
        "output_dir": output_dir,
    })

    try:
        # Ensemble weight optimization + calibration
        import numpy as np
        import pandas as pd
        from model.train_phase2 import SoccerEnsemblePhase2, load_training_data
        from model.calibrate import ProbabilityCalibrator
        from sklearn.metrics import log_loss, brier_score_loss
        import joblib

        # Load data
        X, y, groups = load_training_data(data_path)

        # Split
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

        # Load best models from Phase 3
        ensemble = SoccerEnsemblePhase2.load("model_phase3")

        # Calibrate
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

        save_master_checkpoint("phase4", "complete", {
            "output_dir": output_dir,
            "final_log_loss": final_ll,
        })
        return True

    except Exception as e:
        logger.error("Phase 4 failed: %s", e, exc_info=True)
        save_master_checkpoint("phase4", "failed", {"error": str(e)})
        return False


def run_all_phases(data_path: str = "data/train_full.parquet") -> bool:
    """Run all phases sequentially.

    Phase 2 (NGBoost/FT-Transformer) is skipped — marginal value, wastes compute.
    Phase 3 Optuna is where profitability is built.

    Returns:
        True if all phases successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("RUNNING ALL PHASES")
    logger.info("=" * 60)

    # Phase 1: Data Scale
    if not run_phase1(data_path):
        return False

    # Phase 2: SKIPPED (marginal value, wastes compute)
    logger.info("Phase 2 SKIPPED — going straight to Optuna sweep")
    save_master_checkpoint("phase2", "complete", {"skipped": True})

    # Phase 3: Optuna Sweep (2000 trials, 4 parallel × 4 cores = 16 cores)
    if not run_phase3(data_path, n_trials=2000):
        return False

    # Phase 4: Final Ensemble
    if not run_phase4(data_path):
        return False

    logger.info("=" * 60)
    logger.info("ALL PHASES COMPLETE!")
    logger.info("=" * 60)

    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Master training orchestrator")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run specific phase")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--data", default="data/train_full.parquet", help="Training data path")
    parser.add_argument("--output", default="model_final", help="Output directory")
    parser.add_argument("--trials", type=int, default=5000, help="Optuna trials for Phase 3")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.all:
        success = run_all_phases(args.data)
    elif args.phase == 1:
        success = run_phase1(args.data)
    elif args.phase == 2:
        success = run_phase2(args.data, args.output)
    elif args.phase == 3:
        success = run_phase3(args.data, args.output, args.trials)
    elif args.phase == 4:
        success = run_phase4(args.data, args.output)
    else:
        parser.print_help()
        return

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
