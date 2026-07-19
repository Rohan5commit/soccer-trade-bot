"""Configuration for soccer trade bot.

All values sourced from environment variables. No secrets hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Config:
    """Immutable configuration loaded from environment variables."""

    # --- Stream ---
    stream_url: str = ""
    match_id: str = ""

    # --- Mode ---
    dry_run: bool = True

    # --- OVH (training) ---
    ovh_app_key: str = ""
    ovh_app_secret: str = ""
    ovh_consumer_key: str = ""
    ovh_project_id: str = ""
    ovh_region: str = "BHS5"
    ovh_flavor_id: str = ""  # h100-360 flavor ID for the region

    # --- Lightning AI (inference) ---
    lightning_user_id: str = ""
    lightning_api_key: str = ""

    # --- Polymarket ---
    polymarket_api_key: str = ""
    polymarket_private_key: str = ""

    # --- Kalshi ---
    kalshi_api_key: str = ""
    kalshi_private_key: str = ""

    # --- Trading parameters ---
    min_bet_usd: float = 5.0
    max_bet_pct: float = 0.02
    kelly_fraction: float = 0.25
    edge_threshold: float = 0.05
    confidence_threshold: float = 0.70

    # --- OCR parameters ---
    ocr_confidence_threshold: float = 0.70
    score_roi: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) or None
    clock_roi: Optional[Tuple[int, int, int, int]] = None

    # --- Stream parameters ---
    stream_lag_max_seconds: int = 8
    frame_interval_seconds: int = 1

    # --- Kill switch ---
    api_error_threshold: int = 3
    api_error_window_seconds: int = 60
    drawdown_threshold_pct: float = 0.20
    ocr_fail_threshold: int = 5
    final_minutes_skip: int = 5  # no trades in last N minutes

    # --- Model paths ---
    xgb_model_path: str = "model/xgb_soccer.pkl"
    lgbm_model_path: str = "model/lgbm_soccer.pkl"
    calibrator_path: str = "model/calibrator.pkl"
    yolo_model_path: str = "model/yolov10_soccer.pt"
    clip_model_path: str = "model/clip_soccer"

    # --- Database ---
    db_path: str = "data/trades.db"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        def _bool(key: str, default: str = "true") -> bool:
            return os.environ.get(key, default).lower() in ("true", "1", "yes")

        def _float(key: str, default: float) -> float:
            return float(os.environ.get(key, str(default)))

        def _int(key: str, default: int) -> int:
            return int(os.environ.get(key, str(default)))

        def _tuple3(key: str) -> Optional[Tuple[int, int, int, int]]:
            val = os.environ.get(key, "")
            if not val:
                return None
            parts = [int(x.strip()) for x in val.split(",")]
            if len(parts) == 4:
                return (parts[0], parts[1], parts[2], parts[3])
            return None

        return cls(
            stream_url=os.environ.get("STREAM_URL", ""),
            match_id=os.environ.get("MATCH_ID", ""),
            dry_run=_bool("DRY_RUN", "true"),
            ovh_app_key=os.environ.get("OVH_APP_KEY", ""),
            ovh_app_secret=os.environ.get("OVH_APP_SECRET", ""),
            ovh_consumer_key=os.environ.get("OVH_CONSUMER_KEY", ""),
            ovh_project_id=os.environ.get("OVH_PROJECT_ID", ""),
            ovh_region=os.environ.get("OVH_REGION", "BHS5"),
            ovh_flavor_id=os.environ.get("OVH_FLAVOR_ID", ""),
            lightning_user_id=os.environ.get("LIGHTNING_USER_ID", ""),
            lightning_api_key=os.environ.get("LIGHTNING_API_KEY", ""),
            polymarket_api_key=os.environ.get("POLYMARKET_API_KEY", ""),
            polymarket_private_key=os.environ.get("POLYMARKET_PRIVATE_KEY", ""),
            kalshi_api_key=os.environ.get("KALSHI_API_KEY", ""),
            kalshi_private_key=os.environ.get("KALSHI_PRIVATE_KEY", ""),
            min_bet_usd=_float("MIN_BET_USD", 5.0),
            max_bet_pct=_float("MAX_BET_PCT", 0.02),
            kelly_fraction=_float("KELLY_FRACTION", 0.25),
            edge_threshold=_float("EDGE_THRESHOLD", 0.05),
            confidence_threshold=_float("CONFIDENCE_THRESHOLD", 0.70),
            ocr_confidence_threshold=_float("OCR_CONFIDENCE_THRESHOLD", 0.70),
            score_roi=_tuple3("SCORE_ROI"),
            clock_roi=_tuple3("CLOCK_ROI"),
            stream_lag_max_seconds=_int("STREAM_LAG_MAX", 8),
            frame_interval_seconds=_int("FRAME_INTERVAL", 1),
            api_error_threshold=_int("API_ERROR_THRESHOLD", 3),
            api_error_window_seconds=_int("API_ERROR_WINDOW", 60),
            drawdown_threshold_pct=_float("DRAWDOWN_THRESHOLD", 0.20),
            ocr_fail_threshold=_int("OCR_FAIL_THRESHOLD", 5),
            final_minutes_skip=_int("FINAL_MINUTES_SKIP", 5),
            xgb_model_path=os.environ.get("XGB_MODEL_PATH", "model/xgb_soccer.pkl"),
            lgbm_model_path=os.environ.get("LGBM_MODEL_PATH", "model/lgbm_soccer.pkl"),
            calibrator_path=os.environ.get("CALIBRATOR_PATH", "model/calibrator.pkl"),
            yolo_model_path=os.environ.get("YOLO_MODEL_PATH", "model/yolov10_soccer.pt"),
            clip_model_path=os.environ.get("CLIP_MODEL_PATH", "model/clip_soccer"),
            db_path=os.environ.get("DB_PATH", "data/trades.db"),
        )


def load_config() -> Config:
    """Convenience function to load config from environment."""
    return Config.from_env()
