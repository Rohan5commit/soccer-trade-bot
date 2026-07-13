"""Feature engineering from GameState.

Converts raw GameState snapshots into 38-feature vectors for model training/inference.
Handles time-decay interactions, momentum computation, and pre-match enrichment.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.game_state import GameState


# Feature names in canonical order (must match training schema)
FEATURE_NAMES: List[str] = [
    # Live state
    "score_diff",
    "clock_minutes",
    "is_extra_time",
    "home_red_cards",
    "away_red_cards",
    "home_pressure_score",
    "goals_in_last_10min",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_xg_running",
    "away_xg_running",
    # Critical interaction
    "score_diff_x_time_remaining",
    # Pre-match
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_pts",
    "away_form_pts",
    "h2h_home_winrate",
    "is_home_game",
    "referee_cards_per_game",
    # Team quality
    "home_squad_value_EUR",
    "away_squad_value_EUR",
    "squad_value_ratio",
    "home_injuries_count",
    "away_injuries_count",
    # Tactical
    "home_press_pct",
    "away_press_pct",
    "home_xg_last5",
    "away_xg_last5",
    "home_xga_last5",
    "away_xga_last5",
    # Match context
    "competition_tier",
    "match_importance",
    "days_since_last_match_home",
    "days_since_last_match_away",
    # Momentum
    "goals_last_15min",
    "cards_last_15min",
    "score_diff_squared",
    "momentum_shift",
    # === NEW FEATURES (v2) ===
    "xg_diff",                  # home xG - away xG
    "xg_total",                 # total xG in match
    "form_diff",                # home form - away form
    "elo_xg_interaction",       # elo_diff * xg_diff
    "pressure_x_time_remaining",# pressure * time_remaining
    "clock_normalized",         # clock / 90
    "home_dominance",           # (home_xg - away_xg) / (home_xg + away_xg + 0.1)
    "score_xg_consistent",      # 1 if score and xG agree on leader
    "late_game_state",          # 1 if clock > 75 and score_diff != 0
    "home_xg_per_minute",       # home_xg / max(clock, 1)
    "away_xg_per_minute",       # away_xg / max(clock, 1)
    "xg_momentum_ratio",        # (home_xg_recent - away_xg_recent) / (total_xg + 0.1)
]

assert len(FEATURE_NAMES) == 51, f"Expected 51 features, got {len(FEATURE_NAMES)}"


def state_to_vector(state: GameState) -> Dict[str, float]:
    """Convert GameState to feature dictionary.

    Args:
        state: Current game state snapshot.

    Returns:
        Dictionary mapping feature names to float values.
    """
    return state.to_feature_vector()


def state_to_array(state: GameState) -> np.ndarray:
    """Convert GameState to numpy feature array (1, 38).

    Args:
        state: Current game state snapshot.

    Returns:
        numpy array of shape (1, 38) in canonical feature order.
    """
    vec = state.to_feature_vector()
    return np.array([[vec[name] for name in FEATURE_NAMES]], dtype=np.float32)


def batch_states_to_array(states: List[GameState]) -> np.ndarray:
    """Convert multiple GameStates to numpy array (N, 39).

    Args:
        states: List of game state snapshots.

    Returns:
        numpy array of shape (N, 39).
    """
    return np.vstack([state_to_array(s) for s in states])


def compute_momentum(
    xg_history: List[Tuple[float, float]],
    window: int = 5,
) -> float:
    """Compute momentum shift from xG history.

    Momentum = total xG in last N minutes minus total xG in previous N minutes.

    Args:
        xg_history: List of (home_xg, away_xg) tuples over time.
        window: Number of time steps to compare.

    Returns:
        Momentum shift value (positive = home momentum).
    """
    if len(xg_history) < window * 2:
        return 0.0

    recent = xg_history[-window:]
    previous = xg_history[-window * 2 : -window]

    recent_total = sum(h + a for h, a in recent)
    previous_total = sum(h + a for h, a in previous)

    return recent_total - previous_total


def compute_goals_in_window(
    score_history: List[int],
    window_minutes: int,
    current_minute: int,
) -> int:
    """Count goals scored in a time window.

    Args:
        score_history: List of (minute, score_diff) tuples.
        window_minutes: How many minutes back to look.
        current_minute: Current match clock.

    Returns:
        Number of goals scored in the window.
    """
    start_minute = max(0, current_minute - window_minutes)
    goals = 0

    for i in range(1, len(score_history)):
        prev_diff = score_history[i - 1]
        curr_diff = score_history[i]
        if abs(curr_diff - prev_diff) > 0:
            goals += 1

    return goals


def create_training_snapshot(
    clock: int,
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
    home_sot: int = 0,
    away_sot: int = 0,
    home_red: int = 0,
    away_red: int = 0,
    pressure: float = 0.5,
    # Pre-match features
    home_elo: float = 1500.0,
    away_elo: float = 1500.0,
    home_form: int = 0,
    away_form: int = 0,
    h2h_winrate: float = 0.5,
    is_home: bool = True,
    ref_cards: float = 3.5,
    home_value: float = 0.0,
    away_value: float = 0.0,
    home_injuries: int = 0,
    away_injuries: int = 0,
    home_press: float = 0.0,
    away_press: float = 0.0,
    home_xg5: float = 0.0,
    away_xg5: float = 0.0,
    home_xga5: float = 0.0,
    away_xga5: float = 0.0,
    comp_tier: int = 2,
    importance: float = 0.5,
    days_home: int = 7,
    days_away: int = 7,
    # Target
    final_result: Optional[int] = None,
) -> Dict[str, float]:
    """Create a training snapshot from raw values.

    Args:
        clock: Match minute (0-90+).
        home_goals, away_goals: Current score.
        home_xg, away_xg: Cumulative xG.
        ... (other features)
        final_result: 0=home win, 1=draw, 2=away win (None for inference).

    Returns:
        Dictionary of 38 features.
    """
    score_diff = home_goals - away_goals

    return {
        "score_diff": float(score_diff),
        "clock_minutes": float(clock),
        "is_extra_time": float(clock > 90),
        "home_red_cards": float(home_red),
        "away_red_cards": float(away_red),
        "home_pressure_score": pressure,
        "goals_in_last_10min": 0.0,  # Would need historical data
        "home_shots_on_target": float(home_sot),
        "away_shots_on_target": float(away_sot),
        "home_xg_running": home_xg,
        "away_xg_running": away_xg,
        "score_diff_x_time_remaining": float(score_diff * (90 - clock)),
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_form_pts": float(home_form),
        "away_form_pts": float(away_form),
        "h2h_home_winrate": h2h_winrate,
        "is_home_game": float(is_home),
        "referee_cards_per_game": ref_cards,
        "home_squad_value_EUR": home_value,
        "away_squad_value_EUR": away_value,
        "squad_value_ratio": home_value / max(away_value, 1.0),
        "home_injuries_count": float(home_injuries),
        "away_injuries_count": float(away_injuries),
        "home_press_pct": home_press,
        "away_press_pct": away_press,
        "home_xg_last5": home_xg5,
        "away_xg_last5": away_xg5,
        "home_xga_last5": home_xga5,
        "away_xga_last5": away_xga5,
        "competition_tier": float(comp_tier),
        "match_importance": importance,
        "days_since_last_match_home": float(days_home),
        "days_since_last_match_away": float(days_away),
        "goals_last_15min": 0.0,
        "cards_last_15min": 0.0,
        "score_diff_squared": float(score_diff ** 2),
        "momentum_shift": 0.0,
    }
