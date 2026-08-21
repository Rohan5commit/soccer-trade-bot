"""Unified GameState dataclass.

Central data structure shared across all pipeline modules.
Contains 51 features derived from vision, OCR, and pre-match data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GameState:
    """Complete game state at a point in time.

    Attributes:
        match_id: Unique match identifier.
        home_team: Home team name.
        away_team: Away team name.
        clock_minutes: Current match clock (0-90+).
        stoppage_time: Added stoppage time minutes.
        home_score: Home team goals.
        away_score: Away team goals.
        ocr_reliable: Whether OCR confidence meets threshold.
        consecutive_consistent_reads: Number of consecutive matching OCR reads.
        timestamp: Unix timestamp of this snapshot.

    Live vision features:
        home_red_cards: Red cards for home team.
        away_red_cards: Red cards for away team.
        home_pressure_score: Attacking dominance (0.0-1.0).
        goals_in_last_10min: Goals scored in last 10 minutes.
        goals_last_15min: Goals scored in last 15 minutes.
        cards_last_15min: Cards shown in last 15 minutes.
        home_shots_on_target: Home shots on target.
        away_shots_on_target: Away shots on target.
        home_xg_running: Cumulative home xG.
        away_xg_running: Cumulative away xG.
        momentum_shift: xG delta in last 10 minutes.

    Pre-match features (injected at kickoff):
        home_elo: Home team ELO rating.
        away_elo: Away team ELO rating.
        home_form_pts: Points from last 5 matches.
        away_form_pts: Points from last 5 matches.
        h2h_home_winrate: Historical home win rate in H2H.
        is_home_game: Whether home team is actually home.
        referee_cards_per_game: Referee average cards per game.
        home_squad_value_EUR: Home squad market value.
        away_squad_value_EUR: Away squad market value.
        home_injuries_count: Home team injuries.
        away_injuries_count: Away team injuries.
        home_press_pct: Home pressing intensity (PPDA-derived).
        away_press_pct: Away pressing intensity.
        home_xg_last5: Home rolling xG form (last 5).
        away_xg_last5: Away rolling xG form.
        home_xga_last5: Home rolling xG conceded.
        away_xga_last5: Away rolling xG conceded.
        competition_tier: 1=UCL, 2=Big5 league, 3=lower.
        match_importance: 0.0-1.0 importance score.
        days_since_last_match_home: Days since home last match.
        days_since_last_match_away: Days since away last match.

    Event classification:
        event_label: Current detected event (goal, red card, etc.).
        event_confidence: Confidence of event detection (0.0-1.0).
    """

    # Identity
    match_id: str = ""
    home_team: str = ""
    away_team: str = ""

    # Clock
    clock_minutes: float = 0.0  # Float for stoppage time precision (e.g., 90.5 = 90+30s)
    stoppage_time: int = 0
    is_extra_time: bool = False

    # Score
    home_score: int = 0
    away_score: int = 0

    # OCR state
    ocr_reliable: bool = False
    consecutive_consistent_reads: int = 0
    timestamp: float = field(default_factory=time.time)

    # Vision features
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_pressure_score: float = 0.5
    goals_in_last_10min: int = 0
    goals_last_15min: int = 0
    cards_last_15min: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_xg_running: float = 0.0
    away_xg_running: float = 0.0
    momentum_shift: float = 0.0

    # Pre-match features
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_form_pts: int = 0
    away_form_pts: int = 0
    h2h_home_winrate: float = 0.5
    is_home_game: bool = True
    referee_cards_per_game: float = 3.5
    home_squad_value_EUR: float = 0.0
    away_squad_value_EUR: float = 0.0
    home_injuries_count: int = 0
    away_injuries_count: int = 0
    home_press_pct: float = 0.0
    away_press_pct: float = 0.0
    home_xg_last5: float = 0.0  # Actually cumulative match xG (not rolling 5-match)
    away_xg_last5: float = 0.0  # Actually cumulative match xG (not rolling 5-match)
    home_xga_last5: float = 0.0  # Actually away cumulative xG
    away_xga_last5: float = 0.0  # Actually home cumulative xG
    competition_tier: int = 2
    match_importance: float = 0.5
    days_since_last_match_home: int = 7
    days_since_last_match_away: int = 7

    # Event detection
    event_label: Optional[str] = None
    event_confidence: float = 0.0

    # Historical tracking for momentum
    _score_history: list = field(default_factory=list, repr=False)  # [(minute, score_diff), ...]
    _xg_history: list = field(default_factory=list, repr=False)  # [(minute, total_xg), ...]

    @property
    def score_diff(self) -> int:
        """Home score minus away score."""
        return self.home_score - self.away_score

    @property
    def score_diff_squared(self) -> float:
        """Non-linear blowout signal."""
        return float(self.score_diff ** 2)

    @property
    def elo_diff(self) -> float:
        """Home ELO minus away ELO."""
        return self.home_elo - self.away_elo

    @property
    def squad_value_ratio(self) -> float:
        """Home squad value / away squad value."""
        if self.away_squad_value_EUR <= 0:
            return 1.0
        return self.home_squad_value_EUR / self.away_squad_value_EUR

    @property
    def score_diff_x_time_remaining(self) -> float:
        """Critical interaction: score difference × time remaining.

        A team leading by 1 at 85 min is very different from 45 min.
        """
        return float(self.score_diff * (90 - self.clock_minutes))

    def to_feature_vector(self) -> Dict[str, float]:
        """Convert to feature dictionary for model input.

        Returns:
            Dictionary of 51 features matching the training schema.
        """
        return {
            # Live state
            "score_diff": float(self.score_diff),
            "clock_minutes": float(self.clock_minutes),
            "is_extra_time": float(self.is_extra_time),
            "home_red_cards": float(self.home_red_cards),
            "away_red_cards": float(self.away_red_cards),
            "home_pressure_score": self.home_pressure_score,
            "goals_in_last_10min": float(self.goals_in_last_10min),
            "home_shots_on_target": float(self.home_shots_on_target),
            "away_shots_on_target": float(self.away_shots_on_target),
            "home_xg_running": self.home_xg_running,
            "away_xg_running": self.away_xg_running,
            # Critical interaction
            "score_diff_x_time_remaining": self.score_diff_x_time_remaining,
            # Pre-match
            "home_elo": self.home_elo,
            "away_elo": self.away_elo,
            "elo_diff": self.elo_diff,
            "home_form_pts": float(self.home_form_pts),
            "away_form_pts": float(self.away_form_pts),
            "h2h_home_winrate": self.h2h_home_winrate,
            "is_home_game": float(self.is_home_game),
            "referee_cards_per_game": self.referee_cards_per_game,
            # Team quality
            "home_squad_value_EUR": self.home_squad_value_EUR,
            "away_squad_value_EUR": self.away_squad_value_EUR,
            "squad_value_ratio": self.squad_value_ratio,
            "home_injuries_count": float(self.home_injuries_count),
            "away_injuries_count": float(self.away_injuries_count),
            # Tactical
            "home_press_pct": self.home_press_pct,
            "away_press_pct": self.away_press_pct,
            "home_xg_last5": self.home_xg_last5,
            "away_xg_last5": self.away_xg_last5,
            "home_xga_last5": self.home_xga_last5,
            "away_xga_last5": self.away_xga_last5,
            # Match context
            "competition_tier": float(self.competition_tier),
            "match_importance": self.match_importance,
            "days_since_last_match_home": float(self.days_since_last_match_home),
            "days_since_last_match_away": float(self.days_since_last_match_away),
            # Momentum
            "goals_last_15min": float(self.goals_last_15min),
            "cards_last_15min": float(self.cards_last_15min),
            "score_diff_squared": self.score_diff_squared,
            "momentum_shift": self.momentum_shift,
            # v2 interaction features
            "xg_diff": float(self.home_xg_running - self.away_xg_running),
            "xg_total": float(self.home_xg_running + self.away_xg_running),
            "form_diff": float(self.home_form_pts - self.away_form_pts),
            "elo_xg_interaction": float(self.elo_diff * (self.home_xg_running - self.away_xg_running)),
            "pressure_x_time_remaining": float(self.home_pressure_score * max(90 - self.clock_minutes, 0)),
            "clock_normalized": float(self.clock_minutes / 90.0),
            "home_dominance": float((self.home_xg_running - self.away_xg_running) / max(self.home_xg_running + self.away_xg_running, 0.1)),
            "score_xg_consistent": 1.0 if (self.score_diff > 0 and self.home_xg_running > self.away_xg_running) or (self.score_diff < 0 and self.home_xg_running < self.away_xg_running) or (self.score_diff == 0 and abs(self.home_xg_running - self.away_xg_running) < 0.3) else 0.0,
            "late_game_state": 1.0 if (self.clock_minutes > 75 and self.score_diff != 0) else 0.0,
            "home_xg_per_minute": float(self.home_xg_running / max(self.clock_minutes, 1)),
            "away_xg_per_minute": float(self.away_xg_running / max(self.clock_minutes, 1)),
            "xg_momentum_ratio": float(self.momentum_shift / max(self.home_xg_running + self.away_xg_running, 0.1)),
        }

    def update_score(self, home: int, away: int) -> bool:
        """Update score with anti-flicker logic.

        Returns True if score actually changed.
        """
        if home == self.home_score and away == self.away_score:
            self.consecutive_consistent_reads += 1
            return False

        # Require 2 consistent reads before updating
        if (
            self.consecutive_consistent_reads < 2
            and self.home_score > 0
        ):
            self.consecutive_consistent_reads = 1
            return False

        old_diff = self.score_diff
        self.home_score = home
        self.away_score = away
        self.consecutive_consistent_reads = 1

        # Track score history with timestamp for windowed queries
        self._score_history.append((self.clock_minutes, self.score_diff))

        return self.score_diff != old_diff

    def goals_in_window(self, window_minutes: int) -> int:
        """Count goals scored in the last N minutes.

        Args:
            window_minutes: How many minutes back to look.

        Returns:
            Number of goals scored in the window.
        """
        if not self._score_history or self.clock_minutes <= 0:
            return 0

        start_minute = max(0, self.clock_minutes - window_minutes)
        goals = 0

        for i in range(1, len(self._score_history)):
            prev_minute, prev_diff = self._score_history[i - 1]
            curr_minute, curr_diff = self._score_history[i]

            # Only count goals within the time window
            if curr_minute >= start_minute and abs(curr_diff - prev_diff) > 0:
                goals += 1

        return goals

    def cards_in_window(self, window_minutes: int) -> int:
        """Count cards shown in the last N minutes (placeholder).

        Returns:
            Number of cards (currently returns 0 — needs event data).
        """
        # TODO: Track card events with timestamps like score history
        return 0

    def update_xg(self, home_xg: float, away_xg: float) -> None:
        """Update cumulative xG and compute momentum shift."""
        self.home_xg_running = home_xg
        self.away_xg_running = away_xg
        new_total = home_xg + away_xg

        self._xg_history.append((self.clock_minutes, new_total))

        # Compute momentum as delta xG over time window
        if len(self._xg_history) >= 2:
            prev_minute, prev_total = self._xg_history[-2]
            curr_minute, curr_total = self._xg_history[-1]
            delta_minutes = max(curr_minute - prev_minute, 1)
            self.momentum_shift = (curr_total - prev_total) / delta_minutes

    @property
    def is_active(self) -> bool:
        """Whether the match is still in progress."""
        return self.clock_minutes < 90 or self.is_extra_time

    def __repr__(self) -> str:
        return (
            f"GameState({self.home_team} {self.home_score} - {self.away_score} "
            f"{self.away_team} | {self.clock_minutes}' | reliable={self.ocr_reliable})"
        )
