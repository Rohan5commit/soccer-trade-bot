"""Shared live-match data types used across data-source clients.

Kept free of any single provider's client so ESPN, football-data,
SportScore, and BSD can share the same state shape.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MatchEvent:
    """A match event (goal, card, substitution, var)."""
    event_type: str
    detail: str
    team_id: int
    team_name: str
    player_name: str
    minute: int
    comments: Optional[str] = None


@dataclass
class MatchStats:
    """Match statistics for a team."""
    team_id: int
    team_name: str
    possession: float = 0.0
    shots_on: int = 0
    shots_off: int = 0
    fouls: int = 0
    corners: int = 0
    offsides: int = 0


@dataclass
class LiveMatchState:
    """Complete live match state from any data source."""
    fixture_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    clock_minutes: float  # 0-90+
    status: str  # "NS", "1H", "HT", "2H", "ET", "P", "FT"
    is_live: bool
    period: int  # 1=first half, 2=second half, 3=extra time 1, 4=extra time 2
    events: List[MatchEvent] = field(default_factory=list)
    home_stats: Optional[MatchStats] = None
    away_stats: Optional[MatchStats] = None
    home_xg_running: float = 0.0
    away_xg_running: float = 0.0
    home_pressure: float = 0.5
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    last_update: float = field(default_factory=time.time)
