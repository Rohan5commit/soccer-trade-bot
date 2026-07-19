"""API-Football live match state client.

Fetches real-time score, clock, events, and stats from API-Football.
No CV/OCR needed — replaces the vision pipeline with structured data.

API: https://v3.football.api-sports.io
Free tier: 100 requests/day (1 match ≈ 120 requests at 30s polling for 2hr)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"

# World Cup league ID and season
WORLD_CUP_LEAGUE_ID = 1
WORLD_CUP_SEASON = 2025


@dataclass
class MatchEvent:
    """A match event (goal, card, substitution, etc.)."""
    event_type: str  # "Goal", "Card", "subst", "Var"
    detail: str  # "Normal Goal", "Yellow Card", etc.
    team: str  # Team name
    player: str  # Player name
    minute: int  # Event minute
    comments: Optional[str] = None


@dataclass
class MatchStats:
    """Match statistics for a team."""
    team: str
    shots_on: int = 0
    shots_off: int = 0
    possession: float = 0.0
    passes: int = 0
    fouls: int = 0
    corners: int = 0
    offsides: int = 0


@dataclass
class LiveMatchState:
    """Complete live match state from API-Football."""
    fixture_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    clock_minutes: float  # 0-90+ (can be fractional)
    status: str  # "NS", "1H", "HT", "2H", "FT", etc.
    elapsed_minutes: int
    is_live: bool
    period: int  # 1=first half, 2=second half, 3=extra time 1, 4=extra time 2
    events: List[MatchEvent] = field(default_factory=list)
    home_stats: Optional[MatchStats] = None
    away_stats: Optional[MatchStats] = None
    # Running xG (estimated from shots and events)
    home_xg_running: float = 0.0
    away_xg_running: float = 0.0
    # Pressure (possession-based proxy)
    home_pressure: float = 0.5
    # Cards
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    # Timestamp
    last_update: float = field(default_factory=time.time)

    @property
    def is_extra_time(self) -> bool:
        return self.period >= 3


class ApiFootballClient:
    """Client for API-Football live match data.

    Usage:
        client = ApiFootballClient(api_key="your_key")
        state = client.get_live_match(fixture_id=1591866)
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": api_key,
        })
        self._request_count = 0
        self._last_request_time = 0.0

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make an API request with rate limiting."""
        # Rate limit: max 10 req/min on free tier
        elapsed = time.time() - self._last_request_time
        if elapsed < 6.5:  # ~9 req/min to be safe
            time.sleep(6.5 - elapsed)

        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            self._last_request_time = time.time()
            self._request_count += 1

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("API-Football rate limited, waiting 30s")
                time.sleep(30)
                return None
            else:
                logger.error("API-Football %d: %s", resp.status_code, resp.text[:200])
                return None
        except requests.RequestException as e:
            logger.error("API-Football request failed: %s", e)
            return None

    def get_live_match(self, fixture_id: int) -> Optional[LiveMatchState]:
        """Fetch live match state by fixture ID.

        Args:
            fixture_id: The fixture ID (e.g., 1591866 for Spain vs Argentina).

        Returns:
            LiveMatchState with all available data, or None if error.
        """
        data = self._get("fixtures", {"id": fixture_id})
        if not data or "response" not in data or not data["response"]:
            return None

        fixture = data["response"][0]
        return self._parse_fixture(fixture)

    def get_live_events(self, fixture_id: int) -> List[MatchEvent]:
        """Fetch match events (goals, cards, subs)."""
        data = self._get("fixtures/events", {"fixture": fixture_id})
        if not data or "response" not in data:
            return []

        events = []
        for e in data["response"]:
            time_info = e.get("time", {})
            team_info = e.get("team", {})
            player_info = e.get("player", {})
            detail = e.get("detail", "")
            comments = e.get("comments", None)

            events.append(MatchEvent(
                event_type=e.get("type", ""),
                detail=detail,
                team=team_info.get("name", ""),
                player=player_info.get("name", ""),
                minute=time_info.get("elapsed", 0),
                comments=comments,
            ))
        return events

    def get_match_statistics(self, fixture_id: int) -> Tuple[Optional[MatchStats], Optional[MatchStats]]:
        """Fetch match statistics for both teams."""
        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        if not data or "response" not in data:
            return None, None

        stats_list = []
        for team_stats in data["response"]:
            team_name = team_stats.get("team", {}).get("name", "")
            stats_dict = {}
            for stat in team_stats.get("statistics", []):
                key = stat.get("type", "").lower().replace(" ", "_")
                value = stat.get("value")
                # Parse percentage values
                if isinstance(value, str) and value.endswith("%"):
                    try:
                        value = float(value.rstrip("%"))
                    except ValueError:
                        value = 0.0
                elif isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        value = 0
                elif value is None:
                    value = 0
                stats_dict[key] = value

            ms = MatchStats(
                team=team_name,
                shots_on=stats_dict.get("shots_on_goal", 0),
                shots_off=stats_dict.get("shots_off_goal", 0),
                possession=stats_dict.get("ball_possession", 0.0),
                passes=stats_dict.get("total_passes", 0),
                fouls=stats_dict.get("total_fouls", 0),
                corners=stats_dict.get("corner_kicks", 0),
                offsides=stats_dict.get("offsides", 0),
            )
            stats_list.append(ms)

        home_stats = stats_list[0] if len(stats_list) > 0 else None
        away_stats = stats_list[1] if len(stats_list) > 1 else None
        return home_stats, away_stats

    def _parse_fixture(self, fixture: Dict) -> LiveMatchState:
        """Parse a fixture response into LiveMatchState."""
        fix = fixture.get("fixture", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        status_info = fixture.get("status", {})
        score_info = fixture.get("score", {})

        # Extract clock
        elapsed = status_info.get("elapsed") or 0
        status_short = status_info.get("short", "NS")

        # Handle half/extra time periods
        periods = score_info.get("extra", {})
        is_extra = status_short in ("ET", "P", "BT")
        current_period = 1
        if status_short == "2H":
            current_period = 2
        elif status_short == "HT":
            current_period = 1
        elif status_short == "ET":
            current_period = 3
        elif status_short == "P":
            current_period = 4

        clock_minutes = float(elapsed) if elapsed else 0.0

        # Score
        home_score = goals.get("home") or 0
        away_score = goals.get("away") or 0

        is_live = status_short in ("1H", "2H", "HT", "ET", "P", "BT", "ST", "LIVE")

        state = LiveMatchState(
            fixture_id=fix.get("id", 0),
            home_team=teams.get("home", {}).get("name", ""),
            away_team=teams.get("away", {}).get("name", ""),
            home_score=home_score,
            away_score=away_score,
            clock_minutes=clock_minutes,
            status=status_short,
            elapsed_minutes=elapsed or 0,
            is_live=is_live,
            period=current_period,
        )

        # Fetch events for live matches
        if is_live and fix.get("id"):
            state.events = self.get_live_events(fix["id"])

            # Count cards and goals from events
            for event in state.events:
                is_home = event.team == state.home_team
                if event.event_type == "Card":
                    if event.detail == "Red Card":
                        if is_home:
                            state.home_red_cards += 1
                        else:
                            state.away_red_cards += 1
                    elif event.detail == "Yellow Card":
                        if is_home:
                            state.home_yellow_cards += 1
                        else:
                            state.away_yellow_cards += 1

            # Fetch stats for live matches
            home_stats, away_stats = self.get_match_statistics(fix["id"])
            state.home_stats = home_stats
            state.away_stats = away_stats

            # Use possession for pressure proxy
            if home_stats and away_stats and home_stats.possession > 0:
                state.home_pressure = home_stats.possession / 100.0

            # Update running xG from shots (rough estimate)
            if home_stats and away_stats:
                # xG ≈ 0.10 * shots_on + 0.05 * shots_off (simplified)
                state.home_xg_running = home_stats.shots_on * 0.10 + home_stats.shots_off * 0.05
                state.away_xg_running = away_stats.shots_on * 0.10 + away_stats.shots_off * 0.05

        return state

    def search_world_cup_match(
        self,
        home_team: str = "",
        away_team: str = "",
    ) -> Optional[int]:
        """Find a World Cup fixture ID by team names.

        Args:
            home_team: Home team name (partial match, case-insensitive).
            away_team: Away team name (partial match, case-insensitive).

        Returns:
            Fixture ID if found, None otherwise.
        """
        # Search today's fixtures (World Cup fixtures appear in date search)
        import datetime
        today = datetime.date.today().isoformat()
        data = self._get("fixtures", {"date": today})
        if not data or "response" not in data:
            return None

        for fixture in data["response"]:
            teams = fixture.get("teams", {})
            league = fixture.get("league", {}).get("name", "").lower()

            # Only consider World Cup fixtures
            if "world cup" not in league and "world" not in league:
                continue

            h = teams.get("home", {}).get("name", "").lower()
            a = teams.get("away", {}).get("name", "").lower()

            if home_team and away_team:
                if home_team.lower() in h and away_team.lower() in a:
                    return fixture["fixture"]["id"]
            elif home_team:
                if home_team.lower() in h:
                    return fixture["fixture"]["id"]
            elif away_team:
                if away_team.lower() in a:
                    return fixture["fixture"]["id"]

        return None

    @property
    def request_count(self) -> int:
        return self._request_count
