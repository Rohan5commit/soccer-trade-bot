"""API-Football live match state client.

Fetches real-time score, clock, events, and stats from API-Football (api-sports.io).
Replaces the unreliable KickoffAPI client for CI/GitHub Actions usage.

API: https://v3.football.api-sports.io
Free tier: 100 requests/day
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"

# API-Football league IDs for leagues covered by Kalshi
# Kalshi series ticker → API-Football league_id
# Verified against API-Football on 2026-08-02
KALSHI_TO_LEAGUE_ID: Dict[str, int] = {
    # Tier 1: UEFA
    "KXUCLGAME": 2,
    "KXCHAMPIONSLEAGUEGAME": 2,
    "KXUELGAME": 3,
    "KXUECLGAME": 848,
    "KXUEFAGAME": 848,
    "KXUEFANLGAME": 848,
    # Tier 2: Top 5
    "KXPREMIERLEAGUE": 39,
    "KXSERIEAGAME": 71,
    "KXPRIMERALIGAME": 94,
    "KXMLSGAME": 253,
    # Tier 3: Strong European
    "KXEREDIVISIEGAME": 88,
    "KXSUPERLIGGAME": 203,
    "KXBRASILEIROGAME": 71,
    "KXBRASILEIROBGAME": 72,
    "KXALLSVENSKANGAME": 113,
    "KXSCOTTISHPREMGAME": 179,
    "KXSLGREECEGAME": 197,
    "KXSWISSLEAGUEGAME": 207,
    "KXDENSUPERLIGAGAME": 119,
    # Tier 4: Other
    "KXLIGAMXGAME": 262,
    "KXSAUDIPLGAME": 307,
    "KXKLEAGUEGAME": 292,
    "KXISLGAME": 164,
    "KXTHAIL1GAME": 296,
    "KXUAEPLGAME": 1089,
    "KXPERLIGA1GAME": 281,
    "KXVENFUTVEGAME": 300,
    "KXQSTARSGAME": 306,
    "KXSPBGAME": 475,
    "KXWIBPLGAME": 110,
    # Tier 5: Cups & Other
    "KXTACAPORTGAME": 96,
    "KXUSLGAME": 244,
    "KXUSOPENCUPGAME": 257,
    "KXSCOCUPGAME": 1078,
    "KXARGNACBGAME": 130,
    "KXCLUBFGAME": 15,
    "KXWCGAME": 1,
    "KXMENWORLDCUP": 1,
    "KXASEANGAME": 24,
    # Missing: KXCHNSLGAME (Chinese Super League), KXUSLCUPGAME, KXBRASILEIROGAME (search issue)
    # Bot still works via team name fuzzy matching for these
}


@dataclass
class APIFootballEvent:
    """A match event from API-Football."""
    event_type: str  # "Goal", "Card", "subst", "Var"
    detail: str  # "Normal Goal", "Yellow Card", etc.
    team_id: int
    team_name: str
    player_name: str
    minute: int
    comments: Optional[str] = None


@dataclass
class APIFootballStats:
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
    """Complete live match state from API-Football."""
    fixture_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    clock_minutes: float  # 0-90+
    status: str  # "NS", "1H", "HT", "2H", "ET", "P", "FT"
    is_live: bool
    period: int  # 1=first half, 2=second half, 3=extra time 1, 4=extra time 2
    events: List[APIFootballEvent] = field(default_factory=list)
    home_stats: Optional[APIFootballStats] = None
    away_stats: Optional[APIFootballStats] = None
    home_xg_running: float = 0.0
    away_xg_running: float = 0.0
    home_pressure: float = 0.5
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    last_update: float = field(default_factory=time.time)


class APIFootballClient:
    """Client for API-Football live match data.

    Supports dual-key rotation: when the primary key hits rate limit,
    automatically switches to the secondary key.

    Usage:
        client = APIFootballClient(api_key="key1", api_key_2="key2")
        state = client.get_live_match(fixture_id=12345)
    """

    def __init__(self, api_key: str, api_key_2: str = "") -> None:
        self.api_key = api_key
        self.api_key_2 = api_key_2
        self._request_count = 0
        self._remaining = 100  # Free tier: 100/day
        self._active_key_index = 0  # 0 = primary, 1 = secondary
        self._session = requests.Session()
        self._session.headers.update({
            "x-apisports-key": api_key,
            "Accept": "application/json",
        })

    def _switch_key(self) -> bool:
        """Switch to secondary API key if available. Returns True if switched."""
        if self.api_key_2 and self._active_key_index == 0:
            self._active_key_index = 1
            self._session.headers["x-apisports-key"] = self.api_key_2
            self._remaining = 100  # Reset for new key
            logger.info("API-Football: switched to secondary key (remaining=%d)", self._remaining)
            return True
        return False

    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a GET request to API-Football with rate limiting and key rotation."""
        if self._remaining <= 0:
            if self._switch_key():
                return self._get(endpoint, params)
            logger.error("API-Football daily rate limit exhausted (both keys)")
            return None

        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(4):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                self._request_count += 1

                # Parse rate limit headers
                remaining = resp.headers.get("x-ratelimit-requests-remaining")
                if remaining is not None:
                    self._remaining = int(remaining)

                if resp.status_code == 429:
                    # Try switching to secondary key first
                    if self._switch_key():
                        continue
                    try:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                    except (ValueError, TypeError):
                        retry_after = 60
                    logger.warning("API-Football 429 — retry after %ds (remaining=%d, attempt %d/4)",
                                   retry_after, self._remaining, attempt + 1)
                    time.sleep(min(retry_after, 60))
                    continue

                if resp.status_code in (500, 502, 503, 504):
                    logger.warning("API-Football %d server error — retrying (attempt %d/4)",
                                   resp.status_code, attempt + 1)
                    time.sleep(5 * (attempt + 1))
                    continue

                if resp.status_code != 200:
                    logger.error("API-Football %d: %s", resp.status_code, resp.text[:200])
                    return None

                data = resp.json()
                if data.get("errors"):
                    logger.warning("API-Football errors: %s (attempt %d/4)", data["errors"], attempt + 1)
                    # Try switching to secondary key on error (suspended account, etc.)
                    if self._switch_key():
                        continue
                    # If no secondary key or already switched, retry with backoff
                    if attempt < 3:
                        time.sleep(3 * (attempt + 1))
                        continue
                    return None

                return data.get("response", data)

            except requests.RequestException as e:
                logger.warning("API-Football request failed (attempt %d/4): %s", attempt + 1, e)
                if attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                return None

        logger.error("API-Football 429 — all 4 attempts exhausted for %s", endpoint)
        return None

    def get_live_match(self, fixture_id: int) -> Optional[LiveMatchState]:
        """Fetch live match state by fixture ID.

        Args:
            fixture_id: The fixture ID from API-Football.

        Returns:
            LiveMatchState with all available data, or None if error.
        """
        response = self._get("fixtures", {"id": fixture_id})
        if not response:
            return None

        # response is a list; take first fixture
        fixtures = response if isinstance(response, list) else [response]
        if not fixtures:
            return None

        fixture = fixtures[0] if isinstance(fixtures, list) else fixtures
        # API-Football returns {"fixture": {...}, "league": {...}, "teams": {...}, "goals": {...}, ...}
        if isinstance(fixture, dict) and "fixture" in fixture:
            state = self._parse_fixture(fixture)
        else:
            logger.warning("Unexpected API-Football response format")
            return None

        # Fetch events for live matches (1 extra request)
        if state.is_live and fixture_id:
            state.events = self.get_live_events(fixture_id)
            for event in state.events:
                fixture_data = fixture.get("fixture", {}) if isinstance(fixture, dict) else {}
                is_home = event.team_id == fixture.get("teams", {}).get("home", {}).get("id", 0)
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

        return state

    def get_fixtures_by_date(self, date: str) -> List[Dict]:
        """Fetch all fixtures for a given date.

        Args:
            date: Date string in YYYY-MM-DD format.

        Returns:
            List of fixture dicts from the API response.
        """
        response = self._get("fixtures", {"date": date})
        if not response:
            return []
        return response if isinstance(response, list) else [response]

    def get_fixtures_by_league(self, league_id: int, date: str) -> List[Dict]:
        """Fetch fixtures for a specific league and date."""
        response = self._get("fixtures", {"league": league_id, "date": date})
        if not response:
            return []
        return response if isinstance(response, list) else [response]

    def get_live_events(self, fixture_id: int) -> List[APIFootballEvent]:
        """Fetch match events (goals, cards, subs)."""
        response = self._get("fixtures/events", {"fixture": fixture_id})
        if not response:
            return []

        events = []
        event_list = response if isinstance(response, list) else []
        for e in event_list:
            time_data = e.get("time", {})
            team_data = e.get("team", {})
            player_data = e.get("player", {})
            detail = e.get("detail", "")
            event_type = e.get("type", "")

            events.append(APIFootballEvent(
                event_type=event_type,
                detail=detail,
                team_id=team_data.get("id", 0),
                team_name=team_data.get("name", ""),
                player_name=player_data.get("name", ""),
                minute=time_data.get("elapsed", 0),
                comments=e.get("comments"),
            ))
        return events

    def get_match_statistics(self, fixture_id: int) -> Tuple[Optional[APIFootballStats], Optional[APIFootballStats]]:
        """Fetch match statistics for both teams."""
        response = self._get("fixtures/statistics", {"fixture": fixture_id})
        if not response:
            return None, None

        stats_list = response if isinstance(response, list) else []
        home_stats = None
        away_stats = None

        for team_stats in stats_list:
            team_data = team_stats.get("team", {})
            statistics = {s["type"]: s["value"] for s in team_stats.get("statistics", [])}

            ms = APIFootballStats(
                team_id=team_data.get("id", 0),
                team_name=team_data.get("name", ""),
                possession=_parse_pct(statistics.get("Ball Possession", "0%")),
                shots_on=statistics.get("Shots on Goal", 0) or 0,
                shots_off=statistics.get("Shots off Goal", 0) or 0,
                fouls=statistics.get("Total Fouls", 0) or 0,
                corners=statistics.get("Corner Kicks", 0) or 0,
                offsides=statistics.get("Offsides", 0) or 0,
            )

            if home_stats is None:
                home_stats = ms
            else:
                away_stats = ms

        return home_stats, away_stats

    def _parse_fixture(self, fixture: Dict) -> LiveMatchState:
        """Parse an API-Football fixture response into LiveMatchState."""
        fixture_data = fixture.get("fixture", {})
        teams_data = fixture.get("teams", {})
        goals_data = fixture.get("goals", {})
        status_data = fixture_data.get("status", {})

        status_short = status_data.get("short", "NS")
        elapsed = status_data.get("elapsed") or 0

        # Determine if live
        live_statuses = ("1H", "2H", "HT", "ET", "P", "BT", "ST", "LIVE", "AET", "PEN")
        is_live = status_short in live_statuses

        # Determine period
        period = 1
        if status_short == "2H":
            period = 2
        elif status_short == "HT":
            period = 1
        elif status_short == "ET":
            period = 3
        elif status_short == "P":
            period = 4

        clock_minutes = float(elapsed) if elapsed else 0.0

        home_score = goals_data.get("home") or 0
        away_score = goals_data.get("away") or 0

        home_team = teams_data.get("home", {}).get("name", "")
        away_team = teams_data.get("away", {}).get("name", "")
        fixture_id = fixture_data.get("id", 0)

        return LiveMatchState(
            fixture_id=fixture_id,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            clock_minutes=clock_minutes,
            status=status_short,
            is_live=is_live,
            period=period,
        )

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def remaining(self) -> int:
        return self._remaining


def _parse_pct(val) -> float:
    """Parse '54%' to 54.0."""
    if isinstance(val, str) and val.endswith("%"):
        try:
            return float(val.rstrip("%"))
        except ValueError:
            return 0.0
    elif isinstance(val, (int, float)):
        return float(val)
    return 0.0
