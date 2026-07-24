"""KickoffAPI live match state client.

Fetches real-time score, clock, events, and stats from KickoffAPI.
Supports key rotation between multiple API keys for higher daily limits.

API: https://api.kickoffapi.com
Free tier: 100 requests/day per key (PRO trial)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cloudscraper

logger = logging.getLogger(__name__)

BASE_URL = "https://api.kickoffapi.com/api/v1"


@dataclass
class KickoffEvent:
    """A match event from KickoffAPI."""
    event_type: str  # "Goal", "Card", "subst", "Var"
    detail: str  # "Normal Goal", "Yellow Card", etc.
    team_id: int
    player_name: str
    minute: int
    comments: Optional[str] = None


@dataclass
class KickoffStats:
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
    """Complete live match state from KickoffAPI."""
    fixture_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    clock_minutes: float  # 0-90+
    status: str  # "NS", "1H", "HT", "2H", "ET", "P", "FT"
    is_live: bool
    period: int  # 1=first half, 2=second half, 3=extra time 1, 4=extra time 2
    events: List[KickoffEvent] = field(default_factory=list)
    home_stats: Optional[KickoffStats] = None
    away_stats: Optional[KickoffStats] = None
    # Running xG (estimated from shots)
    home_xg_running: float = 0.0
    away_xg_running: float = 0.0
    # Pressure (possession-based)
    home_pressure: float = 0.5
    # Cards
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    # Timestamp
    last_update: float = field(default_factory=time.time)


class KickoffApiClient:
    """Client for KickoffAPI live match data with key rotation.

    Usage:
        client = KickoffApiClient(keys=["key1", "key2"])
        state = client.get_live_match(fixture_id=1591866)
    """

    def __init__(self, keys: List[str]) -> None:
        self.keys = keys
        self._current_key_idx = 0
        self._request_count = 0
        self._last_request_time = 0.0
        self._session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "linux", "mobile": False}
        )
        self._remaining_per_key: Dict[str, int] = {k: 100 for k in keys}

    @property
    def _api_key(self) -> str:
        return self.keys[self._current_key_idx]

    def _rotate_key(self) -> bool:
        """Switch to next key if available. Returns True if rotated."""
        if self._current_key_idx < len(self.keys) - 1:
            self._current_key_idx += 1
            logger.info("Rotated to API key %d/%d (%d remaining)",
                       self._current_key_idx + 1, len(self.keys),
                       self._remaining_per_key[self._api_key])
            return True
        return False

    def _get(self, endpoint: str, params: Dict = None, retries: int = 2) -> Optional[Dict]:
        """Make an API request with rate limiting and key rotation."""
        # Rate limit: max 10 req/min to be safe
        elapsed = time.time() - self._last_request_time
        if elapsed < 6.5:
            time.sleep(6.5 - elapsed)

        url = f"{BASE_URL}/{endpoint}"
        if params is None:
            params = {}

        for attempt in range(len(self.keys)):
            for retry in range(retries):
                try:
                    resp = self._session.get(
                        url,
                        params=params,
                        headers={
                            "x-api-key": self._api_key,
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                        },
                        timeout=15,
                    )
                    self._last_request_time = time.time()
                    self._request_count += 1

                    # Track remaining
                    remaining = resp.headers.get("X-RateLimit-Remaining")
                    if remaining is not None:
                        self._remaining_per_key[self._api_key] = int(remaining)

                    if resp.status_code == 200:
                        return resp.json()
                    elif resp.status_code == 429:
                        logger.warning("KickoffAPI rate limited on key %d, rotating...",
                                      self._current_key_idx + 1)
                        if not self._rotate_key():
                            logger.error("All API keys exhausted!")
                            return None
                        break
                    elif resp.status_code == 403:
                        logger.warning("KickoffAPI 403 (Cloudflare?) on key %d, retry %d/%d...",
                                      self._current_key_idx + 1, retry + 1, retries)
                        if retry >= retries - 1:
                            logger.warning("403 retries exhausted on key %d, rotating", self._current_key_idx + 1)
                            if not self._rotate_key():
                                return None
                            break
                        time.sleep(3)
                        continue
                    else:
                        logger.error("KickoffAPI %d: %s", resp.status_code, resp.text[:200])
                        return None
                except Exception as e:
                    logger.error("KickoffAPI request failed: %s", e)
                    return None

        logger.error("All API keys exhausted")
        return None

    def get_live_match(self, fixture_id: int) -> Optional[LiveMatchState]:
        """Fetch live match state by fixture ID.

        Args:
            fixture_id: The fixture ID (e.g., 1494214 for Kalmar vs Malmo).

        Returns:
            LiveMatchState with all available data, or None if error.
        """
        data = self._get("fixtures", {"id": fixture_id})
        if not data or "response" not in data or not data["response"]:
            return None

        fixture = data["response"][0]
        state = self._parse_fixture(fixture)

        # Fetch events for live matches (1 extra request)
        if state.is_live and fixture_id:
            state.events = self.get_live_events(fixture_id)
            for event in state.events:
                is_home = event.team_id == fixture.get("homeTeamId", 0)
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
        data = self._get("fixtures", {"date": date})
        if not data or "response" not in data:
            return []
        return data["response"]

    def get_live_fixtures_for_date(self, date: str, league_id: Optional[int] = None) -> List[LiveMatchState]:
        """Fetch all fixtures for a date and return as LiveMatchState list.

        Args:
            date: Date string YYYY-MM-DD.
            league_id: Optional filter by league ID (e.g., 113 for Allsvenskan).

        Returns:
            List of LiveMatchState objects.
        """
        fixtures = self.get_fixtures_by_date(date)
        results = []
        for f in fixtures:
            if league_id is not None and f.get("leagueId") != league_id:
                continue
            results.append(self._parse_fixture(f))
        return results

    def get_live_events(self, fixture_id: int) -> List[KickoffEvent]:
        """Fetch match events (goals, cards, subs)."""
        data = self._get(f"fixtures/{fixture_id}/events")
        if not data or "response" not in data:
            return []

        events = []
        for e in data["response"]:
            events.append(KickoffEvent(
                event_type=e.get("type", ""),
                detail=e.get("detail", ""),
                team_id=e.get("teamId", 0),
                player_name=e.get("playerName", ""),
                minute=e.get("time", 0),
                comments=e.get("comments"),
            ))
        return events

    def get_match_statistics(self, fixture_id: int) -> tuple[Optional[KickoffStats], Optional[KickoffStats]]:
        """Fetch match statistics for both teams."""
        data = self._get(f"fixtures/{fixture_id}/statistics")
        if not data or "response" not in data:
            return None, None

        stats_list = []
        for team_stats in data["response"]:
            team_id = team_stats.get("teamId", 0)
            team_name = team_stats.get("teamName", "")
            statistics = team_stats.get("statistics", {})

            ms = KickoffStats(
                team_id=team_id,
                team_name=team_name,
                possession=_parse_pct(statistics.get("Ball Possession", "0%")),
                shots_on=statistics.get("Shots on Goal", 0),
                shots_off=statistics.get("Shots off Goal", 0),
                fouls=statistics.get("Total Fouls", 0),
                corners=statistics.get("Corner Kicks", 0),
                offsides=statistics.get("Offsides", 0),
            )
            stats_list.append(ms)

        home_stats = stats_list[0] if len(stats_list) > 0 else None
        away_stats = stats_list[1] if len(stats_list) > 1 else None
        return home_stats, away_stats

    def _parse_fixture(self, fixture: Dict) -> LiveMatchState:
        """Parse a fixture response into LiveMatchState."""
        status_short = fixture.get("statusShort", "NS")
        elapsed = fixture.get("elapsed") or 0

        # Determine period
        is_live = status_short in ("1H", "2H", "HT", "ET", "P", "BT", "ST", "LIVE", "AET", "PEN")
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

        home_score = fixture.get("goalsHome") or 0
        away_score = fixture.get("goalsAway") or 0

        home_team = fixture.get("homeTeam", {}).get("name", "")
        away_team = fixture.get("awayTeam", {}).get("name", "")
        fixture_id = fixture.get("id", 0)

        state = LiveMatchState(
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

        return state

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def remaining(self) -> int:
        return sum(self._remaining_per_key.values())


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
