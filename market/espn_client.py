"""Live match data client using ESPN public API.

Primary live data source — free, no API key, reliable.
ESPN covers major leagues including Turkish Super Lig, Serie A, EPL, etc.

API: https://site.api.espn.com/apis/site/v2/sports/soccer/
No authentication required.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from market.api_football_client import LiveMatchState, APIFootballEvent

logger = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# ESPN league codes for major competitions
LEAGUE_CODES = {
    "turkish-super-lig": "tur.1",
    "tur": "tur.1",
    "super-lig": "tur.1",
    "premier-league": "eng.1",
    "eng": "eng.1",
    "la-liga": "esp.1",
    "esp": "esp.1",
    "bundesliga": "ger.1",
    "ger": "ger.1",
    "ligue-1": "fra.1",
    "fra": "fra.1",
    "serie-a": "ita.1",
    "ita": "ita.1",
    "eredivisie": "ned.1",
    "ned": "ned.1",
    "primeira-liga": "por.1",
    "por": "por.1",
    "champions-league": "uefa.champions",
    "europa-league": "uefa.europa",
}

# Reverse mapping: ESPN league name -> league code
ESPN_NAME_TO_CODE = {
    "turkish super lig": "tur.1",
    "premier league": "eng.1",
    "la liga": "esp.1",
    "bundesliga": "ger.1",
    "ligue 1": "fra.1",
    "serie a": "ita.1",
    "eredivisie": "ned.1",
    "primeira liga": "por.1",
    "champions league": "uefa.champions",
    "europa league": "uefa.europa",
}

# Status mapping: ESPN status -> API-Football status
STATUS_MAP = {
    "STATUS_SCHEDULED": "NS",
    "STATUS_IN_PROGRESS": "1H",  # Will be refined by period
    "STATUS_HALFTIME": "HT",
    "STATUS_FULL_TIME": "FT",
    "STATUS_POSTPONED": "PST",
    "STATUS_CANCELLED": "CANC",
    "STATUSDELAYED": "PST",
    "STATUS_HALFTIME": "HT",
    "STATUS_END_PERIOD": "HT",  # End of period (half)
}


class ESPNClient:
    """Client for ESPN live match data.

    Usage:
        client = ESPNClient()
        state = client.get_live_match(espn_event_id="401888389")
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
        self._request_count = 0

    def _get(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a GET request to ESPN API."""
        try:
            resp = self._session.get(url, params=params, timeout=15)
            self._request_count += 1

            if resp.status_code != 200:
                logger.error("ESPN %d: %s", resp.status_code, resp.text[:200])
                return None

            return resp.json()

        except requests.RequestException as e:
            logger.error("ESPN request failed: %s", e)
            return None

    def _resolve_league_code(self, league: str) -> str:
        """Convert league name/code to ESPN league code."""
        # Already an ESPN code
        if "." in league:
            return league
        # Check direct mapping
        if league in LEAGUE_CODES:
            return LEAGUE_CODES[league]
        # Check name mapping
        lower = league.lower()
        if lower in ESPN_NAME_TO_CODE:
            return ESPN_NAME_TO_CODE[lower]
        # Default: try as-is
        return league

    def get_fixtures(self, league: str = "tur.1", date: str = None) -> List[Dict]:
        """Get fixtures for a league and date.

        Args:
            league: ESPN league code (e.g., "tur.1", "eng.1")
            date: Date string YYYYMMDD (default: today)

        Returns:
            List of fixture dicts with ESPN event data.
        """
        league_code = self._resolve_league_code(league)
        url = f"{BASE_URL}/{league_code}/scoreboard"
        params = {}
        if date:
            params["dates"] = date

        data = self._get(url, params)
        if not data:
            return []

        return data.get("events", [])

    def find_event_id(self, home_team: str, away_team: str, league: str = None) -> Optional[str]:
        """Find ESPN event ID by team names.

        Searches today's fixtures across major leagues if no league specified.
        Returns ESPN event ID string or None.
        """
        leagues_to_search = [league] if league else [
            "tur.1", "eng.1", "esp.1", "ger.1", "fra.1", "ita.1",
        ]

        for lg in leagues_to_search:
            events = self.get_fixtures(league=lg)
            for e in events:
                name = e.get("name", "").lower()
                home_lower = home_team.lower()
                away_lower = away_team.lower()

                # Match by event name (e.g., "Amed SFK at Kocaelispor")
                if home_lower in name and away_lower in name:
                    return str(e.get("id", ""))
                if away_lower in name and home_lower in name:
                    return str(e.get("id", ""))

                # Match by competitor names
                for comp in e.get("competitions", []):
                    competitors = comp.get("competitors", [])
                    if len(competitors) == 2:
                        teams = [c.get("team", {}).get("displayName", "").lower()
                                 for c in competitors]
                        if (home_lower in teams and away_lower in teams):
                            return str(e.get("id", ""))
                        if (away_lower in teams and home_lower in teams):
                            return str(e.get("id", ""))

        return None

    def get_live_match(self, event_id: str) -> Optional[LiveMatchState]:
        """Fetch live match state by ESPN event ID.

        Returns LiveMatchState in the same format as API-Football client.
        """
        url = f"{BASE_URL}/scoreboard"
        data = self._get(url)
        if not data:
            return None

        # Find the specific event
        target_event = None
        for e in data.get("events", []):
            if str(e.get("id", "")) == str(event_id):
                target_event = e
                break

        if not target_event:
            # Try fetching by league + event ID
            for lg in ["tur.1", "eng.1", "esp.1", "ger.1", "fra.1", "ita.1"]:
                url = f"{BASE_URL}/{lg}/scoreboard"
                data = self._get(url)
                if data:
                    for e in data.get("events", []):
                        if str(e.get("id", "")) == str(event_id):
                            target_event = e
                            break
                if target_event:
                    break

        if not target_event:
            return None

        return self._parse_event(target_event)

    def get_live_match_by_league(self, event_id: str, league: str) -> Optional[LiveMatchState]:
        """Fetch live match state by ESPN event ID and league code (more efficient)."""
        league_code = self._resolve_league_code(league)
        url = f"{BASE_URL}/{league_code}/scoreboard"
        data = self._get(url)
        if not data:
            return None

        for e in data.get("events", []):
            if str(e.get("id", "")) == str(event_id):
                return self._parse_event(e)

        return None

    def _parse_event(self, event: Dict) -> Optional[LiveMatchState]:
        """Parse ESPN event into LiveMatchState."""
        try:
            comps = event.get("competitions", [])
            if not comps:
                return None
            comp = comps[0]

            # Get competitors
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None

            home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

            home_team = home_comp.get("team", {}).get("displayName", "Unknown")
            away_team = away_comp.get("team", {}).get("displayName", "Unknown")
            home_score = int(home_comp.get("score", 0) or 0)
            away_score = int(away_comp.get("score", 0) or 0)

            # Parse status
            status = comp.get("status", {})
            status_type = status.get("type", {})
            espn_status = status_type.get("name", "STATUS_SCHEDULED")
            detail = status_type.get("detail", "")
            clock_str = status.get("displayClock", "0'")
            period = status.get("period", 0)

            # Map ESPN status to our status
            is_live = espn_status == "STATUS_IN_PROGRESS"
            is_finished = espn_status == "STATUS_FULL_TIME"
            is_ht = espn_status in ("STATUS_HALFTIME", "STATUS_END_PERIOD")

            if is_finished:
                status_short = "FT"
            elif is_ht:
                status_short = "HT"
            elif is_live:
                if period == 1:
                    status_short = "1H"
                elif period == 2:
                    status_short = "2H"
                elif period == 3:
                    status_short = "ET"
                elif period == 4:
                    status_short = "ET"
                else:
                    status_short = "1H"
            else:
                status_short = "NS"

            # Parse clock minutes
            clock_minutes = 0.0
            if is_live or is_ht:
                # Extract from displayClock (e.g., "45'+2'" or "67'")
                match = re.search(r"(\d+)", clock_str)
                if match:
                    clock_minutes = float(match.group(1))
                # Add extra time if present
                extra_match = re.search(r"'\+(\d+)'", clock_str)
                if extra_match:
                    clock_minutes += float(extra_match.group(1))

            elif is_finished:
                clock_minutes = 90.0
                if period >= 3:
                    clock_minutes = 120.0

            # Parse events (goals, cards)
            events = []
            home_red = 0
            away_red = 0
            home_yellow = 0
            away_yellow = 0

            # ESPN provides details with events
            details = comp.get("details", [])
            for detail_item in details:
                dtype = detail_item.get("type", {}).get("text", "")
                athlete = detail_item.get("athlete", {}).get("displayName", "")
                team_name = detail_item.get("team", {}).get("displayName", "")
                clock = detail_item.get("clock", {})
                minute = 0.0
                if clock and "displayValue" in clock:
                    m = re.search(r"(\d+)", clock["displayValue"])
                    if m:
                        minute = float(m.group(1))

                if "goal" in dtype.lower() or "score" in dtype.lower():
                    events.append(APIFootballEvent(
                        event_type="Goal",
                        detail=dtype,
                        team_id=0,
                        team_name=team_name,
                        player_name=athlete,
                        minute=minute,
                    ))
                elif "yellow" in dtype.lower():
                    events.append(APIFootballEvent(
                        event_type="Card",
                        detail="Yellow Card",
                        team_id=0,
                        team_name=team_name,
                        player_name=athlete,
                        minute=minute,
                    ))
                    if team_name == home_team:
                        home_yellow += 1
                    else:
                        away_yellow += 1
                elif "red" in dtype.lower():
                    events.append(APIFootballEvent(
                        event_type="Card",
                        detail="Red Card",
                        team_id=0,
                        team_name=team_name,
                        player_name=athlete,
                        minute=minute,
                    ))
                    if team_name == home_team:
                        home_red += 1
                    else:
                        away_red += 1

            return LiveMatchState(
                fixture_id=int(event.get("id", 0)),
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                clock_minutes=clock_minutes,
                status=status_short,
                is_live=is_live and not is_finished,
                period=period,
                events=events,
                home_red_cards=home_red,
                away_red_cards=away_red,
                home_yellow_cards=home_yellow,
                away_yellow_cards=away_yellow,
                last_update=time.time(),
            )

        except Exception as e:
            logger.error("ESPN parse error: %s", e)
            return None
