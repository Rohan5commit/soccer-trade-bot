"""Live match data client using football-data.org API.

Free tier: 10 req/min, covers PL, BL1, SA, FL1, PD, CL, ELC, DED, PPL.
No IP blocking — works from datacenter IPs (GitHub Actions).
Requires free API key from https://www.football-data.org/client/signup

API: https://api.football-data.org/v4/
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import requests

from market.api_football_client import LiveMatchState, APIFootballEvent

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# Free tier competition codes
FREE_TIER_COMPS = {"PL", "BL1", "BL2", "BL3", "SA", "SB", "FL1", "FL2",
                    "PD", "SD", "DED", "PPL", "CL", "EL", "EC", "WC",
                    "ELC", "EL1", "FAC", "DFB"}

# Competition code -> name mapping
COMP_NAMES = {
    "PL": "Premier League",
    "BL1": "Bundesliga",
    "BL2": "2. Bundesliga",
    "BL3": "3. Bundesliga",
    "SA": "Serie A",
    "SB": "Serie B",
    "FL1": "Ligue 1",
    "FL2": "Ligue 2",
    "PD": "La Liga",
    "SD": "Segunda Division",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "CL": "Champions League",
    "EL": "Europa League",
    "ELC": "Championship",
    "EL1": "League One",
}

# Status mapping: football-data.org -> API-Football
STATUS_MAP = {
    "SCHEDULED": "NS",
    "TIMED": "NS",
    "IN_PLAY": "1H",  # Refined by half
    "PAUSED": "HT",
    "HALFTIME": "HT",
    "FINISHED": "FT",
    "SUSPENDED": "PST",
    "POSTPONED": "PST",
    "CANCELLED": "CANC",
    "AWARDED": "AWD",
    "LIVE": "1H",
}


class FootballDataClient:
    """Client for football-data.org live match data.

    Usage:
        client = FootballDataClient(api_key="your-free-key")
        state = client.get_live_match(competition="SA", match_id=12345)
    """

    def __init__(self, api_key: str = "") -> None:
        self._session = requests.Session()
        key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")
        if key:
            self._session.headers["X-Auth-Token"] = key
        self._session.headers["Accept"] = "application/json"
        self._request_count = 0
        self._remaining = 10  # Free tier: 10/min

    @property
    def is_available(self) -> bool:
        """Check if API key is set."""
        return "X-Auth-Token" in self._session.headers

    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a GET request to football-data.org."""
        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            self._request_count += 1

            if resp.status_code == 429:
                logger.warning("football-data.org rate limited (429)")
                return None

            if resp.status_code == 403:
                logger.warning("football-data.org 403 — resource requires paid plan")
                return None

            if resp.status_code != 200:
                logger.error("football-data.org %d: %s", resp.status_code, resp.text[:200])
                return None

            return resp.json()

        except requests.RequestException as e:
            logger.error("football-data.org request failed: %s", e)
            return None

    def get_live_matches(self, competition: str = None) -> List[Dict]:
        """Get all live matches, optionally filtered by competition."""
        params = {"status": "LIVE,IN_PLAY,PAUSED,HALFTIME"}
        if competition:
            params["competitions"] = competition

        data = self._get("matches", params)
        if not data:
            return []
        return data.get("matches", [])

    def get_scheduled_matches(self, competition: str = None, date: str = None) -> List[Dict]:
        """Get scheduled matches for a competition and date range."""
        params = {"status": "SCHEDULED,TIMED"}
        if competition:
            params["competitions"] = competition
        if date:
            params["dateFrom"] = date
            params["dateTo"] = date

        data = self._get("matches", params)
        if not data:
            return []
        return data.get("matches", [])

    def find_match_id(self, home_team: str, away_team: str, competition: str = None) -> Optional[Dict]:
        """Find match by team names across free-tier competitions.

        Returns dict with match_id, competition, league_name or None.
        """
        comps_to_search = [competition] if competition else list(FREE_TIER_COMPS)

        for comp in comps_to_search:
            if comp not in FREE_TIER_COMPS:
                continue

            matches = self.get_scheduled_matches(competition=comp)
            if not matches:
                # Also check live matches
                matches = self.get_live_matches(competition=comp)

            home_lower = home_team.lower()
            away_lower = away_team.lower()

            for m in matches:
                m_home = m.get("homeTeam", {}).get("name", "").lower()
                m_away = m.get("awayTeam", {}).get("name", "").lower()

                # Match both orderings
                if (home_lower in m_home or m_home in home_lower) and \
                   (away_lower in m_away or m_away in away_lower):
                    return {
                        "match_id": m.get("id"),
                        "competition": comp,
                        "league_name": COMP_NAMES.get(comp, comp),
                        "home": m.get("homeTeam", {}).get("name", ""),
                        "away": m.get("awayTeam", {}).get("name", ""),
                    }
                if (away_lower in m_home or m_home in away_lower) and \
                   (home_lower in m_away or m_away in home_lower):
                    return {
                        "match_id": m.get("id"),
                        "competition": comp,
                        "league_name": COMP_NAMES.get(comp, comp),
                        "home": m.get("homeTeam", {}).get("name", ""),
                        "away": m.get("awayTeam", {}).get("name", ""),
                    }

        return None

    def get_match(self, match_id: int) -> Optional[Dict]:
        """Get full match details by ID."""
        data = self._get(f"matches/{match_id}")
        return data

    def get_live_match(self, match_id: int) -> Optional[LiveMatchState]:
        """Fetch live match state by match ID.

        Returns LiveMatchState in the same format as API-Football client.
        """
        match = self.get_match(match_id)
        if not match:
            return None

        return self._parse_match(match)

    def _parse_match(self, match: Dict) -> Optional[LiveMatchState]:
        """Parse football-data.org match into LiveMatchState."""
        try:
            home_team = match.get("homeTeam", {}).get("name", "Unknown")
            away_team = match.get("awayTeam", {}).get("name", "Unknown")

            # Parse status
            status_str = match.get("status", "SCHEDULED")
            is_live = status_str in ("IN_PLAY", "PAUSED", "HALFTIME", "LIVE")
            is_finished = status_str == "FINISHED"

            if is_finished:
                status_short = "FT"
            elif status_str in ("PAUSED", "HALFTIME"):
                status_short = "HT"
            elif is_live:
                status_short = "1H"  # Will refine below
            else:
                status_short = "NS"

            # Parse score
            score = match.get("score", {})
            ft = score.get("fullTime", {})
            home_score = ft.get("homeTeam", 0) or 0
            away_score = ft.get("awayTeam", 0) or 0

            # Parse clock (from minute field or utcDate)
            clock_minutes = 0.0
            minute = match.get("minute")
            if minute is not None:
                clock_minutes = float(minute)
            elif is_live:
                # Estimate from utcDate
                utc_date = match.get("utcDate", "")
                if utc_date:
                    from datetime import datetime, timezone
                    try:
                        match_dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                        elapsed = (datetime.now(timezone.utc) - match_dt).total_seconds() / 60
                        clock_minutes = min(max(elapsed, 0), 120.0)
                    except Exception:
                        pass
            elif is_finished:
                clock_minutes = 90.0

            # Refine status by half
            if is_live:
                if clock_minutes <= 45:
                    status_short = "1H"
                elif clock_minutes <= 60:
                    status_short = "HT"
                else:
                    status_short = "2H"

            # Parse events (goals, bookings, substitutions)
            events = []
            home_red = 0
            away_red = 0
            home_yellow = 0
            away_yellow = 0

            for goal in match.get("goals", []):
                scorer = goal.get("scorer", {})
                team = goal.get("team", {})
                minute_val = goal.get("minute", 0)
                events.append(APIFootballEvent(
                    event_type="Goal",
                    detail=goal.get("type", "REGULAR"),
                    team_id=team.get("id", 0),
                    team_name=team.get("name", ""),
                    player_name=scorer.get("name", ""),
                    minute=minute_val,
                ))

            for booking in match.get("bookings", []):
                player = booking.get("player", {})
                team = booking.get("team", {})
                minute_val = booking.get("minute", 0)
                card = booking.get("card", "")
                detail = "Red Card" if "RED" in card else "Yellow Card"
                events.append(APIFootballEvent(
                    event_type="Card",
                    detail=detail,
                    team_id=team.get("id", 0),
                    team_name=team.get("name", ""),
                    player_name=player.get("name", ""),
                    minute=minute_val,
                ))
                if "RED" in card:
                    if team.get("name") == home_team:
                        home_red += 1
                    else:
                        away_red += 1
                else:
                    if team.get("name") == home_team:
                        home_yellow += 1
                    else:
                        away_yellow += 1

            return LiveMatchState(
                fixture_id=match.get("id", 0),
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                clock_minutes=clock_minutes,
                status=status_short,
                is_live=is_live and not is_finished,
                period=1 if clock_minutes <= 45 else 2,
                events=events,
                home_red_cards=home_red,
                away_red_cards=away_red,
                home_yellow_cards=home_yellow,
                away_yellow_cards=away_yellow,
                last_update=time.time(),
            )

        except Exception as e:
            logger.error("football-data.org parse error: %s", e)
            return None
