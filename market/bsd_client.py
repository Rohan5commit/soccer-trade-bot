"""BSD (bzzoiro) live match state client.

Fetches real-time score, clock, events, and odds from BSD API (sports.bzzoiro.com).
Replaces API-Football as the primary live data source — free, no quota, 83+ leagues,
covers Turkish Super Lig 100%, works from datacenter IPs.

API: https://sports.bzzoiro.com/api/v2/
Auth: Token header (free registration)
"""
from __future__ import annotations

import logging
import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://sports.bzzoiro.com/api/v2"

# BSD league IDs → Kalshi series tickers
# Note: BSD league 5 (Bundesliga) is NOT mapped — Kalshi has no Bundesliga series
BSD_TO_KALSHI_SERIES: Dict[int, str] = {
    11: "KXSUPERLIGGAME",    # Trendyol Super Lig (Turkey)
    1: "KXPREMIERLEAGUE",    # Premier League (England)
    4: "KXSERIEAGAME",       # Serie A (Italy)
    3: "KXPRIMERALIGAME",    # La Liga (Spain)
    6: "KXMLSGAME",          # Ligue 1 (France) — note: Kalshi uses MLS for Ligue 1
    10: "KXEREDIVISIEGAME",  # Eredivisie (Netherlands)
    7: "KXUCLGAME",          # Champions League (Europe)
    8: "KXUELGAME",          # Europa League (Europe)
    83: "KXUECLGAME",        # Conference League (Europe)
    9: "KXBRASILEIROGAME",   # Brasileirão Serie A (Brazil)
    34: "KXBRASILEIROBGAME", # Brasileirão Serie B (Brazil)
    18: "KXMLSGAME",         # MLS (USA)
    24: "KXASEANGAME",       # AFC Asian Cup
    17: "KXSAUDIPLGAME",     # Saudi Pro League
    49: "KXKLEAGAME",        # J1 League (Japan)
    52: "KXCHNSLGAME",       # Chinese Super League
    22: "KXSLGREECEGAME",    # Parva Liga (Bulgaria) — maps to Greek SL
    84: "KXDENSUPERLIGAGAME",# Danish Superliga
    85: "KXARGNACBGAME",     # Liga Profesional (Argentina)
    19: "KXLIGAMXGAME",      # Liga MX (Mexico)
    80: "KXPERLIGA1GAME",    # Categoría Primera A (Colombia)
    15: "KXSWISSLEAGUEGAME", # Super League (Switzerland)
}

# Kalshi series → BSD league IDs (reverse of above)
KALSHI_TO_BSD_LEAGUE: Dict[str, List[int]] = {}
for _bsd_id, _kalshi in BSD_TO_KALSHI_SERIES.items():
    KALSHI_TO_BSD_LEAGUE.setdefault(_kalshi, []).append(_bsd_id)

# All BSD league IDs we care about
ALL_BSD_LEAGUE_IDS = list(BSD_TO_KALSHI_SERIES.keys())


@dataclass
class BSDEvent:
    """A match event from BSD API."""
    event_type: str  # "goal", "card", "substitution", etc.
    detail: str
    team_name: str
    player_name: str
    minute: int


@dataclass
class LiveMatchState:
    """Complete live match state from BSD API.

    Compatible with the LiveMatchState used by run_bot_github.py.
    """
    fixture_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    clock_minutes: float  # 0-90+
    status: str  # "NS", "1H", "HT", "2H", "FT"
    is_live: bool
    period: int  # 1=first half, 2=second half, 3=extra time
    events: List[BSDEvent] = field(default_factory=list)
    home_stats: object = None  # Compatible with APIFootballStats (always None for BSD)
    away_stats: object = None
    home_xg_running: float = 0.0
    away_xg_running: float = 0.0
    home_pressure: float = 0.5
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    last_update: float = field(default_factory=time.time)


# Status mapping: BSD → API-Football compatible
STATUS_MAP = {
    "notstarted": "NS",
    "inprogress": "1H",  # Refined by period
    "1st_half": "1H",
    "2nd_half": "2H",
    "halftime": "HT",
    "HT": "HT",
    "finished": "FT",
    "FT": "FT",
    "postponed": "PST",
    "cancelled": "CANC",
    "awarded": "AWD",
    "afterextratime": "FT",
    "afterpenalties": "FT",
    "extra_time_1st_half": "ET",
    "extra_time_halftime": "ET",
    "extra_time_2nd_half": "ET",
    "ET": "ET",
    "penalties": "P",
    "P": "P",
    "suspended": "SUSP",
    "delayed": "DELAYED",
    "interrupted": "INT",
}

PERIOD_MAP = {
    "": 0,
    "1st_half": 1,
    "1T": 1,
    "halftime": 1,
    "HT": 1,
    "2nd_half": 2,
    "2T": 2,
    "extra_time_1st_half": 3,
    "ET1": 3,
    "extra_time_halftime": 3,
    "extra_time_2nd_half": 4,
    "ET2": 4,
    "penalties": 5,
    "P": 5,
    "FT": 5,
    "finished": 5,
}


class BSDClient:
    """Client for BSD API live match data.

    Usage:
        client = BSDClient(api_key="your-token")
        state = client.get_live_match(event_id=222628)
    """

    def __init__(self, api_key: str = "") -> None:
        self._session = requests.Session()
        key = api_key or os.environ.get("BSD_API_KEY", "")
        if key:
            self._session.headers["Authorization"] = f"Token {key}"
        self._session.headers["Accept"] = "application/json"
        self._request_count = 0
        self._cache: Dict[str, dict] = {}
        self._cache_ttl: Dict[str, float] = {}

    @property
    def is_available(self) -> bool:
        """Check if API key is set."""
        return "Authorization" in self._session.headers

    def _get(self, endpoint: str, params: Dict = None, ttl: int = 30) -> Optional[Dict]:
        """Make a GET request to BSD API with simple caching."""
        url = f"{BASE_URL}/{endpoint}"
        cache_key = f"{endpoint}:{params}"

        # Check cache
        if cache_key in self._cache:
            age = time.time() - self._cache_ttl.get(cache_key, 0)
            if age < ttl:
                return self._cache[cache_key]

        try:
            resp = self._session.get(url, params=params, timeout=15)
            self._request_count += 1

            if resp.status_code == 429:
                logger.warning("BSD API rate limited (429)")
                return None

            if resp.status_code == 401:
                logger.error("BSD API authentication failed (401) — check API key")
                return None

            if resp.status_code != 200:
                logger.warning("BSD API %d: %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            self._cache[cache_key] = data
            self._cache_ttl[cache_key] = time.time()
            return data

        except requests.Timeout:
            logger.warning("BSD API timeout for %s", endpoint)
            return None
        except Exception as e:
            logger.warning("BSD API error: %s", e)
            return None

    def get_live_events(self) -> List[Dict]:
        """Get all currently live events."""
        data = self._get("events/live/", ttl=5)
        if not data:
            return []
        return data.get("events", [])

    def get_upcoming_events(
        self, league_id: int = None, limit: int = 50,
        date_from: str = None, date_to: str = None,
    ) -> List[Dict]:
        """Get upcoming events, optionally filtered by league and date range."""
        params = {"limit": limit, "status": "notstarted"}
        if league_id:
            params["league_id"] = league_id
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        # Cache TTL: shorter when filtering by date (volatile), longer otherwise
        ttl = 60 if (date_from or date_to) else 300
        data = self._get("events/", params=params, ttl=ttl)
        if not data:
            return []
        return data.get("results", [])

    def get_event(self, event_id: int) -> Optional[Dict]:
        """Get a single event by ID."""
        data = self._get(f"events/{event_id}/", ttl=5)
        if not data or "error" in data:
            return None
        return data

    def get_odds(self, event_id: int) -> Optional[Dict]:
        """Get odds for a specific event."""
        data = self._get(f"events/{event_id}/odds/", ttl=5)
        if not data or "error" in data:
            return None
        return data.get("odds", {})

    def get_leagues(self) -> List[Dict]:
        """Get all available leagues."""
        data = self._get("leagues/", params={"limit": 200}, ttl=3600)
        if not data:
            return []
        return data.get("results", [])

    def _date_window(self, days_ahead: int = 7, days_behind: int = 1) -> tuple:
        """Return (date_from, date_to) strings for near-term event search."""
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        date_from = (now - timedelta(days=days_behind)).strftime("%Y-%m-%d")
        date_to = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        return date_from, date_to

    def find_event_by_teams(
        self, home: str, away: str, league_id: int = None
    ) -> Optional[Dict]:
        """Find an event by team names (fuzzy match).

        Searches live events first, then upcoming (date-filtered to ±7 days).
        """
        home_norm = self._normalize_team(home)
        away_norm = self._normalize_team(away)
        date_from, date_to = self._date_window()

        # Search live events first
        for event in self.get_live_events():
            if self._teams_match(event, home_norm, away_norm):
                return event

        # Search upcoming events (date-filtered so we don't match 2027 fixtures)
        search_leagues = [league_id] if league_id else ALL_BSD_LEAGUE_IDS
        for lid in search_leagues:
            for event in self.get_upcoming_events(
                league_id=lid, date_from=date_from, date_to=date_to
            ):
                if self._teams_match(event, home_norm, away_norm):
                    return event

        return None

    def find_event_by_kalshi_teams(
        self, kalshi_home: str, kalshi_away: str, kalshi_series: str
    ) -> Optional[Dict]:
        """Find an event by Kalshi team names and series.

        Maps Kalshi series → BSD league IDs, then searches.
        Date-filtered to avoid matching far-future fixtures (e.g. May 2027).
        """
        league_ids = KALSHI_TO_BSD_LEAGUE.get(kalshi_series, ALL_BSD_LEAGUE_IDS)
        date_from, date_to = self._date_window()

        # Search live events first
        for event in self.get_live_events():
            event_league = event.get("league_id")
            if event_league in league_ids:
                if self._teams_match(event, self._normalize_team(kalshi_home),
                                     self._normalize_team(kalshi_away)):
                    return event

        # Search upcoming events (date-filtered)
        for lid in league_ids:
            for event in self.get_upcoming_events(
                league_id=lid, date_from=date_from, date_to=date_to
            ):
                if self._teams_match(event, self._normalize_team(kalshi_home),
                                     self._normalize_team(kalshi_away)):
                    return event

        return None

    def get_live_match(self, event_id: int) -> Optional[LiveMatchState]:
        """Get live match state for a specific event.

        Prefers the /events/live/ endpoint (accurate live status) over
        the single-event endpoint (which may be stale/cached).

        Returns LiveMatchState compatible with run_bot_github.py.
        """
        # Prefer live endpoint — it has accurate status/period/minute
        for evt in self.get_live_events():
            if evt.get("id") == event_id:
                return self._event_to_live_state(evt)

        event = self.get_event(event_id)
        if not event:
            return None

        return self._event_to_live_state(event)

    def get_live_match_by_teams(
        self, home: str, away: str, series: str = ""
    ) -> Optional[LiveMatchState]:
        """Find and return live match state by team names."""
        event = None

        if series:
            event = self.find_event_by_kalshi_teams(home, away, series)
        else:
            event = self.find_event_by_teams(home, away)

        if not event:
            return None

        return self._event_to_live_state(event)

    def _event_to_live_state(self, event: Dict) -> LiveMatchState:
        """Convert BSD event to LiveMatchState."""
        status_raw = event.get("status", "notstarted")
        period_raw = event.get("period", "")
        minute_raw = event.get("current_minute") or 0.0
        # Handle "45+2" style strings
        if isinstance(minute_raw, str):
            try:
                if "+" in minute_raw:
                    base, extra = minute_raw.split("+", 1)
                    minute = float(base) + float(extra)
                else:
                    minute = float(minute_raw)
            except Exception:
                minute = 0.0
        else:
            minute = float(minute_raw) if minute_raw is not None else 0.0

        # Map status
        status = STATUS_MAP.get(status_raw, "NS")

        # Refine status by period if status is inprogress or a period string
        if status_raw == "inprogress" or status_raw in PERIOD_MAP:
            if period_raw in ("2nd_half", "2T"):
                status = "2H"
            elif period_raw in ("halftime", "HT"):
                status = "HT"
            elif period_raw in ("extra_time_1st_half", "ET1", "extra_time_halftime", "extra_time_2nd_half", "ET2"):
                status = "ET"
            elif period_raw in ("penalties", "P"):
                status = "P"
            elif period_raw in ("1st_half", "1T"):
                status = "1H"
            elif status_raw == "inprogress":
                status = "1H"  # Default to first half for inprogress

        # Map period
        period = PERIOD_MAP.get(period_raw, 0)

        is_live = status_raw in ("inprogress", "1st_half", "2nd_half",
                                  "halftime", "HT", "extra_time_1st_half",
                                  "extra_time_halftime", "extra_time_2nd_half",
                                  "penalties", "ET", "P", "1T", "2T", "ET1", "ET2",
                                  "afterextratime", "afterpenalties")

        # Get events (goals, cards, etc.)
        events = []
        # BSD API doesn't provide detailed events in the event object
        # but we can infer from scores and cards

        return LiveMatchState(
            fixture_id=event.get("id", 0),
            home_team=event.get("home_team", ""),
            away_team=event.get("away_team", ""),
            home_score=event.get("home_score") or 0,
            away_score=event.get("away_score") or 0,
            clock_minutes=minute,
            status=status,
            is_live=is_live,
            period=period,
            events=events,
            home_xg_running=0.0,
            away_xg_running=0.0,
            home_pressure=0.5,
            home_red_cards=0,
            away_red_cards=0,
            home_yellow_cards=0,
            away_yellow_cards=0,
            last_update=time.time(),
        )

    def _normalize_team(self, name: str) -> str:
        """Normalize team name for fuzzy matching."""
        name = name.lower().strip()
        # Remove common suffixes/prefixes
        for remove in ["fc", "cf", "sc", "ac", "as", "ss", "sk", "fk", "tk", "us", "rc", "rcd"]:
            if name.startswith(remove + " "):
                name = name[len(remove):].strip()
            if name.endswith(" " + remove):
                name = name[:-len(remove)].strip()
        # Strip diacritics via NFKD (handles é, ñ, ş, ğ, ı, etc.)
        name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
        # Turkish dotless i
        name = name.replace("ı", "i")
        # Remove non-alphanumeric
        name = "".join(c for c in name if c.isalnum())
        return name

    def _teams_match(self, event: Dict, home_norm: str, away_norm: str) -> bool:
        """Check if team names match (fuzzy)."""
        event_home = self._normalize_team(event.get("home_team", ""))
        event_away = self._normalize_team(event.get("away_team", ""))

        # Guard against empty normalization (e.g. "FC" -> "")
        if not home_norm or not away_norm or not event_home or not event_away:
            return False
        if len(home_norm) < 3 or len(away_norm) < 3 or len(event_home) < 3 or len(event_away) < 3:
            # For very short names, require exact match only
            return (event_home == home_norm and event_away == away_norm) or \
                   (event_home == away_norm and event_away == home_norm)

        # Exact match
        if event_home == home_norm and event_away == away_norm:
            return True

        # Substring match (both need reasonable length)
        if (home_norm in event_home or event_home in home_norm) and \
           (away_norm in event_away or event_away in away_norm):
            return True

        # Reverse (away team at home in BSD?)
        if (home_norm in event_away or event_away in home_norm) and \
           (away_norm in event_home or event_home in away_norm):
            return True

        return False

    def get_league_id_for_kalshi_series(self, series: str) -> Optional[int]:
        """Get BSD league ID for a Kalshi series ticker."""
        ids = KALSHI_TO_BSD_LEAGUE.get(series, [])
        return ids[0] if ids else None

    def get_kalshi_series_for_league(self, league_id: int) -> Optional[str]:
        """Get Kalshi series ticker for a BSD league ID."""
        return BSD_TO_KALSHI_SERIES.get(league_id)
