"""WorldCup26.ir client — unlimited free live data.

Primary live match data source when other APIs are Cloudflare-blocked.
No API key required. No rate limits.

Limitation: Only provides score — no clock minute, no events, no stats.
During live matches, provides:
- home_score, away_score
- time_elapsed (e.g., "45:00" or "1st Half" or "finished")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://worldcup26.ir"


class WorldCup26Client:
    """Client for worldcup26.ir live match data.

    Usage:
        client = WorldCup26Client()
        state = client.get_match(fixture_id="1591866")
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        self._request_count = 0
        self._last_request_time = 0.0

    def get_all_matches(self) -> list:
        """Fetch all matches. Returns list of match dicts."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        try:
            resp = self._session.get(f"{BASE_URL}/get/games", timeout=15)
            self._last_request_time = time.time()
            self._request_count += 1
            if resp.status_code == 200:
                return resp.json()
            logger.warning("worldcup26.ir %d: %s", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            logger.warning("worldcup26.ir request failed: %s", e)
        return []

    def get_match(self, mongodb_id: str) -> Optional[Dict]:
        """Fetch a single match by MongoDB ID.

        WC Final MongoDB ID: 679c9c8a5749c4077500e092
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        try:
            resp = self._session.get(f"{BASE_URL}/get/game/{mongodb_id}", timeout=15)
            self._last_request_time = time.time()
            self._request_count += 1
            if resp.status_code == 200:
                return resp.json()
            logger.warning("worldcup26.ir %d: %s", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            logger.warning("worldcup26.ir request failed: %s", e)
        return None

    def find_fixture(self, home_team: str = "Spain", away_team: str = "Argentina") -> Optional[Dict]:
        """Find a specific match by team names from all matches."""
        matches = self.get_all_matches()
        for m in matches:
            h = m.get("home_team_name_en", "")
            a = m.get("away_team_name_en", "")
            if (home_team.lower() in h.lower() and away_team.lower() in a.lower()) or \
               (home_team.lower() in a.lower() and away_team.lower() in h.lower()):
                return m
        return None

    @staticmethod
    def parse_local_date(match_data: Dict) -> Optional[datetime]:
        """Parse local_date from match data into a UTC datetime.

        local_date format: "07/19/2026 15:00" (MM/DD/YYYY HH:MM, UTC).
        Returns None if parse fails.
        """
        local_date = match_data.get("local_date", "")
        if not local_date:
            return None
        try:
            # Format: "07/19/2026 15:00"
            dt = datetime.strptime(local_date, "%m/%d/%Y %H:%M")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def get_match_status(match_data: Dict) -> str:
        """Determine match status from worldcup26 data.

        Returns: 'notstarted', 'live', 'finished', or 'unknown'.
        Handles the case where time_elapsed is wrong (e.g., "notstarted"
        when match should be live/finished).
        """
        time_elapsed = match_data.get("time_elapsed", "notstarted")
        finished = match_data.get("finished", "FALSE")

        if finished == "TRUE" or time_elapsed == "finished":
            return "finished"
        elif time_elapsed == "notstarted":
            return "notstarted"
        elif ":" in str(time_elapsed):
            return "live"
        return "unknown"

    @staticmethod
    def is_match_scheduled_to_be_live(match_data: Dict, buffer_minutes: int = 5) -> bool:
        """Check if the match is scheduled to be live right now.

        Uses local_date to determine if current time is within the match window.
        A typical soccer match is ~105 min (90 + halftime + stoppage).

        Args:
            match_data: Raw match dict from worldcup26.ir.
            buffer_minutes: Minutes before kickoff to start treating as live.

        Returns:
            True if current time is within the match window.
        """
        kick = WorldCup26Client.parse_local_date(match_data)
        if not kick:
            return False

        now = datetime.now(timezone.utc)
        elapsed = (now - kick).total_seconds() / 60

        # Match window: from buffer_minutes before kickoff to 120 min after
        if elapsed >= -buffer_minutes and elapsed <= 120:
            return True
        return False

    @property
    def request_count(self) -> int:
        return self._request_count
