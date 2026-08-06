"""Live match data client using SportScore API.

Fallback data source when API-Football can't find a fixture.
SportScore: free, no API key, CORS-open, ~10k req/day.

API: https://sportscore.com/api/widget/
Attribution: "Powered by SportScore" link required.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

import requests

from market.api_football_client import LiveMatchState, APIFootballEvent

logger = logging.getLogger(__name__)

BASE_URL = "https://sportscore.com/api/widget"


def _make_slug(home: str, away: str) -> str:
    """Convert team names to SportScore slug format.

    Example: "Manchester United", "Liverpool" -> "manchester-united-vs-liverpool"
    """
    def team_to_slug(name: str) -> str:
        # Remove common suffixes
        name = re.sub(r'\b(fc|cf|sc|ac|ss|rc|rcd|cd|ud|sd|cf|rcd|real|club|de|do|da|del|dos)\b',
                      '', name, flags=re.IGNORECASE)
        # Replace special chars with hyphens
        slug = re.sub(r'[^a-z0-9]+', '-', name.lower().strip())
        # Remove leading/trailing hyphens and collapse multiple
        slug = re.sub(r'-+', '-', slug).strip('-')
        return slug

    return f"{team_to_slug(home)}-vs-{team_to_slug(away)}"


class LiveScoreClient:
    """Client for SportScore live match data.

    Usage:
        client = LiveScoreClient()
        state = client.get_live_match(slug="manchester-united-vs-liverpool")
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

    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a GET request to SportScore API."""
        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            self._request_count += 1

            if resp.status_code != 200:
                logger.error("SportScore %d: %s", resp.status_code, resp.text[:200])
                return None

            return resp.json()

        except requests.RequestException as e:
            logger.error("SportScore request failed: %s", e)
            return None

    def get_matches(self, limit: int = 50) -> List[Dict]:
        """Get recent and live matches."""
        data = self._get("matches/", params={"sport": "football", "limit": limit})
        if not data:
            return []
        return data.get("matches", [])

    def find_match(self, home_team: str, away_team: str) -> Optional[str]:
        """Find match slug by team names.

        Returns slug string or None if not found.
        """
        matches = self.get_matches(limit=50)
        if not matches:
            return None

        home_norm = home_team.lower().strip()
        away_norm = away_team.lower().strip()

        for m in matches:
            m_home = m.get("home", "").lower().strip()
            m_away = m.get("away", "").lower().strip()

            # Check both orderings
            for h, a in [(home_norm, away_norm), (away_norm, home_norm)]:
                # Substring match
                if (h in m_home or m_home in h) and (a in m_away or m_away in a):
                    # Extract slug from URL
                    url = m.get("url", "")
                    if url:
                        # URL format: /football/match/<slug>/
                        slug = url.rstrip("/").split("/")[-1]
                        return slug

                # Word-level match
                h_words = {w for w in h.split() if len(w) > 3}
                a_words = {w for w in a.split() if len(w) > 3}
                m_h_words = {w for w in m_home.split() if len(w) > 3}
                m_a_words = {w for w in m_away.split() if len(w) > 3}

                if (h_words & m_h_words) and (a_words & m_a_words):
                    url = m.get("url", "")
                    if url:
                        slug = url.rstrip("/").split("/")[-1]
                        return slug

        return None

    def get_match_detail(self, slug: str) -> Optional[Dict]:
        """Get full match details including score, status, timeline."""
        data = self._get("match/", params={"sport": "football", "slug": slug})
        if not data:
            return None
        return data.get("match", data)

    def get_live_match(self, slug: str) -> Optional[LiveMatchState]:
        """Fetch live match state by SportScore slug.

        Returns LiveMatchState in the same format as API-Football client.
        """
        details = self.get_match_detail(slug)
        if not details:
            return None

        home_team = details.get("home", "")
        away_team = details.get("away", "")

        # Parse status
        status_text = details.get("status", "")
        is_live = status_text == "inprogress"
        is_finished = status_text == "finished"

        if is_finished:
            status_short = "FT"
        elif is_live:
            # Determine half from status_text or time
            time_str = details.get("time", "")
            if "1st" in str(time_str).lower() or "first" in str(time_str).lower():
                status_short = "1H"
            elif "2nd" in str(time_str).lower() or "second" in str(time_str).lower():
                status_short = "2H"
            elif "extra" in str(time_str).lower():
                status_short = "ET"
            else:
                status_short = "1H"
        else:
            status_short = "NS"

        # Parse score
        home_score = int(details.get("home_score", 0) or 0)
        away_score = int(details.get("away_score", 0) or 0)

        # Parse clock
        clock_minutes = 0.0
        if is_live:
            # Try to extract elapsed time from time field
            time_str = details.get("time", "")
            elapsed_match = re.search(r'(\d+)', str(time_str))
            if elapsed_match:
                clock_minutes = float(elapsed_match.group(1))
            else:
                # Estimate from match time
                match_time = details.get("time", "")
                if match_time:
                    try:
                        from datetime import datetime
                        match_dt = datetime.fromisoformat(str(match_time).replace("Z", "+00:00"))
                        elapsed_secs = time.time() - match_dt.timestamp()
                        clock_minutes = min(max(elapsed_secs / 60.0, 0), 120.0)
                    except Exception:
                        pass
        elif is_finished:
            clock_minutes = 90.0

        # Determine period
        period = 1
        if status_short == "2H":
            period = 2
        elif status_short == "HT":
            period = 1
        elif status_short == "ET":
            period = 3
        elif status_short == "FT":
            period = 0

        # Parse incidents for cards and events
        events = []
        home_red = 0
        away_red = 0
        home_yellow = 0
        away_yellow = 0

        timeline = details.get("timeline", [])
        for inc in timeline:
            inc_type = inc.get("type", "")
            inc_time = inc.get("time", 0)
            player_name = inc.get("player", "")
            team_name = inc.get("team", "")

            if inc_type in ("goal", "penalty_goal"):
                events.append(APIFootballEvent(
                    event_type="Goal",
                    detail="Normal Goal" if inc_type == "goal" else "Penalty",
                    team_id=0,
                    team_name=team_name,
                    player_name=player_name,
                    minute=inc_time,
                ))
            elif inc_type == "card":
                card_color = inc.get("card", "")
                detail = "Red Card" if card_color == "red" else "Yellow Card"
                events.append(APIFootballEvent(
                    event_type="Card",
                    detail=detail,
                    team_id=0,
                    team_name=team_name,
                    player_name=player_name,
                    minute=inc_time,
                ))
                if card_color == "red":
                    if team_name == home_team:
                        home_red += 1
                    else:
                        away_red += 1
                else:
                    if team_name == home_team:
                        home_yellow += 1
                    else:
                        away_yellow += 1

        return LiveMatchState(
            fixture_id=hash(slug) % (2**31),
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            clock_minutes=clock_minutes,
            status=status_short,
            is_live=is_live or is_finished,
            period=period,
            events=events,
            home_red_cards=home_red,
            away_red_cards=away_red,
            home_yellow_cards=home_yellow,
            away_yellow_cards=away_yellow,
            last_update=time.time(),
        )
