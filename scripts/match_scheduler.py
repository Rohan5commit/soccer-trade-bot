#!/usr/bin/env python3
"""Match scheduler: discovers upcoming soccer matches from Kalshi + API-Football.

Used by GitHub Actions workflows:
  - scheduler.yml: Daily discovery, stores today's matches
  - watcher.yml: Checks proximity, dispatches bot when match is ~2hrs away

Outputs JSON to stdout and optionally saves to a file.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

IST = timezone(timedelta(hours=5, minutes=30))

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market.kalshi_client import KalshiClient, SOCCER_SERIES
from market.bsd_client import BSDClient, KALSHI_TO_BSD_LEAGUE
from config import load_config


# Kalshi series → API-Football league ID (for kickoff time lookup)
KALSHI_TO_LEAGUE_ID: Dict[str, int] = {
    "KXUCLGAME": 2, "KXCHAMPIONSLEAGUEGAME": 2,
    "KXUELGAME": 3, "KXUECLGAME": 848, "KXUEFAGAME": 848, "KXUEFANLGAME": 848,
    "KXPREMIERLEAGUE": 39, "KXSERIEAGAME": 71, "KXPRIMERALIGAME": 94,
    "KXMLSGAME": 253, "KXEREDIVISIEGAME": 88, "KXSUPERLIGGAME": 203,
    "KXBRASILEIROGAME": 71, "KXBRASILEIROBGAME": 72,
    "KXALLSVENSKANGAME": 113, "KXSCOTTISHPREMGAME": 179,
    "KXSLGREECEGAME": 197, "KXSWISSLEAGUEGAME": 207,
    "KXDENSUPERLIGAGAME": 119, "KXLIGAMXGAME": 262,
    "KXSAUDIPLGAME": 307, "KXKLEAGUEGAME": 292, "KXISLGAME": 164,
    "KXTHAIL1GAME": 296, "KXUAEPLGAME": 1089,
    "KXPERLIGA1GAME": 281, "KXVENFUTVEGAME": 300,
    "KXQSTARSGAME": 306, "KXSPBGAME": 475, "KXWIBPLGAME": 110,
    "KXTACAPORTGAME": 96, "KXUSLGAME": 244, "KXUSOPENCUPGAME": 257,
    "KXSCOCUPGAME": 1078, "KXARGNACBGAME": 130,
    "KXCLUBFGAME": 15, "KXWCGAME": 1, "KXMENWORLDCUP": 1,
    "KXASEANGAME": 24,
}


def parse_kalshi_event(event: dict, now: datetime, api_football_fixtures: Dict[str, dict] = None,
                       bsd_fixtures: Dict[str, dict] = None) -> Optional[Dict]:
    """Parse a Kalshi event into a match candidate dict.

    Uses API-Football fixtures to get actual kickoff time (Kalshi only has date).

    Returns None if the event is not a valid upcoming match.
    """
    title = event.get("title", "")
    event_ticker = event.get("event_ticker", "")
    series = event.get("series_ticker", "")

    if " vs " not in title:
        return None

    teams = title.split(" vs ", 1)
    home = teams[0].strip()
    away = teams[1].strip().split(" winner")[0].strip()

    # Strip Kalshi suffixes like ": Regulation Time Moneyline", ": Extra Time Winner", etc.
    for suffix_marker in [":", " - "]:
        if suffix_marker in home:
            home = home.split(suffix_marker, 1)[0].strip()
        if suffix_marker in away:
            away = away.split(suffix_marker, 1)[0].strip()

    if not home or not away:
        return None

    # Parse date from sub_title: "(Aug 4)" → Aug 4
    sub_title = event.get("sub_title", "")
    date_match = re.search(r'\((\w{3})\s+(\d{1,2})\)', sub_title)

    kickoff = None
    if date_match:
        try:
            month_str = date_match.group(1).upper()
            day = int(date_match.group(2))
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            month = month_map.get(month_str, 0)
            if month > 0:
                year = now.year
                if month < now.month - 6:
                    year += 1
                elif month > now.month + 6:
                    year -= 1

                # Try to get actual kickoff time from BSD fixtures first
                if bsd_fixtures:
                    kickoff = _find_kickoff_from_bsd(home, away, bsd_fixtures)

                # Fall back to API-Football fixtures
                if kickoff is None and api_football_fixtures:
                    kickoff = _find_kickoff_from_api_football(
                        home, away, year, month, day, api_football_fixtures
                    )

                # Fallback: use 21:00 IST (common European kickoff time)
                if kickoff is None:
                    kickoff = datetime(year, month, day, 21, 0, tzinfo=IST)
        except Exception:
            pass

    if kickoff is None:
        return None

    minutes_until = (kickoff - now).total_seconds() / 60

    # Only include matches within next 24 hours that haven't started yet
    if minutes_until < -10 or minutes_until > 1440:
        return None

    # Markets count will be populated by discover_matches() via separate API call
    # (Kalshi /events endpoint doesn't include market details)

    return {
        "home": home,
        "away": away,
        "event_ticker": event_ticker,
        "series": series,
        "kickoff_ist": kickoff.isoformat(),
        "minutes_until": round(minutes_until, 1),
        "sub_title": sub_title,
        "markets_count": 0,
    }


def _find_kickoff_from_api_football(
    home: str, away: str, year: int, month: int, day: int,
    fixtures: Dict[str, dict]
) -> Optional[datetime]:
    """Find actual kickoff time from API-Football fixtures by matching team names."""
    target_date = f"{year}-{month:02d}-{day:02d}"
    home_norm = home.lower().replace(".", "").replace("'", "").replace("-", " ")
    away_norm = away.lower().replace(".", "").replace("'", "").replace("-", " ")

    for fixture_key, fixture in fixtures.items():
        inner = fixture.get("fixture", {})
        fixture_date = inner.get("date", "")[:10]
        if fixture_date != target_date:
            continue

        f_teams = fixture.get("teams", {})
        f_home = f_teams.get("home", {}).get("name", "").lower()
        f_away = f_teams.get("away", {}).get("name", "").lower()

        # Check if both team names match (substring match)
        if (home_norm in f_home or f_home in home_norm) and \
           (away_norm in f_away or f_away in away_norm):
            # Parse UTC timestamp and convert to IST
            fixture_date_str = inner.get("date", "")
            try:
                utc_time = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))
                return utc_time.astimezone(IST)
            except Exception:
                pass

    return None


def fetch_api_football_fixtures(api_key: str, api_key_2: str = "") -> Dict[str, dict]:
    """Fetch today's and tomorrow's fixtures from API-Football for kickoff time lookup.

    Returns dict keyed by fixture ID. Tries api_key first; falls back to api_key_2
    on error (suspended account, rate limit, etc.).
    """
    def _try_fetch(key: str) -> Dict[str, dict]:
        if not key:
            return {}
        headers = {"x-apisports-key": key}
        base = "https://v3.football.api-sports.io"
        fixtures = {}
        for date_offset in [0, 1]:
            try:
                date = (datetime.now(timezone.utc) + timedelta(days=date_offset)).strftime("%Y-%m-%d")
                resp = requests.get(f"{base}/fixtures", params={"date": date}, headers=headers, timeout=15)
                data = resp.json()
                if resp.status_code == 200 and not data.get("errors"):
                    for f in data.get("response", []):
                        fid = f.get("fixture", {}).get("id")
                        if fid:
                            fixtures[fid] = f
                else:
                    return {}
                time.sleep(0.5)
            except Exception:
                return {}
        return fixtures

    fixtures = _try_fetch(api_key)
    if not fixtures and api_key_2:
        print("[WARN] Primary API-Football key failed, trying secondary", file=sys.stderr)
        fixtures = _try_fetch(api_key_2)
    return fixtures


def fetch_bsd_fixtures() -> Dict[str, dict]:
    """Fetch upcoming fixtures from BSD API for kickoff time lookup.

    Returns dict keyed by (home_team, away_team) for fast lookup.
    BSD covers 83+ leagues — replaces API-Football for kickoff lookup.
    """
    bsd_key = os.environ.get("BSD_API_KEY", "")
    if not bsd_key:
        return {}

    client = BSDClient(api_key=bsd_key)
    fixtures = {}

    # Fetch upcoming events for all BSD-covered leagues
    from market.bsd_client import KALSHI_TO_BSD_LEAGUE
    for league_ids in KALSHI_TO_BSD_LEAGUE.values():
        for league_id in league_ids:
            try:
                events = client.get_upcoming_events(league_id=league_id, limit=20)
                for event in events:
                    home = event.get("home_team", "")
                    away = event.get("away_team", "")
                    if home and away:
                        key = (home.lower(), away.lower())
                        fixtures[key] = event
            except Exception:
                continue

    return fixtures


def _find_kickoff_from_bsd(
    home: str, away: str, bsd_fixtures: Dict[str, dict]
) -> Optional[datetime]:
    """Find actual kickoff time from BSD fixtures by matching team names."""
    home_norm = home.lower().replace(".", "").replace("'", "").replace("-", " ")
    away_norm = away.lower().replace(".", "").replace("'", "").replace("-", " ")

    for (f_home, f_away), event in bsd_fixtures.items():
        if (home_norm in f_home or f_home in home_norm) and \
           (away_norm in f_away or f_away in away_norm):
            event_date = event.get("event_date", "")
            try:
                utc_time = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                return utc_time.astimezone(IST)
            except Exception:
                pass

    return None


def discover_matches() -> List[Dict]:
    """Discover all upcoming soccer matches from Kalshi.

    Uses BSD API (primary) or API-Football (fallback) for actual kickoff times.
    Returns sorted list of match dicts, soonest first.
    """
    cfg = load_config()
    now = datetime.now(IST)

    # Try BSD fixtures first (free, no quota, 83+ leagues)
    bsd_fixtures = fetch_bsd_fixtures()
    if bsd_fixtures:
        print(f"[INFO] Loaded {len(bsd_fixtures)} BSD fixtures for kickoff lookup", file=sys.stderr)

    # Fall back to API-Football if BSD unavailable
    api_football_fixtures = {}
    if not bsd_fixtures:
        api_key = os.environ.get("API_FOOTBALL_API_KEY", "")
        api_key_2 = os.environ.get("API_FOOTBALL_API_KEY_2", "")
        api_football_fixtures = fetch_api_football_fixtures(api_key, api_key_2)
        if api_football_fixtures:
            print(f"[INFO] Loaded {len(api_football_fixtures)} API-Football fixtures for kickoff lookup", file=sys.stderr)
        else:
            print("[WARN] No fixtures from BSD or API-Football — using 21:00 IST default kickoff", file=sys.stderr)

    client = KalshiClient(
        api_key=cfg.kalshi_api_key,
        private_key_pem=cfg.kalshi_private_key,
    )

    matches = []

    for series in SOCCER_SERIES:
        try:
            resp = client._request(
                "GET",
                "/events",
                params={"series_ticker": series, "limit": 50, "status": "open"},
            )
            if not resp or "events" not in resp:
                continue

            for event in resp["events"]:
                match = parse_kalshi_event(event, now, api_football_fixtures, bsd_fixtures)
                if match:
                    # Fetch actual market count (Kalshi /events doesn't include markets)
                    try:
                        markets_resp = client._request(
                            "GET", "/markets",
                            params={"event_ticker": match["event_ticker"], "limit": 50},
                        )
                        match["markets_count"] = len(markets_resp.get("markets", [])) if markets_resp else 0
                    except Exception:
                        match["markets_count"] = 0
                    matches.append(match)
                    time.sleep(0.1)

            # Rate limit: 1 req/sec
            time.sleep(0.5)

        except Exception as e:
            print(f"[WARN] Failed to fetch {series}: {e}", file=sys.stderr)
            continue

    # Sort by kickoff time
    matches.sort(key=lambda x: x["minutes_until"])

    return matches


# ── League tier mapping ────────────────────────────────────────────
# Higher tier = better data quality on Kalshi = tighter markets
LEAGUE_TIERS = {
    # Tier 1: UEFA Champions League
    "KXUCLGAME": 1.0,
    "KXUELGAME": 1.0,
    "KXUECLGAME": 1.0,
    "KXUEFAGAME": 1.0,
    "KXUEFANLGAME": 1.0,
    # Tier 2: Top 5 European leagues
    "KXPREMIERLEAGUE": 0.9,
    "KXPRIMERALIGAME": 0.9,
    "KXSERIEAGAME": 0.85,
    "KXMLSGAME": 0.8,
    # Tier 3: Strong European leagues
    "KXEREDIVISIEGAME": 0.75,
    "KXSUPERLIGGAME": 0.65,
    "KXBRASILEIROGAME": 0.7,
    "KXBRASILEIROBGAME": 0.6,
    "KXALLSVENSKANGAME": 0.55,
    "KXSCOTTISHPREMGAME": 0.55,
    "KXSLGREECEGAME": 0.5,
    "KXSWISSLEAGUEGAME": 0.55,
    # Tier 4: Other leagues
    "KXLIGAMXGAME": 0.5,
    "KXSAUDIPLGAME": 0.5,
    "KXTHAIL1GAME": 0.4,
    "KXUAEPLGAME": 0.45,
    "KXISLGAME": 0.4,
    "KXCHNSLGAME": 0.4,
    "KXKLEAGUEGAME": 0.5,
    "KXPERLIGA1GAME": 0.4,
    "KXSPBGAME": 0.4,
    "KXVENFUTVEGAME": 0.35,
    "KXQSTARSGAME": 0.35,
    "KXTACAPORTGAME": 0.35,
    "KXDENSUPERLIGAGAME": 0.4,
    "KXCLUBFGAME": 0.35,
    "KXASEANGAME": 0.3,
    "KXWIBPLGAME": 0.3,
    # Tier 5: Cups / lower
    "KXUSLGAME": 0.4,
    "KXUSOPENCUPGAME": 0.35,
    "KXSCOCUPGAME": 0.35,
    "KXUSLCUPGAME": 0.35,
    "KXARGNACBGAME": 0.35,
    "KXSERIEBGAME": 0.45,
}


def _score_timing(minutes_until: float) -> float:
    """Score timing: peak at ~3 hours out (180 min), falls off before/after.

    Returns 0.0-1.0.
    """
    if minutes_until < 30:
        return 0.1  # Too soon — bot might not initialize in time
    elif minutes_until < 60:
        return 0.4
    elif minutes_until < 120:
        return 0.7
    elif minutes_until < 240:
        return 1.0  # Sweet spot: 2-4 hours
    elif minutes_until < 360:
        return 0.8
    elif minutes_until < 480:
        return 0.5
    else:
        return 0.2  # Too far out — might not be worth waiting


# Leagues covered by BSD API (primary live data source — free, no quota, 83+ leagues)
# Replaces football-data.org as the free data source filter
FD_COVERED_SERIES = {
    # Tier 1: UEFA
    "KXUCLGAME", "KXCHAMPIONSLEAGUEGAME",
    "KXUELGAME", "KXUECLGAME", "KXUEFAGAME", "KXUEFANLGAME",
    # Tier 2: Top 5
    "KXPREMIERLEAGUE", "KXSERIEAGAME", "KXPRIMERALIGAME",
    "KXMLSGAME", "KXEREDIVISIEGAME", "KXSUPERLIGGAME",
    "KXBRASILEIROGAME", "KXBRASILEIROBGAME",
    # Tier 3: Strong European
    "KXALLSVENSKANGAME", "KXSCOTTISHPREMGAME",
    "KXSLGREECEGAME", "KXSWISSLEAGUEGAME",
    "KXDENSUPERLIGAGAME", "KXLIGAMXGAME",
    "KXSAUDIPLGAME", "KXKLEAGUEGAME",
    # Tier 4: Other
    "KXCHNSLGAME", "KXISLGAME",
    "KXPERLIGA1GAME", "KXVENFUTVEGAME",
    "KXQSTARSGAME", "KXSPBGAME", "KXWIBPLGAME",
    # Tier 5: Cups & Other
    "KXTACAPORTGAME", "KXUSLGAME", "KXUSOPENCUPGAME",
    "KXSCOCUPGAME", "KXARGNACBGAME",
    "KXCLUBFGAME", "KXWCGAME", "KXMENWORLDCUP",
    "KXASEANGAME",
}


def pick_best_match(matches: List[Dict]) -> Optional[Dict]:
    """Pick the single best match to trade today.

    Scoring:
      - Liquidity (40%): markets_count (more markets = tighter spreads)
      - League tier (30%): higher tier = better data quality
      - Timing (30%): sweet spot ~3 hours from now

    Only picks matches in leagues with free live data coverage (football-data.org).

    Returns the best match dict with 'score' field added, or None.
    """
    if not matches:
        return None

    now = datetime.now(IST)
    scored = []

    # Find max markets_count for normalization
    max_markets = max(m.get("markets_count", 0) for m in matches) or 1

    for match in matches:
        minutes_until = match["minutes_until"]

        # Skip matches starting in less than 30 minutes (too close)
        if minutes_until < 30:
            continue

        # Skip matches with no Kalshi markets (illiquid)
        markets_count = match.get("markets_count", 0)
        if markets_count == 0:
            continue

        # Skip matches in leagues without free live data coverage
        series = match.get("series", "")
        if series not in FD_COVERED_SERIES:
            continue

        # Liquidity score: more markets = more liquidity
        liquidity_score = min(markets_count / max(max_markets, 1), 1.0)

        # League tier score
        series = match.get("series", "")
        league_score = LEAGUE_TIERS.get(series, 0.4)

        # Timing score
        timing_score = _score_timing(minutes_until)

        # Weighted score
        total_score = liquidity_score * 0.4 + league_score * 0.3 + timing_score * 0.3

        scored.append({**match, "score": round(total_score, 4)})

    if not scored:
        return None

    # Pick highest score
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    print(
        f"[INFO] Best match: {best['home']} vs {best['away']} "
        f"(score={best['score']}, league={best['series']}, "
        f"kickoff_in={best['minutes_until']:.0f}min)",
        file=sys.stderr,
    )

    return best


def save_schedule(matches: List[Dict], output_path: str) -> None:
    """Save match schedule to JSON file."""
    schedule = {
        "generated_at": datetime.now(IST).isoformat(),
        "match_count": len(matches),
        "matches": matches,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(schedule, indent=2))
    print(f"[INFO] Saved {len(matches)} matches to {output_path}", file=sys.stderr)


def save_best_match(match: Dict, output_path: str) -> None:
    """Save the best match to a separate JSON file for the dispatch step."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(match, indent=2))
    print(f"[INFO] Best match saved to {output_path}", file=sys.stderr)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover upcoming soccer matches")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--best-output", help="Save best match to this path")
    parser.add_argument("--within-hours", type=float, default=24,
                        help="Only include matches within N hours (default: 24)")
    args = parser.parse_args()

    matches = discover_matches()

    # Filter by time window
    if args.within_hours:
        max_mins = args.within_hours * 60
        matches = [m for m in matches if m["minutes_until"] <= max_mins]

    # Pick best match
    best = pick_best_match(matches)

    # Output all matches to stdout
    print(json.dumps(matches, indent=2))

    # Save schedule
    if args.output:
        save_schedule(matches, args.output)

    # Save best match
    if best and args.best_output:
        save_best_match(best, args.best_output)

    print(f"[INFO] Found {len(matches)} matches within {args.within_hours}h",
          file=sys.stderr)


if __name__ == "__main__":
    main()
