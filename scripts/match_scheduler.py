#!/usr/bin/env python3
"""Match scheduler: discovers upcoming soccer matches from Kalshi + KickoffAPI.

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
from typing import Dict, List, Optional, Tuple

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market.kalshi_client import KalshiClient, SOCCER_SERIES
from config import load_config


def parse_kalshi_event(event: dict, now: datetime) -> Optional[Dict]:
    """Parse a Kalshi event into a match candidate dict.

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

    if not home or not away:
        return None

    # Parse kickoff from sub_title: "(Jul 23)" → Jul 23
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
                kickoff = datetime(year, month, day, 18, 0, tzinfo=timezone.utc)
        except Exception:
            pass

    if kickoff is None:
        return None

    minutes_until = (kickoff - now).total_seconds() / 60

    # Only include matches within next 24 hours that haven't started yet
    if minutes_until < -10 or minutes_until > 1440:
        return None

    # Extract markets from event
    markets = event.get("markets", [])

    return {
        "home": home,
        "away": away,
        "event_ticker": event_ticker,
        "series": series,
        "kickoff_utc": kickoff.isoformat(),
        "minutes_until": round(minutes_until, 1),
        "sub_title": sub_title,
        "markets_count": len(markets),
    }


def discover_matches() -> List[Dict]:
    """Discover all upcoming soccer matches from Kalshi.

    Returns sorted list of match dicts, soonest first.
    """
    cfg = load_config()
    now = datetime.now(timezone.utc)

    client = KalshiClient(
        api_key=cfg.kalshi_api_key,
        private_key_pem=cfg.kalshi_private_key,
        use_demo=False,  # Use production for discovery (read-only)
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
                match = parse_kalshi_event(event, now)
                if match:
                    matches.append(match)

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


def pick_best_match(matches: List[Dict]) -> Optional[Dict]:
    """Pick the single best match to trade today.

    Scoring:
      - Liquidity (40%): markets_count (more markets = tighter spreads)
      - League tier (30%): higher tier = better data quality
      - Timing (30%): sweet spot ~3 hours from now

    Returns the best match dict with 'score' field added, or None.
    """
    if not matches:
        return None

    now = datetime.now(timezone.utc)
    scored = []

    # Find max markets_count for normalization
    max_markets = max(m.get("markets_count", 0) for m in matches) or 1

    for match in matches:
        minutes_until = match["minutes_until"]

        # Skip matches starting in less than 30 minutes (too close)
        if minutes_until < 30:
            continue

        # Liquidity score: more markets = more liquidity
        markets_count = match.get("markets_count", 0)
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
