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
from config import get_config


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
    cfg = get_config()
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover upcoming soccer matches")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--within-hours", type=float, default=24,
                        help="Only include matches within N hours (default: 24)")
    args = parser.parse_args()

    matches = discover_matches()

    # Filter by time window
    if args.within_hours:
        max_mins = args.within_hours * 60
        matches = [m for m in matches if m["minutes_until"] <= max_mins]

    # Output to stdout
    print(json.dumps(matches, indent=2))

    # Optionally save to file
    if args.output:
        save_schedule(matches, args.output)

    print(f"[INFO] Found {len(matches)} matches within {args.within_hours}h",
          file=sys.stderr)


if __name__ == "__main__":
    main()
