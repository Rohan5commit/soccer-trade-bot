#!/usr/bin/env python3
"""Watch proximity: checks if any match is starting soon, dispatches bot.

Called by watcher.yml every 30 minutes. Reads the schedule artifact,
finds matches within the 2-hour window, and triggers the bot workflow.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def load_schedule() -> list:
    """Load match schedule from artifact or discover fresh."""
    schedule_file = Path("data/schedule.json")

    if schedule_file.exists():
        try:
            data = json.loads(schedule_file.read_text())
            return data.get("matches", [])
        except Exception as e:
            print(f"[WARN] Failed to load schedule: {e}", file=sys.stderr)

    # Fallback: discover fresh
    print("[INFO] No schedule file, discovering fresh...", file=sys.stderr)
    try:
        result = subprocess.run(
            [sys.executable, "scripts/match_scheduler.py", "--within-hours", "4"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"[ERROR] Fresh discovery failed: {e}", file=sys.stderr)

    return []


def is_bot_already_running(event_ticker: str) -> bool:
    """Check if a bot workflow is already running for this match."""
    try:
        result = subprocess.run(
            ["gh", "run", "list", "--workflow=bot.yml", "--status=in_progress",
             "--limit=10", "--json=name,status,createdAt"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            runs = json.loads(result.stdout)
            for run in runs:
                if event_ticker in run.get("name", ""):
                    return True
    except Exception:
        pass
    return False


def dispatch_bot(match: dict) -> bool:
    """Dispatch the bot workflow via GitHub API."""
    event_ticker = match["event_ticker"]

    if is_bot_already_running(event_ticker):
        print(f"[INFO] Bot already running for {event_ticker}", file=sys.stderr)
        return False

    # Dispatch via gh CLI
    payload = json.dumps({
        "home": match["home"],
        "away": match["away"],
        "kickoff": match["kickoff_utc"],
        "event_ticker": event_ticker,
    })

    try:
        result = subprocess.run(
            [
                "gh", "workflow", "run", "bot.yml",
                "--repo", os.environ.get("GITHUB_REPOSITORY", "Rohan5commit/soccer-trade-bot"),
                "-f", f"home={match['home']}",
                "-f", f"away={match['away']}",
                "-f", f"kickoff={match['kickoff_utc']}",
                "-f", f"event_ticker={event_ticker}",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"[INFO] Dispatched bot for {match['home']} vs {match['away']}", file=sys.stderr)
            return True
        else:
            print(f"[ERROR] Dispatch failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Dispatch exception: {e}", file=sys.stderr)
        return False


def main():
    now = datetime.now(timezone.utc)
    print(f"[INFO] Watcher check at {now.isoformat()}", file=sys.stderr)

    matches = load_schedule()
    print(f"[INFO] Loaded {len(matches)} matches", file=sys.stderr)

    dispatched = []

    for match in matches:
        minutes_until = match.get("minutes_until", 9999)

        # Match must be starting within 2 hours and not already started
        if -10 <= minutes_until <= 120:
            print(f"[MATCH] {match['home']} vs {match['away']} "
                  f"in {minutes_until:.0f} min ({match['event_ticker']})",
                  file=sys.stderr)

            if dispatch_bot(match):
                dispatched.append(f"{match['home']} vs {match['away']}")

    # Output for GitHub Step Summary
    dispatched_str = ", ".join(dispatched) if dispatched else "none"
    print(f"dispatched={dispatched_str}")

    if not dispatched:
        print("[INFO] No matches within 2-hour window", file=sys.stderr)


if __name__ == "__main__":
    main()
