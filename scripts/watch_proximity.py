#!/usr/bin/env python3
"""Watch proximity: checks if any match is starting soon, dispatches bot.

Called by watcher.yml every 30 minutes. Reads the schedule artifact,
picks the SINGLE best match, and triggers the bot workflow.
Only dispatches future matches — never re-dispatches past matches.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

IST = timezone(timedelta(hours=5, minutes=30))


def load_schedule() -> dict:
    """Load match schedule from artifact. Returns {generated_at, matches}."""
    schedule_file = Path("data/schedule.json")

    if schedule_file.exists():
        try:
            return json.loads(schedule_file.read_text())
        except Exception as e:
            print(f"[WARN] Failed to load schedule: {e}", file=sys.stderr)

    return {}


def filter_future_matches(matches: List[Dict]) -> List[Dict]:
    """Only keep matches with positive minutes_until (not yet started).

    Also recalculates minutes_until from current time since schedule may be stale.
    """
    now = datetime.now(IST)
    future = []

    for m in matches:
        kickoff_str = m.get("kickoff_ist", "")
        if not kickoff_str:
            continue

        try:
            kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        except Exception:
            continue

        minutes_until = (kickoff - now).total_seconds() / 60

        # Only future matches (at least 10 min away, max 12 hours)
        if 10 <= minutes_until <= 720:
            future.append({**m, "minutes_until": round(minutes_until, 1)})

    return future


def _score_match(match: Dict) -> float:
    """Score a match for the watcher. Returns 0.0-1.0."""
    minutes_until = match.get("minutes_until", 9999)
    markets_count = match.get("markets_count", 0)

    # Timing: prefer 2-6 hours out
    if minutes_until < 30:
        timing = 0.1
    elif minutes_until < 60:
        timing = 0.4
    elif minutes_until < 120:
        timing = 0.7
    elif minutes_until < 240:
        timing = 1.0
    elif minutes_until < 360:
        timing = 0.8
    else:
        timing = 0.3

    # Liquidity: more markets = better
    liquidity = min(markets_count / 10, 1.0) if markets_count > 0 else 0.3

    return timing * 0.6 + liquidity * 0.4


def pick_best_match(matches: List[Dict]) -> Optional[Dict]:
    """Pick the single best match to trade. Returns None if nothing good."""
    if not matches:
        return None

    scored = [(m, _score_match(m)) for m in matches]
    scored.sort(key=lambda x: x[1], reverse=True)

    best, best_score = scored[0]
    print(
        f"[INFO] Best match: {best['home']} vs {best['away']} "
        f"(score={best_score:.2f}, {best['minutes_until']:.0f}min away)",
        file=sys.stderr,
    )
    return best


def is_bot_already_running() -> bool:
    """Check if ANY bot workflow is already running or queued."""
    for status in ("in_progress", "queued"):
        try:
            result = subprocess.run(
                ["gh", "run", "list", "--workflow=bot.yml", f"--status={status}",
                 "--limit=5", "--json=name,status"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                runs = json.loads(result.stdout)
                if runs:
                    return True
        except Exception:
            pass
    return False


def dispatch_bot(match: dict) -> bool:
    """Dispatch the bot workflow via GitHub API."""
    if is_bot_already_running():
        print(f"[INFO] Bot already running or queued — skipping", file=sys.stderr)
        return False

    try:
        result = subprocess.run(
            [
                "gh", "workflow", "run", "bot.yml",
                "--repo", os.environ.get("GITHUB_REPOSITORY", "Rohan5commit/soccer-trade-bot"),
                "-f", f"home={match['home']}",
                "-f", f"away={match['away']}",
                "-f", f"kickoff={match['kickoff_ist']}",
                "-f", f"event_ticker={match['event_ticker']}",
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
    now = datetime.now(IST)
    print(f"[INFO] Watcher check at {now.strftime('%Y-%m-%d %H:%M IST')}", file=sys.stderr)

    schedule_data = load_schedule()
    raw_matches = schedule_data.get("matches", [])
    generated_at = schedule_data.get("generated_at", "unknown")
    print(f"[INFO] Schedule generated at {generated_at}, loaded {len(raw_matches)} matches", file=sys.stderr)

    # Filter to future matches only
    matches = filter_future_matches(raw_matches)
    print(f"[INFO] {len(matches)} matches still upcoming (filtered from {len(raw_matches)})", file=sys.stderr)

    # Pick best match
    best = pick_best_match(matches)

    # Dispatch
    dispatched = "none"
    if best:
        if dispatch_bot(best):
            dispatched = f"{best['home']} vs {best['away']}"

    print(f"dispatched={dispatched}")


if __name__ == "__main__":
    main()
