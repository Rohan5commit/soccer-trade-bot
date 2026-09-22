#!/usr/bin/env python3
"""Watch proximity: checks if any match is starting soon, dispatches bot.

Called by watcher.yml. GitHub heavily throttles high-frequency crons
(observed */10 → ~3.6h average gaps, max ~6h), so a narrow 10-180 min
window is often missed. We therefore:
  - track matches up to INTEREST_MAX (360 min) out
  - dispatch immediately when a match is within DISPATCH_NOW_MAX (180 min)
  - otherwise sleep until SLEEP_TARGET (150 min) before kickoff, then dispatch

Only dispatches future matches — never re-dispatches past matches.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

IST = timezone(timedelta(hours=5, minutes=30))

# Match must be at least this far out to dispatch (bot needs init time)
MIN_BEFORE_KICKOFF = 10
# Dispatch without sleeping when kickoff is within this many minutes
# (bot.yml timeout 340 = 180 wait + 120 match + 40 buffer)
DISPATCH_NOW_MAX = 180
# Track matches up to this many minutes out (covers worst observed
# GitHub cron gap of ~6h so we don't skip a match entirely)
INTEREST_MAX = 360
# After sleeping, dispatch when this many minutes remain before kickoff
# (sweet spot: bot ready, markets typically open or opening)
SLEEP_TARGET = 150


def load_schedule() -> dict:
    """Load match schedule from artifact. Returns {generated_at, matches}."""
    schedule_file = Path("data/schedule.json")

    if schedule_file.exists():
        try:
            data = json.loads(schedule_file.read_text())
            # Stale check: if generated >22h ago, treat as missing (scheduler outage)
            gen = data.get("generated_at", "")
            if gen:
                try:
                    gen_dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
                    if gen_dt.tzinfo is None:
                        gen_dt = gen_dt.replace(tzinfo=timezone.utc)
                    age_h = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
                    if age_h > 22:
                        print(f"[CRITICAL] schedule.json stale {age_h:.1f}h old (>{22}h) — treating as missing", file=sys.stderr)
                        return {}
                except Exception:
                    pass
            return data
        except Exception as e:
            print(f"[WARN] Failed to load schedule: {e}", file=sys.stderr)

    return {}


def load_preferred_match() -> Optional[Dict]:
    """Load the scheduler's designated best match (best_match.json).

    The scheduler ranks all matches and picks ONE as the night's best.
    The watcher should honor that pick instead of re-scoring on its own.
    Returns None if no best-match artifact is available.
    """
    best_file = Path("data/best_match.json")

    if best_file.exists():
        try:
            return json.loads(best_file.read_text())
        except Exception as e:
            print(f"[WARN] Failed to load best match: {e}", file=sys.stderr)

    return None


def filter_future_matches(matches: List[Dict]) -> List[Dict]:
    """Only keep matches with positive minutes_until (not yet started).

    Also recalculates minutes_until from current time since schedule may be stale.
    """
    now = datetime.now(IST)
    future = []

    for m in matches:
        kickoff_str = m.get("kickoff_ist", "")
        if not kickoff_str:
            print(f"[INFO]   filtered: {m.get('home', '?')} vs {m.get('away', '?')} — no kickoff time", file=sys.stderr)
            continue

        try:
            kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        except Exception:
            print(f"[INFO]   filtered: {m.get('home', '?')} vs {m.get('away', '?')} — bad kickoff format", file=sys.stderr)
            continue

        minutes_until = (kickoff - now).total_seconds() / 60

        # Keep matches from MIN_BEFORE_KICKOFF out through INTEREST_MAX.
        # Beyond DISPATCH_NOW_MAX we sleep in wait_until_ready() rather than
        # dropping the match (GitHub cron gaps can exceed 3h).
        if minutes_until < MIN_BEFORE_KICKOFF:
            print(f"[INFO]   filtered: {m['home']} vs {m['away']} — too close ({minutes_until:.0f}min)", file=sys.stderr)
            continue
        if minutes_until > INTEREST_MAX:
            print(f"[INFO]   filtered: {m['home']} vs {m['away']} — too far ({minutes_until:.0f}min)", file=sys.stderr)
            continue

        # Keep matches even if markets_count==0 — bot polls for markets live
        # (Kalshi opens markets close to kickoff, scheduler may have run early)
        markets_count = m.get("markets_count", 0)
        if markets_count == 0:
            print(f"[INFO]   keep: {m['home']} vs {m['away']} — no markets yet (bot will poll)", file=sys.stderr)

        future.append({**m, "minutes_until": round(minutes_until, 1)})

    return future


def _score_match(match: Dict) -> float:
    """Score a match for the watcher. Returns 0.0-1.0."""
    minutes_until = match.get("minutes_until", 9999)
    markets_count = match.get("markets_count", 0)

    # Timing: prefer 30-90 min out (the dispatch window)
    if minutes_until < 30:
        timing = 0.3
    elif minutes_until < 60:
        timing = 0.7
    else:
        timing = 1.0  # Sweet spot: 60-90 min gives bot time to initialize

    # Liquidity: more markets = better
    liquidity = min(markets_count / 10, 1.0) if markets_count > 0 else 0.3

    return timing * 0.6 + liquidity * 0.4


def pick_best_match(matches: List[Dict]) -> Optional[Dict]:
    """Pick the single best match to trade. Returns None if nothing good."""
    if not matches:
        return None

    # Filter to leagues with free live data coverage
    from match_scheduler import FD_COVERED_SERIES
    covered = [m for m in matches if m.get("series", "") in FD_COVERED_SERIES]
    if not covered:
        print("[INFO] No matches in leagues with free live data coverage", file=sys.stderr)
        return None

    scored = [(m, _score_match(m)) for m in covered]
    scored.sort(key=lambda x: x[1], reverse=True)

    best, best_score = scored[0]
    print(
        f"[INFO] Best match: {best['home']} vs {best['away']} "
        f"(score={best_score:.2f}, {best['minutes_until']:.0f}min away)",
        file=sys.stderr,
    )
    return best


def prioritize(preferred: Optional[Dict], matches: List[Dict]) -> Optional[Dict]:
    """Return the scheduler's preferred match when it's still upcoming.

    Falls back to real-time scoring when there's no preferred match or it
    is no longer in the future window. Uses the match's event_ticker (or
    team names) to match against the filtered future matches so we carry
    the fresh minutes_until value.
    """
    if not preferred:
        return None

    preferred_ticker = preferred.get("event_ticker", "")
    preferred_home = preferred.get("home", "")
    preferred_away = preferred.get("away", "")

    for m in matches:
        if preferred_ticker and m.get("event_ticker") == preferred_ticker:
            print(
                f"[INFO] Honoring scheduler pick: {m['home']} vs {m['away']} "
                f"({m['minutes_until']:.0f}min away)",
                file=sys.stderr,
            )
            return m
        if m.get("home") == preferred_home and m.get("away") == preferred_away:
            print(
                f"[INFO] Honoring scheduler pick: {m['home']} vs {m['away']} "
                f"({m['minutes_until']:.0f}min away)",
                file=sys.stderr,
            )
            return m

    print(
        f"[INFO] Scheduler pick ({preferred_home} vs {preferred_away}) not in "
        f"future window — falling back to real-time scoring",
        file=sys.stderr,
    )
    return None


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


def _session_date(dt: datetime) -> str:
    """Map a datetime to its trading-session date.

    Sessions run from 10:00 IST to the next 10:00 IST (the scheduler
    generates the daily schedule at ~10:00 IST), so a match at 02:00 IST
    belongs to the same session as one at 18:00 IST the previous evening.
    """
    return (dt - timedelta(hours=10)).strftime("%Y-%m-%d")


MAX_MATCHES_PER_SESSION = 2


def _count_session_dispatches(session: str) -> int:
    """Count successful/active bot runs in this trading session."""
    try:
        result = subprocess.run(
            ["gh", "run", "list", "--workflow=bot.yml", "--limit=30",
             "--json=createdAt,conclusion,status"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return 0
        count = 0
        for run in json.loads(result.stdout):
            created = run.get("createdAt", "")
            if not created:
                continue
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                continue
            if _session_date(created_dt.astimezone(IST)) != session:
                continue
            conclusion = (run.get("conclusion") or "").lower()
            status = (run.get("status") or "").lower()
            # failed/cancelled runs don't consume the slot (allow retry)
            if conclusion and conclusion not in ("success", "in_progress", "queued", "waiting"):
                continue
            if not conclusion and status not in ("in_progress", "queued", "waiting", "completed"):
                continue
            count += 1
        return count
    except Exception:
        return 0


def was_dispatched_today() -> bool:
    """Check if this session already hit the match quota.

    Allows up to MAX_MATCHES_PER_SESSION successful runs per session
    (failed/cancelled runs don't consume a slot — retry allowed).
    """
    session = _session_date(datetime.now(IST))
    count = _count_session_dispatches(session)
    if count >= MAX_MATCHES_PER_SESSION:
        print(f"[INFO] Session {session} already has {count}/{MAX_MATCHES_PER_SESSION} "
              f"matches dispatched — skipping.", file=sys.stderr)
        return True
    return False


def was_match_already_dispatched(event_ticker: str) -> bool:
    """Check if we already dispatched a bot run for this event ticker.

    Uses the bot-log artifact name: bot.yml saves 'bot-log-{event_ticker}'.
    If that artifact exists, the bot was already dispatched for this match.
    """
    repo = os.environ.get("GITHUB_REPOSITORY", "Rohan5commit/soccer-trade-bot")
    artifact_name = f"bot-log-{event_ticker}"

    try:
        result = subprocess.run(
            [
                "gh", "api",
                f"repos/{repo}/actions/artifacts?name={artifact_name}&per_page=1",
                "--jq", ".total_count",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            count = int(result.stdout.strip() or "0")
            if count > 0:
                return True
    except Exception:
        pass

    # Fallback: check recent bot runs for this event ticker in display title
    try:
        result = subprocess.run(
            ["gh", "run", "list", "--workflow=bot.yml", "--limit=10",
             "--status=completed", "--json=displayTitle,conclusion"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            runs = json.loads(result.stdout)
            for run in runs:
                title = run.get("displayTitle", "")
                # Bot run titles contain the event ticker (e.g. "Paper Trade Bot - KXUCLGAME-...")
                if event_ticker in title:
                    return True
    except Exception:
        pass

    return False


def dispatch_bot(match: dict) -> bool:
    """Dispatch the bot workflow via GitHub API."""
    if is_bot_already_running():
        print(f"[INFO] Bot already running or queued — skipping", file=sys.stderr)
        return False

    # Session quota: up to MAX_MATCHES_PER_SESSION matches (BSD primary live
    # source, so the old API-Football 100 calls/day constraint no longer applies)
    if was_dispatched_today():
        print(f"[INFO] Session match quota reached — skipping", file=sys.stderr)
        return False

    # Check if this match was already dispatched
    event_ticker = match.get("event_ticker", "")
    if event_ticker and was_match_already_dispatched(event_ticker):
        print(f"[INFO] Match {event_ticker} already dispatched — skipping", file=sys.stderr)
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


def _minutes_until(kickoff_iso: str) -> float:
    kickoff = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=IST)
    return (kickoff - datetime.now(IST)).total_seconds() / 60


def wait_until_ready(match: Dict) -> Optional[Dict]:
    """Return match when it's within DISPATCH_NOW_MAX, sleeping if needed.

    Callers only invoke this for matches already inside INTEREST_MAX.
    Returns None if the match became too close/started during the wait
    (e.g. long GitHub cron delay landed us after kickoff).
    """
    kickoff_iso = match.get("kickoff_ist", "")
    if not kickoff_iso:
        return None

    try:
        mins = _minutes_until(kickoff_iso)
    except Exception as e:
        print(f"[WARN] wait_until_ready: bad kickoff {kickoff_iso}: {e}", file=sys.stderr)
        return None

    if mins < MIN_BEFORE_KICKOFF:
        print(f"[INFO] Match already too close after wait ({mins:.0f}min) — skip", file=sys.stderr)
        return None

    if mins > DISPATCH_NOW_MAX:
        sleep_min = mins - SLEEP_TARGET
        # Cap so the watcher job always finishes inside its timeout
        # (watcher.yml timeout-minutes: 360; setup ~2min + dispatch ~1min)
        max_sleep_min = 340
        if sleep_min > max_sleep_min:
            sleep_min = max_sleep_min
        if sleep_min > 0:
            print(
                f"[INFO] Kickoff in {mins:.0f}min (> {DISPATCH_NOW_MAX}) — "
                f"sleeping {sleep_min:.0f}min until ~{SLEEP_TARGET}min out",
                file=sys.stderr,
            )
            time.sleep(sleep_min * 60)

        try:
            mins = _minutes_until(kickoff_iso)
        except Exception:
            return None

        if mins < MIN_BEFORE_KICKOFF:
            print(f"[INFO] Match started during sleep ({mins:.0f}min) — skip", file=sys.stderr)
            return None
        if mins > DISPATCH_NOW_MAX:
            # Still too far (hit sleep cap) — let the next cron try again
            print(f"[INFO] Still {mins:.0f}min out after sleep cap — deferring", file=sys.stderr)
            return None

    fresh = {**match, "minutes_until": round(mins, 1)}
    print(
        f"[INFO] Ready to dispatch: {fresh['home']} vs {fresh['away']} "
        f"({fresh['minutes_until']}min out)",
        file=sys.stderr,
    )
    return fresh


def main():
    now = datetime.now(IST)
    print(f"[INFO] Watcher check at {now.strftime('%Y-%m-%d %H:%M IST')}", file=sys.stderr)

    schedule_data = load_schedule()
    raw_matches = schedule_data.get("matches", [])
    generated_at = schedule_data.get("generated_at", "unknown")
    print(f"[INFO] Schedule generated at {generated_at}, loaded {len(raw_matches)} matches", file=sys.stderr)

    # Filter to matches within INTEREST_MAX (includes not-yet-ready ones)
    matches = filter_future_matches(raw_matches)
    print(f"[INFO] {len(matches)} matches still upcoming (filtered from {len(raw_matches)})", file=sys.stderr)

    # Log each upcoming match for debugging
    for m in matches:
        print(
            f"[INFO]   upcoming: {m['home']} vs {m['away']} "
            f"(markets={m.get('markets_count', '?')}, kickoff_in={m.get('minutes_until', '?')}min)",
            file=sys.stderr,
        )

    preferred = load_preferred_match()

    # Prefer matches already inside the immediate dispatch window so we
    # never sleep past a match that's ready to trade right now.
    ready_now = [m for m in matches if m.get("minutes_until", 9999) <= DISPATCH_NOW_MAX]
    if ready_now:
        best = prioritize(preferred, ready_now) or pick_best_match(ready_now)
        if best:
            best = wait_until_ready(best)
    else:
        best = prioritize(preferred, matches) or pick_best_match(matches)
        if best:
            print(
                f"[INFO] Nearest eligible match is {best.get('minutes_until', '?')}min out "
                f"(>{DISPATCH_NOW_MAX}) — waiting for dispatch window",
                file=sys.stderr,
            )
            best = wait_until_ready(best)

    # Dispatch
    dispatched = "none"
    if best:
        if dispatch_bot(best):
            dispatched = f"{best['home']} vs {best['away']}"
    else:
        print("[INFO] No suitable match found for dispatch", file=sys.stderr)

    print(f"dispatched={dispatched}")


if __name__ == "__main__":
    main()
