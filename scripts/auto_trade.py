#!/usr/bin/env python3
"""Auto-trade lifecycle manager.

Checks for upcoming soccer matches, starts Lightning studio,
runs the paper trader, and stops the studio after the match ends.

Usage:
    python scripts/auto_trade.py check      # Check for matches, start/stop as needed
    python scripts/auto_trade.py status     # Check if bot is running
    python scripts/auto_trade.py start      # Force start studio + bot
    python scripts/auto_trade.py stop       # Force stop studio + bot
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cloudscraper

# Config
TEAMSPACE = "juy595711-org/deploy-model-project"
STUDIO_NAME = "soccer-trade-train"
DEFAULT_LEAGUE_WEIGHT = 0.5  # Default weight for any soccer league
MATCH_BUFFER_MINUTES = 120  # Start studio 2hr before kickoff
POST_MATCH_BUFFER_MINUTES = 30  # Stop 30min after expected end
KICKOFF_API_KEYS = [
    os.environ.get("KICKOFF_API_KEY", ""),
    os.environ.get("KICKOFF_API_KEY_2", ""),
]
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY", "")
BOT_LOG = Path("data/auto_trade.log")
STATE_FILE = Path("data/auto_trade_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BOT_LOG, mode="a"),
    ],
)
logger = logging.getLogger("auto_trade")

# Lightning SDK imports
try:
    from lightning_sdk import Studio, Machine
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    logger.warning("lightning-sdk not installed, CLI fallback enabled")


def run_cmd(cmd: str, timeout: int = 60) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def get_studio_status() -> str:
    """Check if studio is running, stopped, or unknown."""
    rc, out, err = run_cmd(f"lightning studio list --teamspace {TEAMSPACE} 2>&1")
    if rc != 0:
        return "unknown"

    for line in out.split("\n"):
        if STUDIO_NAME in line:
            if "Running" in line or "running" in line:
                return "running"
            elif "Stopped" in line or "stopped" in line:
                return "stopped"
    return "unknown"


def start_studio() -> bool:
    """Start the Lightning studio."""
    logger.info("Starting studio %s...", STUDIO_NAME)

    if HAS_SDK:
        try:
            studio = Studio(STUDIO_NAME, teamspace=TEAMSPACE)
            studio.start(Machine.CPU)
            logger.info("Studio started via SDK")
            return True
        except Exception as e:
            logger.warning("SDK start failed: %s, trying CLI", e)

    rc, out, err = run_cmd(
        f"lightning studio start --name {STUDIO_NAME} --teamspace {TEAMSPACE} 2>&1",
        timeout=120,
    )
    if rc == 0:
        logger.info("Studio started via CLI")
        return True
    else:
        logger.error("Failed to start studio: %s", err)
        return False


def stop_studio() -> bool:
    """Stop the Lightning studio."""
    logger.info("Stopping studio %s...", STUDIO_NAME)

    if HAS_SDK:
        try:
            studio = Studio(STUDIO_NAME, teamspace=TEAMSPACE)
            studio.stop()
            logger.info("Studio stopped via SDK")
            return True
        except Exception as e:
            logger.warning("SDK stop failed: %s, trying CLI", e)

    rc, out, err = run_cmd(
        f"lightning studio stop --name {STUDIO_NAME} --teamspace {TEAMSPACE} 2>&1",
        timeout=120,
    )
    if rc == 0:
        logger.info("Studio stopped via CLI")
        return True
    else:
        logger.error("Failed to stop studio: %s", err)
        return False


def configure_ssh() -> bool:
    """Regenerate SSH key for current studio session."""
    logger.info("Configuring SSH for Lightning studio...")
    rc, out, err = run_cmd("lightning ssh configure --overwrite 2>&1", timeout=60)
    if rc == 0:
        logger.info("SSH configured successfully: %s", out[:200])
    else:
        logger.error("SSH configure failed (rc=%d): %s | %s", rc, out[:300], err[:300])
    return rc == 0


def ssh_exec(cmd: str, timeout: int = 120, retries: int = 3) -> Tuple[bool, str]:
    """Execute a command on the Lightning studio via SSH with retries.

    Returns (success, output) where output is stdout on success or stderr on failure.
    """
    ssh_user = os.environ.get("LIGHTNING_SSH_USER", "")
    if not ssh_user:
        # Fallback: parse SSH config
        rc2, out2, _ = run_cmd("grep -A 2 'soccer-trade-train' ~/.ssh/config 2>/dev/null | grep User | awk '{print $2}'")
        ssh_user = out2.strip()

    if not ssh_user:
        logger.error("No SSH user found — cannot SSH to studio")
        return False, ""

    # Verify SSH key exists
    key_path = Path("~/.ssh/lightning_rsa").expanduser()
    if not key_path.exists():
        logger.error("SSH key not found at %s — run 'lightning ssh configure'", key_path)
        return False, ""

    for attempt in range(retries):
        ssh_cmd = (
            f'ssh -i ~/.ssh/lightning_rsa -o StrictHostKeyChecking=no '
            f'-o ConnectTimeout=30 {ssh_user}@ssh.lightning.ai "{cmd}"'
        )
        rc, out, err = run_cmd(ssh_cmd, timeout=timeout)
        if rc == 0:
            return True, out
        logger.warning(
            "SSH attempt %d/%d failed (rc=%d): %s",
            attempt + 1, retries, rc, (err or out or "unknown error")[:200],
        )
        if attempt < retries - 1:
            time.sleep(15)
    return False, err or out


def is_bot_running() -> bool:
    """Check if the paper trader is running on the studio."""
    ok, out = ssh_exec("screen -ls 2>/dev/null | grep paper_trade || echo 'not_running'")
    return "paper_trade" in out and "not_running" not in out


def start_bot() -> bool:
    """Start the paper trader on the studio via SSH (with retries)."""
    logger.info("Starting paper trader on studio...")

    # Kill any existing session (ignore failures)
    ssh_exec("screen -S paper_trade -X quit 2>/dev/null || true", timeout=30, retries=1)
    time.sleep(2)

    # Start new session
    cmd = (
        "cd /teamspace/studios/this_studio/soccer-trade-bot && "
        "screen -dmS paper_trade bash -c '"
        "source .env 2>/dev/null || true; "
        "export $(grep -v \"^#\" .env | xargs) 2>/dev/null || true; "
        "python3 run_paper_trade.py > data/paper_trade_cloud.log 2>&1'"
    )
    ok, out = ssh_exec(cmd, timeout=120, retries=3)
    if ok:
        logger.info("Paper trader SSH command sent successfully")
        time.sleep(10)
        running = is_bot_running()
        if running:
            logger.info("Paper trader confirmed running on studio")
        else:
            logger.warning("SSH succeeded but bot not detected in screen sessions — may still be starting")
        return running
    else:
        logger.error("Failed to start paper trader (all SSH retries exhausted): %s", out)
        return False


def stop_bot() -> bool:
    """Stop the paper trader on the studio."""
    logger.info("Stopping paper trader on studio...")
    ssh_exec("screen -S paper_trade -X quit 2>/dev/null || true")
    ssh_exec("pkill -f 'run_paper_trade.py' 2>/dev/null || true")
    return True


def fetch_kickoff_fixtures(date: str) -> List[Dict]:
    """Fetch fixtures from KickoffAPI for a given date."""
    keys = [k for k in KICKOFF_API_KEYS if k]
    if not keys:
        logger.warning("No KickoffAPI keys configured")
        return []

    session = cloudscraper.create_scraper()
    for key in keys:
        try:
            resp = session.get(
                "https://api.kickoffapi.com/api/v1/fixtures",
                params={"date": date},
                headers={"x-api-key": key, "User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", [])
            elif resp.status_code == 429:
                logger.warning("KickoffAPI rate limited on key %s...", key[:10])
                continue
            elif resp.status_code == 403:
                logger.warning("KickoffAPI blocked by Cloudflare (CI environment)")
                return []
        except Exception as e:
            logger.warning("KickoffAPI request failed: %s", e)
    return []


def fetch_kalshi_events() -> List[Dict]:
    """Fetch open soccer events from Kalshi."""
    if not KALSHI_API_KEY:
        return []

    # Import KalshiClient
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from market.kalshi_client import KalshiClient
        client = KalshiClient(
            api_key=KALSHI_API_KEY,
            private_key_pem=KALSHI_PRIVATE_KEY,
            dry_run=True,
            use_demo=True,
        )
    except Exception as e:
        logger.warning("Failed to init KalshiClient: %s", e)
        return []

    series = [
        "KXALLSVENSKANGAME", "KXBRASILEIROBGAME", "KXBRASILEIROGAME",
        "KXSUPERLIGGAME", "KXEREDIVISIEGAME", "KXPRIMERALIGAME",
        "KXCHAMPIONSLEAGUEGAME", "KXPREMIERLEAGUE",
        "KXMLSGAME", "KXSERIEAGAME", "KXSERIEBGAME",
        "KXLIGAMXGAME", "KXSAUDIPLGAME", "KXSCOTTISHPREMGAME",
        "KXSLGREECEGAME", "KXSWISSLEAGUEGAME", "KXTHAIL1GAME",
        "KXUAEPLGAME", "KXUCLGAME", "KXUELGAME", "KXUECLGAME",
        "KXUEFAGAME", "KXUEFANLGAME", "KXUSLGAME", "KXUSOPENCUPGAME",
        "KXPERLIGA1GAME", "KXSPBGAME", "KXVENFUTVEGAME",
        "KXQSTARSGAME", "KXTACAPORTGAME", "KXDENSUPERLIGAGAME",
        "KXISLGAME", "KXCHNSLGAME", "KXKLEAGUEGAME",
        "KXCLUBFGAME", "KXASEANGAME", "KXWIBPLGAME",
        "KXSCOCUPGAME", "KXUSLCUPGAME", "KXARGNACBGAME",
    ]

    events = []
    for s in series:
        try:
            resp = client._request("GET", "/events", params={
                "series_ticker": s, "limit": 50, "status": "open"
            })
            if resp and "events" in resp:
                events.extend(resp["events"])
            time.sleep(1)
        except Exception as e:
            logger.debug("Failed to fetch series %s: %s", s, e)

    return events


def find_best_match(fixtures: List[Dict], kalshi_events: List[Dict]) -> Optional[Dict]:
    """Find the best upcoming match to track.

    Returns dict with keys: fixture_id, home, away, kickoff_utc, league_id, kalshi_score
    """
    now = datetime.now(timezone.utc)
    candidates = []

    # Track which Kalshi events we've already matched to fixtures
    matched_kalshi = set()

    # First, try to match KickoffAPI fixtures to Kalshi events
    for f in fixtures:
        league_id = f.get("leagueId")
        status = f.get("statusShort", "NS")
        home = f.get("homeTeam", {}).get("name", "")
        away = f.get("awayTeam", {}).get("name", "")
        fixture_id = f.get("id", 0)
        date_str = f.get("date", "")

        if status == "FT":
            continue

        # Parse kickoff time
        try:
            kickoff = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            minutes_until = (kickoff - now).total_seconds() / 60
        except Exception:
            continue

        # Only consider matches starting within 24 hours
        if minutes_until < -10 or minutes_until > 1440:
            continue

        # Live matches are highest priority
        if status in ("1H", "2H", "HT", "ET", "PEN", "LIVE"):
            return {
                "fixture_id": fixture_id,
                "home": home,
                "away": away,
                "kickoff_utc": kickoff.isoformat(),
                "league_id": league_id,
                "status": status,
                "minutes_until": minutes_until,
                "kalshi_score": 0.5,
            }

        # Score Kalshi market liquidity
        kalshi_score = 0.0
        for i, event in enumerate(kalshi_events):
            if i in matched_kalshi:
                continue
            title = event.get("title", "").lower()
            if " vs " in title:
                teams = title.split(" vs ")
                t_a = teams[0].strip().lower()
                t_b = teams[1].strip().split(" winner")[0].strip().lower()
                h_words = [w for w in home.lower().split() if len(w) > 3]
                a_words = [w for w in away.lower().split() if len(w) > 3]
                # Check both orderings (home/away and away/home)
                fwd = any(w in t_a for w in h_words) and any(w in t_b for w in a_words)
                rev = any(w in t_b for w in h_words) and any(w in t_a for w in a_words)
                if fwd or rev:
                    kalshi_score = 0.8
                    matched_kalshi.add(i)
                    break

        time_bonus = min(1.0, max(0.0, 1.0 - minutes_until / 300)) if minutes_until > 0 else 0.5
        league_weight = DEFAULT_LEAGUE_WEIGHT
        combined = kalshi_score * 0.6 + league_weight * 0.2 + time_bonus * 0.2

        candidates.append({
            "fixture_id": fixture_id,
            "home": home,
            "away": away,
            "kickoff_utc": kickoff.isoformat(),
            "league_id": league_id,
            "status": status,
            "minutes_until": minutes_until,
            "kalshi_score": kalshi_score,
            "combined": combined,
        })

    # If no non-FT fixtures from KickoffAPI, create candidates from Kalshi events
    non_ft_fixtures = [f for f in fixtures if f.get("statusShort", "NS") != "FT"]
    if not non_ft_fixtures:
        for i, event in enumerate(kalshi_events):
            if i in matched_kalshi:
                continue
            title = event.get("title", "")
            event_ticker = event.get("event_ticker", "")
            if " vs " not in title:
                continue
            teams = title.split(" vs ")
            home = teams[0].strip()
            away = teams[1].strip().split(" winner")[0].strip()

            # Accept all soccer events with " vs " in title
            league_id = 0  # Generic — not used for filtering anymore

            # Parse kickoff from sub_title (format: "TWE vs FTC (Jul 23)")
            try:
                sub_title = event.get("sub_title", "")
                # Extract date from sub_title: "(Jul 23)" → Jul 23
                date_match = re.search(r'\((\w{3})\s+(\d{1,2})\)', sub_title)
                if date_match:
                    month_str = date_match.group(1).upper()
                    day = int(date_match.group(2))
                    month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
                    month = month_map.get(month_str, 0)
                    if month > 0:
                        year = now.year
                        # Handle year rollover (Dec → Jan)
                        if month < now.month - 6:
                            year += 1
                        elif month > now.month + 6:
                            year -= 1
                        kickoff = datetime(year, month, day, 18, 0, tzinfo=timezone.utc)
                        minutes_until = (kickoff - now).total_seconds() / 60
                        if minutes_until < -10 or minutes_until > 1440:
                            continue
                        candidates.append({
                            "fixture_id": hash(event_ticker) % 1000000,
                            "home": home,
                            "away": away,
                            "kickoff_utc": kickoff.isoformat(),
                            "league_id": league_id,
                            "status": "NS",
                            "minutes_until": minutes_until,
                            "kalshi_score": 0.8,
                            "combined": 0.8 * 0.6 + DEFAULT_LEAGUE_WEIGHT * 0.2 + 0.5 * 0.2,
                        })
            except Exception:
                pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["combined"], reverse=True)
    return candidates[0]


def load_state() -> Dict:
    """Load auto-trade state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: Dict) -> None:
    """Save auto-trade state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def cmd_check():
    """Check for upcoming matches and manage studio lifecycle."""
    logger.info("=" * 60)
    logger.info("AUTO-TRADE CHECK")
    logger.info("=" * 60)

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    state = load_state()

    # Fetch data
    logger.info("Fetching Kalshi events for match discovery...")
    kalshi_events = fetch_kalshi_events()
    logger.info("Found %d Kalshi events", len(kalshi_events))

    # Try KickoffAPI (may fail on Cloudflare-protected environments)
    logger.info("Fetching KickoffAPI fixtures for %s...", today)
    fixtures = fetch_kickoff_fixtures(today)
    logger.info("Found %d KickoffAPI fixtures", len(fixtures))

    # Find best match (prioritize Kalshi, fallback to KickoffAPI)
    best = find_best_match(fixtures, kalshi_events)

    if not best:
        logger.info("No matches starting within 4 hours")

        # Log next upcoming match for visibility
        all_candidates = []
        for f in fixtures:
            league_id = f.get("leagueId")
            status = f.get("statusShort", "NS")
            if status == "FT":
                continue
            try:
                kickoff = datetime.fromisoformat(f.get("date", "").replace("Z", "+00:00"))
                mins = (kickoff - now).total_seconds() / 60
                if mins > 0:
                    all_candidates.append((f.get("homeTeam", {}).get("name", ""), f.get("awayTeam", {}).get("name", ""), mins))
            except Exception:
                pass

        # Also check Kalshi events for upcoming matches
        for event in kalshi_events:
            title = event.get("title", "")
            series = event.get("series_ticker", "")
            if " vs " in title:
                teams = title.split(" vs ")
                home = teams[0].strip()
                away = teams[1].strip().split(" winner")[0].strip()
                # Estimate kickoff from sub_title
                sub_title = event.get("sub_title", "")
                date_match = re.search(r'\((\w{3})\s+(\d{1,2})\)', sub_title)
                if date_match:
                    try:
                        month_str = date_match.group(1).upper()
                        day = int(date_match.group(2))
                        month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                                    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
                        month = month_map.get(month_str, 0)
                        if month > 0:
                            year = now.year
                            if month < now.month - 6:
                                year += 1
                            elif month > now.month + 6:
                                year -= 1
                            kickoff = datetime(year, month, day, 18, 0, tzinfo=timezone.utc)
                            mins = (kickoff - now).total_seconds() / 60
                            if mins > 0:
                                all_candidates.append((home, away, mins))
                    except Exception:
                        pass

        if all_candidates:
            all_candidates.sort(key=lambda x: x[2])
            home, away, mins = all_candidates[0]
            logger.info(
                "Next match: %s vs %s in %.1f hours — studio will start when within 2hr",
                home, away, mins / 60,
            )

        # If studio is running and no match, stop it
        status = get_studio_status()
        if status == "running" and state.get("match_active"):
            logger.info("Match ended, stopping studio...")
            stop_bot()
            stop_studio()
            state["match_active"] = False
            state["last_stop"] = now.isoformat()
            save_state(state)
        elif status == "running" and not state.get("match_active"):
            # Studio running but no active match — check if bot is running
            if is_bot_running():
                logger.info("Bot is running but no match detected. Waiting...")
            else:
                logger.info("Studio running but bot not active. Will stop to save credits.")
                stop_studio()
        return

    logger.info(
        "Best match: %s vs %s (league %s, status=%s, %.1fh until kickoff)",
        best["home"], best["away"], best["league_id"],
        best["status"], best["minutes_until"] / 60,
    )

    # Determine action based on match timing
    minutes_until = best["minutes_until"]
    status = get_studio_status()

    if best["status"] in ("1H", "2H", "HT", "ET", "PEN", "LIVE"):
        # Match is LIVE
        logger.info("Match is LIVE!")
        if status != "running":
            logger.info("Starting studio for live match...")
            configure_ssh()
            if start_studio():
                logger.info("Waiting 90s for studio to fully boot...")
                time.sleep(90)
                configure_ssh()
                start_bot()
                state["match_active"] = True
            else:
                logger.error("Failed to start studio for live match")
        elif not is_bot_running():
            logger.info("Studio running but bot not active, starting bot...")
            configure_ssh()
            start_bot()
            state["match_active"] = True
        else:
            state["match_active"] = True

        state["current_match"] = f"{best['home']} vs {best['away']}"
        state["match_status"] = best["status"]
        state["last_check"] = now.isoformat()
        save_state(state)

    elif minutes_until <= MATCH_BUFFER_MINUTES and minutes_until > -POST_MATCH_BUFFER_MINUTES:
        # Match starting within 2 hours
        logger.info("Match starting in %.1f minutes, starting studio...", minutes_until)
        if status != "running":
            configure_ssh()
            if start_studio():
                logger.info("Waiting 90s for studio to fully boot...")
                time.sleep(90)
                configure_ssh()
                start_bot()
                state["match_active"] = True
            else:
                logger.error("Failed to start studio")
        elif not is_bot_running():
            # Auto-recovery: studio running but bot crashed — restart it
            logger.info("Studio running but bot not active, restarting bot...")
            configure_ssh()
            start_bot()
            state["match_active"] = True
        else:
            state["match_active"] = True

        state["current_match"] = f"{best['home']} vs {best['away']}"
        state["match_kickoff"] = best["kickoff_utc"]
        state["last_check"] = now.isoformat()
        save_state(state)

    elif minutes_until < -POST_MATCH_BUFFER_MINUTES:
        # Match ended (more than 30min ago)
        logger.info("Match ended (%.1f min ago), stopping studio...", abs(minutes_until))
        if status == "running":
            stop_bot()
            stop_studio()
        state["match_active"] = False
        state["current_match"] = ""
        state["last_stop"] = now.isoformat()
        save_state(state)

    else:
        # Match is > 2 hours away
        logger.info("Match is %.1f hours away, waiting...", minutes_until / 60)
        state["match_active"] = False
        state["current_match"] = f"{best['home']} vs {best['away']}"
        state["match_kickoff"] = best["kickoff_utc"]
        state["last_check"] = now.isoformat()
        save_state(state)


def cmd_status():
    """Check current status."""
    status = get_studio_status()
    bot_running = False
    if status == "running":
        configure_ssh()
        bot_running = is_bot_running()

    state = load_state()
    print(json.dumps({
        "studio_status": status,
        "bot_running": bot_running,
        "match_active": state.get("match_active", False),
        "current_match": state.get("current_match", ""),
        "last_check": state.get("last_check", ""),
    }, indent=2))


def cmd_start():
    """Force start studio and bot."""
    configure_ssh()
    if start_studio():
        logger.info("Waiting 90s for studio to fully boot...")
        time.sleep(90)
        configure_ssh()
        if start_bot():
            print("Studio and bot started successfully")
        else:
            print("Studio started but bot failed to start")
    else:
        print("Failed to start studio")


def cmd_stop():
    """Force stop studio and bot."""
    stop_bot()
    stop_studio()
    state = load_state()
    state["match_active"] = False
    state["current_match"] = ""
    state["match_kickoff"] = ""
    state["match_status"] = ""
    state["last_stop"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print("Studio and bot stopped")


def main():
    if len(sys.argv) < 2:
        print("Usage: auto_trade.py [check|status|start|stop]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "check":
        cmd_check()
    elif cmd == "status":
        cmd_status()
    elif cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
