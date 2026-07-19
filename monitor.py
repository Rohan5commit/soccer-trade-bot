#!/usr/bin/env python3
"""
Paper Trade Monitor - runs as background watchdog.
Checks every 2 minutes:
  1. Paper trader process alive
  2. Log file growing (no freeze)
  3. Market data flowing
  4. Kalshi API responding
  5. Balance sanity
  6. Auto-restart if dead
"""

import subprocess
import time
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BOT_DIR = Path(__file__).parent
LOG_FILE = BOT_DIR / "data" / "paper_trade_cloud.log"
STATE_FILE = BOT_DIR / "data" / "paper_signals" / "current_state.json"
MONITOR_LOG = BOT_DIR / "data" / "monitor.log"
CHECK_INTERVAL = 120  # seconds
MAX_LOG_AGE = 180  # if log not updated in 180s, trader is frozen
MIN_BALANCE = 100  # alert if balance drops below this


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(MONITOR_LOG, "a") as f:
        f.write(line + "\n")


def is_trader_alive() -> bool:
    """Check if paper_trade process is running in screen."""
    try:
        r = subprocess.run(
            ["screen", "-ls"],
            capture_output=True, text=True, timeout=5
        )
        return "paper_trade" in r.stdout
    except Exception:
        return False


def is_log_growing() -> bool:
    """Check if paper_trade_cloud.log was updated recently."""
    if not LOG_FILE.exists():
        return False
    mtime = LOG_FILE.stat().st_mtime
    age = time.time() - mtime
    return age < MAX_LOG_AGE


def get_log_tail(n: int = 5) -> str:
    """Get last n lines of paper trade log."""
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception:
        return "Cannot read log"


def get_state() -> dict:
    """Read current_state.json."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def check_errors_in_log() -> list:
    """Scan last 20 log lines for errors."""
    errors = []
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
            for line in lines[-20:]:
                if "ERROR" in line or "Traceback" in line or "Exception" in line:
                    errors.append(line.strip())
    except Exception:
        pass
    return errors


def check_kalshi_api() -> bool:
    """Quick health check on Kalshi API."""
    try:
        r = subprocess.run(
            ["python3", "-c", """
import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
from market.kalshi_client import KalshiClient
k = KalshiClient(api_key=os.environ['KALSHI_API_KEY'],
                  private_key_pem=os.environ['KALSHI_PRIVATE_KEY'], dry_run=False)
bal = k.get_balance()
print(f'BALANCE={bal}')
"""],
            capture_output=True, text=True, timeout=30,
            cwd=str(BOT_DIR)
        )
        for line in r.stdout.split("\n"):
            if line.startswith("BALANCE="):
                return True, float(line.split("=")[1])
        return False, 0
    except Exception:
        return False, 0


def restart_trader():
    """Kill and restart paper trade in screen."""
    log("RESTARTING paper trader...")
    try:
        subprocess.run(
            ["screen", "-S", "paper_trade", "-X", "quit"],
            capture_output=True, timeout=5
        )
    except Exception:
        pass
    time.sleep(2)
    subprocess.Popen(
        ["screen", "-dmS", "paper_trade", "bash", "-c",
         "source .venv/bin/activate && python run_paper_trade.py > data/paper_trade_cloud.log 2>&1"],
        cwd=str(BOT_DIR)
    )
    log("Paper trader restarted.")


def run_monitor():
    log("=" * 60)
    log("MONITOR STARTED - checking every %d seconds" % CHECK_INTERVAL)
    log("=" * 60)

    restart_count = 0
    max_restarts = 5
    last_balance = 0
    consecutive_errors = 0

    while True:
        try:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            issues = []

            # 1. Process alive?
            alive = is_trader_alive()
            if not alive:
                issues.append("PROCESS DEAD")
                if restart_count < max_restarts:
                    restart_trader()
                    restart_count += 1
                    log("Auto-restart #%d (max %d)" % (restart_count, max_restarts))
                    time.sleep(30)
                    continue
                else:
                    log("CRITICAL: Max restarts reached. Manual intervention needed.")
                    time.sleep(CHECK_INTERVAL)
                    continue

            # 2. Log growing?
            growing = is_log_growing()
            if not growing:
                issues.append("LOG FROZEN (no update in %ds)" % MAX_LOG_AGE)

            # 3. Errors in log?
            errors = check_errors_in_log()
            if errors:
                consecutive_errors += 1
                for e in errors[-3:]:
                    issues.append("LOG ERROR: %s" % e[:120])
            else:
                consecutive_errors = 0

            # 4. Kalshi API responding?
            api_ok, balance = check_kalshi_api()
            if not api_ok:
                issues.append("KALSHI API UNREACHABLE")
            else:
                if last_balance > 0 and balance < last_balance - 50:
                    issues.append("BALANCE DROP: $%.2f -> $%.2f" % (last_balance, balance))
                if balance < MIN_BALANCE:
                    issues.append("LOW BALANCE: $%.2f" % balance)
                last_balance = balance

            # 5. State file freshness?
            state = get_state()
            last_update = state.get("last_update", "")
            if last_update:
                try:
                    lu = datetime.fromisoformat(last_update)
                    age_s = (datetime.now(timezone.utc) - lu).total_seconds()
                    if age_s > MAX_LOG_AGE:
                        issues.append("STATE STALE (%ds old)" % int(age_s))
                except Exception:
                    pass

            # Report
            status = "OK" if not issues else "ISSUES(%d)" % len(issues)
            match = state.get("match", {})
            score = "%s %s-%s %s" % (
                match.get("home", "?"),
                match.get("home_score", "?"),
                match.get("away_score", "?"),
                match.get("away", "?")
            )
            pred = state.get("prediction", {})
            pred_str = "H%.1f%% D%.1f%% A%.1f%%" % (
                pred.get("home", 0) * 100,
                pred.get("draw", 0) * 100,
                pred.get("away", 0) * 100
            )

            log("[%s] %s | %s | %s | Bal=$%.2f | Restarts=%d" % (
                now, status, score, pred_str, balance, restart_count
            ))

            for issue in issues:
                log("  ISSUE: %s" % issue)

            if consecutive_errors >= 5:
                log("WARNING: %d consecutive error cycles" % consecutive_errors)

        except Exception as e:
            log("MONITOR EXCEPTION: %s" % str(e))

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_monitor()
