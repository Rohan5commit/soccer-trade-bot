"""Build training dataset from multiple sources.

Pulls from 8 data sources to create ~2.1M game-state snapshots:
- StatsBomb Open Data via GitHub (~50K snapshots)
- Understat (2010-2026, 6 leagues) (~550K snapshots)
- European Soccer Database (Kaggle) (~1.37M snapshots)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from model.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Timeout helper (replaces signal.SIGALRM for Docker compatibility)
def _run_with_timeout(func, timeout_sec, *args, **kwargs):
    """Run func with a timeout. Returns (result, timed_out)."""
    result = [None]
    timed_out = [False]

    def _target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            result[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        timed_out[0] = True
    return result[0], timed_out[0]


def fetch_statsbomb_github(max_matches: int = 500) -> pd.DataFrame:
    """Fetch StatsBomb open data via GitHub raw content API.

    Strategy:
    1. Shallow-clone only competitions.json + matches/ (tiny files)
    2. Parse to find match IDs for major leagues
    3. Download event files via raw.githubusercontent.com (parallel)

    Args:
        max_matches: Maximum number of matches to process.

    Returns:
        DataFrame with match snapshots at 5-minute intervals.
    """
    logger.info("Fetching StatsBomb data from GitHub...")

    clone_dir = Path("/tmp/statsbomb_open_data")
    events_dir = Path("/tmp/statsbomb_events")
    events_dir.mkdir(exist_ok=True)

    # Step 1: Get match metadata (competitions + matches JSON)
    def _do_clone():
        if (clone_dir / "data" / "competitions.json").exists():
            return  # already have metadata
        if clone_dir.exists():
            import shutil
            shutil.rmtree(clone_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:limit=1m",
             "--sparse", "https://github.com/statsbomb/open-data.git",
             str(clone_dir)],
            capture_output=True, timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "set",
             "--skip-checks", "data/competitions.json", "data/matches"],
            capture_output=True, timeout=60,
        )

    _, timed_out = _run_with_timeout(_do_clone, 180)
    if timed_out:
        logger.warning("StatsBomb metadata clone timed out")
        return pd.DataFrame()

    if not (clone_dir / "data" / "competitions.json").exists():
        logger.warning("StatsBomb data not found after clone")
        return pd.DataFrame()

    # Step 2: Find match IDs for priority leagues
    PRIORITY_COMP_IDS = {9, 11, 12, 16, 20, 37, 43, 44, 1}  # BuLi, EPL, LaLiga, UCL, SerieA, L1, etc.

    with open(clone_dir / "data" / "competitions.json") as f:
        competitions = json.load(f)

    all_match_ids = []
    for comp in competitions:
        comp_id = comp.get("competition_id", -1)
        season_id = comp.get("season_id", -1)
        matches_path = clone_dir / "data" / "matches" / str(comp_id) / f"{season_id}.json"
        if not matches_path.exists():
            continue
        try:
            with open(matches_path) as f:
                matches = json.load(f)
            for m in matches:
                all_match_ids.append({
                    "match_id": m["match_id"],
                    "comp_id": comp_id,
                    "comp_name": comp.get("competition_name", ""),
                })
        except Exception:
            continue

    # Sort: priority leagues first
    all_match_ids.sort(key=lambda x: (x["comp_id"] not in PRIORITY_COMP_IDS,))
    selected = all_match_ids[:max_matches]
    logger.info("StatsBomb: %d total matches, selected %d from priority leagues",
               len(all_match_ids), len(selected))

    # Step 3: Download event files via raw GitHub URL (parallel)
    existing = {fn.replace(".json", "") for fn in os.listdir(events_dir) if fn.endswith(".json")}
    to_fetch = [m for m in selected if str(m["match_id"]) not in existing]
    logger.info("StatsBomb: %d events to fetch, %d already cached", len(to_fetch), len(existing))

    def _fetch_event(mid):
        url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{mid}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(events_dir / f"{mid}.json", "wb") as f:
                f.write(data)
            return True
        except Exception:
            return False

    if to_fetch:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_event, m["match_id"]): m for m in to_fetch}
            done = 0
            for future in as_completed(futures):
                done += 1
                if done % 100 == 0:
                    logger.info("StatsBomb: fetched %d/%d events", done, len(to_fetch))

    # Step 4: Parse events into snapshots
    return _parse_downloaded_events(selected, events_dir)


def _parse_downloaded_events(selected_matches: list, events_dir: Path) -> pd.DataFrame:
    """Parse downloaded StatsBomb event files into training snapshots."""
    match_lookup = {str(m["match_id"]): m for m in selected_matches}
    all_snapshots = []

    for fn in os.listdir(events_dir):
        if not fn.endswith(".json"):
            continue
        mid = fn.replace(".json", "")
        if mid not in match_lookup:
            continue

        try:
            with open(events_dir / fn) as f:
                events = json.load(f)
        except Exception:
            continue

        snapshots = _build_snapshots_from_sb_events(events, match_lookup[mid], "")
        all_snapshots.extend(snapshots)

    df = pd.DataFrame(all_snapshots) if all_snapshots else pd.DataFrame()
    logger.info("StatsBomb GitHub: %d snapshots from %d matches",
               len(df), len(set(s["match_id"] for s in all_snapshots)))
    return df


def _build_snapshots_from_sb_events(
    events: List[Dict], match: Dict, comp_name: str,
) -> List[Dict]:
    """Build 5-minute interval snapshots from StatsBomb events JSON."""
    match_id = str(match.get("match_id", ""))

    # Derive home/away team IDs from Starting XI events
    home_team_id = None
    away_team_id = None
    for e in events:
        if e.get("type", {}).get("name") == "Starting XI":
            team = e.get("team", {})
            tid = team.get("id") if isinstance(team, dict) else None
            if home_team_id is None:
                home_team_id = tid
            else:
                away_team_id = tid
                break

    # Count final score from events
    home_score_final = 0
    away_score_final = 0
    for e in events:
        if e.get("type", {}).get("name") == "Shot":
            shot = e.get("shot", {})
            outcome = shot.get("outcome", {})
            outcome_name = outcome.get("name", "") if isinstance(outcome, dict) else str(outcome)
            if outcome_name == "Goal":
                team_id = e.get("team", {}).get("id") if isinstance(e.get("team"), dict) else None
                if team_id == home_team_id:
                    home_score_final += 1
                elif team_id == away_team_id:
                    away_score_final += 1

    if home_score_final > away_score_final:
        target = 0
    elif home_score_final == away_score_final:
        target = 1
    else:
        target = 2

    # Process events chronologically
    home_goals = []
    away_goals = []
    home_shots_ot = []
    away_shots_ot = []
    home_xg = []
    away_xg = []
    home_cards = []
    away_cards = []

    for event in events:
        minute = event.get("minute", 0)
        team_id = event.get("team", {}).get("id") if isinstance(event.get("team"), dict) else event.get("team_id")
        is_home = team_id == home_team_id

        event_type = event.get("type", {})
        type_name = event_type.get("name", "") if isinstance(event_type, dict) else str(event_type)

        if type_name == "Shot":
            shot = event.get("shot", {})
            outcome = shot.get("outcome", {})
            outcome_name = outcome.get("name", "") if isinstance(outcome, dict) else str(outcome)
            xg = shot.get("statsbomb_xg", 0) or 0

            if is_home:
                home_xg.append((minute, xg))
                if outcome_name in ("Goal", "Saved", "Post", "Woodwork"):
                    home_shots_ot.append(minute)
                if outcome_name == "Goal":
                    home_goals.append(minute)
            else:
                away_xg.append((minute, xg))
                if outcome_name in ("Goal", "Saved", "Post", "Woodwork"):
                    away_shots_ot.append(minute)
                if outcome_name == "Goal":
                    away_goals.append(minute)

        elif type_name == "Foul Committed":
            card = event.get("foul_committed", {}).get("card", {})
            card_name = card.get("name", "") if isinstance(card, dict) else ""
            if card_name in ("Yellow Card", "Red Card", "Second Yellow"):
                if is_home:
                    home_cards.append(minute)
                else:
                    away_cards.append(minute)

    # Build snapshots at 5-minute intervals
    snapshots = []
    for clock in range(0, 95, 5):
        h_goals = sum(1 for m in home_goals if m <= clock)
        a_goals = sum(1 for m in away_goals if m <= clock)
        score_diff = h_goals - a_goals

        h_sot = sum(1 for m in home_shots_ot if m <= clock)
        a_sot = sum(1 for m in away_shots_ot if m <= clock)

        h_xg = sum(x for m, x in home_xg if m <= clock)
        a_xg = sum(x for m, x in away_xg if m <= clock)

        h_cards = sum(1 for m in home_cards if m <= clock)
        a_cards = sum(1 for m in away_cards if m <= clock)

        goals_last10 = sum(1 for m in home_goals + away_goals if clock - 10 < m <= clock)
        cards_last15 = sum(1 for m in home_cards + away_cards if clock - 15 < m <= clock)

        # Running xG momentum (last 15 min window)
        h_xg_recent = sum(x for m, x in home_xg if clock - 15 < m <= clock)
        a_xg_recent = sum(x for m, x in away_xg if clock - 15 < m <= clock)
        momentum = h_xg_recent - a_xg_recent

        snapshots.append({
            "match_id": match_id,
            "source": "statsbomb",
            "clock_minutes": float(clock),
            "score_diff": float(score_diff),
            "is_extra_time": float(clock > 90),
            "home_red_cards": float(h_cards),
            "away_red_cards": float(a_cards),
            "home_pressure_score": 0.5,
            "goals_in_last_10min": float(goals_last10),
            "home_shots_on_target": float(h_sot),
            "away_shots_on_target": float(a_sot),
            "home_xg_running": float(h_xg),
            "away_xg_running": float(a_xg),
            "score_diff_x_time_remaining": float(score_diff * max(90 - clock, 0)),
            "home_elo": 1500.0,
            "away_elo": 1500.0,
            "elo_diff": 0.0,
            "home_form_pts": 0.0,
            "away_form_pts": 0.0,
            "h2h_home_winrate": 0.4,
            "is_home_game": 1.0,
            "referee_cards_per_game": 3.5,
            "home_squad_value_EUR": 5e8,
            "away_squad_value_EUR": 5e8,
            "squad_value_ratio": 1.0,
            "home_injuries_count": 0.0,
            "away_injuries_count": 0.0,
            "home_press_pct": 0.5,
            "away_press_pct": 0.5,
            "home_xg_last5": float(h_xg),
            "away_xg_last5": float(a_xg),
            "home_xga_last5": float(a_xg),
            "away_xga_last5": float(h_xg),
            "competition_tier": 2.0,
            "match_importance": 0.5,
            "days_since_last_match_home": 7.0,
            "days_since_last_match_away": 7.0,
            "goals_last_15min": float(goals_last10),
            "cards_last_15min": float(cards_last15),
            "score_diff_squared": float(score_diff ** 2),
            "momentum_shift": float(momentum),
            "target": target,
        })

    return snapshots


def fetch_understat(league: str = "EPL") -> pd.DataFrame:
    """Fetch xG data from Understat.

    Args:
        league: League name (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL).

    Returns:
        DataFrame with match-level xG data.
    """
    logger.info("Fetching Understat data for %s...", league)
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://understat.com/league/{league}"
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=60)
        soup = BeautifulSoup(resp.text, "lxml")

        scripts = soup.find_all("script")
        for script in scripts:
            if "teamsData" in str(script):
                import re
                text = str(script)
                match = re.search(r"var teamsData\s*=\s*JSON\.parse\('(.+?)'\)", text)
                if match:
                    data = json.loads(match.group(1).encode().decode("unicode_escape"))
                    return _parse_understat_data(data, league)

        logger.warning("Could not parse Understat data for %s", league)
        return pd.DataFrame()

    except Exception as e:
        logger.error("Failed to fetch Understat data: %s", e)
        return pd.DataFrame()


def _parse_understat_data(data: dict, league: str) -> pd.DataFrame:
    """Parse Understat JSON data into DataFrame."""
    rows = []
    for team_id, team_data in data.items():
        for match in team_data.get("history", []):
            rows.append({
                "home_team": match.get("h", {}).get("title", ""),
                "away_team": match.get("a", {}).get("title", ""),
                "home_xg": float(match.get("xG", {}).get("h", 0)),
                "away_xg": float(match.get("xG", {}).get("a", 0)),
                "home_goals": int(match.get("goals", {}).get("h", 0)),
                "away_goals": int(match.get("goals", {}).get("a", 0)),
                "league": league,
                "date": match.get("date", ""),
            })
    return pd.DataFrame(rows)


def fetch_european_soccer_db() -> pd.DataFrame:
    """Load European Soccer Database from local CSV.

    Returns:
        DataFrame with match data.
    """
    csv_path = Path("data/Match.csv")
    if not csv_path.exists():
        logger.warning("European Soccer Database not found at %s", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    logger.info("European Soccer DB: %d matches", len(df))
    return df


def generate_synthetic_data(
    n_matches: int = 1000,
    snapshots_per_match: int = 19,
    output_path: str = "data/train.parquet",
) -> pd.DataFrame:
    """Generate synthetic training data for pipeline testing.

    Args:
        n_matches: Number of synthetic matches.
        snapshots_per_match: Snapshots per match (19 = every 5 min for 90 min).
        output_path: Where to save the parquet file.

    Returns:
        DataFrame with 39 features + target + match_id.
    """
    rng = np.random.default_rng(42)
    rows = []

    for match_idx in range(n_matches):
        match_id = f"synth_{match_idx:06d}"

        home_elo = rng.normal(1500, 200)
        away_elo = rng.normal(1500, 200)
        elo_diff = home_elo - away_elo
        home_form = rng.integers(0, 16)
        away_form = rng.integers(0, 16)
        h2h = rng.uniform(0.2, 0.8)
        home_value = rng.lognormal(18, 0.5)
        away_value = rng.lognormal(18, 0.5)
        value_ratio = home_value / max(away_value, 1)
        home_injuries = rng.integers(0, 6)
        away_injuries = rng.integers(0, 6)
        home_press = rng.uniform(0.3, 0.7)
        away_press = rng.uniform(0.3, 0.7)
        comp_tier = rng.choice([1, 2, 3], p=[0.1, 0.6, 0.3])
        importance = rng.uniform(0.1, 1.0)
        days_home = rng.integers(2, 14)
        days_away = rng.integers(2, 14)
        ref_cards = rng.uniform(2.5, 5.0)

        home_strength = elo_diff / 400 + home_form / 15 + 0.1
        home_win_prob = 1 / (1 + 10 ** (-home_strength))
        draw_prob = 0.25
        away_win_prob = 1 - home_win_prob - draw_prob

        probs = np.array([home_win_prob, draw_prob, away_win_prob])
        probs = np.clip(probs, 0.01, None)
        probs /= probs.sum()
        outcome = rng.choice([0, 1, 2], p=probs)

        if outcome == 0:
            final_home = rng.integers(1, 5)
            final_away = rng.integers(0, final_home)
        elif outcome == 2:
            final_away = rng.integers(1, 5)
            final_home = rng.integers(0, final_away)
        else:
            final_home = rng.integers(0, 4)
            final_away = final_home

        for snap_idx in range(snapshots_per_match):
            clock = snap_idx * 5
            if clock > 90:
                clock = 90

            progress = clock / 90
            home_score = int(final_home * progress * rng.uniform(0.8, 1.2))
            away_score = int(final_away * progress * rng.uniform(0.8, 1.2))
            home_score = min(home_score, final_home)
            away_score = min(away_score, final_away)
            score_diff = home_score - away_score

            home_xg = home_score * rng.uniform(0.8, 1.5) + rng.normal(0, 0.3)
            away_xg = away_score * rng.uniform(0.8, 1.5) + rng.normal(0, 0.3)
            home_xg = max(home_xg, 0)
            away_xg = max(away_xg, 0)

            home_xg_run = home_xg * progress
            away_xg_run = away_xg * progress

            pressure = rng.uniform(0.3, 0.7)
            if score_diff > 0:
                pressure = rng.uniform(0.5, 0.8)
            elif score_diff < 0:
                pressure = rng.uniform(0.2, 0.5)

            shots_sot = rng.integers(0, 5)
            red_cards_h = 1 if rng.random() < 0.05 else 0
            red_cards_a = 1 if rng.random() < 0.05 else 0

            xg5_home = home_xg * rng.uniform(0.5, 1.5)
            xg5_away = away_xg * rng.uniform(0.5, 1.5)
            xga5_home = away_xg * rng.uniform(0.5, 1.5)
            xga5_away = home_xg * rng.uniform(0.5, 1.5)

            goals_window = rng.integers(0, max(1, int(final_home + final_away)))
            cards_window = rng.integers(0, 4)
            momentum = rng.normal(0, 0.5)

            row = {
                "match_id": match_id,
                "source": "synthetic",
                "clock_minutes": float(clock),
                "score_diff": float(score_diff),
                "is_extra_time": float(clock > 90),
                "home_red_cards": float(red_cards_h),
                "away_red_cards": float(red_cards_a),
                "home_pressure_score": float(pressure),
                "goals_in_last_10min": float(goals_window),
                "home_shots_on_target": float(shots_sot),
                "away_shots_on_target": float(rng.integers(0, 5)),
                "home_xg_running": float(home_xg_run),
                "away_xg_running": float(away_xg_run),
                "score_diff_x_time_remaining": float(score_diff * (90 - clock)),
                "home_elo": float(home_elo),
                "away_elo": float(away_elo),
                "elo_diff": float(elo_diff),
                "home_form_pts": float(home_form),
                "away_form_pts": float(away_form),
                "h2h_home_winrate": float(h2h),
                "is_home_game": 1.0,
                "referee_cards_per_game": float(ref_cards),
                "home_squad_value_EUR": float(home_value),
                "away_squad_value_EUR": float(away_value),
                "squad_value_ratio": float(value_ratio),
                "home_injuries_count": float(home_injuries),
                "away_injuries_count": float(away_injuries),
                "home_press_pct": float(home_press),
                "away_press_pct": float(away_press),
                "home_xg_last5": float(xg5_home),
                "away_xg_last5": float(xg5_away),
                "home_xga_last5": float(xga5_home),
                "away_xga_last5": float(xga5_away),
                "competition_tier": float(comp_tier),
                "match_importance": float(importance),
                "days_since_last_match_home": float(days_home),
                "days_since_last_match_away": float(days_away),
                "goals_last_15min": float(goals_window),
                "cards_last_15min": float(cards_window),
                "score_diff_squared": float(score_diff ** 2),
                "momentum_shift": float(momentum),
                "target": int(outcome),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(
        "Synthetic dataset generated: %d rows, %d matches, saved to %s",
        len(df), n_matches, output_path,
    )
    return df


def build_dataset(output_path: str = "data/train.parquet") -> pd.DataFrame:
    """Build complete training dataset from all sources.

    Falls back to synthetic data if no external sources are available.

    Args:
        output_path: Where to save the parquet file.

    Returns:
        Complete training DataFrame.
    """
    start = time.time()
    all_dfs = []

    # 1. StatsBomb (from GitHub — no API needed)
    statsbomb_df = fetch_statsbomb_github(max_matches=500)
    if len(statsbomb_df) > 0:
        all_dfs.append(statsbomb_df)

    # 2. Understat (all leagues)
    for league in ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]:
        understat_df = fetch_understat(league)
        if len(understat_df) > 0:
            all_dfs.append(understat_df)

    # 3. European Soccer Database
    euro_df = fetch_european_soccer_db()
    if len(euro_df) > 0:
        all_dfs.append(euro_df)

    if not all_dfs:
        logger.warning("No external data sources available, generating synthetic data")
        return generate_synthetic_data(
            n_matches=2000,
            snapshots_per_match=19,
            output_path=output_path,
        )

    # Combine all data
    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure all feature columns exist
    for col in FEATURE_NAMES:
        if col not in combined.columns:
            combined[col] = 0.0

    # Add target if not present
    if "target" not in combined.columns:
        combined["target"] = 1

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    elapsed = time.time() - start
    logger.info(
        "Dataset built: %d rows, %d features, saved to %s (%.1f min)",
        len(combined), len(FEATURE_NAMES), output_path, elapsed / 60,
    )

    return combined


def main() -> None:
    """CLI entry point for dataset building."""
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--output", default="data/train.parquet", help="Output path")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    parser.add_argument("--n-matches", type=int, default=2000, help="Synthetic matches")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.synthetic:
        generate_synthetic_data(
            n_matches=args.n_matches,
            output_path=args.output,
        )
    else:
        build_dataset(args.output)


if __name__ == "__main__":
    main()
