#!/usr/bin/env python3
"""Paper trade bot for GitHub Actions.

Runs as a single match session:
1. Receives match info via env vars (from workflow_dispatch)
2. Loads pre-trained models
3. Discovers Kalshi markets for this specific match
4. Fetches live match data from API-Football (primary) or SofaScore (fallback)
5. Runs model predictions on enriched GameState
6. Places paper trades via Kalshi demo API

Unlike run_paper_trade.py, this does NOT discover matches —
it runs a single match passed in by the watcher workflow.

Data sources (in order of preference):
1. API-Football (api-sports.io) — 100 req/day free tier, requires API key
2. SofaScore (sofascore.com) — public API, no key needed, broader coverage

Fail-safe: Bot never trades without live data. If both sources fail,
predictions are skipped until data becomes available.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

IST = timezone(timedelta(hours=5, minutes=30))

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from market.kalshi_client import KalshiClient, KalshiMarket, ShadowOrderbook
from market.api_football_client import APIFootballClient, LiveMatchState
from market.sofascore_client import LiveScoreClient
from model.predict import WinPredictor
from trading.edge_calculator import EdgeCalculator
from trading.kelly_sizer import KellySizer
from vision.game_state import GameState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/paper_signals")
TRADES_LOG = DATA_DIR / "trades_log.jsonl"
STATE_FILE = DATA_DIR / "current_state.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Persistent state (committed to repo across sessions)
STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
BANKROLL_FILE = STATE_DIR / "bankroll.json"
TRADE_LEDGER = STATE_DIR / "trade_ledger.jsonl"
CALIBRATION_FILE = STATE_DIR / "calibration_report.json"

# How often to update live data from API-Football (seconds)
# Single key budget: 100 calls/day. Each poll = 1 call (events come from the
# fixtures response). 120min / 75s = 96 polls + ~4 discovery calls = 100.
LIVE_UPDATE_INTERVAL = 75
# How often to update Kalshi prices (seconds)
PRICE_UPDATE_INTERVAL = 30
# Trade cooldown per outcome (seconds)
TRADE_COOLDOWN = 120
# Max match duration before auto-stop (minutes)
MAX_MATCH_MINUTES = 120
# Cap on any single probability to prevent extreme predictions
MAX_PROB_CAP = 0.80
# Number of consecutive identical predictions before decay kicks in
DECAY_THRESHOLD = 10
# How much to reduce confidence per extra identical prediction
DECAY_RATE = 0.02
# If clock hasn't advanced in this many minutes after 80', assume FT
STALE_CLOCK_FT_MINUTES = 30


def load_db() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"trades": [], "bankroll": 0.0, "pnl": 0.0}


def save_db(db: dict) -> None:
    STATE_FILE.write_text(json.dumps(db, indent=2))


def log_trade(trade: dict) -> None:
    with open(TRADES_LOG, "a") as f:
        f.write(json.dumps(trade) + "\n")


def _normalize_team_name(name: str) -> str:
    """Normalize team name for fuzzy matching.

    Strips Kalshi suffixes like ': Regulation Time Moneyline' and other noise.
    """
    name = name.strip()
    # Strip everything after ":" or " - " (Kalshi market suffixes)
    for marker in [":", " - "]:
        if marker in name:
            name = name.split(marker, 1)[0].strip()
    return name.lower().replace(".", "").replace("'", "").replace("-", " ")


# Team alias dictionary — maps common Kalshi names to full API names
# Used for fuzzy matching when substring matching fails
TEAM_ALIASES: Dict[str, List[str]] = {
    # English Premier League
    "manchester united": ["man united", "man utd", "manu", "manchester utd"],
    "manchester city": ["man city", "mancity"],
    "tottenham hotspur": ["tottenham", "spurs"],
    "west ham united": ["west ham"],
    "newcastle united": ["newcastle"],
    "brighton & hove albion": ["brighton"],
    "wolverhampton wanderers": ["wolves", "wolverhampton"],
    "nottingham forest": ["nottingham"],
    # La Liga
    "real madrid": ["real madrid cf", "rmadrid", "real madrid castilla"],
    "atlético madrid": ["atletico", "atleti", "atletico madrid"],
    "fc barcelona": ["barca", "barcelona"],
    "real sociedad": ["sociedad"],
    "real betis": ["betis"],
    # Serie A
    "inter milan": ["inter", "internazionale", "inter milan"],
    "ac milan": ["milan", "ac milan"],
    "juventus": ["juve", "jfc", "juventus turin"],
    "as roma": ["roma", "as roma"],
    "ss lazio": ["lazio"],
    # Bundesliga
    "bayern munich": ["bayern", "fc bayern", "bayern munchen"],
    "borussia dortmund": ["bvb", "dortmund", "borussia dortmund"],
    "borussia mönchengladbach": ["gladbach", "monchengladbach"],
    "rb leipzig": ["leipzig"],
    # Ligue 1
    "paris saint-germain": ["psg", "paris sg", "paris saint germain"],
    "olympique lyonnais": ["lyon", "ol"],
    "olympique de marseille": ["marseille", "om"],
    "as monaco": ["monaco"],
    # Eredivisie
    "ajax": ["ajax amsterdam", "afc ajax"],
    "feyenoord": ["feyenoord rotterdam"],
    "psv": ["psv eindhoven"],
    # Portuguese Liga
    "benfica": ["sl benfica"],
    "porto": ["fc porto", "porto"],
    "sporting cp": ["sporting lisbon", "sporting"],
    # Turkish Super Lig
    "galatasaray": ["gs"],
    "fenerbahçe": ["fenerbahce", "fener"],
    "beşiktaş": ["besiktas", "bjk"],
    # Scottish Premiership
    "celtic": ["celtic fc"],
    "rangers": ["rangers fc"],
    # Brazilian Serie A
    "flamengo": ["cr flamengo", "flamengo rio"],
    "palmeiras": ["se palmeiras"],
    "corinthians": ["sc corinthians paulista"],
    "são paulo": ["sao paulo", "spfc"],
    "internacional": ["sc internacional", "internacional rs"],
    "grêmio": ["gremio", "grêmio fbpa"],
    # Argentine Liga
    "boca juniors": ["boca"],
    "river plate": ["river"],
    # MLS
    "inter miami": ["miami", "inter miami cf"],
    "la galaxy": ["galaxy", "la galaxy"],
    # UCL / Europa
    "real madrid cf": ["real madrid"],
    "manchester city": ["man city"],
}

# Reverse alias lookup: alias → canonical name
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in TEAM_ALIASES.items():
    ALIAS_TO_CANONICAL[canonical.lower()] = canonical.lower()
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical.lower()


def _resolve_team_alias(name: str) -> str:
    """Resolve a team name through the alias dictionary.

    Returns the canonical normalized name if found, otherwise the original normalized name.
    """
    norm = _normalize_team_name(name)
    return ALIAS_TO_CANONICAL.get(norm, norm)


def _fuzzy_match_score(name1: str, name2: str) -> float:
    """Compute fuzzy match score between two team names using SequenceMatcher.

    Returns a score between 0.0 (no match) and 1.0 (exact match).
    Also compares the primary (first) tokens, since API names often carry
    suffixes like "SK FK", "FC", etc. ("Vasteras SK FK" vs "Vasteraas").
    """
    from difflib import SequenceMatcher
    n1 = _normalize_team_name(name1)
    n2 = _normalize_team_name(name2)

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    # Direct match
    ratio = _ratio(n1, n2)

    # Also check if any alias matches
    r1 = _ratio(_resolve_team_alias(name1), n2)
    r2 = _ratio(n1, _resolve_team_alias(name2))

    best = max(ratio, r1, r2)

    # Compare primary tokens (first word of each name) — helps with
    # "Vasteraas" vs "Vasteras SK FK" and similar suffix-heavy API names
    tok1 = n1.split()[0] if n1.split() else n1
    tok2 = n2.split()[0] if n2.split() else n2
    token_ratio = _ratio(tok1, tok2)
    for t in n2.split():
        if t in ("fc", "sk", "fk", "cf", "afc", "sc", "if", "bk", "ff", "w", "bvo"):
            continue
        token_ratio = max(token_ratio, _ratio(tok1, t))

    best = max(best, token_ratio * 0.90)
    return best


class GitHubBot:
    """Paper trader for GitHub Actions (single match with live data)."""

    def __init__(self):
        self.config = load_config()
        self.kalshi: Optional[KalshiClient] = None
        self.api_football: Optional[APIFootballClient] = None
        self.livescore: Optional[LiveScoreClient] = None
        self.predictor: Optional[WinPredictor] = None
        self.edge_calc: Optional[EdgeCalculator] = None
        self.kelly: Optional[KellySizer] = None

        # Match info from env vars
        self.match_home = os.environ.get("MATCH_HOME", "")
        self.match_away = os.environ.get("MATCH_AWAY", "")
        self.match_kickoff_str = os.environ.get("MATCH_KICKOFF", "")
        self.event_ticker = os.environ.get("MATCH_EVENT_TICKER", "")

        # Parse kickoff
        self.match_kickoff: Optional[datetime] = None
        if self.match_kickoff_str:
            try:
                self.match_kickoff = datetime.fromisoformat(
                    self.match_kickoff_str.replace("Z", "+00:00")
                )
                if self.match_kickoff.tzinfo is None:
                    self.match_kickoff = self.match_kickoff.replace(tzinfo=IST)
            except Exception:
                pass

        # State
        self._running = True
        self._markets: Dict[str, KalshiMarket] = {}
        self._markets_ready: bool = False
        self._code_to_outcome: Dict[str, str] = {}
        self._bankroll: float = 0.0
        self._start_of_day_bankroll: float = 0.0
        self._trades: List[dict] = []
        self._poll_count = 0
        self._order_cooldown: Dict[str, float] = {}
        self._last_live_update: float = 0

        # Shadow orderbook (simulates fills against production book)
        self._shadow_books: Dict[str, ShadowOrderbook] = {}

        # Settlement rules tracking (Fix 6)
        self._regulation_home_goals: int = 0
        self._regulation_away_goals: int = 0
        self._regulation_snapshot_taken: bool = False

        # Calibration data (Fix 7)
        self._calibration_data: List[dict] = []

        # API-Football fixture tracking
        self._api_football_fixture_id: Optional[int] = None
        self._livescore_slug: Optional[str] = None
        self._prev_live_state: Optional[LiveMatchState] = None
        self._live_data_received: bool = False

        # Prediction stability tracking
        self._last_predicted_outcome: Optional[str] = None
        self._consecutive_same_outcome: int = 0
        self._clock_stuck_since: Optional[float] = None
        self._last_clock_value: float = 0.0

        # Game state — enriched by KickoffAPI live data
        self._game_state = GameState(
            home_team=self.match_home,
            away_team=self.match_away,
        )

    def initialize(self) -> bool:
        """Initialize basics (auth, models, clients). Does NOT discover markets.

        Market discovery is separate so the main loop can poll for markets
        during pre-kickoff wait (Kalshi opens markets closer to kickoff).
        """
        logger.info("=" * 60)
        logger.info("GITHUB ACTIONS PAPER BOT")
        logger.info("Match: %s vs %s", self.match_home, self.match_away)
        logger.info("Event: %s", self.event_ticker)
        if self.match_kickoff:
            logger.info("Kickoff: %s IST", self.match_kickoff.astimezone(IST).strftime("%Y-%m-%d %H:%M"))
        else:
            logger.info("Kickoff: %s", self.match_kickoff_str)
        logger.info("=" * 60)

        # Kalshi client
        self.kalshi = KalshiClient(
            api_key=self.config.kalshi_api_key,
            private_key_pem=self.config.kalshi_private_key,
            dry_run=self.config.dry_run,
            use_demo=self.config.kalshi_use_demo,
        )
        balance = self.kalshi.get_balance()
        if balance is None:
            logger.error("Failed to authenticate with Kalshi demo")
            return False
        self._bankroll = balance
        logger.info("Kalshi demo balance: $%.2f", balance)

        # Load previous bankroll from persistent state (Fix 2 + Fix 8)
        if BANKROLL_FILE.exists():
            try:
                prev = json.loads(BANKROLL_FILE.read_text())
                self._start_of_day_bankroll = prev.get("bankroll", balance)
                logger.info("Loaded previous bankroll: $%.2f (start-of-day: $%.2f)",
                            prev.get("bankroll", 0), self._start_of_day_bankroll)
            except Exception:
                self._start_of_day_bankroll = balance
        else:
            self._start_of_day_bankroll = balance

        # API-Football client (for live match data) — supports dual-key rotation
        api_key = os.environ.get("API_FOOTBALL_API_KEY", "")
        api_key_2 = os.environ.get("API_FOOTBALL_API_KEY_2", "")
        if api_key:
            self.api_football = APIFootballClient(api_key=api_key, api_key_2=api_key_2)
            logger.info("API-Football client initialized (%d remaining, dual-key=%s)",
                        self.api_football.remaining, "yes" if api_key_2 else "no")
        else:
            logger.warning("No API_FOOTBALL_API_KEY — running without live data")

        # SofaScore client (fallback for live match data)
        self.livescore = LiveScoreClient()
        logger.info("LiveScore client initialized (SportScore fallback)")

        # ML models
        try:
            self.predictor = WinPredictor()
            self.predictor.initialize()
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.warning("ML models not available: %s — running market-only", e)
            self.predictor = None

        # Edge calculator + Kelly sizer
        self.edge_calc = EdgeCalculator(
            edge_threshold=self.config.edge_threshold,
            confidence_threshold=self.config.confidence_threshold,
        )
        self.kelly = KellySizer(
            base_kelly=self.config.kelly_fraction,
            max_bet_pct=self.config.max_bet_pct,
            min_bet_usd=self.config.min_bet_usd,
        )

        # Find API-Football fixture for live data
        if self.api_football:
            self._find_api_football_fixture()

        return True

    def _discover_markets(self) -> bool:
        """Discover Kalshi markets for this event. Returns True if markets found.

        Called from initialize() and from the main loop during pre-kickoff wait.
        Markets may not be available immediately — Kalshi opens them closer to kickoff.
        """
        if self._markets_ready:
            return True

        try:
            markets = self.kalshi.get_event_markets(self.event_ticker)
        except Exception as e:
            logger.warning("Market discovery error: %s", e)
            return False

        if not markets:
            return False

        # Build market dict
        for m in markets:
            self._markets[m.ticker] = m
            logger.info(
                "  Market: %s (yes=$%.2f no=$%.2f vol=%d) title=%r sub=%r reg_only=%s",
                m.ticker, m.yes_ask, m.no_ask, m.volume, m.title, m.subtitle,
                m.is_regulation_only,
            )
            # Fetch orderbook depth for fill simulation
            depth = self.kalshi.get_orderbook_depth(m.ticker, depth=self.config.orderbook_depth)
            if depth:
                self._shadow_books[m.ticker] = depth
                logger.info("  Shadow book: %s (yes_depth=%d no_depth=%d)",
                            m.ticker, len(depth.yes_levels), len(depth.no_levels))

        # Build team code → outcome mapping from event ticker
        event_upper = self.event_ticker.upper()
        seen_codes: set = set()
        for t in self._markets:
            suffix = t.split("-")[-1].upper()
            if suffix in ("TIE", "HOME", "AWAY", "YES", "NO", "DRAW"):
                continue
            if suffix not in seen_codes:
                seen_codes.add(suffix)

        code_positions = []
        for code in seen_codes:
            pos = event_upper.find(code)
            if pos >= 0:
                code_positions.append((code, pos))
        code_positions.sort(key=lambda x: x[1])

        if len(code_positions) >= 2:
            self._code_to_outcome[code_positions[0][0]] = "home"
            self._code_to_outcome[code_positions[1][0]] = "away"
            logger.info("  Team codes: %s=home, %s=away", code_positions[0][0], code_positions[1][0])

        # Log outcome mapping
        for t, m in self._markets.items():
            outcome = self._map_outcome(m)
            logger.info("  Mapping %s -> %s", t, outcome or "UNKNOWN")

        self._markets_ready = True
        logger.info("Markets ready: %d markets found for %s", len(self._markets), self.event_ticker)
        return True

    def _find_api_football_fixture(self) -> None:
        """Find API-Football fixture ID by matching team names.

        Uses fuzzy matching with alias dictionary for robust name resolution.
        Falls back to SofaScore if API-Football can't find the match.
        """
        if not self.api_football:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            fixtures = self.api_football.get_fixtures_by_date(today)
            if not fixtures:
                tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
                fixtures = self.api_football.get_fixtures_by_date(tomorrow)

            home_norm = _normalize_team_name(self.match_home)
            away_norm = _normalize_team_name(self.match_away)
            home_resolved = _resolve_team_alias(self.match_home)
            away_resolved = _resolve_team_alias(self.match_away)

            best_match = None
            best_score = 0.0

            for f in fixtures:
                teams = f.get("teams", {})
                f_home = teams.get("home", {}).get("name", "")
                f_away = teams.get("away", {}).get("name", "")

                # Strategy 1: Exact substring match (fast)
                score1 = 0
                if home_norm in _normalize_team_name(f_home) or _normalize_team_name(f_home) in home_norm:
                    score1 += 1
                if away_norm in _normalize_team_name(f_away) or _normalize_team_name(f_away) in away_norm:
                    score1 += 1

                score2 = 0
                if home_norm in _normalize_team_name(f_away) or _normalize_team_name(f_away) in home_norm:
                    score2 += 1
                if away_norm in _normalize_team_name(f_home) or _normalize_team_name(f_home) in away_norm:
                    score2 += 1

                substring_score = max(score1, score2)
                if substring_score >= 2:
                    best_match = f
                    best_score = 2.0
                    break

                # Strategy 2: Alias resolution match
                alias_score = 0
                if home_resolved in _normalize_team_name(f_home) or _normalize_team_name(f_home) in home_resolved:
                    alias_score += 1
                if away_resolved in _normalize_team_name(f_away) or _normalize_team_name(f_away) in away_resolved:
                    alias_score += 1

                alias_score2 = 0
                if home_resolved in _normalize_team_name(f_away) or _normalize_team_name(f_away) in home_resolved:
                    alias_score2 += 1
                if away_resolved in _normalize_team_name(f_home) or _normalize_team_name(f_home) in away_resolved:
                    alias_score2 += 1

                alias_total = max(alias_score, alias_score2)
                if alias_total >= 2:
                    best_match = f
                    best_score = 2.0
                    break

                # Strategy 3: Fuzzy matching (slower, for edge cases)
                fuzzy_home = _fuzzy_match_score(self.match_home, f_home)
                fuzzy_away = _fuzzy_match_score(self.match_away, f_away)
                fuzzy_score = min(fuzzy_home, fuzzy_away)

                # Also check reversed ordering
                fuzzy_home_rev = _fuzzy_match_score(self.match_home, f_away)
                fuzzy_away_rev = _fuzzy_match_score(self.match_away, f_home)
                fuzzy_score_rev = min(fuzzy_home_rev, fuzzy_away_rev)

                fuzzy_best = max(fuzzy_score, fuzzy_score_rev)
                if fuzzy_best > best_score and fuzzy_best >= 0.65:
                    best_score = fuzzy_best
                    best_match = f

            # Accept either an exact match (2.0) or a confident fuzzy match (>= 0.80)
            if best_match and best_score >= 0.80:
                fixture_data = best_match.get("fixture", {})
                self._api_football_fixture_id = fixture_data.get("id")
                teams = best_match.get("teams", {})
                f_home = teams.get("home", {}).get("name", "?")
                f_away = teams.get("away", {}).get("name", "?")
                logger.info("API-Football fixture found: %s vs %s (ID=%s, score=%.2f)",
                            f_home, f_away, self._api_football_fixture_id, best_score)
            else:
                logger.warning("API-Football: no matching fixture for %s vs %s (checked %d fixtures)",
                               self.match_home, self.match_away, len(fixtures))
        except Exception as e:
            logger.warning("API-Football fixture discovery failed: %s", e)

        # LiveScore fallback — find match slug for live data
        # Always try to discover SportScore as fallback, even when API-Football works
        if self.livescore and not self._livescore_slug:
            self._find_livescore_match()

    def _find_livescore_match(self) -> None:
        """Find SportScore match slug for this match (fallback when API-Football fails)."""
        if not self.livescore:
            return

        try:
            slug = self.livescore.find_match(self.match_home, self.match_away)
            if slug:
                self._livescore_slug = slug
                logger.info("LiveScore match found: %s vs %s (slug=%s)",
                            self.match_home, self.match_away, slug)
            else:
                logger.warning("LiveScore: no matching event for %s vs %s", self.match_home, self.match_away)
        except Exception as e:
            logger.warning("LiveScore event discovery failed: %s", e)

    def _fetch_live_state(self) -> Optional[str]:
        """Fetch live match data from API-Football or SofaScore fallback.

        Returns match status string ("NS", "1H", "2H", "FT", etc.) or None on error.
        Tries API-Football first, falls back to SofaScore if unavailable.
        Detects clock-stuck and switches to SportScore if API-Football freezes.
        """
        now = time.time()

        # Throttle: don't fetch more often than LIVE_UPDATE_INTERVAL
        if now - self._last_live_update < LIVE_UPDATE_INTERVAL:
            return self._prev_live_state.status if self._prev_live_state else None

        # Check if API-Football clock is stuck
        clock_stuck = self._detect_clock_stuck()

        # Source 1: API-Football (preferred) — skip if clock is stuck
        if self.api_football and self._api_football_fixture_id and not clock_stuck:
            remaining = self.api_football.remaining
            if remaining <= 10:
                logger.warning("API-Football rate limit critical (%d remaining) — trying SofaScore", remaining)
            elif remaining <= 30:
                # When low, only fetch every other cycle
                if now - self._last_live_update < LIVE_UPDATE_INTERVAL * 2:
                    return self._prev_live_state.status if self._prev_live_state else None
                state = self._fetch_from_api_football(now)
                if state:
                    return state
            else:
                state = self._fetch_from_api_football(now)
                if state:
                    return state

        # Source 2: LiveScore/SportScore (fallback, or primary when clock stuck)
        if self.livescore and self._livescore_slug:
            state = self._fetch_from_livescore(now)
            if state:
                return state

        # If both sources failed but we have a previous state, check for stale clock FT
        if clock_stuck and self._prev_live_state:
            current_clock = self._prev_live_state.clock_minutes
            if current_clock >= 80 and self._clock_stuck_since:
                stuck_min = (now - self._clock_stuck_since) / 60
                if stuck_min >= STALE_CLOCK_FT_MINUTES:
                    logger.info("Clock stuck at %d' for %.0f min. Treating as FT.", current_clock, stuck_min)
                    return "FT"

        return self._prev_live_state.status if self._prev_live_state else None

    def _fetch_from_api_football(self, now: float) -> Optional[str]:
        """Fetch live state from API-Football. Returns status or None."""
        try:
            state = self.api_football.get_live_match(self._api_football_fixture_id)
            if not state:
                return None

            logger.info(
                "LIVE [API-Football]: %s %d - %d %s | %s %.0f' | events=%d | remaining=%d",
                state.home_team, state.home_score, state.away_score, state.away_team,
                state.status, state.clock_minutes,
                len(state.events), self.api_football.remaining,
            )

            self._game_state = self._match_state_to_game_state(state)
            self._prev_live_state = state
            self._last_live_update = now
            if not self._live_data_received:
                self._live_data_received = True
                logger.info("First live data received — predictions enabled")
            return state.status

        except Exception as e:
            logger.warning("API-Football live update failed: %s", e)
            self._last_live_update = now
            return None

    def _fetch_from_livescore(self, now: float) -> Optional[str]:
        """Fetch live state from SportScore (fallback). Returns status or None."""
        try:
            state = self.livescore.get_live_match(self._livescore_slug)
            if not state:
                return None

            logger.info(
                "LIVE [SportScore]: %s %d - %d %s | %s %.0f' | events=%d",
                state.home_team, state.home_score, state.away_score, state.away_team,
                state.status, state.clock_minutes,
                len(state.events),
            )

            self._game_state = self._match_state_to_game_state(state)
            self._prev_live_state = state
            self._last_live_update = now
            if not self._live_data_received:
                self._live_data_received = True
                logger.info("First live data received (SportScore) — predictions enabled")
            return state.status

        except Exception as e:
            logger.warning("SportScore live update failed: %s", e)
            self._last_live_update = now
            return None

    def _match_state_to_game_state(self, ms: LiveMatchState) -> GameState:
        """Convert KickoffAPI LiveMatchState to GameState for model prediction.

        Also tracks regulation-time score for settlement (Fix 6):
        Match winner markets settle at 90 min + stoppage time ONLY.
        We snapshot the score at 90' so we can resolve against regulation result.
        """
        # Track regulation-time score for settlement (Fix 6)
        # Snapshot at 90' — once clock passes 90', lock the regulation score
        if not self._regulation_snapshot_taken and ms.clock_minutes >= 90:
            self._regulation_home_goals = ms.home_score
            self._regulation_away_goals = ms.away_score
            self._regulation_snapshot_taken = True
            logger.info(
                "REGULATION SNAPSHOT at %.0f': %d-%d (settles at this score)",
                ms.clock_minutes, ms.home_score, ms.away_score,
            )

        goals_in_10 = self._count_goals_in_window(ms, 10)
        goals_in_15 = self._count_goals_in_window(ms, 15)
        cards_in_15 = self._count_cards_in_window(ms, 15)
        momentum = self._compute_momentum(ms)

        return GameState(
            match_id=str(ms.fixture_id),
            home_team=ms.home_team or self.match_home,
            away_team=ms.away_team or self.match_away,
            clock_minutes=ms.clock_minutes,
            stoppage_time=0,
            is_extra_time=ms.period >= 3,
            home_score=ms.home_score,
            away_score=ms.away_score,
            ocr_reliable=True,
            consecutive_consistent_reads=10,
            timestamp=ms.last_update,
            home_red_cards=ms.home_red_cards,
            away_red_cards=ms.away_red_cards,
            home_pressure_score=ms.home_pressure,
            goals_in_last_10min=goals_in_10,
            goals_last_15min=goals_in_15,
            cards_last_15min=cards_in_15,
            home_shots_on_target=ms.home_stats.shots_on if ms.home_stats else 0,
            away_shots_on_target=ms.away_stats.shots_on if ms.away_stats else 0,
            home_xg_running=ms.home_xg_running,
            away_xg_running=ms.away_xg_running,
            momentum_shift=momentum,
            home_elo=1600.0,
            away_elo=1600.0,
            home_form_pts=7,
            away_form_pts=7,
            h2h_home_winrate=0.45,
            is_home_game=True,
            referee_cards_per_game=3.5,
            home_squad_value_EUR=50_000_000,
            away_squad_value_EUR=50_000_000,
            home_injuries_count=0,
            away_injuries_count=0,
            home_press_pct=ms.home_pressure,
            away_press_pct=1.0 - ms.home_pressure,
            home_xg_last5=ms.home_xg_running,
            away_xg_last5=ms.away_xg_running,
            home_xga_last5=ms.away_xg_running,
            away_xga_last5=ms.home_xg_running,
            competition_tier=2,
            match_importance=0.5,
            days_since_last_match_home=7,
            days_since_last_match_away=7,
        )

    def _count_goals_in_window(self, ms: LiveMatchState, window_minutes: int) -> int:
        count = 0
        for event in ms.events:
            if event.event_type == "Goal" and event.minute >= (ms.clock_minutes - window_minutes):
                count += 1
        return count

    def _count_cards_in_window(self, ms: LiveMatchState, window_minutes: int) -> int:
        count = 0
        for event in ms.events:
            if event.event_type == "Card" and event.minute >= (ms.clock_minutes - window_minutes):
                count += 1
        return count

    def _compute_momentum(self, ms: LiveMatchState) -> float:
        if not self._prev_live_state:
            return 0.0
        prev_xg = self._prev_live_state.home_xg_running + self._prev_live_state.away_xg_running
        curr_xg = ms.home_xg_running + ms.away_xg_running
        delta_time = max(ms.clock_minutes - self._prev_live_state.clock_minutes, 1)
        return (curr_xg - prev_xg) / delta_time

    def _map_outcome(self, market: KalshiMarket) -> Optional[str]:
        """Map a Kalshi market to home/draw/away outcome."""
        t = market.ticker.upper()

        if t.endswith("-HOME") or t.endswith("-YES"):
            return "home"
        if t.endswith("-DRAW"):
            return "draw"
        if t.endswith("-AWAY") or t.endswith("-NO"):
            return "away"

        suffix = market.ticker.split("-")[-1].upper()
        if suffix == "TIE":
            return "draw"
        if suffix in self._code_to_outcome:
            return self._code_to_outcome[suffix]

        title = market.title
        suffix = title.rsplit(" - ", 1)[-1].strip().lower() if " - " in title else title.lower()
        home = self.match_home.lower()
        away = self.match_away.lower()

        if home and suffix == home:
            return "home"
        if away and suffix == away:
            return "away"
        if suffix in ("tie", "draw"):
            return "draw"

        sub = market.subtitle.lower()
        if home and home in sub:
            return "home"
        if away and away in sub:
            return "away"
        if "tie" in sub or "draw" in sub:
            return "draw"
        if "home" in sub or "1" in sub:
            return "home"
        if "away" in sub or "2" in sub:
            return "away"

        return None

    def _find_ticker_for_outcome(self, outcome: str) -> Optional[str]:
        for t, m in self._markets.items():
            if self._map_outcome(m) == outcome:
                return t
        return None

    def run(self):
        if not self.initialize():
            logger.warning("Bot initialization failed — could not authenticate. Exiting.")
            return

        # Try market discovery immediately (fast path: markets already open)
        self._discover_markets()

        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Starting trading loop...")
        last_status_time = 0
        last_price_time = 0
        last_market_poll_time = 0

        while self._running:
            try:
                now_ts = time.time()
                now_ist = datetime.now(IST)

                # Check if match is over
                if self.match_kickoff:
                    elapsed = (now_ist - self.match_kickoff).total_seconds() / 60
                    if elapsed > MAX_MATCH_MINUTES:
                        logger.info("Match likely over (%.0f min elapsed). Stopping.", elapsed)
                        break

                    # Pre-match: poll for markets until they appear
                    if elapsed < -5:
                        if not self._markets_ready:
                            # Poll every 60s for markets (Kalshi opens them closer to kickoff)
                            if now_ts - last_market_poll_time >= 60:
                                last_market_poll_time = now_ts
                                if self._discover_markets():
                                    logger.info("Markets found! Ready to trade.")
                                else:
                                    remaining = abs(elapsed)
                                    logger.info(
                                        "Pre-match: kickoff in %.0f min. Markets not open yet. Polling in 60s...",
                                        remaining,
                                    )
                            time.sleep(1)
                            continue
                        else:
                            # Markets ready but still pre-kickoff — sleep until 5 min before
                            logger.info(
                                "Pre-match: kickoff in %.0f min. Markets ready. Sleeping 60s...",
                                abs(elapsed),
                            )
                            time.sleep(60)
                            continue

                    # Only update clock if no live data (API-Football overrides this)
                    if elapsed > 0 and not self._api_football_fixture_id:
                        self._game_state.clock_minutes = min(elapsed, 90)

                # Fetch live match data (API-Football primary, SofaScore fallback)
                match_status = self._fetch_live_state()

                # Stop immediately if match is over (FT, AET, PEN, etc.)
                if match_status and match_status in ("FT", "AET", "PEN", "AWD", "CANC", "POST"):
                    logger.info("Match finished (status=%s). Stopping.", match_status)
                    # Resolve PnL for all open trades (Fix 2 + Fix 6)
                    self._resolve_session_pnl()
                    break

                # Update prices (throttled) — only if markets are ready
                if self._markets_ready and now_ts - last_price_time >= PRICE_UPDATE_INTERVAL:
                    self._update_prices()
                    last_price_time = now_ts

                    # Check edges after fresh prices
                    self._check_edges()

                # Status every ~5 min
                if now_ts - last_status_time >= 300:
                    self._print_status()
                    last_status_time = now_ts

                time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Loop error: %s", e, exc_info=True)
                time.sleep(5)

        self._shutdown()

    def _update_prices(self):
        for ticker, market in list(self._markets.items()):
            try:
                resp = self.kalshi._request("GET", f"/markets/{ticker}")
                if resp and "market" in resp:
                    m = resp["market"]
                    if "yes_ask_dollars" in m:
                        market.yes_bid = float(m.get("yes_bid_dollars", m.get("yes_bid", 0)))
                        market.yes_ask = float(m.get("yes_ask_dollars", 1.0))
                        market.no_bid = float(m.get("no_bid_dollars", 1.0 - market.yes_ask))
                        market.no_ask = float(m.get("no_ask_dollars", 1.0 - market.yes_bid))
                    elif "yes_bid" in m:
                        market.yes_bid = float(m.get("yes_bid", 0)) / 100
                        market.yes_ask = float(m.get("yes_ask", 100)) / 100
                        market.no_bid = 1.0 - market.yes_ask
                        market.no_ask = 1.0 - market.yes_bid
                    market.volume = m.get("volume", market.volume)
            except Exception as e:
                logger.debug("Price update failed for %s: %s", ticker, e)

    def _clamp_probabilities(self, probs: tuple) -> tuple:
        """Clamp extreme probabilities and apply confidence decay.

        The model sometimes outputs 99.9% for a single outcome when the
        match is 0-0. This is clearly wrong. We:
        1. Cap any single probability at MAX_PROB_CAP (0.80)
        2. If the same outcome keeps winning, decay its confidence
        3. Redistribute excess probability to other outcomes
        """
        home, draw, away = probs
        raw_home, raw_draw, raw_away = home, draw, away

        # Step 1: Cap at MAX_PROB_CAP
        home = min(home, MAX_PROB_CAP)
        draw = min(draw, MAX_PROB_CAP)
        away = min(away, MAX_PROB_CAP)

        # Step 2: Normalize so probabilities sum to 1
        total = home + draw + away
        if total > 0:
            home /= total
            draw /= total
            away /= total

        # Step 3: Track prediction stability
        best_outcome = max(("home", home), ("draw", draw), ("away", away), key=lambda x: x[1])
        if best_outcome[0] == self._last_predicted_outcome:
            self._consecutive_same_outcome += 1
        else:
            self._last_predicted_outcome = best_outcome[0]
            self._consecutive_same_outcome = 1

        # Step 4: Apply decay if same outcome persists too long
        if self._consecutive_same_outcome > DECAY_THRESHOLD:
            decay = min(0.30, (self._consecutive_same_outcome - DECAY_THRESHOLD) * DECAY_RATE)
            if best_outcome[0] == "home":
                home = max(0.30, home - decay)
            elif best_outcome[0] == "draw":
                draw = max(0.30, draw - decay)
            else:
                away = max(0.30, away - decay)

            # Re-normalize
            total = home + draw + away
            if total > 0:
                home /= total
                draw /= total
                away /= total

            if self._consecutive_same_outcome % 10 == 0:
                logger.info(
                    "DECAY: same outcome=%s for %d preds, applying %.1f%% decay",
                    best_outcome[0], self._consecutive_same_outcome, decay * 100,
                )

        if raw_home != home or raw_draw != draw or raw_away != away:
            logger.info(
                "CLAMP: raw=(%.1f,%.1f,%.1f) -> clamped=(%.1f,%.1f,%.1f)",
                raw_home * 100, raw_draw * 100, raw_away * 100,
                home * 100, draw * 100, away * 100,
            )

        return (home, draw, away)

    def _detect_clock_stuck(self) -> bool:
        """Detect if API-Football clock is stuck. Falls back to SportScore.

        Returns True if clock hasn't advanced in 3+ minutes.
        """
        if not self._prev_live_state:
            return False

        current_clock = self._prev_live_state.clock_minutes
        now = time.time()

        # Check if clock advanced
        if current_clock != self._last_clock_value:
            self._last_clock_value = current_clock
            self._clock_stuck_since = None
            return False

        # Clock hasn't changed — track when it got stuck
        if self._clock_stuck_since is None:
            self._clock_stuck_since = now

        stuck_duration = now - self._clock_stuck_since

        # If stuck for 3+ minutes, try SportScore
        if stuck_duration > 180:
            if self._livescore_slug and stuck_duration < 600:
                logger.warning(
                    "API-Football clock stuck for %.0fs at %d'. Trying SportScore fallback...",
                    stuck_duration, current_clock,
                )
                return True
            elif stuck_duration > STALE_CLOCK_FT_MINUTES * 60 and current_clock >= 80:
                logger.warning(
                    "Clock stuck at %d' for >%d min after 80'. Assuming FT.",
                    current_clock, STALE_CLOCK_FT_MINUTES,
                )
                return True

        return False

    def _adjust_for_regulation(self, score_home: int, score_away: int) -> str:
        """Determine the regulation-time result for settlement (Fix 6).

        Match winner markets on Kalshi settle at 90 minutes + stoppage time.
        If the match goes to extra time or penalties, the regulation result
        (90' snapshot) determines settlement.

        Returns:
            "home", "draw", or "away" based on regulation-time score.
        """
        if score_home > score_away:
            return "home"
        elif score_away > score_home:
            return "away"
        else:
            return "draw"

    def _check_loss_floor(self) -> bool:
        """Check if bankroll has hit the relative loss floor (Fix 8).

        Halts trading if current bankroll is X% below start-of-day bankroll.
        Returns True if loss floor is breached (should halt trading).
        """
        if self._start_of_day_bankroll <= 0:
            return False
        loss_pct = (self._start_of_day_bankroll - self._bankroll) / self._start_of_day_bankroll
        if loss_pct >= self.config.loss_floor_pct:
            logger.warning(
                "LOSS FLOOR BREACHED: bankroll $%.2f is %.1f%% below start-of-day $%.2f (floor=%.1f%%)",
                self._bankroll, loss_pct * 100, self._start_of_day_bankroll,
                self.config.loss_floor_pct * 100,
            )
            return True
        return False

    def _resolve_pnl(self, trade: dict, result: str) -> float:
        """Resolve PnL for a single trade at settlement (Fix 2).

        Args:
            trade: Trade dict with outcome, count, price, fee, regulation_only.
            result: "win", "loss", or "push".

        Returns:
            Net PnL in dollars.
        """
        count = trade["count"]
        price = trade["price"]
        fee = trade.get("fee", 0.0)

        if result == "win":
            # Win: receive $1 per contract, minus what you paid
            pnl = count * (1.0 - price) - fee
        elif result == "loss":
            # Loss: lose your stake
            pnl = -(count * price + fee)
        else:
            # Push: get stake back minus fee
            pnl = -fee

        return pnl

    def _check_edges(self):
        if not self.edge_calc or not self.kelly:
            return

        # Loss floor check (Fix 8)
        if self._check_loss_floor():
            return

        # Don't predict until we have real live data — blank GameState produces garbage predictions
        if not self._live_data_received:
            return

        # Don't predict if markets aren't loaded yet
        if not self._markets_ready:
            return

        # Don't predict before match is live
        if self._prev_live_state and not self._prev_live_state.is_live:
            logger.debug("Match not live yet (status=%s) — skipping prediction", self._prev_live_state.status)
            return

        # RISK GUARD: never trade on stale live data.
        # If the data source (API-Football) is dead/stuck and the fallback
        # (SportScore) has no match, the clock won't advance — trading on a
        # frozen GameState produces garbage predictions. Halt instead.
        if self._clock_stuck_since and self._prev_live_state and self._prev_live_state.is_live:
            stuck_min = (time.time() - self._clock_stuck_since) / 60
            api_dead = self.api_football and self.api_football.remaining <= 15
            no_fallback = not self._livescore_slug
            if stuck_min >= 5 and (api_dead or no_fallback):
                logger.warning(
                    "RISK GUARD: clock stuck at %.0f' for %.0f min with no working fallback "
                    "(API remaining=%d, LS slug=%s) — halting new trades",
                    self._prev_live_state.clock_minutes, stuck_min,
                    self.api_football.remaining if self.api_football else -1,
                    self._livescore_slug or "none",
                )
                return

        market_prices = {}
        market_asks = {}
        market_bids = {}

        for ticker, market in self._markets.items():
            outcome = self._map_outcome(market)
            if not outcome:
                continue
            if market.yes_ask > 0:
                market_prices[outcome] = (market.yes_bid + market.yes_ask) / 2
                market_asks[outcome] = market.yes_ask
                market_bids[outcome] = market.yes_bid

        if not market_prices:
            return

        if self.predictor:
            try:
                probs = self.predictor.predict(self._game_state)
                # Log raw model outputs for diagnosis
                logger.info(
                    "RAW MODEL: home=%.1f%% draw=%.1f%% away=%.1f%% | clock=%.0f'",
                    probs[0] * 100, probs[1] * 100, probs[2] * 100,
                    self._game_state.clock_minutes,
                )
                # Clamp extreme probabilities and apply decay
                probs = self._clamp_probabilities(probs)
                model_probs = {"home": probs[0], "draw": probs[1], "away": probs[2]}
                confidence = max(probs)
                logger.info(
                    "PREDICTION: home=%.1f%% draw=%.1f%% away=%.1f%% (conf=%.1f%%) | clock=%.0f'",
                    probs[0] * 100, probs[1] * 100, probs[2] * 100,
                    confidence * 100, self._game_state.clock_minutes,
                )
            except Exception as e:
                logger.debug("Prediction failed: %s", e)
                return
        else:
            return

        analysis = self.edge_calc.calculate(
            model_probs=model_probs,
            market_prices=market_prices,
            market_asks=market_asks,
            market_bids=market_bids,
            fee_per_contract=self._fee_per_contract(market_asks),
        )

        if not analysis.any_tradable:
            return

        best = analysis.best_edge
        if best:
            self._place_paper_trade(best, model_probs)

    def _fee_per_contract(self, market_asks: dict) -> float:
        """Compute average fee per contract across outcomes with asks."""
        if not market_asks:
            return 0.0
        fees = []
        for outcome, ask in market_asks.items():
            if ask > 0:
                fees.append(KalshiClient._calc_fee(ask, 1))
        return sum(fees) / len(fees) if fees else 0.0

    def _place_paper_trade(self, edge_result, model_probs: dict):
        """Place a simulated trade via shadow orderbook.

        Two-phase edge evaluation (Fix 4):
        1. Initial edge from model vs market ask (already computed)
        2. Simulate fill against shadow book → get fill_price
        3. Re-evaluate edge at fill_price; skip if edge evaporated
        """
        outcome = edge_result.outcome
        now = time.time()

        if outcome in self._order_cooldown:
            if now - self._order_cooldown[outcome] < TRADE_COOLDOWN:
                return

        ticker = self._find_ticker_for_outcome(outcome)
        if not ticker:
            return

        market = self._markets[ticker]
        ask_price = edge_result.market_ask

        # Kelly sizing
        kelly_result = self.kelly.calculate(
            outcome=outcome,
            edge=edge_result.net_edge,
            model_prob=edge_result.model_prob,
            market_prob=edge_result.market_prob,
            bankroll=self._bankroll,
        )

        if not kelly_result or kelly_result.bet_usd < self.config.min_bet_usd:
            return

        bet_usd = kelly_result.bet_usd
        count = max(1, int(bet_usd / ask_price)) if ask_price > 0 else 0
        if count <= 0:
            return

        # ── Phase 2: simulate fill against shadow book ──
        shadow = self._shadow_books.get(ticker)
        if shadow and not shadow.needs_resync():
            fill_count = shadow.simulate_fill(count, ask_price, side="yes")
            if fill_count <= 0:
                logger.debug("Shadow book: no liquidity for %s at $%.2f — skipping", ticker, ask_price)
                return
            fill_price = ask_price
            count = fill_count
        else:
            fill_price = ask_price

        # Re-evaluate edge at fill price (may have changed if slippage)
        fee = KalshiClient._calc_fee(fill_price, count)
        fee_per = fee / count if count > 0 else 0.0
        net_edge_at_fill = edge_result.model_prob - fill_price - fee_per
        if net_edge_at_fill < self.config.edge_threshold:
            logger.debug(
                "Edge evaporated at fill: model=%.3f fill=%.3f fee=%.4f net=%.3f < threshold=%.3f",
                edge_result.model_prob, fill_price, fee_per, net_edge_at_fill, self.config.edge_threshold,
            )
            return

        # ── BUY-YES only: side="bid" means buying YES on this ticker.
        # Each outcome has its own ticker (HOME/DRAW/AWAY). Buying YES on
        # the "away" ticker IS the bet on the away team winning. ──
        logger.info(
            "PAPER TRADE: BUY %s %s x%d @ $%.2f (model=%.3f market=%.3f net_edge=+%.3f fee=$%.4f)",
            outcome.upper(), ticker, count, fill_price,
            edge_result.model_prob, edge_result.market_prob, net_edge_at_fill, fee,
        )

        result = self.kalshi.place_order(
            ticker=ticker,
            side="bid",
            count=count,
            yes_price=fill_price,
        )

        trade = {
            "time": datetime.now(IST).isoformat(),
            "match": f"{self.match_home} vs {self.match_away}",
            "event_ticker": self.event_ticker,
            "ticker": ticker,
            "outcome": outcome,
            "side": "yes",
            "count": count,
            "price": fill_price,
            "bet_usd": bet_usd,
            "model_prob": edge_result.model_prob,
            "market_prob": edge_result.market_prob,
            "gross_edge": edge_result.edge,
            "net_edge": net_edge_at_fill,
            "fee": fee,
            "result": "submitted" if result else "failed",
            "clock": self._game_state.clock_minutes,
            "score": f"{self._game_state.home_score}-{self._game_state.away_score}",
            "regulation_only": market.is_regulation_only,
        }
        self._trades.append(trade)
        log_trade(trade)
        self._order_cooldown[outcome] = now

        # Log calibration data for post-match analysis
        self._calibration_data.append({
            "time": trade["time"],
            "ticker": ticker,
            "outcome": outcome,
            "fill_price": fill_price,
            "model_prob": edge_result.model_prob,
            "market_ask": edge_result.market_ask,
            "net_edge": net_edge_at_fill,
            "count": count,
            "fee": fee,
            "clock": self._game_state.clock_minutes,
            "score": f"{self._game_state.home_score}-{self._game_state.away_score}",
        })

    def _print_status(self):
        elapsed = 0
        if self.match_kickoff:
            elapsed = (datetime.now(IST) - self.match_kickoff).total_seconds() / 60

        logger.info("--- STATUS (T+%.0f min) ---", elapsed)
        logger.info("  Score: %d-%d | Clock: %.0f' | Trades: %d | Bankroll: $%.2f",
                     self._game_state.home_score, self._game_state.away_score,
                     self._game_state.clock_minutes, len(self._trades), self._bankroll)
        source = "API-Football" if self._api_football_fixture_id else (
            "SportScore" if self._livescore_slug else "none")
        logger.info("  Markets: %s | Live data: %s | Source: %s (API=%s, LS=%s)",
                     "ready" if self._markets_ready else "pending",
                     "yes" if self._live_data_received else "no",
                     source,
                     self._api_football_fixture_id or "none",
                     self._livescore_slug or "none")
        for t, m in self._markets.items():
            logger.info("    %s: yes=$%.2f no=$%.2f vol=%d",
                        t, m.yes_ask, m.no_ask, m.volume)

    def _resolve_session_pnl(self):
        """Resolve PnL for all trades at match end (Fix 2 + Fix 6).

        Uses the regulation-time score snapshot (90') to determine winners.
        Match winner markets settle at 90 min + stoppage time ONLY.
        """
        if not self._trades:
            return

        # Use regulation snapshot if available, otherwise use final score
        if self._regulation_snapshot_taken:
            score_home = self._regulation_home_goals
            score_away = self._regulation_away_goals
            logger.info("Resolving trades against REGULATION score: %d-%d", score_home, score_away)
        elif self._prev_live_state:
            score_home = self._prev_live_state.home_score
            score_away = self._prev_live_state.away_score
            logger.info("Resolving trades against final score: %d-%d (no regulation snapshot)", score_home, score_away)
        else:
            logger.warning("No score data available — cannot resolve trades")
            return

        regulation_winner = self._adjust_for_regulation(score_home, score_away)
        total_pnl = 0.0

        for trade in self._trades:
            outcome = trade["outcome"]
            if outcome == regulation_winner:
                result = "win"
            elif (outcome == "draw" and regulation_winner == "draw"):
                result = "win"
            else:
                result = "loss"

            pnl = self._resolve_pnl(trade, result)
            total_pnl += pnl
            trade["settlement_result"] = result
            trade["settlement_pnl"] = round(pnl, 4)
            trade["regulation_score"] = f"{score_home}-{score_away}"

            logger.info(
                "SETTLE: %s %s x%d @ $%.2f → %s (PnL=$%.2f)",
                outcome.upper(), trade["ticker"], trade["count"],
                trade["price"], result.upper(), pnl,
            )

            # Append to persistent ledger
            self._append_trade_ledger(trade)

        self._bankroll += total_pnl
        logger.info(
            "Session PnL: $%.2f | New bankroll: $%.2f | Trades: %d",
            total_pnl, self._bankroll, len(self._trades),
        )

    def _append_trade_ledger(self, trade: dict) -> None:
        """Append a resolved trade to the persistent ledger (Fix 2)."""
        try:
            with open(TRADE_LEDGER, "a") as f:
                f.write(json.dumps(trade) + "\n")
        except Exception as e:
            logger.warning("Failed to write trade ledger: %s", e)

    def _save_bankroll_state(self) -> None:
        """Save bankroll and session metadata to persistent state (Fix 2)."""
        state = {
            "bankroll": round(self._bankroll, 2),
            "start_of_day": round(self._start_of_day_bankroll, 2),
            "session_start": datetime.now(IST).isoformat(),
            "total_pnl": round(self._bankroll - self._start_of_day_bankroll, 2),
            "sessions": [],
            "last_updated": datetime.now(IST).isoformat(),
            "last_match": f"{self.match_home} vs {self.match_away}",
        }

        # Load existing state and append this session
        if BANKROLL_FILE.exists():
            try:
                existing = json.loads(BANKROLL_FILE.read_text())
                state["sessions"] = existing.get("sessions", [])
                state["bankroll"] = existing.get("bankroll", self._bankroll)
            except Exception:
                pass

        state["sessions"].append({
            "match": f"{self.match_home} vs {self.match_away}",
            "trades": len(self._trades),
            "pnl": round(self._bankroll - self._start_of_day_bankroll, 2),
            "time": datetime.now(IST).isoformat(),
        })

        # Keep last 30 sessions
        state["sessions"] = state["sessions"][-30:]

        BANKROLL_FILE.write_text(json.dumps(state, indent=2))
        logger.info("Bankroll state saved: $%.2f", self._bankroll)

    def _save_calibration_data(self) -> None:
        """Save calibration data for post-match analysis (Fix 7)."""
        if not self._calibration_data:
            return

        report = {
            "match": f"{self.match_home} vs {self.match_away}",
            "time": datetime.now(IST).isoformat(),
            "liquidity_buffer": self.config.liquidity_buffer,
            "orderbook_depth": self.config.orderbook_depth,
            "trades": self._calibration_data,
            "summary": {
                "total_trades": len(self._calibration_data),
                "avg_fill_vs_ask": 0.0,
                "avg_net_edge": 0.0,
            },
        }

        # Compute summary stats
        fills = [t["fill_price"] for t in self._calibration_data]
        asks = [t["market_ask"] for t in self._calibration_data]
        edges = [t["net_edge"] for t in self._calibration_data]
        if fills and asks:
            report["summary"]["avg_fill_vs_ask"] = round(
                sum(f - a for f, a in zip(fills, asks)) / len(fills), 4
            )
        if edges:
            report["summary"]["avg_net_edge"] = round(sum(edges) / len(edges), 4)

        CALIBRATION_FILE.write_text(json.dumps(report, indent=2))
        logger.info("Calibration data saved: %d trades", len(self._calibration_data))

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def _shutdown(self):
        logger.info("Shutting down...")
        self._print_status()

        # Resolve any remaining trades
        if self._trades and not any(t.get("settlement_result") for t in self._trades):
            self._resolve_session_pnl()

        db = load_db()
        db["trades"].extend(self._trades)
        db["bankroll"] = self._bankroll
        save_db(db)

        # Persist bankroll and calibration (Fix 2 + Fix 7)
        self._save_bankroll_state()
        self._save_calibration_data()

        logger.info("Total trades this session: %d", len(self._trades))
        logger.info("Bot finished.")


def main():
    bot = GitHubBot()
    bot.run()


if __name__ == "__main__":
    main()
