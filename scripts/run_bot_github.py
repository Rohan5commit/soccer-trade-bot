#!/usr/bin/env python3
"""Paper trade bot for GitHub Actions.

Runs as a single match session:
1. Receives match info via env vars (from workflow_dispatch)
2. Loads pre-trained models
3. Discovers Kalshi markets for this specific match
4. Fetches live match data from KickoffAPI (score, clock, xG, cards, etc.)
5. Runs model predictions on enriched GameState
6. Places paper trades via Kalshi demo API

Unlike run_paper_trade.py, this does NOT discover matches —
it runs a single match passed in by the watcher workflow.
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
from market.kalshi_client import KalshiClient, KalshiMarket
from market.kickoff_api_client import KickoffApiClient, LiveMatchState
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

# How often to update live data from KickoffAPI (seconds)
LIVE_UPDATE_INTERVAL = 30
# How often to update Kalshi prices (seconds)
PRICE_UPDATE_INTERVAL = 30
# Trade cooldown per outcome (seconds)
TRADE_COOLDOWN = 120
# Max match duration before auto-stop (minutes)
MAX_MATCH_MINUTES = 120


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


class GitHubBot:
    """Paper trader for GitHub Actions (single match with live data)."""

    def __init__(self):
        self.config = load_config()
        self.kalshi: Optional[KalshiClient] = None
        self.kickoff: Optional[KickoffApiClient] = None
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
            except Exception:
                pass

        # State
        self._running = True
        self._markets: Dict[str, KalshiMarket] = {}
        self._code_to_outcome: Dict[str, str] = {}
        self._bankroll: float = 0.0
        self._trades: List[dict] = []
        self._poll_count = 0
        self._order_cooldown: Dict[str, float] = {}
        self._last_live_update: float = 0

        # KickoffAPI fixture tracking
        self._kickoff_fixture_id: Optional[int] = None
        self._prev_live_state: Optional[LiveMatchState] = None

        # Game state — enriched by KickoffAPI live data
        self._game_state = GameState(
            home_team=self.match_home,
            away_team=self.match_away,
        )

    def initialize(self) -> bool:
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

        # KickoffAPI client (for live match data)
        kickoff_keys = []
        k1 = os.environ.get("KICKOFF_API_KEY", "")
        k2 = os.environ.get("KICKOFF_API_KEY_2", "")
        if k1:
            kickoff_keys.append(k1)
        if k2:
            kickoff_keys.append(k2)
        if kickoff_keys:
            self.kickoff = KickoffApiClient(keys=kickoff_keys)
            logger.info("KickoffAPI client initialized (%d keys, %d remaining)",
                        len(kickoff_keys), self.kickoff.remaining)
        else:
            logger.warning("No KICKOFF_API_KEY — running without live data")

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

        # Discover markets for this specific event
        try:
            markets = self.kalshi.get_event_markets(self.event_ticker)
            for m in markets:
                self._markets[m.ticker] = m
                logger.info(
                    "  Market: %s (yes=$%.2f no=$%.2f vol=%d) title=%r sub=%r",
                    m.ticker, m.yes_ask, m.no_ask, m.volume, m.title, m.subtitle,
                )
        except Exception as e:
            logger.error("Failed to discover markets: %s", e)
            return False

        if not self._markets:
            logger.warning("No markets found for %s — match may be closed or cancelled", self.event_ticker)
            return False

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

        # Find KickoffAPI fixture for live data
        if self.kickoff:
            self._find_kickoff_fixture()

        return True

    def _find_kickoff_fixture(self) -> None:
        """Find KickoffAPI fixture ID by matching team names."""
        if not self.kickoff:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            fixtures = self.kickoff.get_fixtures_by_date(today)
            if not fixtures:
                # Try tomorrow (matches may span midnight)
                tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
                fixtures = self.kickoff.get_fixtures_by_date(tomorrow)

            home_norm = _normalize_team_name(self.match_home)
            away_norm = _normalize_team_name(self.match_away)

            best_match = None
            best_score = 0

            for f in fixtures:
                f_home = _normalize_team_name(f.get("homeTeam", {}).get("name", ""))
                f_away = _normalize_team_name(f.get("awayTeam", {}).get("name", ""))

                # Check both orderings
                score1 = 0
                if home_norm in f_home or f_home in home_norm:
                    score1 += 1
                if away_norm in f_away or f_away in away_norm:
                    score1 += 1

                score2 = 0
                if home_norm in f_away or f_away in home_norm:
                    score2 += 1
                if away_norm in f_home or f_home in away_norm:
                    score2 += 1

                score = max(score1, score2)
                if score > best_score:
                    best_score = score
                    best_match = f

            if best_match and best_score >= 2:
                self._kickoff_fixture_id = best_match.get("id")
                f_home = best_match.get("homeTeam", {}).get("name", "?")
                f_away = best_match.get("awayTeam", {}).get("name", "?")
                logger.info("KickoffAPI fixture found: %s vs %s (ID=%d)",
                            f_home, f_away, self._kickoff_fixture_id)
            else:
                logger.warning("KickoffAPI: no matching fixture for %s vs %s (checked %d fixtures)",
                               self.match_home, self.match_away, len(fixtures))
        except Exception as e:
            logger.warning("KickoffAPI fixture discovery failed: %s", e)

    def _fetch_live_state(self) -> bool:
        """Fetch live match data from KickoffAPI and update GameState.

        Returns True if live data was successfully fetched.
        """
        if not self.kickoff or not self._kickoff_fixture_id:
            return False

        now = time.time()
        if now - self._last_live_update < LIVE_UPDATE_INTERVAL:
            return True  # Still fresh

        self._last_live_update = now

        try:
            state = self.kickoff.get_live_match(self._kickoff_fixture_id)
            if not state:
                return False

            # Log live state
            logger.info(
                "LIVE: %s %d - %d %s | %s %.0f' | events=%d | API=%d remaining",
                state.home_team, state.home_score, state.away_score, state.away_team,
                state.status, state.clock_minutes,
                len(state.events), self.kickoff.remaining,
            )

            # Update game state from live data
            prev = self._prev_live_state
            self._game_state = self._match_state_to_game_state(state)
            self._prev_live_state = state
            return True

        except Exception as e:
            logger.warning("KickoffAPI live update failed: %s", e)
            return False

    def _match_state_to_game_state(self, ms: LiveMatchState) -> GameState:
        """Convert KickoffAPI LiveMatchState to GameState for model prediction."""
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
            logger.warning("Bot initialization failed — no markets available. Exiting cleanly.")
            return

        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Starting trading loop...")
        last_status_time = 0
        last_price_time = 0

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

                    # Pre-match: sleep until 5 minutes before kickoff
                    if elapsed < -5:
                        logger.info(
                            "Pre-match: kickoff in %.0f min. Sleeping 60s...",
                            abs(elapsed),
                        )
                        time.sleep(60)
                        continue

                    # Only update clock if no live data (KickoffAPI overrides this)
                    if elapsed > 0 and not self._kickoff_fixture_id:
                        self._game_state.clock_minutes = min(elapsed, 90)

                # Fetch live match data from KickoffAPI
                self._fetch_live_state()

                # Update prices (throttled)
                if now_ts - last_price_time >= PRICE_UPDATE_INTERVAL:
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

    def _check_edges(self):
        if not self.edge_calc or not self.kelly:
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
        )

        if not analysis.any_tradable:
            return

        best = analysis.best_edge
        if best:
            self._place_paper_trade(best, model_probs)

    def _place_paper_trade(self, edge_result, model_probs: dict):
        outcome = edge_result.outcome
        now = time.time()

        if outcome in self._order_cooldown:
            if now - self._order_cooldown[outcome] < TRADE_COOLDOWN:
                return

        ticker = self._find_ticker_for_outcome(outcome)
        if not ticker:
            return

        market = self._markets[ticker]
        price = edge_result.market_ask

        kelly_result = self.kelly.calculate(
            outcome=outcome,
            edge=edge_result.edge,
            model_prob=edge_result.model_prob,
            market_prob=edge_result.market_prob,
            bankroll=self._bankroll,
        )

        if not kelly_result or kelly_result.bet_usd < self.config.min_bet_usd:
            return

        bet_usd = kelly_result.bet_usd
        count = max(1, int(bet_usd / price)) if price > 0 else 0
        if count <= 0:
            return

        logger.info(
            "PAPER TRADE: BUY %s %s x%d @ $%.2f (model=%.3f market=%.3f edge=+%.3f)",
            outcome.upper(), ticker, count, price,
            edge_result.model_prob, edge_result.market_prob, edge_result.edge,
        )

        result = self.kalshi.place_order(
            ticker=ticker,
            side="bid",
            count=count,
            yes_price=price,
        )

        trade = {
            "time": datetime.now(IST).isoformat(),
            "match": f"{self.match_home} vs {self.match_away}",
            "event_ticker": self.event_ticker,
            "ticker": ticker,
            "outcome": outcome,
            "side": "yes",
            "count": count,
            "price": price,
            "bet_usd": bet_usd,
            "model_prob": edge_result.model_prob,
            "market_prob": edge_result.market_prob,
            "edge": edge_result.edge,
            "result": "submitted" if result else "failed",
            "clock": self._game_state.clock_minutes,
            "score": f"{self._game_state.home_score}-{self._game_state.away_score}",
        }
        self._trades.append(trade)
        log_trade(trade)
        self._order_cooldown[outcome] = now

    def _print_status(self):
        elapsed = 0
        if self.match_kickoff:
            elapsed = (datetime.now(IST) - self.match_kickoff).total_seconds() / 60

        logger.info("--- STATUS (T+%.0f min) ---", elapsed)
        logger.info("  Score: %d-%d | Clock: %.0f' | Trades: %d | Bankroll: $%.2f",
                     self._game_state.home_score, self._game_state.away_score,
                     self._game_state.clock_minutes, len(self._trades), self._bankroll)
        logger.info("  KickoffAPI: fixture=%s remaining=%d",
                     self._kickoff_fixture_id or "none",
                     self.kickoff.remaining if self.kickoff else 0)
        for t, m in self._markets.items():
            logger.info("    %s: yes=$%.2f no=$%.2f vol=%d",
                        t, m.yes_ask, m.no_ask, m.volume)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def _shutdown(self):
        logger.info("Shutting down...")
        self._print_status()

        db = load_db()
        db["trades"].extend(self._trades)
        db["bankroll"] = self._bankroll
        save_db(db)

        logger.info("Total trades this session: %d", len(self._trades))
        logger.info("Bot finished.")


def main():
    bot = GitHubBot()
    bot.run()


if __name__ == "__main__":
    main()
