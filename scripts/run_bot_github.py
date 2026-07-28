#!/usr/bin/env python3
"""Paper trade bot for GitHub Actions.

Runs as a single match session:
1. Receives match info via env vars (from workflow_dispatch)
2. Loads pre-trained models
3. Discovers Kalshi markets for this specific match
4. Polls Kalshi prices every 30s
5. Runs model predictions + edge detection
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


class GitHubBot:
    """Simplified paper trader for GitHub Actions (single match)."""

    def __init__(self):
        self.config = load_config()
        self.kalshi: Optional[KalshiClient] = None
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
        self._last_price_update: float = 0
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
        # Kalshi GAME tickers: SERIES-YYMDD + TEAM1CODE + TEAM2CODE
        # Market tickers: EVENTTICKER-TIE / -CODE1 / -CODE2
        # Home code appears FIRST in the event ticker string.
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

        return True

    def _map_outcome(self, market: KalshiMarket) -> Optional[str]:
        """Map a Kalshi market to home/draw/away outcome.

        Tries ticker suffix first, then market title suffix, then subtitle.
        Kalshi GAME market titles: "Home vs Away - <outcome>"
        where outcome is team name or "Tie"/"Draw".
        """
        t = market.ticker.upper()

        if t.endswith("-HOME") or t.endswith("-YES"):
            return "home"
        if t.endswith("-DRAW"):
            return "draw"
        if t.endswith("-AWAY") or t.endswith("-NO"):
            return "away"

        # Team code fallback: -TIE, -HAC, -AIK, -BRA, etc.
        suffix = market.ticker.split("-")[-1].upper()
        if suffix == "TIE":
            return "draw"
        if suffix in self._code_to_outcome:
            return self._code_to_outcome[suffix]

        # Title suffix after last " - " is the outcome label
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

        # Subtitle fallback
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
        """Find the Kalshi ticker for a given outcome."""
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

        logger.info("Starting trading loop (poll every 60s)...")
        last_price_update = 0

        while self._running:
            try:
                now_ts = time.time()

                # Check if match is over
                if self.match_kickoff:
                    elapsed = (datetime.now(IST) - self.match_kickoff).total_seconds() / 60
                    if elapsed > 120:
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

                    # Update game state clock
                    if elapsed > 0:
                        self._game_state.clock_minutes = min(elapsed, 90)

                # Update prices (throttled to every 30s)
                if now_ts - last_price_update >= 30:
                    self._update_prices()
                    self._last_price_update = now_ts
                    last_price_update = now_ts

                    # Check edges after fresh prices
                    self._check_edges()

                # Status every 300 cycles (~5 min)
                self._poll_count += 1
                if self._poll_count % 300 == 0:
                    self._print_status()

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

        # Build market prices dict for edge calculator
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

        # Get model predictions if available
        if self.predictor:
            try:
                probs = self.predictor.predict(self._game_state)
                model_probs = {"home": probs[0], "draw": probs[1], "away": probs[2]}
                confidence = max(probs)
                logger.debug(
                    "Prediction: home=%.3f draw=%.3f away=%.3f (conf=%.3f)",
                    probs[0], probs[1], probs[2], confidence,
                )
            except Exception as e:
                logger.debug("Prediction failed: %s", e)
                return
        else:
            return

        # Calculate edge
        analysis = self.edge_calc.calculate(
            model_probs=model_probs,
            market_prices=market_prices,
            market_asks=market_asks,
            market_bids=market_bids,
        )

        if not analysis.any_tradable:
            return

        # Place trade on best edge
        best = analysis.best_edge
        if best:
            self._place_paper_trade(best, model_probs)

    def _place_paper_trade(self, edge_result, model_probs: dict):
        outcome = edge_result.outcome
        now = time.time()

        # Cooldown check
        if outcome in self._order_cooldown:
            if now - self._order_cooldown[outcome] < 30:
                return

        ticker = self._find_ticker_for_outcome(outcome)

        if not ticker:
            return

        market = self._markets[ticker]
        price = edge_result.market_ask

        # Kelly sizing
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

        # Place order on Kalshi demo
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
        }
        self._trades.append(trade)
        log_trade(trade)

        self._order_cooldown[outcome] = now

    def _print_status(self):
        elapsed = 0
        if self.match_kickoff:
            elapsed = (datetime.now(IST) - self.match_kickoff).total_seconds() / 60

        logger.info("--- STATUS (T+%.0f min) ---", elapsed)
        logger.info("  Bankroll: $%.2f | Trades: %d", self._bankroll, len(self._trades))
        logger.info("  Markets: %d", len(self._markets))
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
