"""Paper trading runner for Kalshi demo with live match data.

Two modes:
1. Live mode: Polls API-Football for match state + Kalshi for odds. Full edge detection.
2. Market-only mode: Polls Kalshi for odds, logs signals (no model predictions).

Usage:
    # Live mode (World Cup Final)
    python run_paper_trade.py

    # Market-only mode (no live match)
    DRY_RUN=true python run_paper_trade.py --market-only
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from config import load_config, Config
from market.api_football_client import ApiFootballClient, LiveMatchState
from market.kalshi_client import KalshiClient, KalshiMarket
from model.predict import WinPredictor
from trading.edge_calculator import EdgeCalculator
from trading.kelly_sizer import KellySizer
from vision.game_state import GameState

logger = logging.getLogger(__name__)

# Paths
SIGNALS_DIR = Path("data/paper_signals")
STATE_FILE = SIGNALS_DIR / "current_state.json"
TRADES_LOG = SIGNALS_DIR / "trades_log.jsonl"


@dataclass
class ActiveMarket:
    """Tracks a single Kalshi market we're paper trading."""
    ticker: str
    event_ticker: str
    title: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int
    last_odds: Dict[str, float] = field(default_factory=dict)
    trades_count: int = 0
    last_trade_side: str = ""  # "yes" or "no"
    last_trade_price: float = 0.0


class PaperTrader:
    """Paper trading engine for Kalshi demo with API-Football live data.

    Modes:
    - live: API-Football match state → model prediction → edge detection → paper trades
    - market_only: Polls Kalshi for odds, logs signals (no model)
    """

    def __init__(self, config: Config, market_only: bool = False) -> None:
        self.config = config
        self._running = False
        self._market_only = market_only

        # Components
        self.api_football: Optional[ApiFootballClient] = None
        self.kalshi: Optional[KalshiClient] = None
        self.predictor: Optional[WinPredictor] = None
        self.edge_calculator: Optional[EdgeCalculator] = None
        self.kelly_sizer: Optional[KellySizer] = None

        # State
        self._active_markets: Dict[str, ActiveMarket] = {}
        self._bankroll: float = 1100.0
        self._cycle_count: int = 0
        self._total_signals: int = 0
        self._total_trades: int = 0
        self._total_edge_bets: int = 0
        self._scan_interval: int = 30  # Kalshi event scan
        self._poll_interval: int = 30  # API-Football poll
        self._match_state: Optional[LiveMatchState] = None
        self._prev_match_state: Optional[LiveMatchState] = None
        self._game_state: Optional[GameState] = None
        self._last_prediction: Optional[Dict] = None

        # World Cup Final market tickers
        self._WC_FINAL_TICKERS = {
            "ESP": "KXWCGAME-26JUL19ESPARG-ESP",
            "ARG": "KXWCGAME-26JUL19ESPARG-ARG",
            "TIE": "KXWCGAME-26JUL19ESPARG-TIE",
        }

    def initialize(self) -> bool:
        """Initialize all components. Returns True if ready."""
        logger.info("=" * 60)
        logger.info("PAPER TRADER INITIALIZING")
        logger.info("Mode: %s", "MARKET-ONLY" if self._market_only else "LIVE (API-Football)")
        logger.info("Bankroll: $%.2f", self._bankroll)
        logger.info("=" * 60)

        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

        # API-Football
        if not self._market_only:
            if not self.config.api_football_key:
                logger.error("API_FOOTBALL_KEY not set")
                return False
            self.api_football = ApiFootballClient(api_key=self.config.api_football_key)
            logger.info("API-Football client initialized")

        # Kalshi client
        if not self.config.kalshi_api_key:
            logger.error("KALSHI_API_KEY not set")
            return False

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

        # ML model
        if not self._market_only:
            try:
                self.predictor = WinPredictor(model_dir="model")
                self.predictor.initialize()
                logger.info("ML model loaded (stacking ensemble)")
            except Exception as e:
                logger.warning("ML model not available: %s (falling back to market-only)", e)
                self.predictor = None

        # Trading components
        self.edge_calculator = EdgeCalculator(
            edge_threshold=self.config.edge_threshold,
            confidence_threshold=self.config.confidence_threshold,
        )
        self.kelly_sizer = KellySizer(
            base_kelly=self.config.kelly_fraction,
            max_bet_pct=self.config.max_bet_pct,
            min_bet_usd=self.config.min_bet_usd,
        )

        logger.info("Initialization complete")
        return True

    def run(self) -> None:
        """Main run loop."""
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Pre-discover World Cup Final markets (must be after Kalshi init)
        self._discover_wc_final_markets()

        logger.info("Starting paper trading loop (poll every %ds)...", self._poll_interval)
        self._print_status()

        last_kalshi_scan = 0
        last_api_poll = 0
        while self._running:
            try:
                now = time.time()

                # Poll API-Football for live match state
                if not self._market_only and now - last_api_poll >= self._poll_interval:
                    self._poll_match_state()
                    last_api_poll = now

                    # Check for edge right after fresh data
                    if self.predictor and self._last_prediction:
                        self._check_edges()

                # Scan Kalshi for new events
                if now - last_kalshi_scan >= self._scan_interval:
                    self._scan_events()
                    last_kalshi_scan = now

                # Update prices
                self._update_prices()

                # Status every 60 cycles (~60s)
                self._cycle_count += 1
                if self._cycle_count % 60 == 0:
                    self._print_status()

                time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as e:
                logger.error("Loop error: %s", e, exc_info=True)
                time.sleep(5)

        self._shutdown()

    def _discover_wc_final_markets(self) -> None:
        """Pre-discover World Cup Final markets on Kalshi."""
        logger.info("Discovering World Cup Final markets...")

        for result_key, ticker in self._WC_FINAL_TICKERS.items():
            try:
                markets = self.kalshi.get_event_markets("KXWCGAME-26JUL19ESPARG")
                for m in markets:
                    if m.ticker == ticker:
                        self._active_markets[ticker] = ActiveMarket(
                            ticker=m.ticker,
                            event_ticker="KXWCGAME-26JUL19ESPARG",
                            title=m.title,
                            yes_bid=m.yes_bid,
                            yes_ask=m.yes_ask,
                            no_bid=m.no_bid,
                            no_ask=m.no_ask,
                            volume=m.volume,
                        )
                        logger.info(
                            "WC Final market: %s | %s | YES bid=%.2f ask=%.2f",
                            ticker, m.title, m.yes_bid, m.yes_ask,
                        )
            except Exception as e:
                logger.warning("Failed to fetch market %s: %s", ticker, e)

        if not self._active_markets:
            logger.warning("No WC Final markets found — will retry in scan")

    def _poll_match_state(self) -> None:
        """Poll API-Football for live match state."""
        fixture_id = self.config.api_football_fixture_id
        if not fixture_id:
            # Auto-discover: search for Spain vs Argentina
            if self.api_football:
                fixture_id = self.api_football.search_world_cup_match(
                    home_team="Spain", away_team="Argentina"
                )
                if not fixture_id:
                    fixture_id = self.api_football.search_world_cup_match(home_team="Argentina")
                if fixture_id:
                    logger.info("Found WC Final fixture ID: %d", fixture_id)
                else:
                    logger.debug("WC Final not found yet (match may not have started)")
                    return

        if not self.api_football:
            return

        self._prev_match_state = self._match_state
        self._match_state = self.api_football.get_live_match(fixture_id)

        if not self._match_state:
            return

        ms = self._match_state

        # Log match state
        logger.info(
            "LIVE: %s %d - %d %s | %s %s' | events=%d | API calls=%d",
            ms.home_team, ms.home_score, ms.away_score, ms.away_team,
            ms.status, ms.clock_minutes,
            len(ms.events), self.api_football.request_count,
        )

        # Convert to GameState for prediction
        self._game_state = self._match_state_to_game_state(ms)

        # Log prediction if available
        if self.predictor and self._game_state:
            try:
                p_home, p_draw, p_away = self.predictor.predict(self._game_state)
                self._last_prediction = {
                    "home": p_home,
                    "draw": p_draw,
                    "away": p_away,
                    "confidence": max(p_home, p_draw, p_away),
                    "clock": ms.clock_minutes,
                }
                logger.info(
                    "PREDICTION: home=%.1f%% draw=%.1f%% away=%.1f%% (conf=%.1f%%)",
                    p_home * 100, p_draw * 100, p_away * 100,
                    self._last_prediction["confidence"] * 100,
                )
            except Exception as e:
                logger.warning("Prediction failed: %s", e)
                self._last_prediction = None

        # Update state file
        self._write_state()

    def _match_state_to_game_state(self, ms: LiveMatchState) -> GameState:
        """Convert API-Football LiveMatchState to GameState for model prediction."""
        # Determine if home team is actually home (in WC Final, it's neutral)
        is_neutral = True  # World Cup Final is always neutral venue

        return GameState(
            match_id=str(ms.fixture_id),
            home_team=ms.home_team,
            away_team=ms.away_team,
            clock_minutes=ms.clock_minutes,
            stoppage_time=0,
            is_extra_time=ms.period >= 3,
            home_score=ms.home_score,
            away_score=ms.away_score,
            ocr_reliable=True,  # API data is always reliable
            consecutive_consistent_reads=10,
            timestamp=ms.last_update,
            # Vision features (from API)
            home_red_cards=ms.home_red_cards,
            away_red_cards=ms.away_red_cards,
            home_pressure_score=ms.home_pressure,
            goals_in_last_10min=self._count_goals_in_window(ms, 10),
            goals_last_15min=self._count_goals_in_window(ms, 15),
            cards_last_15min=self._count_cards_in_window(ms, 15),
            home_shots_on_target=ms.home_stats.shots_on if ms.home_stats else 0,
            away_shots_on_target=ms.away_stats.shots_on if ms.away_stats else 0,
            home_xg_running=ms.home_xg_running,
            away_xg_running=ms.away_xg_running,
            momentum_shift=self._compute_momentum(ms),
            # Pre-match features (World Cup Final defaults)
            home_elo=1900.0,  # Spain ~1900
            away_elo=1880.0,  # Argentina ~1880
            home_form_pts=12,
            away_form_pts=13,
            h2h_home_winrate=0.45,
            is_home_game=False,  # Neutral venue
            referee_cards_per_game=3.5,
            home_squad_value_EUR=1_200_000_000,  # Spain ~€1.2B
            away_squad_value_EUR=1_000_000_000,  # Argentina ~€1.0B
            home_injuries_count=0,
            away_injuries_count=0,
            home_press_pct=ms.home_pressure,
            away_press_pct=1.0 - ms.home_pressure,
            home_xg_last5=ms.home_xg_running,
            away_xg_last5=ms.away_xg_running,
            home_xga_last5=ms.away_xg_running,
            away_xga_last5=ms.home_xg_running,
            competition_tier=1,  # World Cup
            match_importance=1.0,  # Final!
            days_since_last_match_home=7,
            days_since_last_match_away=7,
        )

    def _count_goals_in_window(self, ms: LiveMatchState, window_minutes: int) -> int:
        """Count goals scored in last N minutes from events."""
        count = 0
        for event in ms.events:
            if event.event_type == "Goal" and event.minute >= (ms.clock_minutes - window_minutes):
                count += 1
        return count

    def _count_cards_in_window(self, ms: LiveMatchState, window_minutes: int) -> int:
        """Count cards shown in last N minutes from events."""
        count = 0
        for event in ms.events:
            if event.event_type == "Card" and event.minute >= (ms.clock_minutes - window_minutes):
                count += 1
        return count

    def _compute_momentum(self, ms: LiveMatchState) -> float:
        """Compute momentum shift from recent events."""
        if not self._prev_match_state:
            return 0.0
        prev_xg = self._prev_match_state.home_xg_running + self._prev_match_state.away_xg_running
        curr_xg = ms.home_xg_running + ms.away_xg_running
        delta_time = max(ms.clock_minutes - self._prev_match_state.clock_minutes, 1)
        return (curr_xg - prev_xg) / delta_time

    def _scan_events(self) -> None:
        """Scan Kalshi for open soccer events."""
        logger.debug("Scanning Kalshi for soccer events...")

        # Only scan WC Final-related series to avoid rate limits
        wc_series = ["KXWCGAME", "KXMENWORLDCUP"]
        events = []
        for series in wc_series:
            try:
                resp = self.kalshi._request(
                    "GET", "/events",
                    params={"series_ticker": series, "limit": 50, "status": "open"},
                )
                if resp and "events" in resp:
                    events.extend(resp["events"])
            except Exception:
                pass

        for event in events:
            event_ticker = event.get("event_ticker", "")
            # Only track WC Final and tournament futures
            if "KXWCGAME-26JUL19ESPARG" not in event_ticker and "KXMENWORLDCUP-26" not in event_ticker:
                continue

            markets = self.kalshi.get_event_markets(event_ticker)
            for m in markets:
                if m.ticker not in self._active_markets and m.status == "active":
                    self._active_markets[m.ticker] = ActiveMarket(
                        ticker=m.ticker,
                        event_ticker=event_ticker,
                        title=m.title,
                        yes_bid=m.yes_bid,
                        yes_ask=m.yes_ask,
                        no_bid=m.no_bid,
                        no_ask=m.no_ask,
                        volume=m.volume,
                    )
                    logger.info(
                        "New market: %s | %s | YES bid=%.2f ask=%.2f",
                        m.ticker, m.title, m.yes_bid, m.yes_ask,
                    )

    def _update_prices(self) -> None:
        """Update prices for all active markets."""
        for ticker, market in list(self._active_markets.items()):
            book = self.kalshi.get_orderbook(ticker)
            if book:
                market.yes_bid = book.yes_bid
                market.yes_ask = book.yes_ask
                market.no_bid = book.no_bid
                market.no_ask = book.no_ask

                mid = (book.yes_bid + book.yes_ask) / 2 if book.yes_ask > 0 else 0
                market.last_odds = {
                    "yes_mid": mid,
                    "yes_bid": book.yes_bid,
                    "yes_ask": book.yes_ask,
                    "no_bid": book.no_bid,
                    "no_ask": book.no_ask,
                }

    def _check_edges(self) -> None:
        """Check for trading edges and place paper trades."""
        if not self.predictor or not self._game_state or not self._last_prediction:
            return

        pred = self._last_prediction
        probs = (pred["home"], pred["draw"], pred["away"])

        # Skip if clock > final_minutes_skip
        if self._game_state.clock_minutes > (90 - self.config.final_minutes_skip):
            return

        # Map prediction outcomes to Kalshi market tickers
        outcome_map = {
            self._WC_FINAL_TICKERS["ESP"]: probs[0],  # Spain = home
            self._WC_FINAL_TICKERS["ARG"]: probs[2],  # Argentina = away
            self._WC_FINAL_TICKERS["TIE"]: probs[1],  # Draw
        }

        for ticker, model_prob in outcome_map.items():
            if ticker not in self._active_markets:
                continue

            market = self._active_markets[ticker]
            odds = market.last_odds
            if not odds or odds["yes_ask"] <= 0:
                continue

            # Cooldown: don't re-bet same market if we already traded it
            if market.trades_count > 0:
                continue

            # Calculate edge
            market_prob = odds["yes_mid"] if odds["yes_mid"] > 0 else odds["yes_ask"]
            edge = model_prob - market_prob

            if edge < self.config.edge_threshold:
                continue

            # Calculate confidence
            confidence = max(probs)

            if confidence < self.config.confidence_threshold:
                continue

            # Kelly sizing
            outcome = "home" if "ESP" in ticker else ("away" if "ARG" in ticker else "draw")
            kelly = self.kelly_sizer.calculate(
                outcome=outcome,
                edge=edge,
                market_prob=market_prob,
                bankroll=self._bankroll,
                model_prob=model_prob,
            )

            if kelly.bet_usd < self.config.min_bet_usd:
                continue

            # PLACE PAPER TRADE
            self._total_edge_bets += 1

            trade_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "title": market.title,
                "side": "yes",
                "price": odds["yes_ask"],
                "model_prob": round(model_prob, 4),
                "market_prob": round(market_prob, 4),
                "edge": round(edge, 4),
                "confidence": round(confidence, 4),
                "kelly_bet": round(kelly.bet_usd, 2),
                "clock": self._game_state.clock_minutes,
                "score": f"{self._game_state.home_score}-{self._game_state.away_score}",
            }

            # Actually place order on Kalshi demo
            if not self.config.dry_run:
                try:
                    order = self.kalshi.place_order(
                        ticker=ticker,
                        yes_price=f"{odds['yes_ask']:.4f}",
                        count=max(int(kelly.bet_usd / odds["yes_ask"]), 1),
                        side="bid",
                    )
                    if order:
                        trade_record["order_id"] = str(order)
                        trade_record["status"] = "placed"
                        self._total_trades += 1
                        market.trades_count += 1
                        market.last_trade_side = "yes"
                        market.last_trade_price = odds["yes_ask"]
                    else:
                        trade_record["status"] = "failed"
                except Exception as e:
                    trade_record["status"] = f"error: {e}"
            else:
                trade_record["status"] = "dry_run"

            # Log trade
            self._log_trade(trade_record)

            logger.info(
                "EDGE BET: %s | model=%.1f%% market=%.1f%% edge=%.1f%% | $%.2f | %s",
                ticker, model_prob * 100, market_prob * 100, edge * 100,
                kelly.bet_usd, trade_record["status"],
            )

    def _log_trade(self, trade: Dict) -> None:
        """Append trade to JSONL log."""
        with open(TRADES_LOG, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def _write_state(self) -> None:
        """Write current state to JSON file."""
        match_info = {}
        if self._match_state:
            ms = self._match_state
            match_info = {
                "home": ms.home_team,
                "away": ms.away_team,
                "score": f"{ms.home_score}-{ms.away_score}",
                "clock": ms.clock_minutes,
                "status": ms.status,
                "events": len(ms.events),
                "api_calls": self.api_football.request_count if self.api_football else 0,
            }

        state = {
            "last_update": datetime.now(timezone.utc).isoformat(),
            "mode": "market_only" if self._market_only else "live",
            "bankroll": self._bankroll,
            "active_markets": len(self._active_markets),
            "total_signals": self._total_signals,
            "total_trades": self._total_trades,
            "total_edge_bets": self._total_edge_bets,
            "prediction": self._last_prediction,
            "match": match_info,
            "markets": {
                t: {
                    "title": m.title,
                    "odds": m.last_odds,
                    "volume": m.volume,
                    "trades": m.trades_count,
                }
                for t, m in self._active_markets.items()
            },
        }

        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

        self._total_signals += 1

    def _print_status(self) -> None:
        """Print current status."""
        mode = "MARKET-ONLY" if self._market_only else "LIVE"
        logger.info(
            "STATUS [%s]: bankroll=$%.2f | markets=%d | signals=%d | edge_bets=%d | trades=%d",
            mode, self._bankroll, len(self._active_markets),
            self._total_signals, self._total_edge_bets, self._total_trades,
        )

        # Show match state
        if self._match_state:
            ms = self._match_state
            logger.info(
                "  MATCH: %s %d - %d %s | %s %s' | events=%d",
                ms.home_team, ms.home_score, ms.away_score, ms.away_team,
                ms.status, ms.clock_minutes, len(ms.events),
            )

        # Show prediction
        if self._last_prediction:
            p = self._last_prediction
            logger.info(
                "  PREDICTION: home=%.1f%% draw=%.1f%% away=%.1f%% (conf=%.1f%%)",
                p["home"] * 100, p["draw"] * 100, p["away"] * 100, p["confidence"] * 100,
            )

        # Show active markets
        for ticker, market in self._active_markets.items():
            odds = market.last_odds
            if odds:
                logger.info(
                    "  %s: YES bid=%.2f ask=%.2f | vol=%d",
                    ticker,
                    odds.get("yes_bid", 0),
                    odds.get("yes_ask", 0),
                    market.volume,
                )

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received")
        self._running = False

    def _shutdown(self) -> None:
        logger.info("Shutting down paper trader...")
        if self.kalshi:
            self.kalshi.cancel_all_orders()
        self._print_status()
        logger.info("Paper trading stopped")


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("data/paper_trade.log"),
        ],
    )

    # Check for --market-only flag
    market_only = "--market-only" in sys.argv

    config = load_config()

    trader = PaperTrader(config, market_only=market_only)
    if not trader.initialize():
        logger.error("Initialization failed")
        sys.exit(1)

    trader.run()


if __name__ == "__main__":
    main()
