"""Paper trading runner for Kalshi demo with live match data.

Two modes:
1. Live mode: Polls KickoffAPI for match state + Kalshi for odds. Full edge detection.
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
import os
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
from market.kickoff_api_client import KickoffApiClient, LiveMatchState as KickoffMatchState
from market.worldcup26_client import WorldCup26Client
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
    """Paper trading engine for Kalshi demo with live match data.

    Modes:
    - live: worldcup26.ir/KickoffAPI match state -> model prediction -> edge detection -> paper trades
    - market_only: Polls Kalshi for odds, logs signals (no model)
    """

    def __init__(self, config: Config, market_only: bool = False) -> None:
        self.config = config
        self._running = False
        self._market_only = market_only

        # Components
        self.kickoff: Optional[KickoffApiClient] = None
        self.worldcup26: Optional[WorldCup26Client] = None
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
        self._poll_interval: int = 60  # KickoffAPI poll (60s to conserve requests)
        self._match_state: Optional[KickoffMatchState] = None
        self._prev_match_state: Optional[KickoffMatchState] = None
        self._game_state: Optional[GameState] = None
        self._last_prediction: Optional[Dict] = None
        self._order_cooldown: Dict[str, float] = {}  # ticker -> last order attempt time
        self._ORDER_COOLDOWN_SEC = 30  # min seconds between order attempts per ticker

        # World Cup Final market tickers
        self._WC_FINAL_TICKERS = {
            "ESP": "KXWCGAME-26JUL19ESPARG-ESP",
            "ARG": "KXWCGAME-26JUL19ESPARG-ARG",
            "TIE": "KXWCGAME-26JUL19ESPARG-TIE",
        }

        # Match timing
        self._match_kickoff: Optional[datetime] = None
        self._match_started = False
        self._match_ended = False
        self._WC_FINAL_MONGODB_ID = "679c9c8a5749c4077500e092"
        self._WC_FINAL_EVENT_TICKER = "KXWCGAME-26JUL19ESPARG"

    def initialize(self) -> bool:
        """Initialize all components. Returns True if ready."""
        logger.info("=" * 60)
        logger.info("PAPER TRADER INITIALIZING")
        logger.info("Mode: %s", "MARKET-ONLY" if self._market_only else "LIVE (worldcup26 + KickoffAPI)")
        logger.info("Bankroll: $%.2f", self._bankroll)
        logger.info("=" * 60)

        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

        # WorldCup26 client (primary — unlimited, no Cloudflare)
        if not self._market_only:
            self.worldcup26 = WorldCup26Client()
            logger.info("WorldCup26 client initialized (primary)")

        # KickoffAPI client (backup — has events/stats, but may be Cloudflare-blocked)
        if not self._market_only:
            keys = []
            key1 = os.environ.get("KICKOFF_API_KEY", "")
            key2 = os.environ.get("KICKOFF_API_KEY_2", "")
            if key1:
                keys.append(key1)
            if key2:
                keys.append(key2)
            if keys:
                self.kickoff = KickoffApiClient(keys=keys)
                logger.info("KickoffAPI client initialized with %d keys (backup)", len(keys))

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

        # Detect match schedule from worldcup26 data
        self._detect_match_schedule()

        logger.info("Starting paper trading loop (poll every %ds)...", self._poll_interval)
        self._print_status()

        last_kalshi_scan = 0
        last_api_poll = 0
        while self._running:
            try:
                now = time.time()

                # Poll for live match state
                if not self._market_only and now - last_api_poll >= self._poll_interval:
                    self._poll_match_state()
                    last_api_poll = now

                    # If match hasn't started yet, poll more frequently (every 10s)
                    if not self._match_started and not self._match_ended:
                        self._poll_interval = 10
                    elif self._match_started and not self._match_ended:
                        self._poll_interval = 30  # Live: poll every 30s
                    else:
                        self._poll_interval = 300  # Post-match: poll every 5min

                    # Check for edge right after fresh data
                    if self.predictor and self._last_prediction and self._match_started:
                        self._check_edges()

                # Scan Kalshi for new/changed events
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

    def _detect_match_schedule(self) -> None:
        """Detect match schedule from worldcup26.ir data."""
        if not self.worldcup26:
            return

        try:
            match_data = self.worldcup26.get_match(self._WC_FINAL_MONGODB_ID)
            if not match_data:
                logger.warning("Could not fetch WC Final data for schedule detection")
                return

            self._match_kickoff = self.worldcup26.parse_local_date(match_data)
            status = self.worldcup26.get_match_status(match_data)

            if self._match_kickoff:
                now = datetime.now(timezone.utc)
                mins_until = (self._match_kickoff - now).total_seconds() / 60
                logger.info(
                    "WC Final kickoff: %s UTC (%.0f min from now) | status: %s",
                    self._match_kickoff.strftime("%Y-%m-%d %H:%M"),
                    mins_until, status,
                )

                if status == "finished":
                    self._match_ended = True
                    logger.info("Match already finished — will not place trades")
                elif status == "live" or (mins_until <= 0 and mins_until > -120):
                    self._match_started = True
                    logger.info("Match is LIVE — starting trading mode")
                elif mins_until > 0:
                    logger.info("Match starts in %.0f min — waiting...", mins_until)
            else:
                logger.warning("Could not parse kickoff time from worldcup26 data")

        except Exception as e:
            logger.warning("Failed to detect match schedule: %s", e)

    def _discover_wc_final_markets(self) -> None:
        """Pre-discover World Cup Final markets on Kalshi.

        Fetches markets regardless of status (open, finalized, etc.)
        since we need to track them for the entire match lifecycle.
        """
        logger.info("Discovering World Cup Final markets...")

        event_ticker = self._WC_FINAL_EVENT_TICKER

        try:
            # Fetch ALL markets for this event (no status filter)
            resp = self.kalshi._request(
                "GET",
                "/markets",
                params={"event_ticker": event_ticker, "limit": 10},
            )
            if not resp or "markets" not in resp:
                logger.warning("No markets found for %s", event_ticker)
                return

            for item in resp["markets"]:
                ticker = item.get("ticker", "")
                if ticker in self._WC_FINAL_TICKERS.values():
                    market = self.kalshi._parse_market(item)
                    if market:
                        self._active_markets[ticker] = ActiveMarket(
                            ticker=market.ticker,
                            event_ticker=event_ticker,
                            title=market.title,
                            yes_bid=market.yes_bid,
                            yes_ask=market.yes_ask,
                            no_bid=market.no_bid,
                            no_ask=market.no_ask,
                            volume=market.volume,
                        )
                        logger.info(
                            "WC Final market: %s | %s | status=%s | YES bid=%.2f ask=%.2f",
                            ticker, market.title, market.status,
                            market.yes_bid, market.yes_ask,
                        )

        except Exception as e:
            logger.warning("Failed to fetch WC Final markets: %s", e)

        if not self._active_markets:
            logger.warning("No WC Final markets found — will retry in scan")

    def _poll_match_state(self) -> None:
        """Poll for live match state.

        Primary: worldcup26.ir (unlimited, no Cloudflare).
        Fallback: KickoffAPI (backup, may be Cloudflare-blocked).

        Handles stale data: if worldcup26 says "notstarted" but match
        should be live, retries with KickoffAPI.
        """
        fixture_id = int(os.environ.get("KICKOFF_FIXTURE_ID", "1591866"))
        now = datetime.now(timezone.utc)

        # Try worldcup26.ir first
        if self.worldcup26:
            try:
                match_data = self.worldcup26.get_match(self._WC_FINAL_MONGODB_ID)
                if match_data:
                    # Detect match schedule if not done yet
                    if not self._match_kickoff:
                        self._match_kickoff = self.worldcup26.parse_local_date(match_data)

                    wc_status = self.worldcup26.get_match_status(match_data)
                    should_be_live = self._match_kickoff and (
                        (now - self._match_kickoff).total_seconds() / 60 >= -5 and
                        (now - self._match_kickoff).total_seconds() / 60 <= 120
                    )

                    # Detect stale data: worldcup26 says "notstarted" but match should be live
                    if wc_status == "notstarted" and should_be_live:
                        logger.warning(
                            "worldcup26 says 'notstarted' but match should be live "
                            "(kickoff was %s). Retrying with KickoffAPI...",
                            self._match_kickoff.strftime("%H:%M UTC") if self._match_kickoff else "?",
                        )
                        # Fall through to KickoffAPI fallback below
                    elif wc_status == "finished":
                        if not self._match_ended:
                            self._match_ended = True
                            logger.info("MATCH ENDED (worldcup26 reports finished)")
                        self._parse_worldcup26_data(match_data, fixture_id)
                        return
                    elif wc_status == "live" or (":" in str(match_data.get("time_elapsed", ""))):
                        if not self._match_started:
                            self._match_started = True
                            logger.info("MATCH IS LIVE")
                        self._parse_worldcup26_data(match_data, fixture_id)
                        return
                    else:
                        # Not started yet, not stale — just waiting
                        self._parse_worldcup26_data(match_data, fixture_id)
                        return

            except Exception as e:
                logger.warning("worldcup26.ir failed: %s", e)

        # Fallback: try KickoffAPI directly
        if self.kickoff:
            try:
                self._prev_match_state = self._match_state
                self._match_state = self.kickoff.get_live_match(fixture_id)
                if self._match_state:
                    ms = self._match_state
                    if ms.status in ("1H", "2H", "HT", "ET", "PEN", "LIVE"):
                        if not self._match_started:
                            self._match_started = True
                            logger.info("MATCH IS LIVE (via KickoffAPI)")
                    elif ms.status == "FT":
                        if not self._match_ended:
                            self._match_ended = True
                            logger.info("MATCH ENDED (via KickoffAPI)")

                    logger.info(
                        "LIVE: %s %d - %d %s | %s %s' | events=%d | API calls=%d",
                        ms.home_team, ms.home_score, ms.away_score, ms.away_team,
                        ms.status, ms.clock_minutes,
                        len(ms.events), self.kickoff.request_count,
                    )
                    self._game_state = self._match_state_to_game_state(ms)
                    if self.predictor and self._game_state:
                        try:
                            p_home, p_draw, p_away = self.predictor.predict(self._game_state)
                            self._last_prediction = {
                                "home": p_home, "draw": p_draw, "away": p_away,
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
                    self._write_state()
            except Exception as e:
                logger.warning("KickoffAPI fallback failed: %s", e)

    def _parse_worldcup26_data(self, match_data: Dict, fixture_id: int) -> None:
        """Parse worldcup26 match data into internal state."""
        home_team = match_data.get("home_team_name_en", "Spain")
        away_team = match_data.get("away_team_name_en", "Argentina")
        home_score = int(match_data.get("home_score", 0) or 0)
        away_score = int(match_data.get("away_score", 0) or 0)
        time_elapsed = str(match_data.get("time_elapsed", "notstarted"))

        # Determine status and clock
        status = "NS"
        clock_minutes = 0.0
        is_live = False
        period = 1

        if time_elapsed == "finished":
            status = "FT"
            clock_minutes = 90.0
        elif time_elapsed == "notstarted":
            status = "NS"
            clock_minutes = 0.0
        elif ":" in time_elapsed:
            parts = time_elapsed.split(":")
            try:
                clock_minutes = float(parts[0])
                is_live = True
                status = "1H" if clock_minutes <= 45 else "2H"
                period = 1 if clock_minutes <= 45 else 2
            except ValueError:
                pass

        from dataclasses import dataclass as dc

        @dc
        class WC26MatchState:
            fixture_id: int
            home_team: str
            away_team: str
            home_score: int
            away_score: int
            clock_minutes: float
            status: str
            is_live: bool
            period: int
            events: list = field(default_factory=list)
            home_stats: object = None
            away_stats: object = None
            home_xg_running: float = 0.0
            away_xg_running: float = 0.0
            home_pressure: float = 0.5
            home_red_cards: int = 0
            away_red_cards: int = 0
            home_yellow_cards: int = 0
            away_yellow_cards: int = 0
            last_update: float = field(default_factory=time.time)

        self._prev_match_state = self._match_state
        self._match_state = WC26MatchState(
            fixture_id=fixture_id,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            clock_minutes=clock_minutes,
            status=status,
            is_live=is_live,
            period=period,
        )

        logger.info(
            "WC26: %s %d - %d %s | %s %s' | API calls=%d",
            home_team, home_score, away_score, away_team,
            status, clock_minutes, self.worldcup26.request_count,
        )

        # Try to get events/stats from KickoffAPI (backup) if live
        if self.kickoff and is_live:
            try:
                kickoff_state = self.kickoff.get_live_match(fixture_id)
                if kickoff_state and kickoff_state.events:
                    self._match_state.events = kickoff_state.events
                    self._match_state.home_stats = kickoff_state.home_stats
                    self._match_state.away_stats = kickoff_state.away_stats
                    self._match_state.home_xg_running = kickoff_state.home_xg_running
                    self._match_state.away_xg_running = kickoff_state.away_xg_running
                    self._match_state.home_pressure = kickoff_state.home_pressure
                    self._match_state.home_red_cards = kickoff_state.home_red_cards
                    self._match_state.away_red_cards = kickoff_state.away_red_cards
                    self._match_state.home_yellow_cards = kickoff_state.home_yellow_cards
                    self._match_state.away_yellow_cards = kickoff_state.away_yellow_cards
                    logger.info("  + KickoffAPI events/stats overlaid")
            except Exception as e:
                logger.debug("KickoffAPI backup failed: %s", e)

        # Convert to GameState for prediction
        self._game_state = self._match_state_to_game_state(self._match_state)

        # Log prediction if available
        if self.predictor and self._game_state:
            try:
                p_home, p_draw, p_away = self.predictor.predict(self._game_state)
                self._last_prediction = {
                    "home": p_home,
                    "draw": p_draw,
                    "away": p_away,
                    "confidence": max(p_home, p_draw, p_away),
                    "clock": clock_minutes,
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

    def _match_state_to_game_state(self, ms: KickoffMatchState) -> GameState:
        """Convert KickoffAPI LiveMatchState to GameState for model prediction."""
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

    def _count_goals_in_window(self, ms: KickoffMatchState, window_minutes: int) -> int:
        """Count goals scored in last N minutes from events."""
        count = 0
        for event in ms.events:
            if event.event_type == "Goal" and event.minute >= (ms.clock_minutes - window_minutes):
                count += 1
        return count

    def _count_cards_in_window(self, ms: KickoffMatchState, window_minutes: int) -> int:
        """Count cards shown in last N minutes from events."""
        count = 0
        for event in ms.events:
            if event.event_type == "Card" and event.minute >= (ms.clock_minutes - window_minutes):
                count += 1
        return count

    def _compute_momentum(self, ms: KickoffMatchState) -> float:
        """Compute momentum shift from recent events."""
        if not self._prev_match_state:
            return 0.0
        prev_xg = self._prev_match_state.home_xg_running + self._prev_match_state.away_xg_running
        curr_xg = ms.home_xg_running + ms.away_xg_running
        delta_time = max(ms.clock_minutes - self._prev_match_state.clock_minutes, 1)
        return (curr_xg - prev_xg) / delta_time

    def _scan_events(self) -> None:
        """Scan Kalshi for soccer events and markets.

        Fetches ALL market statuses (open, finalized, etc.) since we need
        to track markets through their entire lifecycle.
        """
        logger.debug("Scanning Kalshi for soccer events...")

        # Only scan WC Final-related series to avoid rate limits
        wc_series = ["KXWCGAME", "KXMENWORLDCUP"]
        events = []
        for series in wc_series:
            try:
                resp = self.kalshi._request(
                    "GET", "/events",
                    params={"series_ticker": series, "limit": 50},
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

            # Fetch ALL markets for this event (no status filter)
            try:
                resp = self.kalshi._request(
                    "GET", "/markets",
                    params={"event_ticker": event_ticker, "limit": 10},
                )
                if not resp or "markets" not in resp:
                    continue

                for item in resp["markets"]:
                    ticker = item.get("ticker", "")
                    if ticker not in self._active_markets:
                        market = self.kalshi._parse_market(item)
                        if market:
                            self._active_markets[ticker] = ActiveMarket(
                                ticker=market.ticker,
                                event_ticker=event_ticker,
                                title=market.title,
                                yes_bid=market.yes_bid,
                                yes_ask=market.yes_ask,
                                no_bid=market.no_bid,
                                no_ask=market.no_ask,
                                volume=market.volume,
                            )
                            logger.info(
                                "New market: %s | %s | status=%s | YES bid=%.2f ask=%.2f",
                                ticker, market.title, market.status,
                                market.yes_bid, market.yes_ask,
                            )
                    else:
                        # Update existing market's status/volume
                        market = self.kalshi._parse_market(item)
                        if market:
                            existing = self._active_markets[ticker]
                            existing.yes_bid = market.yes_bid
                            existing.yes_ask = market.yes_ask
                            existing.no_bid = market.no_bid
                            existing.no_ask = market.no_ask
                            existing.volume = market.volume

            except Exception as e:
                logger.debug("Failed to scan event %s: %s", event_ticker, e)

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
        """Check for trading edges and place paper trades.

        Checks ALL active markets on every call (no trades_count skip).
        Per-ticker cooldown prevents rapid-fire orders.
        """
        if not self.predictor or not self._game_state or not self._last_prediction:
            return

        pred = self._last_prediction
        probs = (pred["home"], pred["draw"], pred["away"])

        # Skip if clock > final_minutes_skip
        if self._game_state.clock_minutes > (90 - self.config.final_minutes_skip):
            return

        # Refresh balance once per edge check
        fresh_balance = self.kalshi.get_balance()
        if fresh_balance:
            self._bankroll = fresh_balance

        # Map prediction outcomes to Kalshi market tickers
        outcome_map = {
            self._WC_FINAL_TICKERS["ESP"]: probs[0],  # Spain = home
            self._WC_FINAL_TICKERS["ARG"]: probs[2],  # Argentina = away
            self._WC_FINAL_TICKERS["TIE"]: probs[1],  # Draw
        }

        now = time.time()

        for ticker, model_prob in outcome_map.items():
            if ticker not in self._active_markets:
                continue

            market = self._active_markets[ticker]
            odds = market.last_odds
            if not odds or odds["yes_ask"] <= 0:
                continue

            # Per-ticker cooldown: don't retry failed orders too quickly
            last_attempt = self._order_cooldown.get(ticker, 0)
            if now - last_attempt < self._ORDER_COOLDOWN_SEC:
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
            self._order_cooldown[ticker] = now  # set cooldown NOW

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
                    contract_count = max(int(kelly.bet_usd / odds["yes_ask"]), 1)

                    # Safety: never bet more than 20% of bankroll on a single market
                    max_cost = self._bankroll * 0.20
                    if contract_count * odds["yes_ask"] > max_cost:
                        contract_count = max(int(max_cost / odds["yes_ask"]), 1)

                    order = self.kalshi.place_order(
                        ticker=ticker,
                        yes_price=f"{odds['yes_ask']:.4f}",
                        count=contract_count,
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
                        trade_record["status"] = "failed (no response)"
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
                "api_calls": (self.worldcup26.request_count if self.worldcup26 else 0) + (self.kickoff.request_count if self.kickoff else 0),
            }

        state = {
            "last_update": datetime.now(timezone.utc).isoformat(),
            "mode": "market_only" if self._market_only else "live",
            "bankroll": self._bankroll,
            "active_markets": len(self._active_markets),
            "total_signals": self._total_signals,
            "total_trades": self._total_trades,
            "total_edge_bets": self._total_edge_bets,
            "match_started": self._match_started,
            "match_ended": self._match_ended,
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
        match_state = "WAITING"
        if self._match_ended:
            match_state = "FINISHED"
        elif self._match_started:
            match_state = "IN-PLAY"

        logger.info(
            "STATUS [%s]: bankroll=$%.2f | markets=%d | signals=%d | edge_bets=%d | trades=%d | match=%s",
            mode, self._bankroll, len(self._active_markets),
            self._total_signals, self._total_edge_bets, self._total_trades,
            match_state,
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
                    "  %s: YES bid=%.2f ask=%.2f | vol=%d trades=%d",
                    ticker,
                    odds.get("yes_bid", 0),
                    odds.get("yes_ask", 0),
                    market.volume,
                    market.trades_count,
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
