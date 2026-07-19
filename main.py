"""Soccer Vision-to-Trade Bot: Main Entrypoint.

Orchestrates the full pipeline:
1. Load configuration from environment
2. Initialize vision pipeline (frame extractor, OCR, CLIP, YOLO)
3. Initialize ML model (XGBoost + LightGBM ensemble)
4. Initialize market clients (Polymarket, Kalshi)
5. Initialize trading logic (edge calc, Kelly sizing, kill switch)
6. Run main loop: frame → game state → predict → trade
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from config import load_config, Config
from data.logger import TradeLogger
from market.kalshi_client import KalshiClient
from market.market_selector import MarketSelector
from market.order_manager import OrderManager
from market.polymarket_client import PolymarketClient
from model.predict import WinPredictor
from trading.edge_calculator import EdgeCalculator
from trading.kill_switch import KillSwitch
from trading.kelly_sizer import KellySizer
from trading.signal_engine import SignalEngine
from vision.event_classifier import EventClassifier
from vision.frame_extractor import FrameExtractor
from vision.game_state import GameState
from vision.ocr_pipeline import OCRPipeline
from vision.player_detector import PlayerDetector

logger = logging.getLogger(__name__)


class SoccerTradeBot:
    """Main bot orchestrator."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._running = False

        # Vision
        self.frame_extractor: FrameExtractor | None = None
        self.ocr_pipeline: OCRPipeline | None = None
        self.event_classifier: EventClassifier | None = None
        self.player_detector: PlayerDetector | None = None

        # ML
        self.predictor: WinPredictor | None = None

        # Market
        self.polymarket: PolymarketClient | None = None
        self.kalshi: KalshiClient | None = None
        self.market_selector: MarketSelector | None = None
        self.order_manager: OrderManager | None = None

        # Trading
        self.edge_calculator: EdgeCalculator | None = None
        self.kelly_sizer: KellySizer | None = None
        self.kill_switch: KillSwitch | None = None
        self.signal_engine: SignalEngine | None = None

        # Data
        self.trade_logger: TradeLogger | None = None

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Soccer Trade Bot...")
        logger.info("Mode: %s", "DRY RUN" if self.config.dry_run else "LIVE")

        # Data logger
        self.trade_logger = TradeLogger(db_path=self.config.db_path)

        # Vision pipeline
        self.frame_extractor = FrameExtractor(
            stream_url=self.config.stream_url,
            lag_threshold=self.config.stream_lag_max_seconds,
        )
        self.ocr_pipeline = OCRPipeline(
            confidence_threshold=self.config.ocr_confidence_threshold,
            score_roi=self.config.score_roi,
            clock_roi=self.config.clock_roi,
        )
        self.event_classifier = EventClassifier(confidence_threshold=0.75)
        self.player_detector = PlayerDetector(model_path=self.config.yolo_model_path)

        # ML model
        self.predictor = WinPredictor(model_dir="model")

        # Market clients
        if self.config.polymarket_private_key:
            self.polymarket = PolymarketClient(
                api_key=self.config.polymarket_api_key,
                private_key=self.config.polymarket_private_key,
                dry_run=self.config.dry_run,
            )
            self.polymarket.initialize()

        if self.config.kalshi_api_key and self.config.kalshi_private_key:
            self.kalshi = KalshiClient(
                api_key=self.config.kalshi_api_key,
                private_key_pem=self.config.kalshi_private_key,
                dry_run=self.config.dry_run,
            )

        self.market_selector = MarketSelector(
            polymarket=self.polymarket,
            kalshi=self.kalshi,
        )
        self.order_manager = OrderManager(
            polymarket=self.polymarket,
            kalshi=self.kalshi,
            min_bet=self.config.min_bet_usd,
            max_bet_pct=self.config.max_bet_pct,
            dry_run=self.config.dry_run,
        )

        # Trading logic
        self.edge_calculator = EdgeCalculator(
            edge_threshold=self.config.edge_threshold,
            confidence_threshold=self.config.confidence_threshold,
        )
        self.kelly_sizer = KellySizer(
            base_kelly=self.config.kelly_fraction,
            max_bet_pct=self.config.max_bet_pct,
            min_bet_usd=self.config.min_bet_usd,
        )
        self.kill_switch = KillSwitch(
            stream_lag_max=self.config.stream_lag_max_seconds,
            ocr_fail_threshold=self.config.ocr_fail_threshold,
            ocr_confidence_threshold=self.config.ocr_confidence_threshold,
            api_error_threshold=self.config.api_error_threshold,
            api_error_window=self.config.api_error_window_seconds,
            drawdown_threshold=self.config.drawdown_threshold_pct,
        )
        self.signal_engine = SignalEngine(
            predictor=self.predictor,
            market_selector=self.market_selector,
            order_manager=self.order_manager,
            edge_calculator=self.edge_calculator,
            kelly_sizer=self.kelly_sizer,
            kill_switch=self.kill_switch,
            trade_logger=self.trade_logger,
            dry_run=self.config.dry_run,
            bankroll=1000.0,
            final_minutes_skip=self.config.final_minutes_skip,
        )

        logger.info("All components initialized")

    def run(self) -> None:
        """Main run loop."""
        self._running = True

        # Handle signals
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Start frame extraction
        self.frame_extractor.start()

        # Initialize game state
        game_state = GameState(
            match_id=self.config.match_id,
            home_team="",
            away_team="",
        )

        logger.info("Starting main loop...")
        logger.info("Stream URL: %s", self.config.stream_url)

        cycle = 0
        while self._running:
            try:
                cycle += 1

                # Get latest frame
                frame = self.frame_extractor.get_latest_frame()
                if frame is None:
                    time.sleep(0.5)
                    continue

                # Run OCR
                from PIL import Image
                img = Image.open(frame)
                ocr_result = self.ocr_pipeline.extract(img)

                # Update game state from OCR
                home_score = int(ocr_result.home_score.value) if ocr_result.home_score.value.isdigit() else game_state.home_score
                away_score = int(ocr_result.away_score.value) if ocr_result.away_score.value.isdigit() else game_state.away_score
                clock = self.ocr_pipeline.parse_clock_minutes(ocr_result.clock.value)

                score_changed = game_state.update_score(home_score, away_score)
                game_state.clock_minutes = clock
                game_state.ocr_reliable = self.ocr_pipeline.is_reliable(ocr_result)

                # Run event classifier (every 5 seconds)
                if cycle % 5 == 0:
                    event_result = self.event_classifier.classify(img)
                    game_state.event_label = event_result.label
                    game_state.event_confidence = event_result.confidence

                    # Critical event triggers re-evaluation
                    if self.event_classifier.is_critical(event_result):
                        logger.warning(
                            "Critical event detected: %s (%.1f%%)",
                            event_result.label,
                            event_result.confidence * 100,
                        )

                # Run player detector (every 10 seconds)
                if cycle % 10 == 0:
                    pressure_result = self.player_detector.detect(img)
                    game_state.home_pressure_score = pressure_result.pressure_score

                # Run trading signal engine
                result = self.signal_engine.run_cycle(game_state)

                if result.kill_switch_triggered:
                    logger.critical("Kill switch triggered, stopping...")
                    self._running = False
                    break

                # Check if match ended
                if game_state.clock_minutes >= 90 and not game_state.is_extra_time:
                    logger.info("Match ended (90+ minutes)")
                    break

                time.sleep(1)  # 1 second between cycles

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error("Main loop error: %s", e, exc_info=True)
                self.kill_switch.record_exception(e)
                if self.kill_switch.is_halted:
                    break

        self._shutdown()

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received")
        self._running = False

    def _shutdown(self):
        """Cleanup and save final state."""
        logger.info("Shutting down...")

        # Stop frame extraction
        if self.frame_extractor:
            self.frame_extractor.stop()

        # Cancel open orders
        if self.order_manager:
            self.order_manager.cancel_all_orders()

        # Log final outcome
        if self.trade_logger and self.config.match_id:
            gs = GameState(match_id=self.config.match_id)
            # Would need final score here

        logger.info("Shutdown complete")


def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("data/bot.log"),
        ],
    )

    config = load_config()

    if not config.stream_url:
        logger.error("STREAM_URL not set")
        sys.exit(1)

    bot = SoccerTradeBot(config)
    bot.initialize()
    bot.run()


if __name__ == "__main__":
    main()
