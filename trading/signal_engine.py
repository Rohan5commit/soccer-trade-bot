"""Main trading signal engine.

Orchestrates the full trading loop:
1. Get latest GameState
2. Run kill switch checks
3. Predict win probabilities
4. Fetch live market prices
5. Calculate edge
6. Size positions (quarter-Kelly)
7. Place orders
8. Log everything to SQLite

Runs every 2 seconds during live matches.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from data.logger import TradeLogger
from market.market_selector import MarketSelector
from market.order_manager import OrderManager
from model.predict import WinPredictor
from trading.edge_calculator import EdgeCalculator, EdgeAnalysis
from trading.kill_switch import KillSwitch
from trading.kelly_sizer import KellySizer, BetSize
from vision.game_state import GameState

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Result of a single signal engine cycle."""

    timestamp: float
    game_state: GameState
    model_probs: Dict[str, float]
    market_prices: Dict[str, float]
    edge_analysis: Optional[EdgeAnalysis]
    bets_placed: list[BetSize]
    kill_switch_triggered: bool
    error: Optional[str] = None


class SignalEngine:
    """Main trading loop: state -> signal -> order.

    Args:
        predictor: Win probability predictor.
        market_selector: Market discovery and price fetching.
        order_manager: Order execution.
        edge_calculator: Edge calculation.
        kelly_sizer: Position sizing.
        kill_switch: Safety mechanism.
        logger: Trade logger.
        dry_run: If True, log without placing orders.
        bankroll: Starting bankroll.
    """

    def __init__(
        self,
        predictor: WinPredictor,
        market_selector: MarketSelector,
        order_manager: OrderManager,
        edge_calculator: EdgeCalculator,
        kelly_sizer: KellySizer,
        kill_switch: KillSwitch,
        trade_logger: TradeLogger,
        dry_run: bool = True,
        bankroll: float = 1000.0,
        final_minutes_skip: int = 5,
    ) -> None:
        self.predictor = predictor
        self.market_selector = market_selector
        self.order_manager = order_manager
        self.edge_calculator = edge_calculator
        self.kelly_sizer = kelly_sizer
        self.kill_switch = kill_switch
        self.trade_logger = trade_logger
        self.dry_run = dry_run
        self.bankroll = bankroll
        self._cycle_count = 0
        self._final_minutes_skip = final_minutes_skip

    def run_cycle(self, state: GameState) -> SignalResult:
        """Execute one trading cycle.

        Args:
            state: Latest GameState from vision pipeline.

        Returns:
            SignalResult with all actions taken.
        """
        self._cycle_count += 1
        start = time.time()

        result = SignalResult(
            timestamp=time.time(),
            game_state=state,
            model_probs={},
            market_prices={},
            edge_analysis=None,
            bets_placed=[],
            kill_switch_triggered=False,
        )

        try:
            # 1. Kill switch check
            if self.kill_switch.check_all(
                stream_lag=time.time() - state.timestamp,
                ocr_confidence=0.95 if state.ocr_reliable else 0.5,
                current_bankroll=self.bankroll,
            ):
                result.kill_switch_triggered = True
                logger.warning("Kill switch triggered, skipping cycle")
                return result

            # 2. Skip if OCR not reliable
            if not state.ocr_reliable:
                logger.debug("OCR not reliable, skipping")
                return result

            # 3. Dynamic clock cutoff — best edge in 75-85 window when markets lag
            clock = state.clock_minutes
            score_diff = abs(state.score_diff)
            skip_after = 90 - self._final_minutes_skip
            if clock > skip_after:
                logger.debug("Clock > %d min, skipping", skip_after)
                return result
            elif clock > (skip_after - 5) and score_diff == 0:
                # Draw game late: still have edge as market lags
                pass
            elif clock > 78 and score_diff >= 2:
                # Blowout: market should be efficient by now
                logger.debug("Blowout at minute %d, skipping", clock)
                return result

            # 4. Predict win probabilities
            p_home, p_draw, p_away = self.predictor.predict(state)
            model_probs = {"home": p_home, "draw": p_draw, "away": p_away}
            result.model_probs = model_probs

            # 5. Fetch live market prices
            market_prices = self.market_selector.get_current_prices()
            if not market_prices:
                logger.debug("No market prices available")
                return result
            result.market_prices = market_prices

            # 6. Calculate edge (with spread awareness)
            # Build bid/ask dicts from market_prices (None = no market)
            market_bids = {}
            market_asks = {}
            for outcome, price in market_prices.items():
                if price is not None and price > 0:
                    # Use midpoint as both bid/ask for now
                    # TODO: fetch actual orderbook depth
                    market_bids[outcome] = price
                    market_asks[outcome] = price

            edge_analysis = self.edge_calculator.calculate(
                model_probs, market_prices, market_bids, market_asks,
            )
            result.edge_analysis = edge_analysis

            # 7. Log signal
            signal_id = self.trade_logger.log_signal(
                match_id=state.match_id,
                clock_minutes=state.clock_minutes,
                home_score=state.home_score,
                away_score=state.away_score,
                ocr_reliable=state.ocr_reliable,
                model_probs=model_probs,
                market_prices=market_prices,
                edges={k: v.edge for k, v in edge_analysis.edges.items()},
                event_label=state.event_label,
                event_confidence=state.event_confidence,
                pressure_score=state.home_pressure_score,
            )

            # 8. Place bets if edge exists
            if edge_analysis.any_tradable and edge_analysis.best_edge:
                best = edge_analysis.best_edge
                bets = self.kelly_sizer.calculate_all(
                    edges={k: v.net_edge for k, v in edge_analysis.edges.items()},
                    market_probs=market_prices,
                    bankroll=self.bankroll,
                    model_probs=model_probs,
                )

                for bet in bets:
                    if bet.bet_usd > 0:
                        # Place order
                        platform = self.market_selector.platform
                        market_id = self.market_selector.selected.market_id if self.market_selector.selected else ""

                        order = self.order_manager.place_order(
                            platform=platform or "polymarket",
                            market_id=market_id,
                            outcome=bet.outcome,
                            side="buy",
                            price=bet.market_prob + bet.edge / 2,  # Limit between market and model
                            size_usd=bet.bet_usd,
                        )

                        if order:
                            self.trade_logger.log_trade(
                                match_id=state.match_id,
                                signal_id=signal_id,
                                platform=platform or "polymarket",
                                market_id=market_id,
                                outcome=bet.outcome,
                                side="buy",
                                price=order.price,
                                size_usd=bet.bet_usd,
                                edge=bet.edge,
                                kelly_fraction=bet.fractional_kelly,
                                order_id=order.order_id,
                                status=order.status,
                                dry_run=self.dry_run,
                            )
                            result.bets_placed.append(bet)

            elapsed = (time.time() - start) * 1000
            if self._cycle_count % 10 == 0:
                logger.info(
                    "Cycle %d: %s %d-%d %d' | p=%.2f/%.2f/%.2f | bets=%d | %.0fms",
                    self._cycle_count,
                    state.home_team[:12],
                    state.home_score,
                    state.away_score,
                    state.clock_minutes,
                    p_home, p_draw, p_away,
                    len(result.bets_placed),
                    elapsed,
                )

        except Exception as e:
            result.error = str(e)
            self.kill_switch.record_exception(e)
            logger.error("Signal engine error: %s", e, exc_info=True)

        return result

    def update_bankroll(self, pnl: float) -> None:
        """Update bankroll after trade outcome."""
        self.bankroll = self.kelly_sizer.update_bankroll(self.bankroll, pnl)
        self.order_manager.set_bankroll(self.bankroll)

    def emergency_stop(self) -> None:
        """Emergency stop: cancel all orders and halt."""
        logger.critical("EMERGENCY STOP triggered")
        self.order_manager.cancel_all_orders()
        self.kill_switch._halt("Emergency stop")

    @property
    def is_active(self) -> bool:
        """Whether the engine is active (not halted)."""
        return not self.kill_switch.is_halted
