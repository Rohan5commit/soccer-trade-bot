"""Kelly position sizing with confidence scaling.

Implements Kelly criterion with:
- Fractional Kelly (default 0.25)
- Dynamic scaling based on model confidence
- Single-bet mode (only best outcome) to prevent over-allocation
- Maximum bet cap (default 2% of bankroll)
- Maximum exposure per match cap
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BetSize:
    """Calculated bet size for a trade."""

    outcome: str
    edge: float
    market_prob: float
    kelly_fraction: float  # Full Kelly fraction
    fractional_kelly: float  # Scaled Kelly
    bet_usd: float  # Final bet size
    bankroll_pct: float  # As percentage of bankroll


class KellySizer:
    """Kelly position sizing with confidence-based scaling.

    Formula:
        f = edge / (1 - market_implied_prob)
        scaled_f = f * base_kelly * confidence_multiplier
        bet = scaled_f * bankroll
        cap at max_bet_pct of bankroll

    Args:
        base_kelly: Base fraction of full Kelly (default 0.25).
        max_bet_pct: Maximum bet as percentage of bankroll (default 0.02).
        min_bet_usd: Minimum bet size in USD (default 5.0).
        max_exposure_pct: Maximum total exposure per match (default 0.05).
        single_bet_mode: If True, only bet on the single best outcome.
    """

    def __init__(
        self,
        base_kelly: float = 0.25,
        max_bet_pct: float = 0.05,
        min_bet_usd: float = 5.0,
        max_exposure_pct: float = 0.05,
        single_bet_mode: bool = True,
    ) -> None:
        self.base_kelly = base_kelly
        self.max_bet_pct = max_bet_pct
        self.min_bet_usd = min_bet_usd
        self.max_exposure_pct = max_exposure_pct
        self.single_bet_mode = single_bet_mode

    def _confidence_multiplier(self, model_prob: float, edge: float) -> float:
        """Scale Kelly fraction based on model confidence.

        Args:
            model_prob: Model probability for the outcome.
            edge: Edge (model - market).

        Returns:
            Multiplier between 0.5 and 1.5.
        """
        # Higher confidence = closer to full Kelly
        # Lower confidence = more conservative
        confidence = model_prob + edge  # Combined signal strength

        if confidence > 0.8:
            return 1.3  # Very confident
        elif confidence > 0.7:
            return 1.1  # Confident
        elif confidence > 0.6:
            return 1.0  # Normal
        elif confidence > 0.5:
            return 0.8  # Somewhat uncertain
        else:
            return 0.5  # Very uncertain

    def calculate(
        self,
        outcome: str,
        edge: float,
        market_prob: float,
        bankroll: float,
        model_prob: float = 0.5,
    ) -> BetSize:
        """Calculate optimal bet size.

        Args:
            outcome: 'home', 'draw', or 'away'.
            edge: Model probability minus market probability.
            market_prob: Market implied probability.
            bankroll: Current bankroll in USD.
            model_prob: Model probability (for confidence scaling).

        Returns:
            BetSize with calculated position.
        """
        if edge <= 0 or market_prob <= 0 or market_prob >= 1:
            return BetSize(
                outcome=outcome,
                edge=edge,
                market_prob=market_prob,
                kelly_fraction=0.0,
                fractional_kelly=0.0,
                bet_usd=0.0,
                bankroll_pct=0.0,
            )

        # Full Kelly: f = edge / (1 - market_prob)
        full_kelly = edge / (1.0 - market_prob)

        # Apply confidence scaling
        conf_mult = self._confidence_multiplier(model_prob, edge)
        scaled_kelly = full_kelly * self.base_kelly * conf_mult

        # Calculate bet size
        bet_usd = scaled_kelly * bankroll

        # Apply cap
        max_bet = bankroll * self.max_bet_pct
        if bet_usd > max_bet:
            bet_usd = max_bet
            logger.debug(
                "Bet capped at $%.2f (%.1f%% of bankroll)",
                max_bet, self.max_bet_pct * 100,
            )

        # Apply minimum
        if bet_usd < self.min_bet_usd:
            bet_usd = 0.0

        bankroll_pct = (bet_usd / bankroll * 100) if bankroll > 0 else 0.0

        return BetSize(
            outcome=outcome,
            edge=edge,
            market_prob=market_prob,
            kelly_fraction=full_kelly,
            fractional_kelly=scaled_kelly,
            bet_usd=bet_usd,
            bankroll_pct=bankroll_pct,
        )

    def calculate_all(
        self,
        edges: Dict[str, float],
        market_probs: Dict[str, float],
        bankroll: float,
        model_probs: Optional[Dict[str, float]] = None,
    ) -> list[BetSize]:
        """Calculate bet sizes for outcomes with positive edge.

        In single_bet_mode (default), only the best outcome is bet on
        to prevent over-allocation across correlated outcomes.

        Args:
            edges: {'home': 0.08, 'draw': -0.02, 'away': -0.05}
            market_probs: {'home': 0.60, 'draw': 0.22, 'away': 0.18}
            bankroll: Current bankroll.
            model_probs: Optional model probabilities for confidence scaling.

        Returns:
            List of BetSize for outcomes with edge > 0.
        """
        if model_probs is None:
            model_probs = {k: 0.5 for k in edges}

        candidates = []

        for outcome in ["home", "draw", "away"]:
            edge = edges.get(outcome, 0.0)
            market_prob = market_probs.get(outcome, 0.0)
            model_prob = model_probs.get(outcome, 0.5)

            # Skip outcomes with no market (None or 0.0)
            if market_prob is None or market_prob <= 0.0:
                continue

            if edge > 0:
                bet = self.calculate(outcome, edge, market_prob, bankroll, model_prob)
                if bet.bet_usd > 0:
                    candidates.append(bet)

        if not candidates:
            return []

        # Sort by edge (best first)
        candidates.sort(key=lambda b: b.edge, reverse=True)

        if self.single_bet_mode:
            # Only bet on the single best outcome
            best = candidates[0]
            logger.info(
                "Single-bet mode: betting on %s (edge=%.3f, $%.2f)",
                best.outcome, best.edge, best.bet_usd,
            )
            return [best]

        # Multi-bet mode: cap total exposure
        total_exposure = 0.0
        max_exposure = bankroll * self.max_exposure_pct
        selected = []

        for bet in candidates:
            if total_exposure + bet.bet_usd <= max_exposure:
                selected.append(bet)
                total_exposure += bet.bet_usd
            else:
                # Reduce bet to fit within exposure cap
                remaining = max_exposure - total_exposure
                if remaining >= self.min_bet_usd:
                    adjusted = BetSize(
                        outcome=bet.outcome,
                        edge=bet.edge,
                        market_prob=bet.market_prob,
                        kelly_fraction=bet.kelly_fraction,
                        fractional_kelly=bet.fractional_kelly,
                        bet_usd=remaining,
                        bankroll_pct=(remaining / bankroll * 100) if bankroll > 0 else 0.0,
                    )
                    selected.append(adjusted)
                break

        return selected

    def update_bankroll(
        self,
        current_bankroll: float,
        pnl: float,
    ) -> float:
        """Update bankroll after a trade outcome.

        Args:
            current_bankroll: Previous bankroll.
            pnl: Profit/loss from trade.

        Returns:
            New bankroll amount.
        """
        new_bankroll = current_bankroll + pnl
        if new_bankroll < 0:
            logger.warning("Bankroll went negative: $%.2f", new_bankroll)
            return 0.0
        return new_bankroll
