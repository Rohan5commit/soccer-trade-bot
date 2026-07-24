"""Edge calculator for model vs market probability comparison.

Computes the difference between model-predicted probabilities
and market-implied probabilities for each outcome.

Supports:
- Bid/ask spread-aware edge calculation
- Skip outcomes with no market (e.g., draw when unavailable)
- Minimum edge threshold
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    """Edge calculation for a single outcome."""

    outcome: str  # 'home', 'draw', 'away'
    model_prob: float
    market_prob: float
    market_ask: float  # Price you'd buy at (execution price)
    market_bid: float  # Price you'd sell at
    edge: float  # model_prob - market_ask (realistic edge for buying)
    has_edge: bool  # edge > threshold
    has_market: bool  # Whether this outcome has a tradeable market


@dataclass
class EdgeAnalysis:
    """Complete edge analysis for all outcomes."""

    edges: Dict[str, EdgeResult]
    best_edge: Optional[EdgeResult]
    any_tradable: bool


class EdgeCalculator:
    """Calculates betting edge from model vs market probabilities.

    Args:
        edge_threshold: Minimum edge to consider tradable (default 5%).
        confidence_threshold: Minimum model probability to consider an outcome (default 0.70).
    """

    def __init__(self, edge_threshold: float = 0.05, confidence_threshold: float = 0.70) -> None:
        self.edge_threshold = edge_threshold
        self.confidence_threshold = confidence_threshold

    def calculate(
        self,
        model_probs: Dict[str, float],
        market_prices: Dict[str, float],
        market_bids: Optional[Dict[str, float]] = None,
        market_asks: Optional[Dict[str, float]] = None,
    ) -> EdgeAnalysis:
        """Calculate edge for all outcomes.

        Args:
            model_probs: Model predictions {'home': 0.65, 'draw': 0.20, 'away': 0.15}.
            market_prices: Market midpoint prices {'home': 0.60, 'away': 0.18}.
                           Draw can be None if no market exists.
            market_bids: Optional bid prices (what you can sell at).
            market_asks: Optional ask prices (what you can buy at).

        Returns:
            EdgeAnalysis with edge for each outcome.
        """
        edges: Dict[str, EdgeResult] = {}

        for outcome in ["home", "draw", "away"]:
            model_p = model_probs.get(outcome, 0.0)
            market_p = market_prices.get(outcome, 0.0)

            # Skip outcomes with no market (None or 0.0)
            if market_p is None or market_p <= 0.0:
                edges[outcome] = EdgeResult(
                    outcome=outcome,
                    model_prob=model_p,
                    market_prob=0.0,
                    market_ask=0.0,
                    market_bid=0.0,
                    edge=0.0,
                    has_edge=False,
                    has_market=False,
                )
                continue

            # Use ask price for edge calculation (price you'd actually buy at)
            if market_asks and outcome in market_asks and market_asks[outcome] > 0:
                execution_price = market_asks[outcome]
            else:
                execution_price = market_p  # Fallback to midpoint

            if market_bids and outcome in market_bids and market_bids[outcome] > 0:
                bid_price = market_bids[outcome]
            else:
                bid_price = market_p

            # Edge = model probability - execution price (what you'd pay)
            edge = model_p - execution_price
            has_edge = edge >= self.edge_threshold and model_p >= self.confidence_threshold

            edges[outcome] = EdgeResult(
                outcome=outcome,
                model_prob=model_p,
                market_prob=market_p,
                market_ask=execution_price,
                market_bid=bid_price,
                edge=edge,
                has_edge=has_edge,
                has_market=True,
            )

            if has_edge:
                logger.info(
                    "Edge found: %s model=%.3f market_ask=%.3f edge=+%.3f",
                    outcome, model_p, execution_price, edge,
                )

        # Find best edge (only from outcomes with markets)
        tradable = [e for e in edges.values() if e.has_edge and e.has_market]
        best = max(tradable, key=lambda e: e.edge) if tradable else None

        return EdgeAnalysis(
            edges=edges,
            best_edge=best,
            any_tradable=len(tradable) > 0,
        )

    def has_significant_edge(
        self,
        model_probs: Dict[str, float],
        market_prices: Dict[str, float],
        min_outcomes: int = 1,
    ) -> bool:
        """Quick check if any outcome has significant edge.

        Args:
            model_probs: Model predictions.
            market_prices: Market prices (None values skipped).
            min_outcomes: Minimum number of outcomes with edge required.

        Returns:
            True if at least min_outcomes have edge > threshold.
        """
        count = 0
        for outcome in ["home", "draw", "away"]:
            model_p = model_probs.get(outcome, 0.0)
            market_p = market_prices.get(outcome, 0.0)
            if market_p is None or market_p <= 0.0:
                continue
            if model_p - market_p >= self.edge_threshold and model_p >= self.confidence_threshold:
                count += 1
        return count >= min_outcomes
