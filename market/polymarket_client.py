"""Polymarket CLOB API client.

Handles:
- REST: Market discovery via Gamma API
- WebSocket: Real-time orderbook updates
- Order placement via CLOB (EIP-712 signed)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
WS_BASE = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class Market:
    """Polymarket market representation."""

    market_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    active: bool


@dataclass
class OrderbookSnapshot:
    """Current orderbook state for a token."""

    token_id: str
    bids: List[Dict[str, float]]  # [{"price": 0.5, "size": 100}, ...]
    asks: List[Dict[str, float]]
    best_bid: float
    best_ask: float
    spread: float
    timestamp: float


class PolymarketClient:
    """Polymarket CLOB API client for market data and order placement.

    Args:
        api_key: Polymarket API key for CLOB authentication.
        private_key: Wallet private key for EIP-712 signing.
        dry_run: If True, log orders without placing them.
    """

    def __init__(
        self,
        api_key: str = "",
        private_key: str = "",
        dry_run: bool = True,
    ) -> None:
        self.api_key = api_key
        self.private_key = private_key
        self.dry_run = dry_run
        self._clob_client = None
        self._ws_connection = None
        self._orderbooks: Dict[str, OrderbookSnapshot] = {}
        self._callbacks: List[Callable] = []

    def initialize(self) -> None:
        """Initialize CLOB client with wallet credentials."""
        if not self.private_key:
            logger.warning("No Polymarket private key provided, read-only mode")
            return

        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                host=CLOB_API_BASE,
                chain_id=137,  # Polygon mainnet
                key=self.private_key,
            )

            # Use provided API key or derive from wallet
            if self.api_key:
                from py_clob_client.creds import ApiCreds
                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_key,
                    api_passphrase=self.api_key,
                )
                self._clob_client.set_api_creds(creds)
                logger.info("Polymarket CLOB client initialized with API key")
            else:
                creds = self._clob_client.create_or_derive_api_creds()
                self._clob_client.set_api_creds(creds)
                logger.info("Polymarket CLOB client initialized (derived creds)")

        except ImportError:
            logger.warning("py-clob-client not installed, read-only mode")
        except Exception as e:
            logger.error("Failed to initialize CLOB client: %s", e)

    def search_soccer_markets(
        self, team_home: str, team_away: str
    ) -> List[Market]:
        """Search for active soccer match winner markets.

        Args:
            team_home: Home team name.
            team_away: Away team name.

        Returns:
            List of matching markets.
        """
        markets = []

        try:
            # Search Gamma API
            params = {
                "closed": "false",
                "limit": 100,
                "tag": "soccer",
            }
            resp = requests.get(f"{GAMMA_API_BASE}/markets", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data:
                question = item.get("question", "").lower()
                # Check if both teams are mentioned
                if (
                    team_home.lower() in question
                    and team_away.lower() in question
                ):
                    outcomes = item.get("outcomes", [])
                    outcome_prices = item.get("outcomePrices", [])

                    if len(outcomes) >= 2 and len(outcome_prices) >= 2:
                        prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices

                        market = Market(
                            market_id=item.get("id", ""),
                            question=item.get("question", ""),
                            token_id_yes=item.get("clobTokenIds", ["", ""])[0],
                            token_id_no=item.get("clobTokenIds", ["", ""])[1] if len(item.get("clobTokenIds", [])) > 1 else "",
                            yes_price=float(prices[0]) if prices[0] else 0.0,
                            no_price=float(prices[1]) if len(prices) > 1 and prices[1] else 0.0,
                            volume=float(item.get("volume", 0)),
                            liquidity=float(item.get("liquidity", 0)),
                            active=item.get("active", False),
                        )
                        markets.append(market)

        except Exception as e:
            logger.error("Failed to search Polymarket: %s", e)

        return markets

    def get_orderbook(self, token_id: str) -> Optional[OrderbookSnapshot]:
        """Get current orderbook for a token.

        Args:
            token_id: Token ID to query.

        Returns:
            OrderbookSnapshot or None if unavailable.
        """
        try:
            if self._clob_client:
                book = self._clob_client.get_order_book(token_id)
                bids = [{"price": float(b.price), "size": float(b.size)} for b in book.bids[:10]]
                asks = [{"price": float(a.price), "size": float(a.size)} for a in book.asks[:10]]

                best_bid = bids[0]["price"] if bids else 0.0
                best_ask = asks[0]["price"] if asks else 1.0

                return OrderbookSnapshot(
                    token_id=token_id,
                    bids=bids,
                    asks=asks,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=best_ask - best_bid,
                    timestamp=time.time(),
                )
            else:
                # REST fallback
                resp = requests.get(
                    f"{CLOB_API_BASE}/order-book/{token_id}",
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()

                bids = [{"price": float(b["price"]), "size": float(b["size"])} for b in data.get("bids", [])[:10]]
                asks = [{"price": float(a["price"]), "size": float(a["size"])} for a in data.get("asks", [])[:10]]

                best_bid = bids[0]["price"] if bids else 0.0
                best_ask = asks[0]["price"] if asks else 1.0

                return OrderbookSnapshot(
                    token_id=token_id,
                    bids=bids,
                    asks=asks,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=best_ask - best_bid,
                    timestamp=time.time(),
                )

        except Exception as e:
            logger.error("Failed to get orderbook for %s: %s", token_id, e)
            return None

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Optional[str]:
        """Place a limit order.

        Args:
            token_id: Token to trade.
            side: 'BUY' or 'SELL'.
            price: Limit price (0.01-0.99).
            size: Number of contracts.

        Returns:
            Order ID if placed, None if failed or dry run.
        """
        if self.dry_run:
            logger.info(
                "[DRY RUN] Polymarket order: %s %s @ %.3f x %.0f",
                side, token_id[:8], price, size,
            )
            return f"dry_run_{int(time.time())}"

        if not self._clob_client:
            logger.error("CLOB client not initialized")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id,
            )

            resp = self._clob_client.create_order(order_args)
            order_id = resp.get("orderID", "")

            logger.info("Order placed: %s", order_id)
            return order_id

        except Exception as e:
            logger.error("Failed to place order: %s", e)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order to cancel.

        Returns:
            True if cancelled successfully.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Cancel order: %s", order_id)
            return True

        if not self._clob_client:
            return False

        try:
            self._clob_client.cancel(order_id)
            logger.info("Order cancelled: %s", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False
