"""Kalshi REST API client.

Handles:
- Market discovery for soccer match winner markets
- RSA-PSS signed request authentication
- Order placement and management
- Dual-mode: production API for prices (real liquidity), demo API for orders

Flow:
1. GET /events → find soccer game events
2. GET /markets?event_ticker=... → get markets inside that event
3. Read yes_ask_dollars → convert to cents for pricing
4. POST /portfolio/orders → place limit orders on demo
"""
from __future__ import annotations

import base64
import datetime
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, utils

logger = logging.getLogger(__name__)

# Production: real liquidity, real bid/ask spreads
KALSHI_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Demo: paper trading only
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"

# Known series tickers for soccer
SOCCER_SERIES = ["KXWCGAME", "KXMENWORLDCUP", "KXSOCCER", "KXMLBSOCCER", "KXMLS", "KXPREMIERLEAGUE"]


@dataclass
class KalshiMarket:
    """Kalshi market representation."""

    ticker: str
    title: str
    subtitle: str
    event_ticker: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int
    open_interest: int
    status: str
    expiration_time: str


@dataclass
class KalshiOrderbook:
    """Kalshi orderbook state."""

    ticker: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    spread: float
    timestamp: float


class KalshiClient:
    """Kalshi REST API client with RSA-PSS authentication.

    Dual-mode architecture:
    - Price fetching always hits production (real liquidity)
    - Order placement hits demo (paper trading)

    Args:
        api_key: Kalshi API key ID (e.g., 'dc990621...').
        private_key_pem: RSA private key in PEM format.
        dry_run: If True, log orders without placing them.
        use_demo: If True, place orders on demo (default True).
    """

    def __init__(
        self,
        api_key: str = "",
        private_key_pem: str = "",
        dry_run: bool = True,
        use_demo: bool = True,
    ) -> None:
        self.api_key = api_key
        self.private_key_pem = private_key_pem
        self.dry_run = dry_run
        self.use_demo = use_demo

        # Dual URLs: prices from prod, orders from demo
        self._price_url = KALSHI_PROD_BASE
        self._trade_url = KALSHI_DEMO_BASE if use_demo else KALSHI_PROD_BASE

        self._private_key = None
        self._session = requests.Session()
        self._market_cache: Dict[str, KalshiMarket] = {}

        if private_key_pem:
            try:
                self._private_key = serialization.load_pem_private_key(
                    private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
                    password=None,
                )
                logger.info("Kalshi RSA key loaded")
            except Exception as e:
                logger.error("Failed to load Kalshi private key: %s", e)

    def _sign_request(self, method: str, path: str, base_url: Optional[str] = None) -> Dict[str, str]:
        """Generate RSA-PSS signed headers.

        Signs: timestamp + method + full_path (including /trade-api/v2 prefix).
        Kalshi requires the full API path in the signature, not just the relative path.
        """
        if not self._private_key or not self.api_key:
            return {}

        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        # Build full URL path for signing: base_url + path
        full_url = f"{base_url or self._price_url}{path}"
        sign_path = urlparse(full_url).path  # e.g., /trade-api/v2/portfolio/balance
        message = f"{timestamp}{method}{sign_path}".encode()

        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        base_url: Optional[str] = None,
    ) -> Optional[Dict]:
        """Make authenticated request to Kalshi API.

        Args:
            method: HTTP method.
            path: API path.
            params: Query parameters.
            json_data: JSON body for POST/PUT.
            base_url: Override base URL (default: self._price_url).

        Returns:
            Response JSON or None on error.
        """
        url = f"{base_url or self._price_url}{path}"
        headers = self._sign_request(method, path, base_url=base_url)

        try:
            resp = self._session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            logger.error("Kalshi API error: %s - %s", e.response.status_code, e.response.text[:200])
            return None
        except Exception as e:
            logger.error("Kalshi request failed: %s", e)
            return None

    # ── Event-based market discovery ──────────────────────────────

    def get_game_events(self, sport: str = "soccer") -> List[Dict]:
        """Fetch all open game events for a sport.

        Uses Kalshi's dedicated series (e.g., KXSOCCER for soccer).
        Each event represents a match with multiple markets inside.

        Args:
            sport: Sport to search ('soccer').

        Returns:
            List of event dicts with event_ticker, title, etc.
        """
        events = []
        series_tickers = SOCCER_SERIES if sport == "soccer" else [f"KX{sport.upper()}"]

        for series in series_tickers:
            try:
                resp = self._request(
                    "GET",
                    "/events",
                    params={"series_ticker": series, "limit": 100, "status": "open"},
                )
                if resp and "events" in resp:
                    events.extend(resp["events"])
                    logger.info("Found %d events in series %s", len(resp["events"]), series)
            except Exception as e:
                logger.warning("Failed to fetch events for %s: %s", series, e)

        return events

    def get_event_markets(self, event_ticker: str) -> List[KalshiMarket]:
        """Fetch all markets inside an event (e.g., a specific match).

        Args:
            event_ticker: Event ticker (e.g., 'KXSOCCER-GAME-123').

        Returns:
            List of KalshiMarket objects.
        """
        markets = []

        try:
            resp = self._request(
                "GET",
                "/markets",
                params={"event_ticker": event_ticker, "limit": 100, "status": "open"},
            )
            if not resp or "markets" not in resp:
                return markets

            for item in resp["markets"]:
                market = self._parse_market(item)
                if market:
                    markets.append(market)
                    self._market_cache[market.ticker] = market

        except Exception as e:
            logger.error("Failed to fetch markets for %s: %s", event_ticker, e)

        return markets

    def search_soccer_markets(
        self, team_home: str, team_away: str
    ) -> List[KalshiMarket]:
        """Search for soccer match winner markets by team names.

        Flow: events → markets → filter by team names.

        Args:
            team_home: Home team name.
            team_away: Away team name.

        Returns:
            List of matching markets.
        """
        all_markets = []

        # Get all soccer events
        events = self.get_game_events("soccer")

        for event in events:
            event_ticker = event.get("event_ticker", "")
            event_title = event.get("title", "").lower()

            # Check if both teams are mentioned in event title
            if (
                team_home.lower() in event_title
                and team_away.lower() in event_title
            ):
                # Get markets inside this event
                markets = self.get_event_markets(event_ticker)
                all_markets.extend(markets)

        # Also search by direct market title/subtitle if event search found nothing
        if not all_markets:
            try:
                resp = self._request(
                    "GET",
                    "/markets",
                    params={"limit": 100, "status": "open", "series_ticker": "KXSOCCER"},
                )
                if resp and "markets" in resp:
                    for item in resp["markets"]:
                        title = item.get("title", "").lower()
                        subtitle = item.get("subtitle", "").lower()
                        if (
                            team_home.lower() in title
                            and team_away.lower() in title
                        ) or (
                            team_home.lower() in subtitle
                            and team_away.lower() in subtitle
                        ):
                            market = self._parse_market(item)
                            if market:
                                all_markets.append(market)
            except Exception as e:
                logger.error("Failed to search Kalshi markets: %s", e)

        return all_markets

    def _parse_market(self, item: Dict) -> Optional[KalshiMarket]:
        """Parse a market dict from Kalshi API into KalshiMarket.

        Handles both cents (yes_bid/yes_ask as int) and
        dollar format (yes_ask_dollars as string like "0.5600").
        """
        try:
            # Handle dollar format: "0.5600" → 0.56
            if "yes_ask_dollars" in item:
                yes_ask = float(item["yes_ask_dollars"])
                yes_bid = float(item.get("yes_bid_dollars", 1.0 - yes_ask))
            elif "yes_bid" in item:
                # Cents format: 56 → 0.56
                yes_bid = float(item.get("yes_bid", 0)) / 100
                yes_ask = float(item.get("yes_ask", 100)) / 100
            else:
                return None

            return KalshiMarket(
                ticker=item.get("ticker", ""),
                title=item.get("title", ""),
                subtitle=item.get("subtitle", ""),
                event_ticker=item.get("event_ticker", ""),
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=1.0 - yes_ask,
                no_ask=1.0 - yes_bid,
                volume=item.get("volume", 0),
                open_interest=item.get("open_interest", 0),
                status=item.get("status", ""),
                expiration_time=item.get("expiration_time", ""),
            )
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse market: %s", e)
            return None

    # ── Orderbook ─────────────────────────────────────────────────

    def get_orderbook(self, ticker: str) -> Optional[KalshiOrderbook]:
        """Get current orderbook for a market.

        Always fetches from production for real liquidity.

        Args:
            ticker: Market ticker.

        Returns:
            KalshiOrderbook or None.
        """
        try:
            resp = self._request(
                "GET", f"/markets/{ticker}/orderbook",
                base_url=self._price_url,
            )
            if not resp:
                return None

            # Try both API response formats
            orderbook = resp.get("orderbook_fp") or resp.get("orderbook", {})
            yes_book = orderbook.get("yes_dollars") or orderbook.get("yes", [])
            no_book = orderbook.get("no_dollars") or orderbook.get("no", [])

            if not yes_book and not no_book:
                return None

            # Best YES bid: highest price in yes_dollars (sorted ascending by price)
            yes_bid = float(yes_book[-1][0]) if yes_book else 0.0
            # Best YES ask: 1 - highest NO bid (buying NO = selling YES)
            no_bid = float(no_book[-1][0]) if no_book else 0.0
            yes_ask = 1.0 - no_bid if no_bid > 0 else 1.0

            return KalshiOrderbook(
                ticker=ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=1.0 - yes_ask,
                no_ask=1.0 - yes_bid,
                spread=yes_ask - yes_bid,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error("Failed to get Kalshi orderbook for %s: %s", ticker, e)
            return None

    def get_yes_price_cents(self, market: Dict) -> int:
        """Extract yes ask price in cents from market dict.

        Reads yes_ask_dollars (e.g., "0.5600") and converts to 56 cents.

        Args:
            market: Raw market dict from Kalshi API.

        Returns:
            Price in cents (1-99).
        """
        if "yes_ask_dollars" in market:
            return int(float(market["yes_ask_dollars"]) * 100)
        elif "yes_ask" in market:
            return int(market["yes_ask"])
        return 50  # fallback

    def get_implied_probability(self, market: Dict) -> float:
        """Get implied probability from market dict.

        Divides yes_ask_dollars by 100 → 0.56 = 56%.

        Args:
            market: Raw market dict from Kalshi API.

        Returns:
            Implied probability (0.0-1.0).
        """
        cents = self.get_yes_price_cents(market)
        return cents / 100.0

    # ── Order placement (always on demo) ─────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        yes_price,
        count: int,
        time_in_force: str = "immediate_or_cancel",
    ) -> Optional[str]:
        """Place a limit order on Kalshi demo.

        - side="bid" → buy YES (you think home team wins)
        - side="ask" → buy NO (you think away team wins)
        - yes_price: cents (int 1-99) or dollar string ("0.4300")
        - Endpoint: POST /portfolio/events/orders
        - Order type: immediate_or_cancel by default (fills or cancels)

        Args:
            ticker: Market ticker.
            side: 'bid' or 'ask'.
            yes_price: Price in cents (int) or dollar string (e.g., "0.4300").
            count: Number of contracts.
            time_in_force: 'immediate_or_cancel' (default) or 'good_till_canceled'.

        Returns:
            Order ID if placed, None otherwise.
        """
        if self.dry_run:
            logger.info(
                "[DRY RUN] Kalshi order: %s %s @ %s x %d",
                side, ticker, yes_price, count,
            )
            return f"dry_run_{int(time.time())}"

        # Normalize price to dollar string
        if isinstance(yes_price, str):
            price_str = yes_price  # Already "0.4300" format
        else:
            price_str = f"{yes_price / 100:.4f}"  # Cents → dollars

        # V2 endpoint: POST /portfolio/events/orders
        order_data = {
            "ticker": ticker,
            "side": side.lower(),  # "bid" or "ask"
            "count": f"{count:.2f}",  # Fixed-point string
            "price": price_str,
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
        }

        resp = self._request(
            "POST",
            "/portfolio/events/orders",
            json_data=order_data,
            base_url=self._trade_url,
        )
        if resp and "order_id" in resp:
            order_id = resp["order_id"]
            fill_count = resp.get("fill_count", "0.00")
            remaining = resp.get("remaining_count", f"{count:.2f}")
            logger.info(
                "Kalshi order placed: %s | filled=%s remaining=%s",
                order_id, fill_count, remaining,
            )
            return order_id

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order to cancel.

        Returns:
            True if cancelled successfully.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Cancel Kalshi order: %s", order_id)
            return True

        resp = self._request(
            "DELETE", f"/portfolio/events/orders/{order_id}",
            base_url=self._trade_url,
        )
        return resp is not None

    def get_balance(self) -> Optional[float]:
        """Get current demo account balance.

        Returns:
            Balance in dollars, or None on error.
        """
        resp = self._request(
            "GET", "/portfolio/balance",
            base_url=self._trade_url,
        )
        if resp:
            return float(resp.get("balance", 0)) / 100
        return None

    def get_positions(self) -> List[Dict]:
        """Get current open positions.

        Returns:
            List of position dicts.
        """
        resp = self._request(
            "GET", "/portfolio/positions",
            base_url=self._trade_url,
        )
        if resp:
            return resp.get("market_positions", [])
        return []
