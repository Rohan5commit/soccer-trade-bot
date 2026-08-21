"""Kalshi REST API client (production only).

Handles:
- Market discovery for soccer match winner markets
- RSA-PSS signed request authentication
- Orderbook depth for local fill simulation

Flow:
1. GET /events → find soccer game events
2. GET /markets?event_ticker=... → get markets inside that event
3. Read yes_ask_dollars → convert to cents for pricing
4. GET /markets/{ticker}/orderbook → fetch depth for shadow book
"""
from __future__ import annotations

import base64
import datetime
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

# Production API (public market data, requires RSA auth)
KALSHI_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Known series tickers for soccer
SOCCER_SERIES = [
    "KXALLSVENSKANGAME", "KXBRASILEIROBGAME", "KXBRASILEIROGAME",
    "KXSUPERLIGGAME", "KXEREDIVISIEGAME", "KXPRIMERALIGAME",
    "KXCHAMPIONSLEAGUEGAME", "KXPREMIERLEAGUE",
    "KXMLSGAME", "KXSERIEAGAME", "KXSERIEBGAME",
    "KXLIGAMXGAME", "KXSAUDIPLGAME", "KXSCOTTISHPREMGAME",
    "KXSLGREECEGAME", "KXSWISSLEAGUEGAME", "KXTHAIL1GAME",
    "KXUAEPLGAME", "KXUCLGAME", "KXUELGAME", "KXUECLGAME",
    "KXUEFAGAME", "KXUEFANLGAME", "KXUSLGAME", "KXUSOPENCUPGAME",
    "KXPERLIGA1GAME", "KXSPBGAME", "KXVENFUTVEGAME",
    "KXQSTARSGAME", "KXTACAPORTGAME", "KXDENSUPERLIGAGAME",
    "KXISLGAME", "KXCHNSLGAME", "KXKLEAGUEGAME",
    "KXCLUBFGAME", "KXASEANGAME", "KXWIBPLGAME",
    "KXSCOCUPGAME", "KXUSLCUPGAME", "KXARGNACBGAME",
    "KXWCGAME", "KXMENWORLDCUP",
]


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
    rules_primary: str = ""
    is_regulation_only: bool = False


@dataclass
class OrderbookLevel:
    """Single price level in the orderbook."""

    price: float
    count: int


@dataclass
class ShadowOrderbook:
    """Local copy of the orderbook that simulates fills.

    Decrements liquidity when orders are placed. Resyncs from
    production when the real book changes or after a timeout.
    """

    ticker: str
    yes_levels: List[OrderbookLevel] = field(default_factory=list)
    no_levels: List[OrderbookLevel] = field(default_factory=list)
    last_sync: float = 0.0

    def simulate_fill(self, count: int, price: float, side: str = "yes") -> int:
        """Simulate filling `count` contracts at `price` or better.

        Walks the orderbook from best price down. Returns number of contracts
        actually fillable (may be less than `count` if book is thin).
        """
        levels = self.yes_levels if side == "yes" else self.no_levels
        remaining = count
        filled = 0

        for lvl in levels:
            if remaining <= 0:
                break
            if side == "yes" and lvl.price > price:
                break
            if side == "no" and lvl.price > price:
                break
            take = min(remaining, lvl.count)
            filled += take
            remaining -= take
            lvl.count -= take

        return filled

    def best_ask(self, side: str = "yes") -> float:
        """Best ask price from the shadow book."""
        levels = self.yes_levels if side == "yes" else self.no_levels
        if not levels:
            return 0.0
        return levels[0].price if levels else 0.0

    def total_depth(self, side: str = "yes", max_levels: int = 5) -> int:
        """Total contracts available across top N levels."""
        levels = self.yes_levels if side == "yes" else self.no_levels
        return sum(l.count for l in levels[:max_levels])

    def needs_resync(self, timeout: float = 30.0) -> bool:
        """Check if book is stale and needs resync from production."""
        return (time.time() - self.last_sync) > timeout


class KalshiClient:
    """Kalshi REST API client — production only.

    Reads market data, prices, and orderbook depth from production.
    Fill simulation and PnL resolution happen locally (shadow book).

    Args:
        api_key: Kalshi API key ID (e.g., 'dc990621...').
        private_key_pem: RSA private key in PEM format.
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

        # Production URL for all reads
        self._price_url = KALSHI_PROD_BASE

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

    # Transient HTTP status codes that warrant retry
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        base_url: Optional[str] = None,
    ) -> Optional[Dict]:
        """Make authenticated request to Kalshi API with automatic retry.

        Retries on transient errors: 429 (rate limit), 500/502/503/504 (server),
        connection errors, and timeouts. Uses exponential backoff.

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
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=15,
                )

                # Success
                if resp.status_code == 200:
                    return resp.json()

                # Rate limit: respect Retry-After header
                if resp.status_code == 429:
                    try:
                        retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                    except (ValueError, TypeError):
                        retry_after = 60
                    logger.warning(
                        "Kalshi 429 rate limited, retry %d/%d in %ds",
                        attempt + 1, max_attempts, retry_after,
                    )
                    time.sleep(retry_after)
                    headers = self._sign_request(method, path, base_url=base_url)
                    continue

                # Transient server errors: retry with backoff
                if resp.status_code in self._RETRYABLE_STATUS:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(
                        "Kalshi %d error, retry %d/%d in %ds",
                        resp.status_code, attempt + 1, max_attempts, backoff,
                    )
                    time.sleep(backoff)
                    headers = self._sign_request(method, path, base_url=base_url)
                    continue

                # Non-retryable error (400, 401, 403, 404, etc.)
                logger.error(
                    "Kalshi API error: %s - %s",
                    resp.status_code, resp.text[:200],
                )
                return None

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                backoff = 2 ** (attempt + 1)
                logger.warning(
                    "Kalshi connection error (%s), retry %d/%d in %ds",
                    type(e).__name__, attempt + 1, max_attempts, backoff,
                )
                time.sleep(backoff)
                headers = self._sign_request(method, path, base_url=base_url)
                continue

            except Exception as e:
                logger.error("Kalshi request failed: %s", e)
                return None

        logger.error("Kalshi request failed after %d attempts: %s %s", max_attempts, method, path)
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
        Searches across all known soccer series.

        Args:
            team_home: Home team name.
            team_away: Away team name.

        Returns:
            List of matching markets.
        """
        all_markets = []

        # Get all soccer events across multiple series
        events = []
        for series in SOCCER_SERIES:
            try:
                resp = self._request(
                    "GET",
                    "/events",
                    params={"series_ticker": series, "limit": 50, "status": "open"},
                )
                if resp and "events" in resp:
                    events.extend(resp["events"])
            except Exception:
                pass

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

        return all_markets

    def _parse_market(self, item: Dict) -> Optional[KalshiMarket]:
        """Parse a market dict from Kalshi API into KalshiMarket.

        Handles both cents (yes_bid/yes_ask as int) and
        dollar format (yes_ask_dollars as string like "0.5600").
        Returns None if price data is missing or invalid.
        """
        try:
            ticker = item.get("ticker", "")

            # Handle dollar format: "0.5600" → 0.56
            if "yes_ask_dollars" in item and "yes_bid_dollars" in item:
                yes_ask = float(item["yes_ask_dollars"])
                yes_bid = float(item["yes_bid_dollars"])
            elif "yes_ask_dollars" in item:
                yes_ask = float(item["yes_ask_dollars"])
                yes_bid = yes_ask  # fallback: assume zero spread
            elif "yes_bid" in item:
                # Cents format: 56 → 0.56
                yes_bid = float(item.get("yes_bid", 0)) / 100
                yes_ask = float(item.get("yes_ask", 100)) / 100
            else:
                logger.debug("Market %s has no price data — skipping", ticker)
                return None

            # Reject degenerate markets (prices out of 0-1 range)
            if not (0.0 <= yes_bid <= 1.0 and 0.0 <= yes_ask <= 1.0):
                logger.warning("Market %s has invalid prices (bid=%.3f ask=%.3f) — skipping",
                               ticker, yes_bid, yes_ask)
                return None

            return KalshiMarket(
                ticker=ticker,
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
                rules_primary=item.get("rules_primary", ""),
                is_regulation_only="regulation time" in item.get("rules_primary", "").lower()
                    or "90 minutes" in item.get("rules_primary", "").lower(),
            )
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse market: %s", e)
            return None

    # ── Orderbook depth ────────────────────────────────────────

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

    @staticmethod
    def _calc_fee(price: float, count: int) -> float:
        """Calculate Kalshi taker fee for a given fill.

        fee_per_contract = ceil(0.07 * price * (1 - price) * 100) / 100
        Total fee = fee_per_contract * count

        Args:
            price: Fill price (0.0 - 1.0).
            count: Number of contracts.

        Returns:
            Total fee in dollars.
        """
        fee_per = math.ceil(0.07 * price * (1 - price) * 100) / 100
        return fee_per * count

    def get_orderbook_depth(self, ticker: str, depth: int = 20) -> Optional[ShadowOrderbook]:
        """Fetch full orderbook with depth for a market.

        Always reads from production for real liquidity.
        Returns a ShadowOrderbook that can simulate fills locally.

        Args:
            ticker: Market ticker.
            depth: Number of price levels to fetch.

        Returns:
            ShadowOrderbook or None.
        """
        try:
            resp = self._request(
                "GET", f"/markets/{ticker}/orderbook",
                params={"depth": depth},
                base_url=self._price_url,
            )
            if not resp:
                return None

            orderbook = resp.get("orderbook_fp") or resp.get("orderbook", {})
            yes_book = orderbook.get("yes_dollars") or orderbook.get("yes", [])
            no_book = orderbook.get("no_dollars") or orderbook.get("no", [])

            yes_levels = []
            no_levels = []

            for price_str, count_str in yes_book:
                try:
                    yes_levels.append(OrderbookLevel(
                        price=float(price_str),
                        count=int(float(count_str)),
                    ))
                except (ValueError, TypeError):
                    continue

            for price_str, count_str in no_book:
                try:
                    no_levels.append(OrderbookLevel(
                        price=float(price_str),
                        count=int(float(count_str)),
                    ))
                except (ValueError, TypeError):
                    continue

            # Sort ascending by price (best ask first)
            yes_levels.sort(key=lambda x: x.price)
            no_levels.sort(key=lambda x: x.price)

            return ShadowOrderbook(
                ticker=ticker,
                yes_levels=yes_levels,
                no_levels=no_levels,
                last_sync=time.time(),
            )

        except Exception as e:
            logger.error("Failed to get orderbook depth for %s: %s", ticker, e)
            return None

    # ── Legacy stubs (kept for backward compatibility) ──────────

    def get_orderbook(self, ticker: str) -> Optional[Dict]:
        """Deprecated: use get_orderbook_depth() instead."""
        logger.warning("get_orderbook() is deprecated — use get_orderbook_depth()")
        return None

    def place_order(self, ticker: str, side: str, yes_price, count: int, **kwargs) -> Optional[str]:
        """Deprecated: fill simulation uses shadow orderbook now."""
        logger.warning("place_order() is deprecated — use shadow orderbook for fills")
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Deprecated: no real orders are placed."""
        logger.warning("cancel_order() is deprecated — no real orders")
        return False

    def get_balance(self) -> Optional[float]:
        """Deprecated: bankroll comes from persistent state."""
        logger.warning("get_balance() is deprecated — use data/state/bankroll.json")
        return None

    def get_positions(self) -> List[Dict]:
        """Deprecated: positions tracked locally."""
        logger.warning("get_positions() is deprecated")
        return []
