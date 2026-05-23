"""
Kraken exchange connector implementation.

Implements both REST API and WebSocket connections for Kraken cryptocurrency exchange.

API Documentation: https://docs.kraken.com/api/
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from ...common.exceptions import ConnectionError as ConnectorConnectionError
from ...common.logging import get_logger
from ...common.rate_limiter import RateLimiter, RetryableRequest
from ..base import (
    OHLCV,
    Balance,
    ExchangeConnector,
    Order,
    OrderBook,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
    Trade,
)

logger = get_logger(__name__)


class KrakenConnector(ExchangeConnector):
    """Kraken exchange connector with REST and WebSocket support."""

    REST_URL = "https://api.kraken.com"
    WS_URL_PUBLIC = "wss://ws.kraken.com"
    WS_URL_PRIVATE = "wss://ws-auth.kraken.com"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialize Kraken connector.

        Args:
            api_key: Kraken API key
            api_secret: Kraken API secret
            testnet: Not supported by Kraken (always False)
        """
        super().__init__(api_key, api_secret, testnet)
        self._session: aiohttp.ClientSession | None = None
        self._ws_public: ClientConnection | None = None
        self._ws_private: ClientConnection | None = None
        self._ws_subscriptions: set[str] = set()
        self._ws_reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 10
        self._reconnect_delay: int = 5  # seconds
        self._ws_dispatcher_task: asyncio.Task | None = None
        self._ws_queues: dict[str, asyncio.Queue] = {}  # channel -> queue mapping
        self._ws_running: bool = False

        # Rate limiting
        self._rate_limiter = RateLimiter(max_requests_per_minute=15)
        self._retryable_request = RetryableRequest(self._rate_limiter, max_retries=3)

    async def connect(self) -> None:
        """Establish HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def disconnect(self) -> None:
        """Close all connections."""
        # Stop the WebSocket dispatcher first
        await self._stop_dispatcher()

        if self._session:
            await self._session.close()
            self._session = None

        if self._ws_public:
            await self._ws_public.close()
            self._ws_public = None

        if self._ws_private:
            await self._ws_private.close()
            self._ws_private = None

    def _generate_signature(self, urlpath: str, data: dict[str, Any], nonce: str) -> str:
        """
        Generate authentication signature for private API calls.

        Args:
            urlpath: API endpoint path
            data: Request data
            nonce: Unique nonce for this request

        Returns:
            Base64-encoded signature
        """
        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    async def _public_request(self, endpoint: str, params: dict | None = None) -> dict:
        """
        Make a public API request.

        Args:
            endpoint: API endpoint (e.g., "/0/public/Ticker")
            params: Query parameters

        Returns:
            Response JSON
        """
        if self._session is None:
            await self.connect()

        assert self._session is not None, "Session not initialized"

        async def _make_request():
            url = f"{self.REST_URL}{endpoint}"
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                if data.get("error"):
                    raise Exception(f"Kraken API error: {data['error']}")
                return data.get("result", {})

        return await self._retryable_request.execute(_make_request)

    async def _private_request(self, endpoint: str, data: dict | None = None) -> dict:
        """
        Make a private (authenticated) API request.

        Args:
            endpoint: API endpoint
            data: POST data

        Returns:
            Response JSON
        """
        if self._session is None:
            await self.connect()

        assert self._session is not None, "Session not initialized"
        if data is None:
            data = {}

        async def _make_request():
            nonce = str(int(time.time() * 1000))
            request_data = data.copy()
            request_data["nonce"] = nonce

            urlpath = endpoint
            signature = self._generate_signature(urlpath, request_data, nonce)

            headers = {
                "API-Key": self.api_key,
                "API-Sign": signature,
            }

            url = f"{self.REST_URL}{endpoint}"
            async with self._session.post(url, data=request_data, headers=headers) as response:
                result = await response.json()
                if result.get("error"):
                    raise Exception(f"Kraken API error: {result['error']}")
                return result.get("result", {})

        return await self._retryable_request.execute(_make_request)

    # Market Data Implementation

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker for a symbol."""
        kraken_symbol = self.normalize_symbol(symbol)
        result = await self._public_request("/0/public/Ticker", {"pair": kraken_symbol})

        # Kraken returns data keyed by the pair name
        pair_data = result[list(result.keys())[0]]

        return Ticker(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=float(pair_data["b"][0]),
            ask=float(pair_data["a"][0]),
            last=float(pair_data["c"][0]),
            volume_24h=float(pair_data["v"][1]),
            high_24h=float(pair_data["h"][1]),
            low_24h=float(pair_data["l"][1]),
            exchange="kraken",
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book for a symbol."""
        kraken_symbol = self.normalize_symbol(symbol)
        result = await self._public_request(
            "/0/public/Depth", {"pair": kraken_symbol, "count": depth}
        )

        pair_data = result[list(result.keys())[0]]

        bids = [(float(price), float(volume)) for price, volume, _ in pair_data["bids"]]
        asks = [(float(price), float(volume)) for price, volume, _ in pair_data["asks"]]

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            exchange="kraken",
        )

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCV]:
        """Get historical OHLCV data."""
        kraken_symbol = self.normalize_symbol(symbol)

        # Kraken interval mapping
        interval_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }

        params = {
            "pair": kraken_symbol,
            "interval": interval_map.get(interval, 1),
        }

        if start_time:
            params["since"] = int(start_time.timestamp())

        result = await self._public_request("/0/public/OHLC", params)
        pair_data = result[list(result.keys())[0]]

        ohlcv_list = []
        for candle in pair_data[:limit]:
            ohlcv_list.append(
                OHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(candle[0])),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[6]),
                    exchange="kraken",
                    interval=interval,
                )
            )

        return ohlcv_list

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol."""
        kraken_symbol = self.normalize_symbol(symbol)
        result = await self._public_request("/0/public/Trades", {"pair": kraken_symbol})

        pair_data = result[list(result.keys())[0]]

        trades = []
        for trade_data in pair_data[:limit]:
            trades.append(
                Trade(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(float(trade_data[2])),
                    price=float(trade_data[0]),
                    quantity=float(trade_data[1]),
                    side=OrderSide.BUY if trade_data[3] == "b" else OrderSide.SELL,
                    trade_id=str(trade_data[2]),
                    exchange="kraken",
                )
            )

        return trades

    # WebSocket Connection Management

    async def _connect_websocket(self, url: str) -> "ClientConnection":
        """
        Establish WebSocket connection with auto-reconnect.

        Args:
            url: WebSocket URL to connect to

        Returns:
            WebSocket connection
        """
        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                logger.info(f"Connecting to WebSocket: {url}")
                ws = await websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                )
                logger.info("WebSocket connected successfully")
                self._ws_reconnect_attempts = 0
                return ws

            except Exception as e:
                self._ws_reconnect_attempts += 1
                logger.error(
                    f"WebSocket connection failed (attempt {self._ws_reconnect_attempts}): {e}"
                )

                if self._ws_reconnect_attempts < self._max_reconnect_attempts:
                    await asyncio.sleep(self._reconnect_delay)
                else:
                    raise ConnectorConnectionError(
                        f"Failed to connect to WebSocket after {self._max_reconnect_attempts} attempts"
                    ) from e

        raise ConnectorConnectionError("WebSocket connection failed")

    async def _ensure_ws_connected(self) -> None:
        """Ensure public WebSocket is connected."""
        # Check if connection is None or closed (websockets 15.x uses close_code)
        needs_connection = (
            self._ws_public is None or getattr(self._ws_public, "close_code", None) is not None
        )
        if needs_connection:
            self._ws_public = await self._connect_websocket(self.WS_URL_PUBLIC)

    async def _ws_message_dispatcher(self) -> None:
        """
        Dispatch WebSocket messages to appropriate channel queues.
        This runs as a background task and routes incoming messages.
        """
        while self._ws_running:
            try:
                assert self._ws_public is not None, "WebSocket not connected"
                message = await self._ws_public.recv()
                data = json.loads(message)

                # Handle subscription status messages (dict format)
                if isinstance(data, dict) and data.get("event") == "subscriptionStatus":
                    # Route subscription status to the relevant channel queue
                    channel_name = data.get("subscription", {}).get("name")
                    if channel_name and channel_name in self._ws_queues:
                        try:
                            await self._ws_queues[channel_name].put(data)
                        except asyncio.QueueFull:
                            logger.warning(
                                f"Queue full for channel {channel_name}, dropping message"
                            )
                    continue

                # Skip other non-list messages (like heartbeats)
                if not isinstance(data, list) or len(data) < 3:
                    continue

                # Message format: [channelID, data, "channel_name", pair]
                channel_name = data[2] if len(data) >= 3 else None

                if channel_name:
                    # Handle channel names with depth suffix (e.g., "book-10" -> "book")
                    base_channel = (
                        channel_name.split("-")[0] if "-" in channel_name else channel_name
                    )

                    if base_channel in self._ws_queues:
                        try:
                            # Put message in the appropriate queue
                            await self._ws_queues[base_channel].put(data)
                        except asyncio.QueueFull:
                            logger.warning(
                                f"Queue full for channel {base_channel}, dropping message"
                            )

            except ConnectionClosed:
                logger.warning("WebSocket connection closed in dispatcher")
                if self._ws_running:
                    await self._ensure_ws_connected()
            except Exception as e:
                logger.error(f"Error in WebSocket dispatcher: {e}")
                await asyncio.sleep(1)

    def _get_or_create_queue(self, channel: str) -> asyncio.Queue:
        """Get or create a queue for a specific channel."""
        if channel not in self._ws_queues:
            self._ws_queues[channel] = asyncio.Queue(maxsize=1000)
        return self._ws_queues[channel]

    def _start_dispatcher(self) -> None:
        """Start the WebSocket message dispatcher if not already running."""
        if not self._ws_running:
            self._ws_running = True
            self._ws_dispatcher_task = asyncio.create_task(self._ws_message_dispatcher())

    async def _stop_dispatcher(self) -> None:
        """Stop the WebSocket message dispatcher."""
        self._ws_running = False
        if self._ws_dispatcher_task:
            self._ws_dispatcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._ws_dispatcher_task
            self._ws_dispatcher_task = None

    async def _subscribe_channel(
        self,
        channel: str,
        symbols: list[str],
    ) -> None:
        """
        Subscribe to a WebSocket channel.

        Args:
            channel: Channel name (ticker, book, trade, ohlc)
            symbols: List of symbols to subscribe to
        """
        await self._ensure_ws_connected()

        # Start dispatcher if not already running
        self._start_dispatcher()

        # Create queue for this channel
        self._get_or_create_queue(channel)

        # Convert symbols to Kraken WebSocket format (with slash separator)
        # WebSocket API expects "XBT/USD" not "XBTUSD"
        kraken_symbols = [self.normalize_symbol_ws(s) for s in symbols]

        subscription_msg = {
            "event": "subscribe",
            "pair": kraken_symbols,
            "subscription": {"name": channel},
        }

        logger.info(f"Subscribing to {channel} for {symbols}")
        assert self._ws_public is not None, "WebSocket not connected"
        await self._ws_public.send(json.dumps(subscription_msg))

        # Wait a bit for subscription to complete
        await asyncio.sleep(0.5)
        logger.info(f"Successfully subscribed to {channel}")

    async def _unsubscribe_channel(
        self,
        channel: str,
        symbols: list[str],
    ) -> None:
        """
        Unsubscribe from a WebSocket channel.

        Args:
            channel: Channel name
            symbols: List of symbols to unsubscribe from
        """
        # Check if connection is closed (websockets 15.x uses close_code)
        if self._ws_public is None or getattr(self._ws_public, "close_code", None) is not None:
            logger.debug("Skipping unsubscribe - connection closed or None")
            return

        kraken_symbols = [self.normalize_symbol_ws(s) for s in symbols]

        unsubscribe_msg = {
            "event": "unsubscribe",
            "pair": kraken_symbols,
            "subscription": {"name": channel},
        }

        try:
            await self._ws_public.send(json.dumps(unsubscribe_msg))
            logger.info(f"Unsubscribed from {channel} for {symbols}")
        except Exception as e:
            logger.warning(f"Failed to send unsubscribe: {e}")

    # WebSocket Streaming Implementation

    async def stream_ticker(self, symbols: list[str]) -> AsyncIterator[Ticker]:
        """
        Stream real-time ticker updates via WebSocket.

        Yields ticker updates as they arrive from Kraken's WebSocket feed.

        Args:
            symbols: List of symbols to subscribe to (e.g., ["BTC/USD", "ETH/USD"])

        Yields:
            Ticker objects with real-time market data
        """
        await self._subscribe_channel("ticker", symbols)

        # Create symbol mapping for denormalization
        symbol_map = {self.normalize_symbol_ws(s): s for s in symbols}

        # Get the queue for ticker messages
        queue = self._get_or_create_queue("ticker")

        try:
            while True:
                try:
                    # Read from queue instead of WebSocket
                    data = await queue.get()

                    # Check for subscription error messages (dict format)
                    if isinstance(data, dict) and data.get("event") == "subscriptionStatus":
                        if data.get("status") == "error":
                            error_msg = data.get("errorMessage", "Unknown error")
                            raise Exception(f"Subscription failed: {error_msg}")
                        # Skip subscription confirmation messages
                        continue

                    # Ticker format: [channelID, data, "ticker", pair]
                    if len(data) >= 4 and data[2] == "ticker":
                        ticker_data = data[1]
                        kraken_pair: str = data[3]
                        standard_symbol = symbol_map.get(kraken_pair, kraken_pair)

                        ticker = Ticker(
                            symbol=standard_symbol,
                            timestamp=datetime.now(),
                            bid=float(ticker_data["b"][0]),
                            ask=float(ticker_data["a"][0]),
                            last=float(ticker_data["c"][0]),
                            volume_24h=float(ticker_data["v"][1]),
                            high_24h=float(ticker_data["h"][1]),
                            low_24h=float(ticker_data["l"][1]),
                            exchange="kraken",
                        )

                        yield ticker

                except Exception as e:
                    logger.error(f"Error processing ticker message: {e}")
                    # Re-raise subscription errors
                    if "Subscription failed" in str(e):
                        raise
                    await asyncio.sleep(1)

        finally:
            await self._unsubscribe_channel("ticker", symbols)

    async def stream_orderbook(
        self, symbols: list[str], depth: int = 10
    ) -> AsyncIterator[OrderBook]:
        """
        Stream real-time order book updates via WebSocket.

        Args:
            symbols: List of symbols to subscribe to
            depth: Order book depth (10, 25, 100, 500, 1000)

        Yields:
            OrderBook objects with updated bid/ask levels
        """
        # Kraken WebSocket book subscription with depth
        await self._ensure_ws_connected()

        # Start dispatcher if not already running
        self._start_dispatcher()

        kraken_symbols = [self.normalize_symbol_ws(s) for s in symbols]
        symbol_map = {self.normalize_symbol_ws(s): s for s in symbols}

        # Create queue for book messages (use "book" as channel name prefix)
        queue = self._get_or_create_queue("book")

        subscription_msg = {
            "event": "subscribe",
            "pair": kraken_symbols,
            "subscription": {"name": "book", "depth": depth},
        }

        assert self._ws_public is not None, "WebSocket not connected"
        await self._ws_public.send(json.dumps(subscription_msg))
        logger.info(f"Subscribing to order book for {symbols} with depth {depth}")

        # Wait for subscription
        await asyncio.sleep(0.5)

        # Store current order book state
        orderbooks: dict[str, dict] = {}

        try:
            while True:
                try:
                    # Read from queue instead of WebSocket
                    data = await queue.get()

                    if not isinstance(data, list):
                        continue

                    # Book format: [channelID, data, "book-depth", pair]
                    if len(data) >= 4 and data[2].startswith("book"):
                        book_data = data[1]
                        kraken_pair: str = data[3]
                        standard_symbol = symbol_map.get(kraken_pair, kraken_pair)

                        # Initialize orderbook if not exists
                        if standard_symbol not in orderbooks:
                            orderbooks[standard_symbol] = {"bids": {}, "asks": {}}

                        # Update orderbook with snapshot or delta
                        if "as" in book_data and "bs" in book_data:
                            # Snapshot
                            orderbooks[standard_symbol]["asks"] = {
                                float(price): float(volume) for price, volume, *_ in book_data["as"]
                            }
                            orderbooks[standard_symbol]["bids"] = {
                                float(price): float(volume) for price, volume, *_ in book_data["bs"]
                            }
                        else:
                            # Delta update
                            if "a" in book_data:
                                for price, volume, *_ in book_data["a"]:
                                    price_f = float(price)
                                    volume_f = float(volume)
                                    if volume_f == 0:
                                        orderbooks[standard_symbol]["asks"].pop(price_f, None)
                                    else:
                                        orderbooks[standard_symbol]["asks"][price_f] = volume_f

                            if "b" in book_data:
                                for price, volume, *_ in book_data["b"]:
                                    price_f = float(price)
                                    volume_f = float(volume)
                                    if volume_f == 0:
                                        orderbooks[standard_symbol]["bids"].pop(price_f, None)
                                    else:
                                        orderbooks[standard_symbol]["bids"][price_f] = volume_f

                        # Convert to sorted lists
                        bids = sorted(
                            orderbooks[standard_symbol]["bids"].items(),
                            key=lambda x: x[0],
                            reverse=True,
                        )[:depth]

                        asks = sorted(
                            orderbooks[standard_symbol]["asks"].items(), key=lambda x: x[0]
                        )[:depth]

                        orderbook = OrderBook(
                            symbol=standard_symbol,
                            timestamp=datetime.now(),
                            bids=bids,
                            asks=asks,
                            exchange="kraken",
                        )

                        yield orderbook

                except ConnectionClosed:
                    logger.warning("WebSocket connection closed, reconnecting...")
                    await self._ensure_ws_connected()
                    # Re-subscribe
                    assert self._ws_public is not None, "WebSocket not connected"
                    await self._ws_public.send(json.dumps(subscription_msg))

                except Exception as e:
                    logger.error(f"Error processing orderbook message: {e}")
                    await asyncio.sleep(1)

        finally:
            await self._unsubscribe_channel("book", symbols)

    async def stream_trades(self, symbols: list[str]) -> AsyncIterator[Trade]:
        """
        Stream real-time trades via WebSocket.

        Args:
            symbols: List of symbols to subscribe to

        Yields:
            Trade objects as they occur
        """
        await self._subscribe_channel("trade", symbols)

        symbol_map = {self.normalize_symbol_ws(s): s for s in symbols}

        # Get the queue for trade messages
        queue = self._get_or_create_queue("trade")

        try:
            while True:
                try:
                    # Read from queue instead of WebSocket
                    data = await queue.get()

                    # Trade format: [channelID, [trade_data, ...], "trade", pair]
                    if len(data) >= 4 and data[2] == "trade":
                        trades_data = data[1]
                        kraken_pair: str = data[3]
                        standard_symbol = symbol_map.get(kraken_pair, kraken_pair)

                        # Process each trade in the message
                        for trade_data in trades_data:
                            # Format: [price, volume, time, side, orderType, misc]
                            trade = Trade(
                                symbol=standard_symbol,
                                timestamp=datetime.fromtimestamp(float(trade_data[2])),
                                price=float(trade_data[0]),
                                quantity=float(trade_data[1]),
                                side=OrderSide.BUY if trade_data[3] == "b" else OrderSide.SELL,
                                trade_id=f"{trade_data[2]}_{trade_data[0]}",
                                exchange="kraken",
                            )

                            yield trade

                except Exception as e:
                    logger.error(f"Error processing trade message: {e}")
                    await asyncio.sleep(1)

        finally:
            await self._unsubscribe_channel("trade", symbols)

    # Trading Methods (Private API)

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        client_order_id: str | None = None,
        **kwargs: Any,
    ) -> Order:
        """Place a new order."""
        kraken_symbol = self.normalize_symbol(symbol)

        order_data = {
            "pair": kraken_symbol,
            "type": side.value,
            "ordertype": order_type.value,
            "volume": str(quantity),
        }

        if price is not None:
            order_data["price"] = str(price)

        if client_order_id:
            order_data["userref"] = client_order_id

        result = await self._private_request("/0/private/AddOrder", order_data)

        return Order(
            order_id=result["txid"][0] if result.get("txid") else "",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
            average_fill_price=None,
            timestamp=datetime.now(),
            exchange="kraken",
            client_order_id=client_order_id,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """Cancel an open order."""
        await self._private_request("/0/private/CancelOrder", {"txid": order_id})

        # Fetch updated order status
        return await self.get_order(order_id, symbol)

    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Get order details."""
        result = await self._private_request("/0/private/QueryOrders", {"txid": order_id})

        order_data = result[order_id]

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if order_data["descr"]["type"] == "buy" else OrderSide.SELL,
            order_type=OrderType(order_data["descr"]["ordertype"]),
            quantity=float(order_data["vol"]),
            price=float(order_data["descr"]["price"]) if order_data["descr"]["price"] else None,
            status=self._map_order_status(order_data["status"]),
            filled_quantity=float(order_data["vol_exec"]),
            average_fill_price=float(order_data["price"]) if order_data.get("price") else None,
            timestamp=datetime.fromtimestamp(float(order_data["opentm"])),
            exchange="kraken",
        )

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        result = await self._private_request("/0/private/OpenOrders")

        orders = []
        for order_id, order_data in result.get("open", {}).items():
            order_symbol = self.denormalize_symbol(order_data["descr"]["pair"])

            if symbol is None or order_symbol == symbol:
                orders.append(
                    Order(
                        order_id=order_id,
                        symbol=order_symbol,
                        side=(
                            OrderSide.BUY
                            if order_data["descr"]["type"] == "buy"
                            else OrderSide.SELL
                        ),
                        order_type=OrderType(order_data["descr"]["ordertype"]),
                        quantity=float(order_data["vol"]),
                        price=(
                            float(order_data["descr"]["price"])
                            if order_data["descr"]["price"]
                            else None
                        ),
                        status=self._map_order_status(order_data["status"]),
                        filled_quantity=float(order_data["vol_exec"]),
                        average_fill_price=None,
                        timestamp=datetime.fromtimestamp(float(order_data["opentm"])),
                        exchange="kraken",
                    )
                )

        return orders

    async def get_balances(self) -> dict[str, Balance]:
        """Get account balances for all assets."""
        result = await self._private_request("/0/private/Balance")

        balances = {}
        for asset, amount in result.items():
            # Kraken uses prefixes like 'X' and 'Z' - normalize them
            # Handle special cases like XXBT -> BTC, XETH -> ETH, ZUSD -> USD
            normalized_asset = asset

            # First check if it's XXBT which should become BTC
            if normalized_asset == "XXBT":
                normalized_asset = "BTC"
            # For other assets starting with XX, strip first X
            elif normalized_asset.startswith("XX") or normalized_asset.startswith(("X", "Z")):
                normalized_asset = normalized_asset[1:]

            balances[normalized_asset] = Balance(
                asset=normalized_asset,
                free=float(amount),
                locked=0.0,  # Kraken doesn't separate free/locked in Balance endpoint
                total=float(amount),
            )

        return balances

    # Helper Methods

    def normalize_symbol(self, symbol: str) -> str:
        """Convert standard symbol format to Kraken REST API format."""
        # Example: "BTC/USD" -> "XBTUSD"
        symbol_map = {
            "BTC": "XBT",  # Kraken uses XBT for Bitcoin
        }

        base, quote = symbol.split("/")
        base = symbol_map.get(base, base)
        quote = symbol_map.get(quote, quote)

        return f"{base}{quote}"

    def normalize_symbol_ws(self, symbol: str) -> str:
        """Convert standard symbol format to Kraken WebSocket format."""
        # Example: "BTC/USD" -> "XBT/USD"
        # WebSocket API requires slash separator unlike REST API
        symbol_map = {
            "BTC": "XBT",  # Kraken uses XBT for Bitcoin
        }

        base, quote = symbol.split("/")
        base = symbol_map.get(base, base)
        quote = symbol_map.get(quote, quote)

        return f"{base}/{quote}"

    def denormalize_symbol(self, symbol: str) -> str:
        """Convert Kraken symbol format to standard format."""
        # This is a simplified implementation
        # Full implementation would need complete symbol mapping
        if "XBT" in symbol:
            symbol = symbol.replace("XBT", "BTC")

        # Try to split into base/quote (heuristic)
        if len(symbol) == 6:
            return f"{symbol[:3]}/{symbol[3:]}"
        return symbol

    @staticmethod
    def _map_order_status(kraken_status: str) -> OrderStatus:
        """Map Kraken order status to standard OrderStatus."""
        status_map = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
        }
        return status_map.get(kraken_status, OrderStatus.PENDING)
