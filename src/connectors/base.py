"""
Base abstract classes for exchange/broker connectors.

This module defines the interfaces that all connector implementations must follow,
ensuring a consistent API regardless of the underlying exchange or broker.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OrderType(Enum):
    """Order types supported across exchanges."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order side (direction)."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Ticker:
    """Real-time ticker data."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume_24h: float
    high_24h: float
    low_24h: float
    exchange: str


@dataclass
class OrderBook:
    """Order book snapshot."""

    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]  # [(price, quantity), ...]
    asks: list[tuple[float, float]]
    exchange: str


@dataclass
class Trade:
    """Individual trade (tick) data."""

    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: OrderSide
    trade_id: str
    exchange: str


@dataclass
class OHLCV:
    """OHLCV candlestick data."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str
    interval: str  # e.g., "1m", "5m", "1h", "1d"


@dataclass
class Order:
    """Order submitted to exchange."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None  # None for market orders
    status: OrderStatus
    filled_quantity: float
    average_fill_price: float | None
    timestamp: datetime
    exchange: str
    client_order_id: str | None = None


@dataclass
class Balance:
    """Account balance for a single asset."""

    asset: str
    free: float  # Available for trading
    locked: float  # In open orders
    total: float


class ExchangeConnector(ABC):
    """
    Abstract base class for all exchange/broker connectors.

    All methods that interact with external APIs should handle exceptions
    and return standardized data structures.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialize the connector.

        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet/sandbox environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the exchange."""
        pass

    # Market Data Methods (Public)

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")

        Returns:
            Ticker object with current market data
        """
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """
        Get current order book for a symbol.

        Args:
            symbol: Trading pair symbol
            depth: Number of price levels to retrieve

        Returns:
            OrderBook object
        """
        pass

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCV]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading pair symbol
            interval: Candlestick interval (e.g., "1m", "1h", "1d")
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of candles to retrieve

        Returns:
            List of OHLCV objects
        """
        pass

    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get recent trades for a symbol.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to retrieve

        Returns:
            List of Trade objects
        """
        pass

    # WebSocket Streaming Methods

    @abstractmethod
    def stream_ticker(self, symbols: list[str]) -> AsyncIterator[Ticker]:
        """
        Stream real-time ticker updates via WebSocket.

        Args:
            symbols: List of symbols to subscribe to

        Yields:
            Ticker objects as they arrive
        """
        # Implementation should be an async generator (async def with yield)
        raise NotImplementedError

    @abstractmethod
    def stream_orderbook(self, symbols: list[str], depth: int = 20) -> AsyncIterator[OrderBook]:
        """
        Stream real-time order book updates via WebSocket.

        Args:
            symbols: List of symbols to subscribe to
            depth: Number of price levels to retrieve (default: 20)

        Yields:
            OrderBook objects as they arrive
        """
        # Implementation should be an async generator (async def with yield)
        raise NotImplementedError

    @abstractmethod
    def stream_trades(self, symbols: list[str]) -> AsyncIterator[Trade]:
        """
        Stream real-time trades via WebSocket.

        Args:
            symbols: List of symbols to subscribe to

        Yields:
            Trade objects as they arrive
        """
        # Implementation should be an async generator (async def with yield)
        raise NotImplementedError

    # Trading Methods (Private - require authentication)

    @abstractmethod
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
        """
        Place a new order.

        Args:
            symbol: Trading pair symbol
            side: BUY or SELL
            order_type: Order type (MARKET, LIMIT, etc.)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            client_order_id: Custom order ID for tracking
            **kwargs: Exchange-specific parameters

        Returns:
            Order object with order details
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol

        Returns:
            Updated Order object
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """
        Get order details.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol

        Returns:
            Order object with current status
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open Order objects
        """
        pass

    @abstractmethod
    async def get_balances(self) -> dict[str, Balance]:
        """
        Get account balances for all assets.

        Returns:
            Dictionary mapping asset symbols to Balance objects
        """
        pass

    # Helper Methods

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format to exchange-specific format.

        Args:
            symbol: Symbol in standard format (e.g., "BTC/USD")

        Returns:
            Exchange-specific symbol format
        """
        # Default implementation - override in subclasses
        return symbol.replace("/", "")

    def denormalize_symbol(self, symbol: str) -> str:
        """
        Convert exchange-specific symbol to standard format.

        Args:
            symbol: Exchange-specific symbol

        Returns:
            Standard format symbol (e.g., "BTC/USD")
        """
        # Default implementation - override in subclasses
        return symbol
