"""
Rate limiting utilities for API requests.
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter with exponential backoff for API requests."""

    def __init__(self, max_requests_per_minute: int = 15, max_backoff: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
            max_backoff: Maximum backoff time in seconds
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_backoff = max_backoff
        self.requests = []  # Timestamps of recent requests
        self._lock = asyncio.Lock()
        self._backoff_multiplier = 1

    def _clean_old_requests(self, now: float) -> None:
        """Remove requests older than 1 minute."""
        cutoff = now - 60
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]

    def _get_wait_time(self, now: float) -> float:
        """Calculate wait time based on rate limits and backoff."""
        self._clean_old_requests(now)

        # Check if we're at rate limit
        if len(self.requests) >= self.max_requests_per_minute:
            # Calculate time until oldest request is older than 1 minute
            oldest_request = min(self.requests)
            wait_time = max(0, 60 - (now - oldest_request))
            return wait_time + (1 * self._backoff_multiplier)

        return 0

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            wait_time = self._get_wait_time(now)

            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

            self.requests.append(time.time())

    def reset_backoff(self) -> None:
        """Reset backoff multiplier after successful request."""
        self._backoff_multiplier = 1

    def increase_backoff(self) -> None:
        """Increase backoff multiplier after rate limit error."""
        self._backoff_multiplier = min(self._backoff_multiplier * 2, self.max_backoff // 2)

    async def handle_rate_limit_error(self) -> None:
        """Handle rate limit error with exponential backoff."""
        self.increase_backoff()
        backoff_time = min(self._backoff_multiplier, self.max_backoff)
        logger.warning(f"Rate limit hit, backing off for {backoff_time} seconds")
        await asyncio.sleep(backoff_time)


class RetryableRequest:
    """Wrapper for making retryable API requests with rate limiting."""

    def __init__(self, rate_limiter: RateLimiter, max_retries: int = 3):
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries

    async def execute(self, request_func, *args, **kwargs):
        """
        Execute a request function with retries and rate limiting.

        Args:
            request_func: Async function to call
            *args: Positional arguments for request_func
            **kwargs: Keyword arguments for request_func

        Returns:
            Result of request_func

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries + 1):
            try:
                # Acquire rate limit permission
                await self.rate_limiter.acquire()

                # Make the request
                result = await request_func(*args, **kwargs)

                # Reset backoff on success
                self.rate_limiter.reset_backoff()
                return result

            except Exception as e:
                # Check if it's a rate limit error
                error_str = str(e).lower()
                is_rate_limit = (
                    "too many requests" in error_str
                    or "rate limit" in error_str
                    or "eGeneral" in str(e)
                )

                if is_rate_limit and attempt < self.max_retries:
                    logger.warning(f"Rate limit error on attempt {attempt + 1}: {e}")
                    await self.rate_limiter.handle_rate_limit_error()
                    continue

                elif attempt < self.max_retries:
                    # Other error, retry with exponential backoff
                    backoff_time = 2**attempt
                    logger.warning(
                        f"Request failed on attempt {attempt + 1}, retrying in {backoff_time}s: {e}"
                    )
                    await asyncio.sleep(backoff_time)
                    continue

                else:
                    # All retries exhausted
                    logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
                    raise

        raise Exception("Unexpected error in retry loop")
