"""
Centralized HTTP client module with automatic retry logic for transient errors.
This module provides drop-in replacements for httpx components with built-in retry support.

Usage:
    from utils import http_client
    
    # Instead of httpx.Client()
    client = http_client.Client()
    
    # Instead of httpx.get()
    response = http_client.get("https://api.example.com/data")
    
    # Exceptions work the same way
    try:
        response = client.get(url)
    except http_client.HTTPStatusError as e:
        handle_error(e)
"""

import logging
import random
import time
from typing import Any, Callable, Iterable, Optional
from contextlib import contextmanager

import httpcore
import httpx

# Re-export httpx exceptions so code doesn't need to change
from httpx import (
    TimeoutException,
    HTTPStatusError, 
    RequestError,
    ConnectError,
    ConnectTimeout,
    ReadTimeout,
    WriteTimeout,
    PoolTimeout,
    NetworkError,
    ProtocolError,
    ProxyError,
    Response,
    Headers,
    Cookies,
    URL,
    Timeout,
    Limits,
    HTTPTransport,
    AsyncHTTPTransport,
)

logger = logging.getLogger("http_client")

# Configuration for retry behavior
RETRYABLE_STATUS_CODES = {429, 502, 503, 504, 529}
BACKOFF_STATUS_CODES = {429, 529}  # Need longer delays
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30


class RetryMixin:
    """Mixin class providing retry logic for HTTP requests."""
    
    def __init__(self, *args, max_retries: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
        
    def _calculate_delay(self, attempt: int, status_code: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        if status_code in BACKOFF_STATUS_CODES:
            # Longer delays for rate limiting: 2, 4, 8 seconds
            base_delay = 2.0
        else:
            # Shorter delays for transient errors: 0.5, 1, 2 seconds
            base_delay = 0.5
            
        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        return min(delay, 30.0)  # Cap at 30 seconds
    
    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """Determine if a request should be retried based on status code and attempt number."""
        return status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries
    
    def _execute_with_retry(self, request_func: Callable[..., Response], *args: Any, **kwargs: Any) -> Response:
        """Execute a request function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return request_func(*args, **kwargs)
                
            except HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code if e.response else 0
                
                if self._should_retry(status_code, attempt):
                    delay = self._calculate_delay(attempt, status_code)
                    
                    if status_code == 529:
                        logger.warning(f"Server overloaded (529), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    elif status_code == 429:
                        logger.warning(f"Rate limited (429), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    else:
                        logger.warning(f"Server error ({status_code}), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    
                    time.sleep(delay)
                    continue
                else:
                    # Not retryable or max retries exceeded
                    raise
                    
            except (ConnectError, ConnectTimeout) as e:
                # Connection errors might be retryable
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt, 503)  # Treat as service unavailable
                    logger.warning(f"Connection error, attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise
                    
            except (TimeoutException, RequestError) as e:
                # Other errors are not retryable
                raise
        
        # If we exhausted retries, raise the last exception
        if last_exception:
            raise last_exception


class Client(RetryMixin, httpx.Client):
    """
    Drop-in replacement for httpx.Client with automatic retry logic.
    
    Automatically retries on:
    - 429 (Rate Limited)
    - 502 (Bad Gateway)
    - 503 (Service Unavailable)
    - 504 (Gateway Timeout)
    - 529 (Server Overloaded)
    - Connection errors
    """
    
    def __init__(self, *args, max_retries: Optional[int] = None, **kwargs):
        # Set default timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = DEFAULT_TIMEOUT
        super().__init__(*args, max_retries=max_retries, **kwargs)
    
    def request(self, *args, **kwargs):
        """Override request method to add retry logic."""
        return self._execute_with_retry(super().request, *args, **kwargs)
    
    def get(self, *args, **kwargs):
        """GET request with retry logic."""
        return self._execute_with_retry(super().get, *args, **kwargs)
    
    def post(self, *args, **kwargs):
        """POST request with retry logic."""
        return self._execute_with_retry(super().post, *args, **kwargs)
    
    def put(self, *args, **kwargs):
        """PUT request with retry logic."""
        return self._execute_with_retry(super().put, *args, **kwargs)
    
    def patch(self, *args, **kwargs):
        """PATCH request with retry logic."""
        return self._execute_with_retry(super().patch, *args, **kwargs)
    
    def delete(self, *args, **kwargs):
        """DELETE request with retry logic."""
        return self._execute_with_retry(super().delete, *args, **kwargs)
    
    def stream(self, *args, **kwargs):
        """
        Stream request with retry on connection establishment.
        Note: Once streaming starts, we can't retry mid-stream.
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return super().stream(*args, **kwargs)
                
            except HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code if e.response else 0
                
                if self._should_retry(status_code, attempt):
                    delay = self._calculate_delay(attempt, status_code)
                    logger.warning(f"Stream connection failed ({status_code}), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise
                    
            except (ConnectError, ConnectTimeout) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt, 503)
                    logger.warning(f"Stream connection error, attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise
        
        if last_exception:
            raise last_exception


# Convenience functions that mirror httpx module-level functions
def get(url: str, **kwargs) -> Response:
    """
    Convenience function for GET requests with automatic retry.
    
    Args:
        url: The URL to request
        **kwargs: Additional arguments passed to httpx.get()
        
    Returns:
        httpx.Response object
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.get(url, **kwargs)


def post(url: str, **kwargs) -> Response:
    """
    Convenience function for POST requests with automatic retry.
    
    Args:
        url: The URL to request
        **kwargs: Additional arguments passed to httpx.post()
        
    Returns:
        httpx.Response object
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.post(url, **kwargs)


def put(url: str, **kwargs) -> Response:
    """
    Convenience function for PUT requests with automatic retry.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.put(url, **kwargs)


def patch(url: str, **kwargs) -> Response:
    """
    Convenience function for PATCH requests with automatic retry.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.patch(url, **kwargs)


def delete(url: str, **kwargs) -> Response:
    """
    Convenience function for DELETE requests with automatic retry.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.delete(url, **kwargs)


@contextmanager
def stream(method: str, url: str, **kwargs):
    """
    Convenience function for streaming requests with automatic retry.

    Usage:
        with http_client.stream('GET', url) as response:
            for line in response.iter_lines():
                process(line)
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    http2 = kwargs.pop('http2', False)
    with Client(max_retries=max_retries, http2=http2) as client:
        with client.stream(method, url, **kwargs) as response:
            yield response


class _IPPinNetworkBackend(httpcore.NetworkBackend):
    """Network backend that connects to a pre-validated IP instead of re-resolving DNS.

    Closes the DNS-rebinding TOCTOU in SSRF validation: `utils.url_safety` resolves
    and validates the hostname once, then callers pass that exact IP here. The TCP
    connection targets the validated IP while TLS SNI and certificate verification
    still use the URL hostname. Connections for any other hostname fail loudly
    rather than silently falling back to fresh DNS resolution.
    """

    def __init__(self, hostname: str, ip: str):
        self._hostnames = {
            hostname.rstrip(".").lower(),
            hostname.encode("idna").decode("ascii").rstrip(".").lower(),
        }
        self._ip = ip
        self._backend = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        if host.rstrip(".").lower() not in self._hostnames:
            raise httpcore.ConnectError(
                f"IP-pinned connection attempted for unvalidated hostname: {host}"
            )
        return self._backend.connect_tcp(
            self._ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return self._backend.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


class PinnedHTTPTransport(httpx.HTTPTransport):
    """HTTP transport pinned to a single pre-validated IP address."""

    def __init__(self, hostname: str, ip: str):
        super().__init__()
        self._pool = httpcore.ConnectionPool(
            network_backend=_IPPinNetworkBackend(hostname, ip)
        )


def pinned_request(method: str, url: str, *, hostname: str, ip: str, **kwargs: Any) -> Response:
    """Issue an HTTP request whose TCP connection targets a pre-validated IP.

    `hostname` and `ip` must come from a `utils.url_safety.ValidatedURL` so the
    connection uses the exact address that passed SSRF validation. TLS SNI and
    certificate verification still use the URL hostname.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    transport = PinnedHTTPTransport(hostname, ip)
    with Client(max_retries=max_retries, transport=transport) as client:
        return client.request(method, url, **kwargs)
