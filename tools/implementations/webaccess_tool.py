"""
Web access tool that combines HTTP requests, web search, and webpage extraction.

This tool provides a comprehensive web interaction interface, allowing the system to:
1. Make HTTP requests to external APIs and web services
2. Perform web searches using Kagi search API
3. Extract and parse content from webpages using Claude's AI capabilities
"""
import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from utils import http_client
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from kagiapi import KagiClient

from tools.repo import Tool
from tools.registry import registry

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class MaterializedResponse:
    """
    A response object that holds all HTTP response data without requiring an open connection.

    This class mimics the interface of http_client.Response but stores all data internally,
    allowing it to be used after the HTTP connection has been closed.
    """
    def __init__(self, status_code: int, url: str, headers: Dict[str, str], content: bytes):
        self.status_code = status_code
        self.url = url
        self.headers = headers
        self._content = content
        self._text = None

    @property
    def text(self) -> str:
        """Decode content to text using UTF-8 encoding."""
        if self._text is None:
            self._text = self._content.decode('utf-8', errors='replace')
        return self._text

    def json(self) -> Any:
        """Parse response text as JSON."""
        import json
        return json.loads(self.text)



def _format_error_message(message: str, details: Optional[Dict[str, Any]] = None) -> str:
    """Combine a human-readable message with optional structured details."""
    if not details:
        return message
    try:
        details_json = json.dumps(details, default=str)
    except TypeError:
        details_json = str(details)
    return f"{message} | details={details_json}"


def _raise_api_response_error(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Raise a RuntimeError for failed HTTP responses."""
    raise RuntimeError(_format_error_message(message, details))


def _raise_api_timeout(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Raise a TimeoutError when remote calls exceed the allotted time."""
    raise TimeoutError(_format_error_message(message, details))


def _raise_api_connection_error(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Raise a ConnectionError when network issues occur."""
    raise ConnectionError(_format_error_message(message, details))


def _raise_invalid_input(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Raise a ValueError for invalid tool parameters."""
    raise ValueError(_format_error_message(message, details))


def _raise_configuration_error(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Raise a RuntimeError when tool configuration is invalid."""
    raise RuntimeError(_format_error_message(message, details))

# Define configuration class for WebAccessTool
class WebAccessConfig(BaseModel):
    """Configuration for the webaccess_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    timeout: int = Field(default=30, description="Timeout in seconds for HTTP requests")
    max_timeout: int = Field(default=120, description="Maximum timeout allowed for HTTP requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    allowed_domains: List[str] = Field(default=[], description="List of allowed domains for requests (empty for all)")
    blocked_domains: List[str] = Field(default=[], description="List of domains to exclude from results")
    max_searches_per_request: int = Field(default=3, description="Maximum number of searches allowed per request")
    default_extraction_prompt: str = Field(
        default="Please extract the main content from this webpage. Focus on the article text, headings, and important information. Ignore navigation, ads, footers, and other non-essential elements.",
        description="Default prompt to use for content extraction"
    )
    user_agent: str = Field(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0",
        description="User-Agent string for browser-like requests"
    )

# Register with registry
registry.register("webaccess_tool", WebAccessConfig)


class WebAccessTool(Tool):
    """
    Web access tool that combines HTTP requests, web search, and webpage extraction.
    
    This integrated tool provides three main capabilities:
    1. HTTP Tool: Make direct HTTP requests to external APIs and web services
    2. Web Search Tool: Perform web searches using Kagi search API
    3. Webpage Extraction Tool: Extract and parse content from webpages
    
    Having these capabilities combined in one tool allows for more efficient web
    interaction workflows, such as searching for information, accessing specific
    URLs found in search results, and extracting content from those pages.
    """
    
    name = "webaccess_tool"
    simple_description = "Access the web with three capabilities: (1) HTTP requests to APIs (GET/POST/PUT/DELETE), (2) web search using Kagi for current information, (3) extract clean content from webpages. Use for APIs, research, or fetching web content."

    anthropic_schema = {
        "name": "webaccess_tool",
        "description": "Provides comprehensive web access with HTTP requests, web searches, and webpage extraction",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["http_request", "web_search", "webpage_extract"],
                        "description": "The web access operation to perform"
                    },
                    # HTTP request parameters
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "description": "HTTP method for http_request operation"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL for http_request or webpage_extract operations"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters for http_request operation"
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers for http_request operation"
                    },
                    "data": {
                        "type": ["object", "string"],
                        "description": "Form data for http_request operation"
                    },
                    "json": {
                        "type": "object",
                        "description": "JSON data for http_request operation"
                    },
                    "response_format": {
                        "type": "string",
                        "enum": ["json", "text", "full"],
                        "description": "Response format for http_request operation"
                    },
                    # Web search parameters
                    "query": {
                        "type": "string",
                        "description": "Search query for web_search operation"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results for web_search operation"
                    },
                    # Webpage extraction parameters
                    "extraction_prompt": {
                        "type": "string",
                        "description": "Content extraction prompt for webpage_extract operation"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["text", "markdown", "html"],
                        "description": "Output format for webpage_extract operation"
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Whether to include page metadata in webpage_extract operation"
                    },
                    # Shared parameters
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds"
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of domains to include in results"
                    },
                    "blocked_domains": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of domains to exclude from results"
                    }
                },
                "required": ["operation"]
            }
        }

    def __init__(self):
        """
        Initialize the web access tool with configuration and setup.
        """
        super().__init__()
        self.logger.info("WebAccessTool initialized")
        
        # List of blocked URL patterns for security
        self._blocked_url_patterns = [
            r'^https?://localhost',
            r'^https?://127\.',
            r'^https?://10\.',
            r'^https?://172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'^https?://192\.168\.',
            r'^https?://0\.0\.0\.0',
        ]
        
        # Initialize Kagi client with API key from vault
        self._kagi_client = None
        self._init_kagi_client()

    def run(self, **params) -> Dict[str, Any]:
        """
        Execute a web access operation based on the specified parameters.

        This is the main entry point for the web access tool. It routes to the
        appropriate operation handler based on the 'operation' parameter.

        Args:
            operation: The operation to perform ('http_request', 'web_search', or 'webpage_extract')
            [Other parameters depend on the operation chosen]

        Returns:
            Dictionary containing the result of the operation (structure depends on operation)

        Raises:
            ValueError: If inputs are invalid or if the operation fails
        """
        # Extract operation
        operation = params.get("operation")
        
        # Validate operation parameter
        if not operation:
            self.logger.error(f"Required parameter 'operation' must be provided. Provided params: {list(params.keys())}")
            raise ValueError("Required parameter 'operation' must be provided")
            
        # Route to appropriate handler based on operation
        if operation == "http_request":
            return self._handle_http_request(params)
        elif operation == "web_search":
            return self._handle_web_search(params)
        elif operation == "webpage_extract":
            return self._handle_webpage_extract(params)
        else:
            self.logger.error(f"Invalid operation: {operation}. Must be one of: http_request, web_search, webpage_extract")
            raise ValueError(f"Invalid operation: {operation}. Must be one of: http_request, web_search, webpage_extract")

    def _make_http_request(self, method, url, params=None, headers=None, data=None, json_data=None,
                       timeout=None, is_browser=False, max_content_size=10*1024*1024,
                       retries=None, retry_status_codes=None, retry_delay=1.0):
        """
        Shared function for making HTTP requests.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: The URL to send the request to
            params: Optional query parameters as a dictionary
            headers: Optional HTTP headers as a dictionary
            data: Optional form data (for POST/PUT)
            json_data: Optional JSON data (for POST/PUT)
            timeout: Request timeout in seconds
            is_browser: Whether to use browser-like headers
            max_content_size: Maximum content size in bytes (default 10MB)
            retries: Number of retries for transient errors (default from config)
            retry_status_codes: HTTP status codes to retry on (default: 429, 500, 502, 503, 504)
            retry_delay: Base delay between retries in seconds (will increase exponentially)

        Returns:
            http_client.Response object on success, or dict with error details on failure

        Raises:
            ValueError: Only for input validation errors
        """
        self.logger.debug(f"Making {method} request to {url}")

        # Get configuration for user agent
        from config import config

        try:
            user_agent = config.webaccess_tool.user_agent
        except AttributeError:
            user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0"

        # Set default headers for browser-like requests
        if is_browser:
            browser_headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "Priority": "u=0, i"
            }

            # Merge with any provided headers (provided headers take precedence)
            if headers:
                browser_headers.update(headers)
            headers = browser_headers
        elif headers is None:
            headers = {}
        
        # Get configuration for retries
        from config import config
        
        # Set up retry configuration
        if retries is None:
            try:
                retries = config.webaccess_tool.max_retries
            except AttributeError:
                retries = 3  # Default to 3 retries
                
        if retry_status_codes is None:
            retry_status_codes = [429, 500, 502, 503, 504]
        
        try:
            # Initialize variables for retry logic
            attempts = 0
            last_exception = None
            
            while attempts <= retries:
                attempts += 1
                try:
                    # Use streaming for large responses to avoid memory issues
                    with http_client.stream(
                        method=method,
                        url=url,
                        params=params,
                        headers=headers,
                        data=data,
                        json=json_data,
                        timeout=timeout,
                        follow_redirects=True,
                        http2=True
                    ) as response:
                        # Check if we got a status code that should trigger a retry
                        if response.status_code in retry_status_codes and attempts <= retries:
                            # Get retry-after header if it exists (used by many APIs for rate limiting)
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                try:
                                    # Try to parse as integer seconds
                                    wait_time = float(retry_after)
                                except ValueError:
                                    # Default to exponential backoff
                                    wait_time = retry_delay * (2 ** (attempts - 1))
                            else:
                                # Use exponential backoff
                                wait_time = retry_delay * (2 ** (attempts - 1))
                                
                            self.logger.warning(
                                f"Request to {url} returned status {response.status_code}, "
                                f"retrying in {wait_time:.1f} seconds (attempt {attempts}/{retries})"
                            )
                            
                            # Small jitter to avoid thundering herd problems
                            import random
                            import time
                            time.sleep(wait_time + random.uniform(0, 0.5))
                            continue
                        
                        # Log response info
                        self.logger.debug(f"Response received: Status {response.status_code}")

                        # Check if status code indicates an error (but not one we retry on)
                        if response.status_code >= 400 and response.status_code not in retry_status_codes:
                            error_msg = f"HTTP {response.status_code}"
                            try:
                                # Try to get more details from response
                                error_msg = f"HTTP {response.status_code}: {response.reason_phrase if hasattr(response, 'reason_phrase') else ''}"
                            except:
                                pass
                            self.logger.error(f"HTTP error {response.status_code} for {url}")
                            return {
                                "error": "http_status_error",
                                "message": error_msg,
                                "url": str(response.url),
                                "method": method,
                                "status_code": response.status_code,
                                "attempts": attempts
                            }

                        # Check content length header
                        content_length = response.headers.get('Content-Length')
                        if content_length and int(content_length) > max_content_size:
                            error_msg = f"Content too large: {int(content_length) // (1024*1024)}MB exceeds limit of {max_content_size // (1024*1024)}MB"
                            self.logger.error(error_msg)
                            return {
                                "error": "content_too_large",
                                "message": error_msg,
                                "url": url,
                                "content_length": int(content_length),
                                "max_size": max_content_size
                            }
                        
                        # Load content in chunks to avoid memory issues
                        content = bytearray()
                        total_size = 0

                        for chunk in response.iter_bytes(chunk_size=1024*1024):  # 1MB chunks
                            if not chunk:
                                continue

                            total_size += len(chunk)
                            if total_size > max_content_size:
                                # Clear any partial content to free memory
                                content = None

                                error_msg = f"Content too large: exceeds limit of {max_content_size // (1024*1024)}MB"
                                self.logger.error(error_msg)
                                return {
                                    "error": "content_too_large",
                                    "message": error_msg,
                                    "url": url,
                                    "max_size": max_content_size
                                }

                            content.extend(chunk)

                        # Materialize all response data before exiting context manager
                        # This ensures data is accessible after the connection closes
                        materialized = MaterializedResponse(
                            status_code=response.status_code,
                            url=str(response.url),
                            headers=dict(response.headers),
                            content=bytes(content)
                        )

                        return materialized
                except (http_client.TimeoutException,
                       http_client.ConnectError,
                       http_client.HTTPStatusError,
                       http_client.RequestError) as e:
                    # Save the exception for potential re-raising
                    last_exception = e
                    
                    # Determine if this exception should trigger a retry
                    should_retry = False
                    
                    # Check exception type to determine if it's retryable
                    if isinstance(e, (http_client.TimeoutException, http_client.ConnectError)):
                        # Network-related errors are generally retryable
                        should_retry = True
                    elif isinstance(e, http_client.HTTPStatusError):
                        # Retry specific HTTP status codes
                        if e.response.status_code in retry_status_codes:
                            should_retry = True
                    
                    # Check if we should retry
                    if should_retry and attempts < retries:
                        # Calculate backoff time
                        wait_time = retry_delay * (2 ** (attempts - 1))
                        
                        self.logger.warning(
                            f"Request to {url} failed with {type(e).__name__}, "
                            f"retrying in {wait_time:.1f} seconds (attempt {attempts}/{retries}). "
                            f"Error: {str(e)}"
                        )
                        
                        # Add jitter to avoid thundering herd problems
                        import random
                        import time
                        time.sleep(wait_time + random.uniform(0, 0.5))
                        continue
                    
                    # If we reach here, we either exhausted retries or the exception isn't retryable
                    if isinstance(e, http_client.TimeoutException):
                        self.logger.error(f"Request to {url} timed out after {timeout} seconds (after {attempts} attempts)")
                        return {
                            "error": "timeout",
                            "message": f"Request timed out after {timeout} seconds",
                            "url": url,
                            "method": method,
                            "timeout": timeout,
                            "attempts": attempts
                        }
                    elif isinstance(e, http_client.ConnectError):
                        self.logger.error(f"Connection error for {url}: {str(e)} (after {attempts} attempts)")
                        return {
                            "error": "connection_error",
                            "message": f"Connection error: {str(e)}",
                            "url": url,
                            "method": method,
                            "details": str(e),
                            "attempts": attempts
                        }
                    elif isinstance(e, http_client.HTTPStatusError):
                        status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else "unknown"
                        self.logger.error(f"HTTP error {status_code} for {url}: {str(e)} (after {attempts} attempts)")
                        return {
                            "error": "http_status_error",
                            "message": f"HTTP error {status_code}: {str(e)}",
                            "url": url,
                            "method": method,
                            "status_code": status_code,
                            "details": str(e),
                            "attempts": attempts
                        }
                    else:
                        self.logger.error(f"Request error for {url}: {str(e)} (after {attempts} attempts)")
                        return {
                            "error": "request_error",
                            "message": f"Request error: {str(e)}",
                            "url": url,
                            "method": method,
                            "details": str(e),
                            "attempts": attempts
                        }
            # If we've exhausted all retries without success
            if last_exception:
                self.logger.error(f"Request to {url} failed after {retries} retries")
                return {
                    "error": "request_failed",
                    "message": f"Request failed after {retries} retries: {str(last_exception)}",
                    "url": url,
                    "method": method,
                    "details": str(last_exception),
                    "attempts": attempts
                }
        except Exception as e:
            self.logger.error(f"Error making HTTP request to {url}: {e}")
            raise

    def _handle_http_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle HTTP request operations.
        
        This function handles direct HTTP requests to external APIs and services.
        
        Args:
            params: Dictionary of parameters for the HTTP request
            
        Returns:
            Dictionary containing the HTTP response
            
        Raises:
            Exception: If inputs are invalid or if the request fails
        """
        # Extract expected parameters
        method = params.get("method")
        url = params.get("url")
        request_params = params.get("params")
        headers = params.get("headers")
        data = params.get("data")
        json_data = params.get("json")
        timeout = params.get("timeout")
        response_format = params.get("response_format", "json")

        # Validate required parameters
        if not method or not url:
            _raise_invalid_input(
                "Required parameters 'method' and 'url' must be provided for http_request operation",
                {"provided_params": list(params.keys())}
            )

        self.logger.info(f"Executing HTTP {method} request to {url}")

        # Input validation
        self._validate_http_inputs(method, url, response_format, timeout)

        # Set default timeout if not provided
        timeout = self._validate_timeout(timeout)

        # Prepare the request
        method = str(method).upper()
        
        # Use shared HTTP request function
        response = self._make_http_request(
            method=method,
            url=url,
            params=request_params,
            headers=headers,
            data=data,
            json_data=json_data,
            timeout=timeout
        )

        # Check if we got an error response
        if isinstance(response, dict) and "error" in response:
            return {
                "success": False,
                "error": response["error"],
                "message": response["message"],
                "url": response.get("url", url),
                "details": {k: v for k, v in response.items() if k not in ["error", "message", "url"]}
            }

        # Format and return the response
        return self._format_http_response(response, response_format)
                
    def _validate_http_inputs(self, method, url, response_format, timeout):
        """
        Validate input parameters before executing an HTTP request.
        
        Args:
            method: HTTP method string
            url: URL string
            response_format: Response format string
            timeout: Timeout value in seconds
            
        Raises:
            Exception: If any inputs are invalid
        """
        # Validate HTTP method
        valid_methods = ["GET", "POST", "PUT", "DELETE"]
        if not method or method.upper() not in valid_methods:
            _raise_invalid_input(
                f"Invalid HTTP method: {method}. Must be one of: {', '.join(valid_methods)}",
                {"provided_method": method, "valid_methods": valid_methods}
            )
        
        # Validate URL with operation context for domain restrictions
        self._validate_url(url, operation="http_request")
            
        # Validate response format
        valid_formats = ["json", "text", "full"]
        if response_format not in valid_formats:
            _raise_invalid_input(
                f"Invalid response format: {response_format}. Must be one of: {', '.join(valid_formats)}",
                {"provided_format": response_format, "valid_formats": valid_formats}
            )
            
        # Timeout validation is handled by the shared _validate_timeout method
        
    def _format_http_response(self, response, response_format):
        """
        Format the HTTP response according to the specified format.
        
        Args:
            response: The http_client.Response object
            response_format: The format to return ("json", "text", or "full")
            
        Returns:
            Formatted response dictionary
        """
        result = {
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": response.url
        }
        
        # Detect content type
        content_info = self._detect_content_type(response=response)
        result["content_type"] = f"{content_info['mimetype']}/{content_info['subtype']}"
        
        # Format based on specified format
        if response_format == "json":
            try:
                # Use response's built-in JSON parser first for efficiency
                result["data"] = response.json()
            except ValueError:
                # If built-in parsing fails, check if content appears to be JSON
                if content_info["format"] == "json":
                    # If content type detection suggests JSON, use our robust parser
                    try:
                        result["data"] = self._parse_json_response(
                            response.text, 
                            expected_format=None  # Auto-detect format
                        )
                    except Exception:
                        # If that also fails, include the raw text
                        result["data"] = response.text
                        result["warning"] = "Response could not be parsed as JSON despite Content-Type"
                else:
                    # Not JSON content type
                    result["data"] = response.text
                    result["warning"] = "Response could not be parsed as JSON"
                
        elif response_format == "text":
            result["data"] = response.text
            
        elif response_format == "full":
            result["data"] = response.text
            result["headers"] = dict(response.headers)
            result["content_info"] = content_info
            
            # Try to include JSON if content might be JSON
            if content_info["format"] == "json":
                try:
                    # Use our robust parser for the JSON field
                    result["json"] = self._parse_json_response(
                        response.text,
                        expected_format=None  # Auto-detect format
                    )
                except Exception:
                    # If parsing fails, don't include JSON
                    pass
            else:
                # Try standard parsing for non-JSON content types that might contain JSON
                try:
                    result["json"] = response.json()
                except ValueError:
                    # Not JSON, so we don't include it
                    pass
        
        return result
        
    def _handle_web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle web search operations.
        
        This function performs web searches using Anthropic's search API.
        
        Args:
            params: Dictionary of parameters for the web search
            
        Returns:
            Dictionary containing the search results
            
        Raises:
            Exception: If inputs are invalid or if the search fails
        """
        # Extract expected parameters
        query = params.get("query")
        max_results = params.get("max_results", 3)
        allowed_domains = params.get("allowed_domains", [])
        blocked_domains = params.get("blocked_domains", [])

        # Validate required parameters
        if not query:
            _raise_invalid_input(
                "Required parameter 'query' must be provided for web_search operation",
                {"provided_params": list(params.keys())}
            )

        self.logger.info(f"Executing web search for: {query}")

        # Validate inputs
        self._validate_web_search_inputs(query, max_results, allowed_domains, blocked_domains)
        
        # Execute the search via Kagi
        search_results = self._execute_kagi_search(query, max_results, allowed_domains, blocked_domains)
        
        # Process and return results
        return {
            "success": True,
            "results": search_results
        }
            
    def _validate_web_search_inputs(self, query, max_results, allowed_domains, blocked_domains):
        """
        Validate input parameters before executing the search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            allowed_domains: List of domains to include
            blocked_domains: List of domains to exclude
            
        Raises:
            Exception: If any inputs are invalid
        """
        # Validate query
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            _raise_invalid_input(
                "Search query must be a non-empty string",
                {"provided_query": str(query)}
            )
        
        # Validate max_results
        if max_results is not None:
            if not isinstance(max_results, int) or max_results <= 0:
                _raise_invalid_input(
                    "max_results must be a positive integer",
                    {"provided_max_results": max_results}
                )
        
        # Get consolidated domains before validating them
        final_allowed, final_blocked = self._get_domain_restrictions(
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            operation="web_search"
        )
        
        # Validate domain restrictions using the shared domain validation method
        self._validate_domains(final_allowed, final_blocked)
    
    def _init_kagi_client(self):
        """
        Initialize the Kagi client with API key from vault.
        """
        try:
            from clients.vault_client import get_api_key
            kagi_api_key = get_api_key("kagi_api_key")
            if kagi_api_key:
                self._kagi_client = KagiClient(kagi_api_key)
                self.logger.info("Kagi client initialized successfully")
            else:
                self.logger.warning("Kagi API key not found in vault")
                self._kagi_client = None
        except Exception as e:
            self.logger.error(f"Failed to initialize Kagi client: {e}")
            self._kagi_client = None

    def _execute_kagi_search(self, query: str, max_results: int, allowed_domains: List[str], blocked_domains: List[str]) -> List[Dict[str, Any]]:
        """
        Execute search using Kagi API with post-filtering for domain restrictions.

        Args:
            query: The search query to execute
            max_results: Maximum number of results to return
            allowed_domains: List of domains to include
            blocked_domains: List of domains to exclude

        Returns:
            List of search result objects

        Raises:
            Exception: If the search fails or Kagi client is not available
        """
        if not self._kagi_client:
            self.logger.warning("Web search attempted but Kagi API key not configured in vault")
            raise ValueError(
                "Web search functionality requires a Kagi API key. "
                "Please ask the user to add their Kagi API key to the vault with the key name 'kagi_api_key'. "
                "Users can obtain a Kagi API key from https://kagi.com/settings?p=api"
            )
        
        try:
            # Get consolidated domain restrictions
            final_allowed, final_blocked = self._get_domain_restrictions(
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                operation="web_search"
            )
            
            # Execute Kagi search - request exactly what was asked for
            response = self._kagi_client.search(query, limit=max_results)
            
            # Transform Kagi response to expected format with domain filtering
            search_results = []
            for item in response.get('data', []):
                # Extract fields from Kagi SearchItem
                result = {
                    "title": item.get('title', ''),
                    "url": item.get('url', ''),
                    "content": item.get('snippet', '')
                }
                
                # Apply domain filtering (allowlist OR blocklist, not both)
                if self._should_include_result(result['url'], final_allowed, final_blocked):
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Kagi search failed: {e}")
            _raise_api_response_error(
                f"Kagi search failed: {str(e)}",
                {"error": str(e), "query": query}
            )
    
    def _should_include_result(self, url: str, allowed_domains: List[str], blocked_domains: List[str]) -> bool:
        """
        Check if a search result URL should be included based on domain restrictions.
        Uses either allowlist OR blocklist, not both.
        
        Args:
            url: The URL to check
            allowed_domains: List of allowed domains (can include regex patterns)
            blocked_domains: List of blocked domains (can include regex patterns)
            
        Returns:
            True if the result should be included, False otherwise
        """
        if not url:
            return False
            
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # If allowlist is specified, use it (ignore blocklist)
            if allowed_domains:
                for allowed in allowed_domains:
                    allowed_lower = allowed.lower()
                    # Try exact match, subdomain match, then regex match
                    if (domain == allowed_lower or 
                        domain.endswith('.' + allowed_lower)):
                        return True
                    # Check regex pattern if it contains regex special characters
                    if any(char in allowed_lower for char in ['*', '+', '?', '^', '$', '[', ']', '(', ')', '|', '\\']):
                        try:
                            if re.match(allowed_lower, domain, re.IGNORECASE):
                                return True
                        except re.error:
                            pass  # Invalid regex, skip
                return False  # Not in allowed list
            
            # Otherwise use blocklist if specified
            elif blocked_domains:
                for blocked in blocked_domains:
                    blocked_lower = blocked.lower()
                    # Try exact match, subdomain match, then regex match
                    if (domain == blocked_lower or 
                        domain.endswith('.' + blocked_lower)):
                        return False
                    # Check regex pattern if it contains regex special characters
                    if any(char in blocked_lower for char in ['*', '+', '?', '^', '$', '[', ']', '(', ')', '|', '\\']):
                        try:
                            if re.match(blocked_lower, domain, re.IGNORECASE):
                                return False
                        except re.error:
                            pass  # Invalid regex, skip
            
            return True  # No restrictions or passed all checks
            
        except Exception as e:
            self.logger.warning(f"Failed to parse URL for domain filtering: {url}, error: {e}")
            return False

    def _detect_content_type(self, response=None, content=None, headers=None):
        """
        Detect the content type of a response or raw content.
        
        This function analyzes response headers and/or content to determine
        the content type and format for better handling.
        
        Args:
            response: Optional http_client.Response object
            content: Optional string content to analyze
            headers: Optional headers dictionary
            
        Returns:
            Dictionary with detected content properties:
            {
                "mimetype": Primary MIME type (e.g., "text", "application")
                "subtype": MIME subtype (e.g., "html", "json")
                "format": Detected format ("json", "html", "xml", "text", etc.)
                "encoding": Detected character encoding
                "is_binary": Boolean indicating if content appears to be binary
            }
        """
        result = {
            "mimetype": "text",
            "subtype": "plain",
            "format": "text",
            "encoding": "utf-8",
            "is_binary": False
        }
        
        # Check response and headers
        if response and response.headers:
            headers = response.headers
        
        # Extract content type from headers
        if headers:
            content_type = headers.get('Content-Type', '').lower()
            if content_type:
                # Parse the content type
                parts = content_type.split(';', 1)
                mimetype_full = parts[0].strip()
                
                # Extract encoding if present
                if len(parts) > 1 and 'charset=' in parts[1].lower():
                    encoding = parts[1].lower().split('charset=', 1)[1].strip()
                    result["encoding"] = encoding
                
                # Split mimetype into main type and subtype
                if '/' in mimetype_full:
                    mimetype, subtype = mimetype_full.split('/', 1)
                    result["mimetype"] = mimetype.strip()
                    result["subtype"] = subtype.strip()
                    
                    # Set format based on mimetype/subtype
                    if subtype in ['json', 'html', 'xml', 'javascript', 'css']:
                        result["format"] = subtype
                    elif 'json' in subtype:  # application/ld+json, etc.
                        result["format"] = "json"
                    elif mimetype == "application" and subtype in ["octet-stream", "pdf", "zip"]:
                        result["is_binary"] = True
                        result["format"] = subtype
                    elif mimetype in ["image", "audio", "video"]:
                        result["is_binary"] = True
                        result["format"] = mimetype
        
        # Analyze content if available and format not definitively determined
        if content and result["format"] == "text":
            # Check for JSON-like content
            if content.strip().startswith('{') and content.strip().endswith('}'):
                result["format"] = "json"
            elif content.strip().startswith('[') and content.strip().endswith(']'):
                result["format"] = "json"
            # Check for HTML-like content
            elif '<html' in content.lower() or '<!doctype html' in content.lower():
                result["format"] = "html"
                result["mimetype"] = "text"
                result["subtype"] = "html"
            # Check for XML-like content
            elif content.strip().startswith('<?xml'):
                result["format"] = "xml"
                result["mimetype"] = "text"
                result["subtype"] = "xml"
                
        return result
    
    def _parse_json_response(self, content, expected_format="list", required_fields=None):
        """
        Parse and validate JSON responses from APIs or LLMs.
        
        This function handles common JSON parsing issues and enforces format requirements.
        
        Args:
            content: String content to parse as JSON
            expected_format: Expected format of the parsed JSON ("list" or "dict")
            required_fields: List of required fields for dict items
            
        Returns:
            Parsed JSON object with validated format
            
        Raises:
            Exception: If parsing fails or format validation fails
        """
        try:
            # Clean up content that might have markdown code blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json", 1)[1]
            elif content.startswith("```"):
                content = content.split("```", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
                
            # Try to parse as JSON
            parsed_content = json.loads(content)
            
            # Validate expected format
            if expected_format == "list" and not isinstance(parsed_content, list):
                self.logger.warning(f"Expected list format but got {type(parsed_content).__name__}")
                # Try to adapt non-list response
                if isinstance(parsed_content, dict):
                    # Extract values if it's a dict
                    parsed_content = list(parsed_content.values())
                else:
                    # Wrap any other type in a list
                    parsed_content = [parsed_content]
            elif expected_format == "dict" and not isinstance(parsed_content, dict):
                self.logger.warning(f"Expected dict format but got {type(parsed_content).__name__}")
                # Try to adapt non-dict response
                if isinstance(parsed_content, list) and len(parsed_content) > 0:
                    # Use first item if it's a list of dicts
                    if isinstance(parsed_content[0], dict):
                        parsed_content = parsed_content[0]
                    else:
                        # Create a generic wrapper
                        parsed_content = {"content": parsed_content}
                else:
                    # Create a generic wrapper for other types
                    parsed_content = {"content": parsed_content}
            
            # Validate required fields if any are specified
            if required_fields and isinstance(parsed_content, list):
                for i, item in enumerate(parsed_content):
                    if not isinstance(item, dict):
                        self.logger.warning(f"Item {i} is not a dictionary: {item}")
                        parsed_content[i] = {field: "N/A" for field in required_fields}
                        parsed_content[i]["content"] = str(item)
                        continue
                    
                    # Check for missing fields
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        self.logger.warning(f"Item {i} is missing fields {missing_fields}: {item}")
                        # Add missing fields with default values
                        for field in missing_fields:
                            item[field] = "N/A"
            
            return parsed_content
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw content: {content}")
            
            # Return a fallback response based on expected format
            if expected_format == "list":
                if required_fields:
                    # Create a valid structure with required fields
                    return [{field: "N/A" for field in required_fields}]
                else:
                    return [{"content": content}]
            else:
                return {"content": content}
    
    def _get_llm_provider(self):
        """
        Get or create an LLM bridge instance.
        
        This helper function centralizes LLM bridge acquisition logic.
        
        Returns:
            LLMProvider instance
        """
        from clients.llm_provider import LLMProvider
        
        # Create or get LLM bridge instance
        try:
            # Create LLM provider instance
            llm_provider = LLMProvider()
        except (ImportError, AttributeError):
            # If not available, create a new instance
            llm_provider = LLMProvider()
            
        return llm_provider
    
        
    def _handle_webpage_extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle webpage extraction operations.
        
        This function extracts content from webpages using Claude's API.
        
        Args:
            params: Dictionary of parameters for the webpage extraction
            
        Returns:
            Dictionary containing the extracted content
            
        Raises:
            Exception: If inputs are invalid or if the extraction fails
        """
        # Extract expected parameters
        url = params.get("url")
        extraction_prompt = params.get("extraction_prompt")
        format_type = params.get("format", "text")
        include_metadata = params.get("include_metadata", False)
        timeout = params.get("timeout")

        # Validate required parameters
        if not url:
            _raise_invalid_input(
                "Required parameter 'url' must be provided for webpage_extract operation",
                {"provided_params": list(params.keys())}
            )

        self.logger.info(f"Extracting content from {url}")

        # Input validation
        self._validate_webpage_extract_inputs(url, format_type, timeout)

        # Set default timeout if not provided
        timeout = self._validate_timeout(timeout)

        # Set default extraction prompt if not provided
        if extraction_prompt is None:
            # Import config when needed (avoids circular imports)
            from config import config
            
            try:
                extraction_prompt = config.webaccess_tool.default_extraction_prompt
            except AttributeError:
                extraction_prompt = (
                    "Please extract the main content from this webpage. "
                    "Focus on the article text, headings, and important information. "
                    "Ignore navigation, ads, footers, and other non-essential elements."
                )

        # Fetch the webpage HTML
        html_content, response = self._fetch_webpage(url, timeout)

        # Check if we got an error
        if html_content is None:
            # response contains the error dict
            return {
                "success": False,
                "error": response["error"],
                "message": response["message"],
                "url": response.get("url", url),
                "details": {k: v for k, v in response.items() if k not in ["error", "message", "url"]}
            }

        if not html_content:
            return {
                "success": False,
                "url": url,
                "error": "empty_content",
                "message": "Failed to fetch webpage content or content was empty"
            }

        # Extract content from HTML using LLM
        extracted_content = self._extract_content_with_llm(
            html_content, 
            url, 
            extraction_prompt, 
            format_type
        )
        
        result = {
            "success": True,
            "url": url,
            "content": extracted_content
        }
        
        # Add metadata if requested
        if include_metadata:
            title = self._extract_title(html_content)
            metadata = self._extract_metadata(html_content, response)
            result["title"] = title
            result["metadata"] = metadata
            
        return result
            
    def _validate_webpage_extract_inputs(self, url, format_type, timeout):
        """
        Validate input parameters for webpage extraction.
        
        Args:
            url: URL string
            format_type: Format type string
            timeout: Timeout value in seconds
            
        Raises:
            Exception: If any inputs are invalid
        """
        # Validate URL with operation context for domain restrictions
        self._validate_url(url, operation="webpage_extract")
            
        # Validate format type
        valid_formats = ["text", "markdown", "html"]
        if format_type not in valid_formats:
            _raise_invalid_input(
                f"Invalid format type: {format_type}. Must be one of: {', '.join(valid_formats)}",
                {"provided_format": format_type, "valid_formats": valid_formats}
            )
            
        # Timeout validation is handled by the shared _validate_timeout method
    
    def _extract_content_with_llm(self, html_content, url, extraction_prompt, format_type):
        """
        Use Groq LLM to extract content from the HTML.

        Uses Groq gpt-oss-20b for ultra-fast, cost-effective content extraction.
        Groq provides 10x faster inference than Claude for this simple task.

        Args:
            html_content: The HTML content to extract from
            url: The URL of the webpage (for context)
            extraction_prompt: The prompt to guide the extraction
            format_type: The desired output format

        Returns:
            Extracted content as string

        Raises:
            Exception: If extraction fails
        """
        self.logger.debug("Extracting content with Groq LLM")

        # Get config and API key for Groq
        from config.config import config
        from clients.vault_client import get_api_key
        from clients.llm_provider import LLMProvider

        groq_api_key = get_api_key(config.api.execution_api_key_name)
        if not groq_api_key:
            raise RuntimeError(f"Groq API key '{config.api.execution_api_key_name}' not found in Vault")

        llm_provider = LLMProvider()

        # Construct the prompt
        format_instruction = ""
        if format_type == "markdown":
            format_instruction = "Format your output as Markdown to preserve the structure."
        elif format_type == "html":
            format_instruction = "Return a filtered, clean HTML that preserves the structure but removes unnecessary elements."

        # Construct specialized system prompt for extraction task
        system_prompt = f"""You are a content extraction assistant. Your sole task is to extract specific information from HTML.

USER REQUEST: {extraction_prompt}

EXTRACTION INSTRUCTIONS:
- Extract only the information requested by the user
- Ignore navigation, headers, footers, ads, sidebars, and decorative elements
- Return the extracted information clearly and concisely
- {format_instruction if format_instruction else "Return as plain text"}
- If the requested information is not found, state that clearly

SOURCE: {url}"""

        # Construct the user message with HTML content
        user_message = f"Extract the requested information from this HTML:\n\n```html\n{html_content}\n```"

        try:
            # Call Groq via LLMProvider with endpoint overrides
            response = llm_provider.generate_response(
                messages=[{"role": "user", "content": user_message}],
                stream=False,
                endpoint_url=config.api.execution_endpoint,
                model_override=config.api.execution_model,
                api_key_override=groq_api_key,
                system_override=system_prompt,
                temperature=0.1  # Low temperature for deterministic extraction
            )

            # Extract the text content
            extracted_content = llm_provider.extract_text_content(response)

            return extracted_content

        except Exception as e:
            self.logger.error(f"Error in LLM content extraction: {str(e)}")
            raise RuntimeError(f"LLM content extraction failed: {str(e)}") from e
    
    def _extract_title(self, html_content):
        """
        Extract the title from HTML content.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Title string or empty string if not found
        """
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return ""
    
    def _extract_metadata(self, html_content, response):
        """
        Extract basic metadata from HTML and response.
        
        Args:
            html_content: HTML content as string
            response: Requests response object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "content_type": response.headers.get('Content-Type', ''),
            "last_modified": response.headers.get('Last-Modified', ''),
            "size": len(html_content),
            "status_code": response.status_code,
            "final_url": response.url
        }
        
        # Extract meta tags
        meta_tags = {}
        description_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if description_match:
            meta_tags["description"] = description_match.group(1).strip()
            
        keywords_match = re.search(r'<meta\s+name=["\']keywords["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if keywords_match:
            meta_tags["keywords"] = keywords_match.group(1).strip()
            
        # Extract Open Graph metadata
        og_title_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if og_title_match:
            meta_tags["og:title"] = og_title_match.group(1).strip()
            
        og_description_match = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if og_description_match:
            meta_tags["og:description"] = og_description_match.group(1).strip()
            
        metadata["meta_tags"] = meta_tags
        
        return metadata
        
    def _validate_url(self, url: str, operation: str = None) -> None:
        """
        Validate URL format and security restrictions.
        
        Args:
            url: The URL to validate
            operation: Optional operation context for domain restrictions
            
        Raises:
            Exception: If the URL is invalid or restricted
        """
        # Validate URL format
        if not url or not isinstance(url, str):
            _raise_invalid_input(
                "URL must be a non-empty string",
                {"provided_url": str(url)}
            )
            
        # Check URL scheme
        parsed_url = urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme not in ["http", "https"]:
            _raise_invalid_input(
                f"Invalid URL scheme: {parsed_url.scheme}. Must be http or https",
                {"url": url, "scheme": parsed_url.scheme}
            )
            
        # Security check - validate against blocked URL patterns
        for pattern in self._blocked_url_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                _raise_invalid_input(
                    "URL is restricted for security reasons (internal/private network)",
                    {"url": url}
                )
                
        # Check domain restrictions if operation is provided
        if operation:
            # Get domain restrictions for this operation
            allowed_domains, blocked_domains = self._get_domain_restrictions(operation=operation)
            
            # Check if domain is explicitly blocked
            if blocked_domains and parsed_url.netloc:
                domain = parsed_url.netloc.lower()
                
                for blocked in blocked_domains:
                    if domain == blocked.lower() or domain.endswith('.' + blocked.lower()):
                        _raise_invalid_input(
                            f"Domain '{domain}' is blocked by configuration",
                            {"url": url, "domain": domain, "blocked_by": blocked}
                        )
            
            # Check if domain is allowed (when allowlist is active)
            if allowed_domains and parsed_url.netloc:
                domain = parsed_url.netloc.lower()
                is_allowed = False
                
                for allowed in allowed_domains:
                    if domain == allowed.lower() or domain.endswith('.' + allowed.lower()):
                        is_allowed = True
                        break
                        
                if not is_allowed:
                    _raise_invalid_input(
                        f"Domain '{domain}' is not in the allowed domains list",
                        {"url": url, "domain": domain, "allowed_domains": allowed_domains}
                    )
    
    def _validate_timeout(self, timeout: Optional[int]) -> int:
        """
        Validate and return the timeout value.
        
        Args:
            timeout: The timeout value to validate
            
        Returns:
            Validated timeout value
            
        Raises:
            Exception: If the timeout is invalid
        """
        from config import config
        
        # Get config
        try:
            tool_config = config.webaccess_tool
            default_timeout = tool_config.timeout
            max_timeout = tool_config.max_timeout
        except AttributeError:
            default_timeout = 30
            max_timeout = 120
            
        # Use default if not provided
        if timeout is None:
            return default_timeout
            
        # Validate timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            _raise_invalid_input(
                "Timeout must be a positive number",
                {"provided_timeout": timeout}
            )
            
        # Check against max timeout
        if timeout > max_timeout:
            _raise_invalid_input(
                f"Timeout value exceeds maximum allowed ({max_timeout} seconds)",
                {"provided_timeout": timeout, "max_timeout": max_timeout}
            )
            
        return timeout
        
    def _get_domain_restrictions(self, allowed_domains=None, blocked_domains=None, operation=None):
        """
        Get and merge domain restrictions from various sources.
        
        This function consolidates domain restrictions from:
        - Provided parameters
        - Tool-specific configuration
        - Global tool configuration
        
        Args:
            allowed_domains: Optional list of explicitly allowed domains
            blocked_domains: Optional list of explicitly blocked domains
            operation: Optional specific operation to get restrictions for
            
        Returns:
            Tuple of (final_allowed_domains, final_blocked_domains)
        """
        from config import config
        
        final_allowed = set()
        final_blocked = set()
        
        # First, add global restrictions from config
        try:
            if config.webaccess_tool.allowed_domains:
                final_allowed.update(config.webaccess_tool.allowed_domains)
                
            if config.webaccess_tool.blocked_domains:
                final_blocked.update(config.webaccess_tool.blocked_domains)
        except AttributeError:
            pass
            
        # Add operation-specific restrictions if specified
        if operation == "http_request":
            # No specific restrictions for HTTP requests beyond globals
            pass
        elif operation == "web_search":
            # Web search might have specific allowed/blocked domains
            try:
                if hasattr(config, 'web_search_tool') and config.web_search_tool.allowed_domains:
                    final_allowed.update(config.web_search_tool.allowed_domains)
                    
                if hasattr(config, 'web_search_tool') and config.web_search_tool.blocked_domains:
                    final_blocked.update(config.web_search_tool.blocked_domains)
            except AttributeError:
                pass
        elif operation == "webpage_extract":
            # Webpage extraction might have specific allowed domains
            try:
                if hasattr(config, 'webpage_extraction_tool') and config.webpage_extraction_tool.allowed_domains:
                    final_allowed.update(config.webpage_extraction_tool.allowed_domains)
            except AttributeError:
                pass
                
        # Finally, add explicitly provided domains (these take precedence)
        if allowed_domains:
            final_allowed.update(allowed_domains)
            
        if blocked_domains:
            final_blocked.update(blocked_domains)
            
        # Convert back to lists for return
        return list(final_allowed) if final_allowed else None, list(final_blocked) if final_blocked else None
                
    def _validate_domains(self, allowed_domains: Optional[List[str]], blocked_domains: Optional[List[str]]) -> None:
        """
        Validate domain restriction lists.
        
        Args:
            allowed_domains: List of allowed domains
            blocked_domains: List of blocked domains
            
        Raises:
            Exception: If domain lists are invalid or conflicting
        """
        # Validate allowed_domains
        if allowed_domains is not None:
            if not isinstance(allowed_domains, list):
                _raise_invalid_input(
                    "allowed_domains must be a list of strings",
                    {"provided_allowed_domains": allowed_domains}
                )
                
            for domain in allowed_domains:
                if not isinstance(domain, str):
                    _raise_invalid_input(
                        "Each allowed domain must be a string",
                        {"invalid_domain": domain}
                    )
        
        # Validate blocked_domains
        if blocked_domains is not None:
            if not isinstance(blocked_domains, list):
                _raise_invalid_input(
                    "blocked_domains must be a list of strings",
                    {"provided_blocked_domains": blocked_domains}
                )
                
            for domain in blocked_domains:
                if not isinstance(domain, str):
                    _raise_invalid_input(
                        "Each blocked domain must be a string",
                        {"invalid_domain": domain}
                    )
        
        # Check for conflict between allowed and blocked domains
        if allowed_domains and blocked_domains:
            overlap = set(allowed_domains).intersection(set(blocked_domains))
            if overlap:
                _raise_invalid_input(
                    "Domain cannot be both allowed and blocked",
                    {"conflicting_domains": list(overlap)}
                )
                
    def _clean_html_content(self, html_content: str) -> str:
        """
        Clean HTML content to reduce token count before LLM processing.

        Removes head, scripts, styles, and other non-content elements while preserving
        the main body content structure.

        Args:
            html_content: Raw HTML content

        Returns:
            Cleaned HTML string with reduced token count
        """
        if not BS4_AVAILABLE:
            self.logger.warning("BeautifulSoup not available, returning uncleaned HTML")
            return html_content

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove entire <head> section (scripts, styles, meta tags, etc.)
            if soup.head:
                soup.head.decompose()

            # Remove elements that don't contribute to content
            for element in soup.find_all(['script', 'style', 'noscript', 'iframe', 'object', 'embed']):
                element.decompose()

            # Remove HTML comments
            from bs4 import Comment
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Get the cleaned HTML
            cleaned_html = str(soup)

            # Log token reduction
            original_length = len(html_content)
            cleaned_length = len(cleaned_html)
            reduction_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
            self.logger.info(f"HTML cleaned: {original_length} -> {cleaned_length} chars ({reduction_pct:.1f}% reduction)")

            return cleaned_html

        except Exception as e:
            self.logger.warning(f"Failed to clean HTML: {e}, returning original content")
            return html_content

    def _fetch_webpage(self, url: str, timeout: int) -> tuple:
        """
        Fetch webpage content with JavaScript rendering, falling back to HTTP if unavailable.

        Attempts to use PlaywrightService for JavaScript-heavy pages. If Playwright/Chromium
        is not available, falls back to simple HTTP request (suitable for static content).

        Args:
            url: The URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Tuple of (html_content, mock_response_dict) or (None, error_dict) on error

        Raises:
            Exception: Only for validation errors
        """
        self.logger.debug(f"Fetching webpage: {url}")

        # Try Playwright first for full JavaScript rendering
        playwright_available = False
        try:
            from utils.playwright_service import PlaywrightService
            playwright_available = True
        except ImportError as e:
            self.logger.warning(f"Playwright not available, will use HTTP fallback: {e}")

        if playwright_available:
            try:
                playwright = PlaywrightService.get_instance()
                html = playwright.fetch_rendered_html(url, timeout=timeout)

                # Clean HTML to reduce token count
                cleaned_html = self._clean_html_content(html)

                # Create a materialized response object for compatibility
                mock_response = MaterializedResponse(
                    status_code=200,
                    url=url,
                    headers={'Content-Type': 'text/html; charset=utf-8'},
                    content=cleaned_html.encode('utf-8')
                )

                return cleaned_html, mock_response

            except TimeoutError as e:
                self.logger.error(f"Playwright timeout for {url}: {e}")
                return None, {
                    "error": "timeout",
                    "message": str(e),
                    "url": url,
                    "method": "GET"
                }
            except RuntimeError as e:
                # Check if this is a Playwright initialization error (missing browser binaries)
                error_str = str(e).lower()
                if 'playwright' in error_str and ('executable' in error_str or 'browser' in error_str or 'chromium' in error_str):
                    self.logger.warning(f"Chromium not installed, falling back to HTTP: {e}")
                    playwright_available = False  # Fall through to HTTP fallback
                else:
                    self.logger.error(f"Playwright error for {url}: {e}")
                    return None, {
                        "error": "playwright_error",
                        "message": str(e),
                        "url": url,
                        "method": "GET"
                    }
            except Exception as e:
                self.logger.error(f"Unexpected Playwright error for {url}: {e}")
                return None, {
                    "error": "request_error",
                    "message": f"Unexpected error: {str(e)}",
                    "url": url,
                    "method": "GET"
                }

        # Fallback to simple HTTP request (for static content)
        if not playwright_available:
            self.logger.info(f"Using HTTP fallback for {url} (JavaScript will not be executed)")
            response = self._make_http_request(
                method="GET",
                url=url,
                timeout=timeout,
                is_browser=True  # Use browser-like headers
            )

            # Check if we got an error response
            if isinstance(response, dict) and "error" in response:
                return None, response

            # Clean HTML to reduce token count
            cleaned_html = self._clean_html_content(response.text)

            # Return cleaned HTML content and response
            return cleaned_html, response