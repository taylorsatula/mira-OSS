"""
Web tool providing search, fetch, and HTTP request capabilities.

Three operations:
- search: Query Kagi search API for current information
- fetch: Extract webpage content via trafilatura (HTTP first, Playwright escalation for JS-heavy pages)
- http: Make direct HTTP requests to APIs
"""
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Literal, Optional
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field, field_validator

from tools.repo import Tool
from tools.registry import registry
from utils import http_client
from utils.url_safety import (
    MAX_REDIRECT_HOPS,
    ValidatedURL,
    domain_matches,
    validate_domain_filters,
    validate_public_http_url,
)

try:
    from kagiapi import KagiClient
    KAGI_AVAILABLE = True
except ImportError:
    KAGI_AVAILABLE = False

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup, Comment
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False


# --- Configuration ---

class WebToolConfig(BaseModel):
    """Configuration for web_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    default_timeout: int = Field(default=30, description="Default timeout in seconds")
    max_timeout: int = Field(default=120, description="Maximum allowed timeout")
    # LLM config for synthesis of long pages (trafilatura text in, compressed text out)
    synthesis_model: str = Field(default="openai/gpt-oss-120b", description="Model for content synthesis")
    synthesis_endpoint: str = Field(default="https://api.groq.com/openai/v1/chat/completions", description="Synthesis LLM endpoint")
    synthesis_api_key_name: Optional[str] = Field(default="subcortical_key", description="Vault key name for API key")


registry.register("web_tool", WebToolConfig)


# --- Input Models ---

class SearchInput(BaseModel):
    """Input for web search operation."""
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return")
    allowed_domains: List[str] = Field(default_factory=list, description="Only include these domains")
    blocked_domains: List[str] = Field(default_factory=list, description="Exclude these domains")


class FetchInput(BaseModel):
    """Input for webpage fetch operation."""
    url: str = Field(..., description="URL to fetch")
    focus: Optional[str] = Field(default=None, description="What to look for on the page (used when content is long)")
    include_metadata: bool = Field(default=False, description="Include page metadata")
    timeout: Optional[int] = Field(default=None, ge=1, description="Request timeout in seconds")

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http or https scheme, got: {parsed.scheme}")
        return v


class HttpInput(BaseModel):
    """Input for HTTP request operation."""
    method: Literal["GET", "POST", "PUT", "DELETE"] = Field(..., description="HTTP method")
    url: str = Field(..., description="Request URL")
    query_params: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")
    headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP headers")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Form data")
    json_body: Optional[Dict[str, Any]] = Field(default=None, description="JSON body")
    timeout: Optional[int] = Field(default=None, ge=1, description="Request timeout")
    response_format: Literal["json", "text", "full"] = Field(default="json", description="Response format")
    allowed_domains: List[str] = Field(default_factory=list, description="Only allow these domains")
    blocked_domains: List[str] = Field(default_factory=list, description="Block these domains")
    # Credential injection - allows LLM to reference stored credentials by name without seeing actual values
    credential_name: Optional[str] = Field(
        default=None,
        description="Name of stored API credential to use for authentication"
    )
    credential_header: str = Field(
        default="Authorization",
        description="HTTP header name to inject the credential into"
    )
    credential_prefix: str = Field(
        default="Bearer ",
        description="Prefix to prepend to credential value (e.g., 'Bearer ', 'token ', '')"
    )

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http or https scheme, got: {parsed.scheme}")
        return v


@dataclass(frozen=True)
class ResolvedCredential:
    """Credential value plus the domains it is allowed to authenticate against."""

    value: str
    allowed_domains: List[str]


# --- Tool Implementation ---

class WebTool(Tool):
    """
    Web tool combining search, fetch, and HTTP capabilities.

    Operations:
    - search: Query Kagi for current web information
    - fetch: URL in, extracted text out (trafilatura; HTTP first, Playwright for JS-heavy pages)
    - http: Make direct HTTP requests to APIs
    """

    name = "web_tool"
    description = "Web search, fetch webpages, and HTTP requests"

    tool_schema = {
        "name": "web_tool",
        "description": "Search the web, fetch and extract webpage content, make HTTP requests to APIs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["search", "fetch", "http"],
                    "description": "'search' = web search, 'fetch' = extract webpage content (pass url, optionally focus), 'http' = direct HTTP API request"
                },
                "query": {"type": "string", "description": "Web search query. Only used with operation='search'. Min 1 character"},
                "max_results": {"type": "integer", "description": "Number of search results to return (1-20, default 5). Ignored unless operation='search'"},
                "url": {"type": "string", "description": "HTTP or HTTPS URL to request. Required for 'fetch' and 'http'. Private/internal network addresses are blocked"},
                "focus": {"type": "string", "description": "What to look for on the page. Guides content selection when the page is long. For 'fetch' only. Examples: 'pricing details', 'API rate limits', 'installation instructions'"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "description": "HTTP verb: GET, POST, PUT, or DELETE. Required for 'http' operation"},
                "query_params": {"type": "object", "description": "Key-value pairs appended as ?key=value query parameters. Only used with operation='http'"},
                "headers": {"type": "object", "description": "Custom HTTP request headers as {name: value} pairs. For 'http' operation. Do not set auth headers here — use credential_name instead"},
                "data": {"type": "object", "description": "Form-encoded request body (application/x-www-form-urlencoded). For 'http' only. Use json_body instead for JSON payloads"},
                "json_body": {"type": "object", "description": "JSON request body (application/json). For 'http' only. Use data instead for form-encoded payloads"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (min 1, max 120, default 30). For 'fetch' and 'http' operations"},
                "response_format": {"type": "string", "enum": ["json", "text", "full"], "description": "'json' (parse response as JSON, fall back to text), 'text' (raw response body), 'full' (body + sanitized headers + JSON if parseable). Default 'json'. For 'http' only"},
                "allowed_domains": {"type": "array", "items": {"type": "string"}, "description": "If set, only these domains are allowed (exact match or subdomain). Takes precedence over blocked_domains if both provided. For 'search' and 'http'"},
                "blocked_domains": {"type": "array", "items": {"type": "string"}, "description": "Domains to exclude (exact match or subdomain). Ignored if allowed_domains is also set. For 'search' and 'http'"},
                "include_metadata": {"type": "boolean", "description": "When true, adds 'title' and 'metadata' keys (content_type, byte size) to the fetch result. Default false"},
                "credential_name": {"type": "string", "description": "Name of a user-stored API credential to inject as an auth header. The credential value is retrieved server-side and never returned to you. For 'http' only"}
            },
            "required": ["operation"]
        }
    }

    # Response headers that should NEVER be returned to LLM (security)
    _SENSITIVE_RESPONSE_HEADERS = {
        'authorization', 'x-api-key', 'api-key', 'x-auth-token',
        'cookie', 'set-cookie', 'www-authenticate', 'proxy-authorization',
    }

    def __init__(self):
        super().__init__()
        self._kagi: Optional[KagiClient] = None
        self._init_kagi()

    def run(self, **params) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = params.pop("operation", None)
        if not operation:
            raise ValueError("Required parameter 'operation' not provided")

        if operation == "search":
            return self._search(SearchInput(**params))
        elif operation == "fetch":
            return self._fetch(FetchInput(**params))
        elif operation == "http":
            return self._http(HttpInput(**params))
        else:
            raise ValueError(f"Unknown operation: {operation}. Must be: search, fetch, or http")

    # --- Search Operation ---

    def _search(self, input: SearchInput) -> Dict[str, Any]:
        """Execute web search with Kagi primary, DuckDuckGo fallback."""
        # Try Kagi first if available
        if self._kagi:
            try:
                response = self._kagi.search(input.query, limit=input.max_results)
                results = []
                for item in response.get("data", []):
                    url = item.get("url", "")
                    if self._should_include_url(url, input.allowed_domains, input.blocked_domains):
                        results.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("snippet", "")
                        })
                return {"success": True, "results": results, "provider": "kagi"}
            except Exception as e:
                self.logger.warning(f"Kagi search failed: {e}, falling back to DuckDuckGo")

        # Fall back to DuckDuckGo
        if not DDGS_AVAILABLE:
            raise ValueError(
                "Web search requires either Kagi API key or DuckDuckGo. "
                "Install ddgs: pip install ddgs"
            )

        try:
            ddgs = DDGS()
            raw_results = ddgs.text(
                input.query,
                max_results=input.max_results,
                backend="auto"
            )

            if not raw_results:
                self.logger.warning(f"DuckDuckGo returned no results for query: {input.query}")

            results = []
            for item in raw_results:
                url = item.get("href", "")
                if self._should_include_url(url, input.allowed_domains, input.blocked_domains):
                    results.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("body", "")
                    })

            return {"success": True, "results": results, "provider": "duckduckgo"}
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            raise ValueError(f"DuckDuckGo search failed: {e}")

    # --- Fetch Operation ---

    # Minimum extracted text length before escalating to Playwright.
    # Below this threshold, the page likely needs JS rendering (SPA shell, JS loader).
    _MIN_EXTRACT_LENGTH = 150

    # Below this, return trafilatura output as-is (faithful dump).
    # Above this, send to LLM for intelligent synthesis.
    _MAX_FAITHFUL_LENGTH = 5000

    # Absolute cap on LLM input to stay within model context.
    _MAX_SYNTHESIS_INPUT = 80000

    _SYNTHESIS_PROMPT = (
        "You are reading the extracted text of a webpage. "
        "Select and return the most important content. "
        "Preserve: names, dates, numbers, prices, URLs, code, key facts. "
        "Drop: repetitive content, boilerplate, low-value sections. "
        "No summaries. No commentary. Return the actual content, condensed."
    )

    def _fetch(self, input: FetchInput) -> Dict[str, Any]:
        """Fetch webpage and extract content. HTTP+trafilatura first, Playwright escalation."""
        self._validate_url(input.url)
        timeout = self._get_timeout(input.timeout)

        # Tier 1: HTTP GET + trafilatura
        html, response_info = self._fetch_http(input.url, timeout)
        if html is not None:
            extracted = self._extract_content(html)
            if self._extraction_sufficient(extracted):
                return self._build_fetch_result(input, html, extracted, response_info)

        # Tier 2: Playwright + trafilatura
        pw_html, pw_info = self._fetch_playwright(input.url, timeout)
        if pw_html is not None:
            extracted = self._extract_content(pw_html)
            if extracted:
                return self._build_fetch_result(input, pw_html, extracted, pw_info)

            # Tier 3: Playwright HTML + BeautifulSoup fallback
            extracted = self._fallback_text_extract(pw_html)
            if extracted:
                return self._build_fetch_result(input, pw_html, extracted, pw_info)

        # All tiers failed — report the most relevant error
        error_info = pw_info or response_info or {}
        if "error" in error_info:
            return {"success": False, "url": input.url, **error_info}
        return {"success": False, "url": input.url, "error": "extraction_failed",
                "message": "Could not extract content from page"}

    def _build_fetch_result(self, input: FetchInput, html: str, content: str,
                            response_info: dict) -> Dict[str, Any]:
        """Build the standard fetch result dict. Long pages or focused queries get LLM synthesis."""
        if input.focus or len(content) > self._MAX_FAITHFUL_LENGTH:
            synthesized = self._synthesize_content(content, input.url, input.focus)
            if synthesized:
                content = synthesized + "\n\n[Page content was condensed. Re-fetch with focus='...' to target specific content]"
            else:
                # Synthesis failed — hard truncate as fallback
                content = content[:self._MAX_FAITHFUL_LENGTH] + "\n\n[Truncated — full page at source]"
        result = {"success": True, "url": input.url, "content": content}
        if input.include_metadata:
            result["title"] = self._extract_title(html)
            result["metadata"] = {
                "content_type": response_info.get("content_type", "text/html"),
                "size": len(html),
            }
        return result

    def _extraction_sufficient(self, extracted: Optional[str]) -> bool:
        """Check if extraction produced enough content to skip Playwright."""
        if not extracted:
            return False
        if len(extracted) < self._MIN_EXTRACT_LENGTH:
            self.logger.info(
                f"Extraction too short ({len(extracted)} chars) — escalating to Playwright"
            )
            return False
        return True

    def _fetch_http(self, url: str, timeout: int) -> tuple:
        """Fetch page via HTTP GET. Returns (html, info_dict) or (None, info_dict)."""
        try:
            response = self._request_with_validated_redirects(
                "GET",
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MIRA/1.0)"},
            )
            response.raise_for_status()
            return response.text, {"content_type": response.headers.get("Content-Type", "text/html")}
        except http_client.TimeoutException:
            return None, {"error": "timeout", "message": f"Request timed out after {timeout}s"}
        except http_client.HTTPStatusError as e:
            return None, {"error": "http_error", "message": f"HTTP {e.response.status_code}"}
        except Exception as e:
            return None, {"error": "request_error", "message": str(e)}

    def _fetch_playwright(self, url: str, timeout: int) -> tuple:
        """Fetch page via Playwright for JS rendering. Returns (html, info_dict) or (None, info_dict)."""
        try:
            from utils.playwright_service import PlaywrightService
            playwright = PlaywrightService.get_instance()
            html = playwright.fetch_rendered_html(url, timeout=timeout)
            return html, {"content_type": "text/html"}
        except ImportError:
            self.logger.info("Playwright not available, cannot escalate")
            return None, {"error": "playwright_unavailable", "message": "Playwright not installed"}
        except RuntimeError as e:
            if "chromium" in str(e).lower() or "executable" in str(e).lower():
                self.logger.info("Chromium not installed, cannot escalate")
                return None, {"error": "chromium_unavailable", "message": "Chromium not installed"}
            return None, {"error": "playwright_error", "message": str(e)}
        except TimeoutError as e:
            return None, {"error": "timeout", "message": str(e)}
        except Exception as e:
            self.logger.warning(f"Playwright failed: {e}")
            return None, {"error": "playwright_error", "message": str(e)}

    def _extract_content(self, html: str) -> Optional[str]:
        """Extract main content from HTML using trafilatura."""
        if not TRAFILATURA_AVAILABLE:
            self.logger.warning("trafilatura not available, skipping extraction")
            return None
        try:
            return trafilatura.extract(
                html,
                include_links=True,
                include_tables=True,
                include_comments=False,
                output_format="txt",
            )
        except Exception as e:
            self.logger.warning(f"trafilatura extraction failed: {e}")
            return None

    def _fallback_text_extract(self, html: str) -> Optional[str]:
        """Last-resort extraction: strip tags, return plain text via BeautifulSoup."""
        if not BS4_AVAILABLE:
            self.logger.warning("BeautifulSoup not available for fallback extraction")
            return None
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(["script", "style", "noscript", "iframe", "object", "embed"]):
                tag.decompose()
            for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
                comment.extract()
            text = soup.get_text(separator='\n', strip=True)
            return text if text else None
        except Exception as e:
            self.logger.warning(f"Fallback text extraction failed: {e}")
            return None

    def _synthesize_content(self, text: str, url: str, focus: Optional[str] = None) -> Optional[str]:
        """Compress long page content via LLM. Receives clean text, returns condensed text."""
        from config import config
        from clients.vault_client import get_api_key
        from clients.llm_provider import LLMProvider

        tool_config = config.web_tool

        if tool_config.synthesis_api_key_name:
            api_key = get_api_key(tool_config.synthesis_api_key_name)
            if not api_key:
                self.logger.warning(f"API key '{tool_config.synthesis_api_key_name}' not found — skipping synthesis")
                return None
        else:
            api_key = None

        if len(text) > self._MAX_SYNTHESIS_INPUT:
            text = text[:self._MAX_SYNTHESIS_INPUT]
            self.logger.info(f"Truncated synthesis input to {self._MAX_SYNTHESIS_INPUT} chars")

        # Build user message: directive fenced above the content firehose
        if focus:
            user_message = (
                f"\U0001F6A8 FOCUS: {focus} \U0001F6A8\n"
                f"{'=' * 60}\n\n"
                f"{text}"
            )
        else:
            user_message = text

        try:
            llm = LLMProvider()
            response = llm.generate_response(
                messages=[{"role": "user", "content": user_message}],
                endpoint_url=tool_config.synthesis_endpoint,
                dialect_name="groq",
                model=tool_config.synthesis_model,
                api_key=api_key,
                max_tokens=2048,
                system_prompt=f"{self._SYNTHESIS_PROMPT}\n\nSource: {url}"
            )
            return llm.extract_text_content(response)
        except Exception as e:
            self.logger.warning(f"Content synthesis failed: {e}")
            return None

    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    # --- HTTP Operation ---

    def _get_credential(self, credential_name: str, request_host: str) -> ResolvedCredential:
        """
        Retrieve credential by name from UserCredentialService.

        The LLM specifies credentials by name; this method retrieves the actual
        value. The value is NEVER returned to the LLM - it's only used for
        HTTP header injection.

        Raises:
            ValueError: If credential does not exist
        """
        from utils.user_credentials import UserCredentialService

        credential_service = UserCredentialService()
        metadata = credential_service.get_credential_metadata(
            credential_type="api_key",
            service_name=credential_name
        )
        allowed_domains = metadata.get("allowed_domains") or []
        if not allowed_domains:
            raise ValueError(
                f"Credential '{credential_name}' is not bound to any allowed domains. "
                f"Re-save it in Settings with explicit allowed domains before using it."
            )
        validate_domain_filters(request_host, allowed_domains=allowed_domains)

        credential_value = credential_service.get_credential(
            credential_type="api_key",
            service_name=credential_name
        )

        if credential_value is None:
            raise ValueError(
                f"Credential '{credential_name}' not found. "
                f"Add it in Settings > API Credentials."
            )
        return ResolvedCredential(value=credential_value, allowed_domains=allowed_domains)

    def _http(self, input: HttpInput) -> Dict[str, Any]:
        """Make HTTP request with optional credential injection."""
        validated = self._validate_url(input.url, input.allowed_domains, input.blocked_domains)
        timeout = self._get_timeout(input.timeout)

        # Build headers dict (copy to avoid mutating input)
        headers = dict(input.headers) if input.headers else {}
        injected_credential_header = None
        credential_allowed_domains = None

        # Credential injection - LLM references by name, we inject actual value
        if input.credential_name:
            credential = self._get_credential(input.credential_name, validated.hostname)
            headers[input.credential_header] = f"{input.credential_prefix}{credential.value}"
            injected_credential_header = input.credential_header
            credential_allowed_domains = credential.allowed_domains
            self.logger.info(f"Injected credential '{input.credential_name}' into '{input.credential_header}'")

        # Build kwargs for the request
        kwargs = {
            "params": input.query_params,
            "headers": headers,
            "timeout": timeout,
            "follow_redirects": True
        }
        if input.data:
            kwargs["data"] = input.data
        if input.json_body:
            kwargs["json"] = input.json_body

        try:
            response = self._request_with_validated_redirects(
                input.method,
                input.url,
                allowed_domains=input.allowed_domains,
                blocked_domains=input.blocked_domains,
                credential_allowed_domains=credential_allowed_domains,
                **kwargs
            )
        except http_client.TimeoutException:
            return {"success": False, "error": "timeout", "message": f"Timed out after {timeout}s"}
        except http_client.ConnectError as e:
            return {"success": False, "error": "connection_error", "message": str(e)}
        except http_client.HTTPStatusError as e:
            return {"success": False, "error": "http_error", "status_code": e.response.status_code, "message": str(e)}

        return self._format_http_response(response, input.response_format, injected_credential_header)

    def _format_http_response(
        self, response, format_type: str, injected_credential_header: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format HTTP response based on requested format.

        Security: Sanitizes response headers to prevent credential leakage.
        Some APIs echo authentication headers back in responses - we strip
        those to ensure the LLM never sees credential values.
        """
        result = {
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": str(response.url)
        }

        if format_type == "json":
            try:
                result["data"] = response.json()
            except ValueError:
                result["data"] = response.text
                result["warning"] = "Response is not valid JSON"
        elif format_type == "text":
            result["data"] = response.text
        elif format_type == "full":
            result["data"] = response.text
            # Sanitize headers - remove sensitive ones to prevent credential leakage
            sanitized_headers = {
                k: v for k, v in response.headers.items()
                if k.lower() not in self._SENSITIVE_RESPONSE_HEADERS
                and not (injected_credential_header and k.lower() == injected_credential_header.lower())
            }
            result["headers"] = sanitized_headers
            try:
                result["json"] = response.json()
            except ValueError:
                pass

        return result

    # --- Helpers ---

    def _init_kagi(self) -> None:
        """Initialize Kagi client from vault."""
        if not KAGI_AVAILABLE:
            self.logger.info("Kagi library not available, will use DuckDuckGo for search")
            return

        try:
            from clients.vault_client import get_api_key
            api_key = get_api_key("kagi_api_key")
            if api_key:
                self._kagi = KagiClient(api_key)
                self.logger.info("Kagi client initialized")
            else:
                self.logger.info("Kagi API key not found, will use DuckDuckGo for search")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Kagi: {e}, will use DuckDuckGo for search")

    def _get_timeout(self, timeout: Optional[int]) -> int:
        """Get validated timeout value."""
        from config import config
        try:
            default = config.web_tool.default_timeout
            max_timeout = config.web_tool.max_timeout
        except AttributeError:
            default, max_timeout = 30, 120

        if timeout is None:
            return default
        return min(timeout, max_timeout)

    def _validate_url(
        self,
        url: str,
        allowed: Optional[List[str]] = None,
        blocked: Optional[List[str]] = None,
    ) -> ValidatedURL:
        """Validate URL against SSRF protection and domain restrictions."""
        return validate_public_http_url(url, allowed, blocked)

    def _domain_matches(self, domain: str, pattern: str) -> bool:
        """Check if domain matches pattern (exact or subdomain)."""
        return domain_matches(domain, pattern)

    def _request_with_validated_redirects(
        self,
        method: str,
        url: str,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        credential_allowed_domains: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Issue an HTTP request while validating every redirect hop."""
        current_url = url
        current_method = method
        body_kwargs = {
            key: kwargs.get(key)
            for key in ("data", "json")
            if key in kwargs
        }

        for _ in range(MAX_REDIRECT_HOPS + 1):
            validated = self._validate_url(current_url, allowed_domains, blocked_domains)
            if credential_allowed_domains:
                validate_domain_filters(validated.hostname, allowed_domains=credential_allowed_domains)
            request_kwargs = dict(kwargs)
            request_kwargs["follow_redirects"] = False
            if current_method == "GET":
                request_kwargs.pop("data", None)
                request_kwargs.pop("json", None)
            else:
                request_kwargs.update(body_kwargs)

            response = {
                "GET": http_client.get,
                "POST": http_client.post,
                "PUT": http_client.put,
                "DELETE": http_client.delete,
            }[current_method](current_url, **request_kwargs)

            if response.status_code not in {301, 302, 303, 307, 308}:
                return response

            location = response.headers.get("Location")
            if not location:
                return response
            current_url = urljoin(str(response.url), location)
            if response.status_code in {301, 302, 303}:
                current_method = "GET"

        raise ValueError(f"Too many redirects after {MAX_REDIRECT_HOPS} hops")

    def _should_include_url(self, url: str, allowed: List[str], blocked: List[str]) -> bool:
        """Check if URL should be included in search results."""
        if not url:
            return False
        try:
            domain = (urlparse(url).hostname or "").lower()
            if not domain:
                return False
            if allowed:
                return any(self._domain_matches(domain, d) for d in allowed)
            if blocked:
                return not any(self._domain_matches(domain, d) for d in blocked)
            return True
        except Exception:
            return False
