"""Public HTTP URL validation for tool-controlled outbound requests."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


MAX_REDIRECT_HOPS = 5

# IPv6 transition ranges embedding an IPv4 address in the low 32 bits. CPython's
# `ipaddress.is_global` returns True for these, so they must be rejected explicitly:
# `64:ff9b::a9fe:a9fe` (NAT64, RFC 6052) embeds 169.254.169.254 and passes is_global.
_EMBEDDED_IPV4_V6_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),  # NAT64 well-known prefix (RFC 6052)
    ipaddress.ip_network("::/96"),  # deprecated IPv4-compatible (RFC 4291)
)

_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
    "169.254.169.254",
}


@dataclass(frozen=True)
class ValidatedURL:
    """Normalized URL components after public-network validation."""

    url: str
    hostname: str
    resolved_ip: str
    """Validated IP to pin the actual connection to.

    For literal-IP hostnames this is the literal itself; otherwise it is the
    first address returned by resolution. Callers MUST connect to this IP
    rather than letting the HTTP client re-resolve DNS, or the validation is
    vulnerable to DNS rebinding between validation and connection.
    """


def normalize_domain_pattern(pattern: str) -> str:
    """Normalize a caller-provided domain pattern to a bare lowercase hostname."""
    raw = pattern.strip().lower()
    if not raw:
        raise ValueError("Domain pattern cannot be empty")

    if "://" in raw:
        parsed = urlparse(raw)
        raw = parsed.hostname or ""
    else:
        raw = raw.split("/", 1)[0].split(":", 1)[0]

    if not raw:
        raise ValueError(f"Invalid domain pattern: {pattern}")
    return raw.rstrip(".")


def domain_matches(hostname: str, pattern: str) -> bool:
    """Return true when hostname is exactly pattern or a subdomain of it."""
    host = hostname.strip().lower().rstrip(".")
    normalized_pattern = normalize_domain_pattern(pattern)
    return host == normalized_pattern or host.endswith(f".{normalized_pattern}")


def validate_domain_filters(
    hostname: str,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> None:
    """Validate hostname against caller-supplied allow/block domain filters."""
    allowed = allowed_domains or []
    blocked = blocked_domains or []

    if allowed:
        if not any(domain_matches(hostname, pattern) for pattern in allowed):
            raise ValueError(f"Domain not in allowed list: {hostname}")
    elif blocked:
        if any(domain_matches(hostname, pattern) for pattern in blocked):
            raise ValueError(f"Domain blocked: {hostname}")


def validate_allowed_domains(allowed_domains: list[str]) -> list[str]:
    """Normalize and require at least one allowed domain for credential binding."""
    normalized = [normalize_domain_pattern(domain) for domain in allowed_domains]
    unique = sorted(set(normalized))
    if not unique:
        raise ValueError("At least one allowed domain is required")
    return unique


def validate_public_http_url(
    url: str,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> ValidatedURL:
    """Validate that a URL uses HTTP(S), targets a public host, and matches filters."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https scheme, got: {parsed.scheme}")
    if parsed.username or parsed.password:
        raise ValueError("URL userinfo is not allowed")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError("URL must include a hostname")

    resolved_ip = _validate_public_hostname(hostname)
    validate_domain_filters(hostname, allowed_domains, blocked_domains)

    return ValidatedURL(url=url, hostname=hostname, resolved_ip=resolved_ip)


def _validate_public_hostname(hostname: str) -> str:
    """Validate hostname and return the validated IP for connection pinning."""
    if hostname in _BLOCKED_HOSTNAMES or hostname.endswith(".localhost"):
        raise ValueError(f"Blocked private hostname: {hostname}")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return _validate_resolved_addresses(hostname)

    _validate_public_ip(ip, hostname)
    return str(ip)


def _validate_resolved_addresses(hostname: str) -> str:
    """Resolve and validate every returned address; return the first for pinning."""
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
        addrinfo = socket.getaddrinfo(ascii_hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from e

    if not addrinfo:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    resolved_ip: str | None = None
    for family, _, _, _, sockaddr in addrinfo:
        raw_ip = sockaddr[0]
        ip = ipaddress.ip_address(raw_ip)
        _validate_public_ip(ip, hostname)
        if resolved_ip is None:
            resolved_ip = str(ip)

    assert resolved_ip is not None
    return resolved_ip


def _validate_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, hostname: str) -> None:
    if isinstance(ip, ipaddress.IPv6Address) and any(ip in net for net in _EMBEDDED_IPV4_V6_NETWORKS):
        raise ValueError(f"Blocked IPv6 transition address embedding IPv4 for {hostname}: {ip}")
    if not ip.is_global:
        raise ValueError(f"Blocked non-public network address for {hostname}: {ip}")
