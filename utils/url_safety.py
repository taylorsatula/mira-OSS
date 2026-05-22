"""Public HTTP URL validation for tool-controlled outbound requests."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


MAX_REDIRECT_HOPS = 5

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

    _validate_public_hostname(hostname)
    validate_domain_filters(hostname, allowed_domains, blocked_domains)

    return ValidatedURL(url=url, hostname=hostname)


def _validate_public_hostname(hostname: str) -> None:
    if hostname in _BLOCKED_HOSTNAMES or hostname.endswith(".localhost"):
        raise ValueError(f"Blocked private hostname: {hostname}")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        _validate_resolved_addresses(hostname)
        return

    _validate_public_ip(ip, hostname)


def _validate_resolved_addresses(hostname: str) -> None:
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
        addrinfo = socket.getaddrinfo(ascii_hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from e

    if not addrinfo:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    for family, _, _, _, sockaddr in addrinfo:
        raw_ip = sockaddr[0]
        ip = ipaddress.ip_address(raw_ip)
        _validate_public_ip(ip, hostname)


def _validate_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, hostname: str) -> None:
    if not ip.is_global:
        raise ValueError(f"Blocked non-public network address for {hostname}: {ip}")
