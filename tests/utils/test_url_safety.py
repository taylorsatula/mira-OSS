"""Tests for SSRF guards against IPv6 transition-address bypasses in url_safety."""

import ipaddress
import socket

import pytest

from utils.url_safety import _validate_public_ip, validate_public_http_url


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::a9fe:a9fe",  # NAT64 -> 169.254.169.254
        "64:ff9b::a00:1",  # NAT64 -> 10.0.0.1
        "64:ff9b::7f00:1",  # NAT64 -> 127.0.0.1
        "::c0a8:1",  # deprecated IPv4-compatible -> 192.168.0.1
        "::a9fe:a9fe",  # deprecated IPv4-compatible -> 169.254.169.254
    ],
)
def test_transition_addresses_embedding_private_ipv4_are_blocked(address: str) -> None:
    """GHSA-rmgf-f8wc-rc3p: is_global is True for these; explicit range rejection required."""
    assert ipaddress.ip_address(address).is_global is True
    with pytest.raises(ValueError, match="transition address"):
        _validate_public_ip(ipaddress.ip_address(address), "evil.example.com")


@pytest.mark.parametrize(
    "address",
    [
        "::ffff:10.0.0.1",  # IPv4-mapped
        "2002:a00:1::",  # 6to4 embedding 10.0.0.1
        "10.0.0.1",
        "127.0.0.1",
        "169.254.169.254",
    ],
)
def test_standard_private_addresses_still_blocked(address: str) -> None:
    with pytest.raises(ValueError):
        _validate_public_ip(ipaddress.ip_address(address), "example.com")


def test_public_addresses_still_pass() -> None:
    _validate_public_ip(ipaddress.ip_address("1.1.1.1"), "example.com")
    _validate_public_ip(ipaddress.ip_address("2606:4700:4700::1111"), "example.com")


def test_validate_public_http_url_accepts_transition_address_hostname() -> None:
    """Literal transition-address hostnames are rejected at the URL level."""
    with pytest.raises(ValueError):
        validate_public_http_url("http://[64:ff9b::a9fe:a9fe]/metadata")


def test_literal_ip_hostname_pins_to_itself() -> None:
    validated = validate_public_http_url("http://1.1.1.1/x")
    assert validated.resolved_ip == "1.1.1.1"


def test_resolved_hostname_pins_to_first_validated_address(monkeypatch) -> None:
    """resolved_ip must be an address that passed validation, for connection pinning."""
    from utils import url_safety

    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host == "example.com"
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1::1", 0, 0, 0)),
        ]

    original = socket.getaddrinfo
    socket.getaddrinfo = fake_getaddrinfo
    try:
        validated = url_safety.validate_public_http_url("http://example.com/x")
    finally:
        socket.getaddrinfo = original

    assert validated.resolved_ip == "93.184.216.34"
