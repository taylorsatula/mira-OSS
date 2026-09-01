"""Tests for DNS-rebinding protection: connection pinning in http_client.

The SSRF guard in utils.url_safety resolves and validates DNS once; these tests
verify that the actual TCP connection uses that validated IP instead of letting
the HTTP client re-resolve (which would let an attacker rebind between the two).
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpcore
import pytest

from utils import http_client


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"pinned")

    def log_message(self, *args):  # silence request logging
        pass


@pytest.fixture()
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join()


def test_pinned_request_connects_to_validated_ip(local_server) -> None:
    """The hostname uses the .invalid TLD, which can never resolve. Success
    proves the TCP connection used the pinned IP rather than DNS."""
    response = http_client.pinned_request(
        "GET",
        f"http://mira-pinning-test.invalid:{local_server.server_address[1]}/",
        hostname="mira-pinning-test.invalid",
        ip="127.0.0.1",
    )
    assert response.status_code == 200
    assert response.text == "pinned"


def test_pinned_request_rejects_unvalidated_hostname(local_server) -> None:
    """A connection attempt for any hostname other than the pinned one fails."""
    with pytest.raises(http_client.ConnectError):
        http_client.pinned_request(
            "GET",
            f"http://other-host.invalid:{local_server.server_address[1]}/",
            hostname="mira-pinning-test.invalid",
            ip="127.0.0.1",
        )


def test_ip_pin_backend_blocks_other_hostnames(local_server) -> None:
    backend = http_client._IPPinNetworkBackend("pinned.invalid", "127.0.0.1")
    with pytest.raises(httpcore.ConnectError, match="unvalidated hostname"):
        backend.connect_tcp("evil.invalid", local_server.server_address[1])


def test_ip_pin_backend_accepts_trailing_dot_and_idna_variants(local_server) -> None:
    """httpx may hand httpcore a trailing-dot or IDNA-encoded host."""
    backend = http_client._IPPinNetworkBackend("pinned.invalid", "127.0.0.1")
    stream = backend.connect_tcp("pinned.invalid.", local_server.server_address[1])
    stream.close()

    backend = http_client._IPPinNetworkBackend("ünïcode.invalid", "127.0.0.1")
    stream = backend.connect_tcp("xn--ncode-cta3g.invalid", local_server.server_address[1])
    stream.close()
