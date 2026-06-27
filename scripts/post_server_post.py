"""Run post-server power-on self-test checks against a live MIRA service."""

from utils.power_on_self_test import post_server_cli


if __name__ == "__main__":
    raise SystemExit(post_server_cli())
