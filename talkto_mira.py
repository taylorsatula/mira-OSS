#!/usr/bin/env python3
"""
MIRA CLI Chat - Simple command-line interface for chatting with MIRA.

Interactive mode automatically starts the MIRA API server in the background if
not already running, then presents a chat interface. The server is cleanly
shutdown when the chat session ends.

One-shot mode (--headless) requires the server to already be running and outputs
only the response for clean stdout usage in scripts.

Usage:
    python talkto_mira.py                     # Interactive chat (auto-starts server)
    python talkto_mira.py --headless "Question" # One-shot query (requires running server)

Prerequisites:
    - API token stored in Vault at mira/api_keys/mira_api
    - VAULT_ADDR, VAULT_ROLE_ID, VAULT_SECRET_ID environment variables set
"""

import argparse
import atexit
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import readline for input editing support (arrow keys, history)
try:
    import readline  # noqa: F401 - imported for side effects
except ImportError:
    # readline not available on Windows, fallback to basic input
    pass

import requests

from clients.vault_client import get_api_key

# Test user constants (from test fixtures)
TEST_USER_1_EMAIL = "test@example.com"
TEST_USER_1_ID = "443a898d-ed56-495a-b9de-0551c80169fe"
TEST_USER_2_EMAIL = "test2@example.com"
TEST_USER_2_ID = "7b8e4c2a-f3d1-4a5b-9e6c-1d2f3a4b5c6d"

# Configuration
MIRA_API_URL = os.getenv("MIRA_API_URL", "http://localhost:1993")
REQUEST_TIMEOUT = 120  # seconds
SERVER_STARTUP_TIMEOUT = 30  # seconds to wait for server to start

# Global server process tracker
_server_process = None

# ANSI color codes for labels
# Wrapped in \001 and \002 to tell readline these are non-printing characters (for input prompts)
CYAN_PROMPT = "\001\033[36m\002"
GREEN_PROMPT = "\001\033[32m\002"
RESET_PROMPT = "\001\033[0m\002"

# Plain ANSI codes for print statements
CYAN = "\033[36m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RED = "\033[31m"
RESET = "\033[0m"


class LoadingIndicator:
    """Scanning loading animation for clean mode."""

    def __init__(self, centered: bool = False):
        self.stop_event = threading.Event()
        self.thread = None
        self.centered = centered

    def _animate(self):
        """Animation loop that scans through LOADING text."""
        base = "LOADING"
        frames = []

        # Create frame with all letters visible
        full_frame_parts = ["-"]
        for letter in base:
            full_frame_parts.extend([letter, "-"])
        frames.append("".join(full_frame_parts))

        # Create frames where each letter gets hidden in sequence
        for hide_idx in range(len(base)):
            frame_parts = ["-"]
            for i, letter in enumerate(base):
                if i == hide_idx:
                    frame_parts.append("--")  # Creates --- with previous dash
                else:
                    frame_parts.extend([letter, "-"])
            frames.append("".join(frame_parts))

        idx = 0
        while not self.stop_event.is_set():
            frame = frames[idx]
            if self.centered:
                # Center the frame
                centered_frame = center_text(frame)
                print(f"\r{centered_frame}", end="", flush=True)
            else:
                print(f"\r{frame}", end="", flush=True)
            idx = (idx + 1) % len(frames)
            time.sleep(0.15)

    def start(self):
        """Start the animation."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the animation and clear the line."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.centered:
            width, _ = get_terminal_size()
            print("\r" + " " * width, end="")  # Clear the entire line
        else:
            print("\r" + " " * 20, end="")  # Clear the line
        print("\r", end="")


def strip_emotion_tag(text: str) -> str:
    """Remove MIRA's emotion tag and preceding newline if present."""
    # Pattern matches optional newline + emotion tag
    pattern = r'\n?<mira:my_emotion>.*?</mira:my_emotion>'
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


def send_message(token: str, message: str) -> dict:
    """Send message to MIRA API and return response."""
    url = f"{MIRA_API_URL}/v0/api/chat"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"message": message},
            timeout=REQUEST_TIMEOUT
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": {"message": "Request timed out after 120 seconds"}
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": {"message": f"Cannot connect to MIRA server at {MIRA_API_URL}"}
        }
    except Exception as e:
        return {
            "success": False,
            "error": {"message": f"Unexpected error: {str(e)}"}
        }


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_terminal_size() -> tuple[int, int]:
    """Get terminal dimensions.

    Returns:
        tuple[int, int]: (width, height) of terminal
    """
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        return 80, 24  # Fallback dimensions


def center_text(text: str, width: int = None) -> str:
    """Center text horizontally in terminal.

    Args:
        text: Text to center
        width: Terminal width (auto-detect if None)

    Returns:
        str: Centered text with padding
    """
    if width is None:
        width, _ = get_terminal_size()

    text_width = len(text)
    if text_width >= width:
        return text

    padding = (width - text_width) // 2
    return " " * padding + text


def print_centered(text: str, color: str = "") -> None:
    """Print text centered in terminal.

    Args:
        text: Text to print
        color: Optional ANSI color code
    """
    centered = center_text(text)
    if color:
        print(f"{color}{centered}{RESET}")
    else:
        print(centered)


def move_cursor_to_center(text_lines: int = 1) -> None:
    """Move cursor to vertical center of screen.

    Args:
        text_lines: Number of text lines to account for
    """
    _, height = get_terminal_size()
    vertical_padding = (height - text_lines) // 2
    print("\n" * vertical_padding, end="")


def show_boot_screen() -> None:
    """Display MIRA boot screen for 2 seconds.

    Skips boot screen if terminal width is less than 80 characters.
    """
    width, _ = get_terminal_size()

    # Skip boot screen if terminal is too narrow for ASCII art
    if width < 80:
        clear_screen()
        return

    clear_screen()

    # ASCII art
    logo = [
        "                                               @@@@@@@@@@@                      ",
        "@@@@@@@        @@        @@@@@@         @      @@@@@@@@@@@    @@@@@@      @@@@@@",
        "@@ @  @        @@        @@@@@@        @ @     @@@@@@@@@@@    @    @      @@@@@@",
        "@@ @  @        @@        @   @        @ @@@    @@@@@@@@@@@    @@@@@@      @@@@@@",
        "                                               @@@@@@@@@@@                       "
    ]

    # Center vertically
    move_cursor_to_center(len(logo))

    # Print each line centered horizontally
    for line in logo:
        print_centered(line, CYAN)

    time.sleep(2)
    clear_screen()


def get_separator() -> str:
    """Get a separator line that spans the terminal width.

    Returns:
        str: Separator with newlines above and below
    """
    width, _ = get_terminal_size()
    return f"\n{'=' * width}\n"


def is_api_running() -> bool:
    """Check if the MIRA API is already running.

    Returns:
        bool: True if API is reachable and healthy, False otherwise
    """
    try:
        response = requests.get(f"{MIRA_API_URL}/v0/api/health", timeout=2)
        # Check for 200 OK or 503 (unhealthy but running)
        return response.status_code in [200, 503]
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def start_api_server() -> subprocess.Popen:
    """Start the MIRA API server as a background process.

    Returns:
        subprocess.Popen: The server process

    Raises:
        RuntimeError: If server fails to start
    """
    global _server_process

    # Use the project root's main.py
    main_py = project_root / "main.py"

    if not main_py.exists():
        raise RuntimeError(f"Cannot find main.py at {main_py}")

    # Start the server process with output suppressed
    _server_process = subprocess.Popen(
        [sys.executable, str(main_py)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(project_root)
    )

    return _server_process


def wait_for_api_ready(timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Wait for the API server to be ready.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        bool: True if server became ready, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_api_running():
            return True
        time.sleep(0.5)

    return False


def shutdown_server():
    """Shutdown the API server if it was started by this script."""
    global _server_process

    if _server_process is not None:
        try:
            _server_process.terminate()
            try:
                _server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _server_process.kill()
                _server_process.wait()
        except Exception:
            pass  # Already terminated
        finally:
            _server_process = None


def select_user() -> dict:
    """Prompt user to select which MIRA user to chat as.

    Returns:
        dict with 'name' and 'email' keys
    """
    clear_screen()
    print(f"{CYAN}═══════════════════════════════════{RESET}")
    print(f"{BOLD}     Select User{RESET}")
    print(f"{CYAN}═══════════════════════════════════{RESET}\n")

    print(f"  {GREEN}1.{RESET} Main User")
    print(f"  {GREEN}2.{RESET} Test User 1 ({TEST_USER_1_EMAIL})")
    print(f"  {GREEN}3.{RESET} Test User 2 ({TEST_USER_2_EMAIL})")
    print()

    while True:
        try:
            choice = input(f"{CYAN}Select user (1-3):{RESET} ").strip()
            if choice == "1":
                return {"name": "Main User", "email": "main"}
            elif choice == "2":
                return {"name": "Test User 1", "email": TEST_USER_1_EMAIL}
            elif choice == "3":
                return {"name": "Test User 2", "email": TEST_USER_2_EMAIL}
            else:
                print(f"  {RED}Invalid choice. Please enter 1, 2, or 3.{RESET}")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)


def get_token_for_user(user_selection: dict) -> str:
    """Get API key from Vault for single-user OSS mode.

    Args:
        user_selection: User selection dict (unused in single-user mode)

    Returns:
        str: API key from Vault

    Raises:
        RuntimeError: If API key cannot be retrieved
    """
    try:
        api_key = get_api_key('mira_api')
        if not api_key:
            raise RuntimeError("API key 'mira_api' not found in Vault")
        return api_key
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve API key from Vault: {e}")


def one_shot(token: str, message: str) -> None:
    """Send a single message and print the response.

    Args:
        token: API authentication token
        message: Message to send to MIRA
    """
    result = send_message(token, message)

    if result.get("success"):
        mira_response = result.get("data", {}).get("response", "")
        clean_response = strip_emotion_tag(mira_response)
        print(clean_response)
    else:
        error = result.get("error", {})
        error_message = error.get("message", "Unknown error")
        print(f"Error: {error_message}", file=sys.stderr)
        sys.exit(1)


def chat_loop(token: str) -> None:
    """Run interactive chat loop in clean mode.

    Args:
        token: API authentication token
    """
    # Track previous exchange
    prev_user_message = None
    prev_mira_response = None

    while True:
        try:
            # Get user input
            user_message = input(f"{GREEN_PROMPT}User:{RESET_PROMPT} ").strip()

            # Check for exit commands
            if user_message.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break

            # Skip empty messages
            if not user_message:
                continue

            # Show previous exchange with waiting indicator
            clear_screen()
            if prev_user_message and prev_mira_response:
                # Show previous exchange
                print(f"{GREEN}User:{RESET} {prev_user_message}")
                print(get_separator(), end="")
                print(f"{CYAN}MIRA:{RESET} {prev_mira_response}\n")
            # Show current message being processed
            print(f"{GREEN}User:{RESET} [message sent...]")

            # Send message with loading animation
            indicator = LoadingIndicator()
            indicator.start()
            result = send_message(token, user_message)
            indicator.stop()

            # Handle response
            if result.get("success"):
                mira_response = result.get("data", {}).get("response", "")
                clean_response = strip_emotion_tag(mira_response)

                # Clear and show current exchange
                clear_screen()
                print(f"{GREEN}User:{RESET} {user_message}")
                print(get_separator(), end="")
                print(f"{CYAN}MIRA:{RESET} {clean_response}\n")

                # Store this exchange for next iteration
                prev_user_message = user_message
                prev_mira_response = clean_response
            else:
                error = result.get("error", {})
                error_message = error.get("message", "Unknown error")

                # Handle specific error cases
                if "401" in str(error) or "Authentication" in error_message:
                    print(f"\n✗ {error_message}")
                    print("Your token may be invalid or expired.")
                    break
                elif "429" in str(error):
                    print(f"\n✗ Rate limit exceeded. Please wait before sending more messages.")
                else:
                    print(f"\n✗ Error: {error_message}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="MIRA CLI Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python talkto_mira.py                     # Interactive chat (auto-starts server)
  python talkto_mira.py --headless "Hello"    # One-shot query (requires running server)
        """
    )
    parser.add_argument(
        '--headless',
        type=str,
        help="Send a single message and exit (server must already be running)"
    )
    args = parser.parse_args()

    # Auto-start server only for interactive mode
    server_started = False
    if not args.headless:
        # Interactive mode - show boot screen
        show_boot_screen()

        # Start server if needed
        if not is_api_running():
            # Center the startup message
            move_cursor_to_center(2)  # Account for message + loading indicator
            print_centered("Starting MIRA API server...", CYAN)

            try:
                start_api_server()
                server_started = True

                # Show centered loading indicator while waiting for server
                indicator = LoadingIndicator(centered=True)
                indicator.start()

                if not wait_for_api_ready():
                    indicator.stop()
                    clear_screen()
                    move_cursor_to_center(1)
                    print_centered(f"✗ Server failed to start within {SERVER_STARTUP_TIMEOUT} seconds", "")
                    shutdown_server()
                    sys.exit(1)

                indicator.stop()
                clear_screen()
                move_cursor_to_center(1)
                print_centered(f"✓ MIRA API server ready", GREEN)
                time.sleep(1)
                clear_screen()

            except Exception as e:
                if 'indicator' in locals():
                    indicator.stop()
                clear_screen()
                move_cursor_to_center(1)
                print_centered(f"✗ Failed to start API server: {e}", "")
                shutdown_server()
                sys.exit(1)

            # Register cleanup handler if we started the server
            atexit.register(shutdown_server)
            signal.signal(signal.SIGINT, lambda sig, frame: (shutdown_server(), sys.exit(0)))
            signal.signal(signal.SIGTERM, lambda sig, frame: (shutdown_server(), sys.exit(0)))
        else:
            # Server already running - just clear screen for chat
            clear_screen()

    try:
        # Get token from Vault
        token = get_api_key('mira_api')
    except Exception as e:
        print(f"✗ Failed to retrieve API token from Vault: {e}", file=sys.stderr)
        print("\nMake sure:", file=sys.stderr)
        print("  1. Vault environment variables are set (VAULT_ADDR, VAULT_ROLE_ID, VAULT_SECRET_ID)", file=sys.stderr)
        print("  2. API token is stored in Vault at: mira/api_keys/mira_api", file=sys.stderr)
        if server_started:
            shutdown_server()
        sys.exit(1)

    # Route to appropriate mode
    if args.headless:
        one_shot(token, args.headless)
    else:
        chat_loop(token)

    # Clean shutdown if we started the server
    if server_started:
        shutdown_server()


if __name__ == "__main__":
    main()
