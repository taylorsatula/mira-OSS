"""
InboxPollerService — Polls IMAP inbox for unread emails during active
conversation segments.

Subclass of SegmentPoller. Publishes unread email headers to EmailTrinket
so Mira can mention them to the user. No email bodies are fetched — just
From, Subject, Date, and UID for token economy.
"""
import email
import email.header
import email.parser
import imaplib
import logging

from cns.services.pollers.segment_poller import SegmentPoller

logger = logging.getLogger(__name__)

# Maximum unread emails to surface in the trinket
MAX_UNREAD_EMAILS = 20


class InboxPollerService(SegmentPoller):
    """IMAP inbox poller — polls unread email headers every 3 minutes."""

    poll_interval_seconds = 180
    target_trinket = 'EmailTrinket'
    poller_name = 'inbox'

    def _try_load_config(self) -> dict | None:
        """Load IMAP credentials from the user's email_tool config.

        Infrastructure failures (Vault down, DB unreachable) propagate —
        the event bus catches and logs them. Only returns None when the
        user genuinely has no email credentials configured.
        """
        from utils.tool_config_store import load_user_tool_config

        config = load_user_tool_config("email_tool", hydrate_secrets=True)
        if not config:
            return None

        # Require minimum viable IMAP config
        required = ('imap_server', 'email_address', 'password')
        missing = [f for f in required if not config.get(f)]
        if missing:
            logger.warning(
                f"inbox poller: email_tool config missing {missing}, "
                "skipping poller start"
            )
            return None

        return {
            'imap_server': config['imap_server'],
            'imap_port': config.get('imap_port', 993),
            'email_address': config['email_address'],
            'password': config['password'],
            'use_ssl': config.get('use_ssl', True),
            'inbox_folder': config.get('inbox_folder', 'INBOX'),
        }

    def _poll_once(self, config: dict) -> list[dict] | None:
        """Fetch unread email headers via IMAP.

        Returns list of email header dicts on success, empty list if no
        unread emails, None on IMAP failure (transient — next cycle retries).
        """
        conn = None
        try:
            # Connect
            if config['use_ssl']:
                conn = imaplib.IMAP4_SSL(
                    config['imap_server'],
                    config['imap_port'],
                    timeout=15,
                )
            else:
                conn = imaplib.IMAP4(
                    config['imap_server'],
                    config['imap_port'],
                    timeout=15,
                )
            conn.login(config['email_address'], config['password'])

            # Select inbox readonly — prevents setting flags
            folder = config['inbox_folder']
            typ, _ = conn.select(folder, readonly=True)
            if typ != 'OK':
                logger.warning(f"inbox poller: could not select {folder}")
                return None

            # Search for unseen
            typ, data = conn.uid('SEARCH', None, 'UNSEEN')
            if typ != 'OK' or not data or not data[0]:
                return []  # No unread emails — valid empty result

            uid_list = data[0].decode('utf-8').split()

            # Cap at most recent
            uid_list = uid_list[-MAX_UNREAD_EMAILS:]

            # Batch fetch headers (PEEK avoids marking as read)
            uid_set = ','.join(uid_list)
            typ, fetch_data = conn.uid(
                'FETCH', uid_set,
                '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])',
            )
            if typ != 'OK':
                logger.warning("inbox poller: FETCH failed")
                return None

            emails = []
            parser = email.parser.BytesParser()

            for item in fetch_data:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue

                uid = _extract_uid(item[0])
                if uid is None:
                    continue

                msg = parser.parsebytes(item[1], headersonly=True)

                emails.append({
                    'uid': f"{folder}:{uid}",
                    'from_addr': _decode_header(msg.get('From', '')),
                    'subject': _decode_header(msg.get('Subject', '')),
                    'date': msg.get('Date', ''),
                })

            return emails

        except Exception as e:
            logger.warning(f"inbox poller: poll failed: {e}")
            return None

        finally:
            if conn is not None:
                try:
                    conn.logout()
                except Exception:
                    pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_uid(response_line: bytes) -> int | None:
    """Extract UID integer from IMAP FETCH response line.

    Response lines look like: b'1 (UID 12345 BODY[HEADER.FIELDS (...)]'
    """
    try:
        text = response_line.decode('utf-8', errors='replace')
        parts = text.split()
        for i, part in enumerate(parts):
            if part.upper() == 'UID' and i + 1 < len(parts):
                return int(parts[i + 1].rstrip(')'))
    except (ValueError, IndexError):
        pass
    return None


def _decode_header(raw: str) -> str:
    """Decode RFC 2047 encoded header value to plain text."""
    try:
        parts = email.header.decode_header(raw)
        decoded = []
        for content, charset in parts:
            if isinstance(content, bytes):
                decoded.append(content.decode(charset or 'utf-8', errors='replace'))
            else:
                decoded.append(content)
        return ' '.join(decoded)
    except Exception:
        return raw
