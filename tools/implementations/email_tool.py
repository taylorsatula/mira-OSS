"""
Email Tool - Email account access and management via IMAP/SMTP.

This tool provides a clean interface to email functionality, with a focus on
allowing the LLM to intelligently categorize and handle emails based on content
rather than rigid pattern matching.
"""
import email
import email.header
import email.message
import email.parser
import email.utils
import imaplib
import json
import re
import smtplib
import ssl
from email.message import EmailMessage, Message
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from tools.repo import Tool
from tools.registry import registry

# Define configuration class for EmailTool
class EmailToolConfig(BaseModel):
    """Configuration for the email_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    email_address: str = Field(
        default="",
        description="Email address for IMAP/SMTP authentication"
    )
    password: str = Field(
        default="",
        description="Email account password or app-specific password",
        json_schema_extra={"secret": True}
    )
    imap_server: str = Field(
        default="",
        description="IMAP server hostname (e.g., imap.gmail.com)"
    )
    imap_port: int = Field(
        default=993,
        description="IMAP server port (typically 993 for SSL/TLS)"
    )
    smtp_server: str = Field(
        default="",
        description="SMTP server hostname (e.g., smtp.gmail.com)"
    )
    smtp_port: int = Field(
        default=465,
        description="SMTP server port (typically 465 for SSL/TLS)"
    )
    use_ssl: bool = Field(
        default=True,
        description="Use SSL/TLS for secure connections"
    )
    # Folders - auto-discovered during validation, can be manually overridden
    inbox_folder: str = Field(default="INBOX", description="Inbox folder name")
    sent_folder: str = Field(default="Sent", description="Sent mail folder name")
    drafts_folder: str = Field(default="Drafts", description="Drafts folder name")
    trash_folder: str = Field(default="Trash", description="Trash folder name")

# Register with registry
registry.register("email_tool", EmailToolConfig)

# Operations that modify mailbox state — these require a reasoning parameter
_MUTATING_OPERATIONS = frozenset({
    'send_email', 'reply_to_email', 'create_draft',
    'delete_email', 'move_email', 'mark_as_read', 'mark_as_unread',
})

# Restricted schema — currently unused.
# Retained as a template for incremental autonomy: when the mailroom agent
# earns enough trust (via audited action+reasoning logs) to act on emails
# autonomously, this schema constrains it to reply-only — no send, search,
# or read. Each autonomy expansion adds operations to the enum; the schema
# is the safety boundary, not the trigger rules.
SIDEBAR_EMAIL_SCHEMA = {
    "name": "email_tool",
    "description": (
        "Reply to the email that triggered this agent. You can only reply "
        "-- not send new emails, search, or read other emails."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["reply_to_email"],
            },
            "email_id": {
                "type": "string",
                "description": (
                    "ID of the email to reply to (provided in your context)."
                ),
            },
            "body": {
                "type": "string",
                "description": "Reply body text.",
            },
        },
        "required": ["operation", "email_id", "body"],
    },
}



class EmailTool(Tool):
    """
    Tool for accessing and managing email through IMAP/SMTP protocols.
    
    This implementation focuses on:
    1. Loading email content into context for LLM categorization
    2. Session state management for referencing emails
    3. Progressive loading for efficiency
    4. Clean, focused API for common email operations
    """
    
    name = "email_tool"
    simple_description = """
    Email management tool that provides access to email accounts via IMAP/SMTP protocols. 
    Use this tool to read, search, send, and manage emails.
    """
    implementation_details = """
    OPERATIONS:
    - get_emails: Retrieve emails from specified folder with options for filtering and content loading
      Parameters:
        folder (optional, default="INBOX"): Email folder to access
        unread_only (optional, default=False): Set to True to only return unread emails
        load_content (optional, default=True): Set to True to load full email content
        sender (optional): Filter by sender email or name
        max_emails (optional, default=20): Maximum number of emails to return
    
    - get_email_content: Get full content of a specific email
      Parameters:
        email_id (required): Email ID in folder:uid format
        folder (optional, default="INBOX"): Email folder containing the email
    
    - mark_as_read: Mark an email as read
      Parameters:
        email_id (required): Email ID in folder:uid format
        folder (optional, default="INBOX"): Email folder containing the email
    
    - mark_as_unread: Mark an email as unread
      Parameters:
        email_id (required): Email ID in folder:uid format
        folder (optional, default="INBOX"): Email folder containing the email
    
    - delete_email: Delete an email
      Parameters:
        email_id (required): Email ID in folder:uid format
        folder (optional, default="INBOX"): Email folder containing the email
    
    - move_email: Move an email to another folder
      Parameters:
        email_id (required): Email ID in folder:uid format
        destination_folder (required): Folder to move the email to
        folder (optional, default="INBOX"): Source folder containing the email
    
    - send_email: Send a new email
      Parameters:
        to (required): Recipient email address(es)
        subject (required): Email subject
        body (required): Email body content
        cc (optional): CC recipient(s)
        bcc (optional): BCC recipient(s)
    
    - reply_to_email: Reply to an existing email
      Parameters:
        email_id (required): Email ID in folder:uid format
        body (required): Reply content
        folder (optional, default="INBOX"): Email folder containing the email
        cc (optional): CC recipient(s)
        bcc (optional): BCC recipient(s)
        
    - create_draft: Create a draft email without sending
      Parameters:
        to (required): Recipient email address(es)
        subject (required): Email subject
        body (required): Email body content
        cc (optional): CC recipient(s)
        bcc (optional): BCC recipient(s)
    
    - search_emails: Search emails with various criteria
      Parameters:
        folder (optional, default="INBOX"): Email folder to search in
        sender (optional): Sender email or name to search for
        subject (optional): Subject text to search for
        start_date (optional): Start date for range search (DD-Mon-YYYY format)
        end_date (optional): End date for range search (DD-Mon-YYYY format)
        unread_only (optional, default=False): Set to True to only return unread emails
        load_content (optional, default=True): Set to True to load full email content
        max_emails (optional, default=20): Maximum number of emails to return
        
    - list_folders: List available email folders
      Parameters: None
      
    - mark_for_later_reply: Mark an email to be replied to later in the continuum
      Parameters:
        email_id (required): Email ID in folder:uid format
        
    - get_emails_for_later_reply: Get list of emails marked for later reply
      Parameters: None
    
    USAGE NOTES:
    - Emails are loaded with full content by default to enable intelligent categorization
    - The LLM should categorize emails into groups like: from humans, priority, notifications, newsletters
    - Use the email_id to reference specific emails throughout the continuum
    - For handling multiple emails efficiently, process them by category
    - Mark emails for later reply to keep track of emails the user wants to address during the continuum
    """
    
    description = simple_description + implementation_details

    tool_schema = {
        "name": "email_tool",
        "description": "Email management tool that provides access to email accounts via IMAP/SMTP protocols. Use this tool to read, search, send, and manage emails.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["get_emails", "get_email_content", "mark_as_read", "mark_as_unread", "delete_email", "move_email", "send_email", "reply_to_email", "create_draft", "search_emails", "list_folders", "mark_for_later_reply", "get_emails_for_later_reply"],
                        "description": "The email operation to perform"
                    },
                    "folder": {
                        "type": "string",
                        "description": "Email folder to access (default: INBOX)"
                    },
                    "email_id": {
                        "type": "string",
                        "description": "Email ID in folder:uid format for operations that work on a single email"
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Whether to only return unread emails (default: false)"
                    },
                    "load_content": {
                        "type": "boolean",
                        "description": "Whether to load full email content (default: true)"
                    },
                    "sender": {
                        "type": "string",
                        "description": "Sender email address or name to search for"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject text to search for"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date for range search (DD-Mon-YYYY format)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date for range search (DD-Mon-YYYY format)"
                    },
                    "max_emails": {
                        "type": "integer",
                        "description": "Maximum number of emails to return (default: 20)"
                    },
                    "to": {
                        "type": "string",
                        "description": "Recipient email address(es) for sending emails"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content for sending or replying to emails"
                    },
                    "cc": {
                        "type": "string",
                        "description": "CC recipient(s) for sending emails"
                    },
                    "bcc": {
                        "type": "string",
                        "description": "BCC recipient(s) for sending emails"
                    },
                    "destination_folder": {
                        "type": "string",
                        "description": "Destination folder for move_email operation"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One-sentence explanation of WHY you are performing this action. Required for send_email, reply_to_email, create_draft, delete_email, move_email, mark_as_read, mark_as_unread. Logged for audit."
                    }
                },
                "required": ["operation"]
            }
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate email configuration by testing IMAP connection and discovering folders.

        Returns discovered data including available folders and suggested mappings.
        """
        imap_server = config.get("imap_server")
        imap_port = config.get("imap_port", 993)
        email_address = config.get("email_address")
        password = config.get("password")
        use_ssl = config.get("use_ssl", True)

        if not all([imap_server, email_address, password]):
            raise ValueError("Missing required fields: imap_server, email_address, password")

        connection = None
        try:
            # Connect to IMAP server
            if use_ssl:
                connection = imaplib.IMAP4_SSL(imap_server, imap_port, timeout=30)
            else:
                connection = imaplib.IMAP4(imap_server, imap_port, timeout=30)

            # Login
            connection.login(email_address, password)

            # Discover folders
            typ, folder_data = connection.list()
            if typ != "OK":
                raise ValueError("Failed to retrieve folder list from server")

            folders = []
            folder_mapping = {"inbox": "INBOX", "sent": None, "drafts": None, "trash": None}

            for item in folder_data:
                if not item:
                    continue

                decoded_item = item.decode("utf-8")
                match = re.match(r'^\((?P<flags>.*?)\) "(?P<delimiter>.*?)" (?P<name>.+)$', decoded_item)

                if match:
                    flags = match.group("flags")
                    folder_name = match.group("name")

                    # Remove quotes if present
                    if folder_name.startswith('"') and folder_name.endswith('"'):
                        folder_name = folder_name[1:-1]

                    folders.append({"name": folder_name, "flags": flags})

                    # Auto-detect standard folders by flags or name
                    flags_lower = flags.lower()
                    name_lower = folder_name.lower()

                    # Inbox: exact "inbox" or IMAP flag
                    if name_lower == "inbox" or "\\inbox" in flags_lower:
                        folder_mapping["inbox"] = folder_name

                    # Sent: check flags first, then name patterns
                    if "\\sent" in flags_lower:
                        folder_mapping["sent"] = folder_name
                    elif folder_mapping["sent"] is None:
                        # Check for common sent folder patterns (including INBOX.* namespace)
                        if name_lower in ("sent", "sent mail", "sent items", "[gmail]/sent mail"):
                            folder_mapping["sent"] = folder_name
                        elif "sent" in name_lower and "inbox" not in name_lower.replace("inbox.", ""):
                            # Matches "INBOX.Sent Messages", "INBOX.Sent", etc.
                            folder_mapping["sent"] = folder_name

                    # Drafts: check flags first, then name patterns
                    if "\\drafts" in flags_lower:
                        folder_mapping["drafts"] = folder_name
                    elif folder_mapping["drafts"] is None:
                        if name_lower in ("drafts", "[gmail]/drafts"):
                            folder_mapping["drafts"] = folder_name
                        elif "draft" in name_lower:
                            folder_mapping["drafts"] = folder_name

                    # Trash: check flags first, then name patterns
                    if "\\trash" in flags_lower:
                        folder_mapping["trash"] = folder_name
                    elif folder_mapping["trash"] is None:
                        if name_lower in ("trash", "deleted", "deleted items", "[gmail]/trash"):
                            folder_mapping["trash"] = folder_name
                        elif "trash" in name_lower or "deleted" in name_lower:
                            folder_mapping["trash"] = folder_name

            # Don't fail on missing folders - let user select manually

            return {
                "folders": folders,
                "discovered_folders": {
                    "inbox_folder": folder_mapping.get("inbox") or "INBOX",
                    "sent_folder": folder_mapping.get("sent"),
                    "drafts_folder": folder_mapping.get("drafts"),
                    "trash_folder": folder_mapping.get("trash")
                },
                "connection_test": "success"
            }

        except imaplib.IMAP4.error as e:
            raise ValueError(f"IMAP authentication failed: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Connection test failed: {e}")
        finally:
            if connection:
                try:
                    connection.logout()
                except Exception:
                    pass

    def __init__(self):
        """Initialize the email tool with configuration."""
        super().__init__()
        
        # Configuration will be loaded from user credentials when needed
        self.imap_server = None
        self.imap_port = None
        self.smtp_server = None
        self.smtp_port = None
        self.email_address = None
        self.use_ssl = None
        self._password = None
        self._config_loaded = False

        # Folder configuration (loaded from config)
        self.inbox_folder = "INBOX"
        self.sent_folder = "Sent"
        self.drafts_folder = "Drafts"
        self.trash_folder = "Trash"
        
        # Session state
        self.connection = None
        self.selected_folder = None
        self.default_max_emails = 20

    def _load_config(self):
        """Load email configuration from unified tool config storage."""
        if self._config_loaded:
            return

        from utils.tool_config_store import load_user_tool_config

        config = load_user_tool_config("email_tool", hydrate_secrets=True)

        if not config:
            self.logger.error("No email configuration found. Configure email in Settings > Tools.")
            raise ValueError("No email configuration found. Configure email in Settings > Tools.")

        # Required fields
        self.imap_server = config.get("imap_server")
        self.smtp_server = config.get("smtp_server")
        self.email_address = config.get("email_address")
        self._password = config.get("password")

        if not all([self.imap_server, self.smtp_server, self.email_address, self._password]):
            missing = [f for f in ["imap_server", "smtp_server", "email_address", "password"]
                       if not config.get(f)]
            raise ValueError(f"Missing required email config fields: {missing}")

        # Optional fields with defaults
        self.imap_port = config.get("imap_port", 993)
        self.smtp_port = config.get("smtp_port", 465)
        self.use_ssl = config.get("use_ssl", True)

        # Folder configuration (auto-discovered during validation, can be manually set)
        self.inbox_folder = config.get("inbox_folder", "INBOX")
        self.sent_folder = config.get("sent_folder", "Sent")
        self.drafts_folder = config.get("drafts_folder", "Drafts")
        self.trash_folder = config.get("trash_folder", "Trash")

        self._config_loaded = True
    
    @property
    def password(self):
        """Lazy-load email password from user credentials."""
        if not self._config_loaded:
            self._load_config()
        return self._password
    
    def _is_connection_alive(self) -> bool:
        """
        Check if the IMAP connection is still alive and responsive.
        
        Returns:
            True if connection is alive, False otherwise
        """
        if not self.connection:
            return False
            
        try:
            # Try a NOOP command to check if connection is still responsive
            status, response = self.connection.noop()
            return status == 'OK'
        except Exception as e:
            self.logger.warning(f"IMAP connection check failed: {e}")
            self.connection = None
            return False
    
    def _connect(self) -> bool:
        """
        Connect to the IMAP server if not already connected or if connection is dead.
        
        Returns:
            True if connection succeeded, False otherwise
        """
        # Check if already connected and connection is alive
        if self.connection and self._is_connection_alive():
            # Already connected and alive
            return True
            
        # Reset connection state
        self.connection = None
        self.selected_folder = None
            
        try:
            if self.use_ssl:
                self.connection = imaplib.IMAP4_SSL(self.imap_server, self.imap_port, timeout=30)
            else:
                self.connection = imaplib.IMAP4(self.imap_server, self.imap_port, timeout=30)
            
            # Login
            self.connection.login(self.email_address, self.password)
            self.logger.info(f"Connected to IMAP server {self.imap_server}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IMAP server: {e}")
            self.connection = None
            return False
    
    def _disconnect(self) -> None:
        """Close the IMAP connection if open."""
        if hasattr(self, 'connection') and self.connection:
            try:
                self.connection.logout()
                self.logger.info("Disconnected from IMAP server")
            except Exception as e:
                self.logger.error(f"Error disconnecting from IMAP server: {e}")
            finally:
                self.connection = None
                self.selected_folder = None
    
    def _ensure_connected(self) -> bool:
        """
        Ensure the IMAP connection is alive, reconnecting if necessary.
        
        Returns:
            True if connection is established, False if connection failed
        """
        # Load config first if not already loaded
        if not self._config_loaded:
            self._load_config()
            
        # First check if the connection is alive
        if self._is_connection_alive():
            return True
            
        # If we get here, we need to connect/reconnect
        return self._connect()
    
    def _select_folder(self, folder_name: str) -> bool:
        """
        Select a folder/mailbox.
        
        Args:
            folder_name: Name of the folder to select
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure connection is established
        if not self._ensure_connected():
            return False
        
        # No need to reselect if already on this folder
        if folder_name == self.selected_folder:
            return True
        
        try:
            status, response = self.connection.select(folder_name)
            if status != 'OK':
                self.logger.warning(f"Failed to select folder '{folder_name}': {response}")
                return False
                
            self.selected_folder = folder_name
            self.logger.info(f"Selected mailbox '{folder_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to select mailbox '{folder_name}': {e}")
            # Connection might have been lost, try to reconnect and retry once
            if self._connect():
                try:
                    status, response = self.connection.select(folder_name)
                    if status == 'OK':
                        self.selected_folder = folder_name
                        self.logger.info(f"Successfully selected mailbox '{folder_name}' after reconnection")
                        return True
                except Exception:
                    pass
            return False
    
    def _create_email_id(self, folder: str, uid: int) -> str:
        """
        Create a stable email identifier from folder and UID.

        The identifier encodes both folder and UID so it's self-contained -
        no external mapping storage needed.

        Args:
            folder: IMAP folder name
            uid: IMAP UID (stable within folder)

        Returns:
            Email identifier string in format "folder:uid"
        """
        return f"{folder}:{uid}"

    def _later_reply_file(self):
        """User-scoped persistence for emails marked for later reply."""
        return self.user_data_path / "later_replies.json"

    def _load_later_reply_ids(self) -> set[str]:
        """Load persisted later-reply IDs."""
        path = self._later_reply_file()
        if not path.exists():
            return set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                return set()
            return {item for item in data if isinstance(item, str)}
        except Exception as e:
            self.logger.warning(f"Failed to load later-reply email IDs: {e}")
            return set()

    def _save_later_reply_ids(self, email_ids: set[str]) -> None:
        """Persist later-reply IDs in stable order."""
        path = self._later_reply_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(sorted(email_ids), handle, indent=2)

    def _add_later_reply_id(self, email_id: str) -> None:
        email_ids = self._load_later_reply_ids()
        email_ids.add(email_id)
        self._save_later_reply_ids(email_ids)

    def _remove_later_reply_id(self, email_id: str) -> None:
        email_ids = self._load_later_reply_ids()
        if email_id in email_ids:
            email_ids.remove(email_id)
            self._save_later_reply_ids(email_ids)

    def _parse_email_id(self, email_id: str) -> tuple[str, int]:
        """
        Parse an email identifier back to folder and UID.

        Args:
            email_id: Email identifier string

        Returns:
            Tuple of (folder, uid)

        Raises:
            ValueError: If email_id format is invalid
        """
        if ":" not in email_id:
            raise ValueError(f"Invalid email ID format: {email_id}")

        # Split on last colon to handle folder names with colons
        last_colon = email_id.rfind(":")
        folder = email_id[:last_colon]
        uid_str = email_id[last_colon + 1:]

        try:
            uid = int(uid_str)
        except ValueError:
            raise ValueError(f"Invalid UID in email ID: {email_id}")

        return folder, uid
    
    def _decode_header(self, header: str) -> str:
        """
        Decode an email header with proper handling of character encodings.
        
        Args:
            header: Raw header string
            
        Returns:
            Decoded header string
        """
        if not header:
            return ""
        
        decoded_parts = []
        
        for decoded_header, charset in email.header.decode_header(header):
            if isinstance(decoded_header, bytes):
                if charset:
                    try:
                        decoded_parts.append(decoded_header.decode(charset))
                    except (LookupError, UnicodeDecodeError):
                        try:
                            decoded_parts.append(decoded_header.decode("utf-8"))
                        except UnicodeDecodeError:
                            decoded_parts.append(decoded_header.decode("latin1", errors="replace"))
                else:
                    try:
                        decoded_parts.append(decoded_header.decode("utf-8"))
                    except UnicodeDecodeError:
                        decoded_parts.append(decoded_header.decode("latin1", errors="replace"))
            else:
                decoded_parts.append(str(decoded_header))
        
        return " ".join(decoded_parts)
    
    def _get_email_body(self, msg: Message) -> Dict[str, Any]:
        """
        Extract the body content and attachment info from a message.

        Args:
            msg: Email message object

        Returns:
            Dictionary with body text and attachment info
        """
        result = {
            "text": "",
            "has_attachments": False,
            "attachments": []
        }

        # Helper class for extracting text from HTML
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []

            def handle_data(self, data):
                self.text_parts.append(data)

            def get_text(self):
                return ' '.join(self.text_parts)

        html_fallback_content = None  # Store HTML content for fallback

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Handle attachments - detect but don't download
                if "attachment" in content_disposition:
                    result["has_attachments"] = True
                    filename = part.get_filename()
                    if filename:
                        payload = part.get_payload(decode=True)
                        attachment_info = {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(payload) if payload else 0
                        }
                        result["attachments"].append(attachment_info)
                    continue

                # Handle text parts
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            text = payload.decode(charset)
                        except UnicodeDecodeError:
                            try:
                                text = payload.decode("utf-8")
                            except UnicodeDecodeError:
                                text = payload.decode("latin1", errors="replace")

                        result["text"] += text

                # Store HTML content for fallback if no text/plain found
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    if html_fallback_content is None:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            try:
                                html_fallback_content = payload.decode(charset)
                            except UnicodeDecodeError:
                                html_fallback_content = payload.decode("utf-8", errors="replace")
        else:
            # Non-multipart - get the payload directly
            content_type = msg.get_content_type()

            if content_type == "text/plain":
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    try:
                        result["text"] = payload.decode(charset)
                    except UnicodeDecodeError:
                        try:
                            result["text"] = payload.decode("utf-8")
                        except UnicodeDecodeError:
                            result["text"] = payload.decode("latin1", errors="replace")
            elif content_type == "text/html":
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    try:
                        html_fallback_content = payload.decode(charset)
                    except UnicodeDecodeError:
                        html_fallback_content = payload.decode("utf-8", errors="replace")

        # If no text/plain was found, fall back to HTML content
        if not result["text"] and html_fallback_content:
            try:
                extractor = HTMLTextExtractor()
                extractor.feed(html_fallback_content)
                result["text"] = extractor.get_text().strip()
            except Exception:
                # If HTML parsing fails, strip tags with simple regex as last resort
                result["text"] = re.sub(r'<[^>]+>', '', html_fallback_content)
                result["text"] = re.sub(r'\s+', ' ', result["text"]).strip()

        return result
    
    def _parse_flags(self, flags_str: str) -> List[str]:
        """
        Parse IMAP flags from a string response.

        Args:
            flags_str: String containing IMAP flags (e.g., "\\Seen \\Flagged")

        Returns:
            List of human-readable flag strings
        """
        flags = []

        if "\\Seen" in flags_str:
            flags.append("read")
        else:
            flags.append("unread")

        if "\\Flagged" in flags_str:
            flags.append("flagged")

        if "\\Answered" in flags_str:
            flags.append("answered")

        if "\\Draft" in flags_str:
            flags.append("draft")

        return flags

    def _get_message_flags(self, uid: int) -> List[str]:
        """
        Get the flags for a message by UID.

        Args:
            uid: IMAP UID

        Returns:
            List of flag strings
        """
        if not self._ensure_connected():
            return []

        try:
            # Fetch the flags using UID
            typ, data = self.connection.uid("FETCH", str(uid), "(FLAGS)")
            if typ != "OK" or not data or not data[0]:
                return []

            flags_str = data[0].decode("utf-8")
            return self._parse_flags(flags_str)
        except Exception as e:
            self.logger.error(f"Error getting flags for UID {uid}: {e}")
            return []
    
    def _search_messages(self, criteria: str) -> List[int]:
        """
        Search for messages in the selected folder using UIDs.

        Args:
            criteria: IMAP search criteria string

        Returns:
            List of message UIDs matching the criteria
        """
        if not self._ensure_connected():
            return []

        try:
            # Execute UID-based search (UIDs are stable, unlike sequence numbers)
            typ, data = self.connection.uid("SEARCH", None, criteria)
            if typ != "OK" or not data or not data[0]:
                return []

            # Parse UIDs
            uids = data[0].decode("utf-8").split()
            return list(map(int, uids))
        except Exception as e:
            self.logger.error(f"Error searching messages with criteria '{criteria}': {e}")
            return []
    
    def _fetch_message_headers(self, uids: List[int], limit: int = None, load_content: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch headers for a list of message UIDs.

        Args:
            uids: List of IMAP UIDs
            limit: Maximum number of messages to fetch
            load_content: Whether to load the full message content

        Returns:
            List of email header dictionaries (with content if requested)
        """
        if not uids or not self._ensure_connected():
            return []

        if limit and len(uids) > limit:
            uids = uids[-limit:]  # Take the most recent messages

        email_items = []
        folder = self.selected_folder  # Capture current folder for ID generation

        for uid in uids:
            try:
                if load_content:
                    # Fetch full message AND flags in single request using UID
                    typ, data = self.connection.uid("FETCH", str(uid), "(RFC822 FLAGS)")
                    if typ != "OK" or not data or not data[0]:
                        continue

                    # Parse flags from the response metadata
                    response_meta = data[0][0].decode("utf-8") if isinstance(data[0][0], bytes) else str(data[0][0])
                    flags = self._parse_flags(response_meta)

                    # Parse the full message
                    email_data = data[0][1]
                    msg = email.message_from_bytes(email_data)

                    # Extract body content
                    body = self._get_email_body(msg)

                    # Create stable email ID from folder and UID
                    email_id = self._create_email_id(folder, uid)

                    # Create the result with content
                    email_info = {
                        "id": email_id,
                        "from": self._decode_header(msg.get("From", "")),
                        "to": self._decode_header(msg.get("To", "")),
                        "cc": self._decode_header(msg.get("Cc", "")),
                        "subject": self._decode_header(msg.get("Subject", "")),
                        "date": self._decode_header(msg.get("Date", "")),
                        "body_text": body["text"],
                        "has_attachments": body["has_attachments"],
                        "flags": flags,
                    }

                    # Add attachment information if present
                    if body["has_attachments"]:
                        email_info["attachments"] = body["attachments"]
                else:
                    # Fetch headers AND flags in single request using UID
                    typ, data = self.connection.uid("FETCH", str(uid), "(BODY.PEEK[HEADER] FLAGS)")
                    if typ != "OK" or not data or not data[0]:
                        continue

                    # Parse flags from the response metadata
                    response_meta = data[0][0].decode("utf-8") if isinstance(data[0][0], bytes) else str(data[0][0])
                    flags = self._parse_flags(response_meta)

                    header_data = data[0][1]
                    if not header_data:
                        continue

                    # Parse headers
                    parser = email.parser.BytesParser()
                    headers = parser.parsebytes(header_data, headersonly=True)

                    # Create stable email ID from folder and UID
                    email_id = self._create_email_id(folder, uid)

                    # Create header dictionary
                    email_info = {
                        "id": email_id,
                        "from": self._decode_header(headers.get("From", "")),
                        "subject": self._decode_header(headers.get("Subject", "")),
                        "date": self._decode_header(headers.get("Date", "")),
                        "flags": flags
                    }

                email_items.append(email_info)
            except Exception as e:
                self.logger.error(f"Error fetching {'full message' if load_content else 'headers'} for UID {uid}: {e}")

        return email_items
    
    def _set_flag(self, email_id: str, flag: str, value: bool) -> bool:
        """
        Set or unset a flag on an email.

        Args:
            email_id: Email identifier (folder:uid format)
            flag: Flag name ('\\Seen', '\\Flagged', etc.)
            value: True to set, False to unset

        Returns:
            True if successful, False otherwise
        """
        try:
            folder, uid = self._parse_email_id(email_id)
        except ValueError as e:
            self.logger.error(f"Invalid email_id format: {e}")
            return False

        if not self._select_folder(folder):
            return False

        try:
            # Set or unset the flag using UID
            command = "+FLAGS" if value else "-FLAGS"
            self.connection.uid("STORE", str(uid), command, flag)
            return True
        except Exception as e:
            self.logger.error(f"Error setting flag {flag} for UID {uid}: {e}")
            return False
    
    def _parse_email_addresses(self, email_param: Optional[str]) -> Optional[str]:
        """
        Parse email address string that might be in JSON array format.
        
        Args:
            email_param: Email address string that might be a JSON array
            
        Returns:
            Properly formatted email string or None if input was None
        """
        if not email_param:
            return None
            
        # Check if the input might be a JSON array
        if email_param.startswith('[') and email_param.endswith(']'):
            try:
                import json
                emails = json.loads(email_param)
                # Ensure we have a list of strings
                if isinstance(emails, list):
                    # Filter empty values and join with commas
                    valid_emails = [e.strip() for e in emails if e and isinstance(e, str) and e.strip()]
                    if valid_emails:
                        return ", ".join(valid_emails)
                    return None
            except json.JSONDecodeError:
                # If not valid JSON, continue with original value
                pass
                
        # Process as comma-separated string, filtering empty addresses
        valid_emails = [addr.strip() for addr in email_param.split(",") if addr and addr.strip()]
        if valid_emails:
            return ", ".join(valid_emails)
        return None
    
    def run(
        self,
        operation: str,
        folder: str = "INBOX",
        email_id: Optional[str] = None,
        unread_only: bool = False,
        load_content: bool = True,
        sender: Optional[str] = None,
        subject: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_emails: Optional[int] = None,
        to: Optional[str] = None,
        body: Optional[str] = None,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        destination_folder: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an email operation.

        Args:
            operation: The operation to perform (get_emails, get_email_content, etc.)
            folder: Email folder to access (default: "INBOX")
            email_id: Email ID in folder:uid format
            unread_only: Whether to only return unread emails
            load_content: Whether to load full email content
            sender: Sender email address or name to search for
            subject: Subject text to search for
            start_date: Start date for range search (DD-Mon-YYYY format)
            end_date: End date for range search (DD-Mon-YYYY format)
            max_emails: Maximum number of emails to return
            to: Recipient for sending emails
            body: Body text for sending emails
            cc: CC recipients for sending emails
            bcc: BCC recipients for sending emails
            destination_folder: Destination folder for move_email
            
        Returns:
            Dictionary with operation results
            
        Raises:
            ValueError: If the operation is invalid or fails
        """
        try:
            # Ensure we're connected to the server
            if not self._ensure_connected():
                self.logger.error("Failed to connect to email server for email_tool")
                raise ValueError("Failed to connect to email server")

            # Translate "INBOX" to configured inbox folder (handles INBOX.* namespace servers)
            if folder == "INBOX" and hasattr(self, 'inbox_folder') and self.inbox_folder:
                folder = self.inbox_folder

            # Make sure we're looking at the right folder for message operations
            if folder and folder != self.selected_folder and operation not in ["list_folders", "send_email", "create_draft", "mark_for_later_reply", "get_emails_for_later_reply"]:
                if not self._select_folder(folder):
                    self.logger.error(f"Failed to select folder '{folder}' in email_tool")
                    raise ValueError(f"Failed to select folder '{folder}'")
            
            # Validate reasoning for mutating operations
            if operation in _MUTATING_OPERATIONS and not (reasoning and reasoning.strip()):
                raise ValueError(
                    f"reasoning is required for '{operation}'. "
                    "Explain why you are performing this action."
                )

            # Set default max_emails if not provided and convert to int if it's a string
            if max_emails is None:
                max_emails = self.default_max_emails
            elif isinstance(max_emails, str):
                max_emails = int(max_emails)
            
            # Handle each operation type
            if operation == "get_emails":
                # Build search criteria
                search_parts = []
                
                if unread_only:
                    search_parts.append("UNSEEN")
                
                if sender:
                    search_parts.append(f'FROM "{sender}"')
                
                if subject:
                    search_parts.append(f'SUBJECT "{subject}"')
                
                if start_date:
                    search_parts.append(f'SINCE "{start_date}"')
                
                if end_date:
                    search_parts.append(f'BEFORE "{end_date}"')
                
                # Default to ALL if no criteria specified
                search_criteria = " ".join(search_parts) if search_parts else "ALL"
                
                # Execute search and fetch emails
                message_ids = self._search_messages(search_criteria)
                emails = self._fetch_message_headers(message_ids, max_emails, load_content)
                
                # LLM handling notes for categorization and summarization
                categorization_note = """
                CATEGORIZATION INSTRUCTIONS:
                
                Group these emails into the following categories:
                1. "humans" - Emails from real people requiring personal attention
                2. "priority" - Important emails needing immediate action
                3. "notifications" - Automated notifications from services
                4. "newsletters" - Marketing and newsletter emails
                
                For each email, consider:
                - Sender address and name patterns
                - Subject line keywords
                - Content patterns and formality
                - Importance to the recipient
                
                Provide a summary of how many emails are in each category before showing details.
                When the user asks to see emails from a specific category, show a numbered list with
                brief summaries of each email in that category.
                
                Example output:
                "You have 3 emails from humans, 2 priority emails, 4 notifications, and 7 newsletters."
                
                When showing emails in a category:
                "Here are your emails from humans:
                1. John Smith - Meeting tomorrow at 2pm
                2. Sarah Lee - Question about the project timeline
                3. Mike Johnson - Kids soccer practice cancelled"
                """
                
                return {
                    "emails": emails,
                    "total": len(message_ids),
                    "showing": len(emails),
                    "content_loaded": load_content,
                    "categorization_note": categorization_note if load_content else "Load content to enable intelligent categorization"
                }
            
            elif operation == "get_email_content":
                # Validate required parameters
                if not email_id:
                    raise ValueError("email_id is required for get_email_content operation")

                # Parse email_id to get folder and UID
                try:
                    email_folder, uid = self._parse_email_id(email_id)
                except ValueError as e:
                    raise ValueError(f"Invalid email_id format: {e}")

                # Select the folder containing this email
                if not self._select_folder(email_folder):
                    raise ValueError(f"Failed to select folder '{email_folder}'")

                try:
                    # Fetch the full message using UID
                    typ, data = self.connection.uid("FETCH", str(uid), "(RFC822)")
                    if typ != "OK" or not data or not data[0]:
                        self.logger.error(f"Failed to fetch email content for ID {email_id} in email_tool")
                        raise ValueError(f"Failed to fetch email content for ID {email_id}")

                    # Parse the message
                    email_data = data[0][1]
                    msg = email.message_from_bytes(email_data)

                    # Extract body content
                    body = self._get_email_body(msg)

                    # Get the flags
                    flags = self._get_message_flags(uid)

                    # Mark the email as read if it wasn't already
                    if "unread" in flags:
                        if not self._set_flag(email_id, "\\Seen", True):
                            self.logger.warning(f"Failed to mark email {email_id} as read")
                        # Update flags to reflect the change
                        flags = [flag for flag in flags if flag != "unread"]
                        flags.append("read")

                    # Create the result
                    result = {
                        "id": email_id,
                        "from": self._decode_header(msg.get("From", "")),
                        "to": self._decode_header(msg.get("To", "")),
                        "cc": self._decode_header(msg.get("Cc", "")),
                        "subject": self._decode_header(msg.get("Subject", "")),
                        "date": self._decode_header(msg.get("Date", "")),
                        "body_text": body["text"],
                        "has_attachments": body["has_attachments"],
                        "flags": flags
                    }

                    # Add attachment information if present
                    if body["has_attachments"]:
                        result["attachments"] = body["attachments"]

                    return result
                except ValueError:
                    raise
                except Exception as e:
                    self.logger.error(f"Error fetching email content in email_tool: {e}")
                    raise ValueError(f"Error fetching email content: {e}")
            
            elif operation == "mark_as_read":
                # Validate required parameters
                if not email_id:
                    raise ValueError("email_id is required for mark_as_read operation")
                
                # Set the \Seen flag
                success = self._set_flag(email_id, "\\Seen", True)
                
                if not success:
                    self.logger.error(f"Failed to mark email {email_id} as read in email_tool")
                    self._log_audit("mark_as_read", f"mark_as_read {email_id}", reasoning, "failed", f"Failed to mark email {email_id} as read")
                    raise ValueError(f"Failed to mark email {email_id} as read")

                self._log_audit("mark_as_read", f"mark_as_read {email_id}", reasoning, "success")
                return {
                    "success": True,
                    "email_id": email_id,
                    "operation": "mark_as_read"
                }

            elif operation == "mark_as_unread":
                # Validate required parameters
                if not email_id:
                    raise ValueError("email_id is required for mark_as_unread operation")
                
                # Remove the \Seen flag
                success = self._set_flag(email_id, "\\Seen", False)
                
                if not success:
                    self.logger.error(f"Failed to mark email {email_id} as unread in email_tool")
                    self._log_audit("mark_as_unread", f"mark_as_unread {email_id}", reasoning, "failed", f"Failed to mark email {email_id} as unread")
                    raise ValueError(f"Failed to mark email {email_id} as unread")

                self._log_audit("mark_as_unread", f"mark_as_unread {email_id}", reasoning, "success")
                return {
                    "success": True,
                    "email_id": email_id,
                    "operation": "mark_as_unread"
                }

            elif operation == "delete_email":
                # Validate required parameters
                if not email_id:
                    raise ValueError("email_id is required for delete_email operation")

                # Parse email_id to get folder and UID
                try:
                    email_folder, uid = self._parse_email_id(email_id)
                except ValueError as e:
                    raise ValueError(f"Invalid email_id format: {e}")

                # Select the folder containing this email
                if not self._select_folder(email_folder):
                    raise ValueError(f"Failed to select folder '{email_folder}'")

                try:
                    # Mark the message as deleted using UID
                    self.connection.uid("STORE", str(uid), "+FLAGS", "\\Deleted")

                    # Expunge the message
                    self.connection.expunge()

                    self._remove_later_reply_id(email_id)

                    self._log_audit("delete_email", f"delete {email_id}", reasoning, "success")
                    return {
                        "success": True,
                        "email_id": email_id,
                        "operation": "delete_email"
                    }
                except Exception as e:
                    self.logger.error(f"Failed to delete email {email_id} in email_tool: {e}")
                    self._log_audit("delete_email", f"delete {email_id}", reasoning, "failed", str(e))
                    raise ValueError(f"Failed to delete email {email_id}: {e}")
            
            elif operation == "move_email":
                # Validate required parameters
                if not email_id:
                    raise ValueError("email_id is required for move_email operation")

                if not destination_folder:
                    raise ValueError("destination_folder is required for move_email operation")

                # Parse email_id to get folder and UID
                try:
                    source_folder, uid = self._parse_email_id(email_id)
                except ValueError as e:
                    raise ValueError(f"Invalid email_id format: {e}")

                # Select the source folder
                if not self._select_folder(source_folder):
                    raise ValueError(f"Failed to select folder '{source_folder}'")

                try:
                    # Try to use UID MOVE command if server supports it
                    move_supported = b'MOVE' in self.connection.capabilities

                    if move_supported:
                        # Use the UID MOVE command
                        self.connection.uid("MOVE", str(uid), destination_folder)
                    else:
                        # Fall back to UID copy and delete
                        # Copy to destination
                        self.connection.uid("COPY", str(uid), destination_folder)

                        # Mark as deleted
                        self.connection.uid("STORE", str(uid), "+FLAGS", "\\Deleted")

                        # Expunge
                        self.connection.expunge()

                    # email_id is no longer valid after move.
                    self._remove_later_reply_id(email_id)

                    self._log_audit("move_email", f"move {email_id} to {destination_folder}", reasoning, "success")
                    return {
                        "success": True,
                        "email_id": email_id,
                        "destination": destination_folder,
                        "operation": "move_email"
                    }
                except Exception as e:
                    self.logger.error(f"Failed to move email {email_id} to {destination_folder} in email_tool: {e}")
                    self._log_audit("move_email", f"move {email_id} to {destination_folder}", reasoning, "failed", str(e))
                    raise ValueError(f"Failed to move email {email_id} to {destination_folder}: {e}")
            
            elif operation == "send_email":
                # Validate required parameters
                if not to:
                    self.logger.error("Missing 'to' field for send_email operation in email_tool")
                    raise ValueError("to is required for send_email operation")
                
                if not subject:
                    self.logger.error("Missing subject for send_email operation in email_tool")
                    raise ValueError("subject is required for send_email operation")
                
                if not body:
                    self.logger.error("Missing body for send_email operation in email_tool")
                    raise ValueError("body is required for send_email operation")
                
                try:
                    # Parse the email addresses before creating the message
                    parsed_to = self._parse_email_addresses(to)
                    if not parsed_to:
                        self.logger.error("No valid email addresses in 'to' field in email_tool")
                        raise ValueError("No valid email addresses in 'to' field")
                    
                    parsed_cc = self._parse_email_addresses(cc) if cc else None
                    parsed_bcc = self._parse_email_addresses(bcc) if bcc else None
                    
                    # Create the message
                    msg = EmailMessage()
                    msg["Subject"] = subject
                    msg["From"] = self.email_address
                    msg["To"] = parsed_to
                    
                    if parsed_cc:
                        msg["Cc"] = parsed_cc
                    
                    if parsed_bcc:
                        msg["Bcc"] = parsed_bcc
                    
                    # Set the date
                    msg["Date"] = email.utils.formatdate(localtime=True)
                    
                    # Add Message-ID
                    domain = self.smtp_server.split(".", 1)[1] if "." in self.smtp_server else self.smtp_server
                    msg["Message-ID"] = email.utils.make_msgid(domain=domain)
                    
                    # Set the content
                    msg.set_content(body)
                    
                    # Build list of recipients (reuse already-parsed addresses)
                    recipients = []
                    if parsed_to:
                        recipients.extend([addr.strip() for addr in parsed_to.split(",") if addr.strip()])
                    if parsed_cc:
                        recipients.extend([addr.strip() for addr in parsed_cc.split(",") if addr.strip()])
                    if parsed_bcc:
                        recipients.extend([addr.strip() for addr in parsed_bcc.split(",") if addr.strip()])
                    
                    # Connect to SMTP server
                    context = ssl.create_default_context()
                    
                    with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context, timeout=30) as server:
                        # Login
                        server.login(self.email_address, self.password)
                        
                        # Send the email
                        server.send_message(msg)
                    
                    # Save to Sent folder
                    try:
                        self.connection.append(self.sent_folder, None, None, msg.as_bytes())
                    except Exception as e:
                        self.logger.warning(f"Failed to save to sent folder: {e}")
                    
                    self._log_audit("send_email", f"to={to}, subject={subject}", reasoning, "success")
                    return {
                        "success": True,
                        "to": to,
                        "subject": subject,
                        "operation": "send_email"
                    }
                except Exception as e:
                    self.logger.error(f"Failed to send email in email_tool: {e}")
                    self._log_audit("send_email", f"to={to}, subject={subject}", reasoning, "failed", str(e))
                    raise ValueError(f"Failed to send email: {e}")
            
            elif operation == "reply_to_email":
                # Validate required parameters
                if not email_id:
                    self.logger.error("Missing email_id for reply_to_email operation in email_tool")
                    raise ValueError("email_id is required for reply_to_email operation")

                if not body:
                    self.logger.error("Missing body for reply_to_email operation in email_tool")
                    raise ValueError("body is required for reply_to_email operation")

                # Parse email_id to get folder and UID
                try:
                    email_folder, uid = self._parse_email_id(email_id)
                except ValueError as e:
                    raise ValueError(f"Invalid email_id format: {e}")

                # Select the folder containing this email
                if not self._select_folder(email_folder):
                    raise ValueError(f"Failed to select folder '{email_folder}'")

                try:
                    # Fetch the original message using UID
                    typ, data = self.connection.uid("FETCH", str(uid), "(RFC822)")
                    if typ != "OK" or not data or not data[0]:
                        self.logger.error(f"Failed to fetch original email for reply in email_tool")
                        raise ValueError(f"Failed to fetch original email for reply")
                    
                    # Parse the message
                    email_data = data[0][1]
                    original_msg = email.message_from_bytes(email_data)
                    
                    # Create reply message
                    msg = EmailMessage()
                    
                    # Set subject with Re: prefix if needed
                    original_subject = self._decode_header(original_msg.get("Subject", ""))
                    if original_subject.lower().startswith("re:"):
                        msg["Subject"] = original_subject
                    else:
                        msg["Subject"] = f"Re: {original_subject}"
                    
                    # Set From
                    msg["From"] = self.email_address
                    
                    # Set To
                    if to:
                        parsed_to = self._parse_email_addresses(to)
                        msg["To"] = parsed_to if parsed_to else original_msg.get("From", "")
                    else:
                        reply_to = original_msg.get("Reply-To")
                        msg["To"] = reply_to if reply_to else original_msg.get("From", "")
                    
                    # Add CC/BCC if specified
                    if cc:
                        parsed_cc = self._parse_email_addresses(cc)
                        if parsed_cc:
                            msg["Cc"] = parsed_cc
                    
                    if bcc:
                        parsed_bcc = self._parse_email_addresses(bcc)
                        if parsed_bcc:
                            msg["Bcc"] = parsed_bcc
                    
                    # Set the date
                    msg["Date"] = email.utils.formatdate(localtime=True)
                    
                    # Set In-Reply-To and References headers for threading
                    message_id_header = original_msg.get("Message-ID")
                    if message_id_header:
                        msg["In-Reply-To"] = message_id_header
                        
                        # Set References
                        references = original_msg.get("References", "")
                        if references:
                            msg["References"] = f"{references} {message_id_header}"
                        else:
                            msg["References"] = message_id_header
                    
                    # Set the content
                    msg.set_content(body)
                    
                    # Connect to SMTP server
                    context = ssl.create_default_context()
                    
                    with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context, timeout=30) as server:
                        # Login
                        server.login(self.email_address, self.password)
                        
                        # Send the email
                        server.send_message(msg)
                    
                    # Save to Sent folder
                    try:
                        self.connection.append(self.sent_folder, None, None, msg.as_bytes())
                    except Exception as e:
                        self.logger.warning(f"Failed to save to sent folder: {e}")
                    
                    # Mark as answered
                    if not self._set_flag(email_id, "\\Answered", True):
                        self.logger.warning(f"Failed to mark email {email_id} as answered")

                    self._remove_later_reply_id(email_id)

                    self._log_audit("reply_to_email", f"reply to {email_id}", reasoning, "success")
                    return {
                        "success": True,
                        "replied_to": email_id,
                        "subject": msg["Subject"],
                        "operation": "reply_to_email"
                    }
                except Exception as e:
                    self.logger.error(f"Failed to reply to email in email_tool: {e}")
                    self._log_audit("reply_to_email", f"reply to {email_id}", reasoning, "failed", str(e))
                    raise ValueError(f"Failed to reply to email: {e}")
            
            elif operation == "create_draft":
                # Validate required parameters
                if not to:
                    raise ValueError("to is required for create_draft operation")
                
                if not subject:
                    raise ValueError("subject is required for create_draft operation")
                
                if not body:
                    raise ValueError("body is required for create_draft operation")
                
                try:
                    # Create the message
                    msg = EmailMessage()
                    msg["Subject"] = subject
                    msg["From"] = self.email_address
                    # Parse email addresses
                    parsed_to = self._parse_email_addresses(to)
                    if parsed_to:
                        msg["To"] = parsed_to
                    else:
                        self.logger.error("No valid email addresses in 'to' field in email_tool")
                        raise ValueError("No valid email addresses in 'to' field")
                    
                    if cc:
                        parsed_cc = self._parse_email_addresses(cc)
                        if parsed_cc:
                            msg["Cc"] = parsed_cc
                    
                    if bcc:
                        parsed_bcc = self._parse_email_addresses(bcc)
                        if parsed_bcc:
                            msg["Bcc"] = parsed_bcc
                    
                    # Set the date
                    msg["Date"] = email.utils.formatdate(localtime=True)
                    
                    # Set the content
                    msg.set_content(body)
                    
                    # Append to drafts folder with \Draft flag
                    self.connection.append(self.drafts_folder, "\\Draft", None, msg.as_bytes())
                    
                    self._log_audit("create_draft", f"draft to={to}, subject={subject}", reasoning, "success")
                    return {
                        "success": True,
                        "to": to,
                        "subject": subject,
                        "operation": "create_draft"
                    }
                except Exception as e:
                    self.logger.error(f"Failed to create draft in email_tool: {e}")
                    self._log_audit("create_draft", f"draft to={to}, subject={subject}", reasoning, "failed", str(e))
                    raise ValueError(f"Failed to create draft: {e}")
            
            elif operation == "search_emails":
                # Build search criteria
                search_parts = []
                
                if unread_only:
                    search_parts.append("UNSEEN")
                
                if sender:
                    search_parts.append(f'FROM "{sender}"')
                
                if subject:
                    search_parts.append(f'SUBJECT "{subject}"')
                
                if start_date:
                    search_parts.append(f'SINCE "{start_date}"')
                
                if end_date:
                    search_parts.append(f'BEFORE "{end_date}"')
                
                # Must have at least one search criterion
                if not search_parts:
                    self.logger.error("No search criteria provided for search_emails operation in email_tool")
                    raise ValueError("At least one search criterion is required")
                
                # Combine criteria with AND
                search_criteria = " ".join(search_parts)
                
                # Execute search and fetch emails
                message_ids = self._search_messages(search_criteria)
                
                # Make sure max_emails is an integer
                if isinstance(max_emails, str):
                    max_emails = int(max_emails)
                    
                emails = self._fetch_message_headers(message_ids, max_emails, load_content)
                
                # LLM handling note for search results
                search_note = """
                For search results:
                1. Display emails in a clear, numbered list
                2. For each email, show: sender, date, subject, and a brief preview
                3. If content is loaded, provide a short summary of each email's purpose
                """
                
                return {
                    "emails": emails,
                    "total": len(message_ids),
                    "showing": len(emails),
                    "criteria": search_criteria,
                    "content_loaded": load_content,
                    "search_note": search_note
                }
            
            elif operation == "list_folders":
                try:
                    # Get list of folders
                    typ, folder_data = self.connection.list()
                    
                    if typ != "OK":
                        self.logger.error("Failed to retrieve folder list in email_tool")
                        raise ValueError("Failed to retrieve folder list")
                    
                    folders = []
                    for item in folder_data:
                        if not item:
                            continue
                        
                        decoded_item = item.decode("utf-8")
                        
                        # Parse folder data
                        match = re.match(r'^\((?P<flags>.*?)\) "(?P<delimiter>.*?)" (?P<name>.+)$', decoded_item)
                        
                        if match:
                            flags = match.group("flags")
                            delimiter = match.group("delimiter")
                            folder_name = match.group("name")
                            
                            # Remove quotes if present
                            if folder_name.startswith('"') and folder_name.endswith('"'):
                                folder_name = folder_name[1:-1]
                            
                            folders.append({
                                "name": folder_name,
                                "flags": flags
                            })
                    
                    return {
                        "folders": folders,
                        "current_folder": self.selected_folder
                    }
                except Exception as e:
                    self.logger.error(f"Error listing folders in email_tool: {e}")
                    raise ValueError(f"Error listing folders: {e}")
            
            elif operation == "mark_for_later_reply":
                # Validate required parameters
                if not email_id:
                    raise ValueError("email_id is required for mark_for_later_reply operation")

                # Validate email_id format
                try:
                    self._parse_email_id(email_id)
                except ValueError as e:
                    raise ValueError(f"Invalid email_id format: {e}")

                self._add_later_reply_id(email_id)

                return {
                    "success": True,
                    "email_id": email_id,
                    "operation": "mark_for_later_reply"
                }

            elif operation == "get_emails_for_later_reply":
                email_ids = sorted(self._load_later_reply_ids())

                # Get details for each email
                emails = []
                for eid in email_ids:
                    try:
                        email_folder, uid = self._parse_email_id(eid)

                        # Select the folder
                        if not self._select_folder(email_folder):
                            self.logger.warning(f"Failed to select folder for later reply email {eid}")
                            continue

                        # Fetch headers using UID
                        typ, data = self.connection.uid("FETCH", str(uid), "(BODY.PEEK[HEADER])")
                        if typ == "OK" and data and data[0]:
                            header_data = data[0][1]
                            parser = email.parser.BytesParser()
                            headers = parser.parsebytes(header_data, headersonly=True)

                            # Get flags
                            flags = self._get_message_flags(uid)

                            # Create header dictionary
                            email_info = {
                                "id": eid,
                                "from": self._decode_header(headers.get("From", "")),
                                "subject": self._decode_header(headers.get("Subject", "")),
                                "date": self._decode_header(headers.get("Date", "")),
                                "flags": flags
                            }

                            emails.append(email_info)
                    except Exception as e:
                        self.logger.error(f"Error fetching headers for later reply email {eid}: {e}")

                # LLM handling note for later reply emails
                later_reply_note = """
                For emails marked for later reply:
                1. Present these emails as a numbered list
                2. Ask the user if they want to reply to any of them now
                3. If there are no emails marked for later reply, inform the user
                """

                return {
                    "emails": emails,
                    "count": len(emails),
                    "later_reply_note": later_reply_note
                }
            
            else:
                self.logger.error(f"Unknown operation '{operation}' in email_tool")
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error executing email_tool operation '{operation}': {e}")
            raise
    
    # ------------------------------------------------------------------
    # Action audit log
    # ------------------------------------------------------------------

    def _ensure_audit_table(self) -> None:
        """Create email_action_audit table if it doesn't exist. Idempotent."""
        self.db.create_table(
            "email_action_audit",
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT NOT NULL DEFAULT (datetime('now')), "
            "operation TEXT NOT NULL, "
            "encrypted__target_summary TEXT, "
            "encrypted__reasoning TEXT NOT NULL, "
            "status TEXT NOT NULL, "
            "encrypted__error_message TEXT",
        )

    def _log_audit(
        self,
        operation: str,
        target_summary: str,
        reasoning: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Log email action to audit table. Non-critical — swallows exceptions."""
        try:
            self._ensure_audit_table()
            self.db.insert("email_action_audit", {
                "operation": operation,
                "encrypted__target_summary": target_summary,
                "encrypted__reasoning": reasoning,
                "status": status,
                "encrypted__error_message": error_message,
            })
        except Exception as e:
            self.logger.error(f"Failed to log email audit: {e}")

    def __del__(self):
        """Ensure we disconnect when the object is destroyed."""
        try:
            self._disconnect()
        except Exception:
            pass  # Suppress errors during destruction


def create_email_tool():
    """
    Factory function to create and initialize an EmailTool instance.
    
    Returns:
        Initialized EmailTool instance
    """
    return EmailTool()
