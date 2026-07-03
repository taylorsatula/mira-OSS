"""
Pager tool for managing virtual pager device messaging.

This tool simulates a pager device system allowing users to send and receive messages
through virtual pagers. Each pager has a unique ID and can send/receive messages with
location tracking and priority levels.

Datetime handling follows the UTC-everywhere approach:
- All datetimes are stored in UTC internally
- Timezone-aware datetime objects are used consistently
- Conversion to local time happens only when displaying to users
- The utility functions from utils.timezone_utils are used consistently
"""

import logging
import json
import uuid
import hashlib
from datetime import timedelta
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from config.config_manager import config
from utils.timezone_utils import (
    get_default_timezone,
    format_datetime, utc_now
)
from clients.sqlite_client import get_sqlite_client
from clients.llm_provider import LLMProvider

# Define configuration class for PagerTool
class PagerToolConfig(BaseModel):
    """Configuration for the pager_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    default_expiry_hours: int = Field(default=24, description="Default message expiry time in hours")
    max_message_length: int = Field(default=300, description="Maximum message length")
    ai_distillation_enabled: bool = Field(default=True, description="Whether to use AI for message distillation")

# Register with registry
registry.register("pager_tool", PagerToolConfig)




class PagerTool(Tool):
    """
    Tool for managing virtual pager devices and messaging.
    
    This tool simulates a pager system where users can create virtual pagers,
    send messages between them, and manage message delivery with priority
    and location tracking.
    """

    name = "pager_tool"
    
    tool_schema = {
        "name": "pager_tool",
        "description": "Send and receive short messages between virtual pager devices.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["register_device", "register_username", "send_message", "get_received_messages", "get_sent_messages", "mark_message_read", "get_devices", "deactivate_device", "cleanup_expired", "list_trusted_devices", "revoke_trust", "send_location"],
                        "description": "The pager operation to perform"
                    },
                    "username": {
                        "type": "string",
                        "description": "Username for federated addressing (3-20 alphanumeric chars, lowercased automatically). Only for register_username"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Friendly name for the pager device (register_device only, required, non-empty)"
                    },
                    "device_description": {
                        "type": "string",
                        "description": "Optional description for the pager device (register_device only)"
                    },
                    "sender_id": {
                        "type": "string",
                        "description": "ID of the sending pager device (dev_XXXXXXXX format). Must be an active device"
                    },
                    "recipient": {
                        "type": "string",
                        "description": "Recipient username (e.g. 'taylor') or federated address (e.g. 'taylor@server'). For send_message. Usernames resolve to the recipient's active pager device"
                    },
                    "content": {
                        "type": "string",
                        "description": "Message body for send_message. Messages over 300 characters are AI-distilled to fit; if distillation is disabled, messages over 300 chars are rejected"
                    },
                    "priority": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "Priority: 0=normal, 1=high, 2=urgent. Default 0 (send_message) or 1 (send_location)"
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location string for send_message. Stored verbatim on the message record"
                    },
                    "expiry_hours": {
                        "type": "integer",
                        "description": "Integer hours until message expiry. Defaults to 24. Only applies to send_message (send_location always uses 6)"
                    },
                    "device_secret": {
                        "type": "string",
                        "description": "Device secret from register_device response. If provided, must match the sender device's stored secret or the call is rejected"
                    },
                    "untrusted_device_id": {
                        "type": "string",
                        "description": "Device ID (dev_XXXXXXXX) to untrust. Used with revoke_trust. Must have an existing trust relationship with pager_id"
                    },
                    "note": {
                        "type": "string",
                        "description": "Short note to attach to a location pin. Max 50 chars; longer values are rejected. send_location only"
                    },
                    "pager_id": {
                        "type": "string",
                        "description": "Pager device ID (dev_XXXXXXXX). Required for: get_received_messages, get_sent_messages, deactivate_device, list_trusted_devices, revoke_trust"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message ID (msg_XXXXXXXX). Used only with mark_message_read"
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Filter to unread messages only. Default false. get_received_messages only"
                    },
                },
                "required": ["operation"]
            }
        }

    simple_description = """
    Manages virtual pager devices for sending and receiving short messages. Use this tool when the user
    wants to simulate pager messaging, send urgent notifications, or manage a virtual pager system."""
    
    implementation_details = """
    
    The tool supports these operations:
    
    1. register_device: Create a new virtual pager device.
       - Required: device_name (friendly name for the pager)
       - Optional: device_description (details about the pager)
       - Returns the created pager device with unique ID (format: dev_XXXXXXXX)

    2. register_username: Register a username for federated pager addressing.
       - Required: username (3-20 alphanumeric characters, e.g., "taylor")
       - One username per user - enables receiving federated messages and local username routing
       - Returns the registered username and federated address (username@server)

    3. send_message: Send a message from one pager to another.
       - Required: sender_id, recipient, content
       - Optional: priority (0=normal, 1=high, 2=urgent), location, expiry_hours
       - Content over 300 chars will be AI-distilled to fit pager constraints
       - Returns the sent message details
    
    3. get_received_messages: Get messages received by a specific pager.
       - Required: pager_id (the pager device ID)
       - Optional: unread_only (boolean), include_expired (boolean)
       - Returns list of received messages
    
    4. get_sent_messages: Get messages sent from a specific pager.
       - Required: pager_id (the pager device ID)
       - Optional: include_expired (boolean)
       - Returns list of sent messages
       
    5. mark_message_read: Mark a message as read.
       - Required: message_id
       - Returns the updated message
       
    6. get_devices: List all registered pager devices.
       - Optional: active_only (boolean, default true)
       - Returns list of pager devices
       
    7. deactivate_device: Deactivate a pager device.
       - Required: pager_id
       - Returns confirmation of deactivation
       
    8. cleanup_expired: Remove expired messages from the system.
       - Returns count of messages cleaned up
    
    9. list_trusted_devices: List devices trusted by a specific pager.
       - Required: pager_id
       - Returns list of trusted devices with fingerprints and status
       
    10. revoke_trust: Revoke trust for a specific device (allows re-establishing trust).
        - Required: pager_id, untrusted_device_id
        - Returns confirmation of trust revocation
        
    11. send_location: Send a location pin message from one pager to another.
        - Required: sender_id, recipient
        - Optional: priority (0=normal, 1=high, 2=urgent), note, device_secret
        - Automatically includes current location as a pin
        - Returns the sent location message
       
    The tool uses AI to automatically distill long messages to fit pager constraints while
    preserving the essential information. Location information can be attached to messages
    for context, and priority levels help indicate message urgency.
    
    Location pins are a special feature that allow sending your current coordinates as a 
    message, perfect for emergencies or meetups. The location is formatted with both a 
    human-readable address and technical coordinates.
    """
    
    description = simple_description + implementation_details

    @staticmethod
    def _pager_name(device_row: Dict[str, Any]) -> Optional[str]:
        """Return display name from the encrypted schema."""
        return device_row.get("encrypted__name")

    @staticmethod
    def _pager_description(device_row: Dict[str, Any]) -> Optional[str]:
        """Return display description from the encrypted schema."""
        return device_row.get("encrypted__description")

    @staticmethod
    def _trusted_name(trust_row: Dict[str, Any]) -> Optional[str]:
        """Return trusted device display name from the encrypted schema."""
        return trust_row.get("encrypted__trusted_name")
    
    usage_examples = [
        {
            "input": {
                "operation": "register_device",
                "device_name": "Field Unit Alpha",
                "device_description": "Emergency response team leader"
            },
            "output": {
                "device": {
                    "id": "dev_a1b2c3d4",
                    "name": "Field Unit Alpha",
                    "description": "Emergency response team leader",
                    "active": True
                }
            }
        },
        {
            "input": {
                "operation": "send_message",
                "sender_id": "dev_a1b2c3d4",
                "recipient": "taylor",
                "content": "Code 3 response needed at Main St",
                "priority": 2,
                "location": "Main St & 5th Ave"
            },
            "output": {
                "message": {
                    "id": "msg_12345678",
                    "sender_id": "dev_a1b2c3d4",
                    "recipient_id": "dev_c3d4e5f6",
                    "content": "Code 3 response needed at Main St",
                    "priority": 2,
                    "priority_label": "urgent",
                    "location": "Main St & 5th Ave"
                }
            }
        },
        {
            "input": {
                "operation": "send_location",
                "sender_id": "dev_a1b2c3d4",
                "recipient": "taylor",
                "note": "Stuck in traffic",
                "priority": 2
            },
            "output": {
                "message": {
                    "id": "msg_87654321",
                    "content": "Location Pin: 1-1 Kitahama, Chuo-ku, Osaka (Near Osaka City Hall)\\nNote: Stuck in traffic\\n[34.6937, 135.5023]",
                    "priority": 2,
                    "location": "{\"lat\": 34.6937, \"lng\": 135.5023, \"accuracy_meters\": 15}"
                }
            }
        }
    ]

    def __init__(self):
        """Initialize the pager tool with database access and LLM provider."""
        super().__init__()
        self.llm = LLMProvider()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Get config
        tool_config = config.get_tool_config(self.name)
        self.default_expiry_hours = tool_config.default_expiry_hours
        self.max_message_length = tool_config.max_message_length
        self.ai_distillation_enabled = tool_config.ai_distillation_enabled
        
        self.logger.info("PagerTool initialized")

    def deliver_federated_message(
        self,
        from_address: str,
        content: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None,
        external_message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deliver a federated message to this user's pager (write-only, no read access).

        This is a security-bounded instance method that ONLY allows writing messages to
        the user's database. The federation adapter calls this method and has NO ability
        to read existing messages, maintaining strict data isolation.

        Each user has exactly one active pager device - this method looks it up automatically.

        Args:
            from_address: Federated sender address (e.g., "user@remote-server")
            content: Message content (already filtered for prompt injection)
            priority: Message priority (0=normal, 1=high, 2=urgent)
            metadata: Optional metadata dict (location, etc.)

        Returns:
            Dict with success status and message_id or error

        Raises:
            ValueError: If parameters are invalid or delivery fails
        """
        logger = logging.getLogger(__name__)

        try:
            if external_message_id:
                existing = self.db.select(
                    'pager_messages',
                    'external_message_id = :external_message_id',
                    {'external_message_id': external_message_id}
                )
                if existing:
                    return {
                        'success': True,
                        'message_id': existing[0]['id'],
                        'delivered_to': existing[0]['recipient_id'],
                        'duplicate': True
                    }

            # Look up the user's single active pager device
            pager_devices = self.db.select(
                'pager_devices',
                'user_id = :user_id AND active = :active',
                {'user_id': self.user_id, 'active': 1}
            )

            if not pager_devices or len(pager_devices) == 0:
                raise ValueError(f"No active pager device found for user {self.user_id}")

            pager_id = pager_devices[0]['id']

            # Generate message ID
            message_id = f"msg_{uuid.uuid4().hex[:8]}"

            # Create federated message record
            message_data = {
                'id': message_id,
                'user_id': self.user_id,
                'sender_id': from_address,  # Federated address as sender
                'recipient_id': pager_id,
                'encrypted__content': content,
                'encrypted__original_content': None,
                'ai_distilled': 0,
                'priority': priority,
                'location': json.dumps(metadata.get('location')) if metadata and 'location' in metadata else None,
                'sent_at': utc_now().isoformat(),
                'expires_at': (utc_now() + timedelta(hours=24)).isoformat(),
                'delivered': 1,
                'read': 0,
                'sender_fingerprint': 'FEDERATED',  # Mark as federated source
                'message_signature': '',  # No signature for federated messages
                'external_message_id': external_message_id
            }

            # Insert through UserDataManager so encrypted__ fields are encrypted at rest.
            try:
                self.db.insert('pager_messages', message_data)
            except Exception:
                if external_message_id:
                    existing = self.db.select(
                        'pager_messages',
                        'external_message_id = :external_message_id',
                        {'external_message_id': external_message_id}
                    )
                    if existing:
                        return {
                            'success': True,
                            'message_id': existing[0]['id'],
                            'delivered_to': existing[0]['recipient_id'],
                            'duplicate': True
                        }
                raise

            logger.info(
                f"Delivered federated message {message_id} from {from_address} "
                f"to user {self.user_id} pager {pager_id}"
            )

            return {
                'success': True,
                'message_id': message_id,
                'delivered_to': pager_id
            }

        except ValueError as e:
            logger.warning(f"Rejected federated message for user {self.user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _device_to_dict(self, device_row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert device row to dictionary (UTC timestamps, frontend handles display conversion)."""
        return {
            "id": device_row["id"],
            "name": self._pager_name(device_row),
            "description": self._pager_description(device_row),
            "created_at": device_row["created_at"],  # UTC timestamp
            "last_active": device_row["last_active"],  # UTC timestamp
            "active": device_row["active"],
            "device_fingerprint": device_row["device_fingerprint"]
        }

    def _decrypt_row_safely(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted__ fields without failing on rows already decrypted by select()."""
        decrypted = {}
        for key, value in row.items():
            if key.startswith("encrypted__") and value is not None:
                try:
                    decrypted[key] = self.db._decrypt_value(value)
                except Exception:
                    decrypted[key] = value
            else:
                decrypted[key] = value
        return decrypted
    
    def _message_to_dict(self, message_row: Dict[str, Any], sender_name: str = None, recipient_name: str = None) -> Dict[str, Any]:
        """Convert message row to dictionary (UTC timestamps, frontend handles display conversion)."""
        decrypted = self._decrypt_row_safely(message_row)
        sender_name = sender_name if sender_name is not None else decrypted.get("encrypted__sender_name")
        recipient_name = recipient_name if recipient_name is not None else decrypted.get("encrypted__recipient_name")
        return {
            "id": decrypted["id"],
            "sender_id": decrypted["sender_id"],
            "sender_name": sender_name,
            "recipient_id": decrypted["recipient_id"],
            "recipient_name": recipient_name,
            "content": decrypted["encrypted__content"],
            "original_content": decrypted["encrypted__original_content"],
            "ai_distilled": decrypted["ai_distilled"],
            "priority": decrypted["priority"],
            "priority_label": ["normal", "high", "urgent"][decrypted["priority"]] if 0 <= decrypted["priority"] <= 2 else "unknown",
            "location": decrypted["location"],
            "sent_at": decrypted["sent_at"],  # UTC timestamp
            "expires_at": decrypted["expires_at"],  # UTC timestamp
            "read_at": decrypted["read_at"],  # UTC timestamp
            "delivered": decrypted["delivered"],
            "read": decrypted["read"],
            "sender_fingerprint": decrypted["sender_fingerprint"],
            "message_signature": decrypted["message_signature"]
        }

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a pager operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ValueError: If operation fails or parameters are invalid

        Valid Operations:

        1. register_device: Create a new pager device
           - Required: device_name
           - Optional: description
           - Returns: Dict with created device

        2. send_message: Send a message between pagers
           - Required: sender_id, recipient, content
           - Optional: priority, location, expiry_hours
           - Returns: Dict with sent message

        3. get_received_messages: Get messages for a pager
           - Required: pager_id
           - Optional: unread_only, include_expired
           - Returns: Dict with list of messages

        4. get_sent_messages: Get messages sent by a pager
           - Required: pager_id
           - Optional: include_expired
           - Returns: Dict with list of messages

        5. mark_message_read: Mark a message as read
           - Required: message_id
           - Returns: Dict with updated message

        6. get_devices: List pager devices
           - Optional: active_only
           - Returns: Dict with list of devices

        7. deactivate_device: Deactivate a pager
           - Required: pager_id
           - Returns: Dict with confirmation

        8. cleanup_expired: Clean up expired messages
           - Returns: Dict with cleanup stats
        """
        try:
            # Route to the appropriate operation
            if operation == "register_device":
                return self._register_device(**kwargs)
            elif operation == "send_message":
                return self._send_message(**kwargs)
            elif operation == "get_received_messages":
                return self._get_received_messages(**kwargs)
            elif operation == "get_sent_messages":
                return self._get_sent_messages(**kwargs)
            elif operation == "mark_message_read":
                return self._mark_message_read(**kwargs)
            elif operation == "get_devices":
                return self._get_devices(**kwargs)
            elif operation == "deactivate_device":
                return self._deactivate_device(**kwargs)
            elif operation == "cleanup_expired":
                return self._cleanup_expired(**kwargs)
            elif operation == "list_trusted_devices":
                return self._list_trusted_devices(**kwargs)
            elif operation == "revoke_trust":
                return self._revoke_trust(**kwargs)
            elif operation == "send_location":
                return self._send_location(**kwargs)
            elif operation == "register_username":
                return self._register_username(**kwargs)
            else:
                self.logger.error(f"Unknown operation: {operation}")
                raise ValueError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "register_device, register_username, send_message, get_received_messages, "
                    "get_sent_messages, mark_message_read, get_devices, "
                    "deactivate_device, cleanup_expired, list_trusted_devices, "
                    "revoke_trust, send_location"
                )
        except Exception as e:
            self.logger.error(f"Error executing {operation} in pager_tool: {e}")
            raise

    def _register_device(
        self,
        device_name: str,
        device_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new pager device.

        Args:
            device_name: Friendly name for the pager device
            device_description: Optional description of the device

        Returns:
            Dict containing the created device

        Raises:
            ValueError: If required fields are missing
        """

        # Validate required parameters
        if not device_name:
            self.logger.error("Name is required for registering a pager device")
            raise ValueError("Name is required for registering a pager device")

        # Generate a unique pager ID
        pager_id = f"dev_{uuid.uuid4().hex[:8]}"

        # Generate device secret and fingerprint
        device_secret = f"SECRET-{uuid.uuid4().hex[:32].upper()}"
        # Fingerprint is hash of device ID + secret (like SSH host key fingerprint)
        device_fingerprint = hashlib.sha256(
            f"{pager_id}{device_secret}".encode()
        ).hexdigest()[:16].upper()

        # Insert device into database
        try:
            device_data = {
                "id": pager_id,
                "user_id": self.user_id,
                "encrypted__name": device_name,
                "encrypted__description": device_description,
                "created_at": utc_now().isoformat(),
                "last_active": utc_now().isoformat(),
                "active": 1,  # SQLite uses 1/0 for boolean
                "device_secret": device_secret,
                "device_fingerprint": device_fingerprint
            }

            row_id = self.db.insert('pager_devices', device_data)
            # Get the inserted row
            result = self.db.select('pager_devices', 'id = :pager_id', {'pager_id': pager_id})[0]

        except Exception as e:
            self.logger.error(f"Error saving pager device: {e}")
            self.logger.error(f"Failed to save pager device: {e}")
            raise ValueError(f"Failed to save pager device: {str(e)}")

        return {
            "device": self._device_to_dict(result),
            "message": f"Pager device '{device_name}' registered successfully with ID {pager_id}"
        }

    def _get_lattice_identity(self) -> "LatticeIdentity":
        """
        Get server identity from Lattice discovery daemon.

        Returns:
            LatticeIdentity with server_id, server_uuid, fingerprint, public_key

        Raises:
            ValueError: If Lattice service unavailable or not configured
        """
        from clients.lattice_client import get_lattice_client, LatticeIdentity
        import httpx

        try:
            client = get_lattice_client()
            return client.get_identity()
        except httpx.ConnectError:
            raise ValueError(
                "Cannot connect to Lattice discovery daemon. "
                "Ensure the lattice-discovery service is running."
            )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to get Lattice identity: {e}")

    def _register_username(self, username: str) -> Dict[str, Any]:
        """
        Register a username for the current user in the global registry.

        This enables:
        - Receiving federated messages addressed to username@server
        - Local routing via username instead of pager device UUID
        - Cross-server identity

        Args:
            username: Desired username (3-20 alphanumeric characters)

        Returns:
            Dict with registered username and federated address

        Raises:
            ValueError: If username invalid or already taken
        """
        # Validate username format
        if not username:
            raise ValueError("Username is required")

        username = username.lower().strip()

        if not username.isalnum():
            raise ValueError("Username must contain only letters and numbers (no spaces or special characters)")

        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")

        if len(username) > 20:
            raise ValueError("Username must be at most 20 characters")

        # Check if user already has a username (MIRA's user registry)
        from clients.postgres_client import PostgresClient
        postgres_db = PostgresClient("mira_service")

        existing = postgres_db.execute_single(
            """
            SELECT username
            FROM global_usernames
            WHERE user_id = %(user_id)s
              AND active = true
            """,
            {'user_id': self.user_id}
        )

        if existing:
            raise ValueError(
                f"You already have a registered username: {existing['username']}. "
                f"Each user can only have one username."
            )

        # Check if username is available
        availability = postgres_db.execute_single(
            """
            SELECT user_id
            FROM global_usernames
            WHERE username = %(username)s
              AND active = true
            """,
            {'username': username}
        )

        if availability:
            raise ValueError(f"Username '{username}' is already taken")

        # Register the username
        try:
            postgres_db.execute_insert(
                """
                INSERT INTO global_usernames (username, user_id, created_at, active)
                VALUES (%(username)s, %(user_id)s, NOW(), true)
                """,
                {'username': username, 'user_id': self.user_id}
            )

            # Get our server's domain from Lattice
            try:
                identity = self._get_lattice_identity()
                federated_address = f"{username}@{identity['server_id']}"
            except ValueError:
                federated_address = f"{username}@[lattice-not-configured]"

            self.logger.info(f"Registered username '{username}' for user {self.user_id}")

            return {
                "success": True,
                "username": username,
                "federated_address": federated_address,
                "message": f"Username '{username}' registered successfully. Your federated address is: {federated_address}"
            }

        except Exception as e:
            self.logger.error(f"Failed to register username '{username}': {e}")
            raise ValueError(f"Failed to register username: {e}")

    def _resolve_recipient_to_pager_id(self, recipient_address: str) -> str:
        """
        Resolve recipient address to pager device ID (internal use only).

        Args:
            recipient_address: Username (e.g., "taylor")

        Returns:
            Pager device ID

        Raises:
            ValueError: If recipient cannot be resolved
        """
        from clients.postgres_client import PostgresClient
        postgres_db = PostgresClient("mira_service")

        # Resolve username to user_id via global_usernames
        username_result = postgres_db.execute_single(
            """
            SELECT user_id
            FROM global_usernames
            WHERE username = %(username)s
              AND active = true
            """,
            {'username': recipient_address.lower()}
        )

        if not username_result:
            raise ValueError(f"Username '{recipient_address}' not found")

        user_id = str(username_result['user_id'])

        # Look up the user's pager device
        pager = self.db.select(
            'pager_devices',
            'user_id = :user_id AND active = :active',
            {'user_id': user_id, 'active': 1}
        )

        if not pager or len(pager) == 0:
            raise ValueError(f"No active pager found for username '{recipient_address}'")

        return pager[0]['id']

    def _send_federated_message(
        self,
        sender_pager_id: str,
        recipient_address: str,
        content: str,
        priority: int,
        location: Optional[str],
        device_secret: Optional[str]
    ) -> Dict[str, Any]:
        """
        Route a message through Lattice for remote delivery.

        Args:
            sender_pager_id: Sender's pager device ID (internal)
            recipient_address: Federated address (user@domain)
            content: Message content
            priority: Message priority
            location: Optional location data
            device_secret: Device secret for authentication

        Returns:
            Dict with message delivery status
        """
        # Verify sender device
        sender = self.db.select(
            'pager_devices',
            'id = :sender_id',
            {'sender_id': sender_pager_id}
        )
        if not sender or not sender[0]['active']:
            raise ValueError(f"Sender device not found or inactive")
        sender = sender[0]

        # Verify device secret
        if device_secret and device_secret != sender['device_secret']:
            raise ValueError("Invalid device secret")

        # Get our server's domain from Lattice
        identity = self._get_lattice_identity()
        our_domain = identity['server_id']

        # Get username for this user (from MIRA's user registry)
        from clients.postgres_client import PostgresClient
        postgres_db = PostgresClient("mira_service")

        username_result = postgres_db.execute_single(
            """
            SELECT username
            FROM global_usernames
            WHERE user_id = %(user_id)s
              AND active = true
            """,
            {'user_id': sender['user_id']}
        )

        if not username_result:
            raise ValueError("Sender must have a registered username for federated messaging")

        from_address = f"{username_result['username']}@{our_domain}"

        # Convert location string to dict if provided
        location_dict = None
        if location:
            try:
                location_dict = json.loads(location) if isinstance(location, str) else location
            except:
                location_dict = {'raw': location}

        # Send via Lattice HTTP client
        from clients.lattice_client import get_lattice_client

        client = get_lattice_client()

        try:
            result = client.send_message(
                from_address=from_address,
                to_address=recipient_address,
                content=content,
                priority=priority,
                metadata={"location": location_dict} if location_dict else None
            )

            self.logger.info(
                f"Queued federated message from {from_address} to {recipient_address} "
                f"(message_id: {result['message_id']})"
            )

            return {
                "success": True,
                "message_id": result["message_id"],
                "status": "queued_for_federation",
                "from_address": from_address,
                "to_address": recipient_address,
                "message": f"Message queued for delivery to {recipient_address}"
            }

        except Exception as e:
            self.logger.error(f"Failed to queue federated message: {e}")
            raise ValueError(f"Failed to queue federated message: {e}")

    def _send_message(
        self,
        sender_id: str,
        recipient: str,
        content: str,
        priority: Optional[int] = 0,
        location: Optional[str] = None,
        expiry_hours: Optional[int] = 24,
        device_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to a recipient (local username or federated address).

        Recipient addressing formats:
        1. Local username: "taylor" - delivers to local user's pager
        2. Federated address: "alex@otherserver" - routes through federation

        Args:
            sender_id: ID of the sending pager device (dev_XXXXXXXX)
            recipient: Recipient username or federated address
            content: Message content (will be distilled if too long)
            priority: Message priority (0=normal, 1=high, 2=urgent)
            location: Optional location information
            expiry_hours: Hours until message expires (default 24)
            device_secret: Device secret for authentication

        Returns:
            Dict containing the sent message

        Raises:
            ValueError: If recipient not found or parameters invalid
        """

        # Route federated messages through federation adapter
        if '@' in recipient:
            return self._send_federated_message(
                sender_pager_id=sender_id,
                recipient_address=recipient,
                content=content,
                priority=priority,
                location=location,
                device_secret=device_secret
            )

        # For local delivery, resolve username to pager device ID
        recipient_pager_id = self._resolve_recipient_to_pager_id(recipient)

        # Validate required parameters
        if not all([sender_id, recipient_pager_id, content]):
            self.logger.error("sender_id, recipient, and content are required for send_message")
            raise ValueError("sender_id, recipient, and content are required")

        # Validate priority
        if priority not in [0, 1, 2]:
            self.logger.error(f"Invalid priority: {priority}. Must be 0, 1, or 2")
            raise ValueError("Priority must be 0 (normal), 1 (high), or 2 (urgent)")

        # Get sender and recipient devices
        sender = self.db.select(
            'pager_devices',
            'id = :sender_id',
            {'sender_id': sender_id}
        )
        if not sender or not sender[0]['active']:
            self.logger.error(f"Sender device not found or inactive")
            raise ValueError(f"Sender device not found or inactive")
        sender = sender[0]

        # Verify device secret (optional but recommended)
        if device_secret and device_secret != sender['device_secret']:
            self.logger.error(f"Invalid device secret")
            raise ValueError(f"Invalid device secret")

        recipient_device = self.db.select(
            'pager_devices',
            'id = :recipient_id',
            {'recipient_id': recipient_pager_id}
        )
        if not recipient_device or not recipient_device[0]['active']:
            self.logger.error(f"Recipient pager not found or inactive")
            raise ValueError(f"Recipient pager not found or inactive")
        recipient_device = recipient_device[0]
            
        # Update sender's last active time
        self.db.update(
            'pager_devices',
            {'last_active': utc_now().isoformat()},
            'id = :sender_id',
            {'sender_id': sender_id}
        )
        
        # Check if content needs distillation
        original_content = None
        ai_distilled = False
        
        max_length = self.max_message_length
        
        if len(content) > max_length:
            if self.ai_distillation_enabled:
                original_content = content
                content = self._distill_message(content, max_length)
                ai_distilled = True
            else:
                # Enforce character limit for human messages without AI
                raise ValueError(f"Message too long: {len(content)} characters (max {max_length})")
            
        # Calculate expiry time
        expires_at = utc_now() + timedelta(hours=expiry_hours)
        
        # Generate message ID
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        # Create message signature using device secret
        message_signature = hashlib.sha256(
            f"{message_id}{sender['device_secret']}{content}{recipient_pager_id}".encode()
        ).hexdigest()[:16].upper()
        
        # Create the message data
        message_data = {
            "id": message_id,
            "user_id": self.user_id,
            "sender_id": sender_id,
            "recipient_id": recipient_pager_id,
            "encrypted__content": content,
            "encrypted__original_content": original_content,
            "ai_distilled": 1 if ai_distilled else 0,  # SQLite boolean
            "priority": priority,
            "location": location,
            "sent_at": utc_now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "delivered": 1,  # SQLite boolean
            "read": 0,  # SQLite boolean
            "message_signature": message_signature,
            "sender_fingerprint": sender['device_fingerprint']
        }
        
        # Save message to database
        try:
            row_id = self.db.insert('pager_messages', message_data)
            # Get the inserted message
            message_result = self.db.select('pager_messages', 'id = :message_id', {'message_id': message_id})[0]
            self.logger.info(f"Sent message with ID: {message_id}")
        except Exception as e:
            self.logger.error(f"Error saving message: {e}")
            raise RuntimeError(f"Failed to send message: {str(e)}") from e
            
        result = {
            "message": self._message_to_dict(
                message_result,
                self._pager_name(sender),
                self._pager_name(recipient_device),
            ),
            "status": "delivered"
        }
        
        if ai_distilled:
            result["distillation_note"] = f"Message was distilled from {len(original_content)} to {len(content)} characters"
            
        return result

    def _get_received_messages(
        self,
        pager_id: str,
        unread_only: bool = False
    ) -> Dict[str, Any]:
        """
        Get messages received by a pager device.
        
        Args:
            pager_id: ID of the pager device
            unread_only: Only return unread messages
            
        Returns:
            Dict containing list of received messages
            
        Raises:
            ValueError: If device not found
        """
        self.logger.info(f"Getting received messages for pager {pager_id}")
        
        # Validate device exists
        device = self.db.select(
            'pager_devices',
            'id = :pager_id',
            {'pager_id': pager_id}
        )
        if not device:
            raise ValueError(f"Pager device '{pager_id}' not found")
        device = device[0]
            
        # Update last active time
        self.db.update(
            'pager_devices',
            {'last_active': utc_now().isoformat()},
            'id = :pager_id',
            {'pager_id': pager_id}
        )
        
        # Query messages
        query = """
        SELECT m.*,
               s.encrypted__name as encrypted__sender_name,
               r.encrypted__name as encrypted__recipient_name
        FROM pager_messages m
        LEFT JOIN pager_devices s ON m.sender_id = s.id
        LEFT JOIN pager_devices r ON m.recipient_id = r.id
        WHERE m.user_id = :user_id AND m.recipient_id = :pager_id
        """
        params = {'user_id': self.user_id, 'pager_id': pager_id}
        
        if unread_only:
            query += " AND m.read = 0"
            
        # Always filter out expired messages
        query += " AND m.expires_at > datetime('now')"
            
        # Sort by sent time descending (newest first)
        query += " ORDER BY m.sent_at DESC"
        
        messages = self.db.execute(query, params)
        
        # Check trust status for each message
        message_list = []
        for msg in messages:
            msg_dict = self._message_to_dict(msg)
            
            # Check if we trust this sender
            trust_status = self._check_trust_status(
                pager_id, 
                msg['sender_id'], 
                msg['sender_fingerprint']
            )
            msg_dict['trust_status'] = trust_status
            
            # Note: Conflicted messages will never reach here as they're rejected during send
            message_list.append(msg_dict)
            
        return {
            "messages": message_list,
            "count": len(message_list),
            "pager_id": pager_id,
            "pager_name": self._pager_name(device),
            "filters": {
                "unread_only": unread_only
            }
        }

    def _get_sent_messages(
        self,
        pager_id: str,
        include_expired: bool = False
    ) -> Dict[str, Any]:
        """
        Get messages sent by a pager device.
        
        Args:
            pager_id: ID of the pager device
            include_expired: Include expired messages
            
        Returns:
            Dict containing list of sent messages
            
        Raises:
            ValueError: If device not found
        """
        self.logger.info(f"Getting sent messages for pager {pager_id}")
        
        # Validate device exists
        device = self.db.select(
            'pager_devices',
            'id = :pager_id',
            {'pager_id': pager_id}
        )
        if not device:
            raise ValueError(f"Pager device '{pager_id}' not found")
        device = device[0]
            
        # Query messages
        query = """
        SELECT m.*,
               s.encrypted__name as encrypted__sender_name,
               r.encrypted__name as encrypted__recipient_name
        FROM pager_messages m
        LEFT JOIN pager_devices s ON m.sender_id = s.id
        LEFT JOIN pager_devices r ON m.recipient_id = r.id
        WHERE m.user_id = :user_id AND m.sender_id = :pager_id
        """
        params = {'user_id': self.user_id, 'pager_id': pager_id}
        
        if not include_expired:
            query += " AND m.expires_at > datetime('now')"
            
        # Sort by sent time descending (newest first)
        query += " ORDER BY m.sent_at DESC"
        
        messages = self.db.execute(query, params)
        message_list = [self._message_to_dict(msg) for msg in messages]
            
        return {
            "messages": message_list,
            "count": len(message_list),
            "pager_id": pager_id,
            "pager_name": self._pager_name(device),
            "filters": {
                "include_expired": include_expired
            }
        }

    def _mark_message_read(self, message_id: str) -> Dict[str, Any]:
        """
        Mark a message as read.
        
        Args:
            message_id: ID of the message to mark as read
            
        Returns:
            Dict containing the updated message
            
        Raises:
            ValueError: If message not found
        """
        self.logger.info(f"Marking message {message_id} as read")
        
        # Get the message
        messages = self.db.select(
            'pager_messages',
            'id = :message_id',
            {'message_id': message_id}
        )
        if not messages:
            raise ValueError(f"Message with ID '{message_id}' not found")
        message = messages[0]
            
        # Update message
        read_at = utc_now().isoformat()
        update_result = self.db.update(
            'pager_messages',
            {'read': 1, 'read_at': read_at},
            'id = :message_id',
            {'message_id': message_id}
        )
        
        # Update recipient device's last active time
        self.db.update(
            'pager_devices',
            {'last_active': utc_now().isoformat()},
            'id = :recipient_id',
            {'recipient_id': message['recipient_id']}
        )
        
        # Get updated message for response
        updated_messages = self.db.select(
            'pager_messages',
            'id = :message_id',
            {'message_id': message_id}
        )
        
        if updated_messages:
            updated_message = updated_messages[0]
            self.logger.info(f"Marked message {message_id} as read")
            return {
                "message": self._message_to_dict(updated_message),
                "status": "Message marked as read"
            }
        else:
            raise RuntimeError(f"Failed to retrieve updated message: {message_id}")

    def _get_devices(self, active_only: bool = True) -> Dict[str, Any]:
        """
        Get list of pager devices.
        
        Args:
            active_only: Only return active devices
            
        Returns:
            Dict containing list of devices
        """
        self.logger.info(f"Getting pager devices (active_only={active_only})")
        
        # Query devices
        if active_only:
            devices = self.db.select(
                'pager_devices',
                'active = :active',
                {'active': 1}
            )
        else:
            devices = self.db.select('pager_devices')
        device_list = [self._device_to_dict(device) for device in devices]
        
        # Sort by last active descending
        device_list.sort(key=lambda x: x.get('last_active', ''), reverse=True)
        
        return {
            "devices": device_list,
            "count": len(device_list),
            "filters": {
                "active_only": active_only
            }
        }

    def _deactivate_device(self, pager_id: str) -> Dict[str, Any]:
        """
        Deactivate a pager device.
        
        Args:
            pager_id: ID of the device to deactivate
            
        Returns:
            Dict containing confirmation
            
        Raises:
            ValueError: If device not found
        """
        self.logger.info(f"Deactivating pager device {pager_id}")
        
        # Get the device
        devices = self.db.select(
            'pager_devices',
            'id = :pager_id',
            {'pager_id': pager_id}
        )
        if not devices:
            raise ValueError(f"Pager device '{pager_id}' not found")
        device = devices[0]
            
        # Update device
        self.db.update(
            'pager_devices',
            {'active': 0},
            'id = :pager_id',
            {'pager_id': pager_id}
        )
        
        self.logger.info(f"Deactivated pager device {pager_id}")
        
        # Get updated device for response
        updated_devices = self.db.select(
            'pager_devices',
            'id = :pager_id',
            {'pager_id': pager_id}
        )
        
        if updated_devices:
            updated_device = updated_devices[0]
            return {
                "device": self._device_to_dict(updated_device),
                "message": (
                    "Pager device "
                    f"'{self._pager_name(updated_device)}' "
                    "deactivated successfully"
                )
            }
        else:
            raise RuntimeError(f"Failed to retrieve updated device: {pager_id}")

    def _cleanup_expired(self) -> Dict[str, Any]:
        """
        Clean up expired messages from the system.
        
        Returns:
            Dict containing cleanup statistics
        """
        self.logger.info("Cleaning up expired messages")
        
        # Count expired messages first
        current_time = utc_now().isoformat()
        expired_messages = self.db.select(
            'pager_messages',
            'expires_at < :current_time',
            {'current_time': current_time}
        )
        expired_count = len(expired_messages)
        
        # Delete expired messages
        if expired_count > 0:
            self.db.execute(
                'DELETE FROM pager_messages WHERE expires_at < :current_time',
                {'current_time': current_time}
            )
                
        self.logger.info(f"Cleaned up {expired_count} expired messages")
        
        return {
            "expired_messages_removed": expired_count,
            "message": f"Cleaned up {expired_count} expired message(s)"
        }

    def _distill_message(self, content: str, max_length: int) -> str:
        """
        Use AI to distill a long message to fit pager constraints.
        
        Args:
            content: Original message content
            max_length: Maximum allowed length
            
        Returns:
            Distilled message content
        """
        prompt = f"""Distill the following message to fit within {max_length} characters while preserving all critical information. Focus on actionable content and key details. Remove unnecessary words but keep all important facts, numbers, names, and instructions.

Original message:
{content}

Provide ONLY the distilled message, no explanations or meta-text."""

        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                internal_llm='tidyup',
                max_tokens=200
            )

            distilled = self.llm.extract_text_content(response).strip()
            
            # Ensure it fits within max_length
            if len(distilled) > max_length:
                distilled = distilled[:max_length-3] + "..."
                
            return distilled
            
        except Exception as e:
            self.logger.warning(f"AI distillation failed: {e}, truncating instead")
            # Fallback to simple truncation
            return content[:max_length-3] + "..."
    
    def _check_trust_status(self, trusting_device_id: str, sender_id: str, sender_fingerprint: str) -> str:
        """
        Check the trust status of a sender for a given device.
        
        Args:
            trusting_device_id: The device receiving the message
            sender_id: The device that sent the message
            sender_fingerprint: The fingerprint claimed by the sender
            
        Returns:
            Trust status: "trusted", "untrusted", "conflicted", or "first_contact"
        """
        # Query for existing trust relationship
        trusts = self.db.select(
            'pager_trust',
            'trusting_device_id = :trusting_device_id AND trusted_device_id = :trusted_device_id',
            {'trusting_device_id': trusting_device_id, 'trusted_device_id': sender_id}
        )
        
        if not trusts or len(trusts) == 0:
            # First contact - add to trust store
            self._add_trust_relationship(trusting_device_id, sender_id, sender_fingerprint)
            return "first_contact"
        
        trust = trusts[0]
        
        if trust['trust_status'] == "revoked":
            return "revoked"
        
        if trust['trusted_fingerprint'] != sender_fingerprint:
            # Fingerprint mismatch! This is a security threat - reject the message completely
            self.db.update(
                'pager_trust',
                {'trust_status': 'conflicted'},
                'id = :trust_id',
                {'trust_id': trust['id']}
            )
            self.logger.error(
                f"SECURITY BREACH: Fingerprint mismatch for {sender_id} on device {trusting_device_id}. "
                f"Expected: {trust['trusted_fingerprint']}, Got: {sender_fingerprint}. MESSAGE REJECTED."
            )
            raise PermissionError(
                f"MESSAGE DELIVERY FAILED: Device {sender_id} fingerprint mismatch detected! "
                f"This could indicate an impersonation attempt. The message has been rejected for security. "
                f"If this device legitimately changed, the recipient must use 'revoke_trust' for device {sender_id} "
                f"and then you can send a new message to re-establish trust."
            )
        
        # Update last verified time
        self.db.update(
            'pager_trust',
            {'last_verified': utc_now().isoformat()},
            'id = :trust_id',
            {'trust_id': trust['id']}
        )
        
        return "trusted"
    
    def _add_trust_relationship(self, trusting_device_id: str, trusted_device_id: str, trusted_fingerprint: str) -> None:
        """
        Add a new trust relationship (TOFU - Trust on First Use).
        
        Args:
            trusting_device_id: The device that trusts
            trusted_device_id: The device being trusted
            trusted_fingerprint: The fingerprint to trust
        """
        # Get sender device info if available
        senders = self.db.select(
            'pager_devices',
            'id = :trusted_device_id',
            {'trusted_device_id': trusted_device_id}
        )
        sender_name = (
            self._pager_name(senders[0])
            if senders else "Unknown"
        )
        
        trust_data = {
            'id': f"trust_{uuid.uuid4().hex[:8]}",
            'user_id': self.user_id,
            'trusting_device_id': trusting_device_id,
            'trusted_device_id': trusted_device_id,
            'trusted_fingerprint': trusted_fingerprint,
            'encrypted__trusted_name': sender_name,
            'first_seen': utc_now().isoformat(),
            'last_verified': utc_now().isoformat(),
            'trust_status': "trusted"
        }
        
        try:
            self.db.insert('pager_trust', trust_data)
            self.logger.info(f"Added trust relationship: {trusting_device_id} trusts {trusted_device_id}")
        except Exception as e:
            self.logger.warning(f"Failed to add trust relationship: {e}")
    
    def _list_trusted_devices(self, pager_id: str) -> Dict[str, Any]:
        """
        List all devices trusted by a specific pager.
        
        Args:
            pager_id: ID of the pager device
            
        Returns:
            Dict containing list of trusted devices
        """
        devices = self.db.select(
            'pager_devices',
            'id = :pager_id',
            {'pager_id': pager_id}
        )
        if not devices:
            raise ValueError(f"Pager device '{pager_id}' not found")
        device = devices[0]
        
        trusts = self.db.select(
            'pager_trust',
            'trusting_device_id = :pager_id',
            {'pager_id': pager_id}
        )
        
        trust_list = []
        for trust in trusts:
            trust_list.append({
                "trusted_device_id": trust["trusted_device_id"],
                "trusted_name": self._trusted_name(trust),
                "trusted_fingerprint": trust["trusted_fingerprint"],
                "first_seen": format_datetime(trust["first_seen"], "date_time", get_default_timezone()),
                "last_verified": format_datetime(trust["last_verified"], "date_time", get_default_timezone()),
                "trust_status": trust["trust_status"]
            })
        
        return {
            "pager_id": pager_id,
            "pager_name": self._pager_name(device),
            "trusted_devices": trust_list,
            "count": len(trust_list)
        }
    
    def _revoke_trust(self, pager_id: str, untrusted_device_id: str) -> Dict[str, Any]:
        """
        Revoke trust for a specific device.
        
        Args:
            pager_id: ID of the pager revoking trust
            untrusted_device_id: ID of the device to untrust
            
        Returns:
            Dict with revocation confirmation
        """
        trusts = self.db.select(
            'pager_trust',
            'trusting_device_id = :pager_id AND trusted_device_id = :untrusted_device_id',
            {'pager_id': pager_id, 'untrusted_device_id': untrusted_device_id}
        )
        
        if not trusts or len(trusts) == 0:
            raise ValueError(f"No trust relationship found between {pager_id} and {untrusted_device_id}")
        
        trust = trusts[0]
        
        # Delete the trust relationship entirely to allow fresh start
        self.db.execute(
            'DELETE FROM pager_trust WHERE id = :trust_id',
            {'trust_id': trust['id']}
        )
        self.logger.info(f"Revoked trust: {pager_id} no longer trusts {untrusted_device_id}")
        
        return {
            "message": f"Trust revoked for device {untrusted_device_id}. They can message again to establish new trust.",
            "pager_id": pager_id,
            "untrusted_device_id": untrusted_device_id
        }
    
    def _get_device_location(self) -> Dict[str, Any]:
        """
        Get the current device location.
        
        Returns:
            Dict with location information including coordinates and address
        """
        # In a real implementation, this would use device GPS or IP geolocation
        # For this simulation, we'll generate realistic location data
        import random
        
        # Simulate some common locations
        locations = [
            {
                "lat": 34.6937,
                "lng": 135.5023,
                "address": "1-1 Kitahama, Chuo-ku, Osaka",
                "description": "Near Osaka City Hall"
            },
            {
                "lat": 35.6762,
                "lng": 139.6503,
                "address": "2-8-1 Nishi-Shinjuku, Tokyo",
                "description": "Tokyo Metropolitan Building"
            },
            {
                "lat": 40.7128,
                "lng": -74.0060,
                "address": "City Hall Park, New York, NY",
                "description": "Near City Hall"
            },
            {
                "lat": 37.7749,
                "lng": -122.4194,
                "address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA",
                "description": "San Francisco City Hall"
            }
        ]
        
        # Pick a random location for simulation
        location = random.choice(locations)
        
        # Add timestamp
        location["timestamp"] = utc_now().isoformat()
        location["accuracy_meters"] = random.randint(5, 50)
        
        return location
    
    def _send_location(
        self,
        sender_id: str,
        recipient: str,
        priority: Optional[int] = 1,  # Default to high priority for location pins
        note: Optional[str] = None,
        device_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a location pin message from one pager to another.

        Args:
            sender_id: ID of the sending pager
            recipient: Recipient username or federated address
            priority: Message priority (default 1=high for location pins)
            note: Optional brief note (max 50 chars)
            device_secret: Device secret for authentication

        Returns:
            Dict containing the sent location message
        """
        self.logger.info(f"Sending location pin from {sender_id} to {recipient}")

        # Get current location
        location_data = self._get_device_location()

        # Format location as JSON string for storage
        location_json = json.dumps({
            "lat": location_data["lat"],
            "lng": location_data["lng"],
            "accuracy_meters": location_data["accuracy_meters"]
        })

        # Create location message content
        content = f"📍 Location Pin: {location_data['address']}"
        if location_data.get('description'):
            content += f" ({location_data['description']})"

        # Add optional note if provided
        if note:
            # Enforce brief note limit for location pins
            if len(note) > 50:
                raise ValueError(f"Location pin note too long: {len(note)} characters (max 50)")
            content += f"\nNote: {note}"

        # Add coordinates for technical reference
        content += f"\n[{location_data['lat']:.4f}, {location_data['lng']:.4f}]"

        # Use the existing send_message method with location data
        return self._send_message(
            sender_id=sender_id,
            recipient=recipient,
            content=content,
            priority=priority,
            location=location_json,
            expiry_hours=6,  # Location pins expire faster (6 hours)
            device_secret=device_secret
        )
