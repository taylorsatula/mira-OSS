"""
Simple contacts management tool.

This tool provides basic contact management functionality including
adding, retrieving, listing, and deleting contacts. Each contact has
a UUID for unique identification and can be linked to other tools.
"""

# Standard library imports
import json
import logging
import uuid
from typing import Dict, List, Any, Optional

# Third-party imports
from pydantic import BaseModel, Field

# Import timezone utilities for UTC-everywhere approach
from utils.timezone_utils import utc_now, format_utc_iso

# Local imports
from tools.repo import Tool
from tools.registry import registry


# -------------------- CONFIGURATION --------------------

class ContactsToolConfig(BaseModel):
    """
    Configuration for the contacts_tool.
    
    Defines the parameters that control the contacts tool's behavior.
    """
    # Standard configuration parameter - all tools should include this
    enabled: bool = Field(
        default=False,
        description="Whether this tool is enabled by default"
    )

# Register with registry
registry.register("contacts_tool", ContactsToolConfig)


# -------------------- MAIN TOOL CLASS --------------------

class ContactsTool(Tool):
    """
    Simple contacts management tool.
    
    This tool provides basic contact management functionality including
    adding, retrieving, listing, and deleting contacts.
    """

    name = "contacts_tool"
    
    simple_description = "Store and retrieve contact information (name, email, phone). Search by name or view all contacts. Link reminders to a specific person's UUID."
    anthropic_schema = {
        "name": "contacts_tool",
        "description": "Manages personal contacts with basic CRUD operations. Each contact has a unique UUID for linking to other tools.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add_contact", "get_contact", "list_contacts", "delete_contact", "update_contact"],
                        "description": "The operation to perform"
                    },
                    "name": {
                        "type": "string",
                        "description": "Contact's full name (required for add_contact, optional for update_contact)"
                    },
                    "email": {
                        "type": "string",
                        "description": "Contact's email address (optional)"
                    },
                    "phone": {
                        "type": "string",
                        "description": "Contact's phone number (optional)"
                    },
                    "pager_address": {
                        "type": "string",
                        "description": "Contact's pager address - local username or user@domain (optional)"
                    },
                    "identifier": {
                        "type": "string",
                        "description": "Contact UUID or name to search for/update/delete (required for get_contact, delete_contact, update_contact)"
                    }
                },
                "required": ["operation"],
                "additionalProperties": False
            }
        }

    def __init__(self):
        """Initialize the contacts tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _find_by_identifier(self, identifier: str) -> Dict[str, Any]:
        """Helper to resolve an identifier to a specific contact or candidates.

        Returns a dict with one of:
        - {"contact": {...}, "matched_by": "id|name|partial"}
        - {"ambiguous": True, "matches": [..]}
        - {}
        """
        if not identifier:
            return {}
        ident = identifier.strip()

        # 1) Try UUID exact match using database WHERE clause
        contacts = self.db.select('contacts', 'id = :identifier', {'identifier': ident})
        if contacts:
            return {"contact": contacts[0], "matched_by": "id"}

        # 2) Try case-insensitive exact name (name decrypted at select time)
        all_contacts = self.db.select('contacts')
        ident_lower = ident.lower()
        exact = [c for c in all_contacts if (c.get('encrypted__name') or '').strip().lower() == ident_lower]
        if exact:
            return {"contact": exact[0], "matched_by": "name"}

        # 3) Partial matches by name (starts-with prioritized)
        starts = [c for c in all_contacts if (c.get('encrypted__name') or '').strip().lower().startswith(ident_lower)]

        # Prioritize starts-with - only check contains if no starts-with matches
        if starts:
            if len(starts) == 1:
                return {"contact": starts[0], "matched_by": "partial"}
            # Multiple starts-with matches
            formatted = [{
                'uuid': c['id'],
                'encrypted__name': c.get('encrypted__name'),
                'encrypted__email': c.get('encrypted__email'),
                'encrypted__phone': c.get('encrypted__phone'),
                'encrypted__pager_address': c.get('encrypted__pager_address')
            } for c in starts[:10]]
            return {"ambiguous": True, "matches": formatted}

        # No starts-with matches, check contains
        contains = [c for c in all_contacts if ident_lower in (c.get('encrypted__name') or '').strip().lower()]

        if len(contains) == 1:
            return {"contact": contains[0], "matched_by": "partial"}
        if len(contains) > 1:
            formatted = [{
                'uuid': c['id'],
                'encrypted__name': c.get('encrypted__name'),
                'encrypted__email': c.get('encrypted__email'),
                'encrypted__phone': c.get('encrypted__phone'),
                'encrypted__pager_address': c.get('encrypted__pager_address')
            } for c in contains[:10]]
            return {"ambiguous": True, "matches": formatted}

        return {}
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a contacts tool operation.
        
        Args:
            operation: The operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            Dict containing the operation results
            
        Raises:
            ValueError: If operation fails or parameters are invalid
        """
        try:
            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in kwargs for contacts_tool: {e}")
                    raise ValueError(f"Invalid JSON in kwargs: {e}")
            
            # Route to the appropriate operation
            if operation == "add_contact":
                return self._add_contact(**kwargs)
            elif operation == "get_contact":
                return self._get_contact(**kwargs)
            elif operation == "list_contacts":
                return self._list_contacts(**kwargs)
            elif operation == "delete_contact":
                return self._delete_contact(**kwargs)
            elif operation == "update_contact":
                return self._update_contact(**kwargs)
            else:
                self.logger.error(f"Unknown operation '{operation}' in contacts_tool")
                raise ValueError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "add_contact, get_contact, list_contacts, delete_contact, update_contact"
                )
        except Exception as e:
            self.logger.error(f"Error executing contacts_tool operation '{operation}': {e}")
            raise
    
    def _add_contact(self, name: str, email: Optional[str] = None, phone: Optional[str] = None,
                    pager_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new contact.

        Args:
            name: Contact's full name
            email: Contact's email address
            phone: Contact's phone number
            pager_address: Contact's pager address (username or user@domain)

        Returns:
            Dict containing the operation result
        """
        if not name or not isinstance(name, str):
            self.logger.error(f"Invalid contact name provided: {repr(name)}")
            raise ValueError("Contact name is required and must be a non-empty string")

        # Check duplicates by loading and comparing decrypted names (name is encrypted at rest)
        existing = self.db.select('contacts')
        name_lower = name.strip().lower()
        if any((c.get('encrypted__name') or '').strip().lower() == name_lower for c in existing):
            self.logger.error(f"Duplicate contact name '{name}' in contacts_tool")
            raise ValueError(f"Contact with name '{name}' already exists")
        
        # Create new contact
        contact_id = str(uuid.uuid4())
        timestamp = format_utc_iso(utc_now())
        
        contact_data = {
            'id': contact_id,
            'encrypted__name': name,
            'encrypted__email': email,
            'encrypted__phone': phone,
            'encrypted__pager_address': pager_address,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Insert into database
        self.db.insert('contacts', contact_data)
        
        # Return formatted response
        return {
            "success": True,
            "contact": {
                "uuid": contact_id,
                "encrypted__name": name,
                "encrypted__email": email,
                "encrypted__phone": phone,
                "encrypted__pager_address": pager_address,
                "created_at": timestamp,
                "updated_at": timestamp
            },
            "message": f"Added contact {name}"
        }
    
    def _get_contact(self, identifier: str) -> Dict[str, Any]:
        """
        Get contact details by UUID or name.

        Args:
            identifier: Contact UUID or name to search for

        Returns:
            Dict containing the contact information
        """
        if not identifier or not identifier.strip():
            self.logger.error("Missing contact identifier in get_contact operation")
            raise ValueError("Contact identifier is required")
        
        resolved = self._find_by_identifier(identifier)
        if not resolved:
            self.logger.error(f"Contact '{identifier}' not found in get_contact operation")
            return {
                "success": False,
                "message": f"No contact matches '{identifier}'. Try a fuller name or a UUID.",
                "ambiguous": False
            }

        if resolved.get('ambiguous'):
            return {
                "success": False,
                "ambiguous": True,
                "matches": resolved['matches'],
                "message": f"Multiple contacts match '{identifier}'. Please specify one by UUID or full name."
            }

        contact = resolved['contact']
        return {
            "success": True,
            "contact": {
                "uuid": contact['id'],
                "encrypted__name": contact['encrypted__name'],
                "encrypted__email": contact['encrypted__email'],
                "encrypted__phone": contact['encrypted__phone'],
                "encrypted__pager_address": contact.get('encrypted__pager_address'),
                "created_at": contact['created_at'],
                "updated_at": contact['updated_at']
            },
            "matched_by": resolved.get('matched_by'),
            "message": f"Found contact {contact['encrypted__name']} (matched by {resolved.get('matched_by')})"
        }
    
    def _list_contacts(self) -> Dict[str, Any]:
        """
        List all contacts.
        
        Returns:
            Dict containing all contacts
        """
        contacts = self.db.select('contacts')
        
        # Format contacts for response
        formatted_contacts = []
        for contact in contacts:
            formatted_contacts.append({
                "uuid": contact['id'],
                "encrypted__name": contact['encrypted__name'],
                "encrypted__email": contact['encrypted__email'],
                "encrypted__phone": contact['encrypted__phone'],
                "encrypted__pager_address": contact.get('encrypted__pager_address'),
                "created_at": contact['created_at'],
                "updated_at": contact['updated_at']
            })
        
        return {
            "success": True,
            "contacts": formatted_contacts,
            "message": f"Found {len(formatted_contacts)} contact(s)"
        }
    
    def _delete_contact(self, identifier: str) -> Dict[str, Any]:
        """
        Delete a contact by UUID or name.
        
        Args:
            identifier: Contact UUID or name to delete
            
        Returns:
            Dict containing the operation result
        """
        if not identifier:
            self.logger.error("Missing contact identifier in delete_contact operation")
            raise ValueError("Contact identifier is required")
        
        resolved = self._find_by_identifier(identifier)

        if not resolved:
            self.logger.error(f"Contact '{identifier}' not found in delete_contact operation")
            raise ValueError(f"Contact '{identifier}' not found")

        if resolved.get('ambiguous'):
            return {
                "success": False,
                "ambiguous": True,
                "matches": resolved['matches'],
                "message": f"Multiple contacts match '{identifier}'. Please re-run with a UUID to confirm deletion."
            }

        if resolved.get('matched_by') == 'partial':
            c = resolved['contact']
            return {
                "success": False,
                "needs_confirmation": True,
                "candidate": {
                    'uuid': c['id'], 'encrypted__name': c.get('encrypted__name'), 'encrypted__email': c.get('encrypted__email'), 'encrypted__phone': c.get('encrypted__phone'), 'encrypted__pager_address': c.get('encrypted__pager_address')
                },
                "message": f"Delete candidate matched by partial name. Re-run with UUID {c['id']} to confirm."
            }
        contact = resolved['contact']

        # Delete from database
        rows_deleted = self.db.delete(
            'contacts',
            'id = :id',
            {'id': contact['id']}
        )

        return {
            "success": True,
            "deleted_contact": {
                "uuid": contact['id'],
                "encrypted__name": contact['encrypted__name'],
                "encrypted__email": contact['encrypted__email'],
                "encrypted__phone": contact['encrypted__phone'],
                "encrypted__pager_address": contact.get('encrypted__pager_address'),
                "created_at": contact['created_at']
            },
            "message": f"Deleted contact {contact['encrypted__name']}"
        }
    
    def _update_contact(self, identifier: str, name: Optional[str] = None,
                       email: Optional[str] = None, phone: Optional[str] = None,
                       pager_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Update an existing contact.

        Args:
            identifier: Contact UUID or name to update
            name: New name
            email: New email
            phone: New phone
            pager_address: New pager address

        Returns:
            Dict containing the operation result
        """
        if not identifier:
            self.logger.error("Missing contact identifier in update_contact operation")
            raise ValueError("Contact identifier is required")
        
        resolved = self._find_by_identifier(identifier)
        if not resolved:
            self.logger.error(f"Contact '{identifier}' not found in update_contact operation")
            raise ValueError(f"Contact '{identifier}' not found")

        if resolved.get('ambiguous'):
            return {
                "success": False,
                "ambiguous": True,
                "matches": resolved['matches'],
                "message": f"Multiple contacts match '{identifier}'. Please re-run with a UUID to specify which to update."
            }

        if resolved.get('matched_by') == 'partial':
            c = resolved['contact']
            return {
                "success": False,
                "needs_confirmation": True,
                "candidate": {
                    'uuid': c['id'], 'encrypted__name': c.get('encrypted__name'), 'encrypted__email': c.get('encrypted__email'), 'encrypted__phone': c.get('encrypted__phone'), 'encrypted__pager_address': c.get('encrypted__pager_address')
                },
                "message": f"Update candidate matched by partial name. Re-run with UUID {c['id']} to confirm."
            }
        contact = resolved['contact']

        # Require at least one field to update (beyond timestamp)
        if name is None and email is None and phone is None and pager_address is None:
            self.logger.error("No update fields provided in update_contact operation")
            raise ValueError("At least one of name, email, phone, or pager_address must be provided to update")

        # If renaming, prevent duplicates (case-insensitive) to preserve unique names
        if name is not None:
            all_contacts = self.db.select('contacts')
            name_lower = name.strip().lower()
            dupes = [c for c in all_contacts if c['id'] != contact['id'] and (c.get('encrypted__name') or '').strip().lower() == name_lower]
            if dupes:
                self.logger.error(f"Duplicate contact name '{name}' in update_contact operation")
                raise ValueError(f"Contact with name '{name}' already exists")

        # Build update data
        update_data = {'updated_at': format_utc_iso(utc_now())}
        if name is not None:
            update_data['encrypted__name'] = name
        if email is not None:
            update_data['encrypted__email'] = email
        if phone is not None:
            update_data['encrypted__phone'] = phone
        if pager_address is not None:
            update_data['encrypted__pager_address'] = pager_address
        
        # Update in database
        rows_updated = self.db.update(
            'contacts',
            update_data,
            'id = :id',
            {'id': contact['id']}
        )
        
        # Get updated contact
        updated_contacts = self.db.select(
            'contacts',
            'id = :id',
            {'id': contact['id']}
        )
        
        if updated_contacts:
            updated_contact = updated_contacts[0]
            return {
                "success": True,
                "contact": {
                    "uuid": updated_contact['id'],
                    "encrypted__name": updated_contact['encrypted__name'],
                    "encrypted__email": updated_contact['encrypted__email'],
                    "encrypted__phone": updated_contact['encrypted__phone'],
                    "encrypted__pager_address": updated_contact.get('encrypted__pager_address'),
                    "created_at": updated_contact['created_at'],
                    "updated_at": updated_contact['updated_at']
                },
                "message": f"Updated contact {updated_contact['encrypted__name']}"
            }
        
        raise ValueError("Failed to retrieve updated contact")
    
