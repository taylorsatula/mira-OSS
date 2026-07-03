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
from typing import Dict, Any, Optional

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
        default=True,
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
    
    simple_description = "Store and retrieve contact information (name, email, phone, address, pager address). Search by name or view all contacts. Link reminders to a specific person's UUID."
    tool_schema = {
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
                    "street": {
                        "type": "string",
                        "description": "Street address (optional for add_contact and update_contact)"
                    },
                    "city": {
                        "type": "string",
                        "description": "City (optional for add_contact and update_contact)"
                    },
                    "state": {
                        "type": "string",
                        "description": "State (optional for add_contact and update_contact)"
                    },
                    "zip": {
                        "type": "string",
                        "description": "ZIP code (optional for add_contact and update_contact)"
                    },
                    "identifier": {
                        "type": "string",
                        "description": "Contact UUID or name to search for/update/delete (required for get_contact, delete_contact, update_contact)"
                    },
                    "contacts": {
                        "type": "string",
                        "description": "JSON array of contacts for batch add_contact operations. Each contact should have name, email, phone, street, city, state, zip fields. Use this instead of individual fields for bulk imports."
                    }
                },
                "required": ["operation"],
                "additionalProperties": True
            }
        }

    def __init__(self):
        """Initialize the contacts tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _format_contact(self, contact: Dict[str, Any]) -> Dict[str, Any]:
        """Format a contact using the encrypted field contract."""
        formatted = {
            "uuid": contact["id"],
            "encrypted__name": contact.get("encrypted__name"),
            "encrypted__email": contact.get("encrypted__email"),
            "encrypted__phone": contact.get("encrypted__phone"),
            "encrypted__street": contact.get("encrypted__street"),
            "encrypted__city": contact.get("encrypted__city"),
            "encrypted__state": contact.get("encrypted__state"),
            "encrypted__zip": contact.get("encrypted__zip"),
            "encrypted__pager_address": contact.get("encrypted__pager_address"),
        }
        if "created_at" in contact:
            formatted["created_at"] = contact["created_at"]
        if "updated_at" in contact:
            formatted["updated_at"] = contact["updated_at"]
        return formatted

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
            formatted = [self._format_contact(c) for c in starts[:10]]
            return {"ambiguous": True, "matches": formatted}

        # No starts-with matches, check contains
        contains = [c for c in all_contacts if ident_lower in (c.get('encrypted__name') or '').strip().lower()]

        if len(contains) == 1:
            return {"contact": contains[0], "matched_by": "partial"}
        if len(contains) > 1:
            formatted = [self._format_contact(c) for c in contains[:10]]
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
            # Handle batch contact operations
            if operation == "add_contact" and "contacts" in kwargs:
                return self._batch_add_contacts(kwargs.get("contacts"))

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
                    street: Optional[str] = None, city: Optional[str] = None,
                    state: Optional[str] = None, zip: Optional[str] = None,
                    pager_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new contact.

        Args:
            name: Contact's full name
            email: Contact's email address
            phone: Contact's phone number
            street: Street address
            city: City
            state: State
            zip: ZIP code
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
        for contact in existing:
            if (contact.get('encrypted__name') or '').strip().lower() == name_lower:
                # Found duplicate - return existing contact politely
                self.logger.info(f"Contact '{name}' already exists, returning existing contact")
                return {
                    "success": True,
                    "duplicate": True,
                    "contact": self._format_contact(contact),
                    "message": f"Contact '{name}' already exists (returning existing contact)"
                }
        
        # Create new contact
        contact_id = str(uuid.uuid4())
        timestamp = format_utc_iso(utc_now())
        
        contact_data = {
            'id': contact_id,
            'encrypted__name': name,
            'encrypted__email': email,
            'encrypted__phone': phone,
            'encrypted__street': street,
            'encrypted__city': city,
            'encrypted__state': state,
            'encrypted__zip': zip,
            'encrypted__pager_address': pager_address,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Insert into database
        self.db.insert('contacts', contact_data)
        
        # Return formatted response
        return {
            "success": True,
            "contact": self._format_contact(contact_data),
            "message": f"Added contact {name}"
        }

    def _batch_add_contacts(self, contacts_json: str) -> Dict[str, Any]:
        """
        Add multiple contacts from JSON array.

        Args:
            contacts_json: JSON string containing array of contact objects

        Returns:
            Dict containing batch operation results
        """
        try:
            contacts = json.loads(contacts_json)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in contacts array: {e}")
            raise ValueError(f"Invalid JSON in contacts array: {e}")

        if not isinstance(contacts, list):
            raise ValueError("contacts must be a JSON array")

        results = []
        added = 0
        duplicates = 0
        errors = []

        for contact in contacts:
            try:
                result = self._add_contact(
                    name=contact.get('name'),
                    email=contact.get('email'),
                    phone=contact.get('phone'),
                    street=contact.get('street'),
                    city=contact.get('city'),
                    state=contact.get('state'),
                    zip=contact.get('zip'),
                    pager_address=contact.get('pager_address')
                )
                if result.get('duplicate'):
                    duplicates += 1
                else:
                    added += 1
                results.append(result)
            except Exception as e:
                errors.append(f"{contact.get('name', 'unknown')}: {str(e)}")

        return {
            "success": True,
            "added": added,
            "duplicates": duplicates,
            "errors": errors,
            "total": len(contacts),
            "message": f"Batch add complete: {added} added, {duplicates} duplicates, {len(errors)} errors"
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
            "contact": self._format_contact(contact),
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
            formatted_contacts.append(self._format_contact(contact))
        
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
                "candidate": self._format_contact(c),
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
            "deleted_contact": self._format_contact(contact),
            "message": f"Deleted contact {contact['encrypted__name']}"
        }
    
    def _update_contact(self, identifier: str, name: Optional[str] = None,
                       email: Optional[str] = None, phone: Optional[str] = None,
                       street: Optional[str] = None, city: Optional[str] = None,
                       state: Optional[str] = None, zip: Optional[str] = None,
                       pager_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Update an existing contact.

        Args:
            identifier: Contact UUID or name to update
            name: New name
            email: New email
            phone: New phone
            street: New street address
            city: New city
            state: New state
            zip: New ZIP code
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
                "candidate": self._format_contact(c),
                "message": f"Update candidate matched by partial name. Re-run with UUID {c['id']} to confirm."
            }
        contact = resolved['contact']

        # Require at least one field to update (beyond timestamp)
        if all(
            value is None
            for value in (name, email, phone, street, city, state, zip, pager_address)
        ):
            self.logger.error("No update fields provided in update_contact operation")
            raise ValueError(
                "At least one of name, email, phone, street, city, state, zip, "
                "or pager_address must be provided to update"
            )

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
        if street is not None:
            update_data['encrypted__street'] = street
        if city is not None:
            update_data['encrypted__city'] = city
        if state is not None:
            update_data['encrypted__state'] = state
        if zip is not None:
            update_data['encrypted__zip'] = zip
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
                "contact": self._format_contact(updated_contact),
                "message": f"Updated contact {updated_contact['encrypted__name']}"
            }
        
        raise ValueError("Failed to retrieve updated contact")
    
