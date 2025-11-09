"""
Actions API endpoint - domain-routed state mutations.

Executes state-changing operations through domain-specific handlers that
call tools and services directly, just as MIRA does during continuums.
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator

from utils.user_context import get_current_user_id
from .base import BaseHandler, ValidationError, NotFoundError
from tools.repo import ToolRepository
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)

router = APIRouter()



class DomainType(str, Enum):
    """Supported action domains."""
    REMINDER = "reminder"
    MEMORY = "memory"
    USER = "user"
    CONTACTS = "contacts"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CONTINUUM = "continuum"


class ActionRequest(BaseModel):
    """Action request schema."""
    domain: DomainType = Field(..., description="Domain for the action")
    action: str = Field(..., description="Action to perform")
    data: Dict[str, Any] = Field(default_factory=dict, description="Action-specific data")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        if not v.strip():
            raise ValueError("Action cannot be empty")
        return v.strip()


class BaseDomainHandler(BaseHandler):
    """Base handler for domain-specific actions."""
    
    # Define available actions and their required/optional fields
    ACTIONS: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self):
        super().__init__()  # Initialize BaseHandler (logger, thread pool)
        from utils.user_context import get_current_user_id
        self.user_id = get_current_user_id()
    
    def validate_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action and its data against schema."""
        if action not in self.ACTIONS:
            available_actions = list(self.ACTIONS.keys())
            raise ValidationError(
                f"Unknown action '{action}' for {self.__class__.__name__}. "
                f"Available actions: {', '.join(available_actions)}"
            )
        
        # Get schema for this action
        schema = self.ACTIONS[action]
        required_fields = schema.get('required', [])
        optional_fields = schema.get('optional', [])
        all_fields = required_fields + optional_fields
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(
                f"Missing required fields for action '{action}': {', '.join(missing_fields)}"
            )
        
        # Check for unknown fields
        unknown_fields = [field for field in data.keys() if field not in all_fields]
        if unknown_fields:
            raise ValidationError(
                f"Unknown fields for action '{action}': {', '.join(unknown_fields)}. "
                f"Valid fields: {', '.join(all_fields)}"
            )
        
        # Validate field types
        for field, value in data.items():
            if field in schema.get('types', {}):
                expected_type = schema['types'][field]
                
                # Special handling for UUID type
                if expected_type == 'uuid':
                    from uuid import UUID
                    try:
                        UUID(value)
                    except (ValueError, TypeError):
                        raise ValidationError(f"Field '{field}' must be a valid UUID")
                elif not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Field '{field}' has invalid type, got {type(value).__name__}"
                    )
        
        return data
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action. Override in subclasses."""
        raise NotImplementedError(f"Action '{action}' not implemented")


class ReminderDomainHandler(BaseDomainHandler):
    """Handler for reminder domain actions."""
    
    ACTIONS = {
        "complete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": str}
        },
        "bulk_complete": {
            "required": ["ids"],
            "optional": [],
            "types": {"ids": list}
        },
        "create": {
            "required": ["title", "date"],
            "optional": ["description", "contact_name", "additional_notes"],
            "types": {
                "title": str,
                "date": str,
                "description": str,
                "contact_name": str,
                "additional_notes": str
            }
        },
        "update": {
            "required": ["id"],
            "optional": ["title", "date", "description", "contact_name", "additional_notes"],
            "types": {
                "id": str,
                "title": str,
                "date": str,
                "description": str,
                "contact_name": str,
                "additional_notes": str
            }
        },
        "delete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": str}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reminder actions using ReminderTool."""
        # Get reminder tool through ToolRepository
        tool_repo = ToolRepository()
        
        # Discover and enable tools if not already done
        if "reminder_tool" not in tool_repo.tools:
            tool_repo.discover_tools("tools.implementations")
            if "reminder_tool" in tool_repo.tools:
                tool_repo.enable_tool("reminder_tool")
        
        reminder_tool = tool_repo.get_tool("reminder_tool")
        
        if not reminder_tool:
            raise NotFoundError("tool", "reminder_tool")
        
        try:
            if action == "complete":
                # Mark single reminder as completed
                result = reminder_tool.run(
                    operation="mark_completed",
                    reminder_id=data["id"]
                )
                return {
                    "completed": True,
                    "reminder": result.get("reminder"),
                    "message": result.get("message", "Reminder marked as completed")
                }
            
            elif action == "bulk_complete":
                # Mark multiple reminders as completed
                reminder_ids = data["ids"]
                
                # Validate that ids is a non-empty list of strings
                if not reminder_ids:
                    raise ValidationError("At least one reminder ID is required")
                
                if not all(isinstance(id, str) for id in reminder_ids):
                    raise ValidationError("All reminder IDs must be strings")
                
                completed = []
                failed = []
                
                for reminder_id in reminder_ids:
                    try:
                        result = reminder_tool.run(
                            operation="mark_completed",
                            reminder_id=reminder_id
                        )
                        completed.append({
                            "id": reminder_id,
                            "title": result.get("reminder", {}).get("title", "Unknown")
                        })
                    except Exception as e:
                        failed.append({
                            "id": reminder_id,
                            "error": str(e)
                        })
                
                return {
                    "completed_count": len(completed),
                    "failed_count": len(failed),
                    "completed": completed,
                    "failed": failed,
                    "message": f"Completed {len(completed)} of {len(reminder_ids)} reminders"
                }
            
            elif action == "create":
                # Create new reminder with all provided fields
                result = reminder_tool.run(
                    operation="add_reminder",
                    **data  # Pass all validated data
                )
                return {
                    "created": True,
                    "reminder": result.get("reminder"),
                    "contact_found": result.get("contact_found", False),
                    "contact_info": result.get("contact_info"),
                    "message": result.get("message", "Reminder created")
                }
            
            elif action == "update":
                # Update existing reminder
                reminder_id = data.pop("id")  # Remove id from update fields
                
                # Only pass fields that were actually provided
                update_fields = {k: v for k, v in data.items() if v is not None}
                
                if not update_fields:
                    raise ValidationError("At least one field to update must be provided")
                
                result = reminder_tool.run(
                    operation="update_reminder",
                    reminder_id=reminder_id,
                    **update_fields
                )
                return {
                    "updated": True,
                    "reminder": result.get("reminder"),
                    "updated_fields": result.get("updated_fields", []),
                    "message": result.get("message", "Reminder updated")
                }
            
            elif action == "delete":
                # Delete reminder
                result = reminder_tool.run(
                    operation="delete_reminder",
                    reminder_id=data["id"]
                )
                return {
                    "deleted": True,
                    "id": data["id"],
                    "message": result.get("message", "Reminder deleted")
                }
            
            else:
                raise ValidationError(f"Unknown action: {action}")
                
        except ValueError as e:
            # Tool raises ValueError for business logic errors
            raise ValidationError(str(e))


class MemoryDomainHandler(BaseDomainHandler):
    """Handler for memory domain actions."""
    
    ACTIONS = {
        "create": {
            "required": ["content"],
            "optional": ["importance"],
            "types": {
                "content": str,
                "importance": (int, float)
            }
        },
        "delete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": "uuid"}
        },
        "bulk_delete": {
            "required": ["ids"],
            "optional": [],
            "types": {"ids": list}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory actions using LTMemoryDB."""
        from lt_memory.db_access import LTMemoryDB
        from utils.database_session_manager import get_shared_session_manager

        # Set user context for LTMemoryDB operations
        set_current_user_id(self.user_id)

        # Use shared session manager to prevent connection pool exhaustion
        session_manager = get_shared_session_manager()
        lt_db = LTMemoryDB(session_manager)
        
        if action == "create":
            # Manual memory creation
            content = data["content"]
            importance = data.get("importance", 0.5)
            
            # Validate importance score
            if not 0 <= importance <= 1:
                raise ValidationError("Importance score must be between 0 and 1")
            
            # Generate embedding for the memory
            from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
            from lt_memory.models import ExtractedMemory
            embeddings_provider = get_hybrid_embeddings_provider()  # Use singleton
            # Use realtime embeddings for memory storage (384-dim)
            embedding = embeddings_provider.encode_realtime(content)

            # Convert ndarray to list for database storage (serialization boundary)
            embedding_list = embedding.tolist()

            # Create ExtractedMemory object (embedding passed separately to store_memories)
            memory = ExtractedMemory(
                text=content,
                importance_score=importance
            )

            memory_ids = lt_db.store_memories([memory], embeddings=[embedding_list])
            
            if not memory_ids:
                raise ValidationError("Failed to create memory")
            
            # Get the created memory to return
            created_memory = lt_db.get_memory(memory_ids[0])

            return jsonable_encoder({
                "created": True,
                "memory": {
                    "id": created_memory.id,
                    "text": created_memory.text,
                    "importance_score": created_memory.importance_score,
                    "created_at": created_memory.created_at
                },
                "message": "Memory created successfully"
            })
        
        elif action == "delete":
            # Delete single memory
            memory_id = data["id"]
            
            # Verify memory exists
            memory = lt_db.get_memory(memory_id)
            if not memory:
                raise NotFoundError("memory", memory_id)

            # Delete the memory
            deleted_count = lt_db.delete_memories([memory_id])
            
            return {
                "deleted": deleted_count > 0,
                "id": memory_id,
                "message": f"Memory deleted successfully" if deleted_count > 0 else "Failed to delete memory"
            }
        
        elif action == "bulk_delete":
            # Delete multiple memories
            memory_ids = data["ids"]
            
            # Validate that ids is a non-empty list of strings
            if not memory_ids:
                raise ValidationError("At least one memory ID is required")
            
            if not all(isinstance(id, str) for id in memory_ids):
                raise ValidationError("All memory IDs must be strings")
            
            # Delete memories
            deleted_count = lt_db.delete_memories(memory_ids)
            
            return {
                "deleted_count": deleted_count,
                "requested_count": len(memory_ids),
                "ids": memory_ids,
                "message": f"Deleted {deleted_count} of {len(memory_ids)} memories"
            }
        
        else:
            raise ValidationError(f"Unknown action: {action}")


class ContactsDomainHandler(BaseDomainHandler):
    """Handler for contacts domain actions."""
    
    ACTIONS = {
        "create": {
            "required": ["name"],
            "optional": ["email", "phone"],
            "types": {
                "name": str,
                "email": str,
                "phone": str
            }
        },
        "get": {
            "required": ["identifier"],
            "optional": [],
            "types": {"identifier": str}
        },
        "list": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "update": {
            "required": ["identifier"],
            "optional": ["name", "email", "phone"],
            "types": {
                "identifier": str,
                "name": str,
                "email": str,
                "phone": str
            }
        },
        "delete": {
            "required": ["identifier"],
            "optional": [],
            "types": {"identifier": str}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute contacts actions using ContactsTool."""
        # Get contacts tool through ToolRepository
        tool_repo = ToolRepository()
        
        # Discover and enable tools if not already done
        if "contacts_tool" not in tool_repo.tools:
            tool_repo.discover_tools("tools.implementations")
            if "contacts_tool" in tool_repo.tools:
                tool_repo.enable_tool("contacts_tool")
        
        contacts_tool = tool_repo.get_tool("contacts_tool")
        
        if not contacts_tool:
            raise NotFoundError("tool", "contacts_tool")
        
        try:
            if action == "create":
                # Create new contact
                result = contacts_tool.run(
                    operation="add_contact",
                    **data  # Pass all validated data
                )
                return {
                    "created": True,
                    "contact": result.get("contact"),
                    "message": result.get("message", "Contact created")
                }
            
            elif action == "get":
                # Get contact by UUID or name
                result = contacts_tool.run(
                    operation="get_contact",
                    identifier=data["identifier"]
                )
                return {
                    "found": True,
                    "contact": result.get("contact"),
                    "message": result.get("message", "Contact found")
                }
            
            elif action == "list":
                # List all contacts
                result = contacts_tool.run(
                    operation="list_contacts"
                )
                return {
                    "contacts": result.get("contacts", []),
                    "count": len(result.get("contacts", [])),
                    "message": result.get("message", "Contacts retrieved")
                }
            
            elif action == "update":
                # Update existing contact
                identifier = data.pop("identifier")  # Remove identifier from update fields
                
                # Only pass fields that were actually provided
                update_fields = {k: v for k, v in data.items() if v is not None}
                
                if not update_fields:
                    raise ValidationError("At least one field to update must be provided")
                
                result = contacts_tool.run(
                    operation="update_contact",
                    identifier=identifier,
                    **update_fields
                )
                return {
                    "updated": True,
                    "contact": result.get("contact"),
                    "message": result.get("message", "Contact updated")
                }
            
            elif action == "delete":
                # Delete contact
                result = contacts_tool.run(
                    operation="delete_contact",
                    identifier=data["identifier"]
                )
                return {
                    "deleted": True,
                    "deleted_contact": result.get("deleted_contact"),
                    "message": result.get("message", "Contact deleted")
                }
            
            else:
                raise ValidationError(f"Unknown action: {action}")
                
        except ValueError as e:
            # Tool raises ValueError for business logic errors
            raise ValidationError(str(e))


class UserDomainHandler(BaseDomainHandler):
    """Handler for user preference/settings actions."""
    
    ACTIONS = {
        "update_preferences": {
            "required": [],
            "optional": ["theme", "timezone", "language", "calendar_url"],
            "types": {
                "theme": str,
                "timezone": str,
                "language": str,
                "calendar_url": str
            }
        },
        "store_email_config": {
            "required": ["email_address", "password", "imap_server", "smtp_server"],
            "optional": ["imap_port", "smtp_port", "use_ssl"],
            "types": {
                "email_address": str,
                "password": str,
                "imap_server": str,
                "smtp_server": str,
                "imap_port": int,
                "smtp_port": int,
                "use_ssl": bool
            }
        },
        "get_email_config": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "store_calendar_config": {
            "required": ["calendar_url"],
            "optional": [],
            "types": {
                "calendar_url": str
            }
        },
        "get_calendar_config": {
            "required": [],
            "optional": [],
            "types": {}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute user preference actions."""
        if action == "update_preferences":
            # Validate theme if provided
            if "theme" in data:
                valid_themes = ["light", "dark", "auto"]
                if data["theme"] not in valid_themes:
                    raise ValidationError(
                        f"Invalid theme '{data['theme']}'. "
                        f"Valid themes: {', '.join(valid_themes)}"
                    )
            
            # Validate timezone if provided
            if "timezone" in data:
                from utils.timezone_utils import validate_timezone
                try:
                    validate_timezone(data["timezone"])
                except Exception:
                    raise ValidationError(f"Invalid timezone: {data['timezone']}")
            
            # Validate language if provided
            if "language" in data:
                valid_languages = ["en", "es", "fr", "de", "ja", "zh"]
                if data["language"] not in valid_languages:
                    raise ValidationError(
                        f"Invalid language '{data['language']}'. "
                        f"Valid languages: {', '.join(valid_languages)}"
                    )
            
            # Validate calendar URL if provided
            if "calendar_url" in data:
                calendar_url = data["calendar_url"]
                if not calendar_url or not isinstance(calendar_url, str):
                    raise ValidationError("Calendar URL must be a non-empty string")
                # Basic URL validation
                if not (calendar_url.startswith("http://") or calendar_url.startswith("https://")):
                    raise ValidationError("Calendar URL must start with http:// or https://")
            
            # For now, return dummy success response
            # In the future, this would update a user_preferences table
            updated_prefs = {
                "theme": data.get("theme", "light"),
                "timezone": data.get("timezone", "UTC"),
                "language": data.get("language", "en"),
                "calendar_url": data.get("calendar_url")
            }
            
            return {
                "updated": True,
                "preferences": updated_prefs,
                "message": "Preferences updated successfully (placeholder implementation)"
            }
        
        elif action == "store_email_config":
            from utils.user_credentials import store_email_config_for_current_user
            
            # Build config dict with defaults
            config = {
                "email_address": data["email_address"],
                "password": data["password"],
                "imap_server": data["imap_server"],
                "smtp_server": data["smtp_server"],
                "imap_port": data.get("imap_port", 993),
                "smtp_port": data.get("smtp_port", 465),
                "use_ssl": data.get("use_ssl", True)
            }
            
            try:
                from tools.implementations.email_tool import EmailTool
                
                # Validate required fields first
                if not config["imap_server"] or not config["email_address"] or not config["password"]:
                    raise ValidationError("Missing required email configuration. Please provide server, email address, and password.")
                
                # Create temporary tool instance for validation
                email_tool = EmailTool()
                email_tool.imap_server = config["imap_server"]
                email_tool.imap_port = config["imap_port"]
                email_tool.smtp_server = config["smtp_server"] 
                email_tool.smtp_port = config["smtp_port"]
                email_tool.email_address = config["email_address"]
                email_tool.use_ssl = config["use_ssl"]
                email_tool._password = config["password"]
                email_tool._config_loaded = True
                
                # Test IMAP connection
                if not email_tool._connect():
                    raise ValidationError("Cannot connect to IMAP server. Please check your server settings and credentials.")
                
                # Clean up test connection
                email_tool._disconnect()
                
                # If validation passes, store the configuration
                store_email_config_for_current_user(config)
                return {
                    "success": True,
                    "message": "Email configuration validated and stored successfully"
                }
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Email validation failed: {str(e)}")
        
        elif action == "get_email_config":
            from utils.user_credentials import get_email_config_for_current_user

            config = get_email_config_for_current_user()
            if config:
                # Remove password from response for security
                safe_config = {k: v for k, v in config.items() if k != "password"}
                return {
                    "success": True,
                    "config": safe_config
                }
            else:
                return {
                    "success": True,
                    "config": None,
                    "message": "No email configuration found"
                }
        
        elif action == "store_calendar_config":
            from utils.user_credentials import UserCredentialService

            calendar_url = data["calendar_url"]

            # Store calendar URL in user credentials
            credential_service = UserCredentialService()
            credential_service.store_credential(
                user_id=self.user_id,
                credential_type="calendar_url",
                service_name="calendar",
                credential_value=calendar_url
            )

            return {
                "success": True,
                "message": "Calendar URL stored successfully"
            }
        
        elif action == "get_calendar_config":
            from utils.user_credentials import UserCredentialService

            credential_service = UserCredentialService()
            calendar_url = credential_service.get_credential(
                user_id=self.user_id,
                credential_type="calendar_url",
                service_name="calendar"
            )

            return {
                "success": True,
                "calendar_url": calendar_url if calendar_url else None,
                "message": "Calendar configuration retrieved" if calendar_url else "No calendar URL configured"
            }
        
        else:
            raise ValidationError(f"Unknown action: {action}")


class DomainKnowledgeDomainHandler(BaseDomainHandler):
    """Handler for domain knowledge block actions."""

    ACTIONS = {
        "create": {
            "required": ["domain_label", "domain_name", "block_description"],
            "optional": [],
            "types": {
                "domain_label": str,
                "domain_name": str,
                "block_description": str
            }
        },
        "enable": {
            "required": ["domain_label"],
            "optional": [],
            "types": {"domain_label": str}
        },
        "disable": {
            "required": ["domain_label"],
            "optional": [],
            "types": {"domain_label": str}
        },
        "delete": {
            "required": ["domain_label"],
            "optional": [],
            "types": {"domain_label": str}
        },
        "update": {
            "required": ["domain_label", "content"],
            "optional": [],
            "types": {
                "domain_label": str,
                "content": str
            }
        }
    }

    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domain knowledge actions using DomainKnowledgeService."""
        from cns.services.domain_knowledge_service import get_domain_knowledge_service

        service = get_domain_knowledge_service()
        if not service:
            raise ValidationError("Domain knowledge feature is not available (check Letta API key configuration)")

        try:
            if action == "create":
                # Create new domain block
                result = service.create_domain_block(
                    domain_label=data["domain_label"],
                    domain_name=data["domain_name"],
                    block_description=data["block_description"]
                )
                return {
                    "created": True,
                    "domain": result,
                    "message": f"Domain block '{data['domain_name']}' created successfully"
                }

            elif action == "enable":
                # Enable domain block (inject into system prompt)
                service.enable_domain(data["domain_label"])
                return {
                    "enabled": True,
                    "domain_label": data["domain_label"],
                    "message": "Domain block enabled"
                }

            elif action == "disable":
                # Disable domain block (remove from system prompt)
                service.disable_domain(data["domain_label"])
                return {
                    "disabled": True,
                    "domain_label": data["domain_label"],
                    "message": "Domain block disabled"
                }

            elif action == "delete":
                # Delete domain block and sleeptime agent
                service.delete_domain(data["domain_label"])
                return {
                    "deleted": True,
                    "domain_label": data["domain_label"],
                    "message": "Domain block deleted"
                }

            elif action == "update":
                # Manually update domain block content
                service.update_block_content(
                    domain_label=data["domain_label"],
                    new_content=data["content"]
                )
                return {
                    "updated": True,
                    "domain_label": data["domain_label"],
                    "message": "Domain block content updated"
                }

            else:
                raise ValidationError(f"Unknown action: {action}")

        except ValueError as e:
            # Service raises ValueError for business logic errors
            raise ValidationError(str(e))


class ContinuumDomainHandler(BaseDomainHandler):
    """Handler for continuum-level configuration actions."""

    ACTIONS = {
        "set_thinking_budget_preference": {
            "required": [],
            "optional": ["budget"],
            "types": {
                "budget": (int, type(None))
            }
        },
        "get_thinking_budget_preference": {
            "required": [],
            "optional": [],
            "types": {}
        }
    }

    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute continuum configuration actions."""
        if action == "set_thinking_budget_preference":
            from cns.infrastructure.continuum_pool import get_continuum_pool

            budget = data.get("budget")

            # Validate budget if provided
            if budget is not None:
                if not isinstance(budget, int):
                    raise ValidationError("Budget must be an integer or null")
                if budget < 0:
                    raise ValidationError("Budget must be None, 0, or a positive integer")
                # Validate against allowed values
                valid_budgets = [0, 1024, 4096, 32000]
                if budget not in valid_budgets:
                    raise ValidationError(
                        f"Budget must be one of {valid_budgets} or null. Got: {budget}"
                    )

            pool = get_continuum_pool()
            continuum = pool.get_or_create()

            # Set the preference in Valkey
            pool.set_thinking_budget_preference(budget)

            return {
                "success": True,
                "continuum_id": str(continuum.id),
                "budget": budget,
                "message": f"Thinking budget preference set to {budget}"
            }

        elif action == "get_thinking_budget_preference":
            from cns.infrastructure.continuum_pool import get_continuum_pool

            pool = get_continuum_pool()
            continuum = pool.get_or_create()

            # Get the preference from Valkey
            budget = pool.get_thinking_budget_preference()

            return {
                "success": True,
                "continuum_id": str(continuum.id),
                "budget": budget
            }

        else:
            raise ValidationError(f"Unknown action: {action}")


class ActionsEndpoint(BaseHandler):
    """Main actions endpoint handler with domain-based routing."""

    def __init__(self):
        super().__init__()
        self.domain_handlers = {
            DomainType.REMINDER: ReminderDomainHandler,
            DomainType.MEMORY: MemoryDomainHandler,
            DomainType.USER: UserDomainHandler,
            DomainType.CONTACTS: ContactsDomainHandler,
            DomainType.DOMAIN_KNOWLEDGE: DomainKnowledgeDomainHandler,
            DomainType.CONTINUUM: ContinuumDomainHandler
        }
    
    def process_request(self, **params) -> Dict[str, Any]:
        """Route request to appropriate domain handler."""
        request_data = params['request_data']
        user_id = get_current_user_id()
        
        domain = request_data.domain
        action = request_data.action
        data = request_data.data
        
        # Get domain handler
        handler_class = self.domain_handlers.get(domain)
        if not handler_class:
            raise ValidationError(f"Unknown domain: {domain}")

        # Create handler instance
        handler = handler_class()
        
        # Validate action and data
        validated_data = handler.validate_action(action, data)
        
        # Execute action
        result = handler.execute_action(action, validated_data)
        
        # Add metadata
        result["meta"] = {
            "domain": domain.value,
            "action": action,
            "timestamp": format_utc_iso(utc_now())
        }
        
        return result


def get_actions_handler() -> ActionsEndpoint:
    """Get actions endpoint handler instance."""
    return ActionsEndpoint()


@router.post("/actions")
async def actions_endpoint(request_data: ActionRequest):
    """Execute state-changing operations through domain-routed actions."""
    try:
        handler = get_actions_handler()
        response = handler.handle_request(request_data=request_data)
        return response.to_dict()
        
    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": e.message
                }
            }
        )
    except NotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": e.message
                }
            }
        )
    except NotImplementedError as e:
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "error": {
                    "code": "NOT_IMPLEMENTED",
                    "message": str(e)
                }
            }
        )
    except Exception as e:
        logger.error(f"Actions endpoint error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Action execution failed"
                }
            }
        )
