"""
Data API endpoint - unified data access with type-based routing.

Uses proper repository methods for safe data access with user isolation.
"""
import logging
from typing import Any
from enum import Enum

from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from utils.user_context import get_current_user_id
from auth.api import get_current_user
from auth.types import SessionData, APITokenContext
from .base import BaseHandler, ValidationError, NotFoundError
from utils.timezone_utils import utc_now, format_utc_iso, parse_utc_time_string

logger = logging.getLogger(__name__)

router = APIRouter()


class DataType(str, Enum):
    """Supported data types."""
    HISTORY = "history"
    MEMORIES = "memories"
    DASHBOARD = "dashboard"
    USER = "user"
    DOMAINDOCS = "domaindocs"
    WORKING_MEMORY = "working_memory"
    LORA = "lora"


class DataEndpoint(BaseHandler):
    """Main data endpoint handler with type-based routing."""
    
    def process_request(self, **params) -> dict[str, Any]:
        """Route request to appropriate handler."""
        # Set user context from params (provided by endpoint)
        from utils.user_context import set_current_user_id
        user_id = params.get('user_id')
        if user_id:
            set_current_user_id(user_id)

        data_type = params['data_type']
        request_params = params.get('request_params', {})

        if data_type == DataType.HISTORY:
            return self._get_history(**request_params)
        elif data_type == DataType.MEMORIES:
            return self._get_memories(**request_params)
        elif data_type == DataType.DASHBOARD:
            return self._get_dashboard(**request_params)
        elif data_type == DataType.USER:
            return self._get_user(**request_params)
        elif data_type == DataType.DOMAINDOCS:
            return self._get_domains(**request_params)
        elif data_type == DataType.WORKING_MEMORY:
            return self._get_working_memory(**request_params)
        elif data_type == DataType.LORA:
            return self._get_lora(**request_params)
        else:
            raise ValidationError(f"Invalid data type: {data_type}")
    
    def _get_history(self, **params) -> dict[str, Any]:
        """Get continuum history using ContinuumRepository."""
        user_id = get_current_user_id()

        limit = params.get('limit', 50)
        offset = params.get('offset', 0)
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        search_query = params.get('search')
        message_type = params.get('message_type', 'regular')
        from cns.infrastructure.continuum_repository import get_continuum_repository

        repo = get_continuum_repository()  # Use singleton

        # If search query provided, use search instead of regular history
        if search_query:
            history_data = repo.search_continuums(
                user_id=user_id,
                search_query=search_query,
                offset=offset,
                limit=limit,
                message_type=message_type
            )
        else:
            # Parse dates if provided
            start_dt = None
            end_dt = None
            if start_date:
                start_dt = parse_utc_time_string(start_date.replace('Z', '+00:00'))
            if end_date:
                end_dt = parse_utc_time_string(end_date.replace('Z', '+00:00'))

            history_data = repo.get_history(
                user_id=user_id,
                offset=offset,
                limit=limit,
                start_date=start_dt,
                end_date=end_dt,
                message_type=message_type
            )
        
        return {
            "messages": history_data.get("messages", []),
            "meta": {
                "total_returned": len(history_data.get("messages", [])),
                "has_more": history_data.get("has_more", False),
                "next_offset": history_data.get("next_offset"),
                "limit": limit,
                "offset": offset,
                "search_query": history_data.get("search_query")
            }
        }
    
    def _get_memories(self, **params) -> dict[str, Any]:
        """Get memories using LTMemoryDB."""
        from lt_memory.db_access import LTMemoryDB
        from utils.database_session_manager import get_shared_session_manager

        user_id = get_current_user_id()
        limit = params.get('limit', 50)
        offset = params.get('offset', 0)
        subtype = params.get('subtype')  # 'active', 'expired'
        search_query = params.get('search')

        session_manager = get_shared_session_manager()
        lt_db = LTMemoryDB(session_manager)

        memory_data = lt_db.get_memories_paginated(
            limit=limit,
            offset=offset,
            user_id=user_id
        )

        return jsonable_encoder({
            "memories": memory_data.get("memories", []),
            "meta": {
                "total_returned": len(memory_data.get("memories", [])),
                "has_more": memory_data.get("has_more", False),
                "next_offset": memory_data.get("next_offset"),
                "limit": limit,
                "offset": offset,
                "subtype": subtype,
                "search_query": memory_data.get("search_query")
            }
        })
    
    def _get_dashboard(self, **params) -> dict[str, Any]:
        """Get dashboard data - system health and context usage metrics."""
        from clients.postgres_client import PostgresClient
        from config import config

        user_id = get_current_user_id()

        # DB health check — let infrastructure failures propagate
        from utils.instance import database_name
        db = PostgresClient(database_name(), user_id=user_id)
        db.execute_single("SELECT 1")

        # Context usage metrics are managed by Anthropic's prompt caching
        # Token tracking is stateless - cache markers are applied unconditionally
        # and Anthropic handles threshold logic server-side
        return {
            "system_health": "healthy",
            "context_usage": {
                "max_tokens": config.api.context_window_tokens,
                "note": "Token tracking delegated to Anthropic prompt caching"
            },
            "meta": {
                "timestamp": format_utc_iso(utc_now())
            }
        }
    
    def _get_user(self, **params) -> dict[str, Any]:
        """Get user profile and preferences."""
        from clients.postgres_client import PostgresClient
        from config.announcement import get_cached_announcement

        user_id = get_current_user_id()

        # Fetch real user data from database
        from utils.instance import database_name
        db = PostgresClient(database_name(), user_id=user_id)
        user = db.execute_single(
            """SELECT id, email, first_name, last_name, created_at, last_login_at, timezone
               FROM users WHERE id = %(user_id)s""",
            {"user_id": user_id}
        )

        if not user:
            raise NotFoundError("user", str(user_id))

        # Build full name if available
        name = None
        if user.get('first_name') or user.get('last_name'):
            name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()

        return {
            "profile": {
                "id": str(user["id"]),
                "email": user["email"],
                "name": name,
                "created_at": format_utc_iso(user["created_at"]) if user.get("created_at") else None,
                "last_login": format_utc_iso(user["last_login_at"]) if user.get("last_login_at") else None
            },
            "preferences": {
                # Note: Preferences system not implemented yet - only timezone stored in users table
                "theme": None,
                "timezone": user.get("timezone", "UTC"),
                "display_preferences": None,
                "announcement": get_cached_announcement()
            },
            "meta": {
                "loaded_at": format_utc_iso(utc_now())
            }
        }

    def _get_domains(self, **params) -> dict[str, Any]:
        """Get domaindocs from SQLite storage, including shared docs from other users."""
        from utils.userdata_manager import get_user_data_manager
        from utils.domaindoc_shares import resolve_domaindoc, get_accepted_shares

        user_id = get_current_user_id()
        db = get_user_data_manager(user_id)

        label = params.get('label')

        if label:
            try:
                resolved = resolve_domaindoc(user_id, label, require_enabled=False)
            except ValueError:
                raise NotFoundError("domaindoc", label)

            doc = resolved.doc
            sections = resolved.db.fetchall(
                "SELECT header, encrypted__content, encrypted__summary, sort_order, collapsed, parent_section_id "
                "FROM domaindoc_sections WHERE domaindoc_id = :doc_id ORDER BY sort_order",
                {"doc_id": doc["id"]}
            )

            content_parts = []
            for sec in sections:
                decrypted = resolved.db._decrypt_dict(sec)
                header = decrypted.get("header", "")
                section_content = decrypted.get("encrypted__content", "")
                content_parts.append(f"## {header}\n{section_content}")

            result = {
                "label": label,
                "name": doc.get("name", label),
                "content": "\n\n".join(content_parts),
                "enabled": doc.get("enabled", False),
                "archived": doc.get("archived", False),
                "description": doc.get("description"),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "shared": resolved.is_shared
            }
            if resolved.is_shared:
                # Look up owner display name from the share records
                shares = get_accepted_shares(user_id)
                for s in shares:
                    if s.collaborator_label == label:
                        result["shared_by"] = s.owner_display_name
                        break
            return result

        # List domaindocs — own docs + shared docs
        show_archived = params.get('archived', False)
        if show_archived:
            all_docs = db.fetchall("SELECT * FROM domaindocs WHERE archived = TRUE ORDER BY label")
        else:
            all_docs = db.fetchall("SELECT * FROM domaindocs WHERE archived = FALSE ORDER BY label")

        domain_list = [
            {
                "label": doc.get("label"),
                "name": doc.get("name", doc.get("label")),
                "description": doc.get("description", ""),
                "enabled": doc.get("enabled", False),
                "archived": doc.get("archived", False),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "shared": False
            }
            for doc in all_docs
        ]

        # Add shared docs (only for non-archived view)
        if not show_archived:
            shares = get_accepted_shares(user_id)
            for share in shares:
                try:
                    owner_db = get_user_data_manager(share.owner_user_id)
                    owner_docs = owner_db.select("domaindocs", "label = :label AND enabled = TRUE AND archived = FALSE", {"label": share.domaindoc_label})
                    if owner_docs:
                        owner_doc = owner_db._decrypt_dict(owner_docs[0])
                        domain_list.append({
                            "label": share.collaborator_label,
                            "name": owner_doc.get("name", share.domaindoc_label),
                            "description": owner_doc.get("encrypted__description", ""),
                            "enabled": True,
                            "archived": False,
                            "created_at": owner_doc.get("created_at"),
                            "updated_at": owner_doc.get("updated_at"),
                            "shared": True,
                            "shared_by": share.owner_display_name
                        })
                except Exception:
                    logger.warning(f"Failed to load shared domaindoc '{share.domaindoc_label}' from owner {share.owner_user_id}", exc_info=True)
                    continue

        enabled_count = sum(1 for d in domain_list if d.get("enabled", False))

        return {
            "domaindocs": domain_list,
            "total_count": len(domain_list),
            "enabled_count": enabled_count
        }

    def _get_working_memory(self, **params) -> dict[str, Any]:
        """Get current working memory trinket states.

        If 'section' param provided, returns only that trinket's content.
        Otherwise returns all trinkets.
        """
        from cns.services.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        working_memory = orchestrator.working_memory

        section_filter = params.get('section')
        if section_filter:
            state = working_memory.get_trinket_state(section_filter)
            if state is None:
                raise NotFoundError("trinket", section_filter)
            return state

        return working_memory.get_all_trinket_states()

    def _get_lora(self, **params) -> dict[str, Any]:
        """Get user model from the assessment pipeline.

        Returns the current user model observations that provide contextual
        knowledge about the user. Auto-updated every 7 use-days via the
        assessment/synthesis/critic pipeline.
        """
        import re
        from cns.infrastructure.feedback_tracker import FeedbackTracker

        user_id = get_current_user_id()
        tracker = FeedbackTracker()

        lora_content = tracker.get_lora_content(user_id)
        synthesis_xml = lora_content['synthesis_xml']
        needs_checkin = lora_content['needs_checkin']
        tracking_status = tracker.get_tracking_status(user_id)

        # Parse observations from user model XML
        observations = []
        checkin_topics = []
        if synthesis_xml:
            obs_pattern = r'<mira:observation\s+section="([^"]+)"\s+confidence="([^"]+)">(.*?)</mira:observation>'
            for match in re.finditer(obs_pattern, synthesis_xml, re.DOTALL):
                body = match.group(3)
                changelog_match = re.search(r'<changelog>(.*?)</changelog>', body, re.DOTALL)
                changelog = changelog_match.group(1).strip() if changelog_match else ""
                obs_text = re.sub(r'<changelog>.*?</changelog>', '', body, flags=re.DOTALL).strip()

                observations.append({
                    "section_id": match.group(1).strip(),
                    "confidence": match.group(2).strip(),
                    "observation": obs_text,
                    "changelog": changelog
                })

            topic_pattern = r'<mira:topic\s+section="([^"]+)"\s+reason="([^"]+)"\s*(?:/>|>\s*</mira:topic>)'
            for match in re.finditer(topic_pattern, synthesis_xml, re.DOTALL):
                checkin_topics.append({
                    "section_id": match.group(1).strip(),
                    "reason": match.group(2).strip()
                })

        tracking = {
            "use_days_since_synthesis": tracking_status.get('use_days_since_synthesis', 0),
            "last_synthesis_at": format_utc_iso(tracking_status['last_synthesis_at']) if tracking_status.get('last_synthesis_at') else None,
            "has_previous_synthesis": tracking_status.get('has_previous_synthesis', False)
        }

        if not observations:
            return {
                "has_observations": False,
                "observations": [],
                "checkin_topics": [],
                "needs_checkin": False,
                "raw_xml": None,
                "tracking": tracking,
                "message": "No user model yet. Observations emerge after 7 use-days."
            }

        return {
            "has_observations": True,
            "observations": observations,
            "checkin_topics": checkin_topics,
            "needs_checkin": needs_checkin,
            "raw_xml": synthesis_xml,
            "tracking": tracking,
            "message": f"User model: {len(observations)} observations across {len(set(o['section_id'] for o in observations))} sections"
        }


def get_data_handler() -> DataEndpoint:
    """Get data endpoint handler instance."""
    return DataEndpoint()


@router.get("/data")
async def data_endpoint(
    type: DataType = Query(..., description="Data type to retrieve"),
    limit: int | None = Query(None, ge=1, le=100, description="Pagination limit"),
    offset: int | None = Query(None, ge=0, description="Pagination offset"),
    start_date: str | None = Query(None, description="Start date filter (ISO-8601)"),
    end_date: str | None = Query(None, description="End date filter (ISO-8601)"),
    subtype: str | None = Query(None, description="Type-specific filtering"),
    fields: str | None = Query(None, description="Comma-separated field selection"),
    search: str | None = Query(None, description="Search query for full-text search"),
    message_type: str = Query("regular", description="Message type filter: 'regular', 'summaries', or 'all'"),
    label: str | None = Query(None, description="Domain label to retrieve (for type=domaindocs)"),
    archived: bool | None = Query(None, description="Filter archived domaindocs: true=archived only, false/absent=non-archived (for type=domaindocs)"),
    section: str | None = Query(None, description="Specific trinket section to retrieve (for type=working_memory)"),
    current_user: SessionData | APITokenContext = Depends(get_current_user)
):
    """Unified data access endpoint."""
    try:
        handler = get_data_handler()

        # Build request parameters
        request_params: dict[str, object] = {}
        if limit is not None:
            request_params['limit'] = limit
        if offset is not None:
            request_params['offset'] = offset
        if start_date is not None:
            request_params['start_date'] = start_date
        if end_date is not None:
            request_params['end_date'] = end_date
        if subtype is not None:
            request_params['subtype'] = subtype
        if fields is not None:
            request_params['fields'] = fields
        if search is not None:
            request_params['search'] = search
        if message_type is not None:
            request_params['message_type'] = message_type
        if label is not None:
            request_params['label'] = label
        if archived is not None:
            request_params['archived'] = archived
        if section is not None:
            request_params['section'] = section

        response = handler.handle_request(
            data_type=type,
            request_params=request_params,
            user_id=current_user.user_id
        )
        
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
    except Exception as e:
        logger.error(f"Data endpoint error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Data retrieval failed"
                }
            }
        )
