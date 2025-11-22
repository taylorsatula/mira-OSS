"""
Data API endpoint - unified data access with type-based routing.

Uses proper repository methods for safe data access with user isolation.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from utils.user_context import get_current_user_id
from main import get_current_user
from .base import BaseHandler, ValidationError, NotFoundError
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)

router = APIRouter()


class DataType(str, Enum):
    """Supported data types."""
    HISTORY = "history"
    MEMORIES = "memories"
    DASHBOARD = "dashboard"
    USER = "user"
    DOMAINS = "domains"


class DataEndpoint(BaseHandler):
    """Main data endpoint handler with type-based routing."""
    
    def process_request(self, **params) -> Dict[str, Any]:
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
        elif data_type == DataType.DOMAINS:
            return self._get_domains(**request_params)
        else:
            raise ValidationError(f"Invalid data type: {data_type}")
    
    def _get_history(self, **params) -> Dict[str, Any]:
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
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if end_date:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

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
    
    def _get_memories(self, **params) -> Dict[str, Any]:
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

        # If search query provided, use search instead of regular listing
        if search_query:
            memory_data = lt_db.search_memories(
                search_query=search_query,
                offset=offset,
                limit=limit,
                user_id=user_id
            )
        else:
            memory_data = lt_db.get_all_memories(
                offset=offset,
                limit=limit,
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
    
    def _get_dashboard(self, **params) -> Dict[str, Any]:
        """Get dashboard data - system health and context usage metrics."""
        from clients.postgres_client import PostgresClient
        from cns.infrastructure.continuum_repository import get_continuum_repository
        from config.config import get_config

        user_id = get_current_user_id()

        # Simple system health check
        db_healthy = True
        try:
            db = PostgresClient("mira_service", user_id=user_id)
            db.execute_single("SELECT 1")
        except Exception:
            db_healthy = False

        # Get context usage from continuum runtime metrics
        repo = get_continuum_repository()
        continuum = repo.get_continuum(user_id)
        config = get_config()

        current_tokens = 0
        cached_tokens = 0
        max_tokens = config.context_window_tokens

        if continuum:
            current_tokens = continuum._cumulative_tokens
            cached_tokens = continuum._cached_up_to_tokens

        # Calculate percentage used
        percentage_used = (current_tokens / max_tokens * 100) if max_tokens > 0 else 0

        return {
            "system_health": "healthy" if db_healthy else "degraded",
            "context_usage": {
                "current_tokens": current_tokens,
                "cached_tokens": cached_tokens,
                "max_tokens": max_tokens,
                "percentage_used": round(percentage_used, 2)
            },
            "meta": {
                "timestamp": format_utc_iso(utc_now())
            }
        }
    
    def _get_user(self, **params) -> Dict[str, Any]:
        """Get user profile and preferences."""
        from utils.database_session_manager import get_shared_session_manager

        user_id = get_current_user_id()

        # Fetch real user data from database
        session_manager = get_shared_session_manager()
        with session_manager.get_session() as session:
            user = session.execute_single(
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
                "display_preferences": None
            },
            "meta": {
                "loaded_at": format_utc_iso(utc_now())
            }
        }

    def _get_domains(self, **params) -> Dict[str, Any]:
        """Get domain knowledge blocks."""
        from cns.services.domain_knowledge_service import get_domain_knowledge_service

        user_id = get_current_user_id()

        service = get_domain_knowledge_service()
        if not service:
            return {
                "domains": [],
                "enabled": False,
                "message": "Domain knowledge feature not available"
            }

        domain_label = params.get('domain_label')

        # If specific domain requested, get its content
        if domain_label:
            try:
                content = service.get_block_content(
                    domain_label=domain_label,
                    prompt_formatted=False
                )

                if content is None:
                    raise NotFoundError("domain", domain_label)

                # Get domain metadata
                all_domains = service.get_all_domains()
                domain_info = next((d for d in all_domains if d['domain_label'] == domain_label), None)

                return {
                    "domain_label": domain_label,
                    "domain_name": domain_info.get('domain_name') if domain_info else domain_label,
                    "content": content,
                    "enabled": domain_info.get('enabled', False) if domain_info else False,
                    "block_description": domain_info.get('block_description') if domain_info else None
                }
            except Exception as e:
                logger.error(f"Failed to get domain content: {e}")
                raise

        # Otherwise list all domains
        domains = service.get_all_domains()
        enabled_count = sum(1 for d in domains if d.get('enabled', False))

        return {
            "domains": domains,
            "total_count": len(domains),
            "enabled_count": enabled_count,
            "enabled": True
        }


def get_data_handler() -> DataEndpoint:
    """Get data endpoint handler instance."""
    return DataEndpoint()


@router.get("/data")
async def data_endpoint(
    type: DataType = Query(..., description="Data type to retrieve"),
    limit: Optional[int] = Query(None, ge=1, le=100, description="Pagination limit"),
    offset: Optional[int] = Query(None, ge=0, description="Pagination offset"),
    start_date: Optional[str] = Query(None, description="Start date filter (ISO-8601)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO-8601)"),
    subtype: Optional[str] = Query(None, description="Type-specific filtering"),
    fields: Optional[str] = Query(None, description="Comma-separated field selection"),
    search: Optional[str] = Query(None, description="Search query for full-text search"),
    message_type: Optional[str] = Query(None, description="Message type filter: 'regular', 'summaries', or 'all' (default='regular')"),
    domain_label: Optional[str] = Query(None, description="Specific domain label to retrieve (for type=domains)"),
    current_user: dict = Depends(get_current_user)
):
    """Unified data access endpoint."""
    try:
        handler = get_data_handler()
        
        # Build request parameters
        request_params = {}
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
        if domain_label is not None:
            request_params['domain_label'] = domain_label

        response = handler.handle_request(
            data_type=type,
            request_params=request_params,
            user_id=current_user["user_id"]
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
