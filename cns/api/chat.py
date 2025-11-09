"""
Chat API endpoint - simple request/response over HTTP.

Provides a non-streaming JSON API to send a user message and receive
the assistant's response plus structured metadata. Authenticated via
Bearer token (header) or session cookie.
"""
import base64
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils.distributed_lock import UserRequestLock
from utils.user_context import get_current_user_id
from utils.text_sanitizer import sanitize_message_content
from utils.timezone_utils import utc_now, format_utc_iso
from .base import BaseHandler, APIResponse, ValidationError, create_success_response
from cns.services.orchestrator import get_orchestrator
from cns.infrastructure.continuum_pool import get_continuum_pool


logger = logging.getLogger(__name__)

router = APIRouter()


# Image validation constants (keep consistent with websocket implementation)
SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_MB = 5

# Distributed per-user request lock (coordinates across workers)
_user_request_lock = UserRequestLock(ttl=60)


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., description="User message text")
    image: Optional[str] = Field(None, description="Optional image as base64 string (no data: prefix)")
    image_type: Optional[str] = Field(None, description="MIME type for image (e.g., image/jpeg)")


class ChatEndpoint(BaseHandler):
    """Handler for HTTP chat requests (non-streaming)."""

    def process_request(self, *, user_id: str, message: str, image: Optional[str], image_type: Optional[str]) -> APIResponse:
        start_time = utc_now()

        # Set user context for RLS and utility functions
        from utils.user_context import set_current_user_id
        set_current_user_id(user_id)

        # Basic validation
        msg = (message or "").strip()
        if not msg:
            raise ValidationError("Message cannot be empty")

        # Sanitize text
        msg = sanitize_message_content(msg)

        # Validate image if provided
        if image:
            if not image_type:
                raise ValidationError("image_type is required when image is provided")
            if image_type not in SUPPORTED_IMAGE_FORMATS:
                raise ValidationError(
                    f"Unsupported image format. Supported: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
                )
            try:
                decoded = base64.b64decode(image, validate=True)
                if len(decoded) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                    raise ValidationError(f"Image exceeds maximum size of {MAX_IMAGE_SIZE_MB}MB")
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Invalid base64 image: {str(e)}")

        # Concurrency control: one active request per user
        if not _user_request_lock.acquire(user_id):
            # Use a validation error to preserve consistent error envelope
            raise ValidationError("Another chat request is already in progress for this user")

        try:
            # Resolve dependencies
            orchestrator = get_orchestrator()
            continuum_pool = get_continuum_pool()

            # Get the user's continuum
            continuum = continuum_pool.get_or_create()

            # Build content (text or multimodal array)
            if image and image_type:
                message_content = [
                    {"type": "text", "text": msg},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_type,
                            "data": image
                        }
                    }
                ]
            else:
                message_content = msg

            # Create a Unit of Work and process via orchestrator
            uow = continuum_pool.begin_work(continuum)

            from config.config_manager import config as app_config
            continuum, response_text, metadata = orchestrator.process_message(
                continuum,
                message_content,
                app_config.system_prompt,
                stream=True,           # orchestrator currently streams internally
                stream_callback=None,   # no external streaming for HTTP endpoint
                unit_of_work=uow,
            )

            # Commit batched changes
            uow.commit()

            processing_time_ms = int((utc_now() - start_time).total_seconds() * 1000)

            # Build response
            data: Dict[str, Any] = {
                "continuum_id": str(continuum.id),
                "response": response_text,
                "metadata": {
                    "tools_used": metadata.get("tools_used", []),
                    "referenced_memories": metadata.get("referenced_memories", []),
                    "surfaced_memories": metadata.get("surfaced_memories", []),
                    "processing_time_ms": processing_time_ms,
                },
            }

            return create_success_response(
                data=data,
                meta={
                    "timestamp": format_utc_iso(utc_now()),
                },
            )

        finally:
            _user_request_lock.release(user_id)


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Send a message and receive assistant response as JSON."""
    try:
        user_id = get_current_user_id()
        handler = ChatEndpoint()
        response = handler.handle_request(
            user_id=user_id,
            message=request.message,
            image=request.image,
            image_type=request.image_type,
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
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Chat processing failed"
                }
            }
        )

