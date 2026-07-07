"""
WebSocket chat endpoint - Real-time bidirectional communication for MIRA.

Provides persistent connections with streaming responses, eliminating the
complexity of SSE and dual code paths. Direct service integration with
proper user context management.
"""
import base64
import logging
import asyncio
import contextvars
import threading
from collections.abc import Callable
from functools import partial
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from cns.core.continuum import Continuum
from cns.core.message import ContentBlock
from cns.services.async_work_barrier import get_async_work_barrier
from cns.services.orchestrator import get_orchestrator
from cns.infrastructure.continuum_pool import get_continuum_pool
from cns.infrastructure.continuum_repository import get_continuum_repository
from clients.files_manager import FilesManager
from config.config_manager import config as app_config
from utils.distributed_lock import UserRequestLock
from utils.document_processing import process_document, ProcessedDocument, SUPPORTED_DOCUMENT_FORMATS, MAX_DOCUMENT_SIZE_MB
from utils.image_compression import compress_image, CompressedImage
from clients.llm.events import GenerationCancelled
from utils.user_context import set_current_user_id, clear_user_context, set_cancel_event
from utils.timezone_utils import utc_now
from utils.text_sanitizer import sanitize_message_content

# Import billing exception for proper type checking (None if OSS mode)
try:
    from billing.exceptions import InsufficientBalanceError
except ImportError:
    InsufficientBalanceError = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

router = APIRouter()


class WebSocketAuthError(Exception):
    """Raised when WebSocket authentication fails."""
    pass

def get_friendly_error_message(error: Exception) -> str:
    """
    Convert technical error messages into user-friendly explanations.
    
    Args:
        error: The exception that occurred
        
    Returns:
        A friendly error message for the user
    """
    error_str = str(error).lower()
    
    # API usage limit errors
    if "usage limit" in error_str or "rate limit" in error_str:
        return ("I'm currently rate limited. Please try again in a few moments. "
               "If this persists, the API usage limits may have been reached.")
    
    # Authentication errors (third-party provider key issues)
    if "authentication failed" in error_str or "401" in error_str:
        return ("The API provider outside Mira had a problem. "
               "Please try again in a couple minutes.")
    
    # Model availability errors
    if "no allowed providers" in error_str or "model" in error_str and "404" in error_str:
        return ("The AI model I'm trying to use isn't available. "
               "Please contact support to update the configuration.")
    
    # Network errors
    if "connection" in error_str or "network" in error_str:
        return ("I'm having trouble connecting to the AI service. "
               "Please check your internet connection and try again.")
    
    # Timeout errors
    if "timeout" in error_str:
        return ("The request took too long to process. "
               "Please try again with a simpler message.")
    
    # Server errors
    if any(code in error_str for code in ["500", "502", "503"]):
        return ("The AI service is experiencing technical difficulties. "
               "Please try again in a few moments.")
    
    # Default message for unknown errors
    return ("I encountered an unexpected error while processing your message. "
           "Please try again, and if the problem persists, contact support.")

# Distributed per-user request lock
_user_request_lock = UserRequestLock(ttl=60)

# Connection tracking for graceful shutdown
_active_connections: dict[str, WebSocket] = {}

# Image validation constants
SUPPORTED_IMAGE_FORMATS = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
MAX_IMAGE_SIZE_MB = 5


async def close_all_connections() -> None:
    """Close all active WebSocket connections gracefully."""
    if not _active_connections:
        return

    logger.info(f"Closing {len(_active_connections)} active WebSocket connections")

    # Send shutdown message to all connections
    send_tasks = []
    for conn_id, websocket in list(_active_connections.items()):
        try:
            # Create coroutine for sending shutdown message
            send_tasks.append(websocket.send_json({
                "type": "server_shutdown",
                "message": "Server is shutting down"
            }))
        except:
            pass

    # Wait for all messages to be sent (best-effort, 2s timeout)
    if send_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*send_tasks, return_exceptions=True),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timed out sending shutdown messages to WebSocket clients")

    # Close all connections (TCP close handshake, 3s timeout)
    close_tasks = []
    for conn_id, websocket in list(_active_connections.items()):
        try:
            close_tasks.append(websocket.close())
        except:
            pass

    if close_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*close_tasks, return_exceptions=True),
                timeout=3.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timed out closing WebSocket connections")

    _active_connections.clear()


class WebSocketChatHandler:
    """Handler for WebSocket chat connections."""
    
    def __init__(self):
        """Initialize with singleton service dependencies."""
        self.orchestrator = get_orchestrator()
        self.continuum_pool = get_continuum_pool()
        self.continuum_repo = get_continuum_repository()
    
    async def authenticate(self, websocket: WebSocket) -> str:
        """
        Authenticate WebSocket connection via first message.

        Returns user_id on success. Sends error to websocket and raises
        WebSocketAuthError on failure.
        """
        try:
            # Wait for auth message (with timeout)
            auth_data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=10.0
            )

            if auth_data.get("type") != "auth":
                await websocket.send_json({
                    "type": "error",
                    "message": "First message must be authentication"
                })
                raise WebSocketAuthError("First message must be authentication")

            # OSS single-user: validate token against the app-stored API key
            token = auth_data.get("token")
            if not token:
                await websocket.send_json({
                    "type": "error",
                    "message": "Missing authentication token"
                })
                raise WebSocketAuthError("Missing authentication token")

            api_key = getattr(websocket.app.state, "api_key", None)
            single_user_id = getattr(websocket.app.state, "single_user_id", None)
            if not api_key or not single_user_id:
                await websocket.send_json({
                    "type": "error",
                    "message": "Single-user credentials not configured"
                })
                raise WebSocketAuthError("Single-user credentials not configured")

            if token != api_key:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid or expired session"
                })
                raise WebSocketAuthError("Invalid or expired session")

            # Send auth success
            user_id = str(single_user_id)
            await websocket.send_json({
                "type": "auth_success",
                "user_id": user_id
            })

            # Set user context for the websocket connection
            set_current_user_id(user_id)

            return user_id

        except asyncio.TimeoutError:
            await websocket.send_json({
                "type": "error",
                "message": "Authentication timeout"
            })
            raise WebSocketAuthError("Authentication timeout")
        except WebSocketAuthError:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": f"Authentication failed: {str(e)}"
            })
            raise WebSocketAuthError(f"Authentication failed: {e}") from e
    
    async def process_message_streaming(
        self,
        websocket: WebSocket,
        user_id: str,
        message_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process a chat message with real-time streaming via queue.

        Returns metadata about the completed request.
        """
        try:
            # Extract include_thinking flag (frontend opts in per-request)
            include_thinking = message_data.get("include_thinking", False)

            # Extract and validate message components
            content = message_data.get("content", "").strip()
            if not content:
                await websocket.send_json({"type": "error", "message": "Message cannot be empty"})
                return {"error": "Message cannot be empty"}

            # Sanitize content
            content = sanitize_message_content(content)

            # Enforce content length limit (100KB ~= 25K tokens) to prevent context overflow
            MAX_CONTENT_LENGTH = 100_000
            if len(content) > MAX_CONTENT_LENGTH:
                msg = f"Message too long ({len(content):,} chars). Maximum is {MAX_CONTENT_LENGTH:,} characters."
                await websocket.send_json({"type": "error", "message": msg})
                return {"error": msg}

            # Extract optional image data
            image_base64 = message_data.get("image")
            image_type = message_data.get("image_type")

            # Validate and compress image data if provided
            compressed: CompressedImage | None = None
            if image_base64:
                if not image_type:
                    await websocket.send_json({"type": "error", "message": "image_type is required when image is provided"})
                    return {"error": "image_type is required"}

                if image_type not in SUPPORTED_IMAGE_FORMATS:
                    msg = f"Unsupported image format. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
                    await websocket.send_json({"type": "error", "message": msg})
                    return {"error": msg}

                # Validate base64 encoding, size, and compress
                try:
                    decoded = base64.b64decode(image_base64, validate=True)
                    if len(decoded) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                        msg = f"Image exceeds maximum size of {MAX_IMAGE_SIZE_MB}MB"
                        await websocket.send_json({"type": "error", "message": msg})
                        return {"error": msg}

                    # Compress to both tiers: inference (1200px) and storage (512px WebP)
                    compressed = compress_image(decoded, image_type)

                except ValueError as e:
                    # compress_image raises ValueError on failure
                    msg = f"Image compression failed: {e}"
                    await websocket.send_json({"type": "error", "message": msg})
                    return {"error": msg}
                except Exception as e:
                    msg = f"Invalid base64 image: {str(e)}"
                    await websocket.send_json({"type": "error", "message": msg})
                    return {"error": msg}

            ctx = contextvars.copy_context()

            # Give previous-turn tool-result compaction a bounded head start before
            # loading the hot continuum. Timeout is intentionally fail-open.
            wait_for_background_work = partial(
                get_async_work_barrier().wait_for_user,
                user_id,
                timeout=app_config.api.async_work_barrier_timeout_seconds,
                source="tool_summarizer",
            )
            await run_in_threadpool(ctx.run, wait_for_background_work)

            # Get continuum after the gate so it observes any completed cache patch.
            logger.info(f"Getting continuum for user {user_id}")
            continuum = await run_in_threadpool(
                ctx.run,
                self._get_user_continuum
            )
            logger.info(f"Got continuum {continuum.id}")

            # Increment segment turn counter
            # increment_segment_turn also sets current_segment_id contextvar and
            # (for new segments) defers sentinel persistence to commit-time.
            result = await run_in_threadpool(
                ctx.run,
                self.continuum_pool.repository.increment_segment_turn,
                continuum.id,
                user_id
            )
            segment_turn_number = result.turn_number
            segment_id = result.segment_id

            # Extract and process document if provided
            document_base64 = message_data.get("document")
            document_type = message_data.get("document_type")

            processed_doc: ProcessedDocument | None = None
            files_manager: FilesManager | None = None
            if document_base64:
                if not document_type:
                    await websocket.send_json({"type": "error", "message": "document_type is required when document is provided"})
                    return {"error": "document_type is required"}

                if document_type not in SUPPORTED_DOCUMENT_FORMATS:
                    msg = "Unsupported document format. Supported: PDF, DOCX, XLSX, TXT, CSV, JSON"
                    await websocket.send_json({"type": "error", "message": msg})
                    return {"error": msg}

                try:
                    decoded = base64.b64decode(document_base64, validate=True)
                    if len(decoded) > MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                        msg = f"Document exceeds maximum size of {MAX_DOCUMENT_SIZE_MB}MB"
                        await websocket.send_json({"type": "error", "message": msg})
                        return {"error": msg}

                    files_manager = self.orchestrator.llm_provider.create_files_manager()

                    processed_doc = process_document(
                        decoded,
                        document_type,
                        files_manager=files_manager,
                        filename=f"document.{document_type.split('/')[-1]}",
                        segment_id=segment_id
                    )

                except ValueError as e:
                    msg = f"Document processing failed: {e}"
                    await websocket.send_json({"type": "error", "message": msg})
                    return {"error": msg}
                except Exception as e:
                    msg = f"Document upload failed: {str(e)}"
                    await websocket.send_json({"type": "error", "message": msg})
                    return {"error": msg}

            # Create queue for streaming
            queue = asyncio.Queue(maxsize=100)
            loop = asyncio.get_event_loop()

            # Cancellation: threading.Event set by cancel listener, read by LLM provider via contextvar.
            # Must be set BEFORE ctx snapshot so copy_context captures it.
            cancel_event = threading.Event()
            set_cancel_event(cancel_event)
            # Re-snapshot context so the cancel_event contextvar is included
            ctx = contextvars.copy_context()

            # Create callback that pushes to queue
            def stream_to_queue(event_data: dict[str, Any]):
                """Push event to queue from sync context."""
                future = asyncio.run_coroutine_threadsafe(
                    queue.put(event_data),
                    loop
                )
                # Wait for put to complete (natural backpressure)
                future.result()

            # Listen for cancel messages from client during streaming.
            # Only reader on the WebSocket during streaming (handle_connection is awaiting us).
            async def cancel_listener():
                try:
                    while not cancel_event.is_set():
                        data = await websocket.receive_json()
                        if data.get("type") == "cancel":
                            logger.info(f"Cancel requested by user {user_id}")
                            cancel_event.set()
                            return
                except (WebSocketDisconnect, RuntimeError):
                    # Client disconnected — treat as implicit cancel
                    cancel_event.set()
                except asyncio.CancelledError:
                    pass  # Normal cleanup when streaming completes

            listener_task = asyncio.create_task(cancel_listener())

            # Start orchestrator processing in thread pool with streaming (with context)
            process_task = asyncio.create_task(
                run_in_threadpool(
                    ctx.run,
                    self._process_with_orchestrator,
                    continuum,
                    content,
                    compressed,  # Pass compressed image (or None)
                    processed_doc,  # Pass processed document (or None)
                    document_base64,  # Original document for storage
                    document_type,  # Document MIME type
                    stream_to_queue,  # Pass our callback
                    segment_turn_number,  # Turn count within segment
                )
            )

            # Consume queue and stream to websocket.
            # Client may disconnect mid-stream (tab close, navigation). The
            # orchestrator must still finish and commit — we just stop sending.
            client_gone = False
            error_result = None
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=60.0)
                    except asyncio.TimeoutError:
                        if process_task.done():
                            break
                        continue

                    event_type = event.get("type")

                    if event_type == "complete":
                        break

                    if event_type == "cancelled":
                        break

                    if event_type == "interrupted":
                        if not client_gone:
                            try:
                                await websocket.send_json(event)
                            except (WebSocketDisconnect, RuntimeError):
                                client_gone = True
                        error_result = {
                            "interrupted": True,
                            "response": event.get("response", ""),
                            "message": event.get("message", "Unknown error")
                        }
                        break

                    if event_type == "error":
                        if not client_gone:
                            try:
                                await websocket.send_json(event)
                            except (WebSocketDisconnect, RuntimeError):
                                client_gone = True
                        error_result = {"error": event.get("message")}
                        break

                    # Best-effort send to client. On failure, keep draining
                    # the queue so the orchestrator thread doesn't deadlock.
                    if client_gone:
                        continue

                    try:
                        if event_type == "text":
                            await websocket.send_json({
                                "type": "text",
                                "content": event.get("content", "")
                            })
                        elif event_type == "thinking":
                            if include_thinking:
                                await websocket.send_json({
                                    "type": "thinking",
                                    "content": event.get("content", "")
                                })
                        elif event_type == "tool_event":
                            await websocket.send_json({
                                "type": "tool",
                                "event": event.get("event"),
                                "name": event.get("tool")
                            })
                        elif event_type == "model_error":
                            await websocket.send_json({
                                "type": "model_error",
                                "message": "The AI model made an invalid tool call. Attempting to recover..."
                            })
                        elif event_type == "provider_switch":
                            await websocket.send_json({
                                "type": "provider_switch",
                                "backup_model": event.get("backup_model"),
                                "reason": event.get("reason", "")
                            })
                    except (WebSocketDisconnect, RuntimeError):
                        client_gone = True
                        logger.info(f"Client disconnected mid-stream for user {user_id}, completing processing")

                # Get final result from orchestrator
                result = await process_task
                if error_result:
                    return error_result
                return result

            except asyncio.CancelledError:
                process_task.cancel()
                raise
            finally:
                # Stop the cancel listener (no longer needed)
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass
                if not process_task.done():
                    process_task.cancel()

        except Exception as e:
            logger.error(f"Streaming message error: {e}", exc_info=True)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": get_friendly_error_message(e)
                })
            except (WebSocketDisconnect, RuntimeError):
                pass  # Client already gone
            return {"error": str(e)}

    def _process_with_orchestrator(
        self,
        continuum,
        content: str,
        compressed: CompressedImage | None = None,
        processed_doc: ProcessedDocument | None = None,
        document_base64: str | None = None,
        document_type: str | None = None,
        stream_callback: Callable[[dict[str, object]], None] | None = None,
        segment_turn_number: int = 1
    ) -> dict[str, Any]:
        """Process message through orchestrator with streaming callback."""
        # Get system prompt from config
        from config.config_manager import config
        if not config.system_prompt:
            raise ValueError("System prompt not configured")

        # Build content arrays (inference tier for LLM, storage tier for persistence)
        inference_content: str | list[ContentBlock]
        storage_content: str | list[ContentBlock] | None = None

        if compressed:
            # Image: Inference tier (1200px) for current LLM call
            inference_content = [
                {"type": "text", "text": content},
                {
                    "type": "image",
                    "media_type": compressed.inference_media_type,
                    "data": compressed.inference_base64,
                }
            ]
            # Storage tier (512px WebP) for persistence and multi-turn context
            storage_content = [
                {"type": "text", "text": content},
                {
                    "type": "image",
                    "media_type": compressed.storage_media_type,
                    "data": compressed.storage_base64,
                }
            ]
        elif processed_doc:
            # Document handling based on content_type
            if processed_doc.content_type == "container_upload":
                # Structured data: Files API with file_id (CSV, XLSX, JSON for code execution)
                doc_block: ContentBlock = {
                    "type": "file_ref",
                    "file_id": processed_doc.data  # file_id from Files API
                }
            elif processed_doc.content_type == "document":
                # PDF: Base64 document block
                doc_block = {
                    "type": "document",
                    "media_type": processed_doc.media_type,
                    "data": processed_doc.data,
                }
            else:
                # DOCX/plain text: Extracted text
                doc_block = {
                    "type": "text",
                    "text": f"[Document: {processed_doc.media_type}]\n{processed_doc.data}",
                }

            inference_content = [{"type": "text", "text": content}, doc_block]
            # Storage: Use same block as inference (file_id persists until segment collapse)
            storage_content = [
                {"type": "text", "text": content},
                doc_block  # Reuse same block (file_id or base64)
            ]
        else:
            inference_content = content

        # Track streamed text for error recovery
        streamed_parts: list[str] = []
        original_callback = stream_callback

        def tracking_callback(event_data: dict[str, object]):
            if event_data.get("type") == "text":
                streamed_parts.append(str(event_data.get("content", "")))
            if original_callback:
                original_callback(event_data)

        # Create unit of work for batch operations
        unit_of_work = self.continuum_pool.begin_work(continuum)

        try:
            # Process message with streaming callback
            continuum, response_text, metadata = self.orchestrator.process_message(
                continuum,
                inference_content,
                config.system_prompt,
                stream=True,
                stream_callback=tracking_callback,
                unit_of_work=unit_of_work,
                storage_content=storage_content,  # 512px WebP for persistence
                segment_turn_number=segment_turn_number,  # Turn count within segment
            )

            # Commit all changes atomically
            unit_of_work.commit()

            # Signal completion
            if stream_callback:
                stream_callback({"type": "complete"})

            return {
                "continuum": continuum,
                "response": response_text,
                "metadata": metadata
            }
        except GenerationCancelled:
            # User cancelled — save partial response without error annotation
            accumulated = "".join(streamed_parts).strip()
            if accumulated:
                try:
                    from cns.core.message import Message
                    user_msg = Message(content=content, role="user")
                    assistant_msg = Message(
                        content=accumulated,
                        role="assistant",
                        metadata={"partial_response": True, "cancelled": True}
                    )
                    unit_of_work.add_messages(user_msg, assistant_msg)
                    unit_of_work.commit()
                    logger.info(f"Saved cancelled response ({len(accumulated)} chars)")
                except Exception as save_err:
                    logger.warning(f"Failed to save cancelled response: {save_err}")

            if stream_callback:
                stream_callback({"type": "cancelled"})

            return {
                "cancelled": True,
                "continuum": continuum,
                "response": accumulated,
                "metadata": {"tools_used": [], "cancelled": True}
            }

        except Exception as e:
            # Save partial response if text was already streamed to the user
            accumulated = "".join(streamed_parts).strip()
            friendly_message = get_friendly_error_message(e)
            logger.error(f"Orchestrator processing error: {e}", exc_info=True)
            if accumulated:
                error_note = f"\n\n---\n*[Response interrupted: {type(e).__name__}]*"
                interrupted_response = accumulated + error_note
                try:
                    from cns.core.message import Message
                    user_msg = Message(content=content, role="user")
                    assistant_msg = Message(
                        content=interrupted_response,
                        role="assistant",
                        metadata={"partial_response": True}
                    )
                    unit_of_work.add_messages(user_msg, assistant_msg)
                    unit_of_work.commit()
                    logger.info(f"Saved partial response ({len(accumulated)} chars) after {type(e).__name__}")
                except Exception as save_err:
                    logger.warning(f"Failed to save partial response: {save_err}")

                interruption_event: dict[str, object] = {
                    "type": "interrupted",
                    "continuum_id": str(continuum.id),
                    "message": friendly_message,
                    "response": interrupted_response,
                }

                if InsufficientBalanceError is not None and isinstance(e, InsufficientBalanceError):
                    interruption_event.update({
                        "error_type": "insufficient_balance",
                        "balance": str(e.balance),
                        "next_drip_at": e.next_drip_at.isoformat(),
                        "seconds_until_drip": int(e.time_until_drip.total_seconds())
                    })

                if stream_callback:
                    stream_callback(interruption_event)

                return {
                    "interrupted": True,
                    "continuum": continuum,
                    "response": interrupted_response,
                    "continuum_id": str(continuum.id),
                    "metadata": {"tools_used": [], "interrupted": True}
                }

            # Handle InsufficientBalanceError with specific error type for frontend
            if InsufficientBalanceError is not None and isinstance(e, InsufficientBalanceError):
                if stream_callback:
                    stream_callback({
                        "type": "error",
                        "error_type": "insufficient_balance",
                        "message": str(e),
                        "balance": str(e.balance),
                        "next_drip_at": e.next_drip_at.isoformat(),
                        "seconds_until_drip": int(e.time_until_drip.total_seconds())
                    })
                return {"error": str(e)}

            # Send error through callback
            if stream_callback:
                stream_callback({
                    "type": "error",
                    "message": friendly_message
                })
            return {"error": friendly_message}
    
    def _get_user_continuum(self) -> Continuum:
        """Get user's single continuum."""
        # Context copied from async handler via contextvars.copy_context()
        # Get or create the user's single continuum
        continuum = self.continuum_pool.get_or_create()
        return continuum
    
    async def handle_connection(self, websocket: WebSocket, user_id: str):
        """
        Main message loop for authenticated WebSocket connection.
        """
        connection_id = str(uuid4())
        _active_connections[connection_id] = websocket

        # Acquire user lock
        if not _user_request_lock.acquire(user_id):
            logger.warning(f"Failed to acquire lock for user {user_id} - stale lock from previous connection still active")
            await websocket.send_json({
                "type": "error",
                "message": "MIRA is designed in a way where each user has a 'lock' on a connection to the server. For some reason yours didn't expire last time you disconnected. It will clear in 60 seconds. Please refresh the page in one minute."
            })
            await websocket.close()
            return
        
        try:
            while True:
                # Receive message
                message_data = await websocket.receive_json()
                
                if message_data.get("type") == "ping":
                    # Simple keepalive
                    await websocket.send_json({"type": "pong"})
                    continue
                
                if message_data.get("type") != "message":
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_data.get('type')}"
                    })
                    continue

                logger.info(f"Received message from user {user_id}: {message_data.get('content', '')[:100]}")

                # Process the message with real-time streaming
                start_time = utc_now()
                result = await self.process_message_streaming(websocket, user_id, message_data)

                logger.info(f"Message processing result: {result.get('error', 'success')}")

                if "error" in result:
                    # Error already sent via websocket in process_message_streaming
                    continue

                if result.get("interrupted"):
                    # Partial response already sent via websocket in process_message_streaming
                    continue

                processing_time_ms = int((utc_now() - start_time).total_seconds() * 1000)

                if result.get("cancelled"):
                    # Generation was cancelled by user — send partial response
                    try:
                        await websocket.send_json({
                            "type": "cancelled",
                            "continuum_id": str(result["continuum"].id),
                            "response": result.get("response", ""),
                            "metadata": {"processing_time_ms": processing_time_ms}
                        })
                    except (WebSocketDisconnect, RuntimeError):
                        pass  # Client already gone (disconnect-triggered cancel)
                    continue

                # Send completion message
                await websocket.send_json({
                    "type": "complete",
                    "continuum_id": str(result["continuum"].id),
                    "response": result.get("response", ""),  # Include the response text!
                    "metadata": {
                        "tools_used": result["metadata"].get("tools_used", []),
                        "processing_time_ms": processing_time_ms,
                        "emotion": result["metadata"].get("emotion")  # Include emotion emoji
                    }
                })
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for user {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {e}", exc_info=True)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": get_friendly_error_message(e)
                })
            except:
                pass
        finally:
            # Cleanup
            clear_user_context()
            _user_request_lock.release(user_id)
            _active_connections.pop(connection_id, None)


# WebSocket endpoint
@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket chat endpoint with authentication and streaming.
    
    Protocol:
    1. Accept connection
    2. Receive auth message: {"type": "auth", "token": "..."}
    3. Send auth result
    4. Enter message loop
    
    Message format:
    - Client: {"type": "message", "content": "...", "stream": bool, "image": "base64...", "image_type": "image/jpeg"}
    - Server: Various message types (text, tool, error, complete)
    """
    await websocket.accept()
    
    try:
        handler = WebSocketChatHandler()

        # Authenticate
        try:
            user_id = await handler.authenticate(websocket)
        except WebSocketAuthError:
            await websocket.close()
            return

        # Handle connection
        await handler.handle_connection(websocket, user_id)

    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
        await websocket.close()
