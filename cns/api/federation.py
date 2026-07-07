"""
Federation webhook endpoint for receiving messages from Lattice.

This endpoint receives validated, de-duplicated messages from the Lattice
discovery daemon and delivers them to local users via the pager tool.
"""

import logging
import secrets
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

LATTICE_DELIVERY_TOKEN_HEADER = "X-Lattice-Delivery-Token"


class FederationDeliveryPayload(BaseModel):
    """Payload from Lattice for federated message delivery."""
    from_address: str = Field(..., min_length=3, max_length=254, description="Sender's federated address (user@domain)")
    to_user_id: UUID = Field(..., description="Resolved recipient user_id (UUID)")
    content: str = Field(..., min_length=1, max_length=1000, description="Message content")
    priority: int = Field(default=0, ge=0, le=2, description="Priority: 0=normal, 1=high, 2=urgent")
    message_id: str = Field(..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$", description="Unique message ID for idempotency")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")
    sender_verified: bool = Field(default=False, description="Whether sender signature was verified")
    sender_server_id: str = Field(..., min_length=1, max_length=253, description="Sending server's domain")


class FederationDeliveryResponse(BaseModel):
    """Response to Lattice after delivery attempt."""
    status: Literal["delivered", "failed"] = Field(..., description="delivered or failed")
    message_id: str = Field(..., description="Echo back message_id")
    error: str | None = Field(default=None, description="Error message if failed")


def _require_lattice_delivery_token(header_value: str | None) -> None:
    from clients.vault_client import get_service_config

    try:
        expected_token = get_service_config("lattice_delivery_token")
    except Exception as e:
        logger.error(f"Lattice delivery token is not configured: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Federation delivery not configured")

    if not header_value:
        raise HTTPException(status_code=401, detail=f"Missing {LATTICE_DELIVERY_TOKEN_HEADER} header")
    if not secrets.compare_digest(header_value, expected_token):
        raise HTTPException(status_code=403, detail="Invalid Lattice delivery token")


@router.post("/federation/deliver", response_model=FederationDeliveryResponse)
def receive_federation_delivery(
    payload: FederationDeliveryPayload,
    x_lattice_delivery_token: str | None = Header(default=None)
) -> FederationDeliveryResponse:
    """
    Receive a federated message from Lattice and deliver to local user.

    Lattice has already:
    - Verified the sender's signature
    - Checked rate limits
    - De-duplicated the message
    - Resolved the username to user_id

    This endpoint just needs to write to the user's pager.

    Returns:
        200 + status=delivered: Success
        4xx: Permanent failure (Lattice won't retry)
        5xx: Temporary failure (Lattice will retry)
    """
    try:
        _require_lattice_delivery_token(x_lattice_delivery_token)
        if not payload.sender_verified:
            raise HTTPException(status_code=400, detail="Sender verification is required")

        logger.info(
            f"Receiving federated message {payload.message_id} "
            f"from {payload.from_address} to user {payload.to_user_id}"
        )

        # Import here to avoid circular imports
        from tools.implementations.pager_tool import PagerTool
        from utils.user_context import clear_user_context, set_current_user_id

        # Set user context so PagerTool resolves the correct user via contextvar
        try:
            set_current_user_id(str(payload.to_user_id))
            pager = PagerTool()

            # Deliver the message
            result = pager.deliver_federated_message(
                from_address=payload.from_address,
                content=payload.content,
                priority=payload.priority,
                metadata=payload.metadata,
                external_message_id=payload.message_id
            )
        finally:
            clear_user_context()

        if result.get("success"):
            logger.info(
                f"Delivered federated message {payload.message_id} "
                f"to user {payload.to_user_id} pager {result.get('delivered_to')}"
            )
            return FederationDeliveryResponse(
                status="delivered",
                message_id=payload.message_id
            )
        else:
            error_msg = result.get("error", "Unknown delivery error")
            logger.warning(f"Failed to deliver {payload.message_id}: {error_msg}")

            # Return 400 for user-related issues (permanent)
            raise HTTPException(status_code=400, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error delivering federated message {payload.message_id}: {e}", exc_info=True)
        # Return 500 for server errors (Lattice will retry)
        raise HTTPException(status_code=500, detail="Internal delivery error")
