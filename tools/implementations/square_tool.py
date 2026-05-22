"""
Square integration tool for customer management and appointment scheduling.

This tool connects to Square via their MCP (Model Context Protocol) server,
providing access to customers, appointments, locations, and services.

Each user provides their own Square developer app credentials (client_id, client_secret)
and authenticates via OAuth to connect their Square account.
"""

import asyncio
import json
import logging
from datetime import timedelta
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now, format_utc_iso, parse_utc_time_string
from utils.user_context import get_current_user_id, set_current_user_id
from utils.user_credentials import UserCredentialService

logger = logging.getLogger(__name__)

# MCP server configuration
SQUARE_MCP_URL = "https://mcp.squareup.com/sse"


# -------------------- CONFIGURATION --------------------

class SquareToolConfig(BaseModel):
    """
    Configuration for the square_tool.

    Users provide their own Square developer app credentials and connect
    via OAuth. Each user can connect their own Square account.
    """
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled"
    )
    client_id: str = Field(
        default="",
        description="Your Square Application ID (from Square Developer Dashboard)"
    )
    client_secret: str = Field(
        default="",
        description="Your Square Application Secret",
        json_schema_extra={"secret": True}
    )
    oauth_status: str = Field(
        default="disconnected",
        description="OAuth connection status (managed automatically)"
    )
    merchant_id: Optional[str] = Field(
        default=None,
        description="Connected Square Merchant ID (set after OAuth)"
    )
    default_location_id: Optional[str] = Field(
        default=None,
        description="Default Square location for bookings"
    )
    customer_search_limit: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Default limit for customer searches"
    )
    booking_fetch_limit: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Default limit for booking queries"
    )


registry.register("square_tool", SquareToolConfig)


# -------------------- MAIN TOOL CLASS --------------------

class SquareTool(Tool):
    """
    Square integration tool for customer management and appointment scheduling.

    Connects to Square via MCP server using OAuth authentication.
    Supports customer CRUD, appointment booking, and analytics queries.
    """

    name = "square_tool"

    simple_description = "Manage Square customers and appointments. Search, create, update customers. Book, reschedule, cancel appointments. Query analytics like inactive customers."

    anthropic_schema = {
        "name": "square_tool",
        "description": """Square business management: customers, appointments, and scheduling.

CUSTOMER OPERATIONS:
- search_customers: Find customers by name, email, phone, or custom filters
- get_customer: Get details by ID or fuzzy name match
- create_customer: Add new customer profile
- update_customer: Modify existing customer
- delete_customer: Remove customer record
- list_customers: Paginated customer list with filters
- customer_analytics: Queries like "inactive 30+ days", "visited this month"

APPOINTMENT OPERATIONS:
- search_availability: Find open booking slots
- create_booking: Book an appointment
- get_booking: Retrieve booking details
- update_booking: Modify booking (reschedule, change service)
- cancel_booking: Cancel with optional reason
- list_bookings: By date range, customer, or team member
- get_business_profile: Retrieve booking settings

SUPPORT OPERATIONS:
- list_locations: Get Square locations
- list_team_members: Get bookable staff
- list_services: Get available services""",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        # Customer operations
                        "search_customers", "get_customer", "create_customer",
                        "update_customer", "delete_customer", "list_customers",
                        "customer_analytics",
                        # Appointment operations
                        "search_availability", "create_booking", "get_booking",
                        "update_booking", "cancel_booking", "list_bookings",
                        "get_business_profile",
                        # Support operations
                        "list_locations", "list_team_members", "list_services"
                    ],
                    "description": "The operation to perform"
                },
                # Customer parameters
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID or name for lookup/update/delete"
                },
                "query": {
                    "type": "string",
                    "description": "Search term for customers (searches name, email, phone)"
                },
                "email": {
                    "type": "string",
                    "description": "Customer email address"
                },
                "phone": {
                    "type": "string",
                    "description": "Customer phone number"
                },
                "given_name": {
                    "type": "string",
                    "description": "Customer first name"
                },
                "family_name": {
                    "type": "string",
                    "description": "Customer last name"
                },
                "company_name": {
                    "type": "string",
                    "description": "Customer company name"
                },
                "note": {
                    "type": "string",
                    "description": "Note to attach to customer profile"
                },
                # Analytics parameters
                "analytics_query": {
                    "type": "string",
                    "enum": ["inactive_30_days", "visited_this_month", "new_this_week", "top_customers"],
                    "description": "Type of analytics query to run"
                },
                # Booking parameters
                "booking_id": {
                    "type": "string",
                    "description": "Booking ID for get/update/cancel"
                },
                "start_at": {
                    "type": "string",
                    "description": "Start datetime in ISO 8601 format"
                },
                "end_at": {
                    "type": "string",
                    "description": "End datetime in ISO 8601 format"
                },
                "service_id": {
                    "type": "string",
                    "description": "Service/catalog item ID for booking"
                },
                "team_member_id": {
                    "type": "string",
                    "description": "Team member ID for booking"
                },
                "location_id": {
                    "type": "string",
                    "description": "Location ID (uses default if not specified)"
                },
                "customer_note": {
                    "type": "string",
                    "description": "Note from customer for booking"
                },
                # Pagination
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor for next page"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Number of results to return"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute a Square operation."""
        # Capture user_id for thread context propagation in run_in_executor()
        user_id = get_current_user_id()
        # Check OAuth connection first (fail fast if not connected)
        credential_service = UserCredentialService()
        oauth_token = credential_service.get_credential("oauth_token", "square")

        if not oauth_token:
            return {
                "success": False,
                "error": "not_connected",
                "message": "Not connected to Square. Please configure the Square tool in Settings and complete OAuth authentication."
            }

        # Get client_id from tool config (needed by mcp_client to trigger OAuth)
        config = self._get_tool_config()
        client_id = config.get("client_id")

        if not client_id:
            return {
                "success": False,
                "error": "missing_config",
                "message": "Square Application ID not configured. Please add it in Settings."
            }

        # Run async operation using concurrent.futures for proper thread isolation
        # The MCP SDK's SSE client uses anyio task groups that require proper event loop management

        import concurrent.futures

        def run_in_executor():
            """Run the async operation in a dedicated thread with its own event loop."""
            # Re-establish user context in this thread (contextvars don't propagate)
            set_current_user_id(user_id)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._execute_operation(operation, client_id=client_id, **kwargs)
                )
            finally:
                # Properly shut down the loop including pending tasks
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    # Give cancelled tasks a chance to clean up
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                finally:
                    loop.close()

        # Use ThreadPoolExecutor to run in a separate thread
        # This gives the MCP SDK's background tasks proper isolation
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_executor)
            try:
                return future.result(timeout=60)  # 60 second timeout
            except concurrent.futures.TimeoutError:
                return {
                    "success": False,
                    "error": "timeout",
                    "message": "Square operation timed out after 60 seconds"
                }
            except Exception as e:
                self.logger.error(f"Square operation failed: {type(e).__name__}: {e}")
                return {
                    "success": False,
                    "error": "operation_failed",
                    "message": str(e)
                }

    async def _execute_operation(
        self,
        operation: str,
        client_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute operation with MCP session.

        OAuth token is automatically retrieved by mcp_client via UserCredentialService
        when oauth_client_id is provided.
        """
        from utils import mcp_client

        try:
            async with mcp_client.create_session(
                server_name="square",
                server_url=SQUARE_MCP_URL,
                oauth_client_id=client_id
            ) as session:
                # Route to operation handler
                handlers = {
                    # Customer operations
                    "search_customers": self._search_customers,
                    "get_customer": self._get_customer,
                    "create_customer": self._create_customer,
                    "update_customer": self._update_customer,
                    "delete_customer": self._delete_customer,
                    "list_customers": self._list_customers,
                    "customer_analytics": self._customer_analytics,
                    # Appointment operations
                    "search_availability": self._search_availability,
                    "create_booking": self._create_booking,
                    "get_booking": self._get_booking,
                    "update_booking": self._update_booking,
                    "cancel_booking": self._cancel_booking,
                    "list_bookings": self._list_bookings,
                    "get_business_profile": self._get_business_profile,
                    # Support operations
                    "list_locations": self._list_locations,
                    "list_team_members": self._list_team_members,
                    "list_services": self._list_services,
                }

                handler = handlers.get(operation)
                if not handler:
                    return {
                        "success": False,
                        "error": "unknown_operation",
                        "message": f"Unknown operation: {operation}. Valid operations: {list(handlers.keys())}"
                    }

                return await handler(session, **kwargs)
                # Note: create_session context manager handles disconnect on exit

        except Exception as e:
            error_str = str(e).lower()

            # Handle specific error types
            if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
                return {
                    "success": False,
                    "error": "authentication_failed",
                    "message": "Square OAuth token is invalid or expired. Please reconnect in Settings."
                }
            elif "429" in error_str or "rate" in error_str:
                return {
                    "success": False,
                    "error": "rate_limited",
                    "message": "Square API rate limit exceeded. Please wait before retrying."
                }
            elif "connection" in error_str or "timeout" in error_str:
                return {
                    "success": False,
                    "error": "connection_failed",
                    "message": "Unable to connect to Square. Check network connectivity."
                }
            else:
                raise

    async def _make_square_request(
        self,
        session,
        service: str,
        method: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a Square API request via MCP."""
        result = await session.call_tool("make_api_request", {
            "service": service,
            "method": method,
            "request": request
        })

        # Parse MCP result
        if hasattr(result, 'isError') and result.isError:
            error_text = result.content[0].text if result.content else "Unknown error"
            raise RuntimeError(f"Square API error: {error_text}")

        # Extract JSON response
        if hasattr(result, 'content') and result.content:
            return json.loads(result.content[0].text)

        return {}

    # -------------------- CUSTOMER OPERATIONS --------------------

    async def _search_customers(
        self,
        session,
        query: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Search customers by name, email, or phone."""
        config = self._get_tool_config()

        request: Dict[str, Any] = {
            "limit": limit or config.get("customer_search_limit", 50)
        }

        if cursor:
            request["cursor"] = cursor

        # Build query filters
        filters = []

        if email:
            filters.append({
                "email_address": {"exact": email}
            })

        if phone:
            filters.append({
                "phone_number": {"exact": phone}
            })

        # For text search, we need to use a different approach
        # Square doesn't have a full-text search, so we list and filter locally
        if query and not email and not phone:
            # Fall back to list + local filter
            return await self._search_customers_local(session, query, limit or 50)

        if filters:
            request["query"] = {
                "filter": {"and": filters} if len(filters) > 1 else filters[0]
            }

        response = await self._make_square_request(
            session, "customers", "search", request
        )

        customers = response.get("customers", [])

        return {
            "success": True,
            "customers": [self._format_customer(c) for c in customers],
            "count": len(customers),
            "cursor": response.get("cursor"),
            "has_more": response.get("cursor") is not None,
            "message": f"Found {len(customers)} customers"
        }

    async def _search_customers_local(
        self,
        session,
        query: str,
        limit: int
    ) -> Dict[str, Any]:
        """Search customers by text (local filtering)."""
        # Fetch customers and filter locally
        all_customers = []
        cursor = None
        query_lower = query.lower()

        # Paginate through results until we have enough matches
        while len(all_customers) < limit:
            request: Dict[str, Any] = {"limit": 100}
            if cursor:
                request["cursor"] = cursor

            response = await self._make_square_request(
                session, "customers", "list", request
            )

            for customer in response.get("customers", []):
                # Check if query matches name, email, or phone
                name = f"{customer.get('given_name', '')} {customer.get('family_name', '')}".strip().lower()
                email = (customer.get("email_address") or "").lower()
                phone = customer.get("phone_number") or ""
                company = (customer.get("company_name") or "").lower()

                if (query_lower in name or
                    query_lower in email or
                    query_lower in phone or
                    query_lower in company):
                    all_customers.append(customer)
                    if len(all_customers) >= limit:
                        break

            cursor = response.get("cursor")
            if not cursor:
                break

        return {
            "success": True,
            "customers": [self._format_customer(c) for c in all_customers[:limit]],
            "count": len(all_customers[:limit]),
            "query": query,
            "message": f"Found {len(all_customers[:limit])} customers matching '{query}'"
        }

    async def _get_customer(
        self,
        session,
        customer_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get customer by ID or fuzzy name match."""
        if not customer_id:
            raise ValueError("customer_id is required")

        # Try direct ID lookup first
        try:
            response = await self._make_square_request(
                session, "customers", "get", {"customer_id": customer_id}
            )
            customer = response.get("customer")
            if customer:
                return {
                    "success": True,
                    "customer": self._format_customer(customer),
                    "matched_by": "id"
                }
        except Exception:
            pass  # Fall through to fuzzy search

        # Fuzzy name search
        result = await self._search_customers_local(session, customer_id, 10)
        customers = result.get("customers", [])

        if not customers:
            return {
                "success": False,
                "message": f"Customer '{customer_id}' not found"
            }

        if len(customers) == 1:
            return {
                "success": True,
                "customer": customers[0],
                "matched_by": "name"
            }

        # Multiple matches - return ambiguity
        return {
            "success": False,
            "ambiguous": True,
            "matches": customers,
            "message": f"Multiple customers match '{customer_id}'. Please be more specific or use customer ID."
        }

    async def _create_customer(
        self,
        session,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        company_name: Optional[str] = None,
        note: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new customer."""
        # At least one identifier required
        if not any([given_name, family_name, email, phone, company_name]):
            raise ValueError("At least one of given_name, family_name, email, phone, or company_name is required")

        request: Dict[str, Any] = {}

        if given_name:
            request["given_name"] = given_name
        if family_name:
            request["family_name"] = family_name
        if email:
            request["email_address"] = email
        if phone:
            request["phone_number"] = phone
        if company_name:
            request["company_name"] = company_name
        if note:
            request["note"] = note

        response = await self._make_square_request(
            session, "customers", "create", request
        )

        customer = response.get("customer")
        if not customer:
            raise RuntimeError("Failed to create customer - no customer in response")

        name = f"{customer.get('given_name', '')} {customer.get('family_name', '')}".strip() or customer.get("company_name", "Customer")

        return {
            "success": True,
            "customer": self._format_customer(customer),
            "message": f"Created customer: {name}"
        }

    async def _update_customer(
        self,
        session,
        customer_id: str,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        company_name: Optional[str] = None,
        note: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update an existing customer."""
        if not customer_id:
            raise ValueError("customer_id is required")

        # First get the customer to ensure it exists
        get_result = await self._get_customer(session, customer_id)
        if not get_result.get("success"):
            return get_result

        actual_id = get_result["customer"]["id"]

        request: Dict[str, Any] = {"customer_id": actual_id}
        updates = {}

        if given_name is not None:
            updates["given_name"] = given_name
        if family_name is not None:
            updates["family_name"] = family_name
        if email is not None:
            updates["email_address"] = email
        if phone is not None:
            updates["phone_number"] = phone
        if company_name is not None:
            updates["company_name"] = company_name
        if note is not None:
            updates["note"] = note

        if not updates:
            return {
                "success": False,
                "message": "No fields to update"
            }

        request.update(updates)

        response = await self._make_square_request(
            session, "customers", "update", request
        )

        customer = response.get("customer")
        if not customer:
            raise RuntimeError("Failed to update customer")

        return {
            "success": True,
            "customer": self._format_customer(customer),
            "updated_fields": list(updates.keys()),
            "message": f"Updated customer: {list(updates.keys())}"
        }

    async def _delete_customer(
        self,
        session,
        customer_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Delete a customer."""
        if not customer_id:
            raise ValueError("customer_id is required")

        # First get the customer to ensure it exists
        get_result = await self._get_customer(session, customer_id)
        if not get_result.get("success"):
            return get_result

        actual_id = get_result["customer"]["id"]
        customer_name = get_result["customer"].get("name", actual_id)

        await self._make_square_request(
            session, "customers", "delete", {"customer_id": actual_id}
        )

        return {
            "success": True,
            "deleted_id": actual_id,
            "message": f"Deleted customer: {customer_name}"
        }

    async def _list_customers(
        self,
        session,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """List customers with pagination."""
        config = self._get_tool_config()

        request: Dict[str, Any] = {
            "limit": min(limit or config.get("customer_search_limit", 50), 100)
        }

        if cursor:
            request["cursor"] = cursor

        response = await self._make_square_request(
            session, "customers", "list", request
        )

        customers = response.get("customers", [])

        return {
            "success": True,
            "customers": [self._format_customer(c) for c in customers],
            "count": len(customers),
            "cursor": response.get("cursor"),
            "has_more": response.get("cursor") is not None,
            "message": f"Retrieved {len(customers)} customers"
        }

    async def _customer_analytics(
        self,
        session,
        analytics_query: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run analytics queries on customers."""
        if not analytics_query:
            raise ValueError("analytics_query is required")

        valid_queries = ["inactive_30_days", "visited_this_month", "new_this_week", "top_customers"]
        if analytics_query not in valid_queries:
            raise ValueError(f"Invalid analytics_query. Valid options: {valid_queries}")

        # Fetch all customers (paginated)
        all_customers = []
        cursor = None

        while True:
            request: Dict[str, Any] = {"limit": 100}
            if cursor:
                request["cursor"] = cursor

            response = await self._make_square_request(
                session, "customers", "list", request
            )

            all_customers.extend(response.get("customers", []))
            cursor = response.get("cursor")

            if not cursor or len(all_customers) >= 1000:  # Cap at 1000 for performance
                break

        now = utc_now()
        result_customers = []

        if analytics_query == "inactive_30_days":
            cutoff = now - timedelta(days=30)
            for customer in all_customers:
                updated_at = customer.get("updated_at")
                if updated_at:
                    try:
                        updated = parse_utc_time_string(updated_at)
                        if updated < cutoff:
                            result_customers.append(customer)
                    except Exception:
                        pass

        elif analytics_query == "visited_this_month":
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            for customer in all_customers:
                updated_at = customer.get("updated_at")
                if updated_at:
                    try:
                        updated = parse_utc_time_string(updated_at)
                        if updated >= month_start:
                            result_customers.append(customer)
                    except Exception:
                        pass

        elif analytics_query == "new_this_week":
            week_start = now - timedelta(days=7)
            for customer in all_customers:
                created_at = customer.get("created_at")
                if created_at:
                    try:
                        created = parse_utc_time_string(created_at)
                        if created >= week_start:
                            result_customers.append(customer)
                    except Exception:
                        pass

        elif analytics_query == "top_customers":
            # Sort by creation date (oldest = most loyal)
            sorted_customers = sorted(
                all_customers,
                key=lambda c: c.get("created_at", "9999"),
                reverse=False
            )
            result_customers = sorted_customers[:20]

        final_limit = limit or 50

        return {
            "success": True,
            "customers": [self._format_customer(c) for c in result_customers[:final_limit]],
            "count": len(result_customers[:final_limit]),
            "total_found": len(result_customers),
            "query_type": analytics_query,
            "message": f"Found {len(result_customers)} customers for '{analytics_query}'"
        }

    # -------------------- APPOINTMENT OPERATIONS --------------------

    async def _search_availability(
        self,
        session,
        start_at: str,
        end_at: Optional[str] = None,
        service_id: Optional[str] = None,
        team_member_id: Optional[str] = None,
        location_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Search for available booking slots."""
        if not start_at:
            raise ValueError("start_at is required")

        config = self._get_tool_config()

        # Default end time to 7 days from start
        if not end_at:
            start_dt = parse_utc_time_string(start_at)
            end_dt = start_dt + timedelta(days=7)
            end_at = format_utc_iso(end_dt)

        request: Dict[str, Any] = {
            "query": {
                "filter": {
                    "start_at_range": {
                        "start_at": start_at,
                        "end_at": end_at
                    }
                }
            }
        }

        if location_id or config.get("default_location_id"):
            request["query"]["filter"]["location_id"] = location_id or config.get("default_location_id")

        if service_id:
            request["query"]["filter"]["segment_filters"] = [{
                "service_variation_id": service_id
            }]

        if team_member_id:
            if "segment_filters" not in request["query"]["filter"]:
                request["query"]["filter"]["segment_filters"] = [{}]
            request["query"]["filter"]["segment_filters"][0]["team_member_id_filter"] = {
                "all": [team_member_id]
            }

        response = await self._make_square_request(
            session, "bookings", "searchAvailability", request
        )

        availabilities = response.get("availabilities", [])

        return {
            "success": True,
            "availabilities": availabilities,
            "count": len(availabilities),
            "message": f"Found {len(availabilities)} available slots"
        }

    async def _create_booking(
        self,
        session,
        start_at: str,
        customer_id: str,
        service_id: Optional[str] = None,
        team_member_id: Optional[str] = None,
        location_id: Optional[str] = None,
        customer_note: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new booking."""
        if not start_at:
            raise ValueError("start_at is required")
        if not customer_id:
            raise ValueError("customer_id is required")

        config = self._get_tool_config()

        # Resolve customer ID if it's a name
        get_result = await self._get_customer(session, customer_id)
        if not get_result.get("success"):
            return get_result

        actual_customer_id = get_result["customer"]["id"]

        request: Dict[str, Any] = {
            "booking": {
                "start_at": start_at,
                "customer_id": actual_customer_id,
                "location_id": location_id or config.get("default_location_id")
            }
        }

        if service_id or team_member_id:
            request["booking"]["appointment_segments"] = [{
                "duration_minutes": 60,  # Default duration
                "team_member_id": team_member_id,
                "service_variation_id": service_id
            }]

        if customer_note:
            request["booking"]["customer_note"] = customer_note

        response = await self._make_square_request(
            session, "bookings", "create", request
        )

        booking = response.get("booking")
        if not booking:
            raise RuntimeError("Failed to create booking")

        return {
            "success": True,
            "booking": self._format_booking(booking),
            "message": f"Created booking for {get_result['customer'].get('name', actual_customer_id)}"
        }

    async def _get_booking(
        self,
        session,
        booking_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get booking details."""
        if not booking_id:
            raise ValueError("booking_id is required")

        response = await self._make_square_request(
            session, "bookings", "get", {"booking_id": booking_id}
        )

        booking = response.get("booking")
        if not booking:
            return {
                "success": False,
                "message": f"Booking '{booking_id}' not found"
            }

        return {
            "success": True,
            "booking": self._format_booking(booking)
        }

    async def _update_booking(
        self,
        session,
        booking_id: str,
        start_at: Optional[str] = None,
        customer_note: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update an existing booking."""
        if not booking_id:
            raise ValueError("booking_id is required")

        # Get current booking
        get_result = await self._get_booking(session, booking_id)
        if not get_result.get("success"):
            return get_result

        request: Dict[str, Any] = {
            "booking_id": booking_id,
            "booking": {}
        }

        if start_at:
            request["booking"]["start_at"] = start_at
        if customer_note is not None:
            request["booking"]["customer_note"] = customer_note

        if not request["booking"]:
            return {
                "success": False,
                "message": "No fields to update"
            }

        response = await self._make_square_request(
            session, "bookings", "update", request
        )

        booking = response.get("booking")
        if not booking:
            raise RuntimeError("Failed to update booking")

        return {
            "success": True,
            "booking": self._format_booking(booking),
            "message": "Booking updated"
        }

    async def _cancel_booking(
        self,
        session,
        booking_id: str,
        booking_version: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Cancel a booking.

        Args:
            booking_id: The booking to cancel
            booking_version: Optional optimistic concurrency version. If not provided,
                            we fetch the current version first to avoid VERSION_MISMATCH errors.
        """
        if not booking_id:
            raise ValueError("booking_id is required")

        # If no version provided, fetch current booking to get its version
        if booking_version is None:
            current = await self._get_booking(session, booking_id)
            if current.get("booking"):
                booking_version = current["booking"].get("version")

        request: Dict[str, Any] = {
            "booking_id": booking_id
        }

        if booking_version is not None:
            request["booking_version"] = booking_version

        response = await self._make_square_request(
            session, "bookings", "cancel", request
        )

        booking = response.get("booking")

        return {
            "success": True,
            "booking": self._format_booking(booking) if booking else None,
            "message": f"Cancelled booking {booking_id}"
        }

    async def _list_bookings(
        self,
        session,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
        customer_id: Optional[str] = None,
        team_member_id: Optional[str] = None,
        location_id: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """List bookings with filters."""
        config = self._get_tool_config()

        request: Dict[str, Any] = {
            "limit": min(limit or config.get("booking_fetch_limit", 100), 100)
        }

        if cursor:
            request["cursor"] = cursor

        if location_id or config.get("default_location_id"):
            request["location_id"] = location_id or config.get("default_location_id")

        if start_at:
            request["start_at_min"] = start_at
        if end_at:
            request["start_at_max"] = end_at
        if team_member_id:
            request["team_member_id"] = team_member_id

        response = await self._make_square_request(
            session, "bookings", "list", request
        )

        bookings = response.get("bookings", [])

        # Filter by customer if specified
        if customer_id:
            # Resolve customer ID
            get_result = await self._get_customer(session, customer_id)
            if get_result.get("success"):
                actual_id = get_result["customer"]["id"]
                bookings = [b for b in bookings if b.get("customer_id") == actual_id]

        return {
            "success": True,
            "bookings": [self._format_booking(b) for b in bookings],
            "count": len(bookings),
            "cursor": response.get("cursor"),
            "has_more": response.get("cursor") is not None,
            "message": f"Retrieved {len(bookings)} bookings"
        }

    async def _get_business_profile(
        self,
        session,
        **kwargs
    ) -> Dict[str, Any]:
        """Get business booking profile."""
        response = await self._make_square_request(
            session, "bookings", "getBusinessBookingProfile", {}
        )

        profile = response.get("business_booking_profile")

        return {
            "success": True,
            "profile": profile,
            "seller_level_writes": profile.get("support_seller_level_writes", False) if profile else False,
            "message": "Retrieved business booking profile"
        }

    # -------------------- SUPPORT OPERATIONS --------------------

    async def _list_locations(
        self,
        session,
        **kwargs
    ) -> Dict[str, Any]:
        """List Square locations."""
        response = await self._make_square_request(
            session, "locations", "list", {}
        )

        locations = response.get("locations", [])

        return {
            "success": True,
            "locations": [{
                "id": loc.get("id"),
                "name": loc.get("name"),
                "address": loc.get("address"),
                "timezone": loc.get("timezone"),
                "status": loc.get("status")
            } for loc in locations],
            "count": len(locations),
            "message": f"Found {len(locations)} locations"
        }

    async def _list_team_members(
        self,
        session,
        location_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """List bookable team members."""
        config = self._get_tool_config()

        request: Dict[str, Any] = {}

        if location_id or config.get("default_location_id"):
            request["query"] = {
                "filter": {
                    "location_ids": [location_id or config.get("default_location_id")]
                }
            }

        response = await self._make_square_request(
            session, "team", "search", request
        )

        members = response.get("team_members", [])

        return {
            "success": True,
            "team_members": [{
                "id": m.get("id"),
                "given_name": m.get("given_name"),
                "family_name": m.get("family_name"),
                "email": m.get("email_address"),
                "phone": m.get("phone_number"),
                "status": m.get("status"),
                "is_owner": m.get("is_owner", False)
            } for m in members],
            "count": len(members),
            "message": f"Found {len(members)} team members"
        }

    async def _list_services(
        self,
        session,
        location_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """List available services from catalog."""
        response = await self._make_square_request(
            session, "catalog", "list", {
                "types": ["ITEM", "ITEM_VARIATION"]
            }
        )

        objects = response.get("objects", [])

        # Filter to service items (typically marked for appointments)
        services = []
        for obj in objects:
            if obj.get("type") == "ITEM":
                item_data = obj.get("item_data", {})
                if item_data.get("product_type") == "APPOINTMENTS_SERVICE":
                    services.append({
                        "id": obj.get("id"),
                        "name": item_data.get("name"),
                        "description": item_data.get("description"),
                        "variations": item_data.get("variations", [])
                    })

        return {
            "success": True,
            "services": services,
            "count": len(services),
            "message": f"Found {len(services)} bookable services"
        }

    # -------------------- HELPER METHODS --------------------

    def _get_tool_config(self) -> Dict[str, Any]:
        """Get user's tool configuration."""
        from utils.tool_config_store import load_user_tool_config

        return load_user_tool_config("square_tool", hydrate_secrets=True) or {}

    def _format_customer(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Format customer for response."""
        given = customer.get("given_name", "")
        family = customer.get("family_name", "")
        name = f"{given} {family}".strip() or customer.get("company_name", "")

        return {
            "id": customer.get("id"),
            "name": name,
            "given_name": given,
            "family_name": family,
            "email": customer.get("email_address"),
            "phone": customer.get("phone_number"),
            "company": customer.get("company_name"),
            "note": customer.get("note"),
            "created_at": customer.get("created_at"),
            "updated_at": customer.get("updated_at"),
            "version": customer.get("version")
        }

    def _format_booking(self, booking: Dict[str, Any]) -> Dict[str, Any]:
        """Format booking for response."""
        return {
            "id": booking.get("id"),
            "start_at": booking.get("start_at"),
            "customer_id": booking.get("customer_id"),
            "location_id": booking.get("location_id"),
            "status": booking.get("status"),
            "customer_note": booking.get("customer_note"),
            "seller_note": booking.get("seller_note"),
            "appointment_segments": booking.get("appointment_segments", []),
            "created_at": booking.get("created_at"),
            "updated_at": booking.get("updated_at"),
            "version": booking.get("version")
        }
