"""
Home Assistant smart home control via REST API.

Queries device states and controls devices on a locally hosted Home Assistant
instance. A local SQLite entity registry caches device topology (what exists,
names, domains) to enable fuzzy name resolution without discovery round trips.
State is always fetched live — only topology is cached.

Scope: state queries and control only. Device management stays in the HA UI.
"""

import json
import logging
import re
from datetime import timedelta
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils import http_client
from utils.timezone_utils import utc_now, format_utc_iso


# -------------------- CONFIGURATION --------------------

class HomeassistantToolConfig(BaseModel):
    """Configuration for the homeassistant_tool."""
    enabled: bool = Field(default=False, description="Whether this tool is enabled")
    ha_url: str = Field(default="", description="Home Assistant instance URL (e.g., http://192.168.1.100:8123)")
    request_timeout: int = Field(default=10, description="HTTP request timeout in seconds")


registry.register("homeassistant_tool", HomeassistantToolConfig)


# Entity ID format: domain.object_id (e.g., light.kitchen_main)
_ENTITY_ID_PATTERN = re.compile(r'^[a-z_]+\.[a-z0-9_]+$')

# 24-hour staleness threshold for entity registry GC
_REGISTRY_STALE_HOURS = 24


# -------------------- MAIN TOOL CLASS --------------------

class HomeassistantTool(Tool):
    """
    Controls and queries a locally hosted Home Assistant instance.

    Maintains a local SQLite entity registry for fuzzy name resolution,
    so the LLM can say "kitchen light" instead of "light.kitchen_main"
    and resolve it locally without a round trip to HA.
    """

    name = "homeassistant_tool"

    _parallel_safe_operations = frozenset({
        "get_state", "get_states", "find_entities", "list_services"
    })

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") in cls._parallel_safe_operations

    simple_description = "Query and control Home Assistant smart home devices. Get entity states, toggle lights/switches/fans, and call domain-specific services."

    tool_schema = {
        "name": "homeassistant_tool",
        "description": "Query and control Home Assistant smart home devices. Get entity states, toggle devices, call domain-specific services, and search the local device registry by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "get_state",
                        "get_states",
                        "find_entities",
                        "turn_on",
                        "turn_off",
                        "toggle",
                        "call_service",
                        "list_services"
                    ],
                    "description": "The Home Assistant operation to execute"
                },
                "entity_id": {
                    "type": "string",
                    "description": "Entity identifier: either a HA entity_id (light.kitchen_main) or a friendly name (Kitchen Light). Friendly names are auto-resolved from the local registry. Required for get_state, turn_on, turn_off, toggle. Optional for call_service."
                },
                "domain": {
                    "type": "string",
                    "description": "Entity domain (light, switch, fan, climate, sensor, cover, lock, media_player, etc.). Filters get_states and find_entities results. Specifies the service domain for call_service."
                },
                "service": {
                    "type": "string",
                    "description": "Service name for call_service (e.g., open_cover, set_temperature, media_play). Combined with domain to form the full service path."
                },
                "service_data": {
                    "type": "object",
                    "description": "Additional parameters passed to HA for turn_on or call_service. Contents vary by domain. Examples: {\"brightness\": 128} for lights, {\"temperature\": 72} for climate, {\"rgb_color\": [255, 0, 0]} for colored lights."
                },
                "query": {
                    "type": "string",
                    "description": "Search term for find_entities. Matches against friendly names in the local registry. No HTTP call — instant results."
                }
            },
            "required": ["operation"]
        }
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._config_loaded = False
        self._ha_url: str = ""
        self._token: str = ""
        self._timeout: int = 10

        from utils.user_context import has_user_context
        if has_user_context():
            self._ensure_registry_table()

    # -------------------- DEFERRED INIT --------------------

    def _ensure_registry_table(self):
        """Create the entity_registry table if it doesn't exist."""
        self.db.create_table('entity_registry', """
            entity_id TEXT PRIMARY KEY,
            domain TEXT NOT NULL,
            friendly_name TEXT,
            device_class TEXT,
            supported_features INTEGER DEFAULT 0,
            last_seen TEXT NOT NULL
        """)

    # -------------------- CONFIG & AUTH --------------------

    def _load_config(self):
        """Load HA URL and token from UserCredentialService. Fail-fast if missing."""
        if self._config_loaded:
            return

        from utils.user_credentials import UserCredentialService
        cred_service = UserCredentialService()

        config_json = cred_service.get_credential(
            credential_type="tool_config",
            service_name="homeassistant_tool"
        )
        if not config_json:
            raise ValueError(
                "Home Assistant not configured. Set up in Settings > Tools "
                "with your HA instance URL."
            )

        config = json.loads(config_json)
        self._ha_url = config.get("ha_url", "").rstrip("/")
        self._timeout = config.get("request_timeout", 10)

        if not self._ha_url:
            raise ValueError("Home Assistant URL is empty in tool configuration.")

        token = cred_service.get_credential(
            credential_type="api_key",
            service_name="home_assistant"
        )
        if not token:
            raise ValueError(
                "Home Assistant access token not found. Add a Long-Lived Access Token "
                "in Settings > API Credentials with service name 'home_assistant'."
            )
        self._token = token
        self._config_loaded = True

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        }

    def _build_url(self, path: str) -> str:
        return f"{self._ha_url}{path}"

    # -------------------- HTTP --------------------

    def _ha_request(
        self, method: str, path: str, json_body: dict = None
    ) -> Optional[Any]:
        """Central HTTP helper with HA-specific error translation.

        Returns parsed JSON on success. Returns None on 404 (caller handles).
        Raises on all other errors.
        """
        url = self._build_url(path)
        headers = self._get_headers()

        try:
            if method == "GET":
                response = http_client.get(url, headers=headers, timeout=self._timeout)
            elif method == "POST":
                response = http_client.post(
                    url, headers=headers, json=json_body or {}, timeout=self._timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text

        except http_client.ConnectError:
            raise ConnectionError(
                f"Cannot reach Home Assistant at {self._ha_url}. "
                "Verify the URL and network connectivity."
            )
        except http_client.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Home Assistant authentication failed. "
                    "Check your Long-Lived Access Token."
                )
            if e.response.status_code == 404:
                return None
            raise

    # -------------------- ENTITY REGISTRY --------------------

    def _update_registry(self, states: List[Dict[str, Any]]):
        """Upsert entity_registry from a GET /api/states response, then GC stale entries."""
        self._ensure_registry_table()
        now = format_utc_iso(utc_now())

        for state_obj in states:
            entity_id = state_obj.get("entity_id", "")
            if not entity_id:
                continue

            domain = entity_id.split(".")[0] if "." in entity_id else ""
            attrs = state_obj.get("attributes", {})

            existing = self.db.select(
                'entity_registry', 'entity_id = :eid', {'eid': entity_id}
            )
            row = {
                'entity_id': entity_id,
                'domain': domain,
                'friendly_name': attrs.get("friendly_name", ""),
                'device_class': attrs.get("device_class", ""),
                'supported_features': attrs.get("supported_features", 0),
                'last_seen': now,
            }

            if existing:
                self.db.update(
                    'entity_registry',
                    {k: v for k, v in row.items() if k != 'entity_id'},
                    'entity_id = :eid',
                    {'eid': entity_id}
                )
            else:
                self.db.insert('entity_registry', row)

        # GC: prune entities not seen in the last 24 hours
        cutoff = format_utc_iso(utc_now() - timedelta(hours=_REGISTRY_STALE_HOURS))
        self.db.delete('entity_registry', 'last_seen < :cutoff', {'cutoff': cutoff})

    def _resolve_entity(self, identifier: str) -> Dict[str, Any]:
        """Resolve an identifier to an entity_id.

        Accepts either a HA entity_id (light.kitchen) or a friendly name (Kitchen Light).

        Returns:
            {"entity_id": "light.kitchen", "matched_by": "direct|exact|partial"}
            {"ambiguous": True, "matches": [...]}
            {"error": "message"}
        """
        if not identifier or not identifier.strip():
            return {"error": "entity_id is required"}

        identifier = identifier.strip()

        # 1) Looks like a valid entity_id format → use directly
        if _ENTITY_ID_PATTERN.match(identifier):
            return {"entity_id": identifier, "matched_by": "direct"}

        # 2) Fuzzy search against the local registry
        self._ensure_registry_table()
        all_entities = self.db.select('entity_registry')
        ident_lower = identifier.lower()

        # 2a) Case-insensitive exact match on friendly_name
        exact = [
            e for e in all_entities
            if (e.get('friendly_name') or '').strip().lower() == ident_lower
        ]
        if len(exact) == 1:
            return {"entity_id": exact[0]['entity_id'], "matched_by": "exact"}
        if len(exact) > 1:
            return {
                "ambiguous": True,
                "matches": [
                    {"entity_id": e['entity_id'], "friendly_name": e.get('friendly_name')}
                    for e in exact[:10]
                ]
            }

        # 2b) Starts-with match
        starts = [
            e for e in all_entities
            if (e.get('friendly_name') or '').strip().lower().startswith(ident_lower)
        ]
        if len(starts) == 1:
            return {"entity_id": starts[0]['entity_id'], "matched_by": "partial"}
        if len(starts) > 1:
            return {
                "ambiguous": True,
                "matches": [
                    {"entity_id": e['entity_id'], "friendly_name": e.get('friendly_name')}
                    for e in starts[:10]
                ]
            }

        # 2c) Contains match
        contains = [
            e for e in all_entities
            if ident_lower in (e.get('friendly_name') or '').strip().lower()
        ]
        if len(contains) == 1:
            return {"entity_id": contains[0]['entity_id'], "matched_by": "partial"}
        if len(contains) > 1:
            return {
                "ambiguous": True,
                "matches": [
                    {"entity_id": e['entity_id'], "friendly_name": e.get('friendly_name')}
                    for e in contains[:10]
                ]
            }

        return {"error": f"No entity matching '{identifier}' found in registry. "
                         "Try get_states to refresh the registry, or use the exact entity_id."}

    def _require_entity(self, identifier: str) -> str:
        """Resolve identifier to entity_id or raise with a helpful message.

        Returns the resolved entity_id string.
        Raises ValueError on ambiguity or no match.
        """
        result = self._resolve_entity(identifier)

        if "entity_id" in result:
            return result["entity_id"]

        if result.get("ambiguous"):
            matches_str = ", ".join(
                f"{m['entity_id']} ({m.get('friendly_name', '')})"
                for m in result["matches"]
            )
            raise ValueError(
                f"Multiple entities match '{identifier}': {matches_str}. "
                "Use the exact entity_id."
            )

        raise ValueError(result.get("error", f"Cannot resolve '{identifier}'"))

    # -------------------- VALIDATE CONFIG --------------------

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test HA connectivity and populate the initial entity registry."""
        ha_url = config.get("ha_url", "").rstrip("/")
        if not ha_url:
            raise ValueError("Home Assistant URL is required")

        from utils.user_credentials import UserCredentialService
        cred_service = UserCredentialService()
        token = cred_service.get_credential("api_key", "home_assistant")
        if not token:
            raise ValueError(
                "Home Assistant access token not configured. Add a Long-Lived "
                "Access Token in Settings > API Credentials with service name "
                "'home_assistant'."
            )

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        timeout = config.get("request_timeout", 10)

        # 1) Connectivity check
        try:
            response = http_client.get(f"{ha_url}/api/", headers=headers, timeout=timeout)
            response.raise_for_status()
        except http_client.ConnectError:
            raise ValueError(f"Cannot connect to Home Assistant at {ha_url}. Check URL and network.")
        except http_client.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Authentication failed. Check your Long-Lived Access Token.")
            raise ValueError(f"Home Assistant API error: {e.response.status_code}")

        # 2) Get HA config for version info
        ha_version = None
        location_name = None
        try:
            config_resp = http_client.get(
                f"{ha_url}/api/config", headers=headers, timeout=timeout
            )
            config_resp.raise_for_status()
            ha_config = config_resp.json()
            ha_version = ha_config.get("version")
            location_name = ha_config.get("location_name")
        except Exception:
            pass  # Discovery is best-effort — connectivity already confirmed

        # 3) Get all states for entity count + initial registry population
        try:
            states_resp = http_client.get(
                f"{ha_url}/api/states", headers=headers, timeout=timeout
            )
            states_resp.raise_for_status()
            states = states_resp.json()

            domain_counts: Dict[str, int] = {}
            for state_obj in states:
                domain = state_obj.get("entity_id", "").split(".")[0]
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

            # Populate the registry for the current user
            try:
                tool_instance = cls()
                tool_instance._ensure_registry_table()
                tool_instance._update_registry(states)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Registry population during validation failed: {e}")

            return {
                "ha_version": ha_version,
                "location_name": location_name,
                "entity_count": len(states),
                "domains": domain_counts
            }
        except Exception:
            return {"ha_version": ha_version, "entity_count": "unknown"}

    # -------------------- RUN DISPATCH --------------------

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        self._load_config()
        self._ensure_registry_table()

        try:
            if operation == "get_state":
                return self._get_state(**kwargs)
            elif operation == "get_states":
                return self._get_states(**kwargs)
            elif operation == "find_entities":
                return self._find_entities(**kwargs)
            elif operation == "turn_on":
                return self._turn_on(**kwargs)
            elif operation == "turn_off":
                return self._turn_off(**kwargs)
            elif operation == "toggle":
                return self._toggle(**kwargs)
            elif operation == "call_service":
                return self._call_service(**kwargs)
            elif operation == "list_services":
                return self._list_services(**kwargs)
            else:
                raise ValueError(
                    f"Unknown operation: {operation}. Valid: get_state, get_states, "
                    "find_entities, turn_on, turn_off, toggle, call_service, list_services"
                )
        except (ValueError, ConnectionError) as e:
            self.logger.error(f"homeassistant_tool {operation} failed: {e}")
            return {"success": False, "message": str(e)}

    # -------------------- READ OPERATIONS --------------------

    def _get_state(self, entity_id: str = "", **kwargs) -> Dict[str, Any]:
        if not entity_id:
            raise ValueError("entity_id is required for get_state")

        resolved_id = self._require_entity(entity_id)
        result = self._ha_request("GET", f"/api/states/{resolved_id}")

        if result is None:
            return {
                "success": False,
                "message": f"Entity '{resolved_id}' not found in Home Assistant"
            }

        return {
            "success": True,
            "entity_id": result.get("entity_id"),
            "state": result.get("state"),
            "attributes": result.get("attributes", {}),
            "last_changed": result.get("last_changed"),
            "last_updated": result.get("last_updated")
        }

    def _get_states(self, domain: str = "", **kwargs) -> Dict[str, Any]:
        result = self._ha_request("GET", "/api/states")
        if result is None:
            return {"success": False, "message": "Failed to retrieve states from Home Assistant"}

        # Refresh the entity registry from this response
        self._update_registry(result)

        if domain:
            # Filtered by domain: return full state objects with attributes
            filtered = [
                s for s in result
                if s.get("entity_id", "").startswith(f"{domain}.")
            ]
            return {
                "success": True,
                "domain": domain,
                "count": len(filtered),
                "entities": [
                    {
                        "entity_id": s["entity_id"],
                        "state": s.get("state"),
                        "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
                        "attributes": s.get("attributes", {}),
                        "last_changed": s.get("last_changed"),
                    }
                    for s in filtered
                ]
            }

        # No domain filter: summary only (entity_id + friendly_name + state)
        return {
            "success": True,
            "count": len(result),
            "entities": [
                {
                    "entity_id": s["entity_id"],
                    "state": s.get("state"),
                    "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
                }
                for s in result
            ]
        }

    def _find_entities(
        self, query: str = "", domain: str = "", **kwargs
    ) -> Dict[str, Any]:
        """Search the local entity registry. Zero HTTP calls."""
        all_entities = self.db.select('entity_registry')

        if domain:
            all_entities = [e for e in all_entities if e.get('domain') == domain]

        if query:
            query_lower = query.lower()
            all_entities = [
                e for e in all_entities
                if query_lower in (e.get('friendly_name') or '').lower()
                or query_lower in (e.get('entity_id') or '').lower()
            ]

        return {
            "success": True,
            "count": len(all_entities),
            "entities": [
                {
                    "entity_id": e['entity_id'],
                    "domain": e.get('domain'),
                    "friendly_name": e.get('friendly_name'),
                    "device_class": e.get('device_class'),
                }
                for e in all_entities
            ]
        }

    def _list_services(self, domain: str = "", **kwargs) -> Dict[str, Any]:
        result = self._ha_request("GET", "/api/services")
        if result is None:
            return {"success": False, "message": "Failed to retrieve services from Home Assistant"}

        if domain:
            result = [s for s in result if s.get("domain") == domain]

        return {
            "success": True,
            "count": len(result),
            "services": result
        }

    # -------------------- CONTROL OPERATIONS --------------------

    def _turn_on(
        self, entity_id: str = "", service_data: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        if not entity_id:
            raise ValueError("entity_id is required for turn_on")

        resolved_id = self._require_entity(entity_id)
        body = {"entity_id": resolved_id}
        if service_data:
            body.update(service_data)

        result = self._ha_request("POST", "/api/services/homeassistant/turn_on", body)
        return {
            "success": True,
            "message": f"Turned on {resolved_id}",
            "changed_states": self._summarize_changed_states(result)
        }

    def _turn_off(self, entity_id: str = "", **kwargs) -> Dict[str, Any]:
        if not entity_id:
            raise ValueError("entity_id is required for turn_off")

        resolved_id = self._require_entity(entity_id)
        result = self._ha_request(
            "POST", "/api/services/homeassistant/turn_off",
            {"entity_id": resolved_id}
        )
        return {
            "success": True,
            "message": f"Turned off {resolved_id}",
            "changed_states": self._summarize_changed_states(result)
        }

    def _toggle(self, entity_id: str = "", **kwargs) -> Dict[str, Any]:
        if not entity_id:
            raise ValueError("entity_id is required for toggle")

        resolved_id = self._require_entity(entity_id)
        result = self._ha_request(
            "POST", "/api/services/homeassistant/toggle",
            {"entity_id": resolved_id}
        )
        return {
            "success": True,
            "message": f"Toggled {resolved_id}",
            "changed_states": self._summarize_changed_states(result)
        }

    def _call_service(
        self,
        domain: str = "",
        service: str = "",
        entity_id: str = "",
        service_data: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not domain or not service:
            raise ValueError("domain and service are required for call_service")

        body = {}
        if entity_id:
            resolved_id = self._require_entity(entity_id)
            body["entity_id"] = resolved_id

        if service_data:
            body.update(service_data)

        result = self._ha_request(
            "POST", f"/api/services/{domain}/{service}?return_response", body
        )

        # With ?return_response, HA returns {changed_states, service_response}
        # Without response data, it returns just the changed states array
        if isinstance(result, dict) and "service_response" in result:
            return {
                "success": True,
                "message": f"Called {domain}.{service}",
                "changed_states": self._summarize_changed_states(
                    result.get("changed_states", [])
                ),
                "service_response": result.get("service_response")
            }

        return {
            "success": True,
            "message": f"Called {domain}.{service}",
            "changed_states": self._summarize_changed_states(result)
        }

    # -------------------- HELPERS --------------------

    def _summarize_changed_states(self, states: Any) -> List[Dict[str, str]]:
        """Extract entity_id + new state from a service call response."""
        if not isinstance(states, list):
            return []
        return [
            {
                "entity_id": s.get("entity_id", ""),
                "state": s.get("state", ""),
            }
            for s in states
            if isinstance(s, dict)
        ]
