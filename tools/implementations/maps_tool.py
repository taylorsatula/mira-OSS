"""
Maps API integration tool.

This tool enables the bot to interact with Maps APIs to resolve
natural language location queries to coordinates, retrieve place details,
and perform geocoding operations.

Requires Maps API key in the config file.
"""

import logging
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field
from tools.repo import Tool
from tools.registry import registry

# Define configuration class for  MapsTool
class MapsToolConfig(BaseModel):
    """Configuration for the maps_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    timeout: int = Field(default=60, description="Timeout in seconds for Google Maps API requests")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    backoff_factor: float = Field(default=2.0, description="Backoff factor for retries")
    cache_timeout: int = Field(default=86400, description="Cache timeout in seconds (default: 24 hours)")

# Register with registry
registry.register("maps_tool", MapsToolConfig)


# --- Input Models ---

class GeocodeInput(BaseModel):
    """Input for geocode operation."""
    query: str = Field(..., min_length=1, description="Address, landmark name, or place description")


class ReverseGeocodeInput(BaseModel):
    """Input for reverse geocode operation."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")


class PlaceDetailsInput(BaseModel):
    """Input for place details operation."""
    place_id: str = Field(..., min_length=1, description="Google Places ID")


class PlacesNearbyInput(BaseModel):
    """Input for places nearby operation."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude of center point")
    lng: float = Field(..., ge=-180, le=180, description="Longitude of center point")
    radius: int = Field(default=1000, ge=1, le=50000, description="Search radius in meters")
    keyword: Optional[str] = Field(default=None, description="Keywords to search for")
    type: Optional[str] = Field(default=None, description="Place type filter (e.g., 'restaurant')")
    language: Optional[str] = Field(default=None, description="Language code for results")
    open_now: Optional[bool] = Field(default=None, description="Filter to only open places")


class FindPlaceInput(BaseModel):
    """Input for find place operation."""
    query: str = Field(..., min_length=1, description="Place name or description")
    fields: List[str] = Field(
        default=["place_id", "name", "formatted_address", "geometry", "types", "business_status", "rating"],
        description="Specific fields to request"
    )


class CalculateDistanceInput(BaseModel):
    """Input for calculate distance operation."""
    lat1: float = Field(..., ge=-90, le=90, description="First point latitude")
    lng1: float = Field(..., ge=-180, le=180, description="First point longitude")
    lat2: float = Field(..., ge=-90, le=90, description="Second point latitude")
    lng2: float = Field(..., ge=-180, le=180, description="Second point longitude")


# --- Tool Implementation ---

class MapsTool(Tool):
    """
    Tool for interacting with Maps APIs to resolve locations and places.

    Features:
    1. Geocoding:
       - Convert natural language queries to lat/long coordinates
       - Resolve place names to specific locations
       - Support for structured and unstructured address inputs

    2. Place Details:
       - Get detailed information about places
       - Retrieve business information, opening hours, ratings
       - Get contact information for businesses

    3. Reverse Geocoding:
       - Convert coordinates to formatted addresses
       - Get neighborhood, city, state information from coordinates
    """

    name = "maps_tool"
    
    tool_schema = {
        "name": "maps_tool",
        "description": "Provides comprehensive location intelligence and geographical services through Maps API integration. Use this tool for geocoding, place details, distance calculations, and location-based searches.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "geocode",
                            "reverse_geocode",
                            "place_details",
                            "places_nearby",
                            "find_place",
                            "calculate_distance"
                        ],
                        "description": "The Maps API operation to perform"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for geocode, find_place operations. Can be an address, landmark name, or place description"
                    },
                    "place_id": {
                        "type": "string",
                        "description": "Google Places ID for place_details operation"
                    },
                    "lat": {
                        "type": "number",
                        "description": "Latitude for reverse_geocode, places_nearby operations",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "lng": {
                        "type": "number",
                        "description": "Longitude for reverse_geocode, places_nearby operations",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "lat1": {
                        "type": "number",
                        "description": "First point latitude for calculate_distance operation",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "lng1": {
                        "type": "number",
                        "description": "First point longitude for calculate_distance operation",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "lat2": {
                        "type": "number",
                        "description": "Second point latitude for calculate_distance operation",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "lng2": {
                        "type": "number",
                        "description": "Second point longitude for calculate_distance operation",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Search radius in meters for places_nearby (default: 1000)",
                        "default": 1000
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of place to filter (e.g., 'restaurant', 'cafe', 'hospital') for places_nearby"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keywords to search for in places_nearby operation"
                    },
                    "open_now": {
                        "type": "boolean",
                        "description": "Filter to only show places currently open for places_nearby"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for results (e.g., 'en', 'fr', 'es')"
                    },
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Specific fields to request for find_place operation",
                        "default": ["place_id", "name", "formatted_address", "geometry", "types", "business_status", "rating"]
                    }
                },
                "required": ["operation"]
            }
        }

    description = "Location services: geocoding, place details, nearby search, and distance calculation"

    def __init__(self):
        """Initialize the Google Maps tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._client = None

    @property
    def client(self):
        """
        Get the Google Maps client, initializing it if needed.
        Lazy loading approach.

        Returns:
            Google Maps client instance

        Raises:
            ValueError: If Google Maps API key is not set or client initialization fails
        """
        if self._client is None:
            try:
                from googlemaps import Client

                # Get API key from config
                from config import config
                api_key = config.google_maps_api_key
                if not api_key:
                    self.logger.error("Google Maps API key not found in configuration for maps_tool")
                    raise ValueError("Google Maps API key not found in configuration.")

                # Create client with API key
                self.logger.info("Creating Google Maps client with API key")
                self._client = Client(key=api_key)
            except ImportError:
                self.logger.error("googlemaps library not installed for maps_tool")
                raise ValueError("googlemaps library not installed. Run: pip install googlemaps")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Maps client: {e}")
                raise ValueError(f"Failed to initialize Google Maps client: {e}")
        return self._client

        
    def _geocode(self, input: GeocodeInput) -> Dict[str, Any]:
        """Convert a natural language query to geographic coordinates."""
        try:
            results = self.client.geocode(address=input.query)
            processed_results = []

            for result in results:
                processed_result = {
                    "formatted_address": result.get("formatted_address", ""),
                    "place_id": result.get("place_id", ""),
                    "location": result.get("geometry", {}).get("location", {}),
                    "types": result.get("types", [])
                }
                processed_results.append(processed_result)

            return {"results": processed_results}
        except Exception as e:
            self.logger.error(f"Geocoding failed for '{input.query}': {e}")
            raise ValueError(f"Failed to geocode query: {e}")

    def _reverse_geocode(self, input: ReverseGeocodeInput) -> Dict[str, Any]:
        """Convert geographic coordinates to an address."""
        try:
            results = self.client.reverse_geocode((input.lat, input.lng))
            processed_results = []

            for result in results:
                processed_result = {
                    "formatted_address": result.get("formatted_address", ""),
                    "place_id": result.get("place_id", ""),
                    "types": result.get("types", [])
                }
                processed_results.append(processed_result)

            return {"results": processed_results}
        except Exception as e:
            self.logger.error(f"Reverse geocoding failed for ({input.lat}, {input.lng}): {e}")
            raise ValueError(f"Failed to reverse geocode coordinates: {e}")

    def _place_details(self, input: PlaceDetailsInput) -> Dict[str, Any]:
        """Get detailed information about a place."""
        try:
            result = self.client.place(place_id=input.place_id)

            if "result" not in result:
                raise ValueError(f"No details found for place ID: {input.place_id}")

            place = result["result"]
            details = {
                "name": place.get("name", ""),
                "formatted_address": place.get("formatted_address", ""),
                "formatted_phone_number": place.get("formatted_phone_number", ""),
                "international_phone_number": place.get("international_phone_number", ""),
                "website": place.get("website", ""),
                "url": place.get("url", ""),
                "rating": place.get("rating", 0),
                "types": place.get("types", []),
            }

            if "geometry" in place and "location" in place["geometry"]:
                details["location"] = place["geometry"]["location"]

            if "opening_hours" in place:
                details["opening_hours"] = place["opening_hours"]

            return details

        except Exception as e:
            self.logger.error(f"Place details failed for '{input.place_id}': {e}")
            raise ValueError(f"Failed to get place details: {e}")

    def _places_nearby(self, input: PlacesNearbyInput) -> Dict[str, Any]:
        """Find places near a specific location."""
        try:
            params = {
                "location": (input.lat, input.lng),
                "radius": input.radius
            }

            if input.keyword:
                params["keyword"] = input.keyword
            if input.type:
                params["type"] = input.type
            if input.language:
                params["language"] = input.language
            if input.open_now is not None:
                params["open_now"] = input.open_now

            results = self.client.places_nearby(**params)
            processed_results = []

            for place in results.get("results", []):
                processed_place = {
                    "name": place.get("name", ""),
                    "place_id": place.get("place_id", ""),
                    "vicinity": place.get("vicinity", ""),
                    "types": place.get("types", []),
                }

                if "geometry" in place and "location" in place["geometry"]:
                    processed_place["location"] = place["geometry"]["location"]

                if "rating" in place:
                    processed_place["rating"] = place["rating"]

                if "opening_hours" in place and "open_now" in place["opening_hours"]:
                    processed_place["open_now"] = place["opening_hours"]["open_now"]

                processed_results.append(processed_place)

            return {"results": processed_results}

        except Exception as e:
            self.logger.error(f"Places nearby search failed for ({input.lat}, {input.lng}): {e}")
            raise ValueError(f"Failed to search nearby places: {e}")

    def _find_place(self, input: FindPlaceInput) -> Dict[str, Any]:
        """Find a specific place using a text query."""
        try:
            params = {
                "input": input.query,
                "input_type": "textquery",
                "fields": input.fields
            }

            results = self.client.find_place(**params)
            processed_results = []

            for place in results.get("candidates", []):
                processed_place = {
                    "name": place.get("name", ""),
                    "place_id": place.get("place_id", ""),
                    "formatted_address": place.get("formatted_address", ""),
                    "types": place.get("types", []),
                }

                if "geometry" in place and "location" in place["geometry"]:
                    processed_place["location"] = place["geometry"]["location"]

                if "rating" in place:
                    processed_place["rating"] = place["rating"]

                processed_results.append(processed_place)

            return {"results": processed_results}

        except Exception as e:
            self.logger.error(f"Find place failed for '{input.query}': {e}")
            raise ValueError(f"Failed to find place: {e}")

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate the great circle distance between two points using the haversine formula."""
        import math

        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters
        return c * r

    def _calculate_distance(self, input: CalculateDistanceInput) -> Dict[str, Any]:
        """Calculate the distance between two geographic points."""
        distance = self._haversine_distance(input.lat1, input.lng1, input.lat2, input.lng2)
        return {
            "distance_meters": distance,
            "distance_kilometers": distance / 1000,
            "distance_miles": distance / 1609.34
        }

    def run(self, **params) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = params.pop("operation", None)
        if not operation:
            raise ValueError("Required parameter 'operation' not provided")

        if operation == "geocode":
            return self._geocode(GeocodeInput(**params))
        elif operation == "reverse_geocode":
            return self._reverse_geocode(ReverseGeocodeInput(**params))
        elif operation == "place_details":
            return self._place_details(PlaceDetailsInput(**params))
        elif operation == "places_nearby":
            return self._places_nearby(PlacesNearbyInput(**params))
        elif operation == "find_place":
            return self._find_place(FindPlaceInput(**params))
        elif operation == "calculate_distance":
            return self._calculate_distance(CalculateDistanceInput(**params))
        else:
            raise ValueError(
                f"Unknown operation: {operation}. Must be: geocode, reverse_geocode, "
                "place_details, places_nearby, find_place, or calculate_distance"
            )