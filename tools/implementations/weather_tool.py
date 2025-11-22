"""
Weather tool for retrieving forecast data and calculating heat stress indices.

This tool provides weather forecast information from OpenMeteo API and calculates
Wet Bulb Globe Temperature (WBGT) for heat stress assessment, which is particularly
useful for field technicians working in hot conditions.

Datetime handling follows the UTC-everywhere approach:
- All datetimes are stored in UTC internally
- Timezone-aware datetime objects are used consistently
- Conversion to local time happens only when displaying to users
- The utility functions from utils.timezone_utils are used consistently
"""

# Standard library imports
import logging
import json
import os
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta

# Third-party imports
from utils import http_client
from pydantic import BaseModel, Field

# Import timezone utilities for UTC-everywhere approach
from utils.timezone_utils import (
    utc_now,
    ensure_utc,
    convert_from_utc,
    format_datetime,
    parse_utc_time_string,
    parse_time_string,
)

# Local imports
from tools.repo import Tool
from tools.registry import registry


# -------------------- OAI SCHEMA --------------------


# -------------------- CONFIGURATION --------------------

class WeatherToolConfig(BaseModel):
    """
    Configuration for the weather_tool.
    
    Defines the parameters that control the weather tool's behavior,
    including API endpoints, caching, and default settings.
    """
    # Standard configuration parameter - all tools should include this
    enabled: bool = Field(
        default=False,
        description="Whether this tool is enabled by default"
    )
    
    # API configuration
    api_endpoint: str = Field(
        default="https://api.open-meteo.com/v1/forecast", 
        description="The OpenMeteo API endpoint"
    )
    
    # Caching configuration
    cache_enabled: bool = Field(
        default=True, 
        description="Whether to cache weather data"
    )
    cache_duration: int = Field(
        default=3600, 
        description="Cache duration in seconds (default: 1 hour)"
    )
    cache_directory: str = Field(
        default="data/tools/weather_tool/cache", 
        description="Directory to store cached weather data"
    )
    
    # Default parameters
    timezone: str = Field(
        default="auto", 
        description="Default timezone (auto or IANA time zone name)"
    )
    forecast_days: int = Field(
        default=7, 
        description="Number of forecast days to retrieve"
    )
    
    # Heat stress risk thresholds (NIOSH/OSHA guidelines - in Celsius)
    wbgt_risk_thresholds: Dict[str, float] = Field(
        default={
            "low": 0,
            "moderate": 25.0,
            "high": 27.8,
            "very_high": 30.0,
            "extreme": 32.2
        },
        description="WBGT thresholds for heat stress risk levels in Celsius"
    )

# Register with registry
registry.register("weather_tool", WeatherToolConfig)


# -------------------- CACHE MANAGER --------------------

class WeatherCache:
    """
    Manages caching of weather data to minimize API requests.
    """
    
    def __init__(self, cache_dir: str, cache_duration: int):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            cache_duration: Cache validity duration in seconds
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str) -> str:
        """
        Get the cache file path for a given key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Create a hash of the key for the filename to avoid invalid characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def is_valid(self, cache_path: str) -> bool:
        """
        Check if a cache file is valid and not expired.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(cache_path):
            return False
            
        # Check if cache is expired based on file modification time
        # Use utc_now timestamp for consistent timezone handling
        cache_mtime = os.path.getmtime(cache_path)
        cache_age = utc_now().timestamp() - cache_mtime
        return cache_age < self.cache_duration
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if available and valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not available
        """
        cache_path = self.get_cache_path(key)
        
        # Check if cache is valid
        if not self.is_valid(cache_path):
            return None
            
        # Read from cache
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache for {key}: {str(e)}")
            return None
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        cache_path = self.get_cache_path(key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to cache data for {key}: {str(e)}")


# -------------------- VALIDATION FUNCTIONS --------------------

class ValidationUtils:
    """
    Utility methods for validating weather tool parameters.
    """
    
    @staticmethod
    def validate_coordinates(latitude: Any, longitude: Any) -> tuple:
        """
        Validate latitude and longitude coordinates.
        
        Args:
            latitude: Latitude value to validate
            longitude: Longitude value to validate
            
        Returns:
            Tuple of validated (latitude, longitude)
            
        Raises:
            ValueError: If coordinates are invalid
        """
        errors = {}
        
        # Check if latitude is missing or invalid
        if latitude is None:
            errors["latitude"] = "Latitude is required"
        else:
            try:
                lat_float = float(latitude)
                if lat_float < -90 or lat_float > 90:
                    errors["latitude"] = f"Latitude must be between -90 and 90, got {lat_float}"
                latitude = lat_float
            except (ValueError, TypeError):
                errors["latitude"] = f"Latitude must be a valid number, got {type(latitude).__name__}: {latitude}"
        
        # Check if longitude is missing or invalid
        if longitude is None:
            errors["longitude"] = "Longitude is required"
        else:
            try:
                lon_float = float(longitude)
                if lon_float < -180 or lon_float > 180:
                    errors["longitude"] = f"Longitude must be between -180 and 180, got {lon_float}"
                longitude = lon_float
            except (ValueError, TypeError):
                errors["longitude"] = f"Longitude must be a valid number, got {type(longitude).__name__}: {longitude}"
        
        # If there are any errors, raise exception
        if errors:
            error_message = "; ".join(f"{key}: {value}" for key, value in errors.items())
            raise ValueError(error_message)
        
        return (latitude, longitude)
    
    @staticmethod
    def validate_date(date_str: Optional[str]) -> Optional[str]:
        """
        Validate a date string for forecasts.
        
        Args:
            date_str: Date string in ISO format (YYYY-MM-DD)
            
        Returns:
            Validated date string or None
            
        Raises:
            ValueError: If date is invalid
        """
        # Allow None value
        if date_str is None:
            return None
            
        # Check if parameter is a string
        if not isinstance(date_str, str):
            raise ValueError(
                f"Date must be a string in ISO format (YYYY-MM-DD)",
                {"date": date_str}
            )
            
        try:
            # Parse date in ISO format using our timezone utilities
            date_obj = parse_utc_time_string(date_str)

            # Ensure we're working with UTC time for consistency
            now_utc = utc_now()
            
            # Ensure date is not too far in the future
            max_days = 16  # OpenMeteo typically supports up to 16 days
            if date_obj.date() > now_utc.date() + timedelta(days=max_days):
                raise ValueError(
                    f"Date cannot be more than {max_days} days in the future",
                        {"date": date_str, "max_days": max_days}
                )
                
            # Return the validated date string
            return date_str
        except ValueError:
            raise ValueError(
                f"Invalid date format: '{date_str}'. Use ISO format (YYYY-MM-DD)",
                {"date": date_str}
            )
    
    @staticmethod
    def validate_parameters(parameters: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """
        Validate requested weather parameters.
        
        Args:
            parameters: List of parameters or comma-separated string
            
        Returns:
            List of validated parameters or None for all parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        # If None, return None (all parameters)
        if parameters is None:
            return None
        
        # If string, split by comma
        if isinstance(parameters, str):
            # Empty string means all parameters
            if not parameters.strip():
                return None
            parameters = [p.strip() for p in parameters.split(',')]
        
        # Validate it's a list
        if not isinstance(parameters, list):
            raise ValueError(
                f"Parameters must be a list or comma-separated string, got {type(parameters).__name__}",
                {"parameters": parameters}
            )
        
        # Check if list is empty
        if not parameters:
            return None
        
        # Validate each parameter is a string
        for i, param in enumerate(parameters):
            if not isinstance(param, str):
                raise ValueError(
                    f"Parameter at index {i} must be a string, got {type(param).__name__}",
                        {"parameter": param}
                )
        
        # Return list of validated parameters
        return parameters
    
    @staticmethod
    def validate_forecast_type(forecast_type: Optional[str]) -> str:
        """
        Validate the forecast type parameter.
        
        Args:
            forecast_type: Type of forecast to retrieve (hourly, daily)
            
        Returns:
            Validated forecast type
            
        Raises:
            ValueError: If forecast type is invalid
        """
        valid_types = ["hourly", "daily"]
        
        if not forecast_type:
            return "hourly"  # Default
        
        if not isinstance(forecast_type, str):
            raise ValueError(
                f"Forecast type must be a string, got {type(forecast_type).__name__}",
                {"forecast_type": forecast_type}
            )
        
        forecast_type = forecast_type.lower()
        
        if forecast_type not in valid_types:
            raise ValueError(
                f"Invalid forecast type: {forecast_type}. Must be one of: {', '.join(valid_types)}",
                {"forecast_type": forecast_type, "valid_types": valid_types}
            )
        
        return forecast_type

    @staticmethod
    def validate_forecast_days(forecast_days: Optional[Union[int, str]]) -> Optional[int]:
        """
        Validate the forecast days parameter.

        Args:
            forecast_days: Number of days to include in the forecast

        Returns:
            Validated forecast days as an integer or None

        Raises:
            ValueError: If the value is not within supported bounds
        """
        if forecast_days is None:
            return None

        try:
            days_int = int(forecast_days)
        except (TypeError, ValueError):
            raise ValueError(
                "Forecast days must be an integer between 1 and 16",
                {"forecast_days": forecast_days}
            )

        # OpenMeteo supports up to 16 days of forecast data
        if days_int < 1 or days_int > 16:
            raise ValueError(
                "Forecast days must be between 1 and 16",
                {"forecast_days": forecast_days}
            )

        return days_int
    
    @staticmethod
    def validate_location_input(latitude: Optional[Union[float, str]], longitude: Optional[Union[float, str]], location: Optional[str]) -> None:
        """
        Validate that either coordinates OR location is provided, but not both.
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
            location: Location name
            
        Raises:
            ValueError: If validation fails
        """
        has_coordinates = latitude is not None or longitude is not None
        has_location = location is not None and location.strip()
        
        if not has_coordinates and not has_location:
            raise ValueError(
                "Either coordinates (latitude, longitude) or location must be provided",
                {"latitude": latitude, "longitude": longitude, "location": location}
            )
        
        if has_coordinates and has_location:
            raise ValueError(
                "Provide either coordinates (latitude, longitude) OR location, not both",
                {"latitude": latitude, "longitude": longitude, "location": location}
            )
        
        # If coordinates provided, both must be present
        if has_coordinates and (latitude is None or longitude is None):
            raise ValueError(
                "Both latitude and longitude must be provided when using coordinates",
                {"latitude": latitude, "longitude": longitude}
            )


# -------------------- MAIN TOOL CLASS --------------------

class WeatherTool(Tool):
    """
    Tool for retrieving weather forecast data and calculating heat stress indices.
    
    This tool provides weather forecast information from OpenMeteo API and calculates
    Wet Bulb Globe Temperature (WBGT) for heat stress assessment, particularly
    useful for field technicians working outdoors in hot conditions.
    """
    
    name = "weather_tool"

    simple_description = "Get weather forecasts and heat stress analysis for any location."

    anthropic_schema = {
        "name": "weather_tool",
        "description": "Retrieves weather forecast data and calculates heat stress indices for specified locations. Use this tool to get weather forecasts, heat stress information, and related data for planning field work activities based on expected weather conditions.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["get_forecast", "get_heat_stress"],
                        "description": "The operation to perform"
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the location (-90 to 90)",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the location (-180 to 180)",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "location": {
                        "type": "string",
                        "description": "City name or location (e.g., 'New York', 'Paris, France'). Alternative to latitude/longitude."
                    },
                    "forecast_type": {
                        "type": "string",
                        "enum": ["hourly", "daily"],
                        "default": "hourly",
                        "description": "Type of forecast to retrieve (hourly or daily)"
                    },
                    "forecast_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 16,
                        "description": "Number of days to include in the forecast (1-16). Overrides the configured default of 7 days."
                    },
                    "date": {
                        "type": "string",
                        "description": "Specific date for forecast in ISO format (YYYY-MM-DD). If not provided, returns forecast from current date."
                    },
                    "parameters": {
                        "type": "string",
                        "description": "Comma-separated list of specific parameters to retrieve. If not provided, returns all available parameters."
                    }
                },
                "required": ["operation"]
            }
        }

    # Common hourly parameters available in the API
    _available_hourly_params = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
        "precipitation_probability", "precipitation", "rain", "showers", "snowfall",
        "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility",
        "evapotranspiration", "reference_evapotranspiration", "vapor_pressure_deficit",
        "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
        "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
        "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m",
        "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm",
        "soil_temperature_54cm", "soil_moisture_0_1cm", "soil_moisture_1_3cm",
        "soil_moisture_3_9cm", "soil_moisture_9_27cm", "soil_moisture_27_81cm",
        "shortwave_radiation", "direct_radiation", "direct_normal_irradiance",
        "diffuse_radiation", "global_tilted_irradiance", "terrestrial_radiation",
        "shortwave_radiation_instant", "direct_radiation_instant", "direct_normal_irradiance_instant",
        "diffuse_radiation_instant", "global_tilted_irradiance_instant", "terrestrial_radiation_instant",
        "is_day", "wet_bulb_temperature_2m", "sunshine_duration"
    ]
    
    # Common daily parameters available in the API
    _available_daily_params = [
        "weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
        "sunrise", "sunset", "daylight_duration", "sunshine_duration", "precipitation_sum",
        "rain_sum", "snowfall_sum", "precipitation_hours", "precipitation_probability_max",
        "precipitation_probability_min", "precipitation_probability_mean", "wind_speed_10m_max",
        "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "uv_index_max", "uv_index_clear_sky_max",
        "wind_speed_10m_mean"
    ]
    
    # Parameters required for WBGT calculation
    _wbgt_required_params = [
        "temperature_2m", "wet_bulb_temperature_2m", 
        "shortwave_radiation", "wind_speed_10m"
    ]

    def __init__(self):
        """Initialize the weather tool."""
        super().__init__()
        self.logger.info("WeatherTool initialized")
        
        # Defer user-context-dependent operations to first use
        self._cache_directory = None
        self._maps_client = None
    
    @property
    def cache_directory(self):
        """Lazy-load cache directory."""
        if self._cache_directory is None:
            self._cache_directory = self.make_dir("cache")
        return self._cache_directory
    
    @property
    def maps_client(self):
        """
        Get the Google Maps client, initializing it if needed.
        Lazy loading approach extracted from maps_tool pattern.

        Returns:
            Google Maps client instance

        Raises:
            ValueError: If Google Maps API key is not set or client initialization fails
        """
        if self._maps_client is None:
            try:
                from googlemaps import Client

                # Get API key from config
                from config import config
                api_key = config.google_maps_api_key
                if not api_key:
                    self.logger.error("Google Maps API key not found in configuration for weather_tool geocoding")
                    raise ValueError("Google Maps API key not found in configuration.")

                # Create client with API key
                self.logger.info("Creating Google Maps client for weather_tool geocoding")
                self._maps_client = Client(key=api_key)
            except ImportError:
                self.logger.error("googlemaps library not installed for weather_tool geocoding")
                raise ValueError("googlemaps library not installed. Run: pip install googlemaps")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Maps client for weather_tool: {e}")
                raise ValueError(f"Failed to initialize Google Maps client: {e}")
        return self._maps_client
    
    def _geocode_location(self, location: str) -> tuple:
        """
        Convert a location name to coordinates using Google Maps geocoding API.
        
        Args:
            location: Location name (e.g., "New York", "Paris, France")
            
        Returns:
            Tuple of (latitude, longitude)
            
        Raises:
            ValueError: If location cannot be found or geocoding fails
        """
        self.logger.info(f"Geocoding location using Google Maps: {location}")
        
        try:
            # Call Google Maps geocoding API (extracted from maps_tool pattern)
            results = self.maps_client.geocode(address=location.strip())
            
            if not results:
                raise ValueError(
                    f"Location '{location}' not found. Please check the spelling or try a more specific location (e.g., 'Paris, France')",
                    {"location": location}
                )
            
            # Get the first (best) result
            result = results[0]
            geometry = result.get("geometry", {})
            location_data = geometry.get("location", {})
            
            latitude = location_data.get("lat")
            longitude = location_data.get("lng")
            
            if latitude is None or longitude is None:
                raise ValueError(
                    f"Invalid geocoding response for location '{location}'",
                    {"location": location, "result": result}
                )
            
            # Log the resolved location info
            formatted_address = result.get("formatted_address", "")
            self.logger.info(f"Resolved '{location}' to {formatted_address} ({latitude}, {longitude})")
            
            return (float(latitude), float(longitude))
            
        except Exception as e:
            if isinstance(e, ValueError) and "Location" in str(e):
                # Re-raise our custom ValueError messages
                raise
            self.logger.error(f"Failed to geocode location '{location}': {e}")
            raise ValueError(f"Failed to geocode location '{location}': {str(e)}")
    
    def _resolve_coordinates(
        self, 
        latitude: Optional[Union[float, str]], 
        longitude: Optional[Union[float, str]], 
        location: Optional[str]
    ) -> tuple:
        """
        Resolve coordinates from either lat/lon parameters or location name.
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
            location: Location name
            
        Returns:
            Tuple of validated (latitude, longitude)
            
        Raises:
            ValueError: If input validation or geocoding fails
        """
        # Validate input parameters
        ValidationUtils.validate_location_input(latitude, longitude, location)
        
        # If coordinates provided, validate them
        if latitude is not None and longitude is not None:
            return ValidationUtils.validate_coordinates(latitude, longitude)
        
        # If location provided, geocode it
        if location:
            return self._geocode_location(location)
        
        # This should not happen due to validation, but just in case
        raise ValueError("Neither coordinates nor location provided")
    
    def run(
        self,
        operation: str,
        latitude: Optional[Union[float, str]] = None,
        longitude: Optional[Union[float, str]] = None,
        location: Optional[str] = None,
        forecast_type: Optional[str] = None,
        forecast_days: Optional[Union[int, str]] = None,
        date: Optional[str] = None,
        parameters: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute the weather tool with the specified operation.
        
        Args:
            operation: The operation to perform (get_forecast, get_heat_stress)
            latitude: Latitude of the location (-90 to 90). Required if location not provided.
            longitude: Longitude of the location (-180 to 180). Required if location not provided.
            location: City name or location (e.g., "New York", "Paris, France"). Alternative to lat/lon.
            forecast_type: Type of forecast to retrieve (hourly, daily)
            forecast_days: Number of days to include in the forecast (1-16)
            date: Specific date for forecast in ISO format (YYYY-MM-DD)
            parameters: Specific parameters to retrieve (comma-separated or list)
            
        Returns:
            Dictionary containing the operation results
            
        Raises:
            ValueError: If parameters are invalid or the operation fails
        """
        self.logger.info(f"Running weather tool with operation: {operation}")
        
        try:
            # Import config inside the method to avoid circular imports
            from config import config
            
            # Validate and resolve coordinates
            lat, lon = self._resolve_coordinates(latitude, longitude, location)
            forecast_type = ValidationUtils.validate_forecast_type(forecast_type)
            validated_forecast_days = ValidationUtils.validate_forecast_days(forecast_days)
            validated_date = ValidationUtils.validate_date(date)
            validated_params = ValidationUtils.validate_parameters(parameters)
            
            # Execute appropriate operation
            if operation == "get_forecast":
                result = self._get_forecast(
                    lat,
                    lon,
                    forecast_type,
                    date=validated_date,
                    parameters=validated_params,
                    forecast_days=validated_forecast_days
                )
            elif operation == "get_heat_stress":
                result = self._get_heat_stress(
                    lat,
                    lon,
                    forecast_type,
                    date=validated_date,
                    parameters=validated_params,
                    forecast_days=validated_forecast_days
                )
            else:
                raise ValueError(
                    f"Invalid operation: {operation}. Must be one of: get_forecast, get_heat_stress",
                        {"operation": operation}
                )
            
            return {
                "success": True,
                **result
            }
        
        except Exception as e:
            self.logger.error(f"Weather tool execution failed: {e}")
            raise
    
    def _get_weather_data(
        self,
        latitude: float,
        longitude: float,
        forecast_type: str,
        date: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get weather data from OpenMeteo API with caching.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            forecast_type: Type of forecast to retrieve (hourly, daily)
            date: Specific date for forecast
            parameters: Specific parameters to retrieve
            forecast_days: Number of days to include in the forecast (1-16)
            
        Returns:
            Dictionary with API response data
            
        Raises:
            RuntimeError: If API request fails
        """
        from config import config

        days_to_request = forecast_days or config.weather_tool.forecast_days

        # Create cache key
        cache_key = f"{latitude}_{longitude}_{forecast_type}_{days_to_request}d"
        if date:
            cache_key += f"_{date}"
        if parameters:
            cache_key += f"_{','.join(sorted(parameters))}"

        # Create cache if enabled
        if config.weather_tool.cache_enabled:
            cache = WeatherCache(str(self.cache_directory), config.weather_tool.cache_duration)
            cached_data = cache.get(cache_key)
            if cached_data:
                self.logger.info(f"Using cached weather data for: {cache_key}")
                return cached_data

        # Determine parameters to request
        hourly_params = []
        daily_params = []
        
        if parameters is None:
            # Default parameters if none specified
            if forecast_type == "hourly":
                hourly_params = self._available_hourly_params
            else:  # daily
                daily_params = self._available_daily_params
        else:
            # Requested parameters
            if forecast_type == "hourly":
                hourly_params = [p for p in parameters if p in self._available_hourly_params]
            else:  # daily
                daily_params = [p for p in parameters if p in self._available_daily_params]
        
        # Ensure we have parameters to request
        if forecast_type == "hourly" and not hourly_params:
            self.logger.warning("No valid hourly parameters specified, using defaults")
            hourly_params = ["temperature_2m", "precipitation_probability", "wind_speed_10m"]
        elif forecast_type == "daily" and not daily_params:
            self.logger.warning("No valid daily parameters specified, using defaults")
            daily_params = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
        
        # Build API request URL
        url = config.weather_tool.api_endpoint
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": config.weather_tool.timezone,
            "forecast_days": days_to_request
        }

        # Add parameters
        if hourly_params:
            params["hourly"] = ",".join(hourly_params)
        if daily_params:
            params["daily"] = ",".join(daily_params)
        
        # Filter to specific date if provided
        if date:
            params["start_date"] = date
            params["end_date"] = date
        
        # Make API request
        self.logger.info(f"Requesting weather data from OpenMeteo API: {url} with params: {params}")
        
        try:
            response = http_client.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache if enabled
            if config.weather_tool.cache_enabled:
                cache.set(cache_key, data)
            
            return data
        except http_client.RequestError as e:
            raise ConnectionError(f"Failed to fetch weather data: {str(e)}")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Invalid response from weather API: {str(e)}")
    
    def _get_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_type: str,
        date: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get weather forecast data from OpenMeteo API.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            forecast_type: Type of forecast to retrieve
            date: Specific date for forecast
            parameters: Specific parameters to retrieve
            forecast_days: Number of days to include in the forecast (1-16)
            
        Returns:
            Dictionary with weather forecast data
        """
        self.logger.info(f"Getting {forecast_type} forecast for coordinates: {latitude}, {longitude}")
        
        # Get weather data
        weather_data = self._get_weather_data(
            latitude,
            longitude,
            forecast_type,
            date=date,
            parameters=parameters,
            forecast_days=forecast_days
        )
        
        # Get the API timezone from the response
        api_timezone = weather_data.get("timezone", "UTC")
        
        # Process forecast data to ensure all timestamps are properly handled
        forecast_data = weather_data.get(forecast_type, {})
        
        # Process timestamps in time data if they exist in the response
        if "time" in forecast_data:
            times = forecast_data["time"]
            processed_times = []
            
            for time_str in times:
                # Parse the time string into a UTC datetime (API returns timezone-specific times)
                try:
                    if 'T' in time_str and ('+' in time_str or time_str.endswith('Z')):
                        dt = parse_utc_time_string(time_str)
                    else:
                        dt = ensure_utc(parse_time_string(time_str, api_timezone))
                    processed_times.append(dt.isoformat())
                except ValueError:
                    # If parsing fails, use the original string
                    processed_times.append(time_str)
            
            # Replace the original times with processed ones if successful
            if processed_times:
                forecast_data["time"] = processed_times
        
        # Return formatted response
        return {
            "location": {
                "latitude": weather_data.get("latitude", latitude),
                "longitude": weather_data.get("longitude", longitude),
                "elevation": weather_data.get("elevation"),
                "timezone": api_timezone
            },
            "forecast_type": forecast_type,
            "forecast": {
                # Include units if available
                f"{forecast_type}_units": weather_data.get(f"{forecast_type}_units", {}),
                
                # Include processed forecast data
                forecast_type: forecast_data
            }
        }
    
    def _get_heat_stress(
        self,
        latitude: float,
        longitude: float,
        forecast_type: str,
        date: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        forecast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get weather forecast and calculate heat stress indices.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            forecast_type: Type of forecast to retrieve
            date: Specific date for forecast
            parameters: Additional parameters to retrieve
            forecast_days: Number of days to include in the forecast (1-16)
            
        Returns:
            Dictionary with weather forecast and heat stress data
        """
        self.logger.info(f"Calculating heat stress for coordinates: {latitude}, {longitude}")
        
        # Only hourly forecasts support WBGT calculation
        if forecast_type != "hourly":
            self.logger.warning("Heat stress calculation requires hourly forecast, switching from daily")
            forecast_type = "hourly"
        
        # Ensure we request the parameters needed for WBGT calculation
        required_params = set(self._wbgt_required_params)
        requested_params = set(parameters or [])
        combined_params = list(required_params.union(requested_params))
        
        # Get weather data with required parameters
        weather_data = self._get_weather_data(
            latitude,
            longitude,
            forecast_type,
            date=date,
            parameters=combined_params,
            forecast_days=forecast_days
        )
        
        # Extract required data for WBGT calculation
        hourly_data = weather_data.get("hourly", {})
        if not all(param in hourly_data for param in self._wbgt_required_params):
            missing_params = [param for param in self._wbgt_required_params if param not in hourly_data]
            raise ValueError(f"Missing required parameters for WBGT calculation: {missing_params}")
        
        # Get the API timezone from the response
        api_timezone = weather_data.get("timezone", "UTC")
        
        # Process timestamps in time data - ensure they're properly formatted
        times = hourly_data.get("time", [])
        processed_times = []
        
        for time_str in times:
            # Parse the time string into a UTC datetime (API returns timezone-specific times)
            try:
                if 'T' in time_str and ('+' in time_str or time_str.endswith('Z')):
                    dt = parse_utc_time_string(time_str)
                else:
                    dt = ensure_utc(parse_time_string(time_str, api_timezone))
                processed_times.append(dt.isoformat())
            except ValueError:
                # If parsing fails, use the original string
                processed_times.append(time_str)
        
        # Replace the original times with processed ones if successful
        if processed_times:
            hourly_data["time"] = processed_times
        
        # Continue with WBGT calculation
        temp_2m = hourly_data.get("temperature_2m", [])
        wet_bulb_temp = hourly_data.get("wet_bulb_temperature_2m", [])
        radiation = hourly_data.get("shortwave_radiation", [])
        wind_speed = hourly_data.get("wind_speed_10m", [])

        time_series = hourly_data.get("time", [])
        series_lengths = [
            len(time_series),
            len(temp_2m),
            len(wet_bulb_temp),
            len(radiation),
            len(wind_speed),
        ]
        valid_length = min(series_lengths) if series_lengths else 0

        wbgt_values = []
        risk_levels = []

        for i in range(valid_length):
            # Calculate WBGT
            try:
                wbgt = self._calculate_wbgt(
                    wet_bulb_temp[i],
                    temp_2m[i],
                    radiation[i],
                    wind_speed[i]
                )
            except (TypeError, ValueError):
                continue
            wbgt_values.append(round(wbgt, 1))
            
            # Determine risk level
            risk_level = self._get_heat_stress_risk_level(wbgt)
            risk_levels.append(risk_level)
        
        # Add WBGT and risk levels to hourly data
        hourly_data["wbgt"] = wbgt_values
        hourly_data["heat_stress_risk"] = risk_levels
        
        # Add WBGT unit to units
        hourly_units = weather_data.get("hourly_units", {})
        hourly_units["wbgt"] = "°C"
        
        # Return formatted response
        return {
            "location": {
                "latitude": weather_data.get("latitude", latitude),
                "longitude": weather_data.get("longitude", longitude),
                "elevation": weather_data.get("elevation"),
                "timezone": api_timezone
            },
            "forecast_type": forecast_type,
            "forecast": {
                "hourly_units": hourly_units,
                "hourly": hourly_data
            }
        }
    
    def _calculate_wbgt(
        self,
        wet_bulb_temperature: float,
        temperature: float,
        shortwave_radiation: float,
        wind_speed: float
    ) -> float:
        """
        Calculate Wet Bulb Globe Temperature (WBGT) for heat stress assessment.
        
        WBGT = 0.7 × wet_bulb_temperature_2m + 
               0.3 × temperature_2m + 
               0.14 × (shortwave_radiation/1000) - 
               0.016 × wind_speed_10m
        
        Args:
            wet_bulb_temperature: Wet bulb temperature in °C
            temperature: Temperature in °C
            shortwave_radiation: Shortwave radiation in W/m²
            wind_speed: Wind speed in km/h
            
        Returns:
            WBGT value in °C
        """
        # Apply the WBGT formula
        wbgt = (
            0.7 * wet_bulb_temperature +
            0.3 * temperature +
            0.14 * (shortwave_radiation / 1000) -
            0.016 * wind_speed
        )
        
        return wbgt
    
    def _get_heat_stress_risk_level(self, wbgt: float) -> str:
        """
        Determine heat stress risk level based on WBGT value according to NIOSH/OSHA guidelines.
        
        Args:
            wbgt: WBGT value in °C
            
        Returns:
            Risk level (Low, Moderate, High, Very High, Extreme)
        """
        from config import config
        
        # Get thresholds from config
        thresholds = config.weather_tool.wbgt_risk_thresholds
        
        # Determine risk level based on thresholds
        if wbgt < thresholds["moderate"]:
            return "Low"
        elif wbgt < thresholds["high"]:
            return "Moderate"
        elif wbgt < thresholds["very_high"]:
            return "High"
        elif wbgt < thresholds["extreme"]:
            return "Very High"
        else:
            return "Extreme"
