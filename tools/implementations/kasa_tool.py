"""
Tool for controlling TP-Link Kasa smart home devices.

This tool provides an interface to discover and control Kasa smart home devices
on the local network including plugs, bulbs, switches, light strips, and multi-outlet
power strips.

Datetime handling follows the UTC-everywhere approach:
- All datetimes are stored in UTC internally
- Timezone-aware datetime objects are used consistently
- Conversion to local time happens only when displaying to users
- The utility functions from utils.timezone_utils are used consistently
"""

# Standard library imports
import asyncio
import json
import logging
import os
import threading
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime, timedelta

# Third-party imports
import aiohttp
from pydantic import BaseModel, Field

# Import timezone utilities for UTC-everywhere approach
from utils.timezone_utils import utc_now, ensure_utc

# Local imports
from tools.repo import Tool
from tools.registry import registry


# -------------------- CONFIGURATION --------------------

class KasaToolConfig(BaseModel):
    """
    Configuration for the kasa_tool.

    Defines the parameters that control the Kasa tool's behavior,
    including device discovery, caching, and authentication.
    """
    # Standard configuration parameter - all tools should include this
    enabled: bool = Field(
        default=False,
        description="Whether this tool is enabled by default"
    )

    # Caching configuration
    cache_enabled: bool = Field(
        default=True,
        description="Whether to cache device information"
    )
    cache_duration: int = Field(
        default=3600,
        description="Cache duration in seconds (default: 1 hour)"
    )
    cache_directory: str = Field(
        default="data/tools/kasa_tool/cache",
        description="Directory to store cached device data"
    )

    # Device mapping configuration
    device_mapping_enabled: bool = Field(
        default=True,
        description="Whether to use device mapping file for device identification"
    )
    device_mapping_path: str = Field(
        default="data/tools/kasa_tool/device_mapping.json",
        description="Path to the device mapping file"
    )

    # Discovery configuration
    discovery_timeout: int = Field(
        default=5,
        description="Timeout in seconds for device discovery"
    )
    discovery_target: str = Field(
        default="255.255.255.255",
        description="Default target for device discovery"
    )
    attempt_discovery_when_not_found: bool = Field(
        default=True,
        description="Whether to attempt discovery when a device is not found in cache"
    )

    # Authentication
    default_username: str = Field(
        default="",
        description="Default username for devices requiring authentication"
    )
    default_password: str = Field(
        default="",
        description="Default password for devices requiring authentication"
    )

    # Operation settings
    verify_changes: bool = Field(
        default=True,
        description="Whether to verify changes after performing operations"
    )
    verification_attempts: int = Field(
        default=3,
        description="Number of attempts to verify changes"
    )
    verification_delay: float = Field(
        default=0.5,
        description="Delay in seconds between verification attempts"
    )

# Register with registry
registry.register("kasa_tool", KasaToolConfig)


# -------------------- CACHE MANAGER --------------------

class DeviceCache:
    """
    Manages caching of discovered devices to minimize network operations.
    Uses a single devices.json file to store all device information.
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
        self.devices_file = os.path.join(cache_dir, "devices.json")
        self.cache_lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize devices cache file if it doesn't exist
        if not os.path.exists(self.devices_file):
            self._save_cache({})
    
    def is_valid(self) -> bool:
        """
        Check if the cache file is valid and not expired.
        
        Returns:
            True if cache is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(self.devices_file):
            return False
            
        # Check if cache is expired based on file modification time
        # Use utc_now timestamp for consistent timezone handling
        cache_mtime = os.path.getmtime(self.devices_file)
        cache_age = utc_now().timestamp() - cache_mtime
        return cache_age < self.cache_duration
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get device data from cache if available and valid.
        
        Args:
            key: Device identifier (host or alias)
            
        Returns:
            Cached device data or None if not available
        """
        # Check if cache is valid
        if not self.is_valid():
            return None
            
        # Read all devices from cache
        devices = self._load_cache()
        
        # Return the specific device if found
        return devices.get(key)
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save device data to cache.
        
        Args:
            key: Device identifier (host or alias)
            data: Device data to cache
        """
        with self.cache_lock:
            # Load current devices cache
            devices = self._load_cache()
            
            # Update device data
            devices[key] = data
            
            # Save the updated cache
            self._save_cache(devices)
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all devices from cache file.
        
        Returns:
            Dictionary of all cached devices
        """
        try:
            with open(self.devices_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Failed to load devices cache: {str(e)}")
            return {}
    
    def _save_cache(self, devices: Dict[str, Dict[str, Any]]) -> None:
        """
        Save all devices to cache file.
        
        Args:
            devices: Dictionary of all devices to cache
        """
        try:
            with open(self.devices_file, 'w') as f:
                json.dump(devices, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save devices cache: {str(e)}")


# -------------------- MAIN TOOL CLASS --------------------

class KasaTool(Tool):
    """
    Tool for controlling TP-Link Kasa smart home devices.

    This tool provides functionality to discover and control Kasa smart home devices
    on your local network including plugs, switches, bulbs, light strips, and multi-outlet
    power strips.
    """


    name = "kasa_tool"
    anthropic_schema = {
        "name": "kasa_tool",
        "description": "Discovers and controls TP-Link Kasa smart home devices including plugs, bulbs, switches, and power strips.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "discover_devices",
                            "get_device_info",
                            "power_control",
                            "set_brightness",
                            "set_color",
                            "set_color_temp",
                            "get_energy_usage",
                            "set_device_alias",
                            "get_child_devices",
                            "control_child_device"
                        ],
                        "description": "Kasa device operation to execute"
                    },
                    "target": {
                        "type": "string",
                        "description": "Subnet, broadcast address, or IP to probe when discovering devices"
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Discovery timeout in seconds"
                    },
                    "username": {
                        "type": "string",
                        "description": "Device login username when authentication is required"
                    },
                    "password": {
                        "type": "string",
                        "description": "Device login password when authentication is required"
                    },
                    "device_id": {
                        "type": "string",
                        "description": "Identifier, IP address, or alias of the target device"
                    },
                    "state": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Desired power state for plugs or child outlets"
                    },
                    "brightness": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Brightness percentage for dimmable lights"
                    },
                    "hue": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 360,
                        "description": "Hue component for HSV color adjustments"
                    },
                    "saturation": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Saturation component for HSV color adjustments"
                    },
                    "value": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Optional brightness component for HSV color adjustments"
                    },
                    "temperature": {
                        "type": "integer",
                        "minimum": 2500,
                        "maximum": 9000,
                        "description": "Color temperature in Kelvin for tunable white lights"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["realtime", "today", "month"],
                        "description": "Energy usage period to retrieve"
                    },
                    "alias": {
                        "type": "string",
                        "description": "New alias to assign to the device"
                    },
                    "child_id": {
                        "type": "string",
                        "description": "Child outlet identifier or index on a power strip"
                    }
                },
                "required": ["operation"]
            }
        }

    simple_description = "Control TP-Link Kasa smart home devices - plugs, bulbs, switches, light strips. Turn on/off, adjust brightness/color, check energy usage, discover devices on local network."

    def __init__(self):
        """Initialize the Kasa tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        from config import config
        
        # Defer user-context-dependent operations to first use
        self._cache_directory = None
        self._device_mapping_path = None
        self._cache = None
        self._cache_duration = config.kasa_tool.cache_duration
        self._device_mapping = None
    
    @property
    def cache_directory(self):
        """Lazy-load cache directory."""
        if self._cache_directory is None:
            self._cache_directory = self.make_dir("cache")
        return self._cache_directory
    
    @property
    def device_mapping_path(self):
        """Lazy-load device mapping path."""
        if self._device_mapping_path is None:
            self._device_mapping_path = self.get_file_path("device_mapping.json")
        return self._device_mapping_path
    
    @property
    def cache(self):
        """Lazy-load device cache."""
        if self._cache is None:
            self._cache = DeviceCache(
                str(self.cache_directory),
                self._cache_duration
            )
        return self._cache
    
    @property
    def device_mapping(self):
        """Lazy-load device mapping."""
        if self._device_mapping is None:
            self._device_mapping = self._load_device_mapping()
        return self._device_mapping

    @property
    def _device_instances(self):
        """
        Get thread-local device instances.

        Returns:
            Dictionary of device instances for the current thread
        """
        return self.get_thread_data('device_instances', {})
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a Kasa tool operation.
        
        Args:
            operation: The operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            Dict containing the operation results
            
        Raises:
            ValueError: If operation fails or parameters are invalid
        """
        try:
            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in kwargs for kasa_tool: {e}")
                    raise ValueError(f"Invalid JSON in kwargs: {e}")
            
            # Route to the appropriate operation
            if operation == "discover_devices":
                return self.run_async(self._discover_devices(**kwargs))
            elif operation == "get_device_info":
                return self.run_async(self._get_device_info(**kwargs))
            elif operation == "power_control":
                return self.run_async(self._power_control(**kwargs))
            elif operation == "set_brightness":
                return self.run_async(self._set_brightness(**kwargs))
            elif operation == "set_color":
                return self.run_async(self._set_color(**kwargs))
            elif operation == "set_color_temp":
                return self.run_async(self._set_color_temp(**kwargs))
            elif operation == "get_energy_usage":
                return self.run_async(self._get_energy_usage(**kwargs))
            elif operation == "set_device_alias":
                return self.run_async(self._set_device_alias(**kwargs))
            elif operation == "get_child_devices":
                return self.run_async(self._get_child_devices(**kwargs))
            elif operation == "control_child_device":
                return self.run_async(self._control_child_device(**kwargs))
            else:
                self.logger.error(f"Unknown operation '{operation}' in kasa_tool")
                raise ValueError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "discover_devices, get_device_info, power_control, set_brightness, "
                    "set_color, set_color_temp, get_energy_usage, set_device_alias, "
                    "get_child_devices, control_child_device"
                )
        except Exception as e:
            self.logger.error(f"Error executing kasa_tool operation '{operation}': {e}")
            raise
    
    async def _discover_devices(
        self,
        target: Optional[str] = None,
        timeout: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover Kasa devices on the local network.
        
        Args:
            target: The network address to target for discovery
            timeout: Timeout in seconds for discovery
            username: Username for devices requiring authentication
            password: Password for devices requiring authentication
            
        Returns:
            Dict containing information about discovered devices
        """
        from kasa import Discover, Credentials
        from config import config
        
        # Use default values from config if not provided
        target = target or config.kasa_tool.discovery_target
        timeout = timeout or config.kasa_tool.discovery_timeout
        
        # Process credentials
        credentials = None
        if username and password:
            credentials = Credentials(username, password)
        elif config.kasa_tool.default_username and config.kasa_tool.default_password:
            credentials = Credentials(
                config.kasa_tool.default_username,
                config.kasa_tool.default_password
            )
        
        self.logger.info(f"Discovering Kasa devices on {target} with timeout {timeout}s")
        
        try:
            # Perform device discovery
            found_devices = await Discover.discover(
                target=target,
                discovery_timeout=timeout,
                credentials=credentials
            )
            
            # Update devices after discovery to get full information
            device_details = []
            for device in found_devices.values():
                try:
                    await device.update()
                    self._cache_device(device)
                    device_details.append(self._serialize_device_summary(device))
                except Exception as e:
                    self.logger.warning(f"Error updating device {device.host}: {e}")
            
            return {
                "success": True,
                "devices": device_details,
                "message": f"Found {len(device_details)} Kasa device(s) on the network"
            }
        except Exception as e:
            self.logger.error(f"Error discovering devices in kasa_tool: {e}")
            raise ValueError(f"Error discovering devices: {str(e)}")
    
    async def _get_device_info(self, device_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific device.
        
        Args:
            device_id: The device identifier (IP address, hostname, or alias)
            
        Returns:
            Dict containing device information
        """
        self.logger.info(f"Getting info for device: {device_id}")
        
        try:
            # Get device and ensure it's updated
            device = await self._get_device_by_id(device_id)
            await device.update()
            
            # Cache the device after update
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_details(device),
                "message": f"Retrieved information for {device.alias or device.host}"
            }
        except Exception as e:
            self.logger.error(f"Error getting device info in kasa_tool: {e}")
            raise ValueError(f"Error getting device info: {str(e)}")
    
    async def _power_control(self, device_id: str, state: str) -> Dict[str, Any]:
        """
        Turn a device on or off.
        
        Args:
            device_id: The device identifier
            state: The desired state ("on" or "off")
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting power state for {device_id} to {state}")
        
        # Validate state
        if state.lower() not in ["on", "off"]:
            self.logger.error(f"Invalid state '{state}' in kasa_tool")
            raise ValueError(f"Invalid state: {state}. Must be 'on' or 'off'")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Set the state
            if state.lower() == "on":
                await device.turn_on()
            else:
                await device.turn_off()
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    device, 
                    state.lower() == "on", 
                    "is_on",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Turned {device.alias or device.host} {state.lower()}"
            }
        except Exception as e:
            self.logger.error(f"Error controlling device in kasa_tool: {e}")
            raise ValueError(f"Error controlling device: {str(e)}")
    
    async def _set_brightness(self, device_id: str, brightness: Union[int, str]) -> Dict[str, Any]:
        """
        Set the brightness of a light bulb or light strip.
        
        Args:
            device_id: The device identifier
            brightness: Brightness level (0-100)
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting brightness for {device_id} to {brightness}")
        
        # Validate and convert brightness
        try:
            brightness_value = int(brightness)
            if brightness_value < 0 or brightness_value > 100:
                raise ValueError("Brightness must be between 0 and 100")
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid brightness value '{brightness}' in kasa_tool")
            raise ValueError(f"Invalid brightness value: {brightness}. Must be an integer between 0 and 100")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports brightness
            light_module = None
            if "Light" in device.modules:
                light_module = device.modules["Light"]
            
            if not light_module or not hasattr(light_module, 'set_brightness'):
                self.logger.error(f"Device {device.alias or device.host} does not support brightness control in kasa_tool")
                raise ValueError(f"Device {device.alias or device.host} does not support brightness control")
            
            # Set brightness
            await light_module.set_brightness(brightness_value)
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    light_module, 
                    brightness_value, 
                    "brightness",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Set brightness of {device.alias or device.host} to {brightness_value}%"
            }
        except Exception as e:
            self.logger.error(f"Error setting brightness in kasa_tool: {e}")
            raise ValueError(f"Error setting brightness: {str(e)}")
    
    async def _set_color(
        self, 
        device_id: str, 
        hue: Union[int, str], 
        saturation: Union[int, str], 
        value: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Set the color of a light bulb or light strip.
        
        Args:
            device_id: The device identifier
            hue: Hue value (0-360)
            saturation: Saturation value (0-100)
            value: Brightness value (0-100)
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting color for {device_id} to H:{hue} S:{saturation} V:{value}")
        
        # Validate and convert color values
        try:
            hue_value = int(hue)
            if hue_value < 0 or hue_value > 360:
                raise ValueError("Hue must be between 0 and 360")
                
            saturation_value = int(saturation)
            if saturation_value < 0 or saturation_value > 100:
                raise ValueError("Saturation must be between 0 and 100")
                
            if value is not None:
                value_value = int(value)
                if value_value < 0 or value_value > 100:
                    raise ValueError("Value must be between 0 and 100")
            else:
                value_value = None
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid color values in kasa_tool: {e}")
            raise ValueError(f"Invalid color values: {str(e)}")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports color
            light_module = None
            if "Light" in device.modules:
                light_module = device.modules["Light"]
            
            if not light_module or not hasattr(light_module, 'set_hsv'):
                self.logger.error(f"Device {device.alias or device.host} does not support color control in kasa_tool")
                raise ValueError(f"Device {device.alias or device.host} does not support color control")
            
            # Set HSV
            if value_value is not None:
                await light_module.set_hsv(hue_value, saturation_value, value_value)
            else:
                # Use current brightness if value not provided
                current_hsv = getattr(light_module, 'hsv', None)
                current_value = current_hsv.value if current_hsv else 100
                await light_module.set_hsv(hue_value, saturation_value, current_value)
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Set color of {device.alias or device.host} to HSV({hue_value}, {saturation_value}, {value_value or current_value})"
            }
        except Exception as e:
            self.logger.error(f"Error setting color in kasa_tool: {e}")
            raise ValueError(f"Error setting color: {str(e)}")
    
    async def _set_color_temp(self, device_id: str, temperature: Union[int, str]) -> Dict[str, Any]:
        """
        Set the color temperature of a light bulb.
        
        Args:
            device_id: The device identifier
            temperature: Color temperature in Kelvin
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting color temperature for {device_id} to {temperature}K")
        
        # Validate and convert temperature
        try:
            temperature_value = int(temperature)
            if temperature_value < 2500 or temperature_value > 9000:
                raise ValueError("Temperature must be between 2500K and 9000K")
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid temperature value '{temperature}' in kasa_tool")
            raise ValueError(f"Invalid temperature value: {temperature}. Must be an integer between 2500 and 9000")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports color temperature
            light_module = None
            if "Light" in device.modules:
                light_module = device.modules["Light"]
            
            if not light_module or not hasattr(light_module, 'set_color_temp'):
                self.logger.error(f"Device {device.alias or device.host} does not support color temperature control in kasa_tool")
                raise ValueError(f"Device {device.alias or device.host} does not support color temperature control")
            
            # Set color temperature
            await light_module.set_color_temp(temperature_value)
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    light_module, 
                    temperature_value, 
                    "color_temp",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Set color temperature of {device.alias or device.host} to {temperature_value}K"
            }
        except Exception as e:
            self.logger.error(f"Error setting color temperature in kasa_tool: {e}")
            raise ValueError(f"Error setting color temperature: {str(e)}")
    
    async def _get_energy_usage(self, device_id: str, period: str = "realtime") -> Dict[str, Any]:
        """
        Get energy usage data for supported devices.
        
        Args:
            device_id: The device identifier
            period: Period to retrieve ("realtime", "today", "month")
            
        Returns:
            Dict containing energy usage data
        """
        self.logger.info(f"Getting energy usage for {device_id} for period: {period}")
        
        # Validate period
        valid_periods = ["realtime", "today", "month"]
        if period.lower() not in valid_periods:
            self.logger.error(f"Invalid period '{period}' for energy usage in kasa_tool")
            raise ValueError(f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports energy monitoring
            if not device.has_emeter:
                self.logger.error(f"Device {device.alias or device.host} does not support energy monitoring in kasa_tool")
                raise ValueError(f"Device {device.alias or device.host} does not support energy monitoring")
            
            # Get the energy module
            energy_module = None
            if "Energy" in device.modules:
                energy_module = device.modules["Energy"]
            else:
                self.logger.error(f"Device {device.alias or device.host} does not have energy module in kasa_tool")
                raise ValueError(f"Device {device.alias or device.host} does not have energy module")
            
            # Get energy data based on period
            energy_data = {}
            if period.lower() == "realtime":
                await energy_module.update()  # Ensure we have the latest data
                energy_data = {
                    "current_power": energy_module.current_consumption,
                    "voltage": getattr(energy_module, 'voltage', None),
                    "current": getattr(energy_module, 'current', None)
                }
                message = f"Current power usage for {device.alias or device.host} is {energy_data['current_power']}W"
            elif period.lower() == "today":
                energy_data = {
                    "consumption_today": energy_module.consumption_today
                }
                message = f"Energy consumption today for {device.alias or device.host} is {energy_data['consumption_today']}kWh"
            elif period.lower() == "month":
                energy_data = {
                    "consumption_month": energy_module.consumption_this_month
                }
                message = f"Energy consumption this month for {device.alias or device.host} is {energy_data['consumption_month']}kWh"
            
            return {
                "success": True,
                "device": {
                    "id": device.host,
                    "alias": device.alias,
                    "model": device.model
                },
                "energy_data": energy_data,
                "message": message
            }
        except Exception as e:
            self.logger.error(f"Error getting energy usage in kasa_tool: {e}")
            raise ValueError(f"Error getting energy usage: {str(e)}")
    
    async def _set_device_alias(self, device_id: str, alias: str) -> Dict[str, Any]:
        """
        Set a new name for the device.
        
        Args:
            device_id: The device identifier
            alias: New name for the device
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting alias for {device_id} to {alias}")
        
        # Validate alias
        if not alias or not isinstance(alias, str):
            self.logger.error("Invalid alias provided for device in kasa_tool")
            raise ValueError("Alias must be a non-empty string")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Save old alias for the response message
            old_alias = device.alias or device.host
            
            # Set new alias
            await device.set_alias(alias)
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Renamed device from '{old_alias}' to '{device.alias}'"
            }
        except Exception as e:
            self.logger.error(f"Error setting device alias in kasa_tool: {e}")
            raise ValueError(f"Error setting device alias: {str(e)}")
    
    async def _get_child_devices(self, device_id: str) -> Dict[str, Any]:
        """
        Get information about child devices for power strips.
        
        Args:
            device_id: The device identifier of the parent device
            
        Returns:
            Dict containing information about child devices
        """
        self.logger.info(f"Getting child devices for {device_id}")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device has children
            if not hasattr(device, 'children') or not device.children:
                self.logger.error(f"Device {device.alias or device.host} does not have child devices in kasa_tool")
                raise ValueError(f"Device {device.alias or device.host} does not have child devices")
            
            # Get information about child devices
            child_devices = []
            for child in device.children:
                child_devices.append(self._serialize_device_summary(child))
            
            return {
                "success": True,
                "parent_device": {
                    "id": device.host,
                    "alias": device.alias,
                    "model": device.model
                },
                "child_devices": child_devices,
                "message": f"Found {len(child_devices)} child devices for {device.alias or device.host}"
            }
        except Exception as e:
            self.logger.error(f"Error getting child devices in kasa_tool: {e}")
            raise ValueError(f"Error getting child devices: {str(e)}")
    
    async def _control_child_device(
        self, 
        device_id: str, 
        child_id: str, 
        state: str
    ) -> Dict[str, Any]:
        """
        Control a specific outlet on a power strip.
        
        Args:
            device_id: The device identifier of the parent device
            child_id: The ID or index of the child device
            state: The desired state ("on" or "off")
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Controlling child device {child_id} of {device_id} with state {state}")
        
        # Validate state
        if state.lower() not in ["on", "off"]:
            self.logger.error(f"Invalid state '{state}' in kasa_tool")
            raise ValueError(f"Invalid state: {state}. Must be 'on' or 'off'")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device has children
            if not hasattr(device, 'children') or not device.children:
                self.logger.error(f"Device {device.alias or device.host} does not have child devices in kasa_tool")
                raise ValueError(f"Device {device.alias or device.host} does not have child devices")
            
            # Get child device by ID or index
            child_device = None
            
            # First try to get by alias
            child_device = device.get_plug_by_name(child_id)
            
            # If not found by alias, try by index
            if not child_device and child_id.isdigit():
                try:
                    index = int(child_id)
                    child_device = device.get_plug_by_index(index)
                except Exception:
                    pass
            
            if not child_device:
                self.logger.error(f"Child device {child_id} not found in kasa_tool")
                raise ValueError(f"Child device {child_id} not found")
            
            # Set the state
            if state.lower() == "on":
                await child_device.turn_on()
            else:
                await child_device.turn_off()
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    child_device, 
                    state.lower() == "on", 
                    "is_on",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "parent_device": {
                    "id": device.host,
                    "alias": device.alias,
                    "model": device.model
                },
                "child_device": self._serialize_device_summary(child_device),
                "message": f"Turned {child_device.alias or child_id} {state.lower()}"
            }
        except Exception as e:
            self.logger.error(f"Error controlling child device in kasa_tool: {e}")
            raise ValueError(f"Error controlling child device: {str(e)}")
    
    # -------------------- HELPER METHODS --------------------

    def _load_device_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the device mapping from file.

        Returns:
            Dictionary mapping device identifiers to their information
        """
        if not self.file_exists("device_mapping.json"):
            self.logger.info("Device mapping file not found")
            return {}

        try:
            with self.open_file("device_mapping.json", 'r') as f:
                mapping = json.load(f)
                self.logger.info(f"Loaded {len(mapping)} devices from mapping file")
                return mapping
        except Exception as e:
            self.logger.warning(f"Error loading device mapping: {e}")
            return {}

    def _calculate_name_similarity(self, query: str, device_name: str) -> float:
        """
        Calculate similarity score between query and device name.
        
        Args:
            query: Normalized query string
            device_name: Normalized device name
            
        Returns:
            Similarity score (0-100, higher is better match)
        """
        # Exact match gets highest score
        if query == device_name:
            return 100
        # Contained words get medium scores
        elif query in device_name:
            return 80
        elif device_name in query:
            return 70
        else:
            # Calculate word overlap
            query_words = set(query.split())
            name_words = set(device_name.split())

            if query_words and name_words:
                common_words = query_words.intersection(name_words)
                total_words = len(query_words.union(name_words))

                if common_words:
                    # Percentage of matching words
                    return 60 * len(common_words) / total_words
        
        return 0  # No match
    
    def _find_best_device_match(self, query: str) -> Union[str, Dict[str, Any]]:
        """
        Find the best device match from all available sources based on name similarity.
        
        This enhanced version:
        1. Checks both in-memory and persistent cache with unified logic
        2. Returns a single best match if clear
        3. Returns multiple options if ambiguous rather than raising an error
        4. Provides available devices when no match is found

        Args:
            query: The device name or identifier to match

        Returns:
            Either the best matching device identifier (str) or a dict with potential matches
        """
        from config import config
        
        # Return exact match if exists
        if query in self._device_instances:
            return query

        # Normalize the query for better matching
        norm_query = query.lower().replace("the ", "").strip()

        # Track potential matches with scores
        candidates = {}
        
        # Get all potential device sources
        sources = []
        
        # 1. Add in-memory devices
        for device_id, device in self._device_instances.items():
            alias = device.alias if hasattr(device, 'alias') else None
            if alias:
                sources.append({
                    "id": device_id,
                    "alias": alias,
                    "source": "memory"
                })
        
        # 2. Add devices from persistent cache if enabled
        if config.kasa_tool.cache_enabled:
            cached_devices = self.cache._load_cache()
            for key, device_data in cached_devices.items():
                # Only add if not already in-memory
                if key not in self._device_instances:
                    alias = device_data.get('alias')
                    if alias:
                        sources.append({
                            "id": key,
                            "alias": alias,
                            "host": device_data.get('host'),
                            "model": device_data.get('model'),
                            "source": "cache"
                        })
        
        # Evaluate all sources with the same logic
        for device in sources:
            device_id = device["id"]
            alias = device["alias"]
            
            # Skip IP addresses - we're looking for name matches
            if '.' in device_id and device_id.replace('.', '').isdigit():
                continue
                
            # Skip empty aliases
            if not alias:
                continue
                
            # Normalize the alias
            norm_alias = alias.lower().replace("the ", "").strip()
            
            # Calculate match score using unified logic
            score = self._calculate_name_similarity(norm_query, norm_alias)
            
            # Record candidate if score is above threshold
            if score > 50:
                candidates[device_id] = {
                    "score": score,
                    "alias": alias,
                    "source": device["source"]
                }
                self.logger.debug(f"Match candidate: '{device_id}' (alias: '{alias}', score: {score}, source: {device['source']})")

        # If no candidates found
        if not candidates:
            # Return information about no match but don't raise an error
            available_devices = [s["alias"] for s in sources if s["alias"] and not ('.' in s["id"] and s["id"].replace('.', '').isdigit())]
            return {
                "match_type": "none",
                "query": query,
                "message": f"No device found matching '{query}'. Try discover_devices or use one of the available devices.",
                "available_devices": sorted(list(set(available_devices)))  # Remove duplicates and sort
            }
        
        # Sort candidates by score (highest first)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Check for clear winner vs ambiguous matches
        if len(sorted_candidates) == 1:
            # Only one match - return it
            best_match_id = sorted_candidates[0][0]
            alias = candidates[best_match_id]["alias"]
            score = candidates[best_match_id]["score"]
            self.logger.info(f"Found match for '{query}': '{alias}' (device_id: '{best_match_id}', score: {score})")
            return best_match_id
            
        # Check if there's a clear winner (significantly higher score)
        best_score = sorted_candidates[0][1]["score"]
        second_best_score = sorted_candidates[1][1]["score"] if len(sorted_candidates) > 1 else 0
        
        if best_score > second_best_score + 20:  # Clear winner if 20+ points higher
            best_match_id = sorted_candidates[0][0]
            alias = candidates[best_match_id]["alias"]
            self.logger.info(f"Found best match for '{query}': '{alias}' (device_id: '{best_match_id}', score: {best_score})")
            return best_match_id
            
        # Ambiguous matches - return options instead of raising an error
        devices_list = []
        for candidate_id, candidate_info in sorted_candidates[:5]:  # Limit to top 5 matches
            devices_list.append({
                "id": candidate_id,
                "alias": candidate_info["alias"],
                "score": candidate_info["score"],
                "source": candidate_info.get("source", "unknown")
            })
            
        return {
            "match_type": "ambiguous",
            "query": query,
            "message": f"Multiple potential devices match '{query}'. Please choose one of the following devices.",
            "devices": devices_list
        }
    
    async def _get_device_by_id(
        self,
        device_id: str,
        force_discovery: bool = False,
        return_match_info: bool = False
    ) -> Any:
        """
        Get a device by its identifier, using cache when available.
        
        Enhanced version that:
        1. Uses the smart fuzzy matcher
        2. Returns helpful options when ambiguous
        3. Provides list of available devices when no match
        4. Falls back to discovery as a last resort
        5. Can return match info instead of raising errors

        Args:
            device_id: The device identifier (IP address, hostname, or alias)
            force_discovery: Whether to force device discovery
            return_match_info: If True, return match info dict instead of raising errors

        Returns:
            Device object, or a dict with match info if ambiguous or not found and return_match_info=True

        Raises:
            ValueError: If device cannot be found and return_match_info=False
        """
        from kasa import Device, Credentials, Discover
        from config import config

        # Check for cached instance first (exact match)
        if device_id in self._device_instances and not force_discovery:
            device = self._device_instances[device_id]
            try:
                await device.update()
                self.logger.info(f"Using cached device: {device_id}")
                return device
            except Exception as e:
                self.logger.warning(f"Error updating cached device {device_id}: {e}")
                # Continue to try other methods if update fails

        # Apply smart fuzzy matching if not forcing discovery
        if not force_discovery:
            # Get the best match or information about available matches
            match_result = self._find_best_device_match(device_id)
            
            # If a string is returned, it's a single best match ID
            if isinstance(match_result, str):
                best_match_id = match_result
                if best_match_id in self._device_instances:
                    device = self._device_instances[best_match_id]
                    try:
                        await device.update()
                        alias = device.alias if hasattr(device, 'alias') else best_match_id
                        self.logger.info(f"Using fuzzy-matched device: '{device_id}' -> '{alias}'")
                        # Cache this match for future use
                        self._device_instances[device_id] = device
                        return device
                    except Exception as e:
                        self.logger.warning(f"Error updating fuzzy-matched device {best_match_id}: {e}")
                        # Continue to other methods
                
                # Try connecting to a device from the cache
                if config.kasa_tool.cache_enabled:
                    cached_devices = self.cache._load_cache()
                    if best_match_id in cached_devices:
                        device_data = cached_devices[best_match_id]
                        host = device_data.get('host')
                        if host:
                            try:
                                # Try to connect to the device
                                device = await Device.connect(host=host, config=None)
                                if device:
                                    self.logger.info(f"Connected to cached device: {best_match_id}")
                                    self._device_instances[device_id] = device
                                    self._device_instances[device.host] = device
                                    if device.alias:
                                        self._device_instances[device.alias] = device
                                    return device
                            except Exception as e:
                                self.logger.warning(f"Error connecting to cached device {host}: {e}")
                                # Continue to other methods
            
            # If a dict is returned, handle different match types
            elif isinstance(match_result, dict):
                match_type = match_result.get("match_type")
                
                # Ambiguous matches - either return info or raise error
                if match_type == "ambiguous":
                    devices = match_result.get("devices", [])
                    if return_match_info:
                        return match_result
                    
                    # Convert to a better error message
                    suggestions = ", ".join([f"'{d['alias']}'" for d in devices])
                    self.logger.error(f"Ambiguous device name '{device_id}' in kasa_tool")
                    raise ValueError(f"Ambiguous device name '{device_id}'. Did you mean one of: {suggestions}?")
                
                # No matches - return info or try discovery
                elif match_type == "none":
                    if return_match_info:
                        return match_result
                        
                    # Log and continue to discovery
                    available = match_result.get("available_devices", [])
                    if available:
                        suggestions = ", ".join([f"'{d}'" for d in available[:5]])  # Show up to 5
                        self.logger.info(f"No match for '{device_id}'. Available: {suggestions}")
                    else:
                        self.logger.info(f"No device found matching '{device_id}'")
                    # Continue to discovery
        
        # Setup credentials for possible use later
        credentials = None
        if config.kasa_tool.default_username and config.kasa_tool.default_password:
            credentials = Credentials(
                config.kasa_tool.default_username,
                config.kasa_tool.default_password
            )

        # Direct connection for IP addresses
        if '.' in device_id and not device_id.startswith('.'):
            try:
                self.logger.info(f"Attempting direct connection to: {device_id}")
                device = await Discover.discover_single(
                    device_id,
                    credentials=credentials
                )

                if device:
                    # Store in thread-local storage
                    self._device_instances[device_id] = device
                    self._device_instances[device.host] = device
                    if device.alias:
                        self._device_instances[device.alias] = device
                    return device
            except Exception as e:
                self.logger.warning(f"Error connecting directly to {device_id}: {e}")
                # Continue to discovery
        
        # If still not found and discovery is allowed
        if config.kasa_tool.attempt_discovery_when_not_found or force_discovery:
            try:
                self.logger.info(f"Device '{device_id}' not found in cache. Running network discovery.")
                # Perform device discovery
                found_devices = await Discover.discover(
                    target=config.kasa_tool.discovery_target,
                    discovery_timeout=config.kasa_tool.discovery_timeout,
                    credentials=credentials
                )
                
                # No devices found during discovery
                if not found_devices:
                    self.logger.error("No devices found on the network in kasa_tool")
                    raise ValueError(f"No devices found on the network. Check that your devices are powered on and connected.")
                
                # Cache and check discovered devices
                device_names = []
                for host, device in found_devices.items():
                    try:
                        await device.update()
                        
                        # Cache the device
                        self._cache_device(device)
                        
                        # Add to in-memory cache
                        self._device_instances[host] = device
                        if device.alias:
                            self._device_instances[device.alias] = device
                            device_names.append(device.alias)
                            
                        # Check if this device matches the requested ID
                        if device.alias == device_id or host == device_id:
                            return device
                    except Exception as e:
                        self.logger.warning(f"Error updating discovered device {host}: {e}")
                
                # If we found devices but none matched, suggest them
                if device_names:
                    suggestions = ", ".join([f"'{name}'" for name in device_names[:5]])
                    self.logger.error(f"Device '{device_id}' not found in kasa_tool")
                    raise ValueError(f"Device '{device_id}' not found. Available devices: {suggestions}")
            except ValueError:
                # Re-raise value errors
                raise
            except Exception as e:
                self.logger.warning(f"Error during device discovery: {e}")
                # Continue to final error
        
        # If we get here, device was not found
        if return_match_info:
            return {
                "match_type": "none",
                "query": device_id,
                "message": f"Device '{device_id}' not found in cache or network. Try discover_devices operation or use a different device name.",
                "available_devices": []
            }
            
        # Otherwise raise error
        self.logger.error(f"Device '{device_id}' not found in cache or network in kasa_tool")
        raise ValueError(f"Device '{device_id}' not found in cache or network. Make sure it is connected to your network or try discover_devices operation first.")

    def cleanup(self) -> None:
        """
        Clean up resources used by this tool.

        This method is called by the automation engine when an automation completes.
        It ensures that all device connections are properly closed.
        """
        # Get the thread-local device instances
        device_instances = self.get_thread_data('device_instances', {})

        # Clean up each device
        for device_id, device in list(device_instances.items()):
            try:
                # Only attempt to close once per physical device (by host)
                if hasattr(device, 'host') and device.host == device_id:
                    self.logger.debug(f"Cleaning up kasa device: {device_id}")
                    # Note: python-kasa doesn't have an explicit close method,
                    # but we should clear our references to these objects
            except Exception as e:
                self.logger.warning(f"Error cleaning up device {device_id}: {e}")

        # Clear the device instances for this thread
        self.set_thread_data('device_instances', {})

        # Call parent cleanup to handle common cleanup tasks
        super().cleanup()
    
    def _serialize_device_summary(self, device: Any) -> Dict[str, Any]:
        """
        Serialize basic device information for response.
        
        Args:
            device: The device object
            
        Returns:
            Dict containing basic device information
        """
        # Basic device information
        result = {
            "id": device.host,
            "alias": device.alias,
            "model": device.model,
            "state": "on" if device.is_on else "off"
        }
        
        # Add device type
        device_type = str(device.device_type).lower()
        if 'bulb' in device_type:
            result["type"] = "bulb"
        elif 'plug' in device_type:
            result["type"] = "plug"
        elif 'strip' in device_type:
            result["type"] = "strip"
        elif 'dimmer' in device_type or 'switch' in device_type:
            result["type"] = "switch"
        elif 'lightstrip' in device_type:
            result["type"] = "lightstrip"
        else:
            result["type"] = "other"
        
        # Add supported features
        features = []
        
        # Check for brightness support
        if "Light" in device.modules and hasattr(device.modules["Light"], "brightness"):
            features.append("brightness")
            result["brightness"] = device.modules["Light"].brightness
        
        # Check for color support
        if "Light" in device.modules and hasattr(device.modules["Light"], "hsv"):
            features.append("color")
            hsv = device.modules["Light"].hsv
            if hsv:
                result["color"] = {
                    "hue": hsv.hue,
                    "saturation": hsv.saturation,
                    "value": hsv.value
                }
        
        # Check for color temperature support
        if "Light" in device.modules and hasattr(device.modules["Light"], "color_temp"):
            features.append("temperature")
            result["color_temp"] = device.modules["Light"].color_temp
        
        # Check for energy monitoring support
        if device.has_emeter:
            features.append("energy")
            if "Energy" in device.modules:
                result["power"] = device.modules["Energy"].current_consumption
        
        # Add features list to result
        result["features"] = features
        
        return result
    
    def _serialize_device_details(self, device: Any) -> Dict[str, Any]:
        """
        Serialize detailed device information for response.
        
        Args:
            device: The device object
            
        Returns:
            Dict containing detailed device information
        """
        # Start with summary
        result = self._serialize_device_summary(device)
        
        # Add additional information
        result["mac"] = device.mac
        result["rssi"] = device.rssi
        result["hardware_version"] = device.device_info.hardware_version
        result["firmware_version"] = device.device_info.firmware_version
        
        # Add time information
        if hasattr(device, "time"):
            result["device_time"] = device.time.isoformat()
        if device.on_since:
            result["on_since"] = device.on_since.isoformat()
        
        # Add modules list
        result["modules"] = list(device.modules.keys())
        
        # Add features detailed information
        result["features_details"] = {}
        for feature_id, feature in device.features.items():
            result["features_details"][feature_id] = {
                "name": feature.name,
                "value": str(feature.value)
            }
        
        # Add energy information if supported
        if device.has_emeter and "Energy" in device.modules:
            energy_module = device.modules["Energy"]
            result["energy"] = {
                "current_consumption": energy_module.current_consumption,
                "consumption_today": energy_module.consumption_today,
                "consumption_month": energy_module.consumption_this_month,
                "voltage": getattr(energy_module, "voltage", None),
                "current": getattr(energy_module, "current", None)
            }
        
        # Add child devices if any
        if hasattr(device, "children") and device.children:
            result["child_devices"] = []
            for child in device.children:
                result["child_devices"].append(self._serialize_device_summary(child))
        
        return result
    
    def _cache_device(self, device: Any) -> None:
        """
        Cache a device for future use.
        
        Args:
            device: The device object to cache
        """
        from config import config
        
        if not config.kasa_tool.cache_enabled:
            return
        
        try:
            # Generate cache key from device host
            cache_key = device.host
            
            # Prepare device data for caching
            device_data = {
                "host": device.host,
                "alias": device.alias,
                "model": device.model,
                "mac": device.mac,
                "device_type": str(device.device_type),
                "last_updated": utc_now().isoformat()
            }
            
            # Add to cache
            self.cache.set(cache_key, device_data)
            
            # Also cache by alias if available
            if device.alias:
                self.cache.set(device.alias, device_data)
                
            # Store in thread-local memory cache
            device_instances = self.get_thread_data('device_instances', {})
            device_instances[device.host] = device
            if device.alias:
                device_instances[device.alias] = device
            self.set_thread_data('device_instances', device_instances)
                
        except Exception as e:
            self.logger.warning(f"Error caching device {device.host}: {e}")
    
    async def _verify_change(
        self, 
        device: Any, 
        expected_value: Any, 
        attribute: str,
        max_attempts: int = 3,
        delay: float = 0.5
    ) -> bool:
        """
        Verify a change was successful by checking the device state.
        
        Args:
            device: The device or module object
            expected_value: The expected value
            attribute: The attribute to check
            max_attempts: Maximum number of verification attempts
            delay: Delay between verification attempts in seconds
            
        Returns:
            True if the change was successful, False otherwise
        """
        for attempt in range(max_attempts):
            try:
                # If device has an update method, use it
                if hasattr(device, "update"):
                    await device.update()
                # If device has a parent with an update method, use that
                elif hasattr(device, "_device") and hasattr(device._device, "update"):
                    await device._device.update()
                    
                # Check if the attribute has the expected value
                current_value = getattr(device, attribute, None)
                if current_value == expected_value:
                    return True
                    
                # If not, wait and retry
                await asyncio.sleep(delay)
            except Exception as e:
                self.logger.warning(f"Error verifying change: {e}")
                await asyncio.sleep(delay)
                
        # If we get here, verification failed
        self.logger.warning(
            f"Failed to verify change: expected {attribute}={expected_value}, "
            f"got {getattr(device, attribute, None)}"
        )
        return False
