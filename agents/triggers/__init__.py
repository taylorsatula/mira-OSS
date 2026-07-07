"""Sidebar dispatcher triggers.

Each trigger discovers work-items for a sidebar agent. Triggers are pure
discovery (no judgment) — the dispatcher owns dedup via sidebar_activity, and
the agent owns all decisions.
"""
from agents.triggers.memory_floor_trigger import MemoryFloorTrigger

__all__ = ["MemoryFloorTrigger"]
