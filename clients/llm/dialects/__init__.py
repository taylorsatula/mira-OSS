"""Provider-specific dialects for the neutral LLM boundary.

Each module in this package contributes a concrete Dialect subclass discovered
by clients.llm.dialect_registry. Adding a new dialect is a drop-a-file operation:
declare the class with a dialect_name in DialectName, implement from_selection,
and the registry picks it up at startup.
"""
