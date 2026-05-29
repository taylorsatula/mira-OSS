"""Capability declarations and validation for LLM adapters."""

from __future__ import annotations

from dataclasses import dataclass


class CapabilityError(RuntimeError):
    """Raised before transport when a request requires unsupported behavior."""


@dataclass(frozen=True)
class Capabilities:
    """Capabilities supported by one dialect."""

    batch: bool = False
    files: bool = False
    container_reuse: bool = False
    server_code_execution: bool = False

    def validate(self, requirements: Requirements, dialect_name: str) -> None:
        unsupported = []
        if requirements.batch and not self.batch:
            unsupported.append("batch")
        if requirements.files and not self.files:
            unsupported.append("files")
        if requirements.container_reuse and not self.container_reuse:
            unsupported.append("container_reuse")
        if requirements.server_code_execution and not self.server_code_execution:
            unsupported.append("server_code_execution")

        if unsupported:
            raise CapabilityError(
                f"Dialect '{dialect_name}' does not support required capabilities: "
                f"{', '.join(unsupported)}"
            )


@dataclass(frozen=True)
class Requirements:
    """Capabilities required by one request."""

    batch: bool = False
    files: bool = False
    container_reuse: bool = False
    server_code_execution: bool = False
