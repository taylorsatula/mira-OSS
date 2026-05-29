"""Provider-neutral artifact persistence contracts for LLM dialects."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FileArtifactSink(Protocol):
    """Stores provider-produced file artifacts for later user retrieval."""

    def save_file_artifact(
        self,
        *,
        file_id: str,
        filename: str,
        mime_type: str,
        size_bytes: int,
        content: bytes,
    ) -> Path:
        """Persist a provider-produced file artifact and return the content path."""
