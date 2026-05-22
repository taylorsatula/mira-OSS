"""
Anthropic Files API Manager.

Manages file uploads, lifecycle, and cleanup for structured data files
(CSV, XLSX, JSON) sent to code execution tool.
"""

import logging
from typing import Set
import anthropic
from anthropic import APIStatusError

from tools.repo import FILES_API_BETA_FLAG
from utils.timezone_utils import format_utc_iso, utc_now


class FilesManager:
    """
    Manages Anthropic Files API operations with segment-scoped lifecycle.

    Responsibilities:
    - Upload files to Anthropic Files API
    - Track uploaded files per segment for cleanup
    - Delete files when segment collapses
    - Handle API errors with recovery guidance

    Lifecycle:
    - Files persist for the duration of the conversation segment
    - Cleanup occurs when segment collapses (history compression)
    - Enables multi-turn code execution on same file
    """

    def __init__(self, anthropic_client: anthropic.Anthropic):
        """
        Initialize FilesManager with Anthropic client.

        Args:
            anthropic_client: Initialized Anthropic SDK client
        """
        self.client = anthropic_client
        self.logger = logging.getLogger("files_manager")
        from utils.user_context import get_current_user_id
        from utils.userdata_manager import get_user_data_manager
        self._db = get_user_data_manager(get_current_user_id())
        self._db._init_files_api_schema()

    def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        media_type: str,
        segment_id: str
    ) -> str:
        """
        Upload file to Anthropic Files API.

        Args:
            file_bytes: File content as bytes
            filename: Original filename for tracking
            media_type: MIME type (e.g., "text/csv")
            segment_id: Segment ID for lifecycle tracking

        Returns:
            file_id for use in container_upload blocks

        Raises:
            ValueError: File too large (>32MB, Anthropic server-enforced limit)
            RuntimeError: API errors (403, 404, etc.)
        """
        try:
            # Upload file with beta API
            self.logger.debug(f"Uploading file {filename} ({media_type}) for segment {segment_id}")

            response = self.client.beta.files.upload(
                file=(filename, file_bytes, media_type),
                betas=[FILES_API_BETA_FLAG]
            )

            file_id = response.id

            self._track_upload(file_id, segment_id, filename, media_type)

            self.logger.info(f"Uploaded file {filename} → file_id: {file_id} (segment: {segment_id})")
            return file_id

        except APIStatusError as e:
            if e.status_code == 413:
                self.logger.error(f"File too large: {filename} ({len(file_bytes)} bytes)", exc_info=True)
                raise ValueError(
                    f"File too large for Files API. Maximum size: 32MB (Anthropic server-enforced limit). "
                    f"Consider splitting the file or using data sampling. "
                    f"Current size: {len(file_bytes) / (1024*1024):.1f}MB"
                )
            elif e.status_code == 403:
                self.logger.error(f"Files API access denied: {e}", exc_info=True)
                raise RuntimeError(
                    "Files API access denied. Check API key permissions for Files API beta access. "
                    "Contact Anthropic support if needed."
                )
            elif e.status_code == 404:
                self.logger.error(f"Files API endpoint not found: {e}", exc_info=True)
                raise RuntimeError(
                    "Files API endpoint not found. Verify beta flag is set correctly."
                )
            else:
                self.logger.error(f"Files API error ({e.status_code}): {e}", exc_info=True)
                raise RuntimeError(f"Files API error ({e.status_code}): {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error uploading file {filename}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to upload file: {str(e)}")

    def delete_file(self, file_id: str) -> bool:
        """
        Delete single file by ID.

        Args:
            file_id: File ID from upload

        Note:
            Handles 404 gracefully (file may already be deleted)
        """
        try:
            self.logger.debug(f"Deleting file: {file_id}")
            self.client.beta.files.delete(
                file_id=file_id,
                betas=[FILES_API_BETA_FLAG]
            )
            self.logger.debug(f"Deleted file: {file_id}")
            return True
        except APIStatusError as e:
            if e.status_code == 404:
                # File already deleted or never existed - not an error
                self.logger.debug(f"File not found (may already be deleted): {file_id}")
                return True
            else:
                self.logger.warning(f"Error deleting file {file_id}: {e}", exc_info=True)
                return False
        except Exception as e:
            # Log but don't fail request on cleanup errors
            self.logger.warning(f"Unexpected error deleting file {file_id}: {e}", exc_info=True)
            return False

    def cleanup_segment_files(self, segment_id: str) -> None:
        """
        Delete all files uploaded during this segment.

        Called when segment collapses (conversation history compression).

        Args:
            segment_id: Segment ID to cleanup files for

        Note:
            Removes tracking and deletes all uploaded files for segment.
            Gracefully handles deletion failures (logs warnings).
        """
        file_ids = self._get_tracked_file_ids(segment_id)

        if not file_ids:
            return

        self.logger.debug(f"Cleaning up {len(file_ids)} files for segment {segment_id}")

        for file_id in file_ids:
            if self.delete_file(file_id):
                self._delete_tracked_upload(file_id)

        self.logger.info(f"Cleanup complete for segment {segment_id}")

    def _track_upload(self, file_id: str, segment_id: str, filename: str, media_type: str) -> None:
        """Persist upload tracking in user SQLite."""
        self._db.insert('files_api_uploads', {
            'file_id': file_id,
            'user_id': self._db.user_id,
            'segment_id': segment_id,
            'filename': filename,
            'media_type': media_type,
            'created_at': format_utc_iso(utc_now()),
        })

    def _get_tracked_file_ids(self, segment_id: str) -> Set[str]:
        """Return persisted file IDs for a segment."""
        rows = self._db.select(
            'files_api_uploads',
            'segment_id = :segment_id',
            {'segment_id': segment_id}
        )
        return {row['file_id'] for row in rows}

    def _delete_tracked_upload(self, file_id: str) -> None:
        """Delete persistent tracking after remote deletion succeeds."""
        self._db.delete(
            'files_api_uploads',
            'file_id = :file_id',
            {'file_id': file_id}
        )
