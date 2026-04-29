"""
File serving endpoints for code execution artifacts and tool-generated images.

Serves files from data/users/{user_id}/artifacts/{file_id}/ that were eagerly
downloaded from Anthropic during response processing. Security layers:
1. Auth gate (Depends(get_current_user)) scopes to user's directory
2. file_id regex validation (alphanumeric + hyphen/underscore only)
3. Path traversal guard (resolved path must stay within base_dir)
4. Safe headers (Content-Disposition varies by endpoint)
"""
import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from auth.api import get_current_user
from auth.types import SessionData, APITokenContext

router = APIRouter()

FILE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


@router.get("/files/{file_id}")
async def download_file(
    file_id: str,
    current_user: SessionData | APITokenContext = Depends(get_current_user)
) -> FileResponse:
    """Download a code execution file artifact.

    Files are stored during response processing and persist across segments.
    """
    if not FILE_ID_PATTERN.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")

    user_id = current_user.user_id
    from utils.instance import user_data_base
    base_dir = (user_data_base() / user_id / "artifacts" / file_id).resolve()

    if not base_dir.is_dir():
        raise HTTPException(status_code=404, detail="File not found")

    # Find the .bin (content) and .meta (metadata) files
    bin_files = list(base_dir.glob("*.bin"))
    meta_files = list(base_dir.glob("*.meta"))
    if not bin_files or not meta_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = bin_files[0].resolve()
    # Path traversal guard — resolved path must be within base_dir
    file_path.relative_to(base_dir)

    # Read original filename from metadata sidecar
    meta: dict[str, str] = json.loads(meta_files[0].read_text())
    original_filename = meta.get("filename", "download")

    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{original_filename}"',
            "Cache-Control": "no-store",
        }
    )


@router.get("/images/{file_id}")
async def view_image(
    file_id: str,
    current_user: SessionData | APITokenContext = Depends(get_current_user)
) -> FileResponse:
    """Serve a tool-generated image for inline display.

    Same storage convention as /files/{file_id} (.bin + .meta sidecar)
    but returns with Content-Disposition: inline so browsers render
    the image instead of downloading it.
    """
    if not FILE_ID_PATTERN.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")

    user_id = current_user.user_id
    from utils.instance import user_data_base
    base_dir = (user_data_base() / user_id / "artifacts" / file_id).resolve()

    if not base_dir.is_dir():
        raise HTTPException(status_code=404, detail="Image not found")

    bin_files = list(base_dir.glob("*.bin"))
    meta_files = list(base_dir.glob("*.meta"))
    if not bin_files or not meta_files:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = bin_files[0].resolve()
    file_path.relative_to(base_dir)  # Path traversal guard

    meta: dict[str, str] = json.loads(meta_files[0].read_text())
    mime_type = meta.get("mime_type", "image/png")
    filename = meta.get("filename", "image.png")

    return FileResponse(
        path=file_path,
        media_type=mime_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "public, max-age=3600",
        }
    )
