"""
OSS browser-auth + minimal asset routes.

Always-on routes that support the full web UI (web/): exposes the
single-user API key at /oss-auth/token for browser auto-login, and
serves vendored JS assets. The / and /chat page routes are owned by
main.py (de-auth-gated FileResponse routes over web/*/index.html).
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter()

# Read and cache assets at import time — no disk I/O per request
_oss_ui_dir = Path(__file__).resolve().parent.parent.parent / "deploy" / "oss_ui"
_marked_js = (_oss_ui_dir / "marked.min.js").read_text()
_purify_js = (_oss_ui_dir / "purify.min.js").read_text()


@router.get("/oss-assets/marked.min.js", include_in_schema=False)
async def serve_marked():
    return Response(_marked_js, media_type="application/javascript")


@router.get("/oss-assets/purify.min.js", include_in_schema=False)
async def serve_purify():
    return Response(_purify_js, media_type="application/javascript")



@router.get("/oss-auth/token", include_in_schema=False)
async def get_oss_token(request: Request):
    """Return API key for browser auto-login. OSS single-user only.

    The key is already accessible to any local process via Vault.
    """
    key = getattr(request.app.state, "api_key", None)
    if not key:
        return JSONResponse(status_code=404, content={"error": "not available"})
    return JSONResponse({"token": key})
