"""
Dropbox tool — process files the user has dropped into a local folder.

The user drops files into a configured local directory, then asks MIRA to
process them. The tool exposes three operations:

  - list:    enumerate files currently in the dropbox
  - read:    return extracted text from a document (paginated / truncatable)
  - archive: move a file to dropbox/archive with a JSON sidecar recording
             what was done with it

The tool does not route content itself. Once `read` returns text, the model
composes with other tools — typically `domaindoc_tool` (operation='append')
to file content into a domain doc, or `memory_tool` to create a memory —
then calls `archive` with a `note` describing the disposition.

Supported document formats for `read`:
  text/plain, text/csv, application/json,
  application/vnd.openxmlformats-officedocument.wordprocessingml.document (DOCX),
  application/vnd.openxmlformats-officedocument.spreadsheetml.sheet (XLSX)

PDFs and images can be listed and archived but not read through this tool.
For PDF content, use chat attachment (Claude reads PDFs natively).

Future direction: this remains an inbox-style triage tool for now. A broader
filesystem tool with explicit permissions may come later, but that expansion is
intentionally out of scope for this PR.
"""
import json
import logging
import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from config.config import InboxToolConfig

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)


registry.register("inbox_tool", InboxToolConfig)


# Extension → MIME mapping (authoritative for this tool; mimetypes.guess_type
# is a fallback since its defaults vary by platform).
EXT_TO_MIME = {
    ".txt": "text/plain",
    ".md": "text/plain",
    ".log": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

READABLE_TEXT = {"text/plain", "text/csv", "application/json"}
READABLE_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
READABLE_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class InboxTool(Tool):
    """Local file drop-off dropbox for user-initiated file processing."""

    name = "inbox_tool"

    _parallel_safe_operations = frozenset({"list", "read"})

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") in cls._parallel_safe_operations

    simple_description = (
        "List, read, and archive files the user has placed in their local MIRA dropbox folder."
    )

    tool_schema = {
        "name": "inbox_tool",
        "description": (
            "Operate on files the user has placed in their local MIRA dropbox folder "
            "(a real filesystem directory on the machine running MIRA, NOT email and "
            "NOT cloud storage). Use when the user says things like 'check my dropbox', "
            "'process the files I dropped', or mentions having put a document somewhere "
            "for you to look at. Three operations: `list` (enumerate what's there), "
            "`read` (extract text from a document, paginated via offset+chars), and "
            "`archive` (move the file to dropbox/archive with a note recording what "
            "happened). Typical flow: `list` → `read` the files that matter → route the "
            "content with another tool (e.g., domaindoc_tool 'append' to file into a "
            "domain doc, or memory_tool to create a memory) → `archive` the original "
            "with a short `note` describing the disposition. PDFs and images can be "
            "listed and archived but cannot be `read` through this tool — for PDF "
            "content, have the user attach the PDF directly in chat instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "read", "archive"],
                    "description": "Which operation to run. `list` takes no other params. `read` requires `filename`. `archive` requires `filename` and should include `note`."
                },
                "filename": {
                    "type": "string",
                    "description": "Exact filename (no path) of a file currently at the dropbox root. Obtain valid names from `list`. Path separators and '..' are rejected."
                },
                "chars": {
                    "type": "integer",
                    "description": "For `read`: maximum characters to return in `content`. Default 10000. Use ~2000 to peek, larger to read in full. If the file's extracted text exceeds `offset + chars`, the response sets `truncated: true` and `next_offset` so you can paginate."
                },
                "offset": {
                    "type": "integer",
                    "description": "For `read`: character offset to start from. Default 0. Use the `next_offset` from a prior `read` response to continue."
                },
                "note": {
                    "type": "string",
                    "description": "For `archive`: short (<200 char) record of what was done with the file, written to a .meta.json sidecar next to the archived copy so the archive is self-documenting. Examples: 'appended to finance domaindoc section Q3', 'summarized into memory mem_a1b2c3d4', 'reviewed, no action needed'."
                }
            },
            "required": ["operation"]
        }
    }

    def _cfg(self):
        from config import config
        return config.inbox_tool

    def _inbox_root(self) -> Path:
        cfg = self._cfg()
        p = Path(cfg.inbox_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / cfg.archive_subdir).mkdir(parents=True, exist_ok=True)
        return p

    def _archive_root(self) -> Path:
        return self._inbox_root() / self._cfg().archive_subdir

    def _safe_file(self, filename: str) -> Path:
        if not filename or not isinstance(filename, str):
            raise ValueError("`filename` is required.")
        if "/" in filename or "\\" in filename or filename.startswith(".."):
            raise ValueError(f"Invalid filename {filename!r}: must be a bare filename with no path separators.")
        inbox = self._inbox_root()
        candidate = (inbox / filename).resolve()
        try:
            candidate.relative_to(inbox)
        except ValueError:
            raise ValueError(f"{filename!r} resolves outside the dropbox root.")
        if candidate == self._archive_root():
            raise ValueError("Cannot operate on the archive directory itself.")
        if not candidate.exists():
            raise ValueError(f"No file named {filename!r} in the dropbox. Run `list` to see what's available.")
        if not candidate.is_file():
            raise ValueError(f"{filename!r} is not a regular file.")
        return candidate

    def _guess_mime(self, path: Path) -> str:
        mime = EXT_TO_MIME.get(path.suffix.lower())
        if mime:
            return mime
        guess, _ = mimetypes.guess_type(path.name)
        return guess or "application/octet-stream"

    def _kind(self, mime: str) -> str:
        if mime.startswith("image/"):
            return "image"
        if mime == "application/pdf":
            return "pdf"
        if mime in READABLE_TEXT or mime == READABLE_DOCX or mime == READABLE_XLSX:
            return "document"
        return "other"

    def run(self, **params) -> Dict[str, Any]:
        operation = params.pop("operation", None)
        if not operation:
            return {"error": "`operation` is required. Valid: list, read, archive."}
        try:
            if operation == "list":
                return self._op_list()
            if operation == "read":
                return self._op_read(
                    filename=params.get("filename"),
                    chars=int(params.get("chars") or 10000),
                    offset=int(params.get("offset") or 0),
                )
            if operation == "archive":
                return self._op_archive(
                    filename=params.get("filename"),
                    note=params.get("note"),
                )
            return {"error": f"Unknown operation {operation!r}. Valid: list, read, archive."}
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            self.logger.exception("inbox_tool %s failed", operation)
            return {"error": f"{operation} failed: {e}"}

    def _op_list(self) -> Dict[str, Any]:
        inbox = self._inbox_root()
        archive = self._archive_root()
        entries = []
        for p in sorted(inbox.iterdir()):
            if p == archive or not p.is_file():
                continue
            stat = p.stat()
            mime = self._guess_mime(p)
            kind = self._kind(mime)
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            entries.append({
                "filename": p.name,
                "size_bytes": stat.st_size,
                "size_human": _human_size(stat.st_size),
                "mime": mime,
                "kind": kind,
                "readable": kind == "document",
                "modified": format_utc_iso(mtime, include_ms=False),
            })
        return {
            "dropbox_path": str(inbox),
            "count": len(entries),
            "files": entries,
            "hint": (
                "Use `read` to extract text from documents, then route the content via "
                "domaindoc_tool or memory_tool as appropriate, then `archive` the "
                "original. Images and PDFs can only be listed/archived here — share PDFs "
                "via chat attachment if you need their content."
            ) if entries else "Dropbox is empty.",
        }

    def _op_read(self, filename: str, chars: int, offset: int) -> Dict[str, Any]:
        path = self._safe_file(filename)
        mime = self._guess_mime(path)
        kind = self._kind(mime)
        cfg = self._cfg()

        if kind == "image":
            raise ValueError(
                f"{filename!r} is an image ({mime}). Cannot extract text. Ask the user to "
                f"attach the image directly in chat if they want you to see it."
            )
        if kind == "pdf":
            raise ValueError(
                f"{filename!r} is a PDF. PDF text extraction is not available in this "
                f"build — ask the user to attach the PDF directly in chat (Claude reads "
                f"PDFs natively there), or `archive` this file if no action is needed."
            )
        if kind == "other":
            raise ValueError(
                f"{filename!r} has unsupported type {mime}. Readable types: txt, md, csv, "
                f"json, docx, xlsx. You can still `archive` it."
            )

        max_size_bytes = cfg.max_read_file_size_mb * 1024 * 1024
        file_size = path.stat().st_size
        if file_size > max_size_bytes:
            raise ValueError(
                f"{filename!r} is {_human_size(file_size)}, which exceeds the configured "
                f"read limit of {cfg.max_read_file_size_mb} MB for inbox_tool."
            )

        requested_chars = int(chars or 10000)
        if requested_chars <= 0:
            requested_chars = 10000
        effective_chars = min(requested_chars, cfg.max_read_chars)
        if offset < 0:
            offset = 0

        text = _extract_text(path, mime)
        total = len(text)
        excerpt = text[offset:offset + effective_chars]
        end = offset + len(excerpt)
        truncated = end < total
        result = {
            "filename": filename,
            "mime": mime,
            "total_chars": total,
            "offset": offset,
            "returned_chars": len(excerpt),
            "truncated": truncated,
            "next_offset": end if truncated else None,
            "content": excerpt,
        }
        if effective_chars != requested_chars:
            result["chars_capped"] = True
            result["requested_chars"] = requested_chars
            result["effective_chars"] = effective_chars
        return result

    def _op_archive(self, filename: str, note: Optional[str]) -> Dict[str, Any]:
        path = self._safe_file(filename)
        archive = self._archive_root()
        archive.mkdir(parents=True, exist_ok=True)

        ts = utc_now().strftime("%Y%m%dT%H%M%SZ")
        dest = archive / f"{ts}__{filename}"
        counter = 1
        while dest.exists():
            dest = archive / f"{ts}__{counter}__{filename}"
            counter += 1

        shutil.move(str(path), str(dest))

        mime = self._guess_mime(dest)
        trimmed_note = (note or "").strip() or None
        if trimmed_note and len(trimmed_note) > 500:
            trimmed_note = trimmed_note[:500]

        sidecar = {
            "original_filename": filename,
            "archived_at": format_utc_iso(utc_now()),
            "mime": mime,
            "kind": self._kind(mime),
            "size_bytes": dest.stat().st_size,
            "note": trimmed_note,
        }
        sidecar_path = dest.parent / f"{dest.name}.meta.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2))

        return {
            "archived": filename,
            "archive_location": str(dest),
            "sidecar": str(sidecar_path),
            "note": trimmed_note,
        }


def _extract_text(path: Path, mime: str) -> str:
    if mime in READABLE_TEXT:
        return path.read_text(encoding="utf-8", errors="replace")
    if mime == READABLE_DOCX:
        return _extract_docx(path)
    if mime == READABLE_XLSX:
        return _extract_xlsx(path)
    raise ValueError(f"No text extractor registered for {mime}")


def _extract_docx(path: Path) -> str:
    from utils.document_processing import extract_docx_text

    return extract_docx_text(path.read_bytes())


def _extract_xlsx(path: Path) -> str:
    from utils.document_processing import extract_xlsx_text

    return extract_xlsx_text(path.read_bytes())


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"
