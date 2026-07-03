"""
Domain Document Tool - Section-aware editing with version control.

Provides section-level management for domain knowledge documents stored in SQLite.
Operations are section-scoped with expand/collapse support and full version history.
Supports shared domaindocs via PostgreSQL domaindoc_shares table.
"""
import json
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from pydantic import BaseModel, Field
from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now, format_utc_iso
from utils.userdata_manager import UserDataManager
from utils.domaindoc_shares import resolve_domaindoc, get_accepted_shares, invalidate_domaindoc_cache, ResolvedDomaindoc

if TYPE_CHECKING:
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)


class DomaindocToolConfig(BaseModel):
    """Configuration for the domaindoc tool."""
    enabled: bool = Field(default=True, description="Enable/disable the domaindoc tool")


registry.register("domaindoc_tool", DomaindocToolConfig)


class DomaindocTool(Tool):
    """Section-aware editing tool for domain knowledge documents."""

    name = "domaindoc_tool"

    # Operations without ordering dependencies — safe for concurrent execution.
    # KNOWN BUG: LLMLifecycle runs sequential tools BEFORE parallel ones.
    # If the model issues expand (parallel) + append (sequential) in the same
    # response, append runs first. Doesn't cause data corruption (writes don't
    # check collapsed state) but is semantically inverted from model intent.
    _parallel_safe_operations = frozenset({"search", "expand", "collapse", "overview", "request_create", "request_delete"})

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") in cls._parallel_safe_operations

    simple_description = """Section-aware editing for domain knowledge documents with expand/collapse support."""

    def _build_domaindoc_catalog(self) -> List[str]:
        """Query user's SQLite for all non-archived domaindoc labels, plus shared doc labels.

        Returns empty list on any failure (no user context, no domaindocs table, etc.)
        so the schema is always valid.
        """
        labels: List[str] = []
        try:
            db = self.db
            results = db.fetchall(
                "SELECT label FROM domaindocs WHERE archived = FALSE ORDER BY label"
            )
            labels.extend(r["label"] for r in results)
        except Exception:
            pass

        try:
            shares = get_accepted_shares(self.user_id)
            labels.extend(s.collaborator_label for s in shares)
        except Exception:
            pass

        return sorted(set(labels))

    def _build_schema(self, labels: List[str]) -> Dict[str, Any]:
        """Construct the full tool schema with live domaindoc catalog."""
        if labels:
            catalog_lines = "\n".join(f"- {lbl}" for lbl in labels)
            description = (
                "Manage domain knowledge documents: browse, enable/disable, edit sections.\n\n"
                f"Available domaindocs:\n{catalog_lines}\n\n"
                "Use 'overview' to preview a domaindoc's structure before enabling. Use 'enable'/'disable' "
                "to control which are loaded into context. This tool cannot create or delete domaindocs — "
                "direct the user to the MIRA app UI for lifecycle operations."
            )
        else:
            description = (
                "Manage domain knowledge documents: browse, enable/disable, edit sections.\n\n"
                "No domaindocs available. Direct the user to create one via the MIRA app UI."
            )

        # Build label property — constrain to valid labels when catalog is available
        label_prop: Dict[str, Any] = {
            "type": "string",
            "description": "The domaindoc's label. Optional for search and request_create; required for all other operations"
        }
        if labels:
            label_prop["enum"] = labels

        return {
            "name": "domaindoc_tool",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "overview", "search", "enable", "disable",
                            "expand", "collapse", "set_expanded_by_default",
                            "pin", "unpin",
                            "create_section", "rename_section",
                            "delete_section", "reorder_sections",
                            "append", "sed", "sed_all", "replace_section",
                            "request_create", "request_delete"
                        ],
                        "description": (
                            "Operation to perform. 'overview' previews a domaindoc's structure (works on disabled docs). "
                            "'request_create' is a noop — do NOT call this operation. Instead, when the user asks to create "
                            "a domaindoc, or when you recognize a topic with enough depth to warrant persistent structured "
                            "reference (e.g., the user is deep in an ongoing project, extended summaries show sustained "
                            "engagement with a domain, or the user keeps referencing the same complex information across "
                            "conversations), tell the user directly: domaindocs are created via the MIRA app UI "
                            "(Settings > Domain Documents > Create New). Suggest a label and description for them. "
                            "Do not suggest creating domaindocs for transient topics or one-off questions. "
                            "'request_delete' is a noop — do NOT call this operation. Instead, when the user asks to delete "
                            "a domaindoc, tell them directly: domaindocs are deleted via the MIRA app UI "
                            "(Settings > Domain Documents > [label] > Delete). Suggest 'disable' as a non-destructive "
                            "alternative that removes the domaindoc from context without losing content."
                        )
                    },
                    "label": label_prop,
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive substring to match against section headers and content. Used with search"
                    },
                    "section": {
                        "type": "string",
                        "description": "Section header to operate on (exact match). Add parent for subsections"
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of section headers for batch expand, collapse, or set_expanded_by_default"
                    },
                    "parent": {
                        "type": "string",
                        "description": "Header of the parent section. Required when targeting a subsection or reordering subsections"
                    },
                    "content": {
                        "type": "string",
                        "description": "The section content. Required for create_section and replace_section; adds to existing section content for append"
                    },
                    "find": {
                        "type": "string",
                        "description": "Literal string to match in section content. For sed and sed_all operations"
                    },
                    "replace": {
                        "type": "string",
                        "description": "Literal replacement text for sed/sed_all. Use empty string to delete the matched text"
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New section header name for rename_section"
                    },
                    "insert_after": {
                        "type": "string",
                        "description": "Existing section header — new section is inserted immediately after it. Omit to place at end of the section list"
                    },
                    "expanded_by_default": {
                        "type": "boolean",
                        "description": "If true, section displays expanded by default but can still be collapsed. Used by create_section and set_expanded_by_default"
                    },
                    "order": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Every section header at the level being reordered, in new order. Must include all headers — no omissions or extras"
                    }
                },
                "required": ["operation"]
            }
        }

    @property
    def tool_schema(self) -> Dict[str, Any]:
        """Dynamic schema with live domaindoc catalog.

        Catalog changes only on domaindoc create/delete/archive (rare lifecycle events).
        Enable/disable state is NOT reflected — MIRA infers active state from the
        domaindoc trinket content already in context.
        """
        labels = self._build_domaindoc_catalog()
        return self._build_schema(labels)

    # =========================================================================
    # Database Helpers
    # =========================================================================

    def _normalize_section_name(self, name: str) -> str:
        """Strip ` | alert` suffix that may be included from trinket display."""
        if ' | ' in name:
            return name.split(' | ')[0].strip()
        return name.strip()

    def _get_domaindoc(self, db: UserDataManager, label: str, require_enabled: bool = True) -> Dict[str, Any]:
        """Get domaindoc by label.

        Args:
            db: UserDataManager instance
            label: Domaindoc label to find
            require_enabled: If True, raises ValueError for disabled domaindocs.
                            Set False for enable/disable operations.
        """
        results = db.select("domaindocs", "label = :label", {"label": label})
        if not results:
            raise ValueError(f"Domaindoc '{label}' not found")
        doc = results[0]
        if require_enabled and not doc.get("enabled", True):
            raise ValueError(f"Domaindoc '{label}' is not enabled")
        return doc

    def _get_section(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        header: str,
        parent_header: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get section by header, optionally under a parent. Raises ValueError if not found."""
        normalized = self._normalize_section_name(header)

        if parent_header:
            # Get parent first, then find child under it
            parent = self._get_section(db, domaindoc_id, parent_header)
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header AND parent_section_id = :parent_id",
                {"doc_id": domaindoc_id, "header": normalized, "parent_id": parent["id"]}
            )
            if not results:
                raise ValueError(f"Subsection '{header}' not found under '{parent_header}'")
        else:
            # Top-level section (no parent)
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header AND parent_section_id IS NULL",
                {"doc_id": domaindoc_id, "header": normalized}
            )
            if not results:
                raise ValueError(f"Section '{header}' not found")

        return db._decrypt_dict(results[0])

    def _get_all_sections(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        parent_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get sections for a domaindoc, optionally filtered by parent. Ordered by sort_order."""
        if parent_id is not None:
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND parent_section_id = :parent_id ORDER BY sort_order",
                {"doc_id": domaindoc_id, "parent_id": parent_id}
            )
        else:
            # Get top-level sections only (parent_section_id IS NULL)
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND parent_section_id IS NULL ORDER BY sort_order",
                {"doc_id": domaindoc_id}
            )
        return [db._decrypt_dict(row) for row in results]

    def _get_subsections(self, db: UserDataManager, parent_id: int) -> List[Dict[str, Any]]:
        """Get all subsections of a parent section."""
        results = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE parent_section_id = :parent_id ORDER BY sort_order",
            {"parent_id": parent_id}
        )
        return [db._decrypt_dict(row) for row in results]

    def _count_subsections(self, db: UserDataManager, parent_id: int) -> int:
        """Count subsections of a parent section."""
        result = db.fetchone(
            "SELECT COUNT(*) as count FROM domaindoc_sections WHERE parent_section_id = :parent_id",
            {"parent_id": parent_id}
        )
        return result.get("count", 0) if result else 0

    def _record_version(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        operation: str,
        diff_data: Dict[str, Any],
        section_id: Optional[int] = None
    ) -> int:
        """Record a version entry. Calculates version_num atomically via subquery."""
        now = format_utc_iso(utc_now())

        db.execute(
            """
            INSERT INTO domaindoc_versions
                (domaindoc_id, section_id, version_num, operation, encrypted__diff_data, created_at)
            VALUES (
                :domaindoc_id,
                :section_id,
                (SELECT COALESCE(MAX(version_num), 0) + 1 FROM domaindoc_versions WHERE domaindoc_id = :domaindoc_id),
                :operation,
                :diff_data,
                :now
            )
            """,
            {
                "domaindoc_id": domaindoc_id,
                "section_id": section_id,
                "operation": operation,
                "diff_data": json.dumps(diff_data),
                "now": now
            }
        )

        result = db.fetchone(
            "SELECT MAX(version_num) as ver FROM domaindoc_versions WHERE domaindoc_id = :doc_id",
            {"doc_id": domaindoc_id}
        )
        return result.get("ver", 1)

    def _update_domaindoc_timestamp(self, db: UserDataManager, domaindoc_id: int) -> None:
        """Update the domaindoc's updated_at timestamp."""
        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindocs SET updated_at = :now WHERE id = :doc_id",
            {"now": now, "doc_id": domaindoc_id}
        )

    def _invalidate_shared_doc_caches(self) -> None:
        """Invalidate trinket caches for both collaborator and owner when editing shared docs."""
        if not self._shared_doc_context:
            return
        invalidate_domaindoc_cache(self.user_id)
        owner_user_id = self._shared_doc_context.owner_user_id
        if owner_user_id:
            invalidate_domaindoc_cache(owner_user_id)

    def _actor_suffix(self) -> Dict[str, str]:
        """Return actor info dict for version records when editing shared docs."""
        if not self._shared_doc_context:
            return {}
        return {"actor": "collaborator", "actor_user_id": self.user_id}

    # =========================================================================
    # Tool Interface
    # =========================================================================

    def run(
        self,
        operation: str,
        label: Optional[str] = None,
        query: Optional[str] = None,
        section: Optional[str] = None,
        sections: Optional[List[str]] = None,
        content: Optional[str] = None,
        find: Optional[str] = None,
        replace: Optional[str] = None,
        new_name: Optional[str] = None,
        insert_after: Optional[str] = None,
        order: Optional[List[str]] = None,
        parent: Optional[str] = None,
        expanded_by_default: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Execute an operation on a domaindoc. Use parent param to target subsections."""
        self._shared_doc_context = None
        db = self.db

        if operation == "search":
            return self._op_search(db, query, label)
        elif operation == "enable":
            if not label:
                raise ValueError("enable requires 'label' parameter")
            return self._op_enable(db, label)
        elif operation == "disable":
            if not label:
                raise ValueError("disable requires 'label' parameter")
            return self._op_disable(db, label)
        elif operation == "overview":
            if not label:
                raise ValueError("overview requires 'label' parameter")
            return self._op_overview(label)
        elif operation == "request_create":
            return self._op_request_create(label)
        elif operation == "request_delete":
            if not label:
                raise ValueError("request_delete requires 'label' parameter")
            return self._op_request_delete(label)

        if not label:
            raise ValueError(f"{operation} requires 'label' parameter")

        resolved = resolve_domaindoc(self.user_id, label)
        db = resolved.db
        domaindoc_id = resolved.doc["id"]

        if resolved.is_shared:
            self._shared_doc_context = resolved
            if operation in ("enable", "disable"):
                raise ValueError(f"Cannot {operation} a shared domaindoc — only the owner can manage document lifecycle")

        try:
            if operation == "expand":
                result = self._op_expand(db, domaindoc_id, section, sections, parent)
            elif operation == "collapse":
                result = self._op_collapse(db, domaindoc_id, section, sections, parent)
            elif operation == "set_expanded_by_default":
                result = self._op_set_expanded_by_default(db, domaindoc_id, section, sections, parent, expanded_by_default)
            elif operation == "pin":
                result = self._op_pin(db, domaindoc_id, section, parent)
            elif operation == "unpin":
                result = self._op_unpin(db, domaindoc_id, section, parent)
            elif operation == "create_section":
                result = self._op_create_section(db, domaindoc_id, section, content, insert_after, parent, expanded_by_default)
            elif operation == "rename_section":
                result = self._op_rename_section(db, domaindoc_id, section, new_name, parent)
            elif operation == "delete_section":
                result = self._op_delete_section(db, domaindoc_id, section, parent)
            elif operation == "reorder_sections":
                result = self._op_reorder_sections(db, domaindoc_id, order, parent)
            elif operation == "append":
                result = self._op_append(db, domaindoc_id, section, content, parent)
            elif operation == "sed":
                result = self._op_sed(db, domaindoc_id, section, find, replace, global_replace=False, parent=parent)
            elif operation == "sed_all":
                result = self._op_sed(db, domaindoc_id, section, find, replace, global_replace=True, parent=parent)
            elif operation == "replace_section":
                result = self._op_replace_section(db, domaindoc_id, section, content, parent)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        finally:
            if self._shared_doc_context:
                self._invalidate_shared_doc_caches()
                self._shared_doc_context = None

        return result

    # =========================================================================
    # Document Management Operations
    # =========================================================================

    def _op_search(
        self,
        db: UserDataManager,
        query: Optional[str],
        label: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for content within domaindocs.

        If label is provided, searches only that domaindoc.
        If no label, searches all enabled domaindocs.
        Returns matches with section context and content snippets.
        """
        if not query:
            raise ValueError("search requires 'query' parameter")

        query_lower = query.lower()
        matches: List[Dict[str, Any]] = []

        if label:
            # Search specific domaindoc (must be enabled)
            doc = self._get_domaindoc(db, label)
            docs_to_search = [doc]
        else:
            # Search all enabled, non-archived domaindocs
            docs_to_search = db.fetchall(
                "SELECT * FROM domaindocs WHERE enabled = TRUE AND archived = FALSE"
            )
            docs_to_search = [db._decrypt_dict(d) for d in docs_to_search]

        for doc in docs_to_search:
            domaindoc_id = doc["id"]
            doc_label = doc["label"]

            # Get all sections for this domaindoc (top-level and subsections)
            all_sections = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id ORDER BY parent_section_id NULLS FIRST, sort_order",
                {"doc_id": domaindoc_id}
            )
            all_sections = [db._decrypt_dict(s) for s in all_sections]

            # Build parent lookup for subsection context
            section_by_id = {s["id"]: s for s in all_sections}

            for sec in all_sections:
                content = sec.get("encrypted__content", "") or ""
                header = sec.get("header", "")
                content_lower = content.lower()
                header_lower = header.lower()

                # Check header match
                header_match = query_lower in header_lower
                # Check content match
                content_match = query_lower in content_lower

                if header_match or content_match:
                    parent_header = None
                    parent_id = sec.get("parent_section_id")
                    if parent_id and parent_id in section_by_id:
                        parent_header = section_by_id[parent_id]["header"]

                    match_entry: Dict[str, Any] = {
                        "domaindoc": doc_label,
                        "section": header,
                        "match_in": []
                    }

                    if parent_header:
                        match_entry["parent"] = parent_header

                    if header_match:
                        match_entry["match_in"].append("header")

                    if content_match:
                        match_entry["match_in"].append("content")
                        # Extract snippet around the match
                        idx = content_lower.find(query_lower)
                        start = max(0, idx - 50)
                        end = min(len(content), idx + len(query) + 50)
                        snippet = content[start:end]
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."
                        match_entry["snippet"] = snippet

                    matches.append(match_entry)

        return {
            "success": True,
            "query": query,
            "searched_domaindocs": [d["label"] for d in docs_to_search],
            "matches": matches,
            "total_matches": len(matches)
        }

    def _op_enable(self, db: UserDataManager, label: str) -> Dict[str, Any]:
        """Enable a disabled domaindoc."""
        doc = self._get_domaindoc(db, label, require_enabled=False)

        if doc.get("archived", False):
            raise ValueError(f"Cannot enable archived domaindoc '{label}'. Unarchive it first.")

        if doc.get("enabled", True):
            return {
                "success": True,
                "label": label,
                "enabled": True,
                "message": "Domaindoc was already enabled"
            }

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindocs SET enabled = TRUE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )

        return {
            "success": True,
            "label": label,
            "enabled": True,
            "message": f"Domaindoc '{label}' is now enabled"
        }

    def _op_disable(self, db: UserDataManager, label: str) -> Dict[str, Any]:
        """Disable an enabled domaindoc."""
        doc = self._get_domaindoc(db, label, require_enabled=False)

        if not doc.get("enabled", True):
            return {
                "success": True,
                "label": label,
                "enabled": False,
                "message": "Domaindoc was already disabled"
            }

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindocs SET enabled = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )

        return {
            "success": True,
            "label": label,
            "enabled": False,
            "message": f"Domaindoc '{label}' is now disabled"
        }

    # =========================================================================
    # Browsing & Lifecycle Guidance Operations
    # =========================================================================

    def _op_overview(self, label: str) -> Dict[str, Any]:
        """Return domaindoc description + section tree (headers and summaries, no full content).

        Works on both enabled and disabled domaindocs (own docs only).
        Shared docs always require enabled+not-archived.
        """
        resolved = resolve_domaindoc(self.user_id, label, require_enabled=False)
        db = resolved.db
        domaindoc_id = resolved.doc["id"]

        # Get all sections with parent relationships
        all_sections = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id ORDER BY parent_section_id NULLS FIRST, sort_order",
            {"doc_id": domaindoc_id}
        )
        all_sections = [db._decrypt_dict(s) for s in all_sections]

        # Build section tree: top-level sections with nested subsections
        section_by_id = {s["id"]: s for s in all_sections}
        section_tree: List[Dict[str, Any]] = []

        for sec in all_sections:
            if sec.get("parent_section_id") is not None:
                continue  # Skip subsections — they'll be nested under parents

            entry: Dict[str, Any] = {
                "header": sec["header"],
                "summary": sec.get("encrypted__summary") or "(no summary)",
                "collapsed": sec.get("collapsed", False),
                "pinned": sec.get("pinned", False),
            }

            # Find subsections
            subsections = [
                {
                    "header": sub["header"],
                    "summary": sub.get("encrypted__summary") or "(no summary)",
                    "collapsed": sub.get("collapsed", False),
                }
                for sub in all_sections
                if sub.get("parent_section_id") == sec["id"]
            ]
            if subsections:
                entry["subsections"] = subsections

            section_tree.append(entry)

        return {
            "success": True,
            "label": label,
            "description": doc.get("encrypted__description") or "",
            "enabled": doc.get("enabled", False),
            "section_count": len(all_sections),
            "sections": section_tree
        }

    def _op_request_create(self, label: str | None) -> Dict[str, Any]:
        """Noop fallback — the operation description already tells the LLM what to do."""
        return {
            "success": False,
            "operation": "request_create",
            "error": "You called request_create, but the operation description says not to. Re-read the 'operation' parameter description and relay the directions to the user directly instead of calling this tool."
        }

    def _op_request_delete(self, label: str) -> Dict[str, Any]:
        """Noop fallback — the operation description already tells the LLM what to do."""
        return {
            "success": False,
            "operation": "request_delete",
            "error": "You called request_delete, but the operation description says not to. Re-read the 'operation' parameter description and relay the directions to the user directly instead of calling this tool."
        }

    # =========================================================================
    # Section Management Operations
    # =========================================================================

    def _op_expand(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        sections: Optional[List[str]],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Expand one or more sections. Use parent param for subsections."""
        targets = self._resolve_section_targets(section, sections)
        if not targets:
            raise ValueError("expand requires 'section' or 'sections' parameter")

        expanded = []
        for header in targets:
            sec = self._get_section(db, domaindoc_id, header, parent)
            db.execute(
                "UPDATE domaindoc_sections SET collapsed = FALSE, updated_at = :now WHERE id = :id",
                {"now": format_utc_iso(utc_now()), "id": sec["id"]}
            )
            expanded.append(sec["header"])

        self._update_domaindoc_timestamp(db, domaindoc_id)
        return {"success": True, "expanded": expanded, "parent": parent}

    def _op_collapse(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        sections: Optional[List[str]],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Collapse one or more sections. First top-level section cannot be collapsed."""
        targets = self._resolve_section_targets(section, sections)
        if not targets:
            raise ValueError("collapse requires 'section' or 'sections' parameter")

        collapsed = []
        skipped = []
        for header in targets:
            sec = self._get_section(db, domaindoc_id, header, parent)
            # Pinned sections cannot be collapsed
            if sec.get("pinned"):
                skipped.append(sec["header"])
                continue
            db.execute(
                "UPDATE domaindoc_sections SET collapsed = TRUE, updated_at = :now WHERE id = :id",
                {"now": format_utc_iso(utc_now()), "id": sec["id"]}
            )
            collapsed.append(sec["header"])

        self._update_domaindoc_timestamp(db, domaindoc_id)
        result = {"success": True, "collapsed": collapsed, "parent": parent}
        if skipped:
            result["skipped"] = skipped
            result["note"] = "Pinned and auto-generated sections cannot be collapsed"
        return result

    def _op_set_expanded_by_default(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        sections: Optional[List[str]],
        parent: Optional[str] = None,
        expanded_by_default: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Set expanded_by_default flag on sections. Also expands them if setting to True."""
        targets = self._resolve_section_targets(section, sections)
        if not targets:
            raise ValueError("set_expanded_by_default requires 'section' or 'sections' parameter")
        if expanded_by_default is None:
            raise ValueError("set_expanded_by_default requires 'expanded_by_default' parameter (true/false)")

        updated = []
        skipped = []
        for header in targets:
            sec = self._get_section(db, domaindoc_id, header, parent)
            # Pinned sections are always expanded - skip setting flag
            if sec.get("pinned"):
                skipped.append(sec["header"])
                continue

            # Update flag and also set collapsed state to match
            db.execute(
                "UPDATE domaindoc_sections SET expanded_by_default = :flag, collapsed = :collapsed, updated_at = :now WHERE id = :id",
                {
                    "flag": expanded_by_default,
                    "collapsed": not expanded_by_default,  # expanded_by_default=True means collapsed=False
                    "now": format_utc_iso(utc_now()),
                    "id": sec["id"]
                }
            )
            updated.append(sec["header"])

        self._update_domaindoc_timestamp(db, domaindoc_id)
        result = {"success": True, "updated": updated, "expanded_by_default": expanded_by_default, "parent": parent}
        if skipped:
            result["skipped"] = skipped
            result["note"] = "Pinned sections are always expanded"
        return result

    def _op_pin(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Pin a section so it's always expanded and cannot be collapsed or deleted."""
        if not section:
            raise ValueError("pin requires 'section' parameter")

        sec = self._get_section(db, domaindoc_id, section, parent)

        # Only top-level sections can be pinned
        if sec.get("parent_section_id") is not None:
            raise ValueError("Only top-level sections can be pinned, not subsections")

        if sec.get("pinned"):
            return {"success": True, "pinned": sec["header"], "note": "Section was already pinned"}

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindoc_sections SET pinned = TRUE, collapsed = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": sec["id"]}
        )

        self._record_version(db, domaindoc_id, "pin", {"section": sec["header"], **self._actor_suffix()}, sec["id"])
        self._update_domaindoc_timestamp(db, domaindoc_id)
        return {"success": True, "pinned": sec["header"]}

    def _op_unpin(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Unpin a section so it can be collapsed and deleted."""
        if not section:
            raise ValueError("unpin requires 'section' parameter")

        sec = self._get_section(db, domaindoc_id, section, parent)

        if not sec.get("pinned"):
            return {"success": True, "unpinned": sec["header"], "note": "Section was not pinned"}

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindoc_sections SET pinned = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": sec["id"]}
        )

        self._record_version(db, domaindoc_id, "unpin", {"section": sec["header"], **self._actor_suffix()}, sec["id"])
        self._update_domaindoc_timestamp(db, domaindoc_id)
        return {"success": True, "unpinned": sec["header"]}

    def _op_create_section(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        content: Optional[str],
        insert_after: Optional[str],
        parent: Optional[str] = None,
        expanded_by_default: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Create a new section or subsection. Use parent param to create subsection."""
        if not section:
            raise ValueError("create_section requires 'section' parameter (the new header)")
        if content is None:
            raise ValueError("create_section requires 'content' parameter")

        header = self._normalize_section_name(section)
        now = format_utc_iso(utc_now())
        parent_section_id = None

        if parent:
            # Creating a nested section - validate parent exists and depth limit
            try:
                parent_sec = self._get_section(db, domaindoc_id, parent)
            except ValueError:
                # Parent not found as top-level section - check if it's a subsection
                subsec_check = db.fetchone(
                    "SELECT id, parent_section_id FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header",
                    {"doc_id": domaindoc_id, "header": self._normalize_section_name(parent)}
                )
                if subsec_check and subsec_check.get("parent_section_id") is not None:
                    # Parent is a subsection - check if it's already at depth 2
                    grandparent = db.fetchone(
                        "SELECT parent_section_id FROM domaindoc_sections WHERE id = :id",
                        {"id": subsec_check["parent_section_id"]}
                    )
                    if grandparent and grandparent.get("parent_section_id") is not None:
                        raise ValueError(f"Maximum nesting depth is 2. '{parent}' is already a sub-subsection.")
                    parent_sec = subsec_check  # Use subsection as parent
                else:
                    raise  # Section truly not found

            # Depth check: if parent has a parent, check grandparent depth
            if parent_sec.get("parent_section_id") is not None:
                grandparent = db.fetchone(
                    "SELECT parent_section_id FROM domaindoc_sections WHERE id = :id",
                    {"id": parent_sec["parent_section_id"]}
                )
                if grandparent and grandparent.get("parent_section_id") is not None:
                    raise ValueError(f"Maximum nesting depth is 2. '{parent}' is already a sub-subsection.")

            parent_section_id = parent_sec["id"]
            # Get siblings for ordering
            all_sections = self._get_all_sections(db, domaindoc_id, parent_id=parent_section_id)
        else:
            # Creating top-level section
            all_sections = self._get_all_sections(db, domaindoc_id)

        if insert_after:
            after_sec = self._get_section(db, domaindoc_id, insert_after, parent)
            new_order = after_sec["sort_order"] + 1
            for sec in all_sections:
                if sec["sort_order"] >= new_order:
                    db.execute(
                        "UPDATE domaindoc_sections SET sort_order = sort_order + 1 WHERE id = :id",
                        {"id": sec["id"]}
                    )
        else:
            new_order = max((s["sort_order"] for s in all_sections), default=-1) + 1

        # Check if section already exists (same header at same level)
        existing = db.fetchone(
            """SELECT id FROM domaindoc_sections
               WHERE domaindoc_id = :doc_id AND header = :header
               AND (parent_section_id IS :parent_id OR (parent_section_id IS NULL AND :parent_id IS NULL))""",
            {"doc_id": domaindoc_id, "header": header, "parent_id": parent_section_id}
        )
        if existing:
            raise ValueError(
                f"Section '{header}' already exists. Use 'replace_section' to overwrite content, "
                f"'sed' to edit content, or 'rename_section' to change the header."
            )

        # expanded_by_default sections start expanded; others start collapsed
        start_expanded = expanded_by_default is True
        section_id = db.insert("domaindoc_sections", {
            "domaindoc_id": domaindoc_id,
            "parent_section_id": parent_section_id,
            "header": header,
            "encrypted__content": content,
            "sort_order": new_order,
            "collapsed": not start_expanded,
            "expanded_by_default": start_expanded,
            "created_at": now,
            "updated_at": now
        })

        self._record_version(db, domaindoc_id, "create_section", {
            "header": header,
            "content_length": len(content),
            "insert_after": insert_after,
            "parent": parent,
            "expanded_by_default": start_expanded,
            **self._actor_suffix()
        }, int(section_id))

        self._update_domaindoc_timestamp(db, domaindoc_id)

        # Generate section summary
        from cns.services.domaindoc_summary_service import update_section_summary
        update_section_summary(db, int(section_id), header, content)

        result = {"success": True, "created": header, "sort_order": new_order, "parent": parent}
        if start_expanded:
            result["expanded_by_default"] = True
        return result

    def _op_rename_section(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        new_name: Optional[str],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rename a section header."""
        if not section:
            raise ValueError("rename_section requires 'section' parameter")
        if not new_name:
            raise ValueError("rename_section requires 'new_name' parameter")

        sec = self._get_section(db, domaindoc_id, section, parent)
        old_name = sec["header"]
        normalized_new = self._normalize_section_name(new_name)
        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindoc_sections SET header = :new_name, updated_at = :now WHERE id = :id",
            {"new_name": normalized_new, "now": now, "id": sec["id"]}
        )

        self._record_version(db, domaindoc_id, "rename_section", {
            "old_name": old_name,
            "new_name": normalized_new,
            "parent": parent,
            **self._actor_suffix()
        }, sec["id"])

        self._update_domaindoc_timestamp(db, domaindoc_id)
        return {"success": True, "renamed": old_name, "to": normalized_new, "parent": parent}

    def _op_delete_section(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete a section. Must be expanded first. If parent, all subsections must be expanded."""
        if not section:
            raise ValueError("delete_section requires 'section' parameter")

        sec = self._get_section(db, domaindoc_id, section, parent)

        if sec["collapsed"]:
            raise ValueError(
                f"Please expand '{sec['header']}' before deleting to confirm you've reviewed its contents"
            )

        # Pinned sections cannot be deleted
        if sec.get("pinned"):
            raise ValueError(f"Cannot delete pinned section '{sec['header']}'. Unpin it first.")

        # If this is a parent with subsections, all subsections must be expanded
        subsections = self._get_subsections(db, sec["id"])
        if subsections:
            collapsed_subs = [s["header"] for s in subsections if s.get("collapsed")]
            if collapsed_subs:
                raise ValueError(
                    f"Please expand all subsections of '{sec['header']}' before deleting: {collapsed_subs}"
                )

        deleted_children = []
        if subsections:
            for sub in subsections:
                self._record_version(db, domaindoc_id, "delete_section", {
                    "header": sub["header"],
                    "deleted_content": sub.get("encrypted__content", ""),
                    "sort_order": sub["sort_order"],
                    "parent": sec["header"],
                    **self._actor_suffix()
                }, sub["id"])
                deleted_children.append(sub["header"])

        self._record_version(db, domaindoc_id, "delete_section", {
            "header": sec["header"],
            "deleted_content": sec.get("encrypted__content", ""),
            "sort_order": sec["sort_order"],
            "parent": parent,
            "deleted_children": deleted_children,
            **self._actor_suffix()
        }, sec["id"])

        db.execute(
            "DELETE FROM domaindoc_sections WHERE id = :id",
            {"id": sec["id"]}
        )

        # Renumber siblings
        parent_id = sec.get("parent_section_id")
        siblings = self._get_all_sections(db, domaindoc_id, parent_id=parent_id) if parent_id else self._get_all_sections(db, domaindoc_id)
        for i, s in enumerate(siblings):
            if s["sort_order"] != i:
                db.execute(
                    "UPDATE domaindoc_sections SET sort_order = :order WHERE id = :id",
                    {"order": i, "id": s["id"]}
                )

        self._update_domaindoc_timestamp(db, domaindoc_id)

        result = {"success": True, "deleted": sec["header"], "parent": parent}
        if deleted_children:
            result["deleted_children"] = deleted_children
        return result

    def _op_reorder_sections(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        order: Optional[List[str]],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reorder sections at a given level. Use parent to reorder subsections."""
        if not order:
            raise ValueError("reorder_sections requires 'order' parameter")

        if parent:
            parent_sec = self._get_section(db, domaindoc_id, parent)
            all_sections = self._get_all_sections(db, domaindoc_id, parent_id=parent_sec["id"])
        else:
            all_sections = self._get_all_sections(db, domaindoc_id)

        existing_headers = {s["header"] for s in all_sections}
        provided_headers = {self._normalize_section_name(h) for h in order}

        missing = existing_headers - provided_headers
        unknown = provided_headers - existing_headers

        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"missing sections {list(missing)}")
            if unknown:
                parts.append(f"unknown sections {list(unknown)}")
            raise ValueError(f"Reorder failed: {' and '.join(parts)}")

        now = format_utc_iso(utc_now())
        for new_order, header in enumerate(order):
            normalized = self._normalize_section_name(header)
            sec = next(s for s in all_sections if s["header"] == normalized)
            db.execute(
                "UPDATE domaindoc_sections SET sort_order = :order, updated_at = :now WHERE id = :id",
                {"order": new_order, "now": now, "id": sec["id"]}
            )

        self._record_version(db, domaindoc_id, "reorder_sections", {"order": order, "parent": parent, **self._actor_suffix()})
        self._update_domaindoc_timestamp(db, domaindoc_id)
        return {"success": True, "new_order": order, "parent": parent}

    # =========================================================================
    # Content Editing Operations
    # =========================================================================

    def _op_append(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        content: Optional[str],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Append content to a section or subsection."""
        if not section:
            raise ValueError("append requires 'section' parameter")
        if not content:
            raise ValueError("append requires 'content' parameter")

        sec = self._get_section(db, domaindoc_id, section, parent)
        current = sec.get("encrypted__content", "")
        if current and not current.endswith('\n'):
            current += '\n'
        new_content = current + content
        now = format_utc_iso(utc_now())

        db.update(
            "domaindoc_sections",
            {"encrypted__content": new_content, "updated_at": now},
            "id = :id",
            {"id": sec["id"]}
        )

        self._record_version(db, domaindoc_id, "append", {
            "section": sec["header"],
            "appended_content": content,
            "result_length": len(new_content),
            "parent": parent,
            **self._actor_suffix()
        }, sec["id"])

        self._update_domaindoc_timestamp(db, domaindoc_id)

        # Update section summary
        from cns.services.domaindoc_summary_service import update_section_summary
        update_section_summary(db, sec["id"], sec["header"], new_content)

        return {
            "success": True,
            "section": sec["header"],
            "appended_chars": len(content),
            "total_chars": len(new_content)
        }

    def _op_sed(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        find: Optional[str],
        replace: Optional[str],
        global_replace: bool,
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Replace text in a section or subsection."""
        if not section:
            raise ValueError("sed requires 'section' parameter")
        if not find:
            raise ValueError("sed requires 'find' parameter")
        if replace is None:
            raise ValueError("sed requires 'replace' parameter")

        sec = self._get_section(db, domaindoc_id, section, parent)
        current = sec.get("encrypted__content", "")

        if global_replace:
            new_content = current.replace(find, replace)
            count = current.count(find)
        else:
            new_content = current.replace(find, replace, 1)
            count = 1 if find in current else 0

        if count == 0:
            return {
                "success": False,
                "section": sec["header"],
                "message": f"Pattern '{find}' not found in section",
                "parent": parent
            }

        now = format_utc_iso(utc_now())
        db.update(
            "domaindoc_sections",
            {"encrypted__content": new_content, "updated_at": now},
            "id = :id",
            {"id": sec["id"]}
        )

        op_name = "sed_all" if global_replace else "sed"
        self._record_version(db, domaindoc_id, op_name, {
            "section": sec["header"],
            "find": find,
            "replace": replace,
            "replacements": count,
            "parent": parent,
            **self._actor_suffix()
        }, sec["id"])

        self._update_domaindoc_timestamp(db, domaindoc_id)

        # Update section summary
        from cns.services.domaindoc_summary_service import update_section_summary
        update_section_summary(db, sec["id"], sec["header"], new_content)

        return {
            "success": True,
            "section": sec["header"],
            "replacements": count,
            "total_chars": len(new_content),
            "parent": parent
        }

    def _op_replace_section(
        self,
        db: UserDataManager,
        domaindoc_id: int,
        section: Optional[str],
        content: Optional[str],
        parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Replace entire section or subsection content."""
        if not section:
            raise ValueError("replace_section requires 'section' parameter")
        if content is None:
            raise ValueError("replace_section requires 'content' parameter")

        sec = self._get_section(db, domaindoc_id, section, parent)
        previous_content = sec.get("encrypted__content", "")  # Capture BEFORE modification
        now = format_utc_iso(utc_now())

        db.update(
            "domaindoc_sections",
            {"encrypted__content": content, "updated_at": now},
            "id = :id",
            {"id": sec["id"]}
        )

        self._record_version(db, domaindoc_id, "replace_section", {
            "section": sec["header"],
            "old_length": len(previous_content),
            "new_length": len(content),
            "previous_content": previous_content,
            "parent": parent,
            **self._actor_suffix()
        }, sec["id"])

        self._update_domaindoc_timestamp(db, domaindoc_id)

        # Update section summary
        from cns.services.domaindoc_summary_service import update_section_summary
        update_section_summary(db, sec["id"], sec["header"], content)

        return {
            "success": True,
            "section": sec["header"],
            "previous_chars": len(previous_content),
            "new_chars": len(content),
            "parent": parent
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _resolve_section_targets(
        self,
        section: Optional[str],
        sections: Optional[List[str]]
    ) -> List[str]:
        """Resolve section or sections parameter to list of headers."""
        if sections:
            return [self._normalize_section_name(s) for s in sections]
        elif section:
            return [self._normalize_section_name(section)]
        return []
