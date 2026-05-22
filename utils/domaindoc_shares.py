"""
Domaindoc sharing resolution — single source of truth for resolving domaindocs
across user boundaries.

When a domaindoc is shared, the content stays in the owner's encrypted SQLite.
Collaborators see shared docs with a '_shared' suffix appended to the label
(e.g., owner's "recipes" appears as "recipes_shared" for the collaborator).
This suffix makes resolution deterministic by format — no fallthrough ambiguity.
"""
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from utils.userdata_manager import UserDataManager, get_user_data_manager

logger = logging.getLogger(__name__)

SHARED_SUFFIX = "_shared"


@dataclass(frozen=True)
class ResolvedDomaindoc:
    """Result of resolving a domaindoc label to its backing database and document."""
    db: UserDataManager
    doc: dict[str, Any]
    is_shared: bool
    owner_user_id: UUID | None  # None if owned by current user


@dataclass(frozen=True)
class AcceptedShare:
    """A domaindoc shared with the current user (accepted status only)."""
    owner_user_id: UUID
    domaindoc_label: str  # Owner's original label (without suffix)
    owner_display_name: str  # first_name or email fallback

    @property
    def collaborator_label(self) -> str:
        """The label as the collaborator sees it (with _shared suffix)."""
        return f"{self.domaindoc_label}{SHARED_SUFFIX}"


def _get_pg(user_id: UUID):
    """Get a PostgresClient with user_id set for RLS enforcement."""
    from clients.postgres_client import PostgresClient
    return PostgresClient("mira_service", user_id=str(user_id))


def is_shared_label(label: str) -> bool:
    """Check if a label uses the shared doc suffix."""
    return label.endswith(SHARED_SUFFIX)


def owner_label_from_shared(label: str) -> str:
    """Strip the _shared suffix to get the owner's original label.

    Raises ValueError if the label doesn't end with _shared.
    """
    if not label.endswith(SHARED_SUFFIX):
        raise ValueError(f"Label '{label}' is not a shared doc label")
    return label[:-len(SHARED_SUFFIX)]


def resolve_domaindoc(
    user_id: UUID,
    label: str,
    require_enabled: bool = True
) -> ResolvedDomaindoc:
    """Resolve a domaindoc by label — own docs or shared docs (via _shared suffix).

    Labels without _shared suffix resolve to the user's own SQLite only.
    Labels with _shared suffix strip the suffix and resolve via PostgreSQL shares.

    Raises ValueError if the domaindoc is not found, not enabled, or not available.
    """
    if is_shared_label(label):
        return _resolve_shared(user_id, owner_label_from_shared(label))

    # Own doc — check user's SQLite. db.select already decrypts via _decrypt_dict;
    # calling _decrypt_dict again would try to decrypt plaintext and fail with
    # InvalidToken.
    db = get_user_data_manager(user_id)
    results = db.select("domaindocs", "label = :label", {"label": label})
    if not results:
        raise ValueError(f"Domaindoc '{label}' not found")

    doc = results[0]
    if require_enabled and not doc.get("enabled", True):
        raise ValueError(f"Domaindoc '{label}' is not enabled")
    return ResolvedDomaindoc(db=db, doc=doc, is_shared=False, owner_user_id=None)


def _resolve_shared(user_id: UUID, base_label: str) -> ResolvedDomaindoc:
    """Resolve a shared domaindoc by the owner's original label.

    Always enforces enabled+not-archived — collaborators can't access
    disabled or archived shared docs.
    """
    pg = _get_pg(user_id)
    share = pg.execute_single(
        "SELECT owner_user_id FROM domaindoc_shares "
        "WHERE collaborator_user_id = %(uid)s AND domaindoc_label = %(label)s AND status = 'accepted'",
        {"uid": str(user_id), "label": base_label}
    )
    if not share:
        raise ValueError(f"Domaindoc '{base_label}{SHARED_SUFFIX}' not found")

    owner_user_id = share["owner_user_id"]
    owner_db = get_user_data_manager(owner_user_id)
    # db.select already decrypts; a second _decrypt_dict would fail on plaintext.
    results = owner_db.select("domaindocs", "label = :label", {"label": base_label})
    if not results:
        raise ValueError(f"Shared domaindoc '{base_label}{SHARED_SUFFIX}' not found (owner may have deleted it)")

    doc = results[0]
    if not doc.get("enabled", True) or doc.get("archived", False):
        raise ValueError(f"Shared domaindoc '{base_label}{SHARED_SUFFIX}' is not available")

    return ResolvedDomaindoc(db=owner_db, doc=doc, is_shared=True, owner_user_id=owner_user_id)


def get_accepted_shares(user_id: UUID) -> list[AcceptedShare]:
    """Get all domaindocs shared with a user (accepted status only).

    Returns share metadata. Use share.collaborator_label for the suffixed label
    the collaborator sees, share.domaindoc_label for the owner's original label.
    """
    pg = _get_pg(user_id)
    shares = pg.execute_query(
        "SELECT ds.owner_user_id, ds.domaindoc_label, u.first_name, u.email "
        "FROM domaindoc_shares ds JOIN users u ON ds.owner_user_id = u.id "
        "WHERE ds.collaborator_user_id = %(uid)s AND ds.status = 'accepted'",
        {"uid": str(user_id)}
    )
    return [
        AcceptedShare(
            owner_user_id=s["owner_user_id"],
            domaindoc_label=s["domaindoc_label"],
            owner_display_name=s.get("first_name") or s.get("email", "another user"),
        )
        for s in shares
    ]


def invalidate_domaindoc_cache(user_id: UUID) -> None:
    """Invalidate a user's domaindoc trinket cache in Valkey."""
    from clients.valkey_client import get_valkey_client
    from working_memory.trinkets.base import TRINKET_KEY_PREFIX

    valkey = get_valkey_client()
    valkey.hdel_with_retry(f"{TRINKET_KEY_PREFIX}:{user_id}", "domaindoc")
