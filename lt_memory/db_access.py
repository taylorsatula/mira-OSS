"""
Database access layer for LT_Memory system.

Single source of truth for all database operations. Returns Pydantic models
for type safety. Uses raw SQL for performance and clarity.
"""
import logging
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from uuid import UUID

from lt_memory.models import (
    Memory,
    ExtractedMemory,
    MemoryLink,
    ExtractionBatch,
    PostProcessingBatch,
    RefinementCandidate
)
from utils.timezone_utils import utc_now, format_utc_iso
from utils.user_context import get_current_user_id
from utils.database_session_manager import LTMemorySessionManager

logger = logging.getLogger(__name__)


def _load_scoring_formula() -> str:
    """
    Load scoring formula SQL from dedicated file.

    The formula is stored in lt_memory/scoring_formula.sql for easy viewing
    and version control. This function is called once at module initialization.

    Returns:
        SQL expression for importance_score calculation
    """
    formula_path = Path(__file__).parent / 'scoring_formula.sql'
    with open(formula_path, 'r') as f:
        return f.read().strip()


# Load scoring formula once at module initialization (zero file I/O after this)
_SCORING_FORMULA_SQL = _load_scoring_formula()


class LTMemoryDB:
    """
    Database gateway for LT_Memory operations.

    Provides type-safe database access with Pydantic model returns.
    All operations support both ambient user context and explicit user_id parameters.
    """

    def __init__(self, session_manager: LTMemorySessionManager):
        """
        Initialize database gateway.

        Args:
            session_manager: Session manager for database connections
        """
        self.session_manager = session_manager

    def _resolve_user_id(self, user_id: Optional[str] = None) -> str:
        """
        Resolve effective user_id from explicit parameter or ambient context.

        Args:
            user_id: Explicit user ID (takes precedence)

        Returns:
            Resolved user ID

        Raises:
            ValueError: If no user_id available from either source
            RuntimeError: If ambient context lookup fails when user_id is None
        """
        if user_id is not None:
            return str(user_id)  # Convert UUID to string if needed

        # Only attempt ambient context lookup when user_id truly absent
        try:
            return get_current_user_id()
        except RuntimeError:
            raise ValueError(
                "No user_id provided and no ambient user context available. "
                "Scheduled tasks must pass explicit user_id."
            )

    @contextmanager
    def transaction(self, user_id: Optional[str] = None):
        """
        Provide transaction context for multi-step operations.

        Usage:
            with db.transaction(user_id) as session:
                # Multiple operations in single transaction
                pass

        Args:
            user_id: User ID for session context

        Yields:
            Database session with active transaction
        """
        resolved_user_id = self._resolve_user_id(user_id)
        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                yield session

    # ==================== MEMORY CRUD ====================

    def store_memories(
        self,
        memories: List[ExtractedMemory],
        embeddings: Optional[List[List[float]]] = None,
        user_id: Optional[str] = None
    ) -> List[UUID]:
        """
        Bulk insert extracted memories with optional embeddings.

        Args:
            memories: List of ExtractedMemory objects
            embeddings: Optional list of embedding vectors (must match length of memories)
            user_id: User ID (uses ambient context if None)

        Returns:
            List of created memory UUIDs

        Raises:
            ValueError: If embeddings provided but length doesn't match memories
        """
        if not memories:
            return []

        if embeddings is not None and len(embeddings) != len(memories):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) must match memories length ({len(memories)})"
            )

        resolved_user_id = self._resolve_user_id(user_id)

        # Capture current activity days for snapshots (vacation-proof scoring)
        from utils.user_context import get_user_cumulative_activity_days
        current_activity_days = get_user_cumulative_activity_days()

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                created_ids = []

                for idx, memory in enumerate(memories):
                    insert_sql = """
                    INSERT INTO memories (
                        user_id, text, embedding, importance_score,
                        expires_at, happens_at, created_at,
                        confidence, is_refined, last_refined_at,
                        activity_days_at_creation, activity_days_at_last_access
                    ) VALUES (
                        %(user_id)s, %(text)s, %(embedding)s, %(importance_score)s,
                        %(expires_at)s, %(happens_at)s, %(created_at)s,
                        %(confidence)s, %(is_refined)s, %(last_refined_at)s,
                        %(activity_days_at_creation)s, %(activity_days_at_last_access)s
                    ) RETURNING id
                    """

                    insert_data = {
                        'user_id': resolved_user_id,
                        'text': memory.text,
                        'embedding': embeddings[idx] if embeddings else None,
                        'importance_score': memory.importance_score,
                        'expires_at': memory.expires_at,
                        'happens_at': memory.happens_at,
                        'created_at': utc_now(),
                        'confidence': memory.confidence,
                        'is_refined': False,
                        'last_refined_at': None,
                        'activity_days_at_creation': current_activity_days,
                        'activity_days_at_last_access': current_activity_days
                    }

                    result = session.execute_single(insert_sql, insert_data)
                    if result:
                        created_ids.append(result['id'])

                logger.info(f"Created {len(created_ids)} memories for user {resolved_user_id}")
                return created_ids

    def get_memory(
        self,
        memory_id: UUID,
        user_id: Optional[str] = None
    ) -> Optional[Memory]:
        """
        Fetch single memory by ID.

        Args:
            memory_id: Memory UUID
            user_id: User ID (uses ambient context if None)

        Returns:
            Memory model or None if not found
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM memories
            WHERE id = %(memory_id)s
            LIMIT 1
            """

            result = session.execute_single(query, {'memory_id': str(memory_id)})

            if not result:
                return None

            return Memory(**result)

    def get_memories_by_ids(
        self,
        memory_ids: List[UUID],
        user_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Fetch multiple memories by IDs.

        Args:
            memory_ids: List of memory UUIDs
            user_id: User ID (uses ambient context if None)

        Returns:
            List of Memory models (may be fewer than requested if some not found)
        """
        if not memory_ids:
            return []

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM memories
            WHERE id = ANY(%(memory_ids)s::uuid[])
            ORDER BY importance_score DESC
            """

            results = session.execute_query(query, {
                'memory_ids': [str(mid) for mid in memory_ids]
            })

            return [Memory(**row) for row in results]

    def update_memory(
        self,
        memory_id: UUID,
        updates: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Memory:
        """
        Update memory fields.

        Args:
            memory_id: Memory UUID
            updates: Dictionary of field -> value updates
            user_id: User ID (uses ambient context if None)

        Returns:
            Updated Memory model

        Raises:
            ValueError: If memory not found
        """
        resolved_user_id = self._resolve_user_id(user_id)

        # Build SET clause dynamically
        set_clauses = []
        params = {'memory_id': str(memory_id)}

        for field, value in updates.items():
            param_name = f'update_{field}'
            set_clauses.append(f"{field} = %({param_name})s")
            params[param_name] = value

        # Always update updated_at
        set_clauses.append("updated_at = NOW()")

        set_clause = ", ".join(set_clauses)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                query = f"""
                UPDATE memories
                SET {set_clause}
                WHERE id = %(memory_id)s
                RETURNING *
                """

                result = session.execute_single(query, params)

                if not result:
                    raise ValueError(f"Memory {memory_id} not found")

                return Memory(**result)

    def archive_memory(
        self,
        memory_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Archive a memory (soft delete).

        Args:
            memory_id: Memory UUID
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            session.execute_update("""
                UPDATE memories
                SET is_archived = TRUE,
                    archived_at = NOW(),
                    updated_at = NOW()
                WHERE id = %(memory_id)s
            """, {'memory_id': str(memory_id)})

        # Clean up dead links while we're here
        self.remove_dead_links([memory_id], user_id=resolved_user_id)

        logger.info(f"Archived memory {memory_id} and cleaned up dead links")

    # ==================== SEARCH & RETRIEVAL ====================

    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        min_importance: float = 0.1,
        user_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Vector similarity search using cosine distance.

        Args:
            query_embedding: Query vector (384d AllMiniLM)
            limit: Maximum results to return
            similarity_threshold: Minimum cosine similarity (0-1)
            min_importance: Minimum importance score filter
            user_id: User ID (uses ambient context if None)

        Returns:
            List of Memory models sorted by similarity
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT m.*,
                   1 - (m.embedding <=> %(query_embedding)s::vector) as similarity_score
            FROM memories m
            WHERE m.importance_score >= %(min_importance)s
              AND (m.expires_at IS NULL OR m.expires_at > NOW())
              AND m.is_archived = FALSE
              AND 1 - (m.embedding <=> %(query_embedding)s::vector) >= %(similarity_threshold)s
            ORDER BY m.embedding <=> %(query_embedding)s::vector
            LIMIT %(limit)s
            """

            results = session.execute_query(query, {
                'query_embedding': query_embedding,
                'limit': limit,
                'similarity_threshold': similarity_threshold,
                'min_importance': min_importance
            })

            return [Memory(**row) for row in results]

    def get_all_memories(
        self,
        include_archived: bool = False,
        offset: int = 0,
        limit: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Union[List[Memory], Dict[str, Any]]:
        """
        Fetch memories for a user with optional pagination.

        Args:
            include_archived: Whether to include archived memories
            offset: Number of records to skip (pagination)
            limit: Maximum number of records to return (None = all records)
            user_id: User ID (uses ambient context if None)

        Returns:
            If limit is None: List[Memory] (backward compatible)
            If limit is provided: Dict with 'memories', 'has_more', 'next_offset'
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            # Build base query
            if include_archived:
                base_query = "SELECT * FROM memories ORDER BY created_at DESC"
            else:
                base_query = """
                SELECT * FROM memories
                WHERE is_archived = FALSE
                ORDER BY created_at DESC
                """

            # Add pagination if limit provided
            if limit is not None:
                # Fetch limit + 1 to check if more results exist
                query = f"{base_query} LIMIT %(limit)s OFFSET %(offset)s"
                results = session.execute_query(query, {
                    'limit': limit + 1,
                    'offset': offset
                })

                # Check if more results exist
                has_more = len(results) > limit
                memories = [Memory(**row) for row in results[:limit]]

                return {
                    'memories': [m.model_dump() for m in memories],
                    'has_more': has_more,
                    'next_offset': offset + limit if has_more else None
                }
            else:
                # No pagination - return all memories as list (backward compatible)
                results = session.execute_query(base_query, {})
                return [Memory(**row) for row in results]

    # ==================== SCORING OPERATIONS ====================

    def _recalculate_importance_scores(
        self,
        memory_ids: List[UUID],
        session
    ) -> int:
        """
        Recalculate importance scores for given memories using activity-based decay formula.

        The formula is loaded from lt_memory/scoring_formula.sql (see that file for
        complete documentation of the scoring algorithm).

        Args:
            memory_ids: List of memory UUIDs to recalculate
            session: Database session

        Returns:
            Number of memories updated
        """
        if not memory_ids:
            return 0

        # Build UPDATE query using formula from scoring_formula.sql
        recalc_query = f"""
        UPDATE memories m
        SET
            importance_score = {_SCORING_FORMULA_SQL},
            updated_at = NOW()
        FROM users u
        WHERE m.id = ANY(%(memory_ids)s::uuid[])
          AND m.user_id = u.id
        """

        result = session.execute_update(recalc_query, {
            'memory_ids': [str(mid) for mid in memory_ids]
        })

        return result if result else 0

    def update_access_stats(
        self,
        memory_id: UUID,
        user_id: Optional[str] = None
    ) -> Memory:
        """
        Record memory access and recalculate importance score.

        Args:
            memory_id: Memory UUID
            user_id: User ID (uses ambient context if None)

        Returns:
            Updated Memory model

        Raises:
            ValueError: If memory not found
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                # Step 1: Update access statistics
                access_update = """
                UPDATE memories m
                SET
                    access_count = access_count + 1,
                    last_accessed = NOW(),
                    activity_days_at_last_access = u.cumulative_activity_days
                FROM users u
                WHERE m.id = %(memory_id)s
                  AND m.user_id = u.id
                RETURNING m.id
                """

                result = session.execute_single(access_update, {
                    'memory_id': str(memory_id)
                })

                if not result:
                    raise ValueError(f"Memory {memory_id} not found")

                # Step 2: Recalculate importance score using shared formula
                self._recalculate_importance_scores([memory_id], session)

                # Step 3: Fetch and return updated memory
                return self.get_memory(memory_id, user_id=resolved_user_id)

    def bulk_recalculate_scores(
        self,
        user_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> int:
        """
        Recalculate importance scores for stale memories.

        Used for periodic maintenance - recalculates scores for memories
        that haven't been accessed in 7+ days using the shared scoring formula.

        Args:
            user_id: User ID (uses ambient context if None)
            batch_size: Number of memories to process

        Returns:
            Number of memories updated
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            # Get stale memories
            stale_query = """
            SELECT id FROM memories
            WHERE (last_accessed < NOW() - INTERVAL '7 days' OR last_accessed IS NULL)
              AND importance_score > 0.001
              AND is_archived = FALSE
            LIMIT %(batch_size)s
            """

            stale_ids = [row['id'] for row in session.execute_query(stale_query, {
                'batch_size': batch_size
            })]

            if not stale_ids:
                return 0

            # Recalculate scores using shared formula
            updated_count = self._recalculate_importance_scores(stale_ids, session)

            # Archive memories that fell below threshold
            archive_query = """
            UPDATE memories
            SET is_archived = TRUE,
                archived_at = NOW()
            WHERE id = ANY(%(memory_ids)s::uuid[])
              AND importance_score <= 0.001
              AND is_archived = FALSE
            RETURNING id
            """

            archived = session.execute_query(archive_query, {
                'memory_ids': [str(mid) for mid in stale_ids]
            })
            archived_count = len(archived)

            logger.info(
                f"Bulk recalculated {updated_count} memories, archived {archived_count}"
            )

            return updated_count

    def recalculate_temporal_scores(
        self,
        user_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> int:
        """
        Recalculate importance scores for memories with temporal fields.

        Ensures memories with happens_at or expires_at dates get score updates
        even if not accessed. Focuses on temporal windows where multipliers matter:
        - 30 days before event (a month to build up importance)
        - 7 days after event (a week of decay before dropping off)

        Args:
            user_id: User ID (uses ambient context if None)
            batch_size: Maximum memories to process per call

        Returns:
            Number of memories updated
        """
        resolved_user_id = self._resolve_user_id(user_id)

        # Get current activity days for scoring
        from utils.user_context import get_user_cumulative_activity_days
        current_activity_days = get_user_cumulative_activity_days()

        with self.session_manager.get_session(resolved_user_id) as session:
            # Query temporal memories within relevant windows
            temporal_query = """
            SELECT id FROM memories
            WHERE is_archived = FALSE
              AND (
                (happens_at IS NOT NULL
                 AND happens_at BETWEEN NOW() - INTERVAL '7 days' AND NOW() + INTERVAL '30 days')
                OR
                (expires_at IS NOT NULL
                 AND expires_at BETWEEN NOW() - INTERVAL '7 days' AND NOW() + INTERVAL '30 days')
              )
            LIMIT %(batch_size)s
            """

            temporal_ids = [row['id'] for row in session.execute_query(
                temporal_query,
                {'batch_size': batch_size}
            )]

            if not temporal_ids:
                logger.debug("No temporal memories in relevant windows")
                return 0

            # Recalculate scores using shared formula
            updated_count = self._recalculate_importance_scores(temporal_ids, session)

            # Archive memories that fell below threshold or expired
            archive_query = """
            UPDATE memories
            SET is_archived = TRUE,
                archived_at = NOW()
            WHERE id = ANY(%(memory_ids)s::uuid[])
              AND importance_score <= 0.001
              AND is_archived = FALSE
            RETURNING id
            """

            archived = session.execute_query(archive_query, {
                'memory_ids': [str(mid) for mid in temporal_ids]
            })
            archived_count = len(archived)

            logger.info(
                f"Temporal recalculation: updated {updated_count} memories, "
                f"archived {archived_count}"
            )

            return updated_count

    # ==================== LINK OPERATIONS ====================

    def create_links(
        self,
        links: List[MemoryLink],
        user_id: Optional[str] = None
    ) -> None:
        """
        Create bidirectional links between memories.

        Links are stored in JSONB arrays (inbound_links, outbound_links)
        on each memory for efficient hub score calculation.

        Args:
            links: List of MemoryLink objects
            user_id: User ID (uses ambient context if None)
        """
        if not links:
            return

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                for link in links:
                    # Create outbound link object for source
                    outbound_obj = {
                        'uuid': str(link.target_id),
                        'type': link.link_type,
                        'confidence': link.confidence,
                        'reasoning': link.reasoning,
                        'created_at': format_utc_iso(link.created_at)
                    }

                    # Create inbound link object for target
                    inbound_obj = {
                        'uuid': str(link.source_id),
                        'type': link.link_type,
                        'confidence': link.confidence,
                        'reasoning': link.reasoning,
                        'created_at': format_utc_iso(link.created_at)
                    }

                    # Add outbound link to source (only if not already present)
                    session.execute_update("""
                        UPDATE memories
                        SET outbound_links = CASE
                            WHEN NOT EXISTS (
                                SELECT 1 FROM jsonb_array_elements(COALESCE(outbound_links, '[]'::jsonb)) AS elem
                                WHERE elem->>'uuid' = %(target_id)s
                                  AND elem->>'type' = %(link_type)s
                            )
                            THEN COALESCE(outbound_links, '[]'::jsonb) || %(outbound_obj)s::jsonb
                            ELSE outbound_links
                        END,
                        updated_at = NOW()
                        WHERE id = %(source_id)s
                          AND is_archived = FALSE
                    """, {
                        'source_id': str(link.source_id),
                        'target_id': str(link.target_id),
                        'link_type': link.link_type,
                        'outbound_obj': json.dumps(outbound_obj)
                    })

                    # Add inbound link to target (only if not already present)
                    session.execute_update("""
                        UPDATE memories
                        SET inbound_links = CASE
                            WHEN NOT EXISTS (
                                SELECT 1 FROM jsonb_array_elements(COALESCE(inbound_links, '[]'::jsonb)) AS elem
                                WHERE elem->>'uuid' = %(source_id)s
                                  AND elem->>'type' = %(link_type)s
                            )
                            THEN COALESCE(inbound_links, '[]'::jsonb) || %(inbound_obj)s::jsonb
                            ELSE inbound_links
                        END,
                        updated_at = NOW()
                        WHERE id = %(target_id)s
                          AND is_archived = FALSE
                    """, {
                        'target_id': str(link.target_id),
                        'source_id': str(link.source_id),
                        'link_type': link.link_type,
                        'inbound_obj': json.dumps(inbound_obj)
                    })

                logger.info(f"Created {len(links)} bidirectional links")

    def get_links_for_memory(
        self,
        memory_id: UUID,
        user_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all links for a memory.

        Args:
            memory_id: Memory UUID
            user_id: User ID (uses ambient context if None)

        Returns:
            Dictionary with 'inbound' and 'outbound' link lists
        """
        memory = self.get_memory(memory_id, user_id)

        if not memory:
            return {'inbound': [], 'outbound': []}

        return {
            'inbound': memory.inbound_links,
            'outbound': memory.outbound_links
        }

    def remove_dead_links(
        self,
        dead_uuids: List[UUID],
        user_id: Optional[str] = None
    ) -> int:
        """
        Remove dead UUIDs from all memory link arrays.

        LAZY CLEANUP PATTERN: This method should be called opportunistically
        when dead links are detected during traversal, not proactively.

        Example usage:
            # During link traversal, when UUIDs don't return memories:
            dead_links = [uuid for uuid in requested_ids if uuid not in found_ids]
            if dead_links:
                self.remove_dead_links(dead_links)

        This approach is more efficient than proactive database scans,
        as it only fixes problems when encountered during normal operations.

        FUTURE ENHANCEMENT: A weekly scheduled cleanup job scanning for dead links
        would tighten hub score accuracy, but the current lazy approach provides
        acceptable variance (±1-2 phantom links per memory). Weekly cleanup is
        recommended but not critical for system operation.

        Args:
            dead_uuids: List of dead memory UUIDs to remove from link arrays
            user_id: User ID (uses ambient context if None)

        Returns:
            Number of memories updated
        """
        if not dead_uuids:
            return 0

        resolved_user_id = self._resolve_user_id(user_id)
        dead_uuid_strs = [str(uuid) for uuid in dead_uuids]

        with self.session_manager.get_session(resolved_user_id) as session:
            # Build JSONB filter to remove dead UUIDs
            query = """
            UPDATE memories
            SET
                inbound_links = (
                    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                    FROM jsonb_array_elements(COALESCE(inbound_links, '[]'::jsonb)) AS elem
                    WHERE elem->>'uuid' != ALL(%(dead_uuids)s)
                ),
                outbound_links = (
                    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                    FROM jsonb_array_elements(COALESCE(outbound_links, '[]'::jsonb)) AS elem
                    WHERE elem->>'uuid' != ALL(%(dead_uuids)s)
                ),
                updated_at = NOW()
            WHERE EXISTS (
                SELECT 1 FROM jsonb_array_elements(COALESCE(inbound_links, '[]'::jsonb)) AS elem
                WHERE elem->>'uuid' = ANY(%(dead_uuids)s)
            ) OR EXISTS (
                SELECT 1 FROM jsonb_array_elements(COALESCE(outbound_links, '[]'::jsonb)) AS elem
                WHERE elem->>'uuid' = ANY(%(dead_uuids)s)
            )
            """

            updated_count = session.execute_update(query, {'dead_uuids': dead_uuid_strs})

            if updated_count > 0:
                logger.info(f"Removed {len(dead_uuids)} dead links from {updated_count} memories")

            return updated_count

    def increment_refinement_rejection_count(
        self,
        memory_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Increment rejection count for memory marked do_nothing during refinement.

        After 3 rejections, the memory will be excluded from future refinement
        candidates (system has decided it's fine as-is).

        Args:
            memory_id: Memory UUID to update
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            session.execute_update(
                """
                UPDATE memories
                SET refinement_rejection_count = refinement_rejection_count + 1,
                    updated_at = NOW()
                WHERE id = %(memory_id)s
                """,
                {"memory_id": str(memory_id)}
            )

        logger.debug(f"Incremented refinement rejection count for memory {memory_id}")

    # ==================== ENTITY OPERATIONS ====================

    def get_or_create_entity(
        self,
        name: str,
        entity_type: str,
        embedding: List[float],
        user_id: Optional[str] = None
    ) -> 'Entity':
        """
        Get existing entity or create new one.

        Uses ON CONFLICT to handle race conditions elegantly.

        Args:
            name: Canonical normalized entity name
            entity_type: spaCy NER type (PERSON, ORG, GPE, etc.)
            embedding: spaCy word vector (300d)
            user_id: User ID (uses ambient context if None)

        Returns:
            Entity model (existing or newly created)
        """
        from lt_memory.models import Entity

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            # Try insert, return existing if conflict
            query = """
            INSERT INTO entities (
                user_id, name, entity_type, embedding, created_at
            ) VALUES (
                %(user_id)s, %(name)s, %(entity_type)s, %(embedding)s::vector, NOW()
            )
            ON CONFLICT (user_id, name, entity_type)
            DO UPDATE SET updated_at = NOW()
            RETURNING *
            """

            result = session.execute_single(query, {
                'user_id': resolved_user_id,
                'name': name,
                'entity_type': entity_type,
                'embedding': embedding
            })

            if not result:
                raise RuntimeError(
                    f"INSERT...RETURNING query failed to return entity for {name} ({entity_type}) - "
                    f"database operation failed (infrastructure issue, not 'entity not found')"
                )

            return Entity(**result)

    def get_entity(
        self,
        entity_id: UUID,
        user_id: Optional[str] = None
    ) -> Optional['Entity']:
        """
        Fetch entity by ID.

        Args:
            entity_id: Entity UUID
            user_id: User ID (uses ambient context if None)

        Returns:
            Entity model or None if not found
        """
        from lt_memory.models import Entity

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM entities
            WHERE id = %(entity_id)s
            LIMIT 1
            """

            result = session.execute_single(query, {'entity_id': str(entity_id)})
            return Entity(**result) if result else None

    def get_entities_by_ids(
        self,
        entity_ids: List[UUID],
        user_id: Optional[str] = None
    ) -> List['Entity']:
        """
        Fetch multiple entities by IDs.

        Args:
            entity_ids: List of entity UUIDs
            user_id: User ID (uses ambient context if None)

        Returns:
            List of Entity models
        """
        from lt_memory.models import Entity

        if not entity_ids:
            return []

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM entities
            WHERE id = ANY(%(entity_ids)s::uuid[])
            ORDER BY link_count DESC
            """

            results = session.execute_query(query, {
                'entity_ids': [str(eid) for eid in entity_ids]
            })

            return [Entity(**row) for row in results]

    def link_memory_to_entity(
        self,
        memory_id: UUID,
        entity_id: UUID,
        entity_name: str,
        entity_type: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Link memory to entity.

        Adds entity to memory's entity_links JSONB array and updates
        entity's link_count and last_linked_at timestamp.

        Args:
            memory_id: Memory UUID
            entity_id: Entity UUID
            entity_name: Entity name for JSONB storage
            entity_type: Entity type for JSONB storage
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                # Add entity to memory's entity_links
                entity_link_obj = {
                    'uuid': str(entity_id),
                    'type': entity_type,
                    'name': entity_name
                }

                session.execute_update("""
                    UPDATE memories
                    SET entity_links = COALESCE(entity_links, '[]'::jsonb) || %(entity_obj)s::jsonb,
                        updated_at = NOW()
                    WHERE id = %(memory_id)s
                """, {
                    'memory_id': str(memory_id),
                    'entity_obj': json.dumps(entity_link_obj)
                })

                # Update entity link_count and last_linked_at
                session.execute_update("""
                    UPDATE entities
                    SET link_count = link_count + 1,
                        last_linked_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %(entity_id)s
                """, {
                    'entity_id': str(entity_id)
                })

    def find_dormant_entities(
        self,
        days_dormant: int,
        min_link_count: int,
        max_link_count: int,
        user_id: Optional[str] = None
    ) -> List['Entity']:
        """
        Find dormant entities for garbage collection.

        Identifies entities that haven't gained new links in N days,
        with link counts in the specified range.

        Args:
            days_dormant: Days since last link
            min_link_count: Minimum link count (inclusive)
            max_link_count: Maximum link count (inclusive)
            user_id: User ID (uses ambient context if None)

        Returns:
            List of dormant Entity models
        """
        from lt_memory.models import Entity

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM entities
            WHERE link_count >= %(min_link_count)s
              AND link_count <= %(max_link_count)s
              AND (last_linked_at IS NULL OR last_linked_at < NOW() - INTERVAL '1 day' * %(days)s)
              AND is_archived = FALSE
            ORDER BY link_count ASC, last_linked_at ASC NULLS FIRST
            """

            results = session.execute_query(query, {
                'min_link_count': min_link_count,
                'max_link_count': max_link_count,
                'days': days_dormant
            })

            return [Entity(**row) for row in results]

    def find_entities_by_vector_similarity(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        user_id: Optional[str] = None
    ) -> List['Entity']:
        """
        Find similar entities by vector similarity.

        Uses cosine distance for semantic similarity search.

        Args:
            query_embedding: Query vector (300d spaCy)
            limit: Maximum results
            similarity_threshold: Minimum cosine similarity (0-1)
            user_id: User ID (uses ambient context if None)

        Returns:
            List of Entity models with similarity_score set
        """
        from lt_memory.models import Entity

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT e.*,
                   1 - (e.embedding <=> %(query_embedding)s::vector) as similarity_score
            FROM entities e
            WHERE e.is_archived = FALSE
              AND 1 - (e.embedding <=> %(query_embedding)s::vector) >= %(similarity_threshold)s
            ORDER BY e.embedding <=> %(query_embedding)s::vector
            LIMIT %(limit)s
            """

            results = session.execute_query(query, {
                'query_embedding': query_embedding,
                'limit': limit,
                'similarity_threshold': similarity_threshold
            })

            return [Entity(**row) for row in results]

    def get_memories_for_entity(
        self,
        entity_id: UUID,
        user_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Get all memories linking to an entity.

        Queries memories.entity_links JSONB array.

        Args:
            entity_id: Entity UUID
            user_id: User ID (uses ambient context if None)

        Returns:
            List of Memory models
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM memories
            WHERE entity_links @> %s::jsonb
              AND is_archived = FALSE
            ORDER BY created_at DESC
            """

            # JSONB containment query
            entity_filter = json.dumps([{"uuid": str(entity_id)}])

            results = session.execute_query(query, (entity_filter,))
            return [Memory(**row) for row in results]

    def merge_entities(
        self,
        source_id: UUID,
        target_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Merge source entity into target entity.

        Updates all memories linking to source to point to target instead,
        updates target's link_count, and archives source entity.

        Args:
            source_id: Source entity UUID (will be archived)
            target_id: Target entity UUID (will receive links)
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                # Get target entity info for replacement
                target = self.get_entity(target_id, user_id=resolved_user_id)
                if not target:
                    raise ValueError(f"Target entity {target_id} not found")

                # Update memories: replace source UUID with target UUID in entity_links
                update_query = """
                UPDATE memories
                SET entity_links = (
                    SELECT jsonb_agg(
                        CASE
                            WHEN elem->>'uuid' = %(source_id)s
                            THEN jsonb_set(elem, '{uuid}', %(target_id_json)s)
                                 || jsonb_set(elem, '{name}', %(target_name_json)s)
                                 || jsonb_set(elem, '{type}', %(target_type_json)s)
                            ELSE elem
                        END
                    )
                    FROM jsonb_array_elements(entity_links) AS elem
                ),
                updated_at = NOW()
                WHERE entity_links @> %(source_filter)s::jsonb
                """

                affected_count = session.execute_update(update_query, {
                    'source_id': str(source_id),
                    'target_id_json': json.dumps(str(target_id)),
                    'target_name_json': json.dumps(target.name),
                    'target_type_json': json.dumps(target.entity_type),
                    'source_filter': json.dumps([{"uuid": str(source_id)}])
                })

                # Update target entity link_count
                session.execute_update("""
                    UPDATE entities
                    SET link_count = link_count + %(affected_count)s,
                        last_linked_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %(target_id)s
                """, {
                    'affected_count': affected_count,
                    'target_id': str(target_id)
                })

                # Archive source entity
                session.execute_update("""
                    UPDATE entities
                    SET is_archived = TRUE,
                        archived_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %(source_id)s
                """, {
                    'source_id': str(source_id)
                })

                logger.info(
                    f"Merged entity {source_id} into {target_id}: "
                    f"updated {affected_count} memories"
                )

    def delete_entity(
        self,
        entity_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Delete entity and remove from all memory entity_links.

        Args:
            entity_id: Entity UUID to delete
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                # Remove entity from memory entity_links
                remove_query = """
                UPDATE memories
                SET entity_links = (
                    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                    FROM jsonb_array_elements(entity_links) AS elem
                    WHERE elem->>'uuid' != %(entity_id)s
                ),
                updated_at = NOW()
                WHERE entity_links @> %(entity_filter)s::jsonb
                """

                session.execute_update(remove_query, {
                    'entity_id': str(entity_id),
                    'entity_filter': json.dumps([{"uuid": str(entity_id)}])
                })

                # Delete entity record
                session.execute_update("""
                    DELETE FROM entities
                    WHERE id = %(entity_id)s
                """, {
                    'entity_id': str(entity_id)
                })

                logger.info(f"Deleted entity {entity_id}")

    def archive_entity(
        self,
        entity_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Archive entity (soft delete).

        Args:
            entity_id: Entity UUID
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            session.execute_update("""
                UPDATE entities
                SET is_archived = TRUE,
                    archived_at = NOW(),
                    updated_at = NOW()
                WHERE id = %(entity_id)s
            """, {'entity_id': str(entity_id)})

        logger.info(f"Archived entity {entity_id}")

    # ==================== BATCH TRACKING ====================

    def create_extraction_batch(
        self,
        batch: ExtractionBatch,
        user_id: Optional[str] = None
    ) -> UUID:
        """
        Create extraction batch tracking record.

        Args:
            batch: ExtractionBatch model
            user_id: User ID (uses ambient context if None)

        Returns:
            Created batch UUID
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                insert_sql = """
                INSERT INTO extraction_batches (
                    batch_id, custom_id, user_id, chunk_index,
                    request_payload, chunk_metadata, memory_context,
                    status, created_at, submitted_at, expires_at
                ) VALUES (
                    %(batch_id)s, %(custom_id)s, %(user_id)s, %(chunk_index)s,
                    %(request_payload)s, %(chunk_metadata)s, %(memory_context)s,
                    %(status)s, %(created_at)s, %(submitted_at)s, %(expires_at)s
                ) RETURNING id
                """

                result = session.execute_single(insert_sql, {
                    'batch_id': batch.batch_id,
                    'custom_id': batch.custom_id,
                    'user_id': resolved_user_id,
                    'chunk_index': batch.chunk_index,
                    'request_payload': json.dumps(batch.request_payload),
                    'chunk_metadata': json.dumps(batch.chunk_metadata) if batch.chunk_metadata else None,
                    'memory_context': json.dumps(batch.memory_context) if batch.memory_context else None,
                    'status': batch.status,
                    'created_at': batch.created_at,
                    'submitted_at': batch.submitted_at,
                    'expires_at': batch.expires_at
                })

                return result['id']

    def get_users_with_pending_extraction_batches(self) -> List[str]:
        """
        Get list of user IDs with pending extraction batches.

        Uses admin session to query across users without bypassing security.
        Returns user IDs that can then be used for scoped batch queries.

        Returns:
            List of user_id strings with batches in submitted/processing state
        """
        with self.session_manager.get_admin_session() as session:
            query = """
            SELECT DISTINCT user_id
            FROM extraction_batches
            WHERE status IN ('submitted', 'processing')
            ORDER BY user_id
            """
            results = session.execute_query(query, {})
            return [str(row['user_id']) for row in results]

    def get_pending_relationship_batches_for_user(
        self,
        user_id: str
    ) -> List[PostProcessingBatch]:
        """
        Get pending relationship batches for specific user.

        Args:
            user_id: User ID to check

        Returns:
            List of PostProcessingBatch models with status 'submitted' or 'processing'
        """
        with self.session_manager.get_session(user_id) as session:
            query = """
            SELECT * FROM post_processing_batches
            WHERE user_id = %(user_id)s
            AND status IN ('submitted', 'processing')
            ORDER BY created_at ASC
            """

            results = session.execute_query(query, {'user_id': user_id})
            return [PostProcessingBatch(**row) for row in results]

    def get_pending_extraction_batches_for_user(
        self,
        user_id: str
    ) -> List[ExtractionBatch]:
        """
        Get pending extraction batches for specific user.

        Args:
            user_id: User ID to check

        Returns:
            List of ExtractionBatch models with status 'submitted' or 'processing'
        """
        with self.session_manager.get_session(user_id) as session:
            query = """
            SELECT * FROM extraction_batches
            WHERE user_id = %(user_id)s
            AND status IN ('submitted', 'processing')
            ORDER BY created_at ASC
            """

            results = session.execute_query(query, {'user_id': user_id})
            return [ExtractionBatch(**row) for row in results]

    def update_extraction_batch_status(
        self,
        batch_id: UUID,
        status: str,
        error_message: Optional[str] = None,
        extracted_memories: Optional[List[Dict[str, Any]]] = None,
        completed_at: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Update extraction batch status with detailed result tracking.

        Args:
            batch_id: Batch UUID
            status: New status (submitted/processing/completed/failed/expired/cancelled)
            error_message: Optional error message for failed batches
            extracted_memories: Optional list of extracted memory dicts
            completed_at: Optional completion timestamp
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            updates = {'status': status}

            if error_message:
                updates['error_message'] = error_message

            if extracted_memories:
                updates['extracted_memories'] = json.dumps(extracted_memories)

            if completed_at:
                updates['completed_at'] = completed_at

            set_clauses = [f"{k} = %({k})s" for k in updates.keys()]
            set_clause = ", ".join(set_clauses)

            query = f"""
            UPDATE extraction_batches
            SET {set_clause}
            WHERE id = %(batch_id)s
            """

            updates['batch_id'] = str(batch_id)
            session.execute_update(query, updates)

    def increment_extraction_batch_retry(
        self,
        batch_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Increment retry counter for extraction batch.

        Used to track processing attempts and prevent infinite retry loops.

        Args:
            batch_id: Batch UUID
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            UPDATE extraction_batches
            SET retry_count = retry_count + 1
            WHERE id = %(batch_id)s
            """
            session.execute_update(query, {'batch_id': str(batch_id)})

    def delete_extraction_batch(
        self,
        batch_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Delete extraction batch record after processing complete.

        Args:
            batch_id: Batch UUID to delete
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                query = """
                DELETE FROM extraction_batches
                WHERE id = %(batch_id)s
                """
                session.execute_update(query, {'batch_id': str(batch_id)})

    def cleanup_old_extraction_batches(
        self,
        retention_hours: int,
        user_id: Optional[str] = None
    ) -> int:
        """
        Delete extraction batches in terminal states older than retention period.

        Terminal states: failed, expired, cancelled

        Args:
            retention_hours: Hours to retain terminal-state batches
            user_id: User ID (uses ambient context if None)

        Returns:
            Number of batches deleted
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                query = """
                DELETE FROM extraction_batches
                WHERE user_id = %(user_id)s
                  AND status IN ('failed', 'expired', 'cancelled')
                  AND created_at < NOW() - INTERVAL '%(retention_hours)s hours'
                """
                result = session.execute_update(query, {
                    'user_id': resolved_user_id,
                    'retention_hours': retention_hours
                })
                return result

    # ==================== RELATIONSHIP BATCH TRACKING ====================

    def create_relationship_batch(
        self,
        batch: PostProcessingBatch,
        user_id: Optional[str] = None
    ) -> UUID:
        """
        Create relationship classification batch tracking record.

        Args:
            batch: PostProcessingBatch model
            user_id: User ID (uses ambient context if None)

        Returns:
            Created batch UUID
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                insert_sql = """
                INSERT INTO post_processing_batches (
                    batch_id, batch_type, user_id, request_payload,
                    input_data, items_submitted, status, created_at,
                    submitted_at, expires_at
                ) VALUES (
                    %(batch_id)s, %(batch_type)s, %(user_id)s, %(request_payload)s,
                    %(input_data)s, %(items_submitted)s, %(status)s, %(created_at)s,
                    %(submitted_at)s, %(expires_at)s
                ) RETURNING id
                """

                result = session.execute_single(insert_sql, {
                    'batch_id': batch.batch_id,
                    'batch_type': batch.batch_type,
                    'user_id': resolved_user_id,
                    'request_payload': json.dumps(batch.request_payload),
                    'input_data': json.dumps(batch.input_data) if batch.input_data else None,
                    'items_submitted': batch.items_submitted,
                    'status': batch.status,
                    'created_at': batch.created_at,
                    'submitted_at': batch.submitted_at,
                    'expires_at': batch.expires_at
                })

                return result['id']

    def get_users_with_pending_relationship_batches(self) -> List[str]:
        """
        Get list of user IDs with pending relationship batches.

        Uses admin session to query across users without bypassing security.
        Returns user IDs that can then be used for scoped batch queries.

        Returns:
            List of user_id strings with batches in submitted/processing state
        """
        with self.session_manager.get_admin_session() as session:
            query = """
            SELECT DISTINCT user_id
            FROM post_processing_batches
            WHERE status IN ('submitted', 'processing')
            ORDER BY user_id
            """
            results = session.execute_query(query, {})
            return [str(row['user_id']) for row in results]

    def update_relationship_batch_status(
        self,
        batch_id: UUID,
        status: str,
        error_message: Optional[str] = None,
        items_completed: Optional[int] = None,
        links_created: Optional[int] = None,
        conflicts_flagged: Optional[int] = None,
        completed_at: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Update relationship batch status with detailed metrics.

        Args:
            batch_id: Batch UUID
            status: New status
            error_message: Optional error message
            items_completed: Number of items processed
            links_created: Number of links created
            conflicts_flagged: Number of conflicts detected
            completed_at: Completion timestamp
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            updates = {'status': status}

            if error_message:
                updates['error_message'] = error_message
            if items_completed is not None:
                updates['items_completed'] = items_completed
            if links_created is not None:
                updates['links_created'] = links_created
            if conflicts_flagged is not None:
                updates['conflicts_flagged'] = conflicts_flagged
            if completed_at:
                updates['completed_at'] = completed_at

            set_clauses = [f"{k} = %({k})s" for k in updates.keys()]
            set_clause = ", ".join(set_clauses)

            query = f"""
            UPDATE post_processing_batches
            SET {set_clause}
            WHERE id = %(batch_id)s
            """

            updates['batch_id'] = str(batch_id)
            session.execute_update(query, updates)

    def increment_relationship_batch_retry(
        self,
        batch_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Increment retry counter for relationship batch.

        Used to track processing attempts and prevent infinite retry loops.

        Args:
            batch_id: Batch UUID
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            UPDATE post_processing_batches
            SET retry_count = retry_count + 1
            WHERE id = %(batch_id)s
            """
            session.execute_update(query, {'batch_id': str(batch_id)})

    def delete_relationship_batch(
        self,
        batch_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Delete relationship batch record after processing complete.

        Args:
            batch_id: Batch UUID to delete
            user_id: User ID (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                query = """
                DELETE FROM post_processing_batches
                WHERE id = %(batch_id)s
                """
                session.execute_update(query, {'batch_id': str(batch_id)})

    def cleanup_old_relationship_batches(
        self,
        retention_hours: int,
        user_id: Optional[str] = None
    ) -> int:
        """
        Delete relationship batches in terminal states older than retention period.

        Terminal states: failed, expired, cancelled

        Args:
            retention_hours: Hours to retain terminal-state batches
            user_id: User ID (uses ambient context if None)

        Returns:
            Number of batches deleted
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                query = """
                DELETE FROM post_processing_batches
                WHERE user_id = %(user_id)s
                  AND status IN ('failed', 'expired', 'cancelled')
                  AND created_at < NOW() - INTERVAL '%(retention_hours)s hours'
                """
                result = session.execute_update(query, {
                    'user_id': resolved_user_id,
                    'retention_hours': retention_hours
                })
                return result

    def get_extraction_timestamp(self, user_id: Optional[str] = None) -> Optional[datetime]:
        """
        Get the last extraction timestamp for incremental processing.

        Args:
            user_id: User ID (uses ambient context if None)

        Returns:
            Last extraction timestamp or None if never run
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_admin_session() as session:
            result = session.execute_single("""
                SELECT daily_manipulation_last_run
                FROM users
                WHERE id = %(user_id)s
            """, {'user_id': resolved_user_id})

            return result['daily_manipulation_last_run'] if result else None

    def update_extraction_timestamp(self, user_id: Optional[str] = None) -> None:
        """
        Update the last extraction timestamp for incremental processing.

        This timestamp tracks when extraction last ran so the system knows
        which messages are "new" for incremental extraction.

        Args:
            user_id: User ID to update (uses ambient context if None)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_admin_session() as session:
            session.execute_update("""
                UPDATE users
                SET daily_manipulation_last_run = %(timestamp)s
                WHERE id = %(user_id)s
            """, {'timestamp': utc_now(), 'user_id': resolved_user_id})

    def get_users_with_memory_enabled(self) -> List[Dict[str, Any]]:
        """
        Get all users with memory extraction enabled.

        Returns:
            List of user dictionaries with id, email, and memory settings
        """
        with self.session_manager.get_admin_session() as session:
            return session.execute_query("""
                SELECT id, email, memory_manipulation_enabled, daily_manipulation_last_run, timezone
                FROM users
                WHERE memory_manipulation_enabled = TRUE
                AND is_active = TRUE
            """)

    def cleanup(self):
        """
        Clean up database resources.

        No-op: Session manager is shared singleton, managed separately.
        Nulling reference breaks in-flight scheduler jobs.
        """
        logger.debug("LTMemoryDB cleanup completed (no-op)")
