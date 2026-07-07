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
from typing import List, Dict, Any, Optional
from uuid import UUID

from lt_memory.models import (
    Memory,
    ExtractedMemory,
    MemoryLink,
    ExtractionBatch,
    BatchStatus,
    MemoryLinkEntry,
    UserMemorySettings,
    MemoryPageResult,
    EntityPairRow,
)

from utils.timezone_utils import utc_now, format_utc_iso
from utils.user_context import get_current_user_id
from utils.tag_parser import parse_memory_id
from utils.database_session_manager import LTMemorySessionManager, LTMemorySession

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

        if embeddings is None:
            embeddings = [None] * len(memories)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                params: Dict[str, Any] = {
                    'user_id': resolved_user_id,
                    'created_at': utc_now(),
                    'activity_days_at_creation': current_activity_days,
                    'activity_days_at_last_access': current_activity_days,
                }
                values_clauses = []

                for idx, memory in enumerate(memories):
                    values_clauses.append(
                        f"(%(user_id)s, %(text_{idx})s, %(embedding_{idx})s, "
                        f"%(importance_score_{idx})s, %(expires_at_{idx})s, "
                        f"%(happens_at_{idx})s, %(created_at)s, "
                        f"%(activity_days_at_creation)s, %(activity_days_at_last_access)s, "
                        f"%(source_segment_id_{idx})s)"
                    )
                    params[f'text_{idx}'] = memory.text
                    params[f'embedding_{idx}'] = embeddings[idx]
                    params[f'importance_score_{idx}'] = memory.importance_score
                    params[f'expires_at_{idx}'] = memory.expires_at
                    params[f'happens_at_{idx}'] = memory.happens_at
                    params[f'source_segment_id_{idx}'] = (
                        str(memory.source_segment_id) if memory.source_segment_id else None
                    )

                insert_sql = f"""
                INSERT INTO memories (
                    user_id, text, embedding, importance_score,
                    expires_at, happens_at, created_at,
                    activity_days_at_creation, activity_days_at_last_access,
                    source_segment_id
                ) VALUES {', '.join(values_clauses)}
                RETURNING id
                """

                results = session.execute_query(insert_sql, params)
                created_ids = [row['id'] for row in results]

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

            result = session.execute_single(query, {'memory_id': memory_id})

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
            WHERE id = ANY(%s::uuid[])
            ORDER BY importance_score DESC
            """

            results = session.execute_query(query, (
                list(memory_ids),
            ))

            return [Memory(**row) for row in results]

    def get_memories_by_segment_id(
        self,
        segment_id: UUID,
        user_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Fetch all memories extracted from a specific segment.

        Args:
            segment_id: Source segment UUID (message ID of segment boundary)
            user_id: User ID (uses ambient context if None)

        Returns:
            List of Memory models ordered by creation time
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM memories
            WHERE source_segment_id = %(segment_id)s
            ORDER BY created_at ASC
            """

            results = session.execute_query(query, {'segment_id': segment_id})
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
        params = {'memory_id': memory_id}

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
            """, {'memory_id': memory_id})

        # Clean up dead links while we're here
        self.remove_dead_links([memory_id], user_id=resolved_user_id)

        logger.info(f"Archived memory {memory_id} and cleaned up dead links")

    # ==================== CURATOR (last_tended_at) ====================

    def update_last_tended(
        self,
        memory_ids: List[UUID],
        user_id: Optional[str] = None
    ) -> None:
        """
        Mark memories as tended by the curator at the current time.

        Called by MemoryCuratorAgent.on_completion after a curation run so the
        floor trigger's 'random among unseen' sampling can exclude recently-
        tended memories. RLS-scoped via the session manager.

        Args:
            memory_ids: Memory UUIDs tended in this run
            user_id: User ID (uses ambient context if None)
        """
        if not memory_ids:
            return

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            session.execute_update("""
                UPDATE memories
                SET last_tended_at = NOW()
                WHERE id = ANY(%(memory_ids)s::uuid[])
            """, {'memory_ids': list(memory_ids)})

        logger.info(f"Updated last_tended_at for {len(memory_ids)} memories")

    def get_floor_candidates(
        self,
        floor_threshold: float,
        unseen_days: int,
        sample_size: int,
        user_id: Optional[str] = None
    ) -> List[UUID]:
        """
        Random sample of low-value memories not tended recently.

        The floor trigger's bounded surface: deterministic, no LLM. A memory is
        eligible when it is below the importance threshold AND either it was
        tended before the cutoff OR it was never tended (NULL) but was created
        before the cutoff (integration never ran for it). Random ordering caps
        cost by giving coverage without a full sweep. RLS-scoped via the
        session manager.

        Args:
            floor_threshold: importance_score strictly below this value
            unseen_days: wall-clock staleness window (days)
            sample_size: maximum number of memories to return
            user_id: User ID (uses ambient context if None)

        Returns:
            Up to sample_size memory UUIDs, randomly ordered.
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
                SELECT id
                FROM memories
                WHERE is_archived = FALSE
                  AND importance_score < %(floor_threshold)s
                  AND (
                        last_tended_at < NOW() - make_interval(days => %(unseen_days)s)
                     OR (last_tended_at IS NULL
                         AND created_at < NOW() - make_interval(days => %(unseen_days)s))
                  )
                ORDER BY random()
                LIMIT %(sample_size)s
            """
            results = session.execute_query(query, {
                'floor_threshold': floor_threshold,
                'unseen_days': unseen_days,
                'sample_size': sample_size,
            })

            return [row['id'] for row in results]

    # ==================== SEARCH & RETRIEVAL ====================

    # TODO(taylor): The UNION queries below were written during a period when
    # Anthropic lobotomized Opus 4.5. Revisit this code when Opus is at full
    # brainpower - there may be a cleaner approach (e.g., a compatibility VIEW
    # for global_memories that adds default columns, allowing SELECT * in UNION).

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

        Searches both personal memories (RLS-filtered) and global memories (no RLS)
        via UNION. Results are tagged with source='personal' or source='global'.

        Args:
            query_embedding: Query vector (768d mdbr-leaf-ir-asym)
            limit: Maximum results to return
            similarity_threshold: Minimum cosine similarity (0-1)
            min_importance: Minimum importance score filter (personal only)
            user_id: User ID (uses ambient context if None)

        Returns:
            List of Memory models sorted by similarity, with source attribute set
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            (
                SELECT m.id, m.user_id, m.text, m.embedding, m.importance_score,
                       m.created_at, m.updated_at, m.expires_at, m.access_count,
                       m.mention_count, m.last_accessed, m.happens_at,
                       m.inbound_links, m.outbound_links, m.entity_links,
                       m.is_archived, m.archived_at,
                       m.activity_days_at_creation, m.activity_days_at_last_access,
                       m.annotations, m.source_segment_id,
                       1 - (m.embedding <=> %(query_embedding)s::vector) as similarity_score,
                       'personal' as source
                FROM memories m
                WHERE m.importance_score >= %(min_importance)s
                  AND (m.expires_at IS NULL OR m.expires_at > NOW())
                  AND m.is_archived = FALSE
                  AND m.embedding IS NOT NULL
                  AND 1 - (m.embedding <=> %(query_embedding)s::vector) >= %(similarity_threshold)s
            )
            UNION ALL
            (
                SELECT gm.id, NULL::uuid as user_id, gm.text, gm.embedding, gm.importance_score,
                       gm.created_at, gm.updated_at, NULL::timestamptz as expires_at, 0 as access_count,
                       0 as mention_count, NULL::timestamptz as last_accessed, gm.happens_at,
                       gm.inbound_links, gm.outbound_links, gm.entity_links,
                       gm.is_archived, gm.archived_at,
                       NULL::int as activity_days_at_creation, NULL::int as activity_days_at_last_access,
                       '[]'::jsonb as annotations, NULL::uuid as source_segment_id,
                       1 - (gm.embedding <=> %(query_embedding)s::vector) as similarity_score,
                       'global' as source
                FROM global_memories gm
                WHERE gm.is_archived = FALSE
                  AND gm.embedding IS NOT NULL
                  AND 1 - (gm.embedding <=> %(query_embedding)s::vector) >= %(similarity_threshold)s
            )
            ORDER BY similarity_score DESC
            LIMIT %(limit)s
            """

            results = session.execute_query(query, {
                'query_embedding': query_embedding,
                'limit': limit,
                'similarity_threshold': similarity_threshold,
                'min_importance': min_importance
            })

            memories = []
            for row in results:
                source = row.pop('source', 'personal')
                memory = Memory(**row)
                memory.source = source
                memories.append(memory)

            return memories

    def get_all_memories(
        self,
        include_archived: bool = False,
        user_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Fetch all memories for a user.

        Args:
            include_archived: Whether to include archived memories
            user_id: User ID (uses ambient context if None)

        Returns:
            List of all Memory objects
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            if include_archived:
                query = "SELECT * FROM memories ORDER BY created_at DESC"
            else:
                query = """
                SELECT * FROM memories
                WHERE is_archived = FALSE
                ORDER BY created_at DESC
                """
            results = session.execute_query(query, {})
            return [Memory(**row) for row in results]

    def get_memories_paginated(
        self,
        limit: int,
        offset: int = 0,
        include_archived: bool = False,
        user_id: Optional[str] = None
    ) -> MemoryPageResult:
        """
        Fetch memories with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            include_archived: Whether to include archived memories
            user_id: User ID (uses ambient context if None)

        Returns:
            MemoryPageResult with memories, has_more, next_offset
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            if include_archived:
                base_query = "SELECT * FROM memories ORDER BY created_at DESC"
            else:
                base_query = """
                SELECT * FROM memories
                WHERE is_archived = FALSE
                ORDER BY created_at DESC
                """

            # Fetch limit + 1 to check if more results exist
            query = f"{base_query} LIMIT %(limit)s OFFSET %(offset)s"
            results = session.execute_query(query, {
                'limit': limit + 1,
                'offset': offset
            })

            has_more = len(results) > limit
            memories = [Memory(**row) for row in results[:limit]]

            return {
                'memories': [m.model_dump() for m in memories],
                'has_more': has_more,
                'next_offset': offset + limit if has_more else None
            }

    # ==================== SCORING OPERATIONS ====================

    def _recalculate_importance_scores(
        self,
        memory_ids: List[UUID],
        session: LTMemorySession
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
        # Use positional param %s with tuple for PostgreSQL array type
        recalc_query = f"""
        UPDATE memories m
        SET
            importance_score = {_SCORING_FORMULA_SQL},
            updated_at = NOW()
        FROM users u
        WHERE m.id = ANY(%s::uuid[])
          AND m.user_id = u.id
        """

        result = session.execute_update(recalc_query, (
            list(memory_ids),
        ))

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
                    'memory_id': memory_id
                })

                if not result:
                    raise ValueError(f"Memory {memory_id} not found")

                # Step 2: Recalculate importance score using shared formula
                self._recalculate_importance_scores([memory_id], session)

                # Step 3: Fetch and return updated memory
                result = self.get_memory(memory_id, user_id=resolved_user_id)
                if not result:
                    raise RuntimeError(
                        f"Memory {memory_id} disappeared between UPDATE and SELECT"
                    )
                return result

    def apply_pin_boost(
        self,
        short_ids: List[str],
        user_id: Optional[str] = None
    ) -> int:
        if not short_ids:
            return 0

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                # Strip mem_ prefix if present before building LIKE patterns
                like_patterns = [f"{parse_memory_id(sid).lower()}%" for sid in short_ids]

                update_query = """
                UPDATE memories m
                SET
                    access_count = access_count + 1,
                    last_accessed = NOW(),
                    activity_days_at_last_access = u.cumulative_activity_days
                FROM users u
                WHERE m.user_id = u.id
                  AND (""" + " OR ".join([
                    f"REPLACE(m.id::text, '-', '') LIKE %(pattern_{i})s"
                    for i in range(len(like_patterns))
                ]) + """)
                RETURNING m.id
                """

                params = {f'pattern_{i}': p for i, p in enumerate(like_patterns)}
                result = session.execute_query(update_query, params)
                updated_count = len(result)

                if updated_count > 0:
                    memory_ids = [row['id'] for row in result]
                    self._recalculate_importance_scores(memory_ids, session)
                    logger.info(f"Applied pin boost to {updated_count} memories")

                return updated_count

    def apply_mention_boost(
        self,
        memory_ids: List[str],
        user_id: Optional[str] = None
    ) -> int:
        if not memory_ids:
            return 0

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                update_query = """
                UPDATE memories
                SET mention_count = mention_count + 1
                WHERE id = ANY(%(memory_ids)s::uuid[])
                RETURNING id
                """

                result = session.execute_query(update_query, {
                    'memory_ids': memory_ids
                })
                updated_count = len(result)

                if updated_count > 0:
                    updated_ids = [row['id'] for row in result]
                    self._recalculate_importance_scores(updated_ids, session)
                    logger.info(f"Applied mention boost to {updated_count} memories")

                return updated_count

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
            # Use positional param %s with tuple for PostgreSQL array type
            archive_query = """
            UPDATE memories
            SET is_archived = TRUE,
                archived_at = NOW()
            WHERE id = ANY(%s::uuid[])
              AND importance_score <= 0.001
              AND is_archived = FALSE
            RETURNING id
            """

            archived = session.execute_query(archive_query, (
                list(stale_ids),
            ))
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
            WHERE id = ANY(%s::uuid[])
              AND importance_score <= 0.001
              AND is_archived = FALSE
            RETURNING id
            """

            archived = session.execute_query(archive_query, (
                list(temporal_ids),
            ))
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
                        'reasoning': link.reasoning,
                        'created_at': format_utc_iso(link.created_at)
                    }
                    if link.extraction_bond:
                        outbound_obj['extraction_bond'] = link.extraction_bond

                    # Create inbound link object for target
                    inbound_obj = {
                        'uuid': str(link.source_id),
                        'type': link.link_type,
                        'reasoning': link.reasoning,
                        'created_at': format_utc_iso(link.created_at)
                    }
                    if link.extraction_bond:
                        inbound_obj['extraction_bond'] = link.extraction_bond

                    # Add or replace outbound link to source (one link per target UUID)
                    session.execute_update("""
                        UPDATE memories
                        SET outbound_links = (
                            SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                            FROM jsonb_array_elements(COALESCE(outbound_links, '[]'::jsonb)) AS elem
                            WHERE elem->>'uuid' != %(target_id)s
                        ) || %(outbound_obj)s::jsonb,
                        updated_at = NOW()
                        WHERE id = %(source_id)s
                          AND is_archived = FALSE
                    """, {
                        'source_id': str(link.source_id),
                        'target_id': str(link.target_id),
                        'outbound_obj': json.dumps(outbound_obj)
                    })

                    # Add or replace inbound link on target (one link per source UUID)
                    session.execute_update("""
                        UPDATE memories
                        SET inbound_links = (
                            SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                            FROM jsonb_array_elements(COALESCE(inbound_links, '[]'::jsonb)) AS elem
                            WHERE elem->>'uuid' != %(source_id)s
                        ) || %(inbound_obj)s::jsonb,
                        updated_at = NOW()
                        WHERE id = %(target_id)s
                          AND is_archived = FALSE
                    """, {
                        'target_id': str(link.target_id),
                        'source_id': str(link.source_id),
                        'inbound_obj': json.dumps(inbound_obj)
                    })

                logger.info(f"Created {len(links)} bidirectional links")

    def get_links_for_memory(
        self,
        memory_id: UUID,
        user_id: Optional[str] = None
    ) -> Dict[str, List[MemoryLinkEntry]]:
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

    # ==================== ENTITY OPERATIONS ====================

    def find_similar_entity_pairs(
        self,
        similarity_threshold: float,
        user_id: Optional[str] = None
    ) -> List[EntityPairRow]:
        """Find pairs of entities with similar names via pg_trgm self-join.

        Returns deduplicated pairs (a.id < b.id) over non-archived entities,
        ordered by trigram similarity. Used by the entity-merge service to
        surface candidate duplicates for LLM review.
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT a.id AS id_a, a.name AS name_a, a.entity_type AS type_a,
                   a.link_count AS links_a,
                   b.id AS id_b, b.name AS name_b, b.entity_type AS type_b,
                   b.link_count AS links_b,
                   similarity(a.name, b.name) AS sim
            FROM entities a
            JOIN entities b ON a.id < b.id
            WHERE a.is_archived = FALSE AND b.is_archived = FALSE
              AND similarity(a.name, b.name) > %(threshold)s
            ORDER BY sim DESC
            LIMIT 500
            """
            return session.execute_query(query, {'threshold': similarity_threshold})

    def get_or_create_entity(
        self,
        name: str,
        entity_type: str,
        user_id: Optional[str] = None,
        similarity_threshold: float = 0.3
    ) -> 'Entity':
        """
        Get existing entity or create new one using fuzzy name matching.

        Uses PostgreSQL trigram similarity (pg_trgm) for fuzzy matching.
        Tries exact match first (fast), then trigram similarity.

        Entity type is stored as metadata but not used for matching - prevents
        fragmentation where "GPT-4o" extracted as different types creates duplicates.

        Args:
            name: Entity name to find or create
            entity_type: Entity type for new entities (PERSON, ORG, GPE, etc.) - stored as metadata only
            user_id: User ID (uses ambient context if None)
            similarity_threshold: Minimum trigram similarity (0.0-1.0, default 0.3)

        Returns:
            Entity model (existing or newly created)
        """
        from lt_memory.models import Entity

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            # Try exact match first (fast path)
            exact_query = """
            SELECT * FROM entities
            WHERE user_id = %(user_id)s AND name = %(name)s
            LIMIT 1
            """
            existing = session.execute_single(exact_query, {
                'user_id': resolved_user_id,
                'name': name
            })

            if existing:
                # Update timestamp and return
                session.execute_update(
                    "UPDATE entities SET updated_at = NOW() WHERE id = %(id)s",
                    {'id': existing['id']}
                )
                return Entity(**existing)

            # Try fuzzy match using trigram similarity
            fuzzy_query = """
            SELECT *, similarity(name, %(name)s) AS sim_score
            FROM entities
            WHERE user_id = %(user_id)s
              AND similarity(name, %(name)s) > %(threshold)s
            ORDER BY sim_score DESC
            LIMIT 1
            """
            fuzzy_match = session.execute_single(fuzzy_query, {
                'user_id': resolved_user_id,
                'name': name,
                'threshold': similarity_threshold
            })

            if fuzzy_match:
                logger.debug(
                    f"Fuzzy matched '{name}' to existing entity '{fuzzy_match['name']}' "
                    f"(similarity: {fuzzy_match.get('sim_score', 'N/A'):.3f})"
                )
                # Update timestamp and return existing entity
                session.execute_update(
                    "UPDATE entities SET updated_at = NOW() WHERE id = %(id)s",
                    {'id': fuzzy_match['id']}
                )
                # Remove sim_score before passing to Entity model
                fuzzy_match.pop('sim_score', None)
                return Entity(**fuzzy_match)

            # No match found - create new entity
            insert_query = """
            INSERT INTO entities (
                user_id, name, entity_type, created_at
            ) VALUES (
                %(user_id)s, %(name)s, %(entity_type)s, NOW()
            )
            RETURNING *
            """

            result = session.execute_single(insert_query, {
                'user_id': resolved_user_id,
                'name': name,
                'entity_type': entity_type
            })

            if not result:
                raise RuntimeError(
                    f"INSERT...RETURNING failed for entity {name} ({entity_type})"
                )

            return Entity(**result)

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
                    'memory_id': memory_id,
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
                    'entity_id': entity_id
                })

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

    def get_entity(
        self,
        entity_id: UUID,
        user_id: Optional[str] = None
    ) -> Optional['Entity']:
        """Fetch an entity by ID (None if not found)."""
        from lt_memory.models import Entity

        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            result = session.execute_single(
                "SELECT * FROM entities WHERE id = %(entity_id)s LIMIT 1",
                {'entity_id': entity_id}
            )
            return Entity(**result) if result else None

    def merge_entities(
        self,
        source_id: UUID,
        target_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """Merge the source entity into the target entity.

        Rewrites every memory's entity_links entry pointing at source to point
        at target instead (deduplicating any memory already linked to both),
        bumps target's link_count by the affected count, and archives the
        source entity. The target keeps its name/type; the source is soft-deleted.
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            with session.transaction():
                target = self.get_entity(target_id, user_id=resolved_user_id)
                if not target:
                    raise ValueError(f"Target entity {target_id} not found")

                update_query = """
                UPDATE memories
                SET entity_links = (
                    SELECT COALESCE(jsonb_agg(replaced_elem), '[]'::jsonb)
                    FROM (
                        SELECT DISTINCT ON (elem_out->>'uuid') elem_out AS replaced_elem
                        FROM (
                            SELECT CASE
                                WHEN elem->>'uuid' = %(source_id)s
                                THEN jsonb_build_object(
                                    'uuid', %(target_id_str)s,
                                    'name', %(target_name)s,
                                    'type', %(target_type)s
                                )
                                ELSE elem
                            END AS elem_out
                            FROM jsonb_array_elements(entity_links) AS elem
                        ) replaced
                    ) deduped
                ),
                updated_at = NOW()
                WHERE entity_links @> %(source_filter)s::jsonb
                """

                affected_count = session.execute_update(update_query, {
                    'source_id': str(source_id),
                    'target_id_str': str(target_id),
                    'target_name': target.name,
                    'target_type': target.entity_type,
                    'source_filter': json.dumps([{"uuid": str(source_id)}])
                })

                session.execute_update("""
                    UPDATE entities
                    SET link_count = link_count + %(affected_count)s,
                        last_linked_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %(target_id)s
                """, {'affected_count': affected_count, 'target_id': target_id})

                session.execute_update("""
                    UPDATE entities
                    SET is_archived = TRUE,
                        archived_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %(source_id)s
                """, {'source_id': source_id})

                logger.info(
                    f"Merged entity {source_id} into {target_id}: "
                    f"updated {affected_count} memories"
                )

    # ==================== BATCH TRACKING =====================

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

    # ==================== EXTRACTION BATCH OPERATIONS ====================

    def get_users_with_pending_batches(self) -> List[str]:
        """
        Get user IDs with pending extraction batches.

        Uses admin session to query across users.

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

    def get_pending_batches_for_user(
        self,
        user_id: str
    ) -> list[ExtractionBatch]:
        """
        Get pending extraction batches for a specific user.

        Args:
            user_id: User ID to check

        Returns:
            List of ExtractionBatch models with status 'submitted', 'processing', or 'result_processing'
        """
        with self.session_manager.get_session(user_id) as session:
            query = """
            SELECT * FROM extraction_batches
            WHERE user_id = %(user_id)s
            AND status IN ('submitted', 'processing', 'result_processing')
            ORDER BY created_at ASC
            """

            results = session.execute_query(query, {'user_id': user_id})
            return [ExtractionBatch(**row) for row in results]

    def claim_batch(self, batch_id: UUID, user_id: str) -> bool:
        """Atomically claim an extraction batch for result processing.

        Transitions status to 'result_processing' only if the batch is
        currently 'submitted' or 'processing'. The atomic UPDATE...RETURNING
        guarantees exactly one concurrent caller succeeds, preventing
        duplicate processing when poll cycles overlap due to timeouts.

        Args:
            batch_id: Batch UUID to claim
            user_id: User ID

        Returns:
            True if this caller claimed the batch, False if already claimed
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            result = session.execute_single(
                "UPDATE extraction_batches SET status = 'result_processing' "
                "WHERE id = %(id)s AND status IN ('submitted', 'processing') "
                "RETURNING id",
                {'id': batch_id}
            )
            return bool(result)

    def update_batch_status(
        self,
        batch_id: UUID,
        status: BatchStatus,
        error_message: Optional[str] = None,
        completed_at: Optional[datetime] = None,
        user_id: Optional[str] = None,
        **extra_fields: Any
    ) -> None:
        """
        Update extraction batch status with optional extra fields.

        Args:
            batch_id: Batch UUID
            status: New status
            error_message: Optional error message for failed batches
            completed_at: Optional completion timestamp
            user_id: User ID (uses ambient context if None)
            **extra_fields: Additional column updates (e.g., items_completed)
        """
        resolved_user_id = self._resolve_user_id(user_id)

        with self.session_manager.get_session(resolved_user_id) as session:
            updates: Dict[str, Any] = {'status': status}

            if error_message:
                updates['error_message'] = error_message
            if completed_at:
                updates['completed_at'] = completed_at
            updates.update(extra_fields)

            set_clauses = [f"{k} = %({k})s" for k in updates.keys()]
            set_clause = ", ".join(set_clauses)

            query = f"""
            UPDATE extraction_batches
            SET {set_clause}
            WHERE id = %(batch_id)s
            """

            updates['batch_id'] = batch_id
            session.execute_update(query, updates)

    def increment_batch_retry(
        self,
        batch_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Increment retry counter for an extraction batch.

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
            session.execute_update(query, {'batch_id': batch_id})

    def delete_batch(
        self,
        batch_id: UUID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Delete an extraction batch record after processing completes.

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
                session.execute_update(query, {'batch_id': batch_id})

    def cleanup_old_batches(
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

    def get_users_with_memory_enabled(self) -> List[UserMemorySettings]:
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

    def cleanup(self) -> None:
        """
        Clean up database resources.

        No-op: Session manager is shared singleton, managed separately.
        Nulling reference breaks in-flight scheduler jobs.
        """
        logger.debug("LTMemoryDB cleanup completed (no-op)")
