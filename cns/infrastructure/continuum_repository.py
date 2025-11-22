"""
Continuum repository for CNS.

Handles persistence and retrieval of continuums and messages
with RLS (Row Level Security) per user.
"""
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from cns.core.continuum import Continuum
from cns.core.state import ContinuumState
from cns.core.message import Message
from cns.core.events import ContinuumEvent
from clients.postgres_client import PostgresClient
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)

# Module-level singleton instance
_continuum_repo_instance = None


def get_continuum_repository() -> 'ContinuumRepository':
    """
    Get or create singleton ContinuumRepository instance.

    Following the pattern from clients/valkey_client.py and utils/bge_embeddings.py,
    this ensures we reuse the same repository instance and its database connection pool.

    Returns:
        Singleton ContinuumRepository instance
    """
    global _continuum_repo_instance
    if _continuum_repo_instance is None:
        logger.info("Creating singleton ContinuumRepository instance")
        _continuum_repo_instance = ContinuumRepository()
    return _continuum_repo_instance


class ContinuumRepository:
    """
    Repository for continuum persistence.

    Handles database operations for continuums and messages
    with automatic user isolation via RLS.
    """
    
    def __init__(self):
        """Initialize repository."""
        self._db_cache = {}
        
    def _get_client(self, user_id: str):
        """Get or create database client for user."""
        if user_id not in self._db_cache:
            self._db_cache[user_id] = PostgresClient("mira_service", user_id=user_id)
        return self._db_cache[user_id]
    
    def get_continuum(self, user_id: str) -> Optional[Continuum]:
        """
        Get most recent continuum for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Most recent continuum or None if no continuums exist
        """
        try:
            db = self._get_client(user_id)
            
            # Get most recent continuum
            existing = db.execute_query(
                "SELECT * FROM continuums ORDER BY created_at DESC LIMIT 1"
            )
            
            if not existing:
                return None
                
            row = existing[0]
            # Parse JSON metadata if it's a string (asyncpg doesn't auto-parse)
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}
            
            # Convert asyncpg UUID to standard UUID
            from uuid import UUID
            continuum_id = UUID(str(row['id'])) if not isinstance(row['id'], UUID) else row['id']
            
            state = ContinuumState(
                id=continuum_id,
                user_id=row['user_id'],
                metadata=metadata
            )
            
            # Create continuum
            continuum = Continuum(state)
            
            logger.debug(f"Retrieved existing continuum {continuum.id} for user {user_id}")
            return continuum
        except Exception as e:
            logger.error(f"Failed to get continuum for user {user_id}: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
    
    def create_continuum(self, user_id: str) -> Continuum:
        """
        Create new continuum for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            New continuum instance
        """
        try:
            # Create new continuum
            continuum = Continuum.create_new(user_id)
            
            # Persist to database
            now = utc_now()
            db = self._get_client(user_id)
            db.execute_query(
                """
                INSERT INTO continuums (id, user_id, created_at, updated_at, metadata)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    continuum.id,  # Keep as UUID - PostgresClient will convert
                    user_id,
                    now,
                    now,
                    json.dumps(continuum._state.metadata)
                )
            )

            logger.info(f"Created new continuum {continuum.id} for user {user_id}")
            return continuum
        except Exception as e:
            logger.error(f"Failed to create continuum for user {user_id}: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
    
    def get_by_id(self, continuum_id: str, user_id: str) -> Optional[Continuum]:
        """
        Get continuum by ID.
        
        Args:
            continuum_id: Continuum ID (string, will be converted to UUID)
            user_id: User ID for access verification
            
        Returns:
            Continuum or None if not found
        """
        try:
            db = self._get_client(user_id)
            
            # Convert string ID to UUID for query
            from uuid import UUID as uuid_type
            try:
                conv_uuid = uuid_type(continuum_id)
            except ValueError:
                logger.warning(f"Invalid continuum ID format: {continuum_id}")
                return None
            
            result = db.execute_query(
                "SELECT * FROM continuums WHERE id = %s",
                (conv_uuid,)
            )
            
            if not result:
                return None
            
            row = result[0]
            # Parse JSON metadata if it's a string (asyncpg doesn't auto-parse)
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata) if metadata else {}
            
            state = ContinuumState(
                id=row['id'],  # Already a UUID from database
                user_id=row['user_id'],
                metadata=metadata
            )
            
            continuum = Continuum(state)
            return continuum
        except Exception as e:
            logger.error(f"Failed to get continuum {continuum_id} for user {user_id}: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
    
    def save_message(self, message: Message, continuum_id: Union[str, UUID], user_id: str) -> None:
        """
        Save message to database.

        Automatically creates segment boundary sentinel when second real message
        is saved (forming a complete user/assistant exchange).

        Args:
            message: Message to save
            continuum_id: Continuum ID (string or UUID)
            user_id: User ID
        """
        try:
            # Additional validation to prevent empty messages from being saved
            if isinstance(message.content, str) and not message.content.strip():
                logger.error(f"Blocking empty {message.role} message for continuum {continuum_id}")
                raise ValueError(f"Cannot save empty {message.role} message to database")

            db = self._get_client(user_id)

            # Convert continuum_id to UUID if it's a string
            if isinstance(continuum_id, str):
                from uuid import UUID as uuid_type
                continuum_id = uuid_type(continuum_id)

            # Check if this is a real conversation message (not segment/system notification)
            is_real_message = not (
                message.metadata.get('is_segment_boundary') or
                message.metadata.get('system_notification')
            )

            # Ensure active segment exists if this is a real message
            if is_real_message:
                self._ensure_active_segment(continuum_id, user_id, message.created_at, db)

            # Extract segment embedding if this is a segment boundary with embedding
            segment_embedding_value = None
            if message.metadata.get('is_segment_boundary') and message.metadata.get('segment_embedding_value') is not None:
                embedding_list = message.metadata['segment_embedding_value']
                # Format as PostgreSQL vector: '[0.1, 0.2, ...]'
                segment_embedding_value = '[' + ','.join(str(x) for x in embedding_list) + ']'

            # Get base message tuple
            base_tuple = message.to_db_tuple(continuum_id, user_id)

            # Upsert message with segment embedding
            # ON CONFLICT handles updates to existing messages (e.g., collapsed segment sentinels)
            db.execute_query(
                """
                INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at, segment_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector(384))
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    segment_embedding = EXCLUDED.segment_embedding
                """,
                base_tuple + (segment_embedding_value,)
            )

            # Track user activity day (upstream activity tracking for vacation-proof scoring)
            if message.role == "user":
                try:
                    from utils.user_activity import increment_user_activity_day
                    logger.info(f"[SINGLE] Attempting to increment activity day for user {user_id}")
                    result = increment_user_activity_day(user_id)
                    logger.info(f"[SINGLE] Activity day incremented successfully for user {user_id}, result: {result}")
                except Exception as e:
                    logger.error(f"[SINGLE] Failed to increment activity day for user {user_id}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Failed to save message to continuum {continuum_id}: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e

    def _ensure_active_segment(self, continuum_id: UUID, user_id: str, current_message_time, db) -> None:
        """
        Ensure active segment exists, creating one if needed when second message arrives.

        Segments represent conversations (2+ messages forming user/assistant pairs).
        Sentinel is created when the second real message is saved.

        Args:
            continuum_id: Continuum UUID
            user_id: User ID
            current_message_time: Timestamp of message being saved
            db: Database client
        """
        # Check if active segment already exists
        active_segment_query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            ORDER BY created_at DESC
            LIMIT 1
        """
        active_segments = db.execute_query(active_segment_query, (continuum_id,))

        if not active_segments:
            # No active segment - check if we have enough real messages to create one
            # Count existing real messages (excluding boundaries, system notifications)
            real_message_count_query = """
                SELECT COUNT(*) FROM messages
                WHERE continuum_id = %s
                    AND (metadata->>'is_segment_boundary' IS NULL OR metadata->>'is_segment_boundary' = 'false')
                    AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' = 'false')
            """
            count_row = db.execute_query(real_message_count_query, (continuum_id,))
            existing_count = count_row[0]['count'] if count_row else 0

            # We're about to save 1 message, so if existing_count >= 1, we'll have 2+ after this save
            if existing_count >= 1:
                # Find the most recent collapsed segment to determine time boundary
                last_segment_query = """
                    SELECT metadata->>'segment_end_time' as segment_end_time
                    FROM messages
                    WHERE continuum_id = %s
                        AND metadata->>'is_segment_boundary' = 'true'
                        AND metadata->>'status' = 'collapsed'
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                last_segment_row = db.execute_query(last_segment_query, (continuum_id,))

                # If there's a previous collapsed segment, only look at messages after it ended
                if last_segment_row and last_segment_row[0].get('segment_end_time'):
                    from datetime import datetime
                    last_segment_end = datetime.fromisoformat(last_segment_row[0]['segment_end_time'])

                    first_message_query = """
                        SELECT created_at FROM messages
                        WHERE continuum_id = %s
                            AND created_at > %s
                            AND (metadata->>'is_segment_boundary' IS NULL OR metadata->>'is_segment_boundary' = 'false')
                            AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' = 'false')
                        ORDER BY created_at ASC
                        LIMIT 1
                    """
                    first_msg_row = db.execute_query(first_message_query, (continuum_id, last_segment_end))
                else:
                    # No previous collapsed segment - use absolute first message
                    first_message_query = """
                        SELECT created_at FROM messages
                        WHERE continuum_id = %s
                            AND (metadata->>'is_segment_boundary' IS NULL OR metadata->>'is_segment_boundary' = 'false')
                            AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' = 'false')
                        ORDER BY created_at ASC
                        LIMIT 1
                    """
                    first_msg_row = db.execute_query(first_message_query, (continuum_id,))

                first_message_time = first_msg_row[0]['created_at'] if first_msg_row else current_message_time

                # Create segment boundary sentinel
                from cns.services.segment_helpers import create_segment_boundary_sentinel

                sentinel = create_segment_boundary_sentinel(
                    first_message_time=first_message_time,
                    continuum_id=str(continuum_id)
                )

                # Save sentinel (recursive call, but sentinel has is_segment_boundary=True so won't recurse into _ensure_active_segment)
                self.save_message(sentinel, continuum_id, user_id)

                logger.info(
                    f"Created segment boundary sentinel for continuum {continuum_id} "
                    f"(now has {existing_count + 1} messages)"
                )
    
    def save_messages_batch(self, messages: List[Message], continuum_id: Union[str, UUID], user_id: str) -> None:
        """
        Save multiple messages to database as a batch operation.

        Args:
            messages: List of messages to save
            continuum_id: Continuum ID (string or UUID)
            user_id: User ID
        """
        if not messages:
            return

        try:
            # Additional validation to prevent empty messages from being saved
            for message in messages:
                if isinstance(message.content, str) and not message.content.strip():
                    logger.error(f"Blocking empty {message.role} message for continuum {continuum_id}")
                    raise ValueError(f"Cannot save empty {message.role} message to database")

            db = self._get_client(user_id)

            # Convert continuum_id to UUID if it's a string
            if isinstance(continuum_id, str):
                from uuid import UUID as uuid_type
                continuum_id = uuid_type(continuum_id)

            # Check if any messages are real conversation messages
            real_messages = [
                msg for msg in messages
                if not (
                    msg.metadata.get('is_segment_boundary') or
                    msg.metadata.get('system_notification')
                )
            ]

            # Ensure active segment exists if we have real messages
            if real_messages:
                # Use the earliest real message timestamp
                earliest_timestamp = min(msg.created_at for msg in real_messages)
                self._ensure_active_segment(continuum_id, user_id, earliest_timestamp, db)
                logger.debug(f"Checked segment boundary for batch save with {len(real_messages)} real messages")

            # Insert each message using the same pattern as save_message
            has_user_message = False
            for message in messages:
                # Extract segment embedding if this is a segment boundary with embedding
                segment_embedding_value = None
                if message.metadata.get('is_segment_boundary') and message.metadata.get('segment_embedding_value') is not None:
                    embedding_list = message.metadata['segment_embedding_value']
                    # Format as PostgreSQL vector: '[0.1, 0.2, ...]'
                    segment_embedding_value = '[' + ','.join(str(x) for x in embedding_list) + ']'

                # Get base message tuple
                base_tuple = message.to_db_tuple(continuum_id, user_id)

                db.execute_query(
                    """
                    INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at, segment_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector(384))
                    """,
                    base_tuple + (segment_embedding_value,)
                )
                if message.role == "user":
                    has_user_message = True

            # Track user activity day if batch contained user message
            if has_user_message:
                try:
                    from utils.user_activity import increment_user_activity_day
                    logger.info(f"Attempting to increment activity day for user {user_id}")
                    result = increment_user_activity_day(user_id)
                    logger.info(f"Activity day incremented successfully for user {user_id}, result: {result}")
                except Exception as e:
                    logger.error(f"Failed to increment activity day for user {user_id}: {e}", exc_info=True)

            logger.debug(f"Saved {len(messages)} messages to continuum {continuum_id}")

        except Exception as e:
            logger.error(f"Failed to save {len(messages)} messages to continuum {continuum_id}: {str(e)}")
            raise RuntimeError(f"Database batch operation failed: {str(e)}") from e

    def _parse_message_rows(self, rows: List[Dict[str, Any]]) -> List[Message]:
        """Convert raw database rows into Message instances."""
        messages: List[Message] = []

        for row in rows:
            message_id = row.get("id")
            try:
                canonical_id = message_id if isinstance(message_id, UUID) else UUID(str(message_id))
            except (ValueError, TypeError):
                logger.warning(f"Skipping message row with invalid ID: {message_id}")
                continue

            content = row.get("content")
            if isinstance(content, str) and content.startswith("["):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse message content JSON; keeping raw string")

            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    logger.debug("Failed to parse message metadata JSON; defaulting to empty dict")
                    metadata = {}

            messages.append(
                Message(
                    id=canonical_id,
                    content=content,
                    role=row.get("role"),
                    created_at=row.get("created_at"),
                    metadata=metadata or {},
                )
            )

        return messages

    def load_messages_with_metadata(self, 
                                  continuum_id: str, 
                                  user_id: str,
                                  metadata_filters: Dict[str, Any],
                                  limit: Optional[int] = None,
                                  order_desc: bool = True) -> List[Message]:
        """
        Load messages with flexible metadata filtering.

        This consolidates loading for segment boundaries, system notifications, and other metadata-based queries.

        Args:
            continuum_id: Continuum ID
            user_id: User ID for RLS
            metadata_filters: Dict of metadata key-value pairs to filter by
            limit: Maximum number of messages to return
            order_desc: If True, order by created_at DESC (most recent first)

        Returns:
            List of messages matching criteria

        Examples:
            # Load collapsed segment boundaries
            load_messages_with_metadata(conv_id, user_id, {
                'is_segment_boundary': 'true',
                'status': 'collapsed'
            }, limit=5)

            # Load system notifications
            load_messages_with_metadata(conv_id, user_id, {
                'system_notification': 'true'
            }, limit=3)
        """
        db = self._get_client(user_id)
        
        # Build query with metadata filters
        where_conditions = ["continuum_id = %s"]
        params = [continuum_id]
        
        # Add metadata conditions
        for key, value in metadata_filters.items():
            # Handle boolean values properly
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            where_conditions.append(f"metadata->>'{key}' = %s")
            params.append(str(value))
        
        order_clause = "DESC" if order_desc else "ASC"
        query = f"""
            SELECT * FROM messages 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY created_at {order_clause}
        """
        
        if limit:
            query += f" LIMIT {limit}"

        message_rows = db.execute_query(query, tuple(params))

        # Reverse if we queried DESC to get chronological order
        if order_desc and len(message_rows) > 1:
            message_rows.reverse()

        return self._parse_message_rows(message_rows)

    def load_messages_by_ids(self,
                              continuum_id: str,
                              user_id: str,
                              message_ids: List[str]) -> List[Message]:
        """Load persisted messages by their IDs."""
        if not message_ids:
            return []

        try:
            canonical_continuum_id = (
                continuum_id
                if isinstance(continuum_id, UUID)
                else UUID(str(continuum_id))
            )
        except (ValueError, TypeError):
            logger.error(f"Invalid continuum ID provided when loading messages: {continuum_id}")
            return []

        try:
            unique_ids = []
            seen = set()
            for message_id in message_ids:
                if not message_id:
                    continue
                uuid_obj = message_id if isinstance(message_id, UUID) else UUID(str(message_id))
                if uuid_obj not in seen:
                    unique_ids.append(uuid_obj)
                    seen.add(uuid_obj)
        except (ValueError, TypeError) as exc:
            logger.error(f"Invalid message ID provided when loading messages: {exc}")
            return []

        if not unique_ids:
            return []

        db = self._get_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
              AND id = ANY(%s::uuid[])
        """

        rows = db.execute_query(query, (canonical_continuum_id, unique_ids))
        messages = self._parse_message_rows(rows)

        by_id = {str(message.id): message for message in messages}
        ordered_messages = []

        for message_id in message_ids:
            try:
                canonical_id = str(message_id) if isinstance(message_id, UUID) else str(UUID(str(message_id)))
            except (ValueError, TypeError):
                continue
            message = by_id.get(canonical_id)
            if message:
                ordered_messages.append(message)

        return ordered_messages

    def update_message_metadata(
        self,
        continuum_id: str,
        user_id: str,
        message_id: str,
        metadata_patch: Dict[str, Any],
    ) -> None:
        """Merge metadata_patch into the message metadata payload."""
        if not metadata_patch:
            return

        db = self._get_client(user_id)

        query = """
            UPDATE messages
            SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
            WHERE continuum_id = %s AND id = %s
        """

        db.execute_update(
            query,
            (
                json.dumps(metadata_patch),
                UUID(str(continuum_id)) if not isinstance(continuum_id, UUID) else continuum_id,
                UUID(str(message_id)) if not isinstance(message_id, UUID) else message_id,
            ),
        )

    def get_history(self, user_id: str, offset: int = 0, limit: int = 50, 
                   start_date: Optional[Any] = None, end_date: Optional[Any] = None,
                   message_type: str = "regular") -> Dict[str, Any]:
        """
        Get continuum history with pagination and date filtering.
        
        Args:
            user_id: User ID for RLS
            offset: Pagination offset
            limit: Maximum number of messages to return
            start_date: Optional start date filter
            end_date: Optional end date filter
            message_type: Type of messages to retrieve ("regular", "summaries", "all")
            
        Returns:
            Dictionary with messages, pagination info, and metadata
        """
        db = self._get_client(user_id)
        
        # Build query with optional date filtering
        where_conditions = ["user_id = %s"]
        params = [user_id]
        param_counter = 2
        
        # Filter by message type
        if message_type == "all":
            # Include all messages, no additional filter
            pass
        else:
            # Default to regular messages (exclude system notifications)
            where_conditions.append("(metadata->>'system_notification' IS NULL OR metadata->>'system_notification' != 'true')")
        
        if start_date:
            where_conditions.append("created_at >= %s")
            params.append(start_date)
            
        if end_date:
            where_conditions.append("created_at <= %s")
            params.append(end_date)
        
        # Get messages with pagination
        query = f"""
            SELECT * FROM messages 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY created_at DESC
            OFFSET %s LIMIT %s
        """
        params.extend([offset, limit + 1])  # Get one extra to check for more results
        
        message_rows = db.execute_query(query, tuple(params))
        
        # Check if there are more results
        has_more = len(message_rows) > limit
        if has_more:
            message_rows = message_rows[:-1]  # Remove the extra row
        
        # Format messages for API
        messages = []
        for row in message_rows:
            messages.append({
                "id": str(row['id']),
                "role": row['role'],
                "content": row['content'],
                "timestamp": format_utc_iso(row['created_at']),
                "metadata": row.get('metadata', {})
            })
        
        return {
            "messages": messages,
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None
        }
    
    def search_continuums(self, user_id: str, search_query: str, 
                           offset: int = 0, limit: int = 50,
                           message_type: str = "regular") -> Dict[str, Any]:
        """
        Search continuums using full-text search.
        
        Args:
            user_id: User ID for RLS
            search_query: Text to search for in message content
            offset: Pagination offset
            limit: Maximum number of messages to return
            message_type: Type of messages to retrieve ("regular", "summaries", "all")
            
        Returns:
            Dictionary with matching messages, pagination info, and search metadata
        """
        db = self._get_client(user_id)
        
        # Build query with message type filtering
        type_filter = ""
        if message_type == "all":
            # Include all messages, no filter
            pass
        else:
            # Default to regular messages (exclude system notifications)
            type_filter = "AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' != 'true')"
        
        # Use PostgreSQL full-text search
        query = f"""
            SELECT * FROM messages 
            WHERE (content ILIKE %s OR metadata::text ILIKE %s)
            {type_filter}
            ORDER BY created_at DESC
            OFFSET %s LIMIT %s
        """
        
        search_pattern = f"%{search_query}%"
        params = (search_pattern, search_pattern, offset, limit + 1)
        
        message_rows = db.execute_query(query, params)
        
        # Check if there are more results
        has_more = len(message_rows) > limit
        if has_more:
            message_rows = message_rows[:-1]  # Remove the extra row
        
        # Format messages for API
        messages = []
        for row in message_rows:
            messages.append({
                "id": str(row['id']),
                "role": row['role'],
                "content": row['content'],
                "timestamp": format_utc_iso(row['created_at']),
                "metadata": row.get('metadata', {})
            })
        
        return {
            "messages": messages,
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
            "search_query": search_query
        }
    
    
    def get_messages_for_dates(self, user_id: str, dates: List[str]) -> Dict[str, List[Dict]]:
        """
        Get messages grouped by date for temporal RAG linking.
        
        Args:
            user_id: User ID for RLS
            dates: List of date strings (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping dates to their messages
        """
        from datetime import datetime
        
        db = self._get_client(user_id)
        messages_by_date = {}
        
        for date_str in dates:
            # Parse date and create date range for SQL query (same as temporal_context.py)
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            start_datetime = datetime.combine(target_date, datetime.min.time())
            end_datetime = datetime.combine(target_date, datetime.max.time())
            
            query = """
                SELECT id, role, content, created_at, metadata
                FROM messages
                WHERE created_at >= %s AND created_at <= %s
                ORDER BY created_at
            """
            
            message_rows = db.execute_query(
                query,
                (start_datetime, end_datetime)
            )
            
            messages = []
            for row in message_rows:
                messages.append({
                    "id": str(row['id']),
                    "role": row['role'],
                    "content": row['content'],
                    "created_at": format_utc_iso(row['created_at']),
                    "metadata": row.get('metadata', {})
                })
            
            if messages:
                messages_by_date[date_str] = messages
                
        return messages_by_date
    
    def update_continuum_metadata(self, continuum: Continuum) -> None:
        """
        Update continuum metadata.

        Args:
            continuum: Continuum with updated metadata
        """
        db = self._get_client(continuum.user_id)

        db.execute_query(
            """
            UPDATE continuums
            SET metadata = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (json.dumps(continuum._state.metadata), continuum.id)
        )

    # =========================================================================
    # Segment Query Methods
    # =========================================================================

    def find_active_segment(self, continuum_id: Union[str, UUID], user_id: str) -> Optional[Message]:
        """
        Find active segment sentinel for a continuum.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            Active segment sentinel or None
        """
        db = self._get_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            ORDER BY created_at DESC
            LIMIT 1
        """

        rows = db.execute_query(query, (str(continuum_id),))
        messages = self._parse_message_rows(rows)

        return messages[0] if messages else None

    def find_collapsed_segments(
        self,
        continuum_id: Union[str, UUID],
        user_id: str,
        limit: int
    ) -> List[Message]:
        """
        Find recent collapsed segment sentinels for a continuum.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            limit: Maximum number of segments to return

        Returns:
            List of collapsed segment sentinels in chronological order (oldest first)
        """
        db = self._get_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'collapsed'
            ORDER BY created_at DESC
            LIMIT %s
        """

        rows = db.execute_query(query, (str(continuum_id), limit))

        # Reverse to get chronological order (oldest first)
        if rows:
            rows.reverse()

        return self._parse_message_rows(rows)

    def find_segment_by_id(
        self,
        continuum_id: Union[str, UUID],
        segment_id: str,
        user_id: str
    ) -> Optional[Message]:
        """
        Find segment sentinel by segment_id.

        Args:
            continuum_id: Continuum ID
            segment_id: Segment UUID
            user_id: User ID

        Returns:
            Segment sentinel or None
        """
        db = self._get_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'segment_id' = %s
            LIMIT 1
        """

        rows = db.execute_query(query, (str(continuum_id), segment_id))
        messages = self._parse_message_rows(rows)

        return messages[0] if messages else None

    def find_all_segments(self, user_id: str, limit: int) -> List[Message]:
        """
        Find all segment sentinels for a user across all continuums.

        Args:
            user_id: User ID
            limit: Maximum number of segments to return

        Returns:
            List of segment sentinels ordered by creation time (newest first)
        """
        db = self._get_client(user_id)

        query = """
            SELECT
                id,
                role,
                content,
                metadata,
                created_at
            FROM messages
            WHERE metadata->>'is_segment_boundary' = 'true'
            ORDER BY created_at DESC
            LIMIT %s
        """

        rows = db.execute_query(query, (limit,))
        return self._parse_message_rows(rows)

    def find_failed_extraction_segments(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Find collapsed segments where memory extraction failed or hasn't been attempted.

        Args:
            user_id: User ID

        Returns:
            List of dicts with segment_id and message_id
        """
        db = self._get_client(user_id)

        query = """
            SELECT id, metadata
            FROM messages
            WHERE metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'collapsed'
                AND (metadata->>'memories_extracted' = 'false'
                     OR metadata->>'memories_extracted' IS NULL)
            ORDER BY created_at DESC
        """

        rows = db.execute_query(query, ())

        segments = []
        for row in rows:
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata) if metadata else {}

            segments.append({
                'message_id': str(row['id']),
                'segment_id': metadata.get('segment_id', str(row['id']))
            })

        return segments

    def find_all_active_segments_admin(self) -> List[Dict[str, Any]]:
        """
        Find all active segments across all users (admin query for timeout service).

        Returns:
            List of dicts with segment data (id, continuum_id, user_id, metadata, created_at)
        """
        from utils.database_session_manager import get_shared_session_manager

        session_manager = get_shared_session_manager()

        with session_manager.get_admin_session() as session:
            rows = session.execute_query("""
                SELECT
                    id,
                    continuum_id,
                    user_id,
                    metadata,
                    created_at
                FROM messages
                WHERE metadata->>'is_segment_boundary' = 'true'
                    AND metadata->>'status' = 'active'
                ORDER BY created_at ASC
            """)

            # Rows already come back as dicts with normalized types
            return rows

    def load_segment_messages(
        self,
        continuum_id: Union[str, UUID],
        user_id: str,
        sentinel_time: datetime
    ) -> List[Message]:
        """
        Load all conversation messages from an active segment.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            sentinel_time: Creation time of segment sentinel

        Returns:
            List of messages in chronological order (excludes boundaries and system notifications)
        """
        db = self._get_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND created_at >= %s
                AND (metadata->>'is_segment_boundary' IS NULL OR metadata->>'is_segment_boundary' = 'false')
                AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' = 'false')
            ORDER BY created_at ASC
        """

        rows = db.execute_query(query, (str(continuum_id), sentinel_time))
        return self._parse_message_rows(rows)

    def load_continuity_messages(
        self,
        continuum_id: Union[str, UUID],
        user_id: str,
        turn_count: int
    ) -> List[Message]:
        """
        Load last N user/assistant message pairs from end of most recent collapsed segment.

        Called only during session cache loading after timeout, when all segments are collapsed.
        Provides conversational continuity by showing the tail end of the previous session.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            turn_count: Number of user/assistant pairs to load

        Returns:
            Last N user/assistant message pairs in chronological order
        """
        db = self._get_client(user_id)

        # Get messages before the most recent collapsed segment's end time
        # Request 4x turn_count to ensure we have enough messages to find complete pairs
        query = """
            WITH boundary_time AS (
                SELECT (metadata->>'segment_end_time')::timestamp as cutoff_time
                FROM messages
                WHERE continuum_id = %s
                    AND metadata->>'is_segment_boundary' = 'true'
                    AND metadata->>'status' = 'collapsed'
                ORDER BY created_at DESC
                LIMIT 1
            )
            SELECT m.* FROM messages m, boundary_time
            WHERE m.continuum_id = %s
                AND m.created_at < boundary_time.cutoff_time
                AND m.role IN ('user', 'assistant')
                AND (m.metadata->>'is_segment_boundary' IS NULL OR m.metadata->>'is_segment_boundary' = 'false')
                AND (m.metadata->>'system_notification' IS NULL OR m.metadata->>'system_notification' = 'false')
            ORDER BY m.created_at DESC
            LIMIT %s
        """

        continuum_id_str = str(continuum_id)
        rows = db.execute_query(query, (continuum_id_str, continuum_id_str, turn_count * 4))

        if not rows:
            return []

        # Parse messages (in reverse chronological order from query)
        all_messages = self._parse_message_rows(rows)

        # Work backwards to find last N assistant messages and their corresponding user messages
        pairs = []
        assistant_count = 0
        i = 0

        while i < len(all_messages) and assistant_count < turn_count:
            msg = all_messages[i]

            if msg.role == 'assistant':
                # Found an assistant message, now find its corresponding user message
                assistant_msg = msg
                user_msg = None

                # Look backwards for the user message
                for j in range(i + 1, len(all_messages)):
                    if all_messages[j].role == 'user':
                        user_msg = all_messages[j]
                        break

                # Only add if we found the user message
                if user_msg:
                    pairs.append((user_msg, assistant_msg))
                    assistant_count += 1

            i += 1

        # Reverse pairs to get chronological order (oldest first)
        pairs.reverse()

        # Flatten the pairs into a single list
        messages = []
        for user_msg, assistant_msg in pairs:
            messages.extend([user_msg, assistant_msg])

        return messages
