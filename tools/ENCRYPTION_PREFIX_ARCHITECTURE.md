# Encryption Prefix Architecture

## Overview

MIRA uses an `encrypted__` prefix pattern to explicitly mark fields containing PII (Personally Identifiable Information) that are encrypted at rest. This prefix is maintained throughout the entire data flow from application code through database storage.

## Design Principles

1. **Explicit > Implicit**: The `encrypted__` prefix makes encryption boundaries immediately visible in code
2. **Consistency**: Same field name throughout the stack prevents confusion
3. **Type Safety**: Prefixed fields clearly indicate they contain encrypted data

## Architecture

### Data Flow

```
Application Code → UserDataManager → SQLite Database
encrypted__name  → encrypted__name → encrypted__name (encrypted value)
```

### Components

#### UserDataManager (`utils/userdata_manager.py`)

The `UserDataManager` handles transparent encryption/decryption:

- **`_encrypt_dict()`**: Encrypts values for fields with `encrypted__` prefix before database storage
- **`_decrypt_dict()`**: Decrypts encrypted field values when reading from database
- **Prefix Preservation**: Column names keep the `encrypted__` prefix in the database schema

#### Tools

Tools use the `encrypted__` prefix for all PII fields:

**reminder_tool**:
- `encrypted__title`: Reminder title
- `encrypted__description`: Reminder description
- `encrypted__additional_notes`: Additional notes

**contacts_tool**:
- `encrypted__name`: Contact name
- `encrypted__email`: Contact email
- `encrypted__phone`: Contact phone number

### Database Schema

Database columns use the `encrypted__` prefix in their names:

```sql
CREATE TABLE contacts (
    id TEXT PRIMARY KEY,
    encrypted__name TEXT NOT NULL,
    encrypted__email TEXT,
    encrypted__phone TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### API Responses

API responses return fields with the `encrypted__` prefix intact:

```json
{
  "contact": {
    "uuid": "1e16caea-b2aa-4107-b1b5-1fe2e58dcc7a",
    "encrypted__name": "John Smith",
    "encrypted__email": "john@example.com",
    "encrypted__phone": "555-1234"
  }
}
```

## Migration

The `migrate_encrypted_columns.py` script migrates existing databases from plain column names to `encrypted__` prefixed names:

```bash
python migrate_encrypted_columns.py
```

The migration:
1. Creates new tables with `encrypted__` prefixed columns
2. Copies data from old tables to new tables
3. Drops old tables and renames new tables
4. Recreates indexes with correct column references
5. Is idempotent - safe to run multiple times

## Benefits

1. **Clear Security Boundaries**: Developers immediately see which fields contain sensitive data
2. **Audit Trail**: Easy to grep for `encrypted__` to find all PII fields
3. **No Magic**: The encryption boundary is explicit in the code, not hidden behind abstractions
4. **Consistency**: Same field name from application layer to database layer
5. **Type Checking**: Can build tooling to enforce encryption for `encrypted__` prefixed fields

## Implementation Notes

- Non-PII fields (IDs, timestamps, flags) do NOT use the prefix
- The prefix is a marker for the `UserDataManager` to apply encryption
- Encryption is session-based using deterministic keys derived from user UUIDs
- The `encrypted__` prefix is only stripped when data leaves the system (e.g., display to user in web UI)

## Future Considerations

- Could extend to other encryption schemes (e.g., `encrypted_sensitive__`, `encrypted_hipaa__`)
- Could build linters to enforce prefix usage for fields containing sensitive data patterns
- Could create type aliases like `EncryptedString` that map to `encrypted__` prefixed fields
