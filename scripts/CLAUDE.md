# scripts/ - Operational and one-time maintenance scripts

## Rules

- One-time migrations must be idempotent and fail loud on unrecoverable data; do not hide partial migration errors.

## Files
- `migrations/2026_05_04_userdata_one_time.py` - Re-encrypts legacy per-user SQLite encrypted fields with the Vault-backed userdata key, migrates pager plaintext columns into encrypted columns, and adds hardening columns/indexes.
