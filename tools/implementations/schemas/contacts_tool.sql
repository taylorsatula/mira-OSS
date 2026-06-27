-- contacts_tool schema
-- Stores user contacts with encryption at rest

CREATE TABLE IF NOT EXISTS contacts (
    id TEXT PRIMARY KEY,
    encrypted__name TEXT NOT NULL,
    encrypted__email TEXT,
    encrypted__phone TEXT,
    encrypted__street TEXT,
    encrypted__city TEXT,
    encrypted__state TEXT,
    encrypted__zip TEXT,
    encrypted__pager_address TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
