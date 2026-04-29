# Running Multiple MIRA Instances

Run independent MIRA instances from a single codebase using the `MIRA_INSTANCE` environment variable.

## Quick Start

```bash
# Instance 1 (default — your existing setup, no changes needed)
python main.py

# Instance 2
MIRA_INSTANCE=kristen MIRA_PORT=1994 python main.py
```

## What MIRA_INSTANCE Controls

| Resource | Default | `MIRA_INSTANCE=kristen` |
|----------|---------|------------------------|
| Database | `mira_service` | `mira_service_kristen` |
| Vault prefix | `mira/` | `mira-kristen/` |
| Data directory | `data/users/` | `data-kristen/users/` |
| Valkey keys | (no prefix) | `kristen:` |
| Port | 1993 | Set via `MIRA_PORT` |

## Setup Steps

### 1. Create the Database

```bash
psql -U postgres -h localhost -c "
  CREATE DATABASE mira_service_kristen OWNER mira_admin;
"
psql -U postgres -h localhost -d mira_service_kristen \
  -f deploy/mira_service_schema.sql
```

Grant the application roles access:
```bash
psql -U postgres -h localhost -d mira_service_kristen -c "
  GRANT CONNECT ON DATABASE mira_service_kristen TO mira_dbuser;
  GRANT CONNECT ON DATABASE mira_service_kristen TO mira_admin;
"
```

### 2. Create Vault Secrets

```bash
# Database connection strings (update password as needed)
vault kv put secret/mira-kristen/database \
  admin_url="postgresql://mira_admin:<password>@localhost:5432/mira_service_kristen" \
  service_url="postgresql://mira_dbuser:<password>@localhost:5432/mira_service_kristen"

# API keys — can share with default instance or use separate keys
vault kv put secret/mira-kristen/api_keys \
  anthropic_key="$(vault kv get -field=anthropic_key secret/mira/api_keys)" \
  anthropic_batch_key="$(vault kv get -field=anthropic_batch_key secret/mira/api_keys)" \
  mira_api="$(uuidgen)"

# Services config
vault kv put secret/mira-kristen/services \
  app_url="http://localhost:1994" \
  valkey_url="valkey://localhost:6379"
```

### 3. Create Data Directory

```bash
mkdir -p data-kristen/users
```

### 4. Start the Instance

```bash
MIRA_INSTANCE=kristen \
MIRA_PORT=1994 \
MIRA_CORS_ORIGINS="http://localhost:1994,http://mira-k:1994" \
python main.py
```

### 5. (Optional) Reverse Proxy

Add to your Caddyfile or nginx config to serve at a clean URL:

```
# Caddyfile
http://mira-k {
    reverse_proxy localhost:1994
}
```

Then add `http://mira-k` to `MIRA_CORS_ORIGINS`.

## Sharing Resources

All instances on the same host share:
- PostgreSQL server (different databases)
- Vault server (different secret paths)
- Valkey server (different key prefixes)
- Python venv and codebase
- Anthropic API key (if configured to share)

Each instance has its own:
- Conversations and memories
- User identity and API key
- Working memory and system prompt personality
- Scheduled background jobs
- File uploads and tool state
