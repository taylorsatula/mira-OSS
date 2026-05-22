# deploy/postgresql.sh
# PostgreSQL service startup, schema deployment, and Vault credential storage
# Source this file - do not execute directly
#
# Requires: lib/output.sh, lib/services.sh, lib/vault.sh sourced first
# Requires: OS, DISTRO, CONFIG_*, LOUD_MODE variables set

# Validate required variables
: "${OS:?Error: OS must be set}"
: "${CONFIG_DB_PASSWORD:?Error: CONFIG_DB_PASSWORD must be set}"
: "${CONFIG_ANTHROPIC_KEY:?Error: CONFIG_ANTHROPIC_KEY must be set}"

if [ "$OS" = "macos" ]; then
    print_header "Step 12: Starting Services"

    start_service valkey brew
    start_service postgresql@17 brew

    sleep 2
fi

# Wait for PostgreSQL to be ready to accept connections
echo -ne "${DIM}${ARROW}${RESET} Waiting for PostgreSQL to be ready... "
PG_READY=0
for i in {1..30}; do
    if [ "$OS" = "linux" ]; then
        # On Linux, check with pg_isready (Fedora PGDG uses /usr/pgsql-17/bin/)
        if sudo -u postgres pg_isready > /dev/null 2>&1 || \
           sudo -u postgres /usr/pgsql-17/bin/pg_isready > /dev/null 2>&1; then
            PG_READY=1
            break
        fi
    elif [ "$OS" = "macos" ]; then
        # On macOS, check with pg_isready as current user
        # Homebrew PostgreSQL 17 uses versioned command name
        if pg_isready-17 > /dev/null 2>&1; then
            PG_READY=1
            break
        fi
    fi
    sleep 1
done

if [ $PG_READY -eq 0 ]; then
    echo -e "${ERROR}"
    print_error "PostgreSQL did not become ready within 30 seconds"
    if [ "$OS" = "linux" ]; then
        if [ "$DISTRO" = "fedora" ]; then
            print_info "Check status: systemctl status postgresql-17"
            print_info "Check logs: journalctl -u postgresql-17 -n 50"
        else
            print_info "Check status: systemctl status postgresql"
            print_info "Check logs: journalctl -u postgresql -n 50"
        fi
    elif [ "$OS" = "macos" ]; then
        print_info "Check status: brew services list | grep postgresql"
        print_info "Check logs: brew services info postgresql@17"
    fi
    exit 1
fi
echo -e "${CHECKMARK} ${DIM}(ready after ${i}s)${RESET}"

print_header "Step 13: PostgreSQL Configuration"

# Run schema file - single source of truth for database structure
# Schema file creates: roles, database, extensions, tables, indexes, RLS policies
echo -ne "${DIM}${ARROW}${RESET} Running database schema (roles, tables, indexes, RLS)... "
SCHEMA_FILE="/opt/mira/app/deploy/mira_service_schema.sql"
if [ -f "$SCHEMA_FILE" ]; then
    if [ "$OS" = "linux" ]; then
        # Run as postgres superuser; schema handles CREATE DATABASE and \c
        if sudo -u postgres psql -f "$SCHEMA_FILE" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to run schema file"
            exit 1
        fi
    elif [ "$OS" = "macos" ]; then
        if psql postgres -f "$SCHEMA_FILE" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to run schema file"
            exit 1
        fi
    fi
else
    echo -e "${ERROR}"
    print_error "Schema file not found: $SCHEMA_FILE"
    exit 1
fi

# Configure LLM endpoints for offline mode (use local Ollama)
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Configuring LLM endpoints for offline mode (Ollama)... "
    OLLAMA_MAIN="${CONFIG_OLLAMA_MODEL:-qwen3:1.7b}"
    OLLAMA_SUB="${CONFIG_OLLAMA_SUBCORTICAL_MODEL:-$OLLAMA_MAIN}"
    OFFLINE_SQL="INSERT INTO conversation_llm (name, model, thinking_budget, description, display_order, provider, endpoint_url, api_key_name, hidden) VALUES ('offline', '${OLLAMA_MAIN}', 0, 'Local (Ollama)', 0, 'generic', 'http://localhost:11434/v1/chat/completions', NULL, FALSE) ON CONFLICT (name) DO UPDATE SET model = '${OLLAMA_MAIN}', endpoint_url = 'http://localhost:11434/v1/chat/completions', api_key_name = NULL; UPDATE users SET conversation_llm = 'offline'; UPDATE internal_llm SET endpoint_url = 'http://localhost:11434/v1/chat/completions', model = '${OLLAMA_MAIN}', api_key_name = NULL WHERE endpoint_url LIKE 'https://%' AND name != 'analysis'; UPDATE internal_llm SET endpoint_url = 'http://localhost:11434/v1/chat/completions', model = '${OLLAMA_SUB}', api_key_name = NULL WHERE name = 'analysis';"
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -d mira_service -c "$OFFLINE_SQL" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_warning "Failed to configure offline mode - you may need to run manually"
        fi
    elif [ "$OS" = "macos" ]; then
        if psql mira_service -c "$OFFLINE_SQL" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_warning "Failed to configure offline mode - you may need to run manually"
        fi
    fi
fi

# Update PostgreSQL passwords if custom password was set
if [ "$CONFIG_DB_PASSWORD" != "changethisifdeployingpwd" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Updating database passwords... "
    if [ "$OS" = "linux" ]; then
        sudo -u postgres psql -c "ALTER USER mira_admin WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1 && \
        sudo -u postgres psql -c "ALTER USER mira_dbuser WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1
    elif [ "$OS" = "macos" ]; then
        psql postgres -c "ALTER USER mira_admin WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1 && \
        psql postgres -c "ALTER USER mira_dbuser WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1
    fi
    if [ $? -eq 0 ]; then
        echo -e "${CHECKMARK}"
    else
        echo -e "${ERROR}"
        print_warning "Failed to update passwords - you may need to update manually"
    fi
fi

print_success "PostgreSQL configured"

print_header "Step 14: Vault Credential Storage"

# Build api_keys arguments based on chat provider type
# Note: mira_api token is generated by the server on first startup via ensure_single_user()
if [ "$CONFIG_CHAT_PROVIDER_TYPE" = "generic" ]; then
    # Generic chat: openaicompat_key = chat key, subcortical_key = subcortical key
    API_KEYS_ARGS="anthropic_key=\"${CONFIG_ANTHROPIC_KEY}\" anthropic_batch_key=\"${CONFIG_ANTHROPIC_BATCH_KEY}\" openaicompat_key=\"${CONFIG_CHAT_API_KEY}\" subcortical_key=\"${CONFIG_SUBCORTICAL_API_KEY}\""
else
    # Anthropic chat: no openaicompat_key needed
    API_KEYS_ARGS="anthropic_key=\"${CONFIG_ANTHROPIC_KEY}\" anthropic_batch_key=\"${CONFIG_ANTHROPIC_BATCH_KEY}\" subcortical_key=\"${CONFIG_SUBCORTICAL_API_KEY}\""
fi
if [ -n "$CONFIG_KAGI_KEY" ]; then
    API_KEYS_ARGS="$API_KEYS_ARGS kagi_api_key=\"${CONFIG_KAGI_KEY}\""
fi
eval vault_put_if_not_exists secret/mira/api_keys $API_KEYS_ARGS

vault_put_if_not_exists secret/mira/database \
    admin_url="postgresql://mira_admin:${CONFIG_DB_PASSWORD}@localhost:5432/mira_service" \
    password="${CONFIG_DB_PASSWORD}" \
    username="mira_dbuser" \
    service_url="postgresql://mira_dbuser:${CONFIG_DB_PASSWORD}@localhost:5432/mira_service"

CONFIG_USERDATA_ENCRYPTION_KEY=$(openssl rand -base64 32)
CONFIG_DIAGNOSTICS_TOKEN=$(openssl rand -base64 32)

vault_put_if_not_exists secret/mira/services \
    app_url="http://localhost:1993" \
    valkey_url="valkey://localhost:6379" \
    userdata_encryption_key="${CONFIG_USERDATA_ENCRYPTION_KEY}" \
    diagnostics_token="${CONFIG_DIAGNOSTICS_TOKEN}"

if ! vault kv get -field=userdata_encryption_key secret/mira/services > /dev/null 2>&1; then
    vault kv patch secret/mira/services userdata_encryption_key="${CONFIG_USERDATA_ENCRYPTION_KEY}" > /dev/null
fi
if ! vault kv get -field=diagnostics_token secret/mira/services > /dev/null 2>&1; then
    vault kv patch secret/mira/services diagnostics_token="${CONFIG_DIAGNOSTICS_TOKEN}" > /dev/null
fi

print_success "All credentials configured in Vault"
