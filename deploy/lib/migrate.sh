# deploy/lib/migrate.sh
# Migration helper functions for MIRA upgrades
# Source this file - do not execute directly
#
# Requires: lib/output.sh, lib/services.sh, lib/vault.sh sourced first
# Requires: OS, LOUD_MODE variables set

# ============================================================================
# Logging Functions
# ============================================================================
# Set up migration logging to capture all output

MIGRATION_LOG_FILE=""
MIGRATE_DB_PASSWORD=""  # Extracted from Vault backup for PostgreSQL auth

setup_migration_logging() {
    local backup_dir="$1"
    MIGRATION_LOG_FILE="${backup_dir}/migration.log"

    # Create log file and write header
    cat > "$MIGRATION_LOG_FILE" << EOF
================================================================================
MIRA Migration Log
Started: $(date '+%Y-%m-%d %H:%M:%S %Z')
Host: $(hostname)
User: $(whoami)
OS: ${OS} ${DISTRO:-}
================================================================================

EOF

    # Tee all output to log file while preserving terminal output
    # Use file descriptor 3 to save original stdout
    exec 3>&1
    exec > >(tee -a "$MIGRATION_LOG_FILE")
    exec 2>&1

    print_info "Logging to: $MIGRATION_LOG_FILE"
}

finalize_migration_log() {
    local status="$1"

    # Append footer to log
    cat >> "$MIGRATION_LOG_FILE" << EOF

================================================================================
Migration ${status}
Completed: $(date '+%Y-%m-%d %H:%M:%S %Z')
================================================================================
EOF
}

# ============================================================================
# Rollback Functions
# ============================================================================
# Emergency rollback if Phase 6 (Fresh Installation) fails midway

# Attempt to restore system from backup after failed installation
# This is a best-effort recovery - manual intervention may still be needed
rollback_from_backup() {
    local backup_dir="$1"

    echo ""
    echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${RED}  INSTALLATION FAILED - ATTEMPTING ROLLBACK${RESET}"
    echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════${RESET}"
    echo ""

    print_info "Backup directory: $backup_dir"
    echo ""

    local rollback_success=true

    # Step 1: Restore old application if it was backed up
    if [ -d "${backup_dir}/old_app" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Restoring old application... "
        if [ -d "/opt/mira/app" ]; then
            sudo rm -rf /opt/mira/app
        fi
        if sudo mv "${backup_dir}/old_app" /opt/mira/app; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to restore old application"
            rollback_success=false
        fi
    else
        print_warning "No old application backup found"
    fi

    # Step 2: Restore Vault data if init keys exist
    if [ -f "${backup_dir}/init-keys.txt" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Restoring Vault init keys... "
        sudo mkdir -p /opt/vault
        if sudo cp "${backup_dir}/init-keys.txt" /opt/vault/init-keys.txt; then
            sudo chmod 600 /opt/vault/init-keys.txt
            [ -f "${backup_dir}/role-id.txt" ] && sudo cp "${backup_dir}/role-id.txt" /opt/vault/
            [ -f "${backup_dir}/secret-id.txt" ] && sudo cp "${backup_dir}/secret-id.txt" /opt/vault/
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            rollback_success=false
        fi
    fi

    # Step 3: Recreate database if it was dropped
    echo -ne "${DIM}${ARROW}${RESET} Checking database state... "
    local db_exists
    if [ "$OS" = "linux" ]; then
        db_exists=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='mira_service'" 2>/dev/null || echo "")
    else
        db_exists=$(psql postgres -tAc "SELECT 1 FROM pg_database WHERE datname='mira_service'" 2>/dev/null || echo "")
    fi

    if [ -z "$db_exists" ] && [ -f "${backup_dir}/postgresql_backup.dump" ]; then
        echo -e "${WARNING} Database missing"
        echo -ne "${DIM}${ARROW}${RESET} Recreating database and restoring data... "

        # Create database
        if [ "$OS" = "linux" ]; then
            sudo -u postgres psql -c "CREATE DATABASE mira_service;" > /dev/null 2>&1 || true
            # Apply schema from backup's old app if available
            if [ -f "${backup_dir}/old_app/deploy/mira_service_schema.sql" ]; then
                sudo -u postgres psql -d mira_service -f "${backup_dir}/old_app/deploy/mira_service_schema.sql" > /dev/null 2>&1 || true
            fi
            # Restore data
            local temp_dump="/tmp/mira_rollback_$$.dump"
            sudo cp "${backup_dir}/postgresql_backup.dump" "$temp_dump"
            sudo chown postgres:postgres "$temp_dump"
            sudo -u postgres pg_restore -d mira_service --data-only --disable-triggers "$temp_dump" 2>/dev/null || true
            sudo rm -f "$temp_dump"
        else
            psql postgres -c "CREATE DATABASE mira_service;" > /dev/null 2>&1 || true
            if [ -f "${backup_dir}/old_app/deploy/mira_service_schema.sql" ]; then
                psql -d mira_service -f "${backup_dir}/old_app/deploy/mira_service_schema.sql" > /dev/null 2>&1 || true
            fi
            PGPASSWORD="$MIGRATE_DB_PASSWORD" pg_restore -U mira_admin -h localhost -d mira_service \
                --data-only --disable-triggers "${backup_dir}/postgresql_backup.dump" 2>/dev/null || true
        fi
        echo -e "${CHECKMARK} ${DIM}(best effort)${RESET}"
    else
        echo -e "${CHECKMARK} ${DIM}(database exists)${RESET}"
    fi

    # Step 4: Restore user data files
    if [ -d "${backup_dir}/user_data" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Restoring user data files... "
        sudo mkdir -p /opt/mira/app/data/users
        if sudo cp -a "${backup_dir}/user_data"/* /opt/mira/app/data/users/ 2>/dev/null; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${WARNING} ${DIM}(may be empty)${RESET}"
        fi
    fi

    echo ""
    if [ "$rollback_success" = true ]; then
        echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${RESET}"
        echo -e "${YELLOW}  ROLLBACK COMPLETED - MANUAL VERIFICATION REQUIRED${RESET}"
        echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${RESET}"
        echo ""
        print_info "The system has been restored to its pre-migration state."
        print_info "You may need to:"
        print_info "  1. Start Vault: sudo systemctl start vault (or manually)"
        print_info "  2. Unseal Vault with keys from: ${backup_dir}/init-keys.txt"
        print_info "  3. Start PostgreSQL if not running"
        print_info "  4. Start MIRA: mira or sudo systemctl start mira"
        echo ""
        print_info "Backup preserved at: $backup_dir"
        print_info "Check migration log: ${backup_dir}/migration.log"
    else
        echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════${RESET}"
        echo -e "${RED}  ROLLBACK FAILED - MANUAL RECOVERY REQUIRED${RESET}"
        echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════${RESET}"
        echo ""
        print_error "Automatic rollback could not fully restore the system."
        print_info "Manual recovery steps:"
        print_info "  1. Restore app: sudo cp -a ${backup_dir}/old_app /opt/mira/app"
        print_info "  2. Restore Vault: sudo cp ${backup_dir}/init-keys.txt /opt/vault/"
        print_info "  3. Restore DB: pg_restore -d mira_service ${backup_dir}/postgresql_backup.dump"
        print_info "  4. Restore user data: sudo cp -a ${backup_dir}/user_data/* /opt/mira/app/data/users/"
    fi

    return 0
}

# ============================================================================
# Dry-Run Mode Functions
# ============================================================================
# When DRY_RUN_MODE=true, show what would happen without making changes

# Check if we're in dry-run mode
is_dry_run() {
    [ "$DRY_RUN_MODE" = true ]
}

# Print dry-run notice for an action
dry_run_notice() {
    local action="$1"
    if is_dry_run; then
        echo -e "${CYAN}[DRY-RUN]${RESET} Would: $action"
        return 0
    fi
    return 1
}

# Skip destructive action in dry-run mode
dry_run_skip() {
    local action="$1"
    if is_dry_run; then
        echo -e "${CYAN}[DRY-RUN]${RESET} Skipping: $action"
        return 0
    fi
    return 1
}

# ============================================================================
# Backup Verification Functions
# ============================================================================
# Verify backups are valid before proceeding with destructive operations

verify_postgresql_backup() {
    local backup_file="$1"

    echo -ne "${DIM}${ARROW}${RESET} Verifying PostgreSQL backup integrity... "

    if [ ! -f "$backup_file" ]; then
        echo -e "${ERROR}"
        print_error "Backup file not found: $backup_file"
        return 1
    fi

    # Check file size (should be non-zero)
    local file_size
    file_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null)
    if [ "$file_size" -eq 0 ]; then
        echo -e "${ERROR}"
        print_error "Backup file is empty"
        return 1
    fi

    # Verify pg_dump format is readable with pg_restore --list
    if ! pg_restore --list "$backup_file" > /dev/null 2>&1; then
        echo -e "${ERROR}"
        print_error "Backup file is corrupted or unreadable"
        return 1
    fi

    # Count objects in backup
    local object_count
    object_count=$(pg_restore --list "$backup_file" 2>/dev/null | grep -c "^[0-9]" || echo "0")

    local size_human
    if [ "$file_size" -gt 1048576 ]; then
        size_human="$((file_size / 1048576)) MB"
    else
        size_human="$((file_size / 1024)) KB"
    fi

    echo -e "${CHECKMARK} ${DIM}($size_human, $object_count objects)${RESET}"
    return 0
}

verify_vault_backup() {
    local backup_dir="$1"

    echo -ne "${DIM}${ARROW}${RESET} Verifying Vault backup integrity... "

    local files_valid=0
    local files_checked=0

    for json_file in "${backup_dir}"/vault_*.json; do
        [ ! -f "$json_file" ] && continue
        files_checked=$((files_checked + 1))

        # Verify it's valid JSON
        if jq -e '.' "$json_file" > /dev/null 2>&1; then
            files_valid=$((files_valid + 1))
        else
            echo -e "${ERROR}"
            print_error "Invalid JSON in: $(basename "$json_file")"
            return 1
        fi
    done

    # Verify snapshot file exists and is valid
    if [ -f "${backup_dir}/vault_snapshot.json" ]; then
        if ! jq -e '.' "${backup_dir}/vault_snapshot.json" > /dev/null 2>&1; then
            echo -e "${ERROR}"
            print_error "Vault snapshot is corrupted"
            return 1
        fi
        files_valid=$((files_valid + 1))
        files_checked=$((files_checked + 1))
    fi

    if [ "$files_checked" -eq 0 ]; then
        echo -e "${WARNING}"
        print_warning "No Vault backup files found"
        return 0
    fi

    echo -e "${CHECKMARK} ${DIM}($files_valid/$files_checked files valid)${RESET}"
    return 0
}

verify_database_backup() {
    local backup_dir="$1"

    echo -ne "${DIM}${ARROW}${RESET} Verifying database snapshot integrity... "

    local snapshot_file="${backup_dir}/db_snapshot.json"

    if [ ! -f "$snapshot_file" ]; then
        echo -e "${ERROR}"
        print_error "Database snapshot not found"
        return 1
    fi

    # Verify it's valid JSON with expected structure
    if ! jq -e '.row_counts and .structural_data' "$snapshot_file" > /dev/null 2>&1; then
        echo -e "${ERROR}"
        print_error "Database snapshot is malformed"
        return 1
    fi

    # Verify checksum file exists
    if [ ! -f "${snapshot_file}.sha256" ]; then
        echo -e "${WARNING}"
        print_warning "Checksum file missing (verification may be less reliable)"
    fi

    local table_count
    table_count=$(jq '.row_counts | keys | length' "$snapshot_file" 2>/dev/null || echo "0")

    echo -e "${CHECKMARK} ${DIM}($table_count tables captured)${RESET}"
    return 0
}

# Master verification function - verify all backups before destructive ops
verify_all_backups() {
    print_header "Verifying Backup Integrity"

    local all_valid=true

    verify_postgresql_backup "${BACKUP_DIR}/postgresql_backup.dump" || all_valid=false
    verify_vault_backup "$BACKUP_DIR" || all_valid=false
    verify_database_backup "$BACKUP_DIR" || all_valid=false

    if [ "$all_valid" = true ]; then
        print_success "All backups verified successfully"
        return 0
    else
        print_error "Backup verification failed - aborting migration"
        print_info "Fix backup issues before proceeding with destructive operations"
        return 1
    fi
}

# ============================================================================
# Vault Snapshot Functions (Critical for Data Integrity)
# ============================================================================
# These functions capture the COMPLETE Vault state and verify it after restore.
# This guarantees no secrets are lost during migration.

# Global variable to hold the pre-migration Vault snapshot
VAULT_SNAPSHOT=""

# Capture complete Vault tree as normalized JSON for comparison
# This enumerates ALL paths under secret/mira/ and captures every field
capture_vault_snapshot() {
    local snapshot_file="$1"

    echo -ne "${DIM}${ARROW}${RESET} Capturing complete Vault snapshot... "

    # Ensure authenticated
    vault_authenticate > /dev/null 2>&1 || {
        echo -e "${ERROR}"
        print_error "Cannot authenticate with Vault for snapshot"
        return 1
    }

    # Start JSON object
    echo "{" > "$snapshot_file"

    # Get list of all secret paths under secret/mira/
    local paths
    paths=$(vault kv list -format=json secret/mira 2>/dev/null | jq -r '.[]' 2>/dev/null || echo "")

    if [ -z "$paths" ]; then
        # No paths found - might be empty or error
        echo "  \"_empty\": true" >> "$snapshot_file"
        echo "}" >> "$snapshot_file"
        echo -e "${CHECKMARK} ${DIM}(empty vault)${RESET}"
        return 0
    fi

    local first=true
    local path_count=0

    for path in $paths; do
        # Remove trailing slash if present
        path="${path%/}"

        # Get the secret data (extract .data.data for KV v2)
        local secret_data
        secret_data=$(vault kv get -format=json "secret/mira/${path}" 2>/dev/null | jq -c '.data.data // {}' 2>/dev/null || echo "{}")

        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$snapshot_file"
        fi

        # Write path and its data
        echo "  \"${path}\": ${secret_data}" >> "$snapshot_file"
        path_count=$((path_count + 1))
    done

    echo "" >> "$snapshot_file"
    echo "}" >> "$snapshot_file"

    # Calculate checksum for quick comparison
    local checksum
    if command -v sha256sum &> /dev/null; then
        checksum=$(jq -cS '.' "$snapshot_file" | sha256sum | cut -d' ' -f1)
    else
        # macOS uses shasum
        checksum=$(jq -cS '.' "$snapshot_file" | shasum -a 256 | cut -d' ' -f1)
    fi

    # Store checksum in separate file
    echo "$checksum" > "${snapshot_file}.sha256"

    echo -e "${CHECKMARK} ${DIM}($path_count paths, checksum: ${checksum:0:12}...)${RESET}"
    return 0
}

# Verify restored Vault matches the pre-migration snapshot exactly
# If differences are found, show colored diff and require user confirmation for each
verify_vault_snapshot() {
    local original_snapshot="$1"
    local current_snapshot="${BACKUP_DIR}/vault_current_snapshot.json"

    echo -ne "${DIM}${ARROW}${RESET} Verifying Vault integrity... "

    # Capture current Vault state
    capture_vault_snapshot "$current_snapshot" > /dev/null 2>&1 || {
        echo -e "${ERROR}"
        print_error "Failed to capture current Vault state for comparison"
        return 1
    }

    # Compare checksums first (fast path)
    local original_checksum current_checksum
    original_checksum=$(cat "${original_snapshot}.sha256" 2>/dev/null || echo "")
    current_checksum=$(cat "${current_snapshot}.sha256" 2>/dev/null || echo "")

    if [ "$original_checksum" = "$current_checksum" ] && [ -n "$original_checksum" ]; then
        echo -e "${CHECKMARK} ${DIM}(checksums match: ${original_checksum:0:12}...)${RESET}"
        return 0
    fi

    # Checksums differ - do detailed comparison with colored output
    echo -e "${WARNING}"
    echo ""
    echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${YELLOW}  VAULT INTEGRITY CHECK: Differences Detected${RESET}"
    echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${RESET}"
    echo ""
    echo -e "${DIM}Original checksum: ${original_checksum}${RESET}"
    echo -e "${DIM}Current checksum:  ${current_checksum}${RESET}"
    echo ""

    # Compare each path
    local original_paths current_paths
    original_paths=$(jq -r 'keys[]' "$original_snapshot" 2>/dev/null | grep -v "^_empty$" | sort)
    current_paths=$(jq -r 'keys[]' "$current_snapshot" 2>/dev/null | grep -v "^_empty$" | sort)

    local has_differences=false
    local user_confirmed_all=true

    # Check for missing paths (RED - these are critical)
    local missing_paths
    missing_paths=$(comm -23 <(echo "$original_paths") <(echo "$current_paths") 2>/dev/null || echo "")
    if [ -n "$missing_paths" ] && [ "$missing_paths" != "" ]; then
        has_differences=true
        echo -e "${BOLD}${RED}╔══ MISSING SECRET PATHS (were in original, not in restored) ══╗${RESET}"
        echo "$missing_paths" | while read path; do
            [ -z "$path" ] && continue
            echo -e "${RED}  ▸ secret/mira/${path}${RESET}"

            # Show what fields were in this path
            local fields
            fields=$(jq -r ".\"$path\" | keys[]" "$original_snapshot" 2>/dev/null || echo "")
            if [ -n "$fields" ]; then
                echo -e "${DIM}${RED}    Fields lost: $(echo $fields | tr '\n' ', ' | sed 's/, $//')${RESET}"
            fi
        done
        echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${RESET}"
        echo ""

        echo -e "${BOLD}${RED}CRITICAL: Missing secrets detected!${RESET}"
        read -p "$(echo -e ${YELLOW}Acknowledge this data loss and continue anyway?${RESET}) (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo -e "${RED}Migration aborted by user.${RESET}"
            user_confirmed_all=false
        fi
        echo ""
    fi

    # Check for extra paths (GREEN - unexpected but not critical)
    local extra_paths
    extra_paths=$(comm -13 <(echo "$original_paths") <(echo "$current_paths") 2>/dev/null || echo "")
    if [ -n "$extra_paths" ] && [ "$extra_paths" != "" ]; then
        has_differences=true
        echo -e "${BOLD}${GREEN}╔══ EXTRA SECRET PATHS (new in restored, not in original) ══╗${RESET}"
        echo "$extra_paths" | while read path; do
            [ -z "$path" ] && continue
            echo -e "${GREEN}  ▸ secret/mira/${path}${RESET}"
        done
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${RESET}"
        echo ""

        read -p "$(echo -e ${YELLOW}Acknowledge extra paths and continue?${RESET}) (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo -e "${RED}Migration aborted by user.${RESET}"
            user_confirmed_all=false
        fi
        echo ""
    fi

    # Check each common path for field differences
    local common_paths
    common_paths=$(comm -12 <(echo "$original_paths") <(echo "$current_paths") 2>/dev/null || echo "")

    if [ -n "$common_paths" ]; then
        for path in $common_paths; do
            [ -z "$path" ] && continue
            [ "$path" = "_empty" ] && continue

            local original_data current_data
            original_data=$(jq -cS ".\"$path\"" "$original_snapshot" 2>/dev/null)
            current_data=$(jq -cS ".\"$path\"" "$current_snapshot" 2>/dev/null)

            if [ "$original_data" != "$current_data" ]; then
                has_differences=true

                echo -e "${BOLD}${YELLOW}╔══ FIELD DIFFERENCES: secret/mira/${path} ══╗${RESET}"

                # Get field lists
                local original_keys current_keys
                original_keys=$(echo "$original_data" | jq -r 'keys[]' 2>/dev/null | sort)
                current_keys=$(echo "$current_data" | jq -r 'keys[]' 2>/dev/null | sort)

                # Missing fields (RED)
                local missing_fields
                missing_fields=$(comm -23 <(echo "$original_keys") <(echo "$current_keys") 2>/dev/null || echo "")
                if [ -n "$missing_fields" ]; then
                    for field in $missing_fields; do
                        [ -z "$field" ] && continue
                        echo -e "${RED}  - ${field} (REMOVED)${RESET}"
                    done
                fi

                # Extra fields (GREEN)
                local extra_fields
                extra_fields=$(comm -13 <(echo "$original_keys") <(echo "$current_keys") 2>/dev/null || echo "")
                if [ -n "$extra_fields" ]; then
                    for field in $extra_fields; do
                        [ -z "$field" ] && continue
                        echo -e "${GREEN}  + ${field} (ADDED)${RESET}"
                    done
                fi

                # Changed fields (YELLOW)
                local common_fields
                common_fields=$(comm -12 <(echo "$original_keys") <(echo "$current_keys") 2>/dev/null || echo "")
                for field in $common_fields; do
                    [ -z "$field" ] && continue
                    local orig_val curr_val
                    orig_val=$(echo "$original_data" | jq -r ".\"$field\"" 2>/dev/null)
                    curr_val=$(echo "$current_data" | jq -r ".\"$field\"" 2>/dev/null)
                    if [ "$orig_val" != "$curr_val" ]; then
                        # Show that value changed but don't show actual values (sensitive)
                        local orig_len=${#orig_val}
                        local curr_len=${#curr_val}
                        echo -e "${YELLOW}  ~ ${field} (VALUE CHANGED: ${orig_len} chars → ${curr_len} chars)${RESET}"
                    fi
                done

                echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${RESET}"
                echo ""

                read -p "$(echo -e ${YELLOW}Acknowledge changes to secret/mira/${path} and continue?${RESET}) (yes/no): " confirm
                if [ "$confirm" != "yes" ]; then
                    echo -e "${RED}Migration aborted by user.${RESET}"
                    user_confirmed_all=false
                    break
                fi
                echo ""
            fi
        done
    fi

    # Final verdict
    if [ "$user_confirmed_all" = false ]; then
        echo ""
        print_error "Migration aborted due to unconfirmed Vault differences"
        print_info "Original snapshot preserved at: $original_snapshot"
        print_info "Backup JSON files available at: ${BACKUP_DIR}/vault_*.json"
        return 1
    fi

    if [ "$has_differences" = true ]; then
        echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${RESET}"
        echo -e "${GREEN}  All Vault differences acknowledged by user${RESET}"
        echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${RESET}"
        echo ""
        return 0
    fi

    # If we get here with different checksums but no detected diffs,
    # it might be ordering/formatting - warn but don't fail
    print_warning "Checksums differ but no structural differences detected (may be JSON formatting)"
    return 0
}

# ============================================================================
# Database Snapshot Functions (Critical for Data Integrity)
# ============================================================================
# These functions capture structural database state and verify after restore.
# We compare structural data (users, relationships) but ignore timestamps.

# Tables to snapshot with their structural columns (excluding timestamps)
# Format: "table:order_by_col:col1,col2,col3"
# order_by_col is the column to ORDER BY (usually primary key)
DB_STRUCTURAL_TABLES=(
    "users:id:id,email,first_name,last_name,timezone,conversation_llm,is_active,memory_manipulation_enabled,cumulative_activity_days"
    "continuums:id:id,user_id,last_message_position"
    "api_tokens:id:id,user_id,name,token_hash"
    "domain_knowledge_blocks:id:id,user_id,domain_label,domain_name,enabled"
    "entities:id:id,user_id,name,entity_type"
    "conversation_llm:name:name,model,dialect_name,endpoint_url,api_key_name"
    "internal_llm:name:name,model,endpoint_url,api_key_name"
)

# Tables where we only verify counts (too large or timestamps dominate)
DB_COUNT_ONLY_TABLES=(
    "messages"
    "memories"
    "user_activity_days"
)

# Helper: Run psql query based on OS
# Uses MIGRATE_DB_PASSWORD on macOS if available (set by backup_vault_secrets)
run_psql() {
    local query="$1"
    if [ "$OS" = "linux" ]; then
        sudo -u postgres psql -d mira_service -tAc "$query" 2>/dev/null
    else
        # Use PGPASSWORD if MIGRATE_DB_PASSWORD is set, otherwise rely on .pgpass/trust
        if [ -n "$MIGRATE_DB_PASSWORD" ]; then
            PGPASSWORD="$MIGRATE_DB_PASSWORD" psql -U mira_admin -h localhost -d mira_service -tAc "$query" 2>/dev/null
        else
            psql -U mira_admin -h localhost -d mira_service -tAc "$query" 2>/dev/null
        fi
    fi
}

# Capture database snapshot to JSON file
capture_database_snapshot() {
    local snapshot_file="$1"

    echo -ne "${DIM}${ARROW}${RESET} Capturing database snapshot... "

    # Start JSON
    echo "{" > "$snapshot_file"
    echo '  "row_counts": {' >> "$snapshot_file"

    # Capture row counts for all backed up tables (must match backup_postgresql_data tables)
    # Includes conversation_llm and internal_llm because offline mode customizes them
    local all_tables="users continuums messages memories entities api_tokens user_activity_days domain_knowledge_blocks domain_knowledge_block_content extraction_batches post_processing_batches conversation_llm internal_llm"
    local first=true

    for table in $all_tables; do
        local count
        count=$(run_psql "SELECT COUNT(*) FROM $table" 2>/dev/null || echo "0")
        count=$(echo "$count" | tr -d ' \n')

        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$snapshot_file"
        fi
        echo -n "    \"$table\": $count" >> "$snapshot_file"
    done

    echo "" >> "$snapshot_file"
    echo "  }," >> "$snapshot_file"

    # Capture structural data for key tables
    echo '  "structural_data": {' >> "$snapshot_file"

    first=true
    for table_spec in "${DB_STRUCTURAL_TABLES[@]}"; do
        # Parse format: "table:order_by_col:col1,col2,col3"
        local table="${table_spec%%:*}"
        local rest="${table_spec#*:}"
        local order_col="${rest%%:*}"
        local columns="${rest#*:}"

        # Query structural columns, order by specified column for consistent comparison
        local data
        data=$(run_psql "SELECT json_agg(row_to_json(t) ORDER BY ${order_col}) FROM (SELECT $columns FROM $table ORDER BY ${order_col}) t" 2>/dev/null || echo "[]")

        # Handle null result
        if [ -z "$data" ] || [ "$data" = "" ]; then
            data="[]"
        fi

        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$snapshot_file"
        fi
        echo "    \"$table\": $data" >> "$snapshot_file"
    done

    echo "" >> "$snapshot_file"
    echo "  }," >> "$snapshot_file"

    # Capture sample IDs from large tables for spot-check verification
    echo '  "sample_ids": {' >> "$snapshot_file"

    first=true
    for table in "${DB_COUNT_ONLY_TABLES[@]}"; do
        # Get first 10 and last 10 IDs for verification
        local first_ids last_ids
        first_ids=$(run_psql "SELECT json_agg(id) FROM (SELECT id FROM $table ORDER BY created_at ASC LIMIT 10) t" 2>/dev/null || echo "[]")
        last_ids=$(run_psql "SELECT json_agg(id) FROM (SELECT id FROM $table ORDER BY created_at DESC LIMIT 10) t" 2>/dev/null || echo "[]")

        [ -z "$first_ids" ] || [ "$first_ids" = "" ] && first_ids="[]"
        [ -z "$last_ids" ] || [ "$last_ids" = "" ] && last_ids="[]"

        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$snapshot_file"
        fi
        echo "    \"${table}\": {\"first\": $first_ids, \"last\": $last_ids}" >> "$snapshot_file"
    done

    echo "" >> "$snapshot_file"
    echo "  }" >> "$snapshot_file"
    echo "}" >> "$snapshot_file"

    # Calculate checksum
    local checksum
    if command -v sha256sum &> /dev/null; then
        checksum=$(jq -cS '.' "$snapshot_file" 2>/dev/null | sha256sum | cut -d' ' -f1)
    else
        checksum=$(jq -cS '.' "$snapshot_file" 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
    fi
    echo "$checksum" > "${snapshot_file}.sha256"

    local table_count=$(echo "$all_tables" | wc -w | tr -d ' ')
    echo -e "${CHECKMARK} ${DIM}($table_count tables, checksum: ${checksum:0:12}...)${RESET}"
    return 0
}

# Verify database matches pre-migration snapshot
# Shows colored diff and requires confirmation for each discrepancy
verify_database_snapshot() {
    local original_snapshot="$1"
    local current_snapshot="${BACKUP_DIR}/db_current_snapshot.json"

    echo -ne "${DIM}${ARROW}${RESET} Verifying database integrity... "

    # Capture current state
    capture_database_snapshot "$current_snapshot" > /dev/null 2>&1 || {
        echo -e "${ERROR}"
        print_error "Failed to capture current database state"
        return 1
    }

    # Compare row counts first
    local original_counts current_counts
    original_counts=$(jq -c '.row_counts' "$original_snapshot" 2>/dev/null)
    current_counts=$(jq -c '.row_counts' "$current_snapshot" 2>/dev/null)

    if [ "$original_counts" = "$current_counts" ]; then
        # Quick check structural data
        local original_struct current_struct
        original_struct=$(jq -cS '.structural_data' "$original_snapshot" 2>/dev/null)
        current_struct=$(jq -cS '.structural_data' "$current_snapshot" 2>/dev/null)

        if [ "$original_struct" = "$current_struct" ]; then
            echo -e "${CHECKMARK} ${DIM}(all counts and structures match)${RESET}"
            return 0
        fi
    fi

    # Differences detected - show detailed comparison
    echo -e "${WARNING}"
    echo ""
    echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${YELLOW}  DATABASE INTEGRITY CHECK: Differences Detected${RESET}"
    echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${RESET}"
    echo ""

    local has_differences=false
    local user_confirmed_all=true

    # Compare row counts
    echo -e "${BOLD}Row Count Comparison:${RESET}"
    echo ""

    local tables
    tables=$(jq -r '.row_counts | keys[]' "$original_snapshot" 2>/dev/null)

    for table in $tables; do
        local orig_count curr_count
        orig_count=$(jq -r ".row_counts.\"$table\"" "$original_snapshot" 2>/dev/null)
        curr_count=$(jq -r ".row_counts.\"$table\"" "$current_snapshot" 2>/dev/null)

        if [ "$orig_count" != "$curr_count" ]; then
            has_differences=true
            local diff=$((curr_count - orig_count))
            local color="$YELLOW"
            local symbol="~"

            if [ "$diff" -lt 0 ]; then
                color="$RED"
                symbol="-"
                echo -e "${color}  ${symbol} ${table}: ${orig_count} → ${curr_count} (${diff} rows LOST)${RESET}"
            else
                color="$GREEN"
                symbol="+"
                echo -e "${color}  ${symbol} ${table}: ${orig_count} → ${curr_count} (+${diff} rows)${RESET}"
            fi
        else
            echo -e "${DIM}  ✓ ${table}: ${orig_count} rows${RESET}"
        fi
    done
    echo ""

    # If row counts differ, require confirmation
    if [ "$has_differences" = true ]; then
        # Check if any critical data was lost
        local users_orig users_curr
        users_orig=$(jq -r '.row_counts.users' "$original_snapshot" 2>/dev/null)
        users_curr=$(jq -r '.row_counts.users' "$current_snapshot" 2>/dev/null)

        if [ "$users_curr" -lt "$users_orig" ]; then
            echo -e "${BOLD}${RED}╔══ CRITICAL: USER DATA LOSS DETECTED ══╗${RESET}"
            echo -e "${RED}  Original users: $users_orig${RESET}"
            echo -e "${RED}  Current users:  $users_curr${RESET}"
            echo -e "${RED}  LOST: $((users_orig - users_curr)) user accounts${RESET}"
            echo -e "${RED}╚════════════════════════════════════════╝${RESET}"
            echo ""

            read -p "$(echo -e ${YELLOW}CRITICAL: Acknowledge USER DATA LOSS and continue?${RESET}) (yes/no): " confirm
            if [ "$confirm" != "yes" ]; then
                echo -e "${RED}Migration aborted by user.${RESET}"
                user_confirmed_all=false
            fi
            echo ""
        fi

        local msgs_orig msgs_curr
        msgs_orig=$(jq -r '.row_counts.messages' "$original_snapshot" 2>/dev/null)
        msgs_curr=$(jq -r '.row_counts.messages' "$current_snapshot" 2>/dev/null)

        if [ "$msgs_curr" -lt "$msgs_orig" ] && [ "$user_confirmed_all" = true ]; then
            local lost=$((msgs_orig - msgs_curr))
            echo -e "${BOLD}${RED}╔══ MESSAGE LOSS DETECTED ══╗${RESET}"
            echo -e "${RED}  Original: $msgs_orig messages${RESET}"
            echo -e "${RED}  Current:  $msgs_curr messages${RESET}"
            echo -e "${RED}  LOST: $lost messages${RESET}"
            echo -e "${RED}╚═══════════════════════════╝${RESET}"
            echo ""

            read -p "$(echo -e ${YELLOW}Acknowledge message loss and continue?${RESET}) (yes/no): " confirm
            if [ "$confirm" != "yes" ]; then
                echo -e "${RED}Migration aborted by user.${RESET}"
                user_confirmed_all=false
            fi
            echo ""
        fi

        local mems_orig mems_curr
        mems_orig=$(jq -r '.row_counts.memories' "$original_snapshot" 2>/dev/null)
        mems_curr=$(jq -r '.row_counts.memories' "$current_snapshot" 2>/dev/null)

        if [ "$mems_curr" -lt "$mems_orig" ] && [ "$user_confirmed_all" = true ]; then
            local lost=$((mems_orig - mems_curr))
            echo -e "${BOLD}${RED}╔══ MEMORY LOSS DETECTED ══╗${RESET}"
            echo -e "${RED}  Original: $mems_orig memories${RESET}"
            echo -e "${RED}  Current:  $mems_curr memories${RESET}"
            echo -e "${RED}  LOST: $lost memories${RESET}"
            echo -e "${RED}╚══════════════════════════╝${RESET}"
            echo ""

            read -p "$(echo -e ${YELLOW}Acknowledge memory loss and continue?${RESET}) (yes/no): " confirm
            if [ "$confirm" != "yes" ]; then
                echo -e "${RED}Migration aborted by user.${RESET}"
                user_confirmed_all=false
            fi
            echo ""
        fi
    fi

    # Compare structural data (user details, etc.)
    if [ "$user_confirmed_all" = true ]; then
        echo -e "${BOLD}Structural Data Comparison:${RESET}"
        echo ""

        for table_spec in "${DB_STRUCTURAL_TABLES[@]}"; do
            local table="${table_spec%%:*}"

            local orig_data curr_data
            orig_data=$(jq -cS ".structural_data.\"$table\"" "$original_snapshot" 2>/dev/null)
            curr_data=$(jq -cS ".structural_data.\"$table\"" "$current_snapshot" 2>/dev/null)

            if [ "$orig_data" != "$curr_data" ]; then
                has_differences=true

                echo -e "${BOLD}${YELLOW}╔══ STRUCTURAL DIFFERENCES: ${table} ══╗${RESET}"

                # For users table, show specific field differences
                if [ "$table" = "users" ]; then
                    local orig_emails curr_emails
                    orig_emails=$(echo "$orig_data" | jq -r '.[].email' 2>/dev/null | sort)
                    curr_emails=$(echo "$curr_data" | jq -r '.[].email' 2>/dev/null | sort)

                    # Missing users
                    local missing_users
                    missing_users=$(comm -23 <(echo "$orig_emails") <(echo "$curr_emails") 2>/dev/null)
                    if [ -n "$missing_users" ]; then
                        echo "$missing_users" | while read email; do
                            [ -z "$email" ] && continue
                            echo -e "${RED}  - User MISSING: ${email}${RESET}"
                        done
                    fi

                    # New users
                    local new_users
                    new_users=$(comm -13 <(echo "$orig_emails") <(echo "$curr_emails") 2>/dev/null)
                    if [ -n "$new_users" ]; then
                        echo "$new_users" | while read email; do
                            [ -z "$email" ] && continue
                            echo -e "${GREEN}  + User ADDED: ${email}${RESET}"
                        done
                    fi

                    # Changed user data (same email, different fields)
                    local common_emails
                    common_emails=$(comm -12 <(echo "$orig_emails") <(echo "$curr_emails") 2>/dev/null)
                    for email in $common_emails; do
                        [ -z "$email" ] && continue
                        local orig_user curr_user
                        orig_user=$(echo "$orig_data" | jq -c ".[] | select(.email == \"$email\")" 2>/dev/null)
                        curr_user=$(echo "$curr_data" | jq -c ".[] | select(.email == \"$email\")" 2>/dev/null)

                        if [ "$orig_user" != "$curr_user" ]; then
                            echo -e "${YELLOW}  ~ User CHANGED: ${email}${RESET}"
                            # Show which fields changed
                            local fields="timezone conversation_llm first_name last_name"
                            for field in $fields; do
                                local orig_val curr_val
                                orig_val=$(echo "$orig_user" | jq -r ".$field // empty" 2>/dev/null)
                                curr_val=$(echo "$curr_user" | jq -r ".$field // empty" 2>/dev/null)
                                if [ "$orig_val" != "$curr_val" ]; then
                                    echo -e "${DIM}${YELLOW}      $field: \"$orig_val\" → \"$curr_val\"${RESET}"
                                fi
                            done
                        fi
                    done
                else
                    # Generic table diff - just show counts changed
                    local orig_count curr_count
                    orig_count=$(echo "$orig_data" | jq 'length' 2>/dev/null || echo "0")
                    curr_count=$(echo "$curr_data" | jq 'length' 2>/dev/null || echo "0")
                    echo -e "${YELLOW}  Rows: $orig_count → $curr_count${RESET}"
                fi

                echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${RESET}"
                echo ""

                read -p "$(echo -e ${YELLOW}Acknowledge changes to ${table} and continue?${RESET}) (yes/no): " confirm
                if [ "$confirm" != "yes" ]; then
                    echo -e "${RED}Migration aborted by user.${RESET}"
                    user_confirmed_all=false
                    break
                fi
                echo ""
            else
                echo -e "${DIM}  ✓ ${table}: structural data matches${RESET}"
            fi
        done
    fi

    # Final verdict
    if [ "$user_confirmed_all" = false ]; then
        echo ""
        print_error "Migration aborted due to unconfirmed database differences"
        print_info "Original snapshot: $original_snapshot"
        print_info "Database backup: ${BACKUP_DIR}/postgresql_backup.dump"
        return 1
    fi

    if [ "$has_differences" = true ]; then
        echo ""
        echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${RESET}"
        echo -e "${GREEN}  All database differences acknowledged by user${RESET}"
        echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${RESET}"
        echo ""
    fi

    return 0
}

# ============================================================================
# Pre-flight Validation Functions
# ============================================================================

# Check for existing MIRA installation
migrate_check_existing_install() {
    echo -ne "${DIM}${ARROW}${RESET} Checking existing installation... "

    if [ ! -d "/opt/mira/app" ]; then
        echo -e "${ERROR}"
        print_error "No MIRA installation found at /opt/mira/app"
        print_info "Use 'deploy.sh' without --migrate for fresh installation"
        return 1
    fi

    echo -e "${CHECKMARK}"
    return 0
}

# Check PostgreSQL connectivity
migrate_check_postgresql_running() {
    echo -ne "${DIM}${ARROW}${RESET} Checking PostgreSQL connectivity... "

    if [ "$OS" = "linux" ]; then
        if ! sudo -u postgres psql -d mira_service -c "SELECT 1" > /dev/null 2>&1; then
            echo -e "${ERROR}"
            print_error "Cannot connect to PostgreSQL mira_service database"
            print_info "Ensure PostgreSQL is running and database exists"
            return 1
        fi
    elif [ "$OS" = "macos" ]; then
        if ! psql -U mira_admin -h localhost -d mira_service -c "SELECT 1" > /dev/null 2>&1; then
            echo -e "${ERROR}"
            print_error "Cannot connect to PostgreSQL as mira_admin"
            print_info "Ensure PostgreSQL is running and mira_service database exists"
            return 1
        fi
    fi

    echo -e "${CHECKMARK}"
    return 0
}

# Check Vault accessibility and secrets
migrate_check_vault_accessible() {
    echo -ne "${DIM}${ARROW}${RESET} Checking Vault accessibility... "

    export VAULT_ADDR='http://127.0.0.1:8200'

    # Check Vault is running
    if ! curl -s http://127.0.0.1:8200/v1/sys/health > /dev/null 2>&1; then
        echo -e "${ERROR}"
        print_error "Vault not accessible at http://127.0.0.1:8200"
        print_info "Start Vault before migration"
        return 1
    fi

    # Check Vault is unsealed
    if vault_is_sealed; then
        echo -e "${WARNING}"
        print_warning "Vault is sealed - attempting unseal"
        vault_unseal || {
            print_error "Cannot unseal Vault - check /opt/vault/init-keys.txt"
            return 1
        }
        echo -ne "${DIM}${ARROW}${RESET} Re-checking Vault accessibility... "
    fi

    # Authenticate and verify secret access
    vault_authenticate > /dev/null 2>&1 || {
        echo -e "${ERROR}"
        print_error "Cannot authenticate with Vault"
        return 1
    }

    if ! vault kv get secret/mira/api_keys > /dev/null 2>&1; then
        echo -e "${ERROR}"
        print_error "Cannot read Vault secrets - authentication issue"
        return 1
    fi

    echo -e "${CHECKMARK}"
    return 0
}

# Check disk space for backup
migrate_check_disk_space() {
    echo -ne "${DIM}${ARROW}${RESET} Checking disk space... "

    # Get PostgreSQL database size in bytes
    local pg_size
    if [ "$OS" = "linux" ]; then
        pg_size=$(sudo -u postgres psql -d mira_service -tAc "SELECT pg_database_size('mira_service')" 2>/dev/null || echo "0")
    else
        pg_size=$(psql -U mira_admin -h localhost -d mira_service -tAc "SELECT pg_database_size('mira_service')" 2>/dev/null || echo "0")
    fi

    # Get Vault data size in KB
    local vault_size_kb
    vault_size_kb=$(du -sk /opt/vault/data 2>/dev/null | cut -f1 || echo "0")

    # Get user data size in KB
    local user_size_kb
    user_size_kb=$(du -sk /opt/mira/app/data/users 2>/dev/null | cut -f1 || echo "0")

    # Get MIRA app size in KB (for backup)
    local app_size_kb
    app_size_kb=$(du -sk /opt/mira/app 2>/dev/null | cut -f1 || echo "0")

    # Calculate total needed (convert pg_size from bytes to KB)
    local pg_size_kb=$((pg_size / 1024))
    local total_kb=$((pg_size_kb + vault_size_kb + user_size_kb + app_size_kb))
    local required_kb=$((total_kb * 2))  # 2x for backup headroom

    # Get available space at /opt
    local available_kb
    if [ "$OS" = "macos" ]; then
        available_kb=$(df -k /opt 2>/dev/null | tail -1 | awk '{print $4}' || df -k / | tail -1 | awk '{print $4}')
    else
        available_kb=$(df -k /opt 2>/dev/null | tail -1 | awk '{print $4}')
    fi

    if [ "$available_kb" -lt "$required_kb" ]; then
        echo -e "${ERROR}"
        print_error "Insufficient disk space"
        print_info "Required: $((required_kb / 1024)) MB, Available: $((available_kb / 1024)) MB"
        return 1
    fi

    echo -e "${CHECKMARK} ${DIM}($((required_kb / 1024)) MB needed, $((available_kb / 1024)) MB available)${RESET}"
    return 0
}

# Check for active database sessions
migrate_check_no_active_sessions() {
    echo -ne "${DIM}${ARROW}${RESET} Checking for active sessions... "

    local active
    if [ "$OS" = "linux" ]; then
        active=$(sudo -u postgres psql -d mira_service -tAc \
            "SELECT COUNT(*) FROM pg_stat_activity WHERE datname='mira_service' AND state='active'" 2>/dev/null || echo "0")
    else
        active=$(psql -U mira_admin -h localhost -d mira_service -tAc \
            "SELECT COUNT(*) FROM pg_stat_activity WHERE datname='mira_service' AND state='active'" 2>/dev/null || echo "0")
    fi

    if [ "$active" -gt 1 ]; then
        echo -e "${WARNING}"
        print_warning "Active database connections detected ($active)"
        print_info "Ensure MIRA application is stopped before migration"
        # Continue - just warn, don't fail
    else
        echo -e "${CHECKMARK}"
    fi

    return 0
}

# ============================================================================
# Backup Functions
# ============================================================================

# Backup PostgreSQL user data tables
backup_postgresql_data() {
    local backup_file="${BACKUP_DIR}/postgresql_backup.dump"
    local error_file="${BACKUP_DIR}/pg_dump_errors.log"

    echo -ne "${DIM}${ARROW}${RESET} Backing up PostgreSQL data... "

    # Tables to backup (user data + system config that may be customized)
    # conversation_llm and internal_llm are included because offline mode customizes them
    local tables="users continuums messages memories entities user_activity_days domain_knowledge_blocks domain_knowledge_block_content api_tokens extraction_batches post_processing_batches conversation_llm internal_llm"

    # Build table arguments for pg_dump
    local table_args=""
    for table in $tables; do
        table_args="$table_args --table=$table"
    done

    local pg_result=0

    if [ "$OS" = "linux" ]; then
        # Write to temp file first - postgres user can't write to backup dir owned by mira_service
        local temp_dump="/tmp/mira_pg_backup_$$.dump"

        if [ "$LOUD_MODE" = true ]; then
            echo ""  # newline for verbose output
            sudo -u postgres pg_dump -d mira_service \
                --format=custom \
                --no-owner \
                --no-privileges \
                --data-only \
                $table_args \
                --file="$temp_dump" --verbose 2>&1
            pg_result=$?
        else
            sudo -u postgres pg_dump -d mira_service \
                --format=custom \
                --no-owner \
                --no-privileges \
                --data-only \
                $table_args \
                --file="$temp_dump" 2>"$error_file"
            pg_result=$?
        fi

        # Move temp file to backup directory if successful
        if [ $pg_result -eq 0 ] && [ -f "$temp_dump" ]; then
            sudo mv "$temp_dump" "$backup_file"
            sudo chown $(whoami) "$backup_file"
        else
            rm -f "$temp_dump" 2>/dev/null
        fi
    else
        # macOS: Use PGPASSWORD if set, otherwise rely on .pgpass/trust auth
        local pg_env=""
        if [ -n "$MIGRATE_DB_PASSWORD" ]; then
            pg_env="PGPASSWORD=$MIGRATE_DB_PASSWORD"
        fi

        if [ "$LOUD_MODE" = true ]; then
            echo ""  # newline for verbose output
            env $pg_env pg_dump -U mira_admin -h localhost -d mira_service \
                --format=custom \
                --no-owner \
                --no-privileges \
                --data-only \
                $table_args \
                --file="$backup_file" --verbose 2>&1
            pg_result=$?
        else
            env $pg_env pg_dump -U mira_admin -h localhost -d mira_service \
                --format=custom \
                --no-owner \
                --no-privileges \
                --data-only \
                $table_args \
                --file="$backup_file" 2>"$error_file"
            pg_result=$?
        fi
    fi

    if [ $pg_result -eq 0 ]; then
        [ "$LOUD_MODE" != true ] && echo -e "${CHECKMARK}"
    else
        [ "$LOUD_MODE" != true ] && echo -e "${ERROR}"
        print_error "Failed to backup PostgreSQL data"
        # Always show error details on failure, even in quiet mode
        if [ -f "$error_file" ] && [ -s "$error_file" ]; then
            print_info "Error details:"
            cat "$error_file" | while read line; do
                print_info "  $line"
            done
        fi
        return 1
    fi

    # Record backup size
    local backup_size=$(du -sh "$backup_file" | cut -f1)
    print_info "Database backup: $backup_size"

    return 0
}

# Backup Vault secrets to JSON files
backup_vault_secrets() {
    echo -ne "${DIM}${ARROW}${RESET} Backing up Vault secrets... "

    # Ensure authenticated
    vault_authenticate > /dev/null 2>&1 || {
        echo -e "${ERROR}"
        print_error "Cannot authenticate with Vault for backup"
        return 1
    }

    local success=true
    local vault_output

    # Export each secret path to JSON
    # The .data.data path extracts the actual secret data from KV v2 response
    # Only create files when secrets exist (prevents empty/invalid JSON files)

    vault_output=$(vault kv get -format=json secret/mira/api_keys 2>/dev/null) || true
    if [ -n "$vault_output" ]; then
        echo "$vault_output" | jq '.data.data' > "${BACKUP_DIR}/vault_api_keys.json"
    else
        print_warning "No api_keys secret found (may be okay for offline mode)"
    fi

    vault_output=$(vault kv get -format=json secret/mira/database 2>/dev/null) || true
    if [ -n "$vault_output" ]; then
        echo "$vault_output" | jq '.data.data' > "${BACKUP_DIR}/vault_database.json"
    else
        echo -e "${ERROR}"
        print_error "Cannot backup database credentials"
        success=false
    fi

    vault_output=$(vault kv get -format=json secret/mira/services 2>/dev/null) || true
    if [ -n "$vault_output" ]; then
        echo "$vault_output" | jq '.data.data' > "${BACKUP_DIR}/vault_services.json"
    else
        print_warning "No services secret found"
    fi

    vault_output=$(vault kv get -format=json secret/mira/auth 2>/dev/null) || true
    if [ -n "$vault_output" ]; then
        echo "$vault_output" | jq '.data.data' > "${BACKUP_DIR}/vault_auth.json"
    else
        print_warning "No auth secret found (credential encryption key may be missing)"
    fi

    # Extract database password for later PostgreSQL operations
    if [ -f "${BACKUP_DIR}/vault_database.json" ]; then
        MIGRATE_DB_PASSWORD=$(jq -r '.password // empty' "${BACKUP_DIR}/vault_database.json" 2>/dev/null || echo "")
    fi

    if [ "$success" = true ]; then
        echo -e "${CHECKMARK}"
    else
        return 1
    fi

    return 0
}

# Backup user data files (SQLite DBs and tool data)
backup_user_data_files() {
    local source_dir="/opt/mira/app/data/users"
    local backup_dir="${BACKUP_DIR}/user_data"

    echo -ne "${DIM}${ARROW}${RESET} Backing up user data files... "

    if [ ! -d "$source_dir" ]; then
        echo -e "${CHECKMARK} ${DIM}(no user data directory)${RESET}"
        return 0
    fi

    mkdir -p "$backup_dir"

    if cp -a "$source_dir"/* "$backup_dir/" 2>/dev/null; then
        local user_count=$(ls -d "$backup_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
        echo -e "${CHECKMARK} ${DIM}($user_count user directories)${RESET}"
    else
        echo -e "${CHECKMARK} ${DIM}(empty)${RESET}"
    fi

    return 0
}

# Backup Vault init keys for emergency recovery
backup_vault_init_keys() {
    echo -ne "${DIM}${ARROW}${RESET} Backing up Vault init keys... "

    if [ -f "/opt/vault/init-keys.txt" ]; then
        cp /opt/vault/init-keys.txt "${BACKUP_DIR}/init-keys.txt"
        chmod 600 "${BACKUP_DIR}/init-keys.txt"
        echo -e "${CHECKMARK}"
    else
        echo -e "${WARNING}"
        print_warning "No init-keys.txt found (Vault may not be initialized)"
    fi

    # Also backup role-id and secret-id if they exist
    [ -f "/opt/vault/role-id.txt" ] && cp /opt/vault/role-id.txt "${BACKUP_DIR}/"
    [ -f "/opt/vault/secret-id.txt" ] && cp /opt/vault/secret-id.txt "${BACKUP_DIR}/"

    return 0
}

# Create backup manifest with metadata
create_backup_manifest() {
    echo -ne "${DIM}${ARROW}${RESET} Creating backup manifest... "

    local mira_version
    mira_version=$(cat /opt/mira/app/VERSION 2>/dev/null || echo "unknown")

    local pg_version
    pg_version=$(psql --version 2>/dev/null | head -1 || echo "unknown")

    local vault_version
    vault_version=$(vault version 2>/dev/null | head -1 || echo "unknown")

    cat > "${BACKUP_DIR}/manifest.json" <<EOF
{
    "backup_timestamp": "${BACKUP_TIMESTAMP}",
    "mira_version": "${mira_version}",
    "postgresql_version": "${pg_version}",
    "vault_version": "${vault_version}",
    "os": "${OS}",
    "contents": {
        "postgresql_backup": "postgresql_backup.dump",
        "vault_secrets": ["vault_api_keys.json", "vault_database.json", "vault_services.json", "vault_auth.json"],
        "vault_init": "init-keys.txt",
        "user_data": "user_data/"
    }
}
EOF

    echo -e "${CHECKMARK}"
    return 0
}

# ============================================================================
# Restore Functions
# ============================================================================

# Restore Vault secrets from JSON backups
restore_vault_secrets() {
    echo -ne "${DIM}${ARROW}${RESET} Restoring Vault secrets... "

    # Ensure authenticated with fresh Vault
    vault_authenticate > /dev/null 2>&1 || {
        echo -e "${ERROR}"
        print_error "Cannot authenticate with new Vault"
        return 1
    }

    local success=true

    # Restore each secret from JSON backup
    # vault kv put accepts @file syntax for JSON input
    if [ -f "${BACKUP_DIR}/vault_api_keys.json" ] && [ -s "${BACKUP_DIR}/vault_api_keys.json" ]; then
        if ! vault kv put secret/mira/api_keys @"${BACKUP_DIR}/vault_api_keys.json" > /dev/null 2>&1; then
            # Try alternative format - some vault versions need different syntax
            local json_content=$(cat "${BACKUP_DIR}/vault_api_keys.json")
            if [ "$json_content" != "null" ]; then
                echo "$json_content" | vault kv put secret/mira/api_keys - > /dev/null 2>&1 || {
                    print_warning "Could not restore api_keys - may need manual configuration"
                }
            fi
        fi
    fi

    if [ -f "${BACKUP_DIR}/vault_database.json" ] && [ -s "${BACKUP_DIR}/vault_database.json" ]; then
        if ! vault kv put secret/mira/database @"${BACKUP_DIR}/vault_database.json" > /dev/null 2>&1; then
            local json_content=$(cat "${BACKUP_DIR}/vault_database.json")
            if [ "$json_content" != "null" ]; then
                echo "$json_content" | vault kv put secret/mira/database - > /dev/null 2>&1 || {
                    print_error "Could not restore database credentials"
                    success=false
                }
            fi
        fi
    fi

    if [ -f "${BACKUP_DIR}/vault_services.json" ] && [ -s "${BACKUP_DIR}/vault_services.json" ]; then
        if ! vault kv put secret/mira/services @"${BACKUP_DIR}/vault_services.json" > /dev/null 2>&1; then
            local json_content=$(cat "${BACKUP_DIR}/vault_services.json")
            if [ "$json_content" != "null" ]; then
                echo "$json_content" | vault kv put secret/mira/services - > /dev/null 2>&1 || {
                    print_warning "Could not restore services config"
                }
            fi
        fi
    fi

    # auth secrets are critical for user data decryption
    if [ -f "${BACKUP_DIR}/vault_auth.json" ] && [ -s "${BACKUP_DIR}/vault_auth.json" ]; then
        if ! vault kv put secret/mira/auth @"${BACKUP_DIR}/vault_auth.json" > /dev/null 2>&1; then
            local json_content=$(cat "${BACKUP_DIR}/vault_auth.json")
            if [ "$json_content" != "null" ]; then
                echo "$json_content" | vault kv put secret/mira/auth - > /dev/null 2>&1 || {
                    print_warning "Could not restore auth secrets - user data may be inaccessible"
                }
            fi
        fi
    fi

    if [ "$success" = true ]; then
        echo -e "${CHECKMARK}"
    else
        return 1
    fi

    # Apply key migrations for schema changes
    migrate_vault_api_keys

    return 0
}

# Migrate Vault API keys to handle schema renames between versions
# This ensures old key names are mapped to new expected names
migrate_vault_api_keys() {
    echo -ne "${DIM}${ARROW}${RESET} Checking for API key migrations... "

    local migrations_applied=0

    # Get current api_keys as JSON
    local current_keys
    current_keys=$(vault kv get -format=json secret/mira/api_keys 2>/dev/null | jq -c '.data.data // {}' 2>/dev/null || echo "{}")

    if [ "$current_keys" = "{}" ] || [ -z "$current_keys" ]; then
        echo -e "${DIM}(no keys to migrate)${RESET}"
        return 0
    fi

    # Define key migrations: old_name -> new_name
    # Canonical names are purpose-named: subcortical_key (subcortical analysis layer)
    # and provider_key (user-facing chat provider). Older installs may have vendor-named
    # (groq_key) or protocol-named (openaicompat_key) keys; rename them to canonical.
    local -a KEY_MIGRATIONS=(
        "groq_key:subcortical_key"
        "openaicompat_key:provider_key"
    )

    for mapping in "${KEY_MIGRATIONS[@]}"; do
        local old_key="${mapping%%:*}"
        local new_key="${mapping##*:}"

        # Check if new key is missing but old key exists
        local has_old has_new
        has_old=$(echo "$current_keys" | jq -r "has(\"$old_key\")" 2>/dev/null)
        has_new=$(echo "$current_keys" | jq -r "has(\"$new_key\")" 2>/dev/null)

        if [ "$has_old" = "true" ] && [ "$has_new" = "false" ]; then
            # Copy old key value to new key name
            local old_value
            old_value=$(echo "$current_keys" | jq -r ".\"$old_key\"" 2>/dev/null)

            if [ -n "$old_value" ] && [ "$old_value" != "null" ]; then
                vault kv patch secret/mira/api_keys "${new_key}=${old_value}" > /dev/null 2>&1 && {
                    migrations_applied=$((migrations_applied + 1))
                    [ "$LOUD_MODE" = true ] && echo -e "\n${DIM}  Migrated: ${old_key} -> ${new_key}${RESET}"
                }
            fi
        fi
    done

    if [ "$migrations_applied" -gt 0 ]; then
        echo -e "${CHECKMARK} ${DIM}($migrations_applied key(s) migrated)${RESET}"
    else
        echo -e "${DIM}(none needed)${RESET}"
    fi

    return 0
}

# Restore PostgreSQL data from backup
restore_postgresql_data() {
    local backup_file="${BACKUP_DIR}/postgresql_backup.dump"
    local error_file="${BACKUP_DIR}/pg_restore_errors.log"
    local script_path="/opt/mira/app/deploy/schema_aware_restore.py"
    local python_path="/opt/mira/app/venv/bin/python3"
    local temp_db="mira_restore_temp"

    echo -ne "${DIM}${ARROW}${RESET} Restoring PostgreSQL data... "

    # Find backup file: prefer full dump from old_app (has schema+data), fall back to data-only dump
    local actual_backup=""

    # Look for any mira_backup_*.dump in old_app directory (handles any timestamp)
    if [ -d "${BACKUP_DIR}/old_app" ]; then
        actual_backup=$(find "${BACKUP_DIR}/old_app" -maxdepth 1 -name "mira_backup_*.dump" -type f ! -empty 2>/dev/null | head -1)
    fi

    # Fall back to postgresql_backup.dump if no old_app dump found
    if [ -z "$actual_backup" ] && [ -f "$backup_file" ]; then
        actual_backup="$backup_file"
    fi

    if [ -z "$actual_backup" ] || [ ! -f "$actual_backup" ]; then
        echo -e "${ERROR}"
        print_error "No backup file found"
        return 1
    fi

    [ "$LOUD_MODE" = true ] && echo -e "\n${DIM}Using backup: $actual_backup${RESET}"

    if [ ! -f "$script_path" ]; then
        echo -e "${ERROR}"
        print_error "Schema-aware restore script not found: $script_path"
        return 1
    fi

    # Step 1: Create temp database (requires superuser)
    [ "$LOUD_MODE" = true ] && echo -e "\n${DIM}Creating temporary database...${RESET}"
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS $temp_db;" > /dev/null 2>&1 || true
    if ! sudo -u postgres psql -c "CREATE DATABASE $temp_db;" > /dev/null 2>&1; then
        echo -e "${ERROR}"
        print_error "Failed to create temporary database"
        return 1
    fi

    # Step 2: Enable vector extension (requires superuser)
    [ "$LOUD_MODE" = true ] && echo -e "${DIM}Enabling vector extension...${RESET}"
    sudo -u postgres psql -d "$temp_db" -c "CREATE EXTENSION IF NOT EXISTS vector;" > /dev/null 2>&1 || true
    sudo -u postgres psql -d "$temp_db" -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" > /dev/null 2>&1 || true

    # Step 3: Restore dump to temp database (requires superuser for some operations)
    [ "$LOUD_MODE" = true ] && echo -e "${DIM}Restoring backup to temporary database...${RESET}"
    local temp_dump="/tmp/mira_temp_restore.dump"
    sudo cp "$actual_backup" "$temp_dump"
    sudo chown postgres:postgres "$temp_dump"
    sudo -u postgres pg_restore -d "$temp_db" --no-owner --no-privileges "$temp_dump" 2>/dev/null || true
    sudo rm -f "$temp_dump"

    # Step 4: Run Python schema-aware migration with --skip-temp-db-setup
    local restore_result=0
    local restore_args=(
        "--backup-file" "$actual_backup"
        "--os-type" "$OS"
        "--skip-temp-db-setup"
    )

    # Add loud flag if in verbose mode
    if [ "$LOUD_MODE" = true ]; then
        restore_args+=("--loud")
    fi

    # Pass password via environment variable (more secure than CLI arg - not visible in ps)
    if [ "$LOUD_MODE" = true ]; then
        PGPASSWORD="$MIGRATE_DB_PASSWORD" "$python_path" "$script_path" "${restore_args[@]}" 2>&1
        restore_result=$?
    else
        PGPASSWORD="$MIGRATE_DB_PASSWORD" "$python_path" "$script_path" "${restore_args[@]}" 2>"$error_file"
        restore_result=$?
    fi

    # Step 5: Clean up temp database
    [ "$LOUD_MODE" = true ] && echo -e "${DIM}Cleaning up temporary database...${RESET}"
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS $temp_db;" > /dev/null 2>&1 || true

    case $restore_result in
        0)
            [ "$LOUD_MODE" != true ] && echo -e "${CHECKMARK}"
            ;;
        2)
            # Conflict detected - script already printed details
            [ "$LOUD_MODE" != true ] && echo -e "${ERROR}"
            print_error "Config table conflict detected during restore"
            print_info "Re-run migration with --prefer-backup or --prefer-schema to resolve"
            print_info "Or manually resolve in database after migration completes"
            return 1
            ;;
        *)
            [ "$LOUD_MODE" != true ] && echo -e "${ERROR}"
            print_error "Failed to restore PostgreSQL data"
            if [ -f "$error_file" ] && [ -s "$error_file" ]; then
                print_info "Error details:"
                cat "$error_file" | while read line; do
                    print_info "  $line"
                done
            fi
            return 1
            ;;
    esac

    return 0
}

# Migrate user IDs after restore to handle UUID changes between installations
# Fresh installs create new UUIDs for users even if email is the same
migrate_user_ids() {
    echo -ne "${DIM}${ARROW}${RESET} Checking for user ID migrations... "

    local snapshot="${BACKUP_DIR}/db_snapshot.json"
    [ ! -f "$snapshot" ] && { echo -e "${DIM}(no snapshot)${RESET}"; return 0; }

    local user_data_dir="/opt/mira/app/data/users"

    # For each old user, remap if UUID changed
    jq -r '.structural_data.users[]? | "\(.id) \(.email)"' "$snapshot" 2>/dev/null | while read -r old_id old_email; do
        [ -z "$old_id" ] || [ -z "$old_email" ] && continue

        local new_id=$(run_psql "SELECT id FROM users WHERE email = '$old_email'" | tr -d ' ')
        [ -z "$new_id" ] || [ "$old_id" = "$new_id" ] && continue

        local orphans=$(run_psql "SELECT COUNT(*) FROM messages WHERE user_id = '$old_id'" | tr -d ' ')
        [ "$orphans" = "0" ] && continue

        [ "$LOUD_MODE" = true ] && echo -e "\n${DIM}  $old_email: $old_id -> $new_id${RESET}"

        # Delete new continuum (unique constraint), then remap all old records
        run_psql "DELETE FROM continuums WHERE user_id = '$new_id'" > /dev/null
        for tbl in continuums messages memories entities api_tokens domain_knowledge_blocks user_activity_days; do
            run_psql "UPDATE $tbl SET user_id = '$new_id' WHERE user_id = '$old_id'" > /dev/null
        done

        # Migrate user data files (SQLite) from old UUID dir to new UUID dir
        local old_dir="${user_data_dir}/${old_id}"
        local new_dir="${user_data_dir}/${new_id}"
        if [ -d "$old_dir" ] && [ -d "$new_dir" ]; then
            cp -a "$old_dir"/* "$new_dir"/ 2>/dev/null && rm -rf "$old_dir"
            [ "$LOUD_MODE" = true ] && echo -e "${DIM}    Migrated user data files${RESET}"
        fi
    done

    local remaining=$(run_psql "SELECT COUNT(*) FROM messages WHERE user_id NOT IN (SELECT id FROM users)" | tr -d ' ')
    [ "$remaining" = "0" ] && echo -e "${CHECKMARK}" || echo -e "${WARNING} ${DIM}($remaining orphaned)${RESET}"
    return 0
}

# Restore user data files
restore_user_data_files() {
    local backup_dir="${BACKUP_DIR}/user_data"
    local target_dir="/opt/mira/app/data/users"

    echo -ne "${DIM}${ARROW}${RESET} Restoring user data files... "

    if [ ! -d "$backup_dir" ]; then
        echo -e "${CHECKMARK} ${DIM}(no user data to restore)${RESET}"
        return 0
    fi

    # Check if backup has content
    if [ -z "$(ls -A "$backup_dir" 2>/dev/null)" ]; then
        echo -e "${CHECKMARK} ${DIM}(empty backup)${RESET}"
        return 0
    fi

    mkdir -p "$target_dir"

    if cp -a "$backup_dir"/* "$target_dir/" 2>/dev/null; then
        # Set ownership
        chown -R ${MIRA_USER}:${MIRA_GROUP} "$target_dir" 2>/dev/null || true

        local user_count=$(ls -d "$target_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
        echo -e "${CHECKMARK} ${DIM}($user_count user directories)${RESET}"
    else
        echo -e "${ERROR}"
        print_error "Failed to restore user data files"
        return 1
    fi

    return 0
}

# ============================================================================
# Verification Functions
# ============================================================================

# Verify Vault secrets are accessible
verify_vault_secrets() {
    echo -ne "${DIM}${ARROW}${RESET} Verifying Vault secrets... "

    if vault kv get secret/mira/api_keys > /dev/null 2>&1 || \
       vault kv get secret/mira/database > /dev/null 2>&1; then
        echo -e "${CHECKMARK}"
        return 0
    else
        echo -e "${ERROR}"
        print_error "Cannot access Vault secrets after migration"
        return 1
    fi
}

# Verify memory embeddings preserved
verify_memory_embeddings() {
    echo -ne "${DIM}${ARROW}${RESET} Verifying vector embeddings... "

    local memories_with_embeddings
    if [ "$OS" = "linux" ]; then
        memories_with_embeddings=$(sudo -u postgres psql -d mira_service -tAc \
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL" 2>/dev/null || echo "0")
    else
        memories_with_embeddings=$(psql -U mira_admin -h localhost -d mira_service -tAc \
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL" 2>/dev/null || echo "0")
    fi

    echo -e "${CHECKMARK} ${DIM}($memories_with_embeddings memories with embeddings)${RESET}"
    return 0
}

# Verify user data files exist
verify_user_data_files() {
    echo -ne "${DIM}${ARROW}${RESET} Verifying user data files... "

    local target_dir="/opt/mira/app/data/users"

    if [ ! -d "$target_dir" ]; then
        echo -e "${CHECKMARK} ${DIM}(no user data directory)${RESET}"
        return 0
    fi

    local user_count=$(ls -d "$target_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${CHECKMARK} ${DIM}($user_count user directories)${RESET}"
    return 0
}

# ============================================================================
# Metric Capture Functions
# ============================================================================

# Capture pre-migration metrics for verification
capture_pre_migration_metrics() {
    if [ "$OS" = "linux" ]; then
        PRE_USER_COUNT=$(sudo -u postgres psql -d mira_service -tAc "SELECT COUNT(*) FROM users" 2>/dev/null || echo "0")
        PRE_MESSAGE_COUNT=$(sudo -u postgres psql -d mira_service -tAc "SELECT COUNT(*) FROM messages" 2>/dev/null || echo "0")
        PRE_MEMORY_COUNT=$(sudo -u postgres psql -d mira_service -tAc "SELECT COUNT(*) FROM memories" 2>/dev/null || echo "0")
    else
        PRE_USER_COUNT=$(psql -U mira_admin -h localhost -d mira_service -tAc "SELECT COUNT(*) FROM users" 2>/dev/null || echo "0")
        PRE_MESSAGE_COUNT=$(psql -U mira_admin -h localhost -d mira_service -tAc "SELECT COUNT(*) FROM messages" 2>/dev/null || echo "0")
        PRE_MEMORY_COUNT=$(psql -U mira_admin -h localhost -d mira_service -tAc "SELECT COUNT(*) FROM memories" 2>/dev/null || echo "0")
    fi

    # Trim whitespace
    PRE_USER_COUNT=$(echo "$PRE_USER_COUNT" | tr -d ' ')
    PRE_MESSAGE_COUNT=$(echo "$PRE_MESSAGE_COUNT" | tr -d ' ')
    PRE_MEMORY_COUNT=$(echo "$PRE_MEMORY_COUNT" | tr -d ' ')

    print_info "Found: $PRE_USER_COUNT users, $PRE_MESSAGE_COUNT messages, $PRE_MEMORY_COUNT memories"
}

# Capture and verify post-migration metrics
verify_post_migration_metrics() {
    local post_user_count post_message_count post_memory_count

    if [ "$OS" = "linux" ]; then
        post_user_count=$(sudo -u postgres psql -d mira_service -tAc "SELECT COUNT(*) FROM users" 2>/dev/null || echo "0")
        post_message_count=$(sudo -u postgres psql -d mira_service -tAc "SELECT COUNT(*) FROM messages" 2>/dev/null || echo "0")
        post_memory_count=$(sudo -u postgres psql -d mira_service -tAc "SELECT COUNT(*) FROM memories" 2>/dev/null || echo "0")
    else
        post_user_count=$(psql -U mira_admin -h localhost -d mira_service -tAc "SELECT COUNT(*) FROM users" 2>/dev/null || echo "0")
        post_message_count=$(psql -U mira_admin -h localhost -d mira_service -tAc "SELECT COUNT(*) FROM messages" 2>/dev/null || echo "0")
        post_memory_count=$(psql -U mira_admin -h localhost -d mira_service -tAc "SELECT COUNT(*) FROM memories" 2>/dev/null || echo "0")
    fi

    # Trim whitespace
    post_user_count=$(echo "$post_user_count" | tr -d ' ')
    post_message_count=$(echo "$post_message_count" | tr -d ' ')
    post_memory_count=$(echo "$post_memory_count" | tr -d ' ')

    local verification_passed=true

    echo -ne "${DIM}${ARROW}${RESET} Verifying user count... "
    if [ "$PRE_USER_COUNT" != "$post_user_count" ]; then
        echo -e "${ERROR}"
        print_error "User count mismatch: $PRE_USER_COUNT -> $post_user_count"
        verification_passed=false
    else
        echo -e "${CHECKMARK} ${DIM}($post_user_count users)${RESET}"
    fi

    echo -ne "${DIM}${ARROW}${RESET} Verifying message count... "
    if [ "$PRE_MESSAGE_COUNT" != "$post_message_count" ]; then
        echo -e "${ERROR}"
        print_error "Message count mismatch: $PRE_MESSAGE_COUNT -> $post_message_count"
        verification_passed=false
    else
        echo -e "${CHECKMARK} ${DIM}($post_message_count messages)${RESET}"
    fi

    echo -ne "${DIM}${ARROW}${RESET} Verifying memory count... "
    if [ "$PRE_MEMORY_COUNT" != "$post_memory_count" ]; then
        echo -e "${ERROR}"
        print_error "Memory count mismatch: $PRE_MEMORY_COUNT -> $post_memory_count"
        verification_passed=false
    else
        echo -e "${CHECKMARK} ${DIM}($post_memory_count memories)${RESET}"
    fi

    if [ "$verification_passed" = true ]; then
        return 0
    else
        return 1
    fi
}
