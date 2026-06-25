# deploy/migrate.sh
# MIRA Migration Orchestrator - Upgrade across versions preserving user data
# Source this file - do not execute directly
#
# Requires: lib/output.sh, lib/services.sh, lib/vault.sh, lib/migrate.sh sourced first
# Requires: LOUD_MODE variable set
#
# Usage: ./deploy/deploy.sh --migrate [--loud] [--dry-run]

set -e

# DRY_RUN_MODE is exported from deploy.sh
: "${DRY_RUN_MODE:=false}"

# ============================================================================
# OS Detection (same logic as config.sh)
# ============================================================================

OS_TYPE=$(uname -s)
case "$OS_TYPE" in
    Linux*)
        OS="linux"
        if [ -f /etc/redhat-release ] || [ -f /etc/fedora-release ]; then
            DISTRO="fedora"
        elif [ -f /etc/debian_version ]; then
            DISTRO="debian"
        else
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                case "$ID" in
                    fedora|rhel|centos|rocky|alma)
                        DISTRO="fedora"
                        ;;
                    debian|ubuntu|linuxmint|pop)
                        DISTRO="debian"
                        ;;
                    *)
                        case "$ID_LIKE" in
                            *fedora*|*rhel*)
                                DISTRO="fedora"
                                ;;
                            *debian*|*ubuntu*)
                                DISTRO="debian"
                                ;;
                            *)
                                DISTRO="unknown"
                                ;;
                        esac
                        ;;
                esac
            else
                DISTRO="unknown"
            fi
        fi
        ;;
    Darwin*)
        OS="macos"
        DISTRO=""
        ;;
    *)
        print_error "Unsupported operating system: $OS_TYPE"
        exit 1
        ;;
esac

# ============================================================================
# Migration Header
# ============================================================================

clear
echo -e "${BOLD}${CYAN}"
if is_dry_run; then
    echo "╔════════════════════════════════════════╗"
    echo "║   MIRA Migration (DRY RUN MODE)        ║"
    echo "╚════════════════════════════════════════╝"
else
    echo "╔════════════════════════════════════════╗"
    echo "║   MIRA Migration (--migrate)           ║"
    echo "╚════════════════════════════════════════╝"
fi
echo -e "${RESET}"

if is_dry_run; then
    echo -e "${BOLD}${CYAN}This is a dry run - no changes will be made${RESET}"
    echo ""
fi

[ "$LOUD_MODE" = true ] && print_info "Running in verbose mode (--loud)"

if [ -n "$DISTRO" ]; then
    print_info "Detected: $OS ($DISTRO)"
else
    print_info "Detected: $OS"
fi
echo ""

# ============================================================================
# Phase 1: Pre-flight Validation
# ============================================================================

print_header "Phase 1: Pre-flight Validation"

migrate_check_existing_install || exit 1
migrate_check_postgresql_running || exit 1
migrate_check_vault_accessible || exit 1
migrate_check_disk_space || exit 1
migrate_check_no_active_sessions || exit 1

# Capture pre-migration metrics
capture_pre_migration_metrics

print_success "Pre-flight validation passed"
echo ""

# ============================================================================
# Phase 2: Backup
# ============================================================================

print_header "Phase 2: Creating Backup"

BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if is_dry_run; then
    # In dry-run mode, use temp directory
    BACKUP_DIR="/tmp/mira_migration_dryrun_${BACKUP_TIMESTAMP}"
    mkdir -p "$BACKUP_DIR"
    print_info "Dry-run: Using temporary directory $BACKUP_DIR"
else
    BACKUP_DIR="/opt/mira/backups/${BACKUP_TIMESTAMP}"
    echo -ne "${DIM}${ARROW}${RESET} Creating backup directory... "
    sudo mkdir -p "$BACKUP_DIR"
    sudo chown $(whoami) "$BACKUP_DIR"
    echo -e "${CHECKMARK}"

    # Set up logging (only for real runs)
    setup_migration_logging "$BACKUP_DIR"
fi

# CRITICAL: Capture complete snapshots FIRST for integrity verification
VAULT_SNAPSHOT_FILE="${BACKUP_DIR}/vault_snapshot.json"
capture_vault_snapshot "$VAULT_SNAPSHOT_FILE" || exit 1

DB_SNAPSHOT_FILE="${BACKUP_DIR}/db_snapshot.json"
capture_database_snapshot "$DB_SNAPSHOT_FILE" || exit 1

if is_dry_run; then
    dry_run_notice "Export Vault secrets to JSON"
    dry_run_notice "Create PostgreSQL backup (pg_dump)"
    dry_run_notice "Copy user data files"
    dry_run_notice "Backup Vault init keys"
else
    # IMPORTANT: backup_vault_secrets MUST run first - it sets MIGRATE_DB_PASSWORD
    # which backup_postgresql_data needs for macOS authentication
    backup_vault_secrets || exit 1
    backup_postgresql_data || exit 1
    backup_user_data_files || exit 1
    backup_vault_init_keys || exit 1
    create_backup_manifest || exit 1
fi

print_success "Backup complete: $BACKUP_DIR"
echo ""

# ============================================================================
# Phase 2.5: Backup Verification (before any destructive operations)
# ============================================================================

if ! is_dry_run; then
    verify_all_backups || exit 1
    echo ""
fi

# ============================================================================
# Phase 3: Extract Configuration from Backup
# ============================================================================

print_header "Phase 3: Preparing Configuration"

# Extract CONFIG_* values from backed-up vault secrets for deploy modules
echo -ne "${DIM}${ARROW}${RESET} Extracting configuration from backup... "

# Read database password from backup
if [ -f "${BACKUP_DIR}/vault_database.json" ]; then
    CONFIG_DB_PASSWORD=$(jq -r '.password // empty' "${BACKUP_DIR}/vault_database.json" 2>/dev/null || echo "")
fi
CONFIG_DB_PASSWORD="${CONFIG_DB_PASSWORD:-changethisifdeployingpwd}"

# Read API keys from backup
if [ -f "${BACKUP_DIR}/vault_api_keys.json" ]; then
    CONFIG_ANTHROPIC_KEY=$(jq -r '.anthropic_key // empty' "${BACKUP_DIR}/vault_api_keys.json" 2>/dev/null || echo "")
    CONFIG_ANTHROPIC_BATCH_KEY=$(jq -r '.anthropic_batch_key // empty' "${BACKUP_DIR}/vault_api_keys.json" 2>/dev/null || echo "")
    CONFIG_PROVIDER_KEY=$(jq -r '.provider_key // .openaicompat_key // empty' "${BACKUP_DIR}/vault_api_keys.json" 2>/dev/null || echo "")
    CONFIG_KAGI_KEY=$(jq -r '.kagi_api_key // empty' "${BACKUP_DIR}/vault_api_keys.json" 2>/dev/null || echo "")
fi

# Set defaults for missing values (allows migration of offline-mode installs)
CONFIG_ANTHROPIC_KEY="${CONFIG_ANTHROPIC_KEY:-OFFLINE_MODE_PLACEHOLDER}"
CONFIG_ANTHROPIC_BATCH_KEY="${CONFIG_ANTHROPIC_BATCH_KEY:-$CONFIG_ANTHROPIC_KEY}"
CONFIG_PROVIDER_KEY="${CONFIG_PROVIDER_KEY:-}"
CONFIG_KAGI_KEY="${CONFIG_KAGI_KEY:-}"

# Set other CONFIG_* for deploy modules
CONFIG_INSTALL_PLAYWRIGHT="no"     # Skip playwright during migration
CONFIG_INSTALL_SYSTEMD="yes"       # Recreate systemd services
CONFIG_START_MIRA_NOW="no"         # Don't auto-start until verified

# Detect offline mode from backed-up API keys
# If anthropic_key is placeholder, original install was offline/local mode
if [ "$CONFIG_ANTHROPIC_KEY" = "OFFLINE_MODE_PLACEHOLDER" ]; then
    CONFIG_OFFLINE_MODE="yes"
    CONFIG_LOCAL_MODEL_CHOICE="auto"  # Default: pre-configured model pair
else
    CONFIG_OFFLINE_MODE="no"
    CONFIG_LOCAL_MODEL_CHOICE=""
fi

echo -e "${CHECKMARK}"
print_info "Database password: preserved from backup"
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    print_info "API keys: offline mode detected (will configure llama-server endpoints)"
else
    print_info "API keys: preserved from backup"
fi
echo ""

# ============================================================================
# Phase 4: Service Shutdown
# ============================================================================

print_header "Phase 4: Stopping Services"

if is_dry_run; then
    dry_run_skip "Stop MIRA service"
    dry_run_skip "Stop Vault service"
    dry_run_skip "Stop Valkey service"
    print_info "PostgreSQL would remain running (needed for schema operations)"
else
    # Check if MIRA was running before we stop it (to restore state at end)
    MIRA_WAS_RUNNING="no"
    if [ "$OS" = "linux" ]; then
        if sudo systemctl is-active --quiet mira.service 2>/dev/null; then
            MIRA_WAS_RUNNING="yes"
        fi
    else
        if lsof -ti :1993 >/dev/null 2>&1; then
            MIRA_WAS_RUNNING="yes"
        fi
    fi

    # Stop MIRA first
    echo -ne "${DIM}${ARROW}${RESET} Stopping MIRA... "
    if [ "$OS" = "linux" ]; then
        sudo systemctl stop mira.service 2>/dev/null || true
    else
        # macOS - kill by port
        lsof -ti :1993 | xargs kill 2>/dev/null || true
    fi
    echo -e "${CHECKMARK}"

    # Stop Vault
    echo -ne "${DIM}${ARROW}${RESET} Stopping Vault... "
    if [ "$OS" = "linux" ]; then
        sudo systemctl stop vault-unseal.service 2>/dev/null || true
        sudo systemctl stop vault.service 2>/dev/null || true
    else
        if [ -f /opt/vault/vault.pid ]; then
            kill $(cat /opt/vault/vault.pid) 2>/dev/null || true
            rm -f /opt/vault/vault.pid
        fi
    fi
    echo -e "${CHECKMARK}"

    # Stop Valkey
    echo -ne "${DIM}${ARROW}${RESET} Stopping Valkey... "
    if [ "$OS" = "linux" ]; then
        sudo systemctl stop valkey 2>/dev/null || true
    else
        brew services stop valkey 2>/dev/null || true
    fi
    echo -e "${CHECKMARK}"

    # PostgreSQL stays running (needed for drop/create operations)
    print_info "PostgreSQL left running (needed for schema operations)"

    print_success "Services stopped"
fi
echo ""

# ============================================================================
# Phase 5: Clean Old Installation
# ============================================================================

print_header "Phase 5: Cleaning Old Installation"

if is_dry_run; then
    dry_run_skip "Move /opt/mira/app to backup"
    dry_run_skip "Remove /opt/vault/data"
    dry_run_skip "Remove Vault credential files"
    dry_run_skip "Drop mira_service database"
else
    # Preserve old app in backup
    echo -ne "${DIM}${ARROW}${RESET} Moving old installation to backup... "
    if [ -d "/opt/mira/app" ]; then
        sudo mv /opt/mira/app "${BACKUP_DIR}/old_app"
    fi
    echo -e "${CHECKMARK}"

    # Remove old Vault data (fresh init coming)
    echo -ne "${DIM}${ARROW}${RESET} Removing old Vault data... "
    sudo rm -rf /opt/vault/data
    sudo rm -f /opt/vault/init-keys.txt /opt/vault/role-id.txt /opt/vault/secret-id.txt
    echo -e "${CHECKMARK}"

    # Drop old database
    echo -ne "${DIM}${ARROW}${RESET} Dropping old database... "
    if [ "$OS" = "linux" ]; then
        # Terminate existing connections first
        sudo -u postgres psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='mira_service' AND pid <> pg_backend_pid();" > /dev/null 2>&1 || true
        sudo -u postgres psql -c "DROP DATABASE IF EXISTS mira_service;" > /dev/null 2>&1 || true
    else
        psql postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='mira_service' AND pid <> pg_backend_pid();" > /dev/null 2>&1 || true
        psql postgres -c "DROP DATABASE IF EXISTS mira_service;" > /dev/null 2>&1 || true
    fi
    echo -e "${CHECKMARK}"

    print_success "Old installation cleaned"
fi
echo ""

# ============================================================================
# Phase 6: Fresh Installation
# ============================================================================

print_header "Phase 6: Fresh Installation"

if is_dry_run; then
    dry_run_skip "Run dependencies.sh (system packages)"
    dry_run_skip "Run python.sh (clone MIRA, create venv)"
    dry_run_skip "Run vault.sh (initialize Vault)"
    dry_run_skip "Run postgresql.sh (deploy schema)"
else
    print_info "Running deployment modules..."
    echo ""

    # Set variables needed by deploy modules
    export OS
    export DISTRO
    export LOUD_MODE
    export CONFIG_DB_PASSWORD
    export CONFIG_ANTHROPIC_KEY
    export CONFIG_ANTHROPIC_BATCH_KEY
    export CONFIG_PROVIDER_KEY
    export CONFIG_KAGI_KEY
    export CONFIG_INSTALL_PLAYWRIGHT
    export CONFIG_INSTALL_SYSTEMD
    export CONFIG_OFFLINE_MODE
    export CONFIG_LOCAL_MODEL_CHOICE

    # Track installation progress for rollback
    PHASE6_FAILED=""

    # Temporarily disable exit-on-error to handle failures gracefully
    set +e

    # dependencies.sh - System packages (idempotent)
    source "${SCRIPT_DIR}/dependencies.sh"
    if [ $? -ne 0 ]; then
        PHASE6_FAILED="dependencies.sh"
    fi
    echo ""

    # python.sh - Fresh MIRA clone, venv
    if [ -z "$PHASE6_FAILED" ]; then
        source "${SCRIPT_DIR}/python.sh"
        if [ $? -ne 0 ]; then
            PHASE6_FAILED="python.sh"
        fi
        echo ""
    fi

    # vault.sh - Fresh Vault initialization
    if [ -z "$PHASE6_FAILED" ]; then
        source "${SCRIPT_DIR}/vault.sh"
        if [ $? -ne 0 ]; then
            PHASE6_FAILED="vault.sh"
        fi
        echo ""
    fi

    # postgresql.sh - Fresh schema deployment
    if [ -z "$PHASE6_FAILED" ]; then
        source "${SCRIPT_DIR}/postgresql.sh"
        if [ $? -ne 0 ]; then
            PHASE6_FAILED="postgresql.sh"
        fi
        echo ""
    fi

    # Re-enable exit-on-error
    set -e

    # Handle Phase 6 failure
    if [ -n "$PHASE6_FAILED" ]; then
        echo ""
        print_error "Fresh installation failed during: $PHASE6_FAILED"
        echo ""
        echo -e "${BOLD}${YELLOW}Would you like to attempt automatic rollback?${RESET}"
        echo -e "${DIM}This will restore your system to its pre-migration state.${RESET}"
        echo ""
        read -p "Attempt rollback? (yes/no): " confirm_rollback
        if [ "$confirm_rollback" = "yes" ]; then
            rollback_from_backup "$BACKUP_DIR"
            finalize_migration_log "FAILED - ROLLED BACK"
        else
            print_info "Rollback skipped. Manual recovery may be needed."
            print_info "Backup available at: $BACKUP_DIR"
            finalize_migration_log "FAILED"
        fi
        exit 1
    fi

    print_success "Fresh installation complete"
fi
echo ""

# ============================================================================
# Phase 7: Data Restoration
# ============================================================================

print_header "Phase 7: Restoring Data"

if is_dry_run; then
    dry_run_skip "Restore Vault secrets from backup"
    dry_run_skip "Restore PostgreSQL data (pg_restore)"
    dry_run_skip "Restore user data files"
else
    # Restore Vault secrets first (needed for user data decryption)
    restore_vault_secrets || exit 1

    # Restore PostgreSQL data
    restore_postgresql_data || exit 1

    # Restore user data files (before migrate_user_ids so dirs exist)
    restore_user_data_files || exit 1

    # Migrate user IDs (fresh installs create new UUIDs for same emails)
    # Must run after restore_user_data_files to rename UUID directories
    migrate_user_ids || exit 1

    print_success "Data restored"
fi
echo ""

# ============================================================================
# Phase 8: Verification
# ============================================================================

print_header "Phase 8: Verification"

if is_dry_run; then
    dry_run_skip "Verify Vault integrity against snapshot"
    dry_run_skip "Verify database integrity against snapshot"
    dry_run_skip "Verify memory embeddings preserved"
    dry_run_skip "Verify user data files restored"
else
    # CRITICAL: Verify Vault integrity by comparing against pre-migration snapshot
    verify_vault_snapshot "$VAULT_SNAPSHOT_FILE" || {
        print_error "VAULT INTEGRITY CHECK FAILED"
        print_info "Pre-migration snapshot: $VAULT_SNAPSHOT_FILE"
        print_info "Backup JSON files available at: $BACKUP_DIR/vault_*.json"
        print_info "You may need to manually restore missing secrets"
        finalize_migration_log "FAILED"
        exit 1
    }

    # CRITICAL: Verify database integrity by comparing against pre-migration snapshot
    verify_database_snapshot "$DB_SNAPSHOT_FILE" || {
        print_error "DATABASE INTEGRITY CHECK FAILED"
        print_info "Pre-migration snapshot: $DB_SNAPSHOT_FILE"
        print_info "Database backup: ${BACKUP_DIR}/postgresql_backup.dump"
        print_info "You may need to manually restore from backup"
        finalize_migration_log "FAILED"
        exit 1
    }

    verify_memory_embeddings || exit 1
    verify_user_data_files || exit 1

    print_success "All verifications passed"
fi
echo ""

# ============================================================================
# Phase 9: Finalization
# ============================================================================

print_header "Phase 9: Finalization"

if is_dry_run; then
    dry_run_skip "Create MIRA CLI wrapper"
    dry_run_skip "Configure shell alias"
    dry_run_skip "Set up systemd services (Linux)"
else
    # Set CONFIG for finalize.sh
    CONFIG_START_MIRA_NOW="no"  # Let user start manually after verification

    source "${SCRIPT_DIR}/finalize.sh"
fi
echo ""

# ============================================================================
# Migration Complete
# ============================================================================

echo ""
if is_dry_run; then
    echo -e "${BOLD}${CYAN}"
    echo "╔════════════════════════════════════════╗"
    echo "║   Dry Run Complete!                    ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${RESET}"
    echo ""
    print_success "Dry run completed successfully"
    echo ""
    print_info "No changes were made to your system"
    print_info "Snapshots captured to: $BACKUP_DIR"
    echo ""
    print_info "To perform the actual migration, run:"
    print_info "  ./deploy/deploy.sh --migrate"
    echo ""
    # Clean up temp directory
    rm -rf "$BACKUP_DIR"
else
    echo -e "${BOLD}${GREEN}"
    echo "╔════════════════════════════════════════╗"
    echo "║   Migration Complete!                  ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${RESET}"

    # Finalize log
    finalize_migration_log "SUCCESSFUL"

    print_success "MIRA has been upgraded successfully"
    echo ""
    print_info "Backup preserved at: $BACKUP_DIR"
    print_info "Old installation at: ${BACKUP_DIR}/old_app"
    print_info "Migration log: ${BACKUP_DIR}/migration.log"
    echo ""
    print_warning "After verifying everything works, delete the backup:"
    print_info "  sudo rm -rf $BACKUP_DIR"
    echo ""

    # Flush Valkey cache (stale user/continuum IDs cause issues after migration)
    echo -ne "${DIM}${ARROW}${RESET} Flushing Valkey cache... "
    if valkey-cli FLUSHALL > /dev/null 2>&1; then
        echo -e "${CHECKMARK}"
    else
        echo -e "${DIM}(skipped - valkey not running)${RESET}"
    fi

    # Restart MIRA if it was running before migration
    if [ "$MIRA_WAS_RUNNING" = "yes" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Restarting MIRA (was running before migration)... "
        if [ "$OS" = "linux" ]; then
            if sudo systemctl start mira.service 2>/dev/null; then
                echo -e "${CHECKMARK}"
                print_success "MIRA is running"
            else
                echo -e "${ERROR}"
                print_warning "Failed to start MIRA automatically"
                print_info "Start manually with: sudo systemctl start mira"
            fi
        else
            # macOS - start in background
            if nohup mira >/dev/null 2>&1 &; then
                echo -e "${CHECKMARK}"
                print_success "MIRA is running"
            else
                echo -e "${ERROR}"
                print_warning "Failed to start MIRA automatically"
                print_info "Start manually with: mira"
            fi
        fi
    else
        print_info "Start MIRA with: mira"
        if [ "$OS" = "linux" ]; then
            print_info "Or via systemd: sudo systemctl start mira"
        fi
    fi
    echo ""
fi
