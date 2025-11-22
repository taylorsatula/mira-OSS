#!/bin/bash
set -e

# MIRA Deployment Script
# This script automates the complete deployment of MIRA OSS

# ============================================================================
# VISUAL OUTPUT CONFIGURATION
# ============================================================================

# Parse arguments
LOUD_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--loud" ]; then
        LOUD_MODE=true
    fi
done

# ANSI color codes (muted/professional palette)
RESET='\033[0m'
DIM='\033[2m'
BOLD='\033[1m'
GRAY='\033[38;5;240m'
BLUE='\033[38;5;75m'
GREEN='\033[38;5;77m'
YELLOW='\033[38;5;186m'
RED='\033[38;5;203m'
CYAN='\033[38;5;80m'

# Visual elements
CHECKMARK="${GREEN}✓${RESET}"
ARROW="${CYAN}→${RESET}"
WARNING="${YELLOW}⚠${RESET}"
ERROR="${RED}✗${RESET}"

# Print colored output
print_header() {
    echo -e "\n${BOLD}${BLUE}$1${RESET}"
}

print_step() {
    echo -e "${DIM}${ARROW}${RESET} $1"
}

print_success() {
    echo -e "${CHECKMARK} ${GREEN}$1${RESET}"
}

print_warning() {
    echo -e "${WARNING} ${YELLOW}$1${RESET}"
}

print_error() {
    echo -e "${ERROR} ${RED}$1${RESET}"
}

print_info() {
    echo -e "${DIM}  $1${RESET}"
}

# Execute command with optional output suppression
run_quiet() {
    if [ "$LOUD_MODE" = true ]; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

run_with_status() {
    local msg="$1"
    shift

    if [ "$LOUD_MODE" = true ]; then
        print_step "$msg"
        "$@"
    else
        echo -ne "${DIM}${ARROW}${RESET} $msg... "
        if "$@" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            return 1
        fi
    fi
}

# Progress spinner for long operations
show_progress() {
    local pid=$1
    local msg=$2
    local spin='-\|/'
    local i=0

    if [ "$LOUD_MODE" = true ]; then
        wait $pid
        return $?
    fi

    echo -ne "${DIM}${ARROW}${RESET} $msg... "
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %4 ))
        echo -ne "\r${DIM}${ARROW}${RESET} $msg... ${spin:$i:1}"
        sleep 0.1
    done

    wait $pid
    local status=$?
    if [ $status -eq 0 ]; then
        echo -e "\r${DIM}${ARROW}${RESET} $msg... ${CHECKMARK}"
    else
        echo -e "\r${DIM}${ARROW}${RESET} $msg... ${ERROR}"
    fi
    return $status
}

# ============================================================================
# UNIFIED HELPER FUNCTIONS
# ============================================================================

# Check if something exists with consistent pattern
# Usage: check_exists TYPE TARGET [EXTRA]
# Types: file, dir, command, package, db, db_user, service_systemctl, service_brew
check_exists() {
    local type="$1"
    local target="$2"
    local extra="$3"

    case "$type" in
        file)
            [ -f "$target" ]
            ;;
        dir)
            [ -d "$target" ]
            ;;
        command)
            command -v "$target" &> /dev/null
            ;;
        package)
            venv/bin/pip3 show "$target" &> /dev/null
            ;;
        db)
            if [ "$OS" = "linux" ]; then
                sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$target"
            else
                psql -lqt | cut -d \| -f 1 | grep -qw "$target"
            fi
            ;;
        db_user)
            if [ "$OS" = "linux" ]; then
                sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$target'" | grep -q 1
            else
                psql postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$target'" 2>/dev/null | grep -q 1
            fi
            ;;
        service_systemctl)
            systemctl is-active --quiet "$target" 2>/dev/null
            ;;
        service_brew)
            brew services list 2>/dev/null | grep -q "${target}.*started"
            ;;
    esac
}

# Start service with idempotency check
# Usage: start_service SERVICE_NAME SERVICE_TYPE
# Types: systemctl, brew, background (for custom processes)
start_service() {
    local service_name="$1"
    local service_type="$2"

    case "$service_type" in
        systemctl)
            if check_exists service_systemctl "$service_name"; then
                print_info "$service_name already running"
                return 0
            fi
            run_with_status "Starting $service_name" \
                sudo systemctl start "$service_name"
            ;;
        brew)
            if check_exists service_brew "$service_name"; then
                print_info "$service_name already running"
                return 0
            fi
            run_with_status "Starting $service_name" \
                brew services start "$service_name"
            ;;
        background)
            print_error "Background service type requires custom implementation"
            return 1
            ;;
    esac
}

# Stop service with consistent pattern
# Usage: stop_service SERVICE_NAME SERVICE_TYPE [EXTRA]
# Types: systemctl, brew, pid_file (EXTRA=pid_file_path), port (EXTRA=port_number)
stop_service() {
    local service_name="$1"
    local service_type="$2"
    local extra="$3"

    case "$service_type" in
        systemctl)
            if ! check_exists service_systemctl "$service_name"; then
                return 0  # Already stopped
            fi
            run_with_status "Stopping $service_name" \
                sudo systemctl stop "$service_name"
            ;;
        brew)
            if ! check_exists service_brew "$service_name"; then
                return 0  # Already stopped
            fi
            run_with_status "Stopping $service_name" \
                brew services stop "$service_name"
            ;;
        pid_file)
            local pid_file="$extra"
            if [ ! -f "$pid_file" ]; then
                return 0  # PID file doesn't exist
            fi
            local pid=$(cat "$pid_file")
            if ! kill -0 "$pid" 2>/dev/null; then
                rm -f "$pid_file"  # Clean up stale PID file
                return 0
            fi
            kill "$pid" 2>/dev/null && rm -f "$pid_file"
            ;;
        port)
            local port="$extra"
            if command -v lsof &> /dev/null; then
                local pids=$(lsof -ti ":$port" 2>/dev/null)
                if [ -z "$pids" ]; then
                    return 0  # Nothing on port
                fi
                kill $pids 2>/dev/null
            fi
            ;;
    esac
}

# Write file only if content has changed
# Usage: write_file_if_changed FILEPATH CONTENT
write_file_if_changed() {
    local target_file="$1"
    local content="$2"

    if [ -f "$target_file" ]; then
        local existing_content=$(cat "$target_file")
        if [ "$existing_content" = "$content" ]; then
            return 1  # File unchanged
        fi
    fi

    echo "$content" > "$target_file"
    return 0
}

# Install Python package if not already installed
# Usage: install_python_package PACKAGE_NAME
install_python_package() {
    local package="$1"

    if check_exists package "$package"; then
        local version=$(venv/bin/pip3 show "$package" | grep Version | awk '{print $2}')
        echo -e "${CHECKMARK} ${DIM}$version (already installed)${RESET}"
        return 0
    fi

    if [ "$LOUD_MODE" = true ]; then
        print_step "Installing $package..."
        venv/bin/pip3 install "$package"
    else
        (venv/bin/pip3 install -q "$package") &
        show_progress $! "Installing $package"
    fi
}

# Vault helper: Check if Vault is initialized
vault_is_initialized() {
    check_exists file "/opt/vault/init-keys.txt"
}

# Vault helper: Check if Vault is sealed
vault_is_sealed() {
    # vault status returns:
    # - exit code 0 when unsealed
    # - exit code 2 when sealed
    # - exit code 1 on error
    vault status > /dev/null 2>&1
    local exit_code=$?

    if [ $exit_code -eq 2 ]; then
        return 0  # Sealed (true)
    elif [ $exit_code -eq 0 ]; then
        return 1  # Unsealed (false)
    else
        # Error - assume sealed to be safe
        return 0
    fi
}

# Vault helper: Extract credential from init-keys.txt
# Usage: vault_extract_credential "Unseal Key 1" or "Initial Root Token"
vault_extract_credential() {
    local cred_type="$1"

    # Debug output in loud mode
    if [ "$LOUD_MODE" = true ]; then
        echo ""
        echo "DEBUG: Contents of /opt/vault/init-keys.txt:"
        cat /opt/vault/init-keys.txt
        echo ""
        echo "DEBUG: Attempting to extract: $cred_type"
    fi

    grep "$cred_type" /opt/vault/init-keys.txt | awk '{print $NF}'
}

# Vault helper: Unseal vault if sealed
vault_unseal() {
    if ! vault_is_sealed; then
        return 0  # Already unsealed
    fi

    local unseal_key=$(vault_extract_credential "Unseal Key 1")
    if [ -z "$unseal_key" ]; then
        print_error "Cannot unseal: unseal key not found in init-keys.txt"
        return 1
    fi

    run_with_status "Unsealing Vault" \
        vault operator unseal "$unseal_key"
}

# Vault helper: Authenticate with root token
vault_authenticate() {
    if ! vault_is_initialized; then
        print_error "Cannot authenticate: Vault not initialized"
        return 1
    fi

    local root_token=$(vault_extract_credential "Initial Root Token")
    if [ -z "$root_token" ]; then
        print_error "Cannot authenticate: root token not found in init-keys.txt"
        return 1
    fi

    run_with_status "Authenticating with Vault" vault login "$root_token"
}

# Vault helper: Check if AppRole exists
vault_approle_exists() {
    vault read auth/approle/role/mira > /dev/null 2>&1
}

# Vault helper: Full initialization orchestration
vault_initialize() {
    if vault_is_initialized; then
        print_info "Vault already initialized - checking state"

        # Unseal if needed (checks sealed state first)
        vault_unseal || return 1

        # Authenticate with root token
        vault_authenticate || return 1

        # Ensure KV2 secrets engine is enabled
        if ! vault secrets list | grep -q "^secret/"; then
            run_with_status "Enabling KV2 secrets engine" \
                vault secrets enable -version=2 -path=secret kv
        fi

        # Ensure AppRole exists
        if ! vault_approle_exists; then
            print_info "AppRole not found - creating it"

            # Enable AppRole if not enabled
            vault auth enable approle 2>/dev/null || true

            # Create policy if needed
            if ! vault policy read mira-policy > /dev/null 2>&1; then
                cat > /tmp/mira-policy.hcl <<'EOF'
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "secret/metadata/*" {
  capabilities = ["list", "read", "delete"]
}
EOF
                run_with_status "Writing policy to Vault" \
                    vault policy write mira-policy /tmp/mira-policy.hcl
            fi

            run_with_status "Creating AppRole" \
                vault write auth/approle/role/mira policies="mira-policy" token_ttl=1h token_max_ttl=4h
        fi

        # Ensure role-id and secret-id files exist
        if [ ! -f /opt/vault/role-id.txt ]; then
            vault read auth/approle/role/mira/role-id > /opt/vault/role-id.txt
        fi
        if [ ! -f /opt/vault/secret-id.txt ]; then
            vault write -f auth/approle/role/mira/secret-id > /opt/vault/secret-id.txt
        fi

        return 0
    fi

    # Full initialization for new Vault
    echo -ne "${DIM}${ARROW}${RESET} Initializing Vault... "
    if vault operator init -key-shares=1 -key-threshold=1 > /opt/vault/init-keys.txt 2>&1; then
        echo -e "${CHECKMARK}"
        chmod 600 /opt/vault/init-keys.txt
    else
        echo -e "${ERROR}"
        print_error "Failed to initialize Vault"
        return 1
    fi

    vault_unseal || return 1
    vault_authenticate || return 1

    # Enable KV2 secrets engine
    run_with_status "Enabling KV2 secrets engine" \
        vault secrets enable -version=2 -path=secret kv

    # Enable AppRole authentication
    run_with_status "Enabling AppRole authentication" \
        vault auth enable approle

    # Create policy
    cat > /tmp/mira-policy.hcl <<'EOF'
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "secret/metadata/*" {
  capabilities = ["list", "read", "delete"]
}
EOF

    run_with_status "Writing policy to Vault" \
        vault policy write mira-policy /tmp/mira-policy.hcl

    run_with_status "Creating AppRole" \
        vault write auth/approle/role/mira policies="mira-policy" token_ttl=1h token_max_ttl=4h

    # Extract credentials
    vault read auth/approle/role/mira/role-id > /opt/vault/role-id.txt
    vault write -f auth/approle/role/mira/secret-id > /opt/vault/secret-id.txt
}

# Vault helper: Store secret only if it doesn't exist
# Usage: vault_put_if_not_exists SECRET_PATH KEY1=VALUE1 KEY2=VALUE2 ...
vault_put_if_not_exists() {
    local secret_path="$1"
    shift

    if vault kv get "$secret_path" &> /dev/null; then
        print_info "Secret already exists at $secret_path (preserving existing values)"
        return 0
    fi

    run_with_status "Storing secret at $secret_path" \
        vault kv put "$secret_path" "$@"
}

# ============================================================================
# DEPLOYMENT START
# ============================================================================

# Initialize structured state management (must be before first usage)
declare -A COMPONENT_CONFIG  # User configuration choices
declare -A COMPONENT_STATUS  # Component status for summary display

clear
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════╗"
echo "║   MIRA Deployment Script (main)        ║"
echo "╚════════════════════════════════════════╝"
echo -e "${RESET}"
[ "$LOUD_MODE" = true ] && print_info "Running in verbose mode (--loud)"
echo ""

print_header "Pre-flight Checks"

# Check available disk space (need at least 10GB)
echo -ne "${DIM}${ARROW}${RESET} Checking disk space... "
AVAILABLE_SPACE=$(df /opt 2>/dev/null | awk 'NR==2 {print $4}' || df / | awk 'NR==2 {print $4}')
REQUIRED_SPACE=10485760  # 10GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo -e "${ERROR}"
    print_error "Insufficient disk space. Need at least 10GB free, found $(($AVAILABLE_SPACE / 1024 / 1024))GB"
    exit 1
fi
echo -e "${CHECKMARK}"

# Check if installation already exists
if [ -d "/opt/mira/app" ]; then
    echo ""
    print_warning "Existing MIRA installation found at /opt/mira/app"
    read -p "$(echo -e ${YELLOW}This will OVERWRITE the existing installation. Continue? ${RESET})(y/n): " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy](es)?$ ]]; then
        print_info "Installation cancelled."
        exit 0
    fi
    print_info "Proceeding with overwrite..."
    echo ""
fi

print_success "Pre-flight checks passed"

print_header "Port Availability Check"

echo -ne "${DIM}${ARROW}${RESET} Checking ports 1993, 8200, 6379, 5432... "
PORTS_IN_USE=""
for PORT in 1993 8200 6379 5432; do
    if command -v lsof &> /dev/null; then
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            PORTS_IN_USE="$PORTS_IN_USE $PORT"
        fi
    elif command -v netstat &> /dev/null; then
        if netstat -an | grep -q "LISTEN.*:$PORT"; then
            PORTS_IN_USE="$PORTS_IN_USE $PORT"
        fi
    fi
done

if [ -n "$PORTS_IN_USE" ]; then
    echo -e "${WARNING}"
    print_warning "The following ports are already in use:$PORTS_IN_USE"
    print_info "MIRA requires: 1993 (app), 8200 (vault), 6379 (valkey), 5432 (postgresql)"
    read -p "$(echo -e ${YELLOW}Stop existing services and continue?${RESET}) (y/n): " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy](es)?$ ]]; then
        print_info "Installation cancelled. Free up the required ports and try again."
        exit 0
    fi
    echo ""

    # Stop services on occupied ports using unified stop_service function
    print_info "Stopping services on occupied ports..."
    for PORT in $PORTS_IN_USE; do
        case $PORT in
            8200)
                # Vault - canonical method per OS, fallback to port-based stop
                if [ "$OS" = "linux" ]; then
                    echo -ne "${DIM}${ARROW}${RESET} Stopping Vault (port 8200)... "
                    if check_exists service_systemctl vault; then
                        stop_service vault systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Vault" port 8200 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                elif [ "$OS" = "macos" ]; then
                    echo -ne "${DIM}${ARROW}${RESET} Stopping Vault (port 8200)... "
                    if [ -f /opt/vault/vault.pid ]; then
                        stop_service "Vault" pid_file /opt/vault/vault.pid && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Vault" port 8200 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                fi
                ;;
            6379)
                # Valkey - canonical method per OS
                echo -ne "${DIM}${ARROW}${RESET} Stopping Valkey (port 6379)... "
                if [ "$OS" = "linux" ]; then
                    if check_exists service_systemctl valkey; then
                        stop_service valkey systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Valkey" port 6379 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                elif [ "$OS" = "macos" ]; then
                    if check_exists service_brew valkey; then
                        stop_service valkey brew && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Valkey" port 6379 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                fi
                ;;
            5432)
                # PostgreSQL - canonical method per OS
                echo -ne "${DIM}${ARROW}${RESET} Stopping PostgreSQL (port 5432)... "
                if [ "$OS" = "linux" ]; then
                    if check_exists service_systemctl postgresql; then
                        stop_service postgresql systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "PostgreSQL" port 5432 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                elif [ "$OS" = "macos" ]; then
                    if check_exists service_brew postgresql@17; then
                        stop_service postgresql@17 brew && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "PostgreSQL" port 5432 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                fi
                ;;
            1993)
                # MIRA - canonical method per OS
                echo -ne "${DIM}${ARROW}${RESET} Stopping MIRA (port 1993)... "
                if [ "$OS" = "linux" ] && check_exists service_systemctl mira; then
                    stop_service mira systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                else
                    stop_service "MIRA" port 1993 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                fi
                ;;
            *)
                # Unknown service - use port-based stop
                echo -ne "${DIM}${ARROW}${RESET} Stopping process on port $PORT... "
                stop_service "Unknown" port $PORT && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                ;;
        esac
    done
    echo ""
else
    echo -e "${CHECKMARK}"
fi

print_success "Port check passed"

print_header "API Key Configuration"

print_info "MIRA requires both Anthropic and Groq API keys to function."
print_info "You can set them now or skip and configure later (MIRA won't work until both are set)."
echo ""

# Anthropic API Key (required)
echo -e "${BOLD}${BLUE}1. Anthropic API Key${RESET} ${DIM}(REQUIRED)${RESET}"
print_info "Used for: Main LLM operations (Claude models)"
print_info "Get your key at: https://console.anthropic.com/settings/keys"
echo ""
read -p "$(echo -e ${CYAN}Enter your Anthropic API key${RESET}) (or press Enter to skip): " ANTHROPIC_KEY_INPUT
if [ -z "$ANTHROPIC_KEY_INPUT" ]; then
    COMPONENT_CONFIG[anthropic_key]="PLACEHOLDER_SET_THIS_LATER"
    COMPONENT_STATUS[anthropic]="${WARNING} NOT SET - You must configure this before using MIRA"
else
    # Basic validation - check if it looks like an Anthropic key
    if [[ $ANTHROPIC_KEY_INPUT =~ ^sk-ant- ]]; then
        COMPONENT_CONFIG[anthropic_key]="$ANTHROPIC_KEY_INPUT"
        COMPONENT_STATUS[anthropic]="${CHECKMARK} Configured"
    else
        print_warning "This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')"
        read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y/n): " CONFIRM
        if [[ ! "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
            COMPONENT_CONFIG[anthropic_key]="PLACEHOLDER_SET_THIS_LATER"
            COMPONENT_STATUS[anthropic]="${WARNING} NOT SET"
        else
            COMPONENT_CONFIG[anthropic_key]="$ANTHROPIC_KEY_INPUT"
            COMPONENT_STATUS[anthropic]="${CHECKMARK} Configured (unvalidated)"
        fi
    fi
fi
echo ""

# Groq API Key (required)
echo -e "${BOLD}${BLUE}2. Groq API Key${RESET} ${DIM}(REQUIRED)${RESET}"
print_info "Used for: Fast inference and web extraction operations"
print_info "Get your key at: https://console.groq.com/keys"
echo ""
read -p "$(echo -e ${CYAN}Enter your Groq API key${RESET}) (or press Enter to skip): " GROQ_KEY_INPUT
if [ -z "$GROQ_KEY_INPUT" ]; then
    COMPONENT_CONFIG[groq_key]="PLACEHOLDER_SET_THIS_LATER"
    COMPONENT_STATUS[groq]="${WARNING} NOT SET - You must configure this before using MIRA"
else
    # Basic validation - check if it looks like a Groq key
    if [[ $GROQ_KEY_INPUT =~ ^gsk_ ]]; then
        COMPONENT_CONFIG[groq_key]="$GROQ_KEY_INPUT"
        COMPONENT_STATUS[groq]="${CHECKMARK} Configured"
    else
        print_warning "This doesn't look like a valid Groq API key (should start with 'gsk_')"
        read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y/n): " CONFIRM
        if [[ ! "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
            COMPONENT_CONFIG[groq_key]="PLACEHOLDER_SET_THIS_LATER"
            COMPONENT_STATUS[groq]="${WARNING} NOT SET"
        else
            COMPONENT_CONFIG[groq_key]="$GROQ_KEY_INPUT"
            COMPONENT_STATUS[groq]="${CHECKMARK} Configured (unvalidated)"
        fi
    fi
fi
echo ""

# Playwright Browser Installation (optional)
echo -e "${BOLD}${BLUE}3. Playwright Browser Installation${RESET} ${DIM}(OPTIONAL)${RESET}"
print_info "Enables advanced webpage extraction for JavaScript-heavy sites"
print_info "MIRA can function without it (basic HTTP requests still work)"
echo ""
read -p "$(echo -e ${CYAN}Install Playwright and browser dependencies?${RESET}) (y/n, default=y): " PLAYWRIGHT_INPUT
# Default to yes if user just presses Enter
if [ -z "$PLAYWRIGHT_INPUT" ]; then
    PLAYWRIGHT_INPUT="y"
fi
if [[ "$PLAYWRIGHT_INPUT" =~ ^[Yy](es)?$ ]]; then
    COMPONENT_CONFIG[install_playwright]="yes"
    COMPONENT_STATUS[playwright]="${CHECKMARK} Will be installed"
else
    COMPONENT_CONFIG[install_playwright]="no"
    COMPONENT_STATUS[playwright]="${YELLOW}Skipped - webpage extraction unavailable${RESET}"
fi
echo ""

# Detect operating system early for systemd prompt
OS_TYPE=$(uname -s)
case "$OS_TYPE" in
    Linux*)
        OS="linux"
        ;;
    Darwin*)
        OS="macos"
        ;;
    *)
        echo ""
        print_error "Unsupported operating system: $OS_TYPE"
        print_info "This script supports Linux (Ubuntu/Debian) and macOS only."
        exit 1
        ;;
esac

# Systemd service option (Linux only)
echo -e "${BOLD}${BLUE}4. Systemd Service${RESET} ${DIM}(OPTIONAL - Linux Only)${RESET}"
if [ "$OS" = "linux" ]; then
    print_info "Configure MIRA to start automatically on system boot?"
    print_info "This creates a systemd service that starts MIRA when the system boots."
    echo ""
    read -p "$(echo -e ${CYAN}Install MIRA as systemd service?${RESET}) (y/n): " SYSTEMD_INPUT
    if [[ "$SYSTEMD_INPUT" =~ ^[Yy](es)?$ ]]; then
        COMPONENT_CONFIG[install_systemd]="yes"
        echo ""
        read -p "$(echo -e ${CYAN}Start MIRA service immediately after installation?${RESET}) (y/n): " START_NOW_INPUT
        if [[ "$START_NOW_INPUT" =~ ^[Yy](es)?$ ]]; then
            COMPONENT_CONFIG[start_mira_now]="yes"
            COMPONENT_STATUS[systemd]="${CHECKMARK} Will be installed and started"
        else
            COMPONENT_CONFIG[start_mira_now]="no"
            COMPONENT_STATUS[systemd]="${CHECKMARK} Will be installed (not started)"
        fi
    else
        COMPONENT_CONFIG[install_systemd]="no"
        COMPONENT_CONFIG[start_mira_now]="no"
        COMPONENT_STATUS[systemd]="${RED}Skipped${RESET}"
    fi
elif [ "$OS" = "macos" ]; then
    COMPONENT_CONFIG[install_systemd]="no"
    COMPONENT_CONFIG[start_mira_now]="no"
    print_info "Systemd service creation only available on Linux (macOS uses launchd)"
    COMPONENT_STATUS[systemd]="${DIM}Not available on macOS${RESET}"
fi
echo ""

echo -e "${BOLD}Configuration Summary:${RESET}"
echo -e "  Anthropic:       ${COMPONENT_STATUS[anthropic]}"
echo -e "  Groq:            ${COMPONENT_STATUS[groq]}"
echo -e "  Playwright:      ${COMPONENT_STATUS[playwright]}"
echo -e "  Systemd Service: ${COMPONENT_STATUS[systemd]}"
echo ""

print_header "System Detection"

# Display detected operating system
echo -ne "${DIM}${ARROW}${RESET} Detecting operating system... "
case "$OS" in
    linux)
        echo -e "${CHECKMARK} ${DIM}Linux (Ubuntu/Debian)${RESET}"
        ;;
    macos)
        echo -e "${CHECKMARK} ${DIM}macOS${RESET}"
        ;;
esac

# Check if running as root
echo -ne "${DIM}${ARROW}${RESET} Checking user privileges... "
if [ "$EUID" -eq 0 ]; then
   echo -e "${ERROR}"
   print_error "Please do not run this script as root."
   exit 1
fi
echo -e "${CHECKMARK}"

print_header "Beginning Installation"

print_info "This script requires sudo privileges for system package installation."
print_info "Please enter your password - the installation will then run unattended."
echo ""
sudo -v

# Keep sudo alive (Linux only)
if [ "$OS" = "linux" ]; then
    while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
fi

echo ""
print_success "All configuration collected"
print_info "Installation will now proceed unattended (estimated 10-15 minutes)"
print_info "Progress will be displayed as each step completes"
[ "$LOUD_MODE" = false ] && print_info "Use --loud flag to see detailed output"
echo ""
sleep 1

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${DIM}Some of these steps will take a long time. If the spinner is still going, it hasn't${RESET}"
echo -e "${DIM}error'd or timed out—everything is okay. It could take 15 minutes or more to complete.${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

print_header "Step 1: System Dependencies"

if [ "$OS" = "linux" ]; then
    # Add PostgreSQL APT repository for PostgreSQL 17
    if [ ! -f /etc/apt/sources.list.d/pgdg.list ]; then
        run_with_status "Adding PostgreSQL APT repository" \
            bash -c 'sudo apt-get install -y ca-certificates wget > /dev/null 2>&1 && \
                     sudo install -d /usr/share/postgresql-common/pgdg && \
                     sudo wget -q -O /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc && \
                     echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list > /dev/null'
    fi

    # Detect Python version to use (newest available, 3.12+ required)
    PYTHON_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating package lists..."
        sudo apt-get update
        print_step "Installing system packages (Python ${PYTHON_VER})..."
        sudo apt-get install -y \
            build-essential \
            python${PYTHON_VER}-venv \
            python${PYTHON_VER}-dev \
            libpq-dev \
            postgresql-server-dev-17 \
            unzip \
            wget \
            curl \
            postgresql-17 \
            postgresql-contrib \
            postgresql-17-pgvector \
            valkey \
            libatk1.0-0t64 \
            libatk-bridge2.0-0t64 \
            libatspi2.0-0t64 \
            libxcomposite1
    else
        # Silent mode with progress indicator
        (sudo apt-get update > /dev/null 2>&1) &
        show_progress $! "Updating package lists"

        (sudo apt-get install -y \
            build-essential python${PYTHON_VER}-venv python${PYTHON_VER}-dev libpq-dev \
            postgresql-server-dev-17 unzip wget curl postgresql-17 \
            postgresql-contrib postgresql-17-pgvector valkey \
            libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 \
            libxcomposite1 > /dev/null 2>&1) &
        show_progress $! "Installing system packages (18 packages)"
    fi
elif [ "$OS" = "macos" ]; then
    # macOS Homebrew package installation
    # Check if Homebrew is installed
    echo -ne "${DIM}${ARROW}${RESET} Checking for Homebrew... "
    if ! command -v brew &> /dev/null; then
        echo -e "${ERROR}"
        print_error "Homebrew is not installed. Please install Homebrew first:"
        print_info "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    echo -e "${CHECKMARK}"

    # Detect Python version to use (newest available, 3.12+ required)
    PYTHON_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating Homebrew..."
        brew update
        print_step "Installing dependencies via Homebrew (Python ${PYTHON_VER})..."
        brew install python@${PYTHON_VER} wget curl postgresql@17 valkey vault
    else
        (brew update > /dev/null 2>&1) &
        show_progress $! "Updating Homebrew"

        (brew install python@${PYTHON_VER} wget curl postgresql@17 valkey vault > /dev/null 2>&1) &
        show_progress $! "Installing dependencies via Homebrew (6 packages)"
    fi

    print_info "Playwright will install its own browser dependencies"
fi

print_success "System dependencies installed"

print_header "Step 2: Python Verification"

echo -ne "${DIM}${ARROW}${RESET} Locating Python ${PYTHON_VER}+... "
if [ "$OS" = "linux" ]; then
    # Use the version detected in Step 1
    if ! command -v python${PYTHON_VER} &> /dev/null; then
        echo -e "${ERROR}"
        print_error "Python ${PYTHON_VER} not found after installation."
        exit 1
    fi
    PYTHON_CMD="python${PYTHON_VER}"
elif [ "$OS" = "macos" ]; then
    # Detect macOS Python version
    PYTHON_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)

    # Check common Homebrew locations
    if command -v python${PYTHON_VER} &> /dev/null; then
        PYTHON_CMD="python${PYTHON_VER}"
    elif [ -f "/opt/homebrew/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}" ]; then
        PYTHON_CMD="/opt/homebrew/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}"
    elif [ -f "/usr/local/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}" ]; then
        PYTHON_CMD="/usr/local/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}"
    else
        echo -e "${ERROR}"
        print_error "Python ${PYTHON_VER} not found. Check Homebrew installation."
        exit 1
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${CHECKMARK} ${DIM}$PYTHON_VERSION${RESET}"

print_header "Step 3: MIRA Download & Installation"

# Determine user/group for ownership
if [ "$OS" = "linux" ]; then
    MIRA_USER="$(whoami)"
    MIRA_GROUP="$(id -gn)"
elif [ "$OS" = "macos" ]; then
    MIRA_USER="$(whoami)"
    MIRA_GROUP="staff"
fi

# Download to /tmp to keep user's home directory clean
cd /tmp

# NOTE: Currently downloads from main branch for active development
# When ready for stable release, change to:
#   wget -q -O mira-X.XX.tar.gz https://github.com/taylorsatula/mira-OSS/archive/refs/tags/X.XX.tar.gz
#   tar -xzf mira-X.XX.tar.gz -C /tmp
#   sudo cp -r /tmp/mira-OSS-X.XX/* /opt/mira/app/
#   rm -f /tmp/mira-X.XX.tar.gz
#   rm -rf /tmp/mira-OSS-X.XX

run_with_status "Downloading MIRA from main branch" \
    wget -q -O mira-main.tar.gz https://github.com/taylorsatula/mira-OSS/archive/refs/heads/main.tar.gz

run_with_status "Creating /opt/mira/app directory" \
    sudo mkdir -p /opt/mira/app

run_with_status "Extracting archive" \
    tar -xzf mira-main.tar.gz -C /tmp

run_with_status "Copying files to /opt/mira/app" \
    sudo cp -r /tmp/mira-OSS-main/* /opt/mira/app/

run_with_status "Setting ownership to $MIRA_USER:$MIRA_GROUP" \
    sudo chown -R $MIRA_USER:$MIRA_GROUP /opt/mira

# Clean up immediately after copying
run_quiet rm -f /tmp/mira-main.tar.gz
run_quiet rm -rf /tmp/mira-OSS-main

print_success "MIRA installed to /opt/mira/app"

print_header "Step 4: Python Environment Setup"

cd /opt/mira/app

# Check if venv already exists
echo -ne "${DIM}${ARROW}${RESET} Checking for existing virtual environment... "
if [ -f venv/bin/python3 ]; then
    VENV_PYTHON_VERSION=$(venv/bin/python3 --version 2>&1 | awk '{print $2}')
    echo -e "${CHECKMARK} ${DIM}$VENV_PYTHON_VERSION (existing)${RESET}"
    print_info "Reusing existing virtual environment"
else
    echo -e "${DIM}(not found)${RESET}"
    run_with_status "Creating virtual environment" \
        $PYTHON_CMD -m venv venv

    run_with_status "Initializing pip" \
        venv/bin/python3 -m ensurepip
fi

echo -ne "${DIM}${ARROW}${RESET} Checking PyTorch installation... "
if check_exists package torch; then
    TORCH_VERSION=$(venv/bin/pip3 show torch | grep Version | awk '{print $2}')
    echo -e "${CHECKMARK} ${DIM}$TORCH_VERSION (existing)${RESET}"
    print_info "Note: If you have CUDA-enabled PyTorch, it will be preserved"
else
    echo -e "${DIM}(not installed yet)${RESET}"
    if [ "$LOUD_MODE" = true ]; then
        print_step "Installing PyTorch CPU-only version..."
        venv/bin/pip3 install torch --index-url https://download.pytorch.org/whl/cpu
    else
        (venv/bin/pip3 install -q torch --index-url https://download.pytorch.org/whl/cpu) &
        show_progress $! "Installing PyTorch CPU-only"
    fi
fi

print_header "Step 5: Python Dependencies"

# Count packages in requirements.txt
PACKAGE_COUNT=$(grep -c '^[^#]' requirements.txt 2>/dev/null || echo "many")
echo -e "${DIM}This is the one that is going to take a while (~${PACKAGE_COUNT} packages)${RESET}"
echo ""

if [ "$LOUD_MODE" = true ]; then
    print_step "Installing from requirements.txt..."
    venv/bin/pip3 install -r requirements.txt
else
    (venv/bin/pip3 install -q -r requirements.txt) &
    show_progress $! "Installing Python packages from requirements.txt"
    if [ $? -ne 0 ]; then
        print_error "Failed to install Python packages from requirements.txt"
        print_info "Run with --loud flag to see detailed error output"
        exit 1
    fi
fi

# Install sentence-transformers separately to ensure proper dependency resolution
# (torch, transformers, tokenizers must be installed first from requirements.txt)
echo -ne "${DIM}${ARROW}${RESET} Checking sentence-transformers... "
if ! check_exists package sentence-transformers; then
    echo ""
    install_python_package sentence-transformers
    if [ $? -ne 0 ]; then
        print_error "Failed to install sentence-transformers"
        print_info "Run with --loud flag to see detailed error output"
        exit 1
    fi
else
    install_python_package sentence-transformers  # This will show version if already installed
fi

echo -ne "${DIM}${ARROW}${RESET} Checking spaCy language model... "
if venv/bin/python3 -c "import spacy.util; exit(0 if spacy.util.is_package('en_core_web_lg') else 1)" 2>/dev/null; then
    echo -e "${CHECKMARK} ${DIM}(already installed)${RESET}"
else
    echo -e "${DIM}(not found)${RESET}"
    if [ "$LOUD_MODE" = true ]; then
        print_step "Installing spaCy language model..."
        venv/bin/python3 -m spacy download en_core_web_lg
    else
        (venv/bin/python3 -m spacy download en_core_web_lg > /dev/null 2>&1) &
        show_progress $! "Installing spaCy language model"
    fi
fi

print_success "Python dependencies installed"

print_header "Step 6: AI Model Downloads"

# Check what's already cached by verifying required files exist
echo -ne "${DIM}${ARROW}${RESET} Checking embedding model cache... "
MODELS_CACHED=$(venv/bin/python3 << 'EOF'
from pathlib import Path
import os

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

def check_model_cached(model_substrings):
    """Check if a model is fully cached by looking for model directories and required files"""
    if not cache_dir.exists():
        return False

    # Find directories matching any of the substrings
    model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and any(s in d.name for s in model_substrings)]

    for model_dir in model_dirs:
        # Check for essential files that indicate a complete download
        # Look in snapshots subdirectory where actual model files are stored
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    # Check for config and model files
                    has_config = (snapshot / "config.json").exists()
                    has_model = (snapshot / "pytorch_model.bin").exists() or (snapshot / "model.safetensors").exists()
                    if has_config and has_model:
                        return True
    return False

has_minilm = check_model_cached(["all-MiniLM-L6-v2"])
has_reranker = check_model_cached(["bge-reranker-base"])

if has_minilm and has_reranker:
    print("all")
elif has_minilm or has_reranker:
    print("partial")
else:
    print("none")
EOF
)

if [ "$MODELS_CACHED" = "all" ]; then
    echo -e "${CHECKMARK} ${DIM}(both models already cached)${RESET}"
    print_info "To re-download: rm -rf ~/.cache/huggingface/hub"
elif [ "$MODELS_CACHED" = "partial" ]; then
    echo -e "${DIM}(some models cached, downloading missing)${RESET}"
else
    echo -e "${DIM}(not found)${RESET}"
fi

# Only download if not all cached
if [ "$MODELS_CACHED" != "all" ]; then
    if [ "$LOUD_MODE" = true ]; then
        print_step "Downloading embedding and reranker models..."
        venv/bin/python3 << 'EOF'
from sentence_transformers import SentenceTransformer
print("→ Loading/downloading all-MiniLM-L6-v2...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✓ all-MiniLM-L6-v2 ready")
print("→ Loading/downloading BAAI/bge-reranker-base...")
SentenceTransformer("BAAI/bge-reranker-base")
print("✓ BGE reranker ready")
EOF
    else
        (venv/bin/python3 << 'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
SentenceTransformer("BAAI/bge-reranker-base")
EOF
) &
        show_progress $! "Downloading embedding models"
    fi
fi

print_header "Step 7: Playwright Browser Setup"

if [ "${COMPONENT_CONFIG[install_playwright]}" = "yes" ]; then
    # Check if Playwright Chromium is already installed
    PLAYWRIGHT_CACHE="$HOME/.cache/ms-playwright"
    echo -ne "${DIM}${ARROW}${RESET} Checking Playwright cache... "
    if [ -d "$PLAYWRIGHT_CACHE" ] && ls "$PLAYWRIGHT_CACHE"/chromium-* >/dev/null 2>&1; then
        echo -e "${CHECKMARK} ${DIM}(already installed)${RESET}"
        print_info "To update browsers: venv/bin/playwright install chromium"
    else
        echo -e "${DIM}(not found)${RESET}"
        if [ "$LOUD_MODE" = true ]; then
            print_step "Installing Playwright Chromium browser..."
            venv/bin/playwright install chromium
        else
            (venv/bin/playwright install chromium > /dev/null 2>&1) &
            show_progress $! "Installing Playwright Chromium"
        fi
    fi

    # System dependencies - optional, may fail on newer Ubuntu
    if [ "$OS" = "linux" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Installing Playwright system dependencies... "
        if sudo venv/bin/playwright install-deps > /tmp/playwright-deps.log 2>&1; then
            echo -e "${CHECKMARK}"
            rm -f /tmp/playwright-deps.log
        else
            echo -e "${WARNING}"
            print_warning "Some system dependencies failed to install"

            # Extract specific failed packages if possible
            FAILED_PACKAGES=$(grep -oP "Unable to locate package \K\S+" /tmp/playwright-deps.log 2>/dev/null | head -3 | tr '\n' ' ')
            if [ -n "$FAILED_PACKAGES" ]; then
                print_info "Missing packages: $FAILED_PACKAGES"
            fi

            print_info "This is common on Ubuntu 24.04+ due to package name changes"
            print_info "Playwright should still work in headless mode for most sites"
            print_info "Full log saved to: /tmp/playwright-deps.log"
        fi
    elif [ "$OS" = "macos" ]; then
        print_info "Playwright browser dependencies are bundled on macOS"
    fi

    print_success "Playwright configured"
else
    print_info "Playwright installation skipped (user opted out)"
    print_info "Note: Advanced webpage extraction will not be available"
    print_info "Basic HTTP requests and web search will still work"
    print_success "Playwright setup skipped"
fi

print_header "Step 8: HashiCorp Vault Setup"

if [ "$OS" = "linux" ]; then
    # Detect architecture
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)
            VAULT_ARCH="amd64"
            ;;
        aarch64|arm64)
            VAULT_ARCH="arm64"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    cd /tmp
    run_with_status "Downloading Vault 1.18.3 (${VAULT_ARCH})" \
        wget -q https://releases.hashicorp.com/vault/1.18.3/vault_1.18.3_linux_${VAULT_ARCH}.zip

    run_with_status "Extracting Vault binary" \
        unzip -o vault_1.18.3_linux_${VAULT_ARCH}.zip

    run_with_status "Installing to /usr/local/bin" \
        sudo mv vault /usr/local/bin/

    run_quiet sudo chmod +x /usr/local/bin/vault
elif [ "$OS" = "macos" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Verifying Vault installation... "
    if ! command -v vault &> /dev/null; then
        echo -e "${ERROR}"
        print_error "Vault installation failed. Please install manually: brew install vault"
        exit 1
    fi
    echo -e "${CHECKMARK}"
fi

run_with_status "Creating Vault directories" \
    sudo mkdir -p /opt/vault/data /opt/vault/config /opt/vault/logs

run_with_status "Setting Vault directory ownership" \
    sudo chown -R $MIRA_USER:$MIRA_GROUP /opt/vault

echo -ne "${DIM}${ARROW}${RESET} Writing Vault configuration... "
cat > /opt/vault/config/vault.hcl <<'EOF'
storage "file" {
  path = "/opt/vault/data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_disable = 1
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
ui = true

log_level = "Info"
EOF
echo -e "${CHECKMARK}"

print_header "Step 9: Vault Service Configuration"

if [ "$OS" = "linux" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Creating systemd service... "
    sudo tee /etc/systemd/system/vault.service > /dev/null <<EOF
[Unit]
Description=HashiCorp Vault
Documentation=https://www.vaultproject.io/docs/
Requires=network-online.target
After=network-online.target
ConditionFileNotEmpty=/opt/vault/config/vault.hcl

[Service]
Type=notify
User=$MIRA_USER
Group=$MIRA_GROUP
ProtectSystem=full
ProtectHome=no
PrivateTmp=yes
ExecStart=/usr/local/bin/vault server -config=/opt/vault/config/vault.hcl
ExecReload=/bin/kill --signal HUP \$MAINPID
KillMode=process
KillSignal=SIGINT
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF
    echo -e "${CHECKMARK}"

    run_quiet sudo systemctl daemon-reload
    run_with_status "Enabling Vault service" \
        sudo systemctl enable vault.service

    start_service vault.service systemctl
    sleep 2
elif [ "$OS" = "macos" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Starting Vault service... "
    # Start Vault in the background
    vault server -config=/opt/vault/config/vault.hcl > /opt/vault/logs/vault.log 2>&1 &
    VAULT_PID=$!
    echo $VAULT_PID > /opt/vault/vault.pid
    sleep 2

    # Verify Vault started
    if ! kill -0 $VAULT_PID 2>/dev/null; then
        echo -e "${ERROR}"
        print_error "Vault failed to start. Check /opt/vault/logs/vault.log for details."
        exit 1
    fi
    echo -e "${CHECKMARK} ${DIM}PID $VAULT_PID${RESET}"
fi

print_success "Vault service configured and running"

# Wait for Vault to be ready and check initialization state
echo -ne "${DIM}${ARROW}${RESET} Waiting for Vault to be ready... "
export VAULT_ADDR='http://127.0.0.1:8200'
VAULT_READY=0
for i in {1..30}; do
    if curl -s http://127.0.0.1:8200/v1/sys/health > /dev/null 2>&1; then
        VAULT_READY=1
        break
    fi
    sleep 1
done

if [ $VAULT_READY -eq 0 ]; then
    echo -e "${ERROR}"
    print_error "Vault did not become ready within 30 seconds"
    print_info "Check Vault logs: /opt/vault/logs/vault.log"
    exit 1
fi
echo -e "${CHECKMARK} ${DIM}(ready after ${i}s)${RESET}"

print_header "Step 10: Vault Initialization"

# Use unified vault_initialize function (handles check, unseal, auth, policy, AppRole)
vault_initialize
print_success "Vault fully configured"

print_header "Step 11: Auto-Unseal Configuration"

echo -ne "${DIM}${ARROW}${RESET} Creating unseal script... "
cat > /opt/vault/unseal.sh <<'EOF'
#!/bin/bash
export VAULT_ADDR='http://127.0.0.1:8200'
sleep 5
UNSEAL_KEY=$(grep 'Unseal Key 1:' /opt/vault/init-keys.txt | awk '{print $4}')
echo "$UNSEAL_KEY" | vault operator unseal -
EOF
echo -e "${CHECKMARK}"

run_quiet chmod +x /opt/vault/unseal.sh

if [ "$OS" = "linux" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Creating auto-unseal systemd service... "
    sudo tee /etc/systemd/system/vault-unseal.service > /dev/null <<'EOF'
[Unit]
Description=Vault Auto-Unseal
After=vault.service
Requires=vault.service

[Service]
Type=oneshot
ExecStart=/opt/vault/unseal.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    echo -e "${CHECKMARK}"

    run_quiet sudo systemctl daemon-reload
    run_with_status "Enabling auto-unseal service" \
        sudo systemctl enable vault-unseal.service
elif [ "$OS" = "macos" ]; then
    print_info "On macOS, manually unseal Vault after restart using: /opt/vault/unseal.sh"
fi

print_success "Auto-unseal configured"

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
        # On Linux, check with pg_isready
        if sudo -u postgres pg_isready > /dev/null 2>&1; then
            PG_READY=1
            break
        fi
    elif [ "$OS" = "macos" ]; then
        # On macOS, check with pg_isready as current user
        if pg_isready > /dev/null 2>&1; then
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
        print_info "Check status: systemctl status postgresql"
        print_info "Check logs: journalctl -u postgresql -n 50"
    elif [ "$OS" = "macos" ]; then
        print_info "Check status: brew services list | grep postgresql"
        print_info "Check logs: brew services info postgresql@17"
    fi
    exit 1
fi
echo -e "${CHECKMARK} ${DIM}(ready after ${i}s)${RESET}"

print_header "Step 13: PostgreSQL Configuration"

# Check if database exists, create if not
echo -ne "${DIM}${ARROW}${RESET} Creating database 'mira_service'... "
if check_exists db mira_service; then
    echo -e "${DIM}(already exists)${RESET}"
else
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -c "CREATE DATABASE mira_service;" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to create database 'mira_service'"
            exit 1
        fi
    elif [ "$OS" = "macos" ]; then
        if createdb mira_service > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to create database 'mira_service'"
            exit 1
        fi
    fi
fi

# Check if user mira_admin exists, create if not
echo -ne "${DIM}${ARROW}${RESET} Creating user 'mira_admin'... "
if check_exists db_user mira_admin; then
    echo -e "${DIM}(already exists)${RESET}"
else
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -c "CREATE USER mira_admin WITH PASSWORD 'changethisifdeployingpwd' SUPERUSER;" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to create user 'mira_admin'"
            exit 1
        fi
    elif [ "$OS" = "macos" ]; then
        if psql postgres -c "CREATE USER mira_admin WITH PASSWORD 'changethisifdeployingpwd' SUPERUSER;" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to create user 'mira_admin'"
            exit 1
        fi
    fi
fi

# Check if user mira_dbuser exists, create if not
echo -ne "${DIM}${ARROW}${RESET} Creating user 'mira_dbuser'... "
if check_exists db_user mira_dbuser; then
    echo -e "${DIM}(already exists)${RESET}"
else
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -c "CREATE USER mira_dbuser WITH PASSWORD 'changethisifdeployingpwd';" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to create user 'mira_dbuser'"
            exit 1
        fi
    elif [ "$OS" = "macos" ]; then
        if psql postgres -c "CREATE USER mira_dbuser WITH PASSWORD 'changethisifdeployingpwd';" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to create user 'mira_dbuser'"
            exit 1
        fi
    fi
fi

# Grant privileges (OS-specific commands)
if [ "$OS" = "linux" ]; then
    run_with_status "Granting privileges to mira_admin" \
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_admin;"

    run_with_status "Granting privileges to mira_dbuser" \
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_dbuser;"

    run_with_status "Enabling pgvector extension" \
        sudo -u postgres psql -d mira_service -c "CREATE EXTENSION IF NOT EXISTS vector;"
elif [ "$OS" = "macos" ]; then
    run_with_status "Granting privileges to mira_admin" \
        psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_admin;"

    run_with_status "Granting privileges to mira_dbuser" \
        psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_dbuser;"

    run_with_status "Enabling pgvector extension" \
        psql -d mira_service -c "CREATE EXTENSION IF NOT EXISTS vector;"
fi

print_success "PostgreSQL configured"

print_header "Step 14: Vault Credential Storage"

vault_put_if_not_exists secret/mira/api_keys \
    anthropic_key="${COMPONENT_CONFIG[anthropic_key]}" \
    groq_key="${COMPONENT_CONFIG[groq_key]}"

# ⚠️  SECURITY WARNING: Change default passwords before deploying to production!
# The password "changethisifdeployingpwd" is a placeholder and MUST be replaced with a strong password.
vault_put_if_not_exists secret/mira/database \
    admin_url="postgresql://mira_admin:changethisifdeployingpwd@localhost:5432/mira_service" \
    password="changethisifdeployingpwd" \
    username="mira_dbuser" \
    service_url="postgresql://mira_dbuser:changethisifdeployingpwd@localhost:5432/mira_service"

vault_put_if_not_exists secret/mira/services \
    app_url="http://localhost:1993" \
    valkey_url="valkey://localhost:6379"

print_success "All credentials configured in Vault"

print_header "Step 15: MIRA CLI Setup"

echo -ne "${DIM}${ARROW}${RESET} Creating mira wrapper script... "

# Create mira wrapper script that sets Vault environment variables
# Note: MIRA's main.py automatically creates and stores the API token in Vault on first startup
cat > /opt/mira/mira.sh <<'WRAPPER_EOF'
#!/bin/bash
# MIRA CLI wrapper - sets Vault environment variables for talkto_mira.py

# Set Vault environment variables
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_ROLE_ID=$(grep 'role_id' /opt/vault/role-id.txt | awk '{print $2}')
export VAULT_SECRET_ID=$(grep 'secret_id' /opt/vault/secret-id.txt | awk '{print $2}')

# Launch MIRA CLI
/opt/mira/app/venv/bin/python3 /opt/mira/app/talkto_mira.py "$@"
WRAPPER_EOF
echo -e "${CHECKMARK}"

run_quiet chmod +x /opt/mira/mira.sh

# Add alias to shell RC
if [ "$OS" = "linux" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ "$OS" = "macos" ]; then
    # macOS typically uses zsh
    if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bash_profile"
    fi
fi

echo -ne "${DIM}${ARROW}${RESET} Adding 'mira' alias to $SHELL_RC... "
if ! grep -q "alias mira=" "$SHELL_RC" 2>/dev/null; then
    echo "alias mira='/opt/mira/mira.sh'" >> "$SHELL_RC"
    echo -e "${CHECKMARK}"
else
    echo -e "${DIM}(already exists)${RESET}"
fi

print_success "MIRA CLI configured"

# Systemd service installation (Linux only, if user opted in)
if [ "${COMPONENT_CONFIG[install_systemd]}" = "yes" ] && [ "$OS" = "linux" ]; then
    print_header "Step 16: Systemd Service Configuration"

    # Extract Vault credentials from files
    echo -ne "${DIM}${ARROW}${RESET} Reading Vault credentials... "
    VAULT_ROLE_ID=$(grep 'role_id' /opt/vault/role-id.txt | awk '{print $2}')
    VAULT_SECRET_ID=$(grep 'secret_id' /opt/vault/secret-id.txt | awk '{print $2}')

    if [ -z "$VAULT_ROLE_ID" ] || [ -z "$VAULT_SECRET_ID" ]; then
        echo -e "${ERROR}"
        print_error "Failed to read Vault credentials from /opt/vault/"
        print_info "Skipping systemd service creation"
        COMPONENT_CONFIG[install_systemd]="failed"
        COMPONENT_STATUS[mira_service]="${ERROR} Configuration failed"
    else
        echo -e "${CHECKMARK}"

        # Create systemd service file
        echo -ne "${DIM}${ARROW}${RESET} Creating systemd service file... "
        sudo tee /etc/systemd/system/mira.service > /dev/null <<EOF
[Unit]
Description=MIRA - AI Assistant with Persistent Memory
Documentation=https://github.com/taylorsatula/mira-OSS
Requires=vault.service postgresql.service valkey.service
After=vault.service postgresql.service valkey.service vault-unseal.service
ConditionPathExists=/opt/mira/app/main.py

[Service]
Type=simple
User=$MIRA_USER
Group=$MIRA_GROUP
WorkingDirectory=/opt/mira/app
Environment="VAULT_ADDR=http://127.0.0.1:8200"
Environment="VAULT_ROLE_ID=$VAULT_ROLE_ID"
Environment="VAULT_SECRET_ID=$VAULT_SECRET_ID"
ExecStart=/opt/mira/app/venv/bin/python3 /opt/mira/app/main.py
Restart=on-failure
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mira

[Install]
WantedBy=multi-user.target
EOF
        echo -e "${CHECKMARK}"

        # Reload systemd and enable service
        run_quiet sudo systemctl daemon-reload

        run_with_status "Enabling MIRA service for auto-start on boot" \
            sudo systemctl enable mira.service

        print_success "Systemd service configured"
        print_info "Service will auto-start on system boot"

        # Start service if user chose to during configuration
        if [ "${COMPONENT_CONFIG[start_mira_now]}" = "yes" ]; then
            echo ""
            start_service mira.service systemctl

            # Give service a moment to start
            sleep 2

            # Check if service started successfully
            if sudo systemctl is-active --quiet mira.service; then
                print_success "MIRA service is running"
                print_info "View logs: journalctl -u mira -f"
                COMPONENT_STATUS[mira_service]="${CHECKMARK} Running"
            else
                print_warning "MIRA service may have failed to start"
                print_info "Check status: systemctl status mira"
                print_info "View logs: journalctl -u mira -n 50"
                COMPONENT_STATUS[mira_service]="${ERROR} Start failed"
            fi
        else
            print_info "To start later: sudo systemctl start mira"
            print_info "To view logs: journalctl -u mira -f"
            COMPONENT_STATUS[mira_service]="${DIM}Not started${RESET}"
        fi
    fi
elif [ "${COMPONENT_CONFIG[install_systemd]}" = "no" ]; then
    print_header "Step 16: Systemd Service Configuration"
    print_info "Skipping systemd service installation (user opted out)"
fi

print_header "Step 17: Cleanup"

if [ "$LOUD_MODE" = true ]; then
    print_step "Flushing pip cache..."
    venv/bin/pip3 cache purge 2>/dev/null || print_info "pip cache purge skipped (cache may be empty)"
else
    run_with_status "Flushing pip cache" \
        venv/bin/pip3 cache purge 2>/dev/null || true
fi

# Remove temporary files silently
run_quiet rm -f /tmp/mira-policy.hcl

if [ "$OS" = "linux" ]; then
    run_quiet rm -f /tmp/vault_1.18.3_linux_*.zip
    run_quiet rm -f /tmp/vault
fi

# Rename deploy script to archive it
echo -ne "${DIM}${ARROW}${RESET} Archiving deploy script... "
SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
SCRIPT_ARCHIVED=""

# Only rename if it's actually a file (not piped from curl)
if [ -f "$SCRIPT_PATH" ] && [ "$SCRIPT_NAME" = "deploy.sh" ]; then
    TIMESTAMP=$(date +%m%d%Y)
    NEW_NAME="deploy-lastrun-${TIMESTAMP}.sh"
    NEW_PATH="$SCRIPT_DIR/$NEW_NAME"

    # If archived version already exists, add a counter
    COUNTER=1
    while [ -f "$NEW_PATH" ]; do
        NEW_NAME="deploy-lastrun-${TIMESTAMP}-${COUNTER}.sh"
        NEW_PATH="$SCRIPT_DIR/$NEW_NAME"
        COUNTER=$((COUNTER + 1))
    done

    mv "$SCRIPT_PATH" "$NEW_PATH"
    SCRIPT_ARCHIVED="$NEW_NAME"
    echo -e "${CHECKMARK} ${DIM}$NEW_NAME${RESET}"
else
    echo -e "${DIM}(skipped - not a file)${RESET}"
fi

print_success "Cleanup complete"

echo ""
echo ""
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════╗"
echo "║       Deployment Complete! 🎉          ║"
echo "╚════════════════════════════════════════╝"
echo -e "${RESET}"
echo ""

print_success "MIRA installed to: /opt/mira/app"
print_success "All temporary files cleaned up"
if [ -n "$SCRIPT_ARCHIVED" ]; then
    print_success "Deploy script archived as: $SCRIPT_ARCHIVED"
fi

echo ""
echo -e "${BOLD}${BLUE}Important Files${RESET} ${DIM}(/opt/vault/)${RESET}"
print_info "init-keys.txt (Vault unseal key and root token)"
print_info "role-id.txt (AppRole role ID)"
print_info "secret-id.txt (AppRole secret ID)"
if [ "$OS" = "macos" ]; then
    print_info "vault.pid (Vault process ID)"
fi

echo ""
echo -e "${BOLD}${BLUE}API Key Configuration${RESET}"
echo -e "  Anthropic: ${COMPONENT_STATUS[anthropic]}"
echo -e "  Groq:      ${COMPONENT_STATUS[groq]}"

if [ "${COMPONENT_CONFIG[anthropic_key]}" = "PLACEHOLDER_SET_THIS_LATER" ] || [ "${COMPONENT_CONFIG[groq_key]}" = "PLACEHOLDER_SET_THIS_LATER" ]; then
    echo ""
    print_warning "Required API keys not configured!"
    print_info "MIRA will not work until you set both API keys."
    print_info "To configure later, use Vault CLI:"
    echo -e "${DIM}    export VAULT_ADDR='http://127.0.0.1:8200'${RESET}"
    echo -e "${DIM}    vault login <root-token-from-init-keys.txt>${RESET}"
    echo -e "${DIM}    vault kv put secret/mira/api_keys \\\\${RESET}"
    echo -e "${DIM}      anthropic_key=\"sk-ant-your-key\" \\\\${RESET}"
    echo -e "${DIM}      groq_key=\"gsk_your-key\"${RESET}"
fi

echo ""
echo -e "${BOLD}${BLUE}Services Running${RESET}"
if [ "$OS" = "linux" ]; then
    print_info "Valkey: localhost:6379"
    print_info "Vault: http://localhost:8200 (systemd service)"
    print_info "PostgreSQL: localhost:5432 (systemd service)"
    if [ "${COMPONENT_CONFIG[install_systemd]}" = "yes" ]; then
        print_info "MIRA: http://localhost:1993 (systemd service - ${COMPONENT_STATUS[mira_service]})"
    fi
elif [ "$OS" = "macos" ]; then
    print_info "Valkey: localhost:6379 (brew services)"
    print_info "Vault: http://localhost:8200 (background process)"
    print_info "PostgreSQL: localhost:5432 (brew services)"
fi

echo ""
echo -e "${BOLD}${GREEN}Next Steps${RESET}"
if [ "${COMPONENT_CONFIG[install_systemd]}" = "yes" ] && [ "$OS" = "linux" ]; then
    if [[ "${COMPONENT_STATUS[mira_service]}" == *"Running"* ]]; then
        echo -e "  ${CYAN}→${RESET} MIRA is running at: ${BOLD}http://localhost:1993${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} View logs: ${BOLD}journalctl -u mira -f${RESET}"
        echo -e "  ${CYAN}→${RESET} Stop MIRA: ${BOLD}sudo systemctl stop mira${RESET}"
    elif [[ "${COMPONENT_STATUS[mira_service]}" == *"failed"* ]]; then
        echo -e "  ${CYAN}→${RESET} Check logs: ${BOLD}journalctl -u mira -n 50${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} Try starting: ${BOLD}sudo systemctl start mira${RESET}"
    else
        echo -e "  ${CYAN}→${RESET} Start MIRA: ${BOLD}sudo systemctl start mira${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} View logs: ${BOLD}journalctl -u mira -f${RESET}"
    fi
    echo ""
    print_info "MIRA will auto-start on system boot (systemd enabled)"
elif [ "$OS" = "linux" ]; then
    echo -e "  ${CYAN}→${RESET} Run: ${BOLD}source ~/.bashrc && mira${RESET}"
elif [ "$OS" = "macos" ]; then
    echo -e "  ${CYAN}→${RESET} Run: ${BOLD}source $SHELL_RC && mira${RESET}"
fi

echo ""
print_warning "IMPORTANT: Secure /opt/vault/ - it contains sensitive credentials!"

if [ "$OS" = "macos" ]; then
    echo ""
    echo -e "${BOLD}${YELLOW}macOS Notes${RESET}"
    print_info "Vault is running as a background process"
    print_info "To stop: kill \$(cat /opt/vault/vault.pid)"
    print_info "After system restart, manually start Vault and unseal:"
    echo -e "${DIM}    /opt/vault/unseal.sh${RESET}"
    print_info "PostgreSQL and Valkey are managed by brew services"
fi

# Prompt to launch MIRA CLI immediately
echo ""
echo -e "${BOLD}${CYAN}Launch MIRA CLI Now?${RESET}"
print_info "MIRA CLI will auto-start the API server and open an interactive chat."
echo ""
read -p "$(echo -e ${CYAN}Start MIRA CLI now?${RESET}) (yes/no): " LAUNCH_MIRA
if [ "$LAUNCH_MIRA" = "yes" ]; then
    echo ""
    print_success "Launching MIRA CLI..."
    echo ""
    # Set up Vault environment and launch
    export VAULT_ADDR='http://127.0.0.1:8200'
    export VAULT_ROLE_ID=$(grep 'role_id' /opt/vault/role-id.txt | awk '{print $2}')
    export VAULT_SECRET_ID=$(grep 'secret_id' /opt/vault/secret-id.txt | awk '{print $2}')
    cd /opt/mira/app
    exec venv/bin/python3 talkto_mira.py
fi

echo ""
