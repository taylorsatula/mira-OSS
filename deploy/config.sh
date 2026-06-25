# deploy/config.sh
# Interactive configuration gathering for MIRA deployment
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: LOUD_MODE variable set
#
# Sets: CONFIG_*, STATUS_*, OS, DISTRO

# Initialize configuration state (using simple variables for Bash 3.x compatibility)
CONFIG_ANTHROPIC_KEY=""
CONFIG_ANTHROPIC_BATCH_KEY=""
CONFIG_KAGI_KEY=""
CONFIG_DB_PASSWORD=""
CONFIG_INSTALL_PLAYWRIGHT=""
CONFIG_INSTALL_SYSTEMD=""
CONFIG_START_MIRA_NOW=""
CONFIG_OFFLINE_MODE=""
CONFIG_LOCAL_MODEL_CHOICE=""       # auto / custom
CONFIG_LLAMA_MAIN_MODEL="Qwopus3.6-27B-v2-MTP-Q5_K_M.gguf"
CONFIG_LLAMA_SMALL_MODEL="Qwen3.5-9B-UD-Q3_K_XL.gguf"
CONFIG_CHAT_PROVIDER_TYPE=""
CONFIG_CHAT_ENDPOINT=""
CONFIG_CHAT_API_KEY=""
CONFIG_CHAT_MODEL=""
CONFIG_SUBCORTICAL_ENDPOINT=""
CONFIG_SUBCORTICAL_API_KEY=""
CONFIG_SUBCORTICAL_MODEL=""
STATUS_CHAT_PROVIDER=""
STATUS_CHAT_KEY=""
STATUS_SUBCORTICAL=""
STATUS_SUBCORTICAL_KEY=""
STATUS_KAGI=""
STATUS_DB_PASSWORD=""
STATUS_PLAYWRIGHT=""
STATUS_SYSTEMD=""
STATUS_MIRA_SERVICE=""

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

# Detect operating system (needed for port stop logic and later steps)
OS_TYPE=$(uname -s)
case "$OS_TYPE" in
    Linux*)
        OS="linux"
        # Detect Linux distribution family
        if [ -f /etc/redhat-release ] || [ -f /etc/fedora-release ]; then
            DISTRO="fedora"
        elif [ -f /etc/debian_version ]; then
            DISTRO="debian"
        else
            # Fall back to checking /etc/os-release
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
                        # Check ID_LIKE for derivatives
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
        echo ""
        print_error "Unsupported operating system: $OS_TYPE"
        print_info "Supported: Linux (Debian/Ubuntu, Fedora/RHEL/CentOS) and macOS"
        print_info "For other platforms, see manual installation: docs/MANUAL_INSTALL.md"
        exit 1
        ;;
esac

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
                    # Fedora/RHEL uses postgresql-17 service name, Debian uses postgresql
                    if check_exists service_systemctl postgresql-17; then
                        stop_service postgresql-17 systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    elif check_exists service_systemctl postgresql; then
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

print_header "LLM Provider Configuration"

echo -e "${BOLD}${BLUE}LLM Provider${RESET}"
read -p "$(echo -e ${CYAN}Use local LLM via llama-server?${RESET}) (y/n, default=n): " USE_LOCAL_LLM_INPUT
if [[ "$USE_LOCAL_LLM_INPUT" =~ ^[Yy](es)?$ ]]; then
    CONFIG_OFFLINE_MODE="yes"
    # Placeholder keys so Vault validation passes
    CONFIG_ANTHROPIC_KEY="OFFLINE_MODE_PLACEHOLDER"
    CONFIG_ANTHROPIC_BATCH_KEY="OFFLINE_MODE_PLACEHOLDER"
    STATUS_CHAT_PROVIDER="${CHECKMARK} Local llama-server"
    STATUS_CHAT_KEY="${DIM}N/A (local)${RESET}"
    STATUS_SUBCORTICAL="${DIM}N/A (local)${RESET}"
    STATUS_SUBCORTICAL_KEY="${DIM}N/A (local)${RESET}"

    echo ""
    echo -e "${BOLD}Local Model Configuration${RESET}"
    echo -e "${DIM}   Two models are needed — a capable main model and a small fast model.${RESET}"
    echo -e "${DIM}   The analysis/subcortical model runs on every message; no gains from${RESET}"
    echo -e "${DIM}   using a larger slow model there — speed is the only thing that matters.${RESET}"
    echo ""
    echo -e "${DIM}   Pre-configured pair (sized for dual RTX 3090 / 48GB VRAM):${RESET}"
    echo -e "${DIM}     Main:      Qwopus3.6-27B-v2-MTP-Q5_K_M.gguf${RESET}"
    echo -e "${DIM}     Small:     Qwen3.5-9B-UD-Q3_K_XL.gguf${RESET}"
    echo ""
    echo -e "${DIM}   If your hardware differs or lacks sufficient VRAM, choose custom${RESET}"
    echo -e "${DIM}   and download your own GGUF model after install completes.${RESET}"
    echo ""
    echo "     1. Pre-configured pair (Qwopus3.6-27B + Qwen3.5-9B)"
    echo "     2. Custom (bring your own GGUF model)"
    read -p "$(echo -e ${CYAN}Model selection${RESET}) [1-2, default=1]: " LOCAL_MODEL_CHOICE
    if [[ "$LOCAL_MODEL_CHOICE" == "2" ]]; then
        CONFIG_LOCAL_MODEL_CHOICE="custom"
        echo ""
        echo -e "${DIM}   Enter the path or URL of your GGUF model file:${RESET}"
        read -p "$(echo -e ${CYAN}Custom GGUF model${RESET}): " CUSTOM_GGUF_INPUT
        if [ -n "$CUSTOM_GGUF_INPUT" ]; then
            CONFIG_CUSTOM_GGUF="$CUSTOM_GGUF_INPUT"
        fi
    else
        CONFIG_LOCAL_MODEL_CHOICE="auto"
    fi
else
    CONFIG_OFFLINE_MODE="no"

    # Chat Provider
    echo -e "${BOLD}${BLUE}1. Chat Provider${RESET}"
    echo -e "${DIM}   Pick your main chat provider:${RESET}"
    echo "     1. Anthropic (default)"
    echo "     2. Other (OpenAI-compatible endpoint)"
    read -p "$(echo -e ${CYAN}Select provider${RESET}) [1-2, default=1]: " CHAT_PROVIDER_CHOICE

    case "${CHAT_PROVIDER_CHOICE:-1}" in
        2)
            CONFIG_CHAT_PROVIDER_TYPE="generic"

            # Generic endpoint URL
            echo ""
            read -p "$(echo -e ${CYAN}Endpoint URL${RESET}) [default: https://openrouter.ai/api/v1/chat/completions]: " CHAT_ENDPOINT_INPUT
            CONFIG_CHAT_ENDPOINT="${CHAT_ENDPOINT_INPUT:-https://openrouter.ai/api/v1/chat/completions}"

            # API key
            echo -e "${BOLD}${BLUE}   Chat API Key${RESET}"
            while true; do
                read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " CHAT_KEY_INPUT
                if [ -z "$CHAT_KEY_INPUT" ]; then
                    CONFIG_CHAT_API_KEY="PLACEHOLDER_SET_THIS_LATER"
                    STATUS_CHAT_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
                    break
                fi
                CONFIG_CHAT_API_KEY="$CHAT_KEY_INPUT"
                STATUS_CHAT_KEY="${CHECKMARK} Configured"
                break
            done

            # Model
            read -p "$(echo -e ${CYAN}Model name${RESET}) [default: qwen/qwen3.5-397b-a17b]: " CHAT_MODEL_INPUT
            CONFIG_CHAT_MODEL="${CHAT_MODEL_INPUT:-qwen/qwen3.5-397b-a17b}"

            # Anthropic placeholders (background tasks won't work without real keys)
            CONFIG_ANTHROPIC_KEY="PLACEHOLDER_NOT_CONFIGURED"
            CONFIG_ANTHROPIC_BATCH_KEY="PLACEHOLDER_NOT_CONFIGURED"

            STATUS_CHAT_PROVIDER="${CHECKMARK} Generic (${CONFIG_CHAT_ENDPOINT})"

            ;;
        *)
            CONFIG_CHAT_PROVIDER_TYPE="anthropic"

            # Anthropic API Key
            echo ""
            echo -e "${BOLD}${BLUE}   Anthropic API Key${RESET} ${DIM}(console.anthropic.com/settings/keys)${RESET}"
            while true; do
                read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " ANTHROPIC_KEY_INPUT
                if [ -z "$ANTHROPIC_KEY_INPUT" ]; then
                    CONFIG_ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
                    CONFIG_CHAT_API_KEY=""
                    STATUS_CHAT_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
                    break
                fi
                if [[ $ANTHROPIC_KEY_INPUT =~ ^sk-ant- ]]; then
                    CONFIG_ANTHROPIC_KEY="$ANTHROPIC_KEY_INPUT"
                    CONFIG_CHAT_API_KEY="$ANTHROPIC_KEY_INPUT"
                    STATUS_CHAT_KEY="${CHECKMARK} Configured"
                    break
                else
                    print_warning "This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')"
                    read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, n=exit, t=try again): " CONFIRM
                    if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
                        CONFIG_ANTHROPIC_KEY="$ANTHROPIC_KEY_INPUT"
                        CONFIG_CHAT_API_KEY="$ANTHROPIC_KEY_INPUT"
                        STATUS_CHAT_KEY="${CHECKMARK} Configured (unvalidated)"
                        break
                    elif [[ "$CONFIRM" =~ ^[Tt](ry)?$ ]]; then
                        continue
                    else
                        CONFIG_ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
                        CONFIG_CHAT_API_KEY=""
                        STATUS_CHAT_KEY="${WARNING} NOT SET"
                        break
                    fi
                fi
            done

            # Model (default: claude-opus-4-6)
            read -p "$(echo -e ${CYAN}Model${RESET}) [default: claude-opus-4-6]: " CHAT_MODEL_INPUT
            CONFIG_CHAT_MODEL="${CHAT_MODEL_INPUT:-claude-opus-4-6}"

            # Batch API Key (optional)
            echo -e "${BOLD}${BLUE}   Batch API Key${RESET} ${DIM}(OPTIONAL - separate key for batch operations)${RESET}"
            echo -e "${DIM}    Leave blank to use the same key. Separate keys allow independent rate limits.${RESET}"
            while true; do
                read -p "$(echo -e ${CYAN}Enter batch key${RESET}) (or Enter to use main key): " ANTHROPIC_BATCH_KEY_INPUT
                if [ -z "$ANTHROPIC_BATCH_KEY_INPUT" ]; then
                    CONFIG_ANTHROPIC_BATCH_KEY="$CONFIG_ANTHROPIC_KEY"
                    break
                fi
                if [[ $ANTHROPIC_BATCH_KEY_INPUT =~ ^sk-ant- ]]; then
                    CONFIG_ANTHROPIC_BATCH_KEY="$ANTHROPIC_BATCH_KEY_INPUT"
                    break
                else
                    print_warning "This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')"
                    read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, n=use main key, t=try again): " CONFIRM
                    if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
                        CONFIG_ANTHROPIC_BATCH_KEY="$ANTHROPIC_BATCH_KEY_INPUT"
                        break
                    elif [[ "$CONFIRM" =~ ^[Tt](ry)?$ ]]; then
                        continue
                    else
                        CONFIG_ANTHROPIC_BATCH_KEY="$CONFIG_ANTHROPIC_KEY"
                        break
                    fi
                fi
            done

            STATUS_CHAT_PROVIDER="${CHECKMARK} Anthropic"
            ;;
    esac

    # Subcortical
    echo -e "${BOLD}${BLUE}2. Subcortical${RESET}"
    echo -e "${DIM}   Runs on every message for memory retrieval (query expansion, entity extraction).${RESET}"
    echo -e "${DIM}   A fast inference provider like Groq works best here.${RESET}"
    echo ""
    read -p "$(echo -e ${CYAN}Endpoint URL${RESET}) [default: https://api.groq.com/openai/v1/chat/completions]: " SUBCORTICAL_ENDPOINT_INPUT
    CONFIG_SUBCORTICAL_ENDPOINT="${SUBCORTICAL_ENDPOINT_INPUT:-https://api.groq.com/openai/v1/chat/completions}"
    STATUS_SUBCORTICAL="${CHECKMARK} ${CONFIG_SUBCORTICAL_ENDPOINT}"

    # Subcortical API Key
    echo -e "${BOLD}${BLUE}   Subcortical API Key${RESET}"
    while true; do
        read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " SUBCORTICAL_KEY_INPUT
        if [ -z "$SUBCORTICAL_KEY_INPUT" ]; then
            CONFIG_SUBCORTICAL_API_KEY="PLACEHOLDER_SET_THIS_LATER"
            STATUS_SUBCORTICAL_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
            break
        fi
        # Validate gsk_ prefix if Groq endpoint detected
        if [[ "$CONFIG_SUBCORTICAL_ENDPOINT" == *"groq.com"* ]] && [[ ! $SUBCORTICAL_KEY_INPUT =~ ^gsk_ ]]; then
            print_warning "This doesn't look like a valid Groq API key (should start with 'gsk_')"
            read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, t=try again): " CONFIRM
            if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
                CONFIG_SUBCORTICAL_API_KEY="$SUBCORTICAL_KEY_INPUT"
                STATUS_SUBCORTICAL_KEY="${CHECKMARK} Configured (unvalidated)"
                break
            elif [[ "$CONFIRM" =~ ^[Tt](ry)?$ ]]; then
                continue
            fi
        else
            CONFIG_SUBCORTICAL_API_KEY="$SUBCORTICAL_KEY_INPUT"
            STATUS_SUBCORTICAL_KEY="${CHECKMARK} Configured"
            break
        fi
    done

    # Subcortical Model
    read -p "$(echo -e ${CYAN}Model${RESET}) [default: qwen/qwen3-32b]: " SUBCORTICAL_MODEL_INPUT
    CONFIG_SUBCORTICAL_MODEL="${SUBCORTICAL_MODEL_INPUT:-qwen/qwen3-32b}"
fi

# Kagi Search API Key (optional — works with any provider)
echo -e "${BOLD}${BLUE}3. Kagi Search API Key${RESET} ${DIM}(OPTIONAL - kagi.com/settings?p=api)${RESET}"
read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " KAGI_KEY_INPUT
if [ -z "$KAGI_KEY_INPUT" ]; then
    CONFIG_KAGI_KEY=""
    STATUS_KAGI="${DIM}Skipped${RESET}"
else
    CONFIG_KAGI_KEY="$KAGI_KEY_INPUT"
    STATUS_KAGI="${CHECKMARK} Configured"
fi

# Database Password (optional - defaults to changethisifdeployingpwd)
echo -e "${BOLD}${BLUE}4. Database Password${RESET} ${DIM}(OPTIONAL - default: changethisifdeployingpwd)${RESET}"
read -p "$(echo -e ${CYAN}Enter password${RESET}) (or Enter for default): " DB_PASSWORD_INPUT
if [ -z "$DB_PASSWORD_INPUT" ]; then
    CONFIG_DB_PASSWORD="changethisifdeployingpwd"
    STATUS_DB_PASSWORD="${DIM}Using default password${RESET}"
else
    CONFIG_DB_PASSWORD="$DB_PASSWORD_INPUT"
    STATUS_DB_PASSWORD="${CHECKMARK} Custom password set"
fi

# Playwright Browser Installation (optional)
echo -e "${BOLD}${BLUE}5. Playwright Browser${RESET} ${DIM}(OPTIONAL - for JS-heavy webpage extraction)${RESET}"
read -p "$(echo -e ${CYAN}Install Playwright?${RESET}) (y/n, default=y): " PLAYWRIGHT_INPUT
# Default to yes if user just presses Enter
if [ -z "$PLAYWRIGHT_INPUT" ]; then
    PLAYWRIGHT_INPUT="y"
fi
if [[ "$PLAYWRIGHT_INPUT" =~ ^[Yy](es)?$ ]]; then
    CONFIG_INSTALL_PLAYWRIGHT="yes"
    STATUS_PLAYWRIGHT="${CHECKMARK} Will be installed"
else
    CONFIG_INSTALL_PLAYWRIGHT="no"
    STATUS_PLAYWRIGHT="${YELLOW}Skipped${RESET}"
fi

# Systemd service option (Linux only)
echo -e "${BOLD}${BLUE}6. Systemd Service${RESET} ${DIM}(OPTIONAL - Linux only, auto-start on boot)${RESET}"
if [ "$OS" = "linux" ]; then
    read -p "$(echo -e ${CYAN}Install as systemd service?${RESET}) (y/n): " SYSTEMD_INPUT
    if [[ "$SYSTEMD_INPUT" =~ ^[Yy](es)?$ ]]; then
        CONFIG_INSTALL_SYSTEMD="yes"
        read -p "$(echo -e ${CYAN}Start MIRA now?${RESET}) (y/n): " START_NOW_INPUT
        if [[ "$START_NOW_INPUT" =~ ^[Yy](es)?$ ]]; then
            CONFIG_START_MIRA_NOW="yes"
            STATUS_SYSTEMD="${CHECKMARK} Will be installed and started"
        else
            CONFIG_START_MIRA_NOW="no"
            STATUS_SYSTEMD="${CHECKMARK} Will be installed (not started)"
        fi
    else
        CONFIG_INSTALL_SYSTEMD="no"
        CONFIG_START_MIRA_NOW="no"
        STATUS_SYSTEMD="${DIM}Skipped${RESET}"
    fi
elif [ "$OS" = "macos" ]; then
    CONFIG_INSTALL_SYSTEMD="no"
    CONFIG_START_MIRA_NOW="no"
    STATUS_SYSTEMD="${DIM}N/A (macOS)${RESET}"
fi

echo ""
echo -e "${BOLD}Configuration Summary:${RESET}"
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    if [ "$CONFIG_LOCAL_MODEL_CHOICE" = "auto" ]; then
        echo -e "  LLM Provider:    ${CYAN}Local llama-server${RESET}"
        echo -e "  Main Model:      ${CYAN}Qwopus3.6-27B-v2-MTP-Q5_K_M${RESET}"
        echo -e "  Small Model:     ${CYAN}Qwen3.5-9B-UD-Q3_K_XL${RESET}"
    else
        echo -e "  LLM Provider:    ${CYAN}Local llama-server (custom)${RESET}"
        echo -e "  Model:           ${CYAN}${CONFIG_CUSTOM_GGUF:-TBD}${RESET}"
    fi
else
    echo -e "  Chat Provider:   ${STATUS_CHAT_PROVIDER}"
    echo -e "  Chat Model:      ${CYAN}${CONFIG_CHAT_MODEL}${RESET}"
    echo -e "  Chat Key:        ${STATUS_CHAT_KEY}"
    echo -e "  Subcortical:     ${STATUS_SUBCORTICAL}"
    echo -e "  Subcortical Key: ${STATUS_SUBCORTICAL_KEY}"
    echo -e "  Subcortical Mdl: ${CYAN}${CONFIG_SUBCORTICAL_MODEL}${RESET}"
fi
echo -e "  Kagi:            ${STATUS_KAGI}"
echo -e "  DB Password:     ${STATUS_DB_PASSWORD}"
echo -e "  Playwright:      ${STATUS_PLAYWRIGHT}"
echo -e "  Systemd Service: ${STATUS_SYSTEMD}"
echo ""
