#!/bin/bash
# MIRA Container Interactive Setup
# Adapted from deploy/config.sh for Docker container context
#
# This script is sourced (not executed) by init-mira.sh when running
# interactively with a TTY attached.
#
# Differences from host deploy/config.sh:
# - No disk space check (container handles this)
# - No port availability check (container manages ports)
# - No systemd/service management options
# - No Playwright install prompt (pre-installed in image)
# - No existing installation check (clean container)

# Source helper libraries
source /opt/mira/app/deploy/lib/output.sh
source /opt/mira/app/deploy/lib/services.sh

LOUD_MODE=false

# Initialize configuration state
CONFIG_ANTHROPIC_KEY=""
CONFIG_ANTHROPIC_BATCH_KEY=""
CONFIG_PROVIDER_KEY=""
CONFIG_KAGI_KEY=""
CONFIG_DB_PASSWORD=""
CONFIG_OFFLINE_MODE=""
CONFIG_PROVIDER_NAME=""
CONFIG_PROVIDER_ENDPOINT=""
CONFIG_PROVIDER_KEY_PREFIX=""
CONFIG_PROVIDER_MODEL=""

# Container-specific settings
OS="linux"
DISTRO="debian"

clear
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════╗"
echo "║   MIRA First-Boot Configuration        ║"
echo "╚════════════════════════════════════════╝"
echo -e "${RESET}"
echo ""

print_header "API Key Configuration"

# Note: Offline mode not supported in standard Docker image
# (would require llama.cpp bundled or external llama-server)
CONFIG_OFFLINE_MODE="no"

# Anthropic API Key (required)
echo -e "${BOLD}${BLUE}1. Anthropic API Key${RESET} ${DIM}(REQUIRED - console.anthropic.com/settings/keys)${RESET}"
while true; do
    read -p "$(echo -e ${CYAN}Enter key${RESET}): " ANTHROPIC_KEY_INPUT
    if [ -z "$ANTHROPIC_KEY_INPUT" ]; then
        print_warning "Anthropic API key is required for MIRA to function."
        continue
    fi
    # Basic validation
    if [[ $ANTHROPIC_KEY_INPUT =~ ^sk-ant- ]]; then
        CONFIG_ANTHROPIC_KEY="$ANTHROPIC_KEY_INPUT"
        print_success "Anthropic key configured"
        break
    else
        print_warning "This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')"
        read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, t=try again): " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
            CONFIG_ANTHROPIC_KEY="$ANTHROPIC_KEY_INPUT"
            print_success "Anthropic key configured (unvalidated)"
            break
        fi
    fi
done

# Anthropic Batch API Key (optional)
echo -e "${BOLD}${BLUE}1b. Anthropic Batch API Key${RESET} ${DIM}(OPTIONAL - separate key for batch operations)${RESET}"
echo -e "${DIM}    Leave blank to use the same key as above.${RESET}"
read -p "$(echo -e ${CYAN}Enter batch key${RESET}) (or Enter to use main key): " ANTHROPIC_BATCH_KEY_INPUT
if [ -z "$ANTHROPIC_BATCH_KEY_INPUT" ]; then
    CONFIG_ANTHROPIC_BATCH_KEY="$CONFIG_ANTHROPIC_KEY"
    print_info "Using main Anthropic key for batch operations"
else
    CONFIG_ANTHROPIC_BATCH_KEY="$ANTHROPIC_BATCH_KEY_INPUT"
    print_success "Separate batch key configured"
fi

# Generic Provider Selection
echo -e "${BOLD}${BLUE}2. Generic Provider${RESET} ${DIM}(for fast inference - OpenAI-compatible)${RESET}"
echo ""
echo -e "${DIM}   Select your preferred provider:${RESET}"
echo "     1. Groq (default, recommended for speed)"
echo "     2. OpenRouter"
echo "     3. Together AI"
echo "     4. Fireworks AI"
echo "     5. Cerebras"
echo "     6. SambaNova"
echo "     7. Other (custom endpoint)"
read -p "$(echo -e ${CYAN}Select provider${RESET}) [1-7, default=1]: " PROVIDER_CHOICE

case "${PROVIDER_CHOICE:-1}" in
    1)
        CONFIG_PROVIDER_NAME="Groq"
        CONFIG_PROVIDER_ENDPOINT="https://api.groq.com/openai/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX="gsk_"
        ;;
    2)
        CONFIG_PROVIDER_NAME="OpenRouter"
        CONFIG_PROVIDER_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX="sk-or-"
        ;;
    3)
        CONFIG_PROVIDER_NAME="Together AI"
        CONFIG_PROVIDER_ENDPOINT="https://api.together.xyz/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    4)
        CONFIG_PROVIDER_NAME="Fireworks AI"
        CONFIG_PROVIDER_ENDPOINT="https://api.fireworks.ai/inference/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    5)
        CONFIG_PROVIDER_NAME="Cerebras"
        CONFIG_PROVIDER_ENDPOINT="https://api.cerebras.ai/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    6)
        CONFIG_PROVIDER_NAME="SambaNova"
        CONFIG_PROVIDER_ENDPOINT="https://api.sambanova.ai/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    7)
        CONFIG_PROVIDER_NAME="Custom"
        read -p "$(echo -e ${CYAN}Enter custom endpoint URL${RESET}): " CONFIG_PROVIDER_ENDPOINT
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    *)
        CONFIG_PROVIDER_NAME="Groq"
        CONFIG_PROVIDER_ENDPOINT="https://api.groq.com/openai/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX="gsk_"
        ;;
esac

print_success "Provider: $CONFIG_PROVIDER_NAME"

# For non-Groq providers, prompt for model name
if [ "$CONFIG_PROVIDER_NAME" != "Groq" ]; then
    echo ""
    print_info "MIRA needs a model name compatible with ${CONFIG_PROVIDER_NAME}."
    case "$CONFIG_PROVIDER_NAME" in
        "OpenRouter")
            print_info "Example: meta-llama/llama-3.3-70b-instruct:free"
            DEFAULT_MODEL="meta-llama/llama-3.3-70b-instruct:free"
            ;;
        "Together AI")
            print_info "Example: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            DEFAULT_MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            ;;
        "Fireworks AI")
            print_info "Example: accounts/fireworks/models/llama-v3p1-70b-instruct"
            DEFAULT_MODEL="accounts/fireworks/models/llama-v3p1-70b-instruct"
            ;;
        "Cerebras")
            print_info "Example: llama-3.3-70b"
            DEFAULT_MODEL="llama-3.3-70b"
            ;;
        "SambaNova")
            print_info "Example: Meta-Llama-3.1-70B-Instruct"
            DEFAULT_MODEL="Meta-Llama-3.1-70B-Instruct"
            ;;
        *)
            DEFAULT_MODEL=""
            ;;
    esac
    if [ -n "$DEFAULT_MODEL" ]; then
        read -p "$(echo -e ${CYAN}Model name${RESET}) [default: ${DEFAULT_MODEL}]: " MODEL_INPUT
        CONFIG_PROVIDER_MODEL="${MODEL_INPUT:-$DEFAULT_MODEL}"
    else
        read -p "$(echo -e ${CYAN}Model name${RESET}): " CONFIG_PROVIDER_MODEL
    fi
fi

# Generic Provider API Key (required)
echo -e "${BOLD}${BLUE}2b. ${CONFIG_PROVIDER_NAME} API Key${RESET} ${DIM}(REQUIRED)${RESET}"
while true; do
    read -p "$(echo -e ${CYAN}Enter key${RESET}): " PROVIDER_KEY_INPUT
    if [ -z "$PROVIDER_KEY_INPUT" ]; then
        print_warning "Provider API key is required for internal LLM operations."
        continue
    fi
    # Validate key prefix if provider has one
    if [ -n "$CONFIG_PROVIDER_KEY_PREFIX" ]; then
        if [[ $PROVIDER_KEY_INPUT =~ ^${CONFIG_PROVIDER_KEY_PREFIX} ]]; then
            CONFIG_PROVIDER_KEY="$PROVIDER_KEY_INPUT"
            print_success "Provider key configured"
            break
        else
            print_warning "This doesn't look like a valid ${CONFIG_PROVIDER_NAME} API key"
            read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, t=try again): " CONFIRM
            if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
                CONFIG_PROVIDER_KEY="$PROVIDER_KEY_INPUT"
                print_success "Provider key configured (unvalidated)"
                break
            fi
        fi
    else
        CONFIG_PROVIDER_KEY="$PROVIDER_KEY_INPUT"
        print_success "Provider key configured"
        break
    fi
done

# Kagi API Key (optional)
echo -e "${BOLD}${BLUE}3. Kagi Search API Key${RESET} ${DIM}(OPTIONAL - kagi.com/settings?p=api)${RESET}"
read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " KAGI_KEY_INPUT
if [ -z "$KAGI_KEY_INPUT" ]; then
    CONFIG_KAGI_KEY=""
    print_info "Kagi search skipped (web search will be limited)"
else
    CONFIG_KAGI_KEY="$KAGI_KEY_INPUT"
    print_success "Kagi key configured"
fi

# Database Password (optional)
echo -e "${BOLD}${BLUE}4. Database Password${RESET} ${DIM}(OPTIONAL - for internal PostgreSQL)${RESET}"
read -p "$(echo -e ${CYAN}Enter password${RESET}) (or Enter for default): " DB_PASSWORD_INPUT
if [ -z "$DB_PASSWORD_INPUT" ]; then
    CONFIG_DB_PASSWORD="changethisifdeployingpwd"
    print_info "Using default database password"
else
    CONFIG_DB_PASSWORD="$DB_PASSWORD_INPUT"
    print_success "Custom database password set"
fi

# Configuration Summary
echo ""
print_header "Configuration Summary"
echo -e "  Anthropic Key:   ${GREEN}****${CONFIG_ANTHROPIC_KEY: -4}${RESET}"
if [ "$CONFIG_ANTHROPIC_BATCH_KEY" = "$CONFIG_ANTHROPIC_KEY" ]; then
    echo -e "  Batch Key:       ${DIM}Using main key${RESET}"
else
    echo -e "  Batch Key:       ${GREEN}****${CONFIG_ANTHROPIC_BATCH_KEY: -4}${RESET}"
fi
echo -e "  Provider:        ${CYAN}$CONFIG_PROVIDER_NAME${RESET}"
echo -e "  Provider Key:    ${GREEN}****${CONFIG_PROVIDER_KEY: -4}${RESET}"
if [ -n "$CONFIG_PROVIDER_MODEL" ]; then
    echo -e "  Provider Model:  ${CYAN}$CONFIG_PROVIDER_MODEL${RESET}"
fi
if [ -n "$CONFIG_KAGI_KEY" ]; then
    echo -e "  Kagi Key:        ${GREEN}Configured${RESET}"
else
    echo -e "  Kagi Key:        ${DIM}Skipped${RESET}"
fi
echo ""

read -p "$(echo -e ${CYAN}Proceed with this configuration?${RESET}) (y/n): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
    print_error "Configuration cancelled. Restarting setup..."
    exec /opt/mira/container-setup.sh
fi

# Export configuration for init-mira.sh to use
export CONFIG_ANTHROPIC_KEY
export CONFIG_ANTHROPIC_BATCH_KEY
export CONFIG_PROVIDER_KEY
export CONFIG_PROVIDER_NAME
export CONFIG_PROVIDER_ENDPOINT
export CONFIG_PROVIDER_MODEL
export CONFIG_KAGI_KEY
export CONFIG_DB_PASSWORD
export CONFIG_OFFLINE_MODE

print_success "Configuration complete. Proceeding with initialization..."
