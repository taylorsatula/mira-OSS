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
# DEPLOYMENT START
# ============================================================================

clear
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════╗"
echo "║   MIRA Deployment Script v0.94         ║"
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
    read -p "$(echo -e ${YELLOW}This will OVERWRITE the existing installation. Continue? ${RESET})(yes/no): " OVERWRITE
    if [ "$OVERWRITE" != "yes" ]; then
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
    read -p "$(echo -e ${YELLOW}Stop existing services and continue?${RESET}) (yes/no): " CONTINUE
    if [ "$CONTINUE" != "yes" ]; then
        print_info "Installation cancelled. Free up the required ports and try again."
        exit 0
    fi
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
read -p "$(echo -e ${CYAN}Enter your Anthropic API key${RESET}) (or press Enter to skip): " ANTHROPIC_KEY
if [ -z "$ANTHROPIC_KEY" ]; then
    ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
    ANTHROPIC_STATUS="${WARNING} NOT SET - You must configure this before using MIRA"
else
    # Basic validation - check if it looks like an Anthropic key
    if [[ $ANTHROPIC_KEY =~ ^sk-ant- ]]; then
        ANTHROPIC_STATUS="${CHECKMARK} Configured"
    else
        print_warning "This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')"
        read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y/n): " CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
            ANTHROPIC_STATUS="${WARNING} NOT SET"
        else
            ANTHROPIC_STATUS="${CHECKMARK} Configured (unvalidated)"
        fi
    fi
fi
echo ""

# Groq API Key (required)
echo -e "${BOLD}${BLUE}2. Groq API Key${RESET} ${DIM}(REQUIRED)${RESET}"
print_info "Used for: Fast inference and web extraction operations"
print_info "Get your key at: https://console.groq.com/keys"
echo ""
read -p "$(echo -e ${CYAN}Enter your Groq API key${RESET}) (or press Enter to skip): " GROQ_KEY
if [ -z "$GROQ_KEY" ]; then
    GROQ_KEY="PLACEHOLDER_SET_THIS_LATER"
    GROQ_STATUS="${WARNING} NOT SET - You must configure this before using MIRA"
else
    # Basic validation - check if it looks like a Groq key
    if [[ $GROQ_KEY =~ ^gsk_ ]]; then
        GROQ_STATUS="${CHECKMARK} Configured"
    else
        print_warning "This doesn't look like a valid Groq API key (should start with 'gsk_')"
        read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y/n): " CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            GROQ_KEY="PLACEHOLDER_SET_THIS_LATER"
            GROQ_STATUS="${WARNING} NOT SET"
        else
            GROQ_STATUS="${CHECKMARK} Configured (unvalidated)"
        fi
    fi
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
echo -e "${BOLD}${BLUE}3. Systemd Service${RESET} ${DIM}(OPTIONAL - Linux Only)${RESET}"
if [ "$OS" = "linux" ]; then
    print_info "Configure MIRA to start automatically on system boot?"
    print_info "This creates a systemd service that starts MIRA when the system boots."
    echo ""
    read -p "$(echo -e ${CYAN}Install MIRA as systemd service?${RESET}) (yes/no): " INSTALL_SYSTEMD
    if [ "$INSTALL_SYSTEMD" = "yes" ]; then
        echo ""
        read -p "$(echo -e ${CYAN}Start MIRA service immediately after installation?${RESET}) (yes/no): " START_MIRA_NOW
        if [ "$START_MIRA_NOW" = "yes" ]; then
            SYSTEMD_STATUS="${CHECKMARK} Will be installed and started"
        else
            START_MIRA_NOW="no"
            SYSTEMD_STATUS="${CHECKMARK} Will be installed (not started)"
        fi
    else
        INSTALL_SYSTEMD="no"
        START_MIRA_NOW="no"
        SYSTEMD_STATUS="${RED}Skipped${RESET}"
    fi
elif [ "$OS" = "macos" ]; then
    INSTALL_SYSTEMD="no"
    START_MIRA_NOW="no"
    print_info "Systemd service creation only available on Linux (macOS uses launchd)"
    SYSTEMD_STATUS="${DIM}Not available on macOS${RESET}"
fi
echo ""

echo -e "${BOLD}Configuration Summary:${RESET}"
echo -e "  Anthropic:       $ANTHROPIC_STATUS"
echo -e "  Groq:            $GROQ_STATUS"
echo -e "  Systemd Service: $SYSTEMD_STATUS"
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
    # Ubuntu/Debian package installation
    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating package lists..."
        sudo apt-get update
        print_step "Installing system packages..."
        sudo apt-get install -y \
            build-essential \
            python3.13-venv \
            python3.13-dev \
            libpq-dev \
            postgresql-server-dev-17 \
            unzip \
            wget \
            curl \
            postgresql \
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
            build-essential python3.13-venv python3.13-dev libpq-dev \
            postgresql-server-dev-17 unzip wget curl postgresql \
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

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating Homebrew..."
        brew update
        print_step "Installing dependencies via Homebrew..."
        brew install python@3.13 wget curl postgresql@17 valkey vault
    else
        (brew update > /dev/null 2>&1) &
        show_progress $! "Updating Homebrew"

        (brew install python@3.13 wget curl postgresql@17 valkey vault > /dev/null 2>&1) &
        show_progress $! "Installing dependencies via Homebrew (6 packages)"
    fi

    print_info "Playwright will install its own browser dependencies"
fi

print_success "System dependencies installed"

print_header "Step 2: Python Verification"

echo -ne "${DIM}${ARROW}${RESET} Locating Python 3.13... "
if [ "$OS" = "linux" ]; then
    # Check if python3.13 is available
    if ! command -v python3.13 &> /dev/null; then
        echo -e "${ERROR}"
        print_error "Python 3.13 not found after installation. Check package availability."
        exit 1
    fi
    PYTHON_CMD="python3.13"
elif [ "$OS" = "macos" ]; then
    # On macOS, Homebrew python@3.13 might be at different paths
    if command -v python3.13 &> /dev/null; then
        PYTHON_CMD="python3.13"
    elif [ -f "/opt/homebrew/opt/python@3.13/bin/python3.13" ]; then
        PYTHON_CMD="/opt/homebrew/opt/python@3.13/bin/python3.13"
    elif [ -f "/usr/local/opt/python@3.13/bin/python3.13" ]; then
        PYTHON_CMD="/usr/local/opt/python@3.13/bin/python3.13"
    else
        echo -e "${ERROR}"
        print_error "Python 3.13 not found. Check Homebrew installation."
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
run_with_status "Downloading MIRA v0.94" \
    wget -q -O mira-0.94.tar.gz https://github.com/taylorsatula/mira-OSS/archive/refs/tags/0.94.tar.gz

run_with_status "Creating /opt/mira/app directory" \
    sudo mkdir -p /opt/mira/app

run_with_status "Extracting archive" \
    tar -xzf mira-0.94.tar.gz -C /tmp

run_with_status "Copying files to /opt/mira/app" \
    sudo cp -r /tmp/mira-OSS-0.94/* /opt/mira/app/

run_with_status "Setting ownership to $MIRA_USER:$MIRA_GROUP" \
    sudo chown -R $MIRA_USER:$MIRA_GROUP /opt/mira

# Clean up immediately after copying
run_quiet rm -f /tmp/mira-0.94.tar.gz
run_quiet rm -rf /tmp/mira-OSS-0.94

print_success "MIRA installed to /opt/mira/app"

print_header "Step 4: Python Environment Setup"

cd /opt/mira/app

run_with_status "Creating virtual environment" \
    $PYTHON_CMD -m venv venv

run_with_status "Initializing pip" \
    venv/bin/python3 -m ensurepip

echo -ne "${DIM}${ARROW}${RESET} Checking PyTorch installation... "
if venv/bin/pip3 show torch &> /dev/null; then
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

# Verify critical packages installed successfully
echo -ne "${DIM}${ARROW}${RESET} Verifying sentence-transformers installation... "
if ! venv/bin/python3 -c "import sentence_transformers" 2>/dev/null; then
    echo -e "${ERROR}"
    print_error "sentence-transformers package not found after installation"
    print_info "Try running: venv/bin/pip3 install sentence-transformers"
    exit 1
fi
echo -e "${CHECKMARK}"

if [ "$LOUD_MODE" = true ]; then
    print_step "Installing spacy language model..."
    venv/bin/python3 -m spacy download en_core_web_lg
else
    (venv/bin/python3 -m spacy download en_core_web_lg > /dev/null 2>&1) &
    show_progress $! "Installing spacy language model"
fi

print_success "Python dependencies installed"

print_header "Step 6: AI Model Downloads"

if [ "$LOUD_MODE" = true ]; then
    print_step "Downloading embedding and reranker models..."
    venv/bin/python3 << 'EOF'
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

cache_dir = os.path.expanduser("~/.cache/huggingface")

# Check if all-MiniLM-L6-v2 is already cached
minilm_path = Path(cache_dir) / "sentence-transformers_all-MiniLM-L6-v2"
if minilm_path.exists():
    print("✓ all-MiniLM-L6-v2 already cached, skipping")
else:
    print("→ Downloading all-MiniLM-L6-v2...")
    fast_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=cache_dir
    )
    print("✓ all-MiniLM-L6-v2 downloaded")

# Check if BGE reranker is already cached
reranker_path = Path(cache_dir) / "BAAI_bge-reranker-base"
if reranker_path.exists():
    print("✓ BGE reranker already cached, skipping")
else:
    print("→ Downloading BAAI/bge-reranker-base...")
    reranker_model = SentenceTransformer(
        "BAAI/bge-reranker-base",
        cache_folder=cache_dir
    )
    print("✓ BGE reranker downloaded")
EOF
else
    (venv/bin/python3 << 'EOF'
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

cache_dir = os.path.expanduser("~/.cache/huggingface")

minilm_path = Path(cache_dir) / "sentence-transformers_all-MiniLM-L6-v2"
if not minilm_path.exists():
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)

reranker_path = Path(cache_dir) / "BAAI_bge-reranker-base"
if not reranker_path.exists():
    SentenceTransformer("BAAI/bge-reranker-base", cache_folder=cache_dir)
EOF
) &
    show_progress $! "Downloading embedding models"
fi

print_header "Step 7: Playwright Browser Setup"

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

if [ "$OS" = "linux" ]; then
    run_with_status "Installing Playwright system dependencies" \
        sudo venv/bin/playwright install-deps || true
elif [ "$OS" = "macos" ]; then
    print_info "Playwright browser dependencies are bundled on macOS"
fi

print_success "Playwright configured"

print_header "Step 8: HashiCorp Vault Setup"

if [ "$OS" = "linux" ]; then
    cd /tmp
    run_with_status "Downloading Vault 1.18.3" \
        wget -q https://releases.hashicorp.com/vault/1.18.3/vault_1.18.3_linux_amd64.zip

    run_with_status "Extracting Vault binary" \
        unzip -o vault_1.18.3_linux_amd64.zip

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

    run_with_status "Starting Vault service" \
        sudo systemctl start vault.service

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

print_header "Step 10: Vault Initialization"

export VAULT_ADDR='http://127.0.0.1:8200'

run_with_status "Initializing Vault" \
    vault operator init -key-shares=1 -key-threshold=1 > /opt/vault/init-keys.txt

run_quiet chmod 600 /opt/vault/init-keys.txt

UNSEAL_KEY=$(grep 'Unseal Key 1' /opt/vault/init-keys.txt | awk '{print $NF}')
run_with_status "Unsealing Vault" \
    vault operator unseal "$UNSEAL_KEY" > /dev/null

ROOT_TOKEN=$(grep 'Initial Root Token' /opt/vault/init-keys.txt | awk '{print $NF}')
run_with_status "Authenticating with root token" \
    vault login "$ROOT_TOKEN" > /dev/null

run_with_status "Enabling KV2 secrets engine" \
    vault secrets enable -version=2 -path=secret kv > /dev/null

run_with_status "Enabling AppRole authentication" \
    vault auth enable approle > /dev/null

echo -ne "${DIM}${ARROW}${RESET} Creating Vault policy... "
cat > /tmp/mira-policy.hcl <<'EOF'
# Allow full access to secret/* path
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/*" {
  capabilities = ["list", "read", "delete"]
}
EOF
echo -e "${CHECKMARK}"

run_with_status "Writing policy to Vault" \
    vault policy write mira-policy /tmp/mira-policy.hcl > /dev/null

run_with_status "Creating AppRole" \
    vault write auth/approle/role/mira policies="mira-policy" token_ttl=1h token_max_ttl=4h > /dev/null

run_with_status "Getting AppRole credentials" \
    vault read auth/approle/role/mira/role-id > /opt/vault/role-id.txt

run_quiet vault write -f auth/approle/role/mira/secret-id > /opt/vault/secret-id.txt

print_success "Vault fully configured"

print_header "Step 11: Auto-Unseal Configuration"

echo -ne "${DIM}${ARROW}${RESET} Creating unseal script... "
cat > /opt/vault/unseal.sh <<'EOF'
#!/bin/bash
export VAULT_ADDR='http://127.0.0.1:8200'
sleep 5
UNSEAL_KEY=$(grep 'Unseal Key 1' /opt/vault/init-keys.txt | awk '{print $NF}')
vault operator unseal "$UNSEAL_KEY"
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

    run_with_status "Starting Valkey" \
        brew services start valkey || true

    run_with_status "Starting PostgreSQL" \
        brew services start postgresql@17 || true

    sleep 2
fi

print_header "Step 13: PostgreSQL Configuration"

if [ "$OS" = "linux" ]; then
    # Linux: use postgres system user
    run_with_status "Creating database 'mira_service'" \
        sudo -u postgres psql -c "CREATE DATABASE mira_service;" || true

    run_with_status "Creating user 'mira_admin'" \
        sudo -u postgres psql -c "CREATE USER mira_admin WITH PASSWORD 'changethisifdeployingpwd' SUPERUSER;" || true

    run_with_status "Creating user 'mira_dbuser'" \
        sudo -u postgres psql -c "CREATE USER mira_dbuser WITH PASSWORD 'changethisifdeployingpwd';" || true

    run_quiet sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_admin;"
    run_quiet sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_dbuser;"

    run_with_status "Enabling pgvector extension" \
        sudo -u postgres psql -d mira_service -c "CREATE EXTENSION IF NOT EXISTS vector;"
elif [ "$OS" = "macos" ]; then
    # macOS: run as current user (services already started above)
    run_with_status "Creating database 'mira_service'" \
        createdb mira_service 2>/dev/null || true

    run_with_status "Creating user 'mira_admin'" \
        psql postgres -c "CREATE USER mira_admin WITH PASSWORD 'changethisifdeployingpwd' SUPERUSER;" || true

    run_with_status "Creating user 'mira_dbuser'" \
        psql postgres -c "CREATE USER mira_dbuser WITH PASSWORD 'changethisifdeployingpwd';" || true

    run_quiet psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_admin;"
    run_quiet psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_dbuser;"

    run_with_status "Enabling pgvector extension" \
        psql -d mira_service -c "CREATE EXTENSION IF NOT EXISTS vector;"
fi

print_success "PostgreSQL configured"

print_header "Step 14: Vault Credential Storage"

run_with_status "Storing API keys" \
    vault kv put secret/mira/api_keys \
        anthropic_key="$ANTHROPIC_KEY" \
        groq_key="$GROQ_KEY" > /dev/null

run_with_status "Storing database credentials" \
    vault kv put secret/mira/database \
        admin_url="postgresql://mira_admin:changethisifdeployingpwd@localhost:5432/mira_service" \
        password="changethisifdeployingpwd" \
        username="mira_dbuser" \
        service_url="postgresql://mira_dbuser:changethisifdeployingpwd@localhost:5432/mira_service" > /dev/null

run_with_status "Storing service URLs" \
    vault kv put secret/mira/services \
        app_url="http://localhost:1993" \
        valkey_url="valkey://localhost:6379" > /dev/null

print_success "All credentials stored in Vault"

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
if [ "$INSTALL_SYSTEMD" = "yes" ] && [ "$OS" = "linux" ]; then
    print_header "Step 16: Systemd Service Configuration"

    # Extract Vault credentials from files
    echo -ne "${DIM}${ARROW}${RESET} Reading Vault credentials... "
    VAULT_ROLE_ID=$(grep 'role_id' /opt/vault/role-id.txt | awk '{print $2}')
    VAULT_SECRET_ID=$(grep 'secret_id' /opt/vault/secret-id.txt | awk '{print $2}')

    if [ -z "$VAULT_ROLE_ID" ] || [ -z "$VAULT_SECRET_ID" ]; then
        echo -e "${ERROR}"
        print_error "Failed to read Vault credentials from /opt/vault/"
        print_info "Skipping systemd service creation"
        INSTALL_SYSTEMD="failed"
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
        if [ "$START_MIRA_NOW" = "yes" ]; then
            echo ""
            run_with_status "Starting MIRA service" \
                sudo systemctl start mira.service

            # Give service a moment to start
            sleep 2

            # Check if service started successfully
            if sudo systemctl is-active --quiet mira.service; then
                print_success "MIRA service is running"
                print_info "View logs: journalctl -u mira -f"
                MIRA_STARTED="yes"
            else
                print_warning "MIRA service may have failed to start"
                print_info "Check status: systemctl status mira"
                print_info "View logs: journalctl -u mira -n 50"
                MIRA_STARTED="failed"
            fi
        else
            print_info "To start later: sudo systemctl start mira"
            print_info "To view logs: journalctl -u mira -f"
            MIRA_STARTED="no"
        fi
    fi
elif [ "$INSTALL_SYSTEMD" = "no" ]; then
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
    run_quiet rm -f /tmp/vault_1.18.3_linux_amd64.zip
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
echo -e "  Anthropic: $ANTHROPIC_STATUS"
echo -e "  Groq:      $GROQ_STATUS"

if [ "$ANTHROPIC_KEY" = "PLACEHOLDER_SET_THIS_LATER" ] || [ "$GROQ_KEY" = "PLACEHOLDER_SET_THIS_LATER" ]; then
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
    if [ "$INSTALL_SYSTEMD" = "yes" ]; then
        if [ "$MIRA_STARTED" = "yes" ]; then
            print_info "MIRA: http://localhost:1993 (systemd service - running)"
        elif [ "$MIRA_STARTED" = "failed" ]; then
            print_info "MIRA: http://localhost:1993 (systemd service - start failed, check logs)"
        else
            print_info "MIRA: http://localhost:1993 (systemd service - enabled, not started yet)"
        fi
    fi
elif [ "$OS" = "macos" ]; then
    print_info "Valkey: localhost:6379 (brew services)"
    print_info "Vault: http://localhost:8200 (background process)"
    print_info "PostgreSQL: localhost:5432 (brew services)"
fi

echo ""
echo -e "${BOLD}${GREEN}Next Steps${RESET}"
if [ "$INSTALL_SYSTEMD" = "yes" ] && [ "$OS" = "linux" ]; then
    if [ "$MIRA_STARTED" = "yes" ]; then
        echo -e "  ${CYAN}→${RESET} MIRA is running at: ${BOLD}http://localhost:1993${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} View logs: ${BOLD}journalctl -u mira -f${RESET}"
        echo -e "  ${CYAN}→${RESET} Stop MIRA: ${BOLD}sudo systemctl stop mira${RESET}"
    elif [ "$MIRA_STARTED" = "failed" ]; then
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

echo ""
