#!/bin/bash
set -e

# MIRA Comprehensive Deployment Script
# Handles complete setup from fresh system to running MIRA instance
#
# This script:
# 1. Detects platform and checks system dependencies
# 2. Sets up Python virtual environment
# 3. Installs Python dependencies and post-install packages
# 4. Initializes HashiCorp Vault with secrets
# 5. Deploys PostgreSQL database with schema
# 6. Verifies all services are running
# 7. Provides next steps to start MIRA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_header() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}▸${NC} $1"
}

# Platform detection
detect_platform() {
    OS="$(uname -s)"
    case "$OS" in
        Linux*)     PLATFORM=Linux;;
        Darwin*)    PLATFORM=macOS;;
        *)          PLATFORM="UNKNOWN:${OS}"
    esac

    log_info "Detected platform: ${PLATFORM}"

    if [ "$PLATFORM" = "Linux" ]; then
        # Detect Linux distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO=$ID
            log_info "Linux distribution: ${DISTRO}"
        fi
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check PostgreSQL
check_postgresql() {
    log_step "Checking PostgreSQL..."

    if command_exists psql; then
        PG_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
        if [ "$PG_VERSION" -ge 14 ]; then
            log_success "PostgreSQL ${PG_VERSION} installed"
            return 0
        else
            log_warning "PostgreSQL version ${PG_VERSION} found, but 14+ required"
            return 1
        fi
    else
        log_warning "PostgreSQL not found"
        return 1
    fi
}

# Check Valkey/Redis
check_valkey() {
    log_step "Checking Valkey/Redis..."

    if command_exists valkey-server; then
        log_success "Valkey installed"
        return 0
    elif command_exists redis-server; then
        log_success "Redis installed (Valkey-compatible)"
        return 0
    else
        log_warning "Valkey/Redis not found"
        return 1
    fi
}

# Check Vault
check_vault() {
    log_step "Checking HashiCorp Vault..."

    if command_exists vault; then
        VAULT_VERSION=$(vault version | head -n1 | awk '{print $2}')
        log_success "Vault ${VAULT_VERSION} installed"
        return 0
    else
        log_warning "HashiCorp Vault not found"
        return 1
    fi
}

# Check Python
check_python() {
    log_step "Checking Python..."

    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            log_success "Python ${PYTHON_VERSION} installed"
            PYTHON_CMD=python3
            return 0
        else
            log_warning "Python ${PYTHON_VERSION} found, but 3.11+ required"
            return 1
        fi
    else
        log_warning "Python 3 not found"
        return 1
    fi
}

# Install dependencies on macOS
install_macos_deps() {
    log_header "Installing macOS Dependencies via Homebrew"

    if ! command_exists brew; then
        log_error "Homebrew not installed. Install from: https://brew.sh"
        exit 1
    fi

    # Update homebrew
    log_step "Updating Homebrew..."
    brew update

    # Install dependencies (latest versions)
    BREW_PKGS=()

    if ! check_postgresql; then
        BREW_PKGS+=(postgresql)  # Latest PostgreSQL
    fi

    if ! check_valkey; then
        BREW_PKGS+=(valkey)
    fi

    if ! check_vault; then
        BREW_PKGS+=(vault)
    fi

    if ! check_python; then
        BREW_PKGS+=(python3)  # Latest Python 3.x
    fi

    if [ ${#BREW_PKGS[@]} -gt 0 ]; then
        log_step "Installing: ${BREW_PKGS[*]}"
        brew install "${BREW_PKGS[@]}"
        log_success "Dependencies installed"
    else
        log_success "All dependencies already installed"
    fi

    # Start services
    log_step "Starting services..."
    if ! brew services start postgresql; then
        log_error "Failed to start PostgreSQL"
        exit 1
    fi
    if ! brew services start valkey && ! brew services start redis; then
        log_error "Failed to start Valkey/Redis"
        exit 1
    fi
}

# Install dependencies on Linux
install_linux_deps() {
    log_header "Installing Linux Dependencies"

    if [ "$DISTRO" != "ubuntu" ] && [ "$DISTRO" != "debian" ]; then
        log_warning "Automatic installation only supported on Ubuntu/Debian"
        log_info "Please install manually: PostgreSQL 14+, Valkey, HashiCorp Vault, Python 3.11+"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return
    fi

    log_step "Updating package lists..."
    sudo apt-get update

    # Install build tools for Python packages
    log_step "Installing build essentials..."
    sudo apt-get install -y build-essential

    # Install PostgreSQL with development headers
    if ! check_postgresql; then
        log_step "Installing PostgreSQL..."
        sudo apt-get install -y postgresql postgresql-contrib libpq-dev
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    else
        # Ensure dev headers are installed even if PostgreSQL exists
        log_step "Installing PostgreSQL development headers..."
        sudo apt-get install -y libpq-dev
    fi

    # Install Valkey/Redis
    if ! check_valkey; then
        log_step "Installing Redis (Valkey-compatible)..."
        sudo apt-get install -y redis-server
        sudo systemctl start redis-server
        sudo systemctl enable redis-server
    fi

    # Install Vault
    if ! check_vault; then
        log_step "Installing HashiCorp Vault..."
        wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
        sudo apt-get update
        sudo apt-get install -y vault
    fi

    # Install Python (try latest first, fall back to 3.11)
    if ! check_python; then
        log_step "Installing latest Python 3..."

        # Try to install latest python3
        sudo apt-get install -y python3 python3-venv python3-dev python3-pip

        # If that didn't give us 3.11+, specifically install 3.11
        if ! command_exists python3 || ! check_python; then
            log_step "Installing Python 3.11 specifically..."
            sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
        fi
    fi

    log_success "Dependencies installed"
}

# Setup Python virtual environment
setup_python_env() {
    log_header "Python Environment Setup"

    # Create venv if it doesn't exist
    if [ ! -d "${PROJECT_ROOT}/venv" ]; then
        log_step "Creating virtual environment..."
        $PYTHON_CMD -m venv "${PROJECT_ROOT}/venv"
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi

    # Activate venv
    source "${PROJECT_ROOT}/venv/bin/activate"
    log_success "Virtual environment activated"

    # Upgrade pip
    log_step "Upgrading pip..."
    pip install --upgrade pip >/dev/null

    # Install requirements
    log_step "Installing Python dependencies (this may take several minutes)..."
    log_info "Installing core packages..."
    pip install -r "${PROJECT_ROOT}/requirements.txt"
    log_success "Python dependencies installed"

    # Install spacy model (check if already installed)
    log_step "Checking spacy language model (en_core_web_lg)..."
    if "${PROJECT_ROOT}/venv/bin/python" -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null; then
        log_success "Spacy model already installed"
    else
        log_info "Downloading spacy model (~800MB, may take a few minutes)..."
        "${PROJECT_ROOT}/venv/bin/python" -m spacy download en_core_web_lg
        log_success "Spacy model installed"
    fi

    # Download AllMiniLM embeddings model (check if already cached)
    log_step "Checking AllMiniLM embeddings model..."
    if "${PROJECT_ROOT}/venv/bin/python" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" 2>/dev/null; then
        log_success "AllMiniLM model already installed"
    else
        log_info "Downloading AllMiniLM model (~80MB)..."
        if "${PROJECT_ROOT}/venv/bin/python" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" 2>&1; then
            log_success "AllMiniLM model ready"
        else
            log_warning "AllMiniLM model download skipped (will download on first MIRA startup)"
        fi
    fi

    # Download BGE reranker model (check if already cached)
    log_step "Checking BGE reranker model..."
    if "${PROJECT_ROOT}/venv/bin/python" -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-reranker-base'); AutoModel.from_pretrained('BAAI/bge-reranker-base')" 2>/dev/null; then
        log_success "BGE reranker already installed"
    else
        log_info "Downloading BGE reranker model (~1.1GB)..."
        if "${PROJECT_ROOT}/venv/bin/python" -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-reranker-base'); AutoModel.from_pretrained('BAAI/bge-reranker-base')" 2>&1; then
            log_success "BGE reranker ready"
        else
            log_warning "BGE reranker download skipped (will download on first MIRA startup)"
        fi
    fi

    # ONNX runtime info
    log_step "ONNX runtime configured"
    log_info "Additional ONNX-optimized models will download on first use if needed"

    # Optional: Install playwright browsers
    echo ""
    read -p "Install Playwright browsers for web scraping? (optional, ~300MB) (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_step "Installing Playwright browsers..."
        playwright install
        log_success "Playwright browsers installed"
    else
        log_info "Skipping Playwright browsers (can install later with: playwright install)"
    fi
}

# Initialize Vault
initialize_vault() {
    log_header "HashiCorp Vault Initialization"

    log_step "Running Vault setup script..."
    log_info "This will start Vault, initialize it, and prompt for API keys"
    echo ""

    if ! bash "${PROJECT_ROOT}/scripts/setup_vault.sh"; then
        log_error "Vault setup failed"
        exit 1
    fi

    # Verify .vault_keys was created
    if [ ! -f "${PROJECT_ROOT}/.vault_keys" ]; then
        log_error "Vault setup did not create .vault_keys file"
        exit 1
    fi

    log_success "Vault setup complete"
}

# Deploy database
deploy_database() {
    log_header "PostgreSQL Database Deployment"

    # Check if database already exists
    if psql -U postgres -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw mira_service; then
        log_warning "Database 'mira_service' already exists"
        read -p "Re-deploy database? This will DROP and recreate it. (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping database deployment"
            return 0
        fi
    fi

    log_step "Running database deployment script..."
    bash "${PROJECT_ROOT}/deploy/deploy_database.sh"

    log_success "Database deployed successfully"
}

# Verify services are running
verify_services() {
    log_header "Service Verification"

    SERVICES_OK=true

    # Check PostgreSQL
    log_step "Verifying PostgreSQL..."
    if psql -U postgres -c "SELECT 1" >/dev/null 2>&1; then
        log_success "PostgreSQL is running"
    else
        log_error "PostgreSQL is not accessible"
        SERVICES_OK=false
    fi

    # Check Valkey/Redis
    log_step "Verifying Valkey/Redis..."
    if command_exists valkey-cli && valkey-cli ping >/dev/null 2>&1; then
        log_success "Valkey is running"
    elif command_exists redis-cli && redis-cli ping >/dev/null 2>&1; then
        log_success "Redis is running"
    else
        log_error "Valkey/Redis is not running"
        log_info "Start with: valkey-server & (or redis-server &)"
        SERVICES_OK=false
    fi

    # Check Vault
    log_step "Verifying HashiCorp Vault..."
    if [ -f "${PROJECT_ROOT}/.vault_keys" ]; then
        source "${PROJECT_ROOT}/.vault_keys"
        export VAULT_ADDR VAULT_TOKEN
        if vault status >/dev/null 2>&1; then
            log_success "Vault is running and unsealed"
        else
            log_warning "Vault is not running or sealed"
            log_info "Start with: vault server -config=${PROJECT_ROOT}/config/vault.hcl &"
            SERVICES_OK=false
        fi
    else
        log_error "Vault credentials not found"
        SERVICES_OK=false
    fi

    if [ "$SERVICES_OK" = false ]; then
        log_error "Some services are not running. Cannot continue."
        exit 1
    fi

    log_success "All services verified"
}

# Print next steps and optionally start MIRA
print_next_steps() {
    echo ""
    echo "=================================================="
    echo "  MIRA Deployment Complete!"
    echo "=================================================="
    echo ""
    log_success "All dependencies installed and configured"
    echo ""

    if [ -f "${PROJECT_ROOT}/.vault_keys" ]; then
        echo "Vault credentials saved to: ${PROJECT_ROOT}/.vault_keys"
        source "${PROJECT_ROOT}/.vault_keys"
        if [ -n "$VAULT_TOKEN" ]; then
            echo ""
            log_info "To use Vault commands, source the credentials:"
            echo "  source .vault_keys"
        fi
    fi

    echo ""
    log_warning "Important Notes:"
    echo "  • Keep .vault_keys secure (already in .gitignore)"
    echo "  • Vault data persists in: ./vault_data/"
    echo "  • Database credentials: mira_admin / mira_password (change in Vault)"
    echo ""
    echo "On first startup, MIRA will:"
    echo "  • Create a default user (user@localhost)"
    echo "  • Generate an API bearer token"
    echo "  • Display the token for API access"
    echo ""
    echo "For help, see:"
    echo "  • docs/SINGLE_USER_IMPLEMENTATION_GUIDE.md"
    echo "  • README.md"
    echo ""

    # Offer to start MIRA
    echo ""
    read -p "Start MIRA now? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        log_step "Starting MIRA interactive chat..."
        echo ""

        # Activate venv and run talkto_mira.py
        source "${PROJECT_ROOT}/venv/bin/activate"

        if [ -f "${PROJECT_ROOT}/scripts/talkto_mira.py" ]; then
            python "${PROJECT_ROOT}/scripts/talkto_mira.py"
        else
            log_warning "talkto_mira.py not found, starting main.py instead..."
            python "${PROJECT_ROOT}/main.py"
        fi
    else
        echo ""
        echo "To start MIRA later:"
        echo "  ${GREEN}1.${NC} Activate virtual environment:"
        echo "     ${CYAN}source venv/bin/activate${NC}"
        echo ""
        echo "  ${GREEN}2.${NC} Start MIRA interactively:"
        echo "     ${CYAN}python scripts/talkto_mira.py${NC}"
        echo ""
        echo "  ${GREEN}Or${NC} start the server:"
        echo "     ${CYAN}python main.py${NC}"
        echo ""
    fi
}

# Main execution
main() {
    echo "=================================================="
    echo "  MIRA Comprehensive Deployment Script"
    echo "  Single-User API Mode"
    echo "=================================================="
    echo ""

    log_info "Starting deployment process..."
    echo ""

    # Platform detection
    detect_platform

    # Check dependencies
    log_header "Checking System Dependencies"

    MISSING_DEPS=false
    check_postgresql || MISSING_DEPS=true
    check_valkey || MISSING_DEPS=true
    check_vault || MISSING_DEPS=true
    check_python || MISSING_DEPS=true

    # Install missing dependencies if needed
    if [ "$MISSING_DEPS" = true ]; then
        echo ""
        read -p "Install missing dependencies automatically? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ "$PLATFORM" = "macOS" ]; then
                install_macos_deps
            elif [ "$PLATFORM" = "Linux" ]; then
                install_linux_deps
            else
                log_error "Unsupported platform: ${PLATFORM}"
                exit 1
            fi
        else
            log_error "Cannot proceed without dependencies. Please install them manually."
            exit 1
        fi
    fi

    # Re-detect Python
    if check_python; then
        # python3 is good enough
        true
    elif command_exists python3.11; then
        PYTHON_CMD=python3.11
        log_success "Using python3.11"
    else
        log_error "Python 3.11+ not found after installation"
        exit 1
    fi

    # Setup Python environment
    setup_python_env

    # Initialize Vault
    initialize_vault

    # Deploy database
    deploy_database

    # Verify all services
    verify_services

    # Print next steps
    print_next_steps
}

# Run main function
main "$@"
