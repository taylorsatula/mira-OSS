# deploy/dependencies.sh
# System package installation and local LLM setup (llama.cpp)
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: OS, DISTRO, CONFIG_OFFLINE_MODE, CONFIG_LOCAL_MODEL_CHOICE, LOUD_MODE variables set
#
# Sets: PYTHON_VER

# Validate required variables
: "${OS:?Error: OS must be set}"
# DISTRO can be empty for macOS, so just check if variable exists
if [ -z "${DISTRO+x}" ]; then
    echo "Error: DISTRO variable must be set (can be empty string for macOS)"
    exit 1
fi

print_header "Step 1: System Dependencies"

if [ "$OS" = "linux" ] && [ "$DISTRO" = "debian" ]; then
    # Debian/Ubuntu: Add PostgreSQL APT repository for PostgreSQL 17
    if [ ! -f /etc/apt/sources.list.d/pgdg.list ]; then
        run_with_status "Adding PostgreSQL APT repository" \
            bash -c 'sudo apt-get install -y ca-certificates wget > /dev/null 2>&1 && \
                     sudo install -d /usr/share/postgresql-common/pgdg && \
                     sudo wget -q -O /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc && \
                     echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list > /dev/null'
    fi

    # Detect Python version to use (newest available, 3.12+ required)
    PYTHON_VER=$(python3 --version 2>&1 | sed -n 's/Python \([0-9]*\.[0-9]*\).*/\1/p')

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating package lists..."
        sudo apt-get update
        print_step "Installing system packages (Python ${PYTHON_VER})..."
        sudo apt-get install -y \
            build-essential \
            cmake \
            git \
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
            build-essential cmake git python${PYTHON_VER}-venv python${PYTHON_VER}-dev libpq-dev \
            postgresql-server-dev-17 unzip wget curl postgresql-17 \
            postgresql-contrib postgresql-17-pgvector valkey \
            libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 \
            libxcomposite1 > /dev/null 2>&1) &
        show_progress $! "Installing system packages (18 packages)"
    fi
elif [ "$OS" = "linux" ] && [ "$DISTRO" = "fedora" ]; then
    # Check minimum Fedora version (PGDG dropped support for F-40 and earlier)
    FEDORA_VER=$(rpm -E %fedora 2>/dev/null || echo 0)
    if [ "$FEDORA_VER" -lt 41 ]; then
        print_error "Fedora $FEDORA_VER is not supported — MIRA requires Fedora 41+"
        print_info "PostgreSQL 17 + PGDG repository are unavailable on older releases."
        exit 1
    fi

    # Fedora/RHEL: Add PostgreSQL PGDG repository for PostgreSQL 17
    if ! rpm -q pgdg-fedora-repo-latest > /dev/null 2>&1 && ! rpm -q pgdg-redhat-repo-latest > /dev/null 2>&1; then
        if [ -f /etc/fedora-release ]; then
            run_with_status "Adding PostgreSQL PGDG repository" \
                sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/F-$(rpm -E %fedora)-x86_64/pgdg-fedora-repo-latest.noarch.rpm
        else
            # RHEL/CentOS/Rocky/Alma
            run_with_status "Adding PostgreSQL PGDG repository" \
                sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-$(rpm -E %rhel)-x86_64/pgdg-redhat-repo-latest.noarch.rpm
        fi
    fi

    # Disable built-in PostgreSQL module to avoid conflicts
    run_quiet sudo dnf -qy module disable postgresql || true

    # Determine correct development tools group name
    # Fedora uses "development-tools", RHEL/Rocky/Alma use "Development Tools"
    if [ -f /etc/fedora-release ]; then
        DEV_TOOLS_GROUP="@development-tools"
    else
        DEV_TOOLS_GROUP="@Development Tools"
    fi

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating package lists..."
        sudo dnf makecache
        print_step "Installing system packages..."
        sudo dnf install -y \
            "$DEV_TOOLS_GROUP" \
            python3-devel \
            python3-pip \
            libpq-devel \
            postgresql17-server \
            postgresql17-contrib \
            postgresql17-devel \
            pgvector_17 \
            unzip \
            wget \
            curl \
            valkey \
            atk \
            at-spi2-atk \
            at-spi2-core \
            libXcomposite
    else
        # Silent mode with progress indicator
        (sudo dnf makecache > /dev/null 2>&1) &
        show_progress $! "Updating package lists"

        (sudo dnf install -y \
            "$DEV_TOOLS_GROUP" python3-devel python3-pip libpq-devel \
            postgresql17-server postgresql17-contrib postgresql17-devel pgvector_17 \
            unzip wget curl valkey \
            atk at-spi2-atk at-spi2-core libXcomposite > /dev/null 2>&1) &
        show_progress $! "Installing system packages (17 packages)"
    fi

    # Initialize PostgreSQL database cluster if not already done
    if [ ! -d /var/lib/pgsql/17/data/base ]; then
        run_with_status "Initializing PostgreSQL database cluster" \
            sudo /usr/pgsql-17/bin/postgresql-17-setup initdb
    fi

    # Configure pg_hba.conf for password authentication (Fedora defaults to ident)
    PG_HBA="/var/lib/pgsql/17/data/pg_hba.conf"
    if [ -f "$PG_HBA" ]; then
        # Check if already configured for scram-sha-256/md5
        if ! grep -q "^local.*all.*all.*scram-sha-256" "$PG_HBA" 2>/dev/null; then
            run_with_status "Configuring PostgreSQL authentication (scram-sha-256)" \
                bash -c "sudo sed -i 's/^local.*all.*all.*ident/local   all             all                                     scram-sha-256/' $PG_HBA && \
                         sudo sed -i 's/^local.*all.*all.*peer/local   all             all                                     scram-sha-256/' $PG_HBA && \
                         sudo sed -i 's/^host.*all.*all.*127.0.0.1.*ident/host    all             all             127.0.0.1\\/32            scram-sha-256/' $PG_HBA && \
                         sudo sed -i 's/^host.*all.*all.*::1.*ident/host    all             all             ::1\\/128                 scram-sha-256/' $PG_HBA"
        fi
    fi

    # Enable and start PostgreSQL service
    run_with_status "Enabling PostgreSQL service" \
        sudo systemctl enable postgresql-17
    run_with_status "Starting PostgreSQL service" \
        sudo systemctl start postgresql-17

    # Enable and start Valkey service
    run_with_status "Enabling Valkey service" \
        sudo systemctl enable valkey
    run_with_status "Starting Valkey service" \
        sudo systemctl start valkey

    # Detect Python version after installation
    PYTHON_VER=$(python3 --version 2>&1 | sed -n 's/Python \([0-9]*\.[0-9]*\).*/\1/p')

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
    # Check for Python 3.12+ in descending order of preference
    PYTHON_VER=""
    for ver in 3.14 3.13 3.12; do
        if command -v python${ver} &> /dev/null; then
            PYTHON_VER="${ver}"
            break
        fi
    done
    
    # If no suitable version found, default to 3.12 for installation
    if [ -z "$PYTHON_VER" ]; then
        PYTHON_VER="3.12"
    fi

    # valkey and redis both install `redis-*` binaries — if redis formula is
    # present, `brew install valkey` aborts with a conflict error. Unlink redis
    # preemptively (the formula stays installed; just its symlinks are removed)
    # so the install can proceed. Users can `brew link redis` later if they
    # want both around.
    if brew list --formula 2>/dev/null | grep -q "^redis$"; then
        echo -ne "${DIM}${ARROW}${RESET} Unlinking redis formula (conflicts with valkey install)... "
        brew unlink redis > /dev/null 2>&1 || true
        echo -e "${CHECKMARK}"
    fi

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating Homebrew..."
        # Tolerate non-zero exit from `brew update` — a single broken third-party
        # tap (e.g. one whose remote has been deleted) makes brew update exit
        # non-zero, which `set -e` would otherwise turn into a silent deploy
        # abort. The subsequent `brew install` does its own index refresh, so
        # stale indices aren't actually a concern here.
        brew update || print_warning "brew update reported errors (continuing; usually a broken tap)"
        print_step "Adding HashiCorp tap..."
        brew tap hashicorp/tap
        print_step "Installing dependencies via Homebrew (Python ${PYTHON_VER})..."
        brew install python@${PYTHON_VER} wget curl postgresql@17 pgvector valkey hashicorp/tap/vault
    else
        (brew update > /dev/null 2>&1 || true) &
        show_progress $! "Updating Homebrew"

        (brew tap hashicorp/tap > /dev/null 2>&1) &
        show_progress $! "Adding HashiCorp tap"

        (brew install python@${PYTHON_VER} wget curl postgresql@17 pgvector valkey hashicorp/tap/vault > /dev/null 2>&1) &
        show_progress $! "Installing dependencies via Homebrew (7 packages)"
    fi

    print_info "Playwright will install its own browser dependencies"
elif [ "$OS" = "linux" ]; then
    # Unsupported Linux distribution
    print_error "Unsupported Linux distribution: $DISTRO"
    print_info "Supported distributions:"
    print_info "  - Debian/Ubuntu and derivatives (apt)"
    print_info "  - Fedora/RHEL/Rocky/Alma/CentOS (dnf)"
    print_info ""
    print_info "For other distributions, install these dependencies manually:"
    print_info "  - Python 3.12+ with venv and dev headers"
    print_info "  - PostgreSQL 17 with pgvector extension"
    print_info "  - Valkey (Redis-compatible)"
    print_info "  - Build tools (gcc, make, etc.)"
    print_info "  - libpq development headers"
    print_info ""
    print_info "Then re-run with: DISTRO=debian ./deploy.sh (to skip package install)"
    exit 1
fi

print_success "System dependencies installed"

# Local LLM setup via llama.cpp (only for offline/local mode)
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    print_header "Step 1b: llama.cpp Setup"

    LLAMA_MODELS_DIR="/opt/mira/models"
    LLAMA_MAIN_PORT=8080
    LLAMA_SMALL_PORT=8081

    # --- Detect or build llama-server ---
    echo -ne "${DIM}${ARROW}${RESET} Checking for llama-server... "
    if command -v llama-server &> /dev/null; then
        echo -e "${CHECKMARK} ${DIM}(found in PATH)${RESET}"
    else
        echo -e "${DIM}(not found, building from source)${RESET}"

        # Check for required build tools
        MISSING_TOOLS=""
        for tool in cmake git g++; do
            if ! command -v $tool &> /dev/null; then
                MISSING_TOOLS="$MISSING_TOOLS $tool"
            fi
        done

        if [ -n "$MISSING_TOOLS" ]; then
            print_error "Missing build tools:$MISSING_TOOLS"
            print_info "Install them first, then re-run deploy."
            if [ "$OS" = "linux" ] && [ "$DISTRO" = "debian" ]; then
                print_info "  sudo apt install -y cmake git build-essential"
            elif [ "$OS" = "linux" ] && [ "$DISTRO" = "fedora" ]; then
                print_info "  sudo dnf install -y cmake git gcc-c++"
            elif [ "$OS" = "macos" ]; then
                print_info "  brew install cmake"
            fi
            exit 1
        fi

        # Detect CUDA availability
        USE_CUDA="OFF"
        if command -v nvcc &> /dev/null; then
            USE_CUDA="ON"
        fi

        # Clone and build llama.cpp
        BUILD_DIR="/tmp/llama.cpp-build"
        rm -rf "$BUILD_DIR"

        if [ "$LOUD_MODE" = true ]; then
            print_step "Cloning llama.cpp..."
            git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$BUILD_DIR"
            if [ "$USE_CUDA" = "ON" ]; then
                print_step "Building llama.cpp with CUDA support (this may take several minutes)..."
            else
                print_step "Building llama.cpp (CPU only, no CUDA detected)..."
            fi
            cd "$BUILD_DIR" && cmake -B build -DGGML_CUDA=$USE_CUDA -DLLAMA_SERVER=ON
            cd "$BUILD_DIR" && cmake --build build --config Release -j$(nproc)
            run_with_status "Installing llama.cpp" \
                sudo cmake --install build
        else
            (git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$BUILD_DIR" > /dev/null 2>&1 && \
             cd "$BUILD_DIR" && cmake -B build -DGGML_CUDA=$USE_CUDA -DLLAMA_SERVER=ON > /dev/null 2>&1 && \
             cmake --build build --config Release -j$(nproc) > /dev/null 2>&1 && \
             sudo cmake --install build > /dev/null 2>&1) &
            if [ "$USE_CUDA" = "ON" ]; then
                PROGRESS_MSG="Building llama.cpp from source (CUDA)"
            else
                PROGRESS_MSG="Building llama.cpp from source (CPU)"
            fi
            if show_progress $! "$PROGRESS_MSG"; then
                echo -e "${CHECKMARK}"
            else
                print_error "llama.cpp build failed"
                exit 1
            fi
        fi
        rm -rf "$BUILD_DIR"

        # Verify installation
        if ! command -v llama-server &> /dev/null; then
            print_error "llama-server not found after build — installation may have failed"
            exit 1
        fi
        echo -e "${CHECKMARK} ${DIM}(built & installed)${RESET}"
    fi

    # Create models directory (downloaded later by standalone script)
    run_with_status "Creating models directory" \
        sudo mkdir -p "$LLAMA_MODELS_DIR"
    run_quiet sudo chown -R $(whoami): "$LLAMA_MODELS_DIR"

    if [ "$CONFIG_LOCAL_MODEL_CHOICE" = "custom" ]; then
        print_info "Custom model mode — place your GGUF files in $LLAMA_MODELS_DIR/"
        if [ -n "${CONFIG_CUSTOM_GGUF:-}" ]; then
            print_info "User-specified model: $CONFIG_CUSTOM_GGUF"
        fi
    fi

    print_success "llama.cpp setup complete"
fi
