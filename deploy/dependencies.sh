# deploy/dependencies.sh
# System package installation and Ollama setup
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: OS, DISTRO, CONFIG_OFFLINE_MODE, CONFIG_OLLAMA_MODEL, CONFIG_OLLAMA_SUBCORTICAL_MODEL, LOUD_MODE variables set
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
elif [ "$OS" = "linux" ] && [ "$DISTRO" = "fedora" ]; then
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

# Ollama setup (only for offline mode)
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    print_header "Step 1b: Ollama Setup"

    # Install Ollama if not present
    echo -ne "${DIM}${ARROW}${RESET} Checking for Ollama... "
    if command -v ollama &> /dev/null; then
        echo -e "${CHECKMARK} ${DIM}(already installed)${RESET}"
        OLLAMA_INSTALLED=true
    else
        echo -e "${DIM}(not found)${RESET}"
        if [ "$LOUD_MODE" = true ]; then
            print_step "Installing Ollama..."
            if curl -fsSL https://ollama.com/install.sh | sh; then
                OLLAMA_INSTALLED=true
            else
                OLLAMA_INSTALLED=false
            fi
        else
            (curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1) &
            if show_progress $! "Installing Ollama"; then
                OLLAMA_INSTALLED=true
            else
                OLLAMA_INSTALLED=false
            fi
        fi
    fi

    if [ "$OLLAMA_INSTALLED" = true ]; then
        # Start Ollama server if not already running
        echo -ne "${DIM}${ARROW}${RESET} Checking Ollama server... "
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${CHECKMARK} ${DIM}(already running)${RESET}"
        else
            echo -e "${DIM}(starting)${RESET}"
            # Start server based on OS/init system
            if [ "$OS" = "linux" ] && systemctl is-enabled ollama &>/dev/null 2>&1; then
                run_with_status "Starting Ollama service" \
                    sudo systemctl start ollama
            else
                # Start in background for macOS or non-systemd Linux
                ollama serve > /dev/null 2>&1 &
                OLLAMA_PID=$!
                print_info "Started Ollama server (PID $OLLAMA_PID)"
            fi

            # Wait for server to be ready
            echo -ne "${DIM}${ARROW}${RESET} Waiting for Ollama server... "
            OLLAMA_READY=0
            for i in {1..30}; do
                if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                    OLLAMA_READY=1
                    break
                fi
                sleep 1
            done

            if [ $OLLAMA_READY -eq 1 ]; then
                echo -e "${CHECKMARK} ${DIM}(ready after ${i}s)${RESET}"
            else
                echo -e "${ERROR}"
                print_warning "Ollama server did not start within 30 seconds"
                print_info "Model pull will be skipped - you can pull manually later:"
                print_info "  ollama serve &"
                print_info "  ollama pull ${CONFIG_OLLAMA_MODEL}"
            fi
        fi

        # Pull the model(s) if server is ready
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            if [ "$LOUD_MODE" = true ]; then
                print_step "Pulling model ${CONFIG_OLLAMA_MODEL}..."
                if ollama pull "$CONFIG_OLLAMA_MODEL"; then
                    print_success "Model ${CONFIG_OLLAMA_MODEL} ready"
                else
                    print_warning "Could not pull model (network unavailable)"
                fi
            else
                (ollama pull "$CONFIG_OLLAMA_MODEL" > /dev/null 2>&1) &
                if show_progress $! "Pulling model ${CONFIG_OLLAMA_MODEL}"; then
                    print_success "Model ${CONFIG_OLLAMA_MODEL} ready"
                else
                    print_warning "Could not pull model (network unavailable)"
                    echo ""
                    print_info "For air-gapped installation, manually transfer the model:"
                    print_info "  1. On a connected machine: ollama pull ${CONFIG_OLLAMA_MODEL}"
                    print_info "  2. Export: ~/.ollama/models -> transfer to this machine"
                    print_info "  3. Or use: ollama create ${CONFIG_OLLAMA_MODEL} -f Modelfile"
                fi
            fi

            # Pull subcortical model if it differs from the main model
            if [ -n "$CONFIG_OLLAMA_SUBCORTICAL_MODEL" ] && [ "$CONFIG_OLLAMA_SUBCORTICAL_MODEL" != "$CONFIG_OLLAMA_MODEL" ]; then
                if [ "$LOUD_MODE" = true ]; then
                    print_step "Pulling subcortical model ${CONFIG_OLLAMA_SUBCORTICAL_MODEL}..."
                    if ollama pull "$CONFIG_OLLAMA_SUBCORTICAL_MODEL"; then
                        print_success "Model ${CONFIG_OLLAMA_SUBCORTICAL_MODEL} ready"
                    else
                        print_warning "Could not pull subcortical model (network unavailable)"
                    fi
                else
                    (ollama pull "$CONFIG_OLLAMA_SUBCORTICAL_MODEL" > /dev/null 2>&1) &
                    if show_progress $! "Pulling subcortical model ${CONFIG_OLLAMA_SUBCORTICAL_MODEL}"; then
                        print_success "Model ${CONFIG_OLLAMA_SUBCORTICAL_MODEL} ready"
                    else
                        print_warning "Could not pull subcortical model (network unavailable)"
                    fi
                fi
            fi
        fi
    else
        print_warning "Could not install Ollama (network unavailable or blocked)"
        echo ""
        print_info "For air-gapped Ollama installation:"
        print_info "  1. Download Ollama binary from https://ollama.com/download"
        print_info "  2. Transfer and install manually"
        print_info "  3. Transfer model files to ~/.ollama/models"
        print_info "  4. Start Ollama: ollama serve"
    fi

    print_success "Ollama setup complete"
fi
