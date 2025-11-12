#!/bin/bash
set -e

# MIRA Deployment Script
# This script automates the complete deployment of MIRA OSS

echo "================================"
echo "MIRA Deployment Script v0.94"
echo "================================"
echo ""

# Pre-flight checks
echo "Running pre-flight checks..."

# Check available disk space (need at least 10GB)
AVAILABLE_SPACE=$(df /opt 2>/dev/null | awk 'NR==2 {print $4}' || df / | awk 'NR==2 {print $4}')
REQUIRED_SPACE=10485760  # 10GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "ERROR: Insufficient disk space. Need at least 10GB free, found $(($AVAILABLE_SPACE / 1024 / 1024))GB"
    exit 1
fi

# Check if installation already exists
if [ -d "/opt/mira/app" ]; then
    echo ""
    echo "WARNING: Existing MIRA installation found at /opt/mira/app"
    read -p "This will OVERWRITE the existing installation. Continue? (yes/no): " OVERWRITE
    if [ "$OVERWRITE" != "yes" ]; then
        echo "Installation cancelled."
        exit 0
    fi
    echo "Proceeding with overwrite..."
fi

echo "✓ Pre-flight checks passed"
echo ""

# Check for port conflicts
echo "Checking for port conflicts..."
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
    echo "WARNING: The following ports are already in use:$PORTS_IN_USE"
    echo "MIRA requires ports: 1993 (app), 8200 (vault), 6379 (valkey), 5432 (postgresql)"
    read -p "Stop existing services and continue? (yes/no): " CONTINUE
    if [ "$CONTINUE" != "yes" ]; then
        echo "Installation cancelled. Free up the required ports and try again."
        exit 0
    fi
fi
echo "✓ Port check passed"
echo ""

# Collect API keys upfront so installation can run unattended
echo "================================"
echo "API Key Configuration"
echo "================================"
echo "MIRA requires both Anthropic and Groq API keys to function."
echo "You can set them now or skip and configure later (MIRA won't work until both are set)."
echo ""

# Anthropic API Key (required)
echo "1. Anthropic API Key (REQUIRED)"
echo "   Used for: Main LLM operations (Claude models)"
echo "   Get your key at: https://console.anthropic.com/settings/keys"
echo ""
read -p "Enter your Anthropic API key (or press Enter to skip): " ANTHROPIC_KEY
if [ -z "$ANTHROPIC_KEY" ]; then
    ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
    ANTHROPIC_STATUS="⚠️  NOT SET - You must configure this before using MIRA"
else
    # Basic validation - check if it looks like an Anthropic key
    if [[ $ANTHROPIC_KEY =~ ^sk-ant- ]]; then
        ANTHROPIC_STATUS="✓ Configured"
    else
        echo "   Warning: This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')"
        read -p "   Continue anyway? (y/n): " CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
            ANTHROPIC_STATUS="⚠️  NOT SET"
        else
            ANTHROPIC_STATUS="✓ Configured (unvalidated)"
        fi
    fi
fi
echo ""

# Groq API Key (required)
echo "2. Groq API Key (REQUIRED)"
echo "   Used for: Fast inference and web extraction operations"
echo "   Get your key at: https://console.groq.com/keys"
echo ""
read -p "Enter your Groq API key (or press Enter to skip): " GROQ_KEY
if [ -z "$GROQ_KEY" ]; then
    GROQ_KEY="PLACEHOLDER_SET_THIS_LATER"
    GROQ_STATUS="⚠️  NOT SET - You must configure this before using MIRA"
else
    # Basic validation - check if it looks like a Groq key
    if [[ $GROQ_KEY =~ ^gsk_ ]]; then
        GROQ_STATUS="✓ Configured"
    else
        echo "   Warning: This doesn't look like a valid Groq API key (should start with 'gsk_')"
        read -p "   Continue anyway? (y/n): " CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            GROQ_KEY="PLACEHOLDER_SET_THIS_LATER"
            GROQ_STATUS="⚠️  NOT SET"
        else
            GROQ_STATUS="✓ Configured (unvalidated)"
        fi
    fi
fi
echo ""

echo "API Key Summary:"
echo "  Anthropic: $ANTHROPIC_STATUS"
echo "  Groq:      $GROQ_STATUS"
echo ""
echo "================================"
echo ""

# Detect operating system
OS_TYPE=$(uname -s)
case "$OS_TYPE" in
    Linux*)
        OS="linux"
        echo "Detected OS: Linux (Ubuntu/Debian)"
        ;;
    Darwin*)
        OS="macos"
        echo "Detected OS: macOS"
        ;;
    *)
        echo "ERROR: Unsupported operating system: $OS_TYPE"
        echo "This script supports Linux (Ubuntu/Debian) and macOS only."
        exit 1
        ;;
esac
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo "ERROR: Please do not run this script as root."
   exit 1
fi

# Prompt for sudo password upfront - all interactive prompts are now complete
echo "================================"
echo "Beginning Installation"
echo "================================"
echo "This script requires sudo privileges for system package installation."
echo "Please enter your password - the installation will then run unattended."
echo ""
sudo -v

# Keep sudo alive (Linux only)
if [ "$OS" = "linux" ]; then
    while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
fi

echo ""
echo "✓ All configuration collected"
echo "✓ Installation will now proceed unattended (estimated 10-15 minutes)"
echo "✓ Progress will be displayed as each step completes"
echo ""
sleep 2

echo "Step 1: Installing system dependencies..."
if [ "$OS" = "linux" ]; then
    # Ubuntu/Debian package installation
    sudo apt-get update
    sudo apt-get install -y \
        python3.13-venv \
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
elif [ "$OS" = "macos" ]; then
    # macOS Homebrew package installation
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "ERROR: Homebrew is not installed. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    echo "Updating Homebrew..."
    brew update

    echo "Installing dependencies via Homebrew..."
    brew install \
        python@3.13 \
        wget \
        curl \
        postgresql@17 \
        valkey \
        vault

    # Playwright dependencies are handled differently on macOS
    echo "Note: Playwright will install its own browser dependencies"
fi

echo ""
echo "Step 2: Verifying Python 3.13..."
if [ "$OS" = "linux" ]; then
    # Check if python3.13 is available
    if ! command -v python3.13 &> /dev/null; then
        echo "ERROR: Python 3.13 not found after installation. Check package availability."
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
        echo "ERROR: Python 3.13 not found. Check Homebrew installation."
        exit 1
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Using Python $PYTHON_VERSION at $PYTHON_CMD"

echo ""
echo "Step 3: Downloading MIRA v0.94..."
# Use appropriate home directory based on OS
if [ "$OS" = "linux" ]; then
    DOWNLOAD_DIR="$HOME"
    MIRA_USER="$(whoami)"
    MIRA_GROUP="$(id -gn)"
elif [ "$OS" = "macos" ]; then
    DOWNLOAD_DIR="$HOME"
    MIRA_USER="$(whoami)"
    MIRA_GROUP="staff"
fi

cd "$DOWNLOAD_DIR"
wget -O mira-0.94.tar.gz https://github.com/taylorsatula/mira-OSS/archive/refs/tags/0.94.tar.gz
tar -xzf mira-0.94.tar.gz

echo ""
echo "Step 4: Installing to /opt/mira/app..."
sudo mkdir -p /opt/mira/app
sudo cp -r mira-OSS-0.94/* /opt/mira/app/
sudo chown -R $MIRA_USER:$MIRA_GROUP /opt/mira

echo ""
echo "Step 5: Creating Python virtual environment..."
cd /opt/mira/app
$PYTHON_CMD -m venv venv
venv/bin/python3 -m ensurepip

echo ""
echo "Step 6: Checking PyTorch installation..."
if venv/bin/pip3 show torch &> /dev/null; then
    TORCH_VERSION=$(venv/bin/pip3 show torch | grep Version | awk '{print $2}')
    echo "PyTorch $TORCH_VERSION already installed, using existing installation"
    echo "Note: If you have CUDA-enabled PyTorch, it will be preserved"
else
    echo "No PyTorch found, installing CPU-only version..."
    venv/bin/pip3 install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Step 7: Installing Python dependencies..."
echo "Pre-installing psycopg2-binary to avoid source compilation..."
venv/bin/pip3 install psycopg2-binary
venv/bin/pip3 install -r requirements.txt

echo ""
echo "Step 8: Installing spacy language model..."
venv/bin/python3 -m spacy download en_core_web_lg

echo ""
echo "Step 9: Downloading embedding and reranker models..."
echo "This will download ~500MB of models from Hugging Face..."
venv/bin/python3 << 'EOF'
from sentence_transformers import SentenceTransformer
import os

cache_dir = os.path.expanduser("~/.cache/huggingface")

print("Downloading all-MiniLM-L6-v2 (fast embeddings)...")
fast_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=cache_dir
)
print("✓ all-MiniLM-L6-v2 ready")

print("Downloading BAAI/bge-reranker-base (search reranking)...")
reranker_model = SentenceTransformer(
    "BAAI/bge-reranker-base",
    cache_folder=cache_dir
)
print("✓ BGE reranker ready")

print("All models downloaded successfully!")
EOF

echo ""
echo "Step 10: Installing Playwright browsers..."
venv/bin/playwright install
if [ "$OS" = "linux" ]; then
    sudo venv/bin/playwright install-deps || true
elif [ "$OS" = "macos" ]; then
    echo "Note: Playwright browser dependencies are bundled on macOS"
fi

echo ""
if [ "$OS" = "linux" ]; then
    echo "Step 11: Installing HashiCorp Vault..."
    cd /tmp
    wget -q https://releases.hashicorp.com/vault/1.18.3/vault_1.18.3_linux_amd64.zip
    unzip -o vault_1.18.3_linux_amd64.zip
    sudo mv vault /usr/local/bin/
    sudo chmod +x /usr/local/bin/vault
elif [ "$OS" = "macos" ]; then
    echo "Step 11: Verifying Vault installation..."
    if ! command -v vault &> /dev/null; then
        echo "ERROR: Vault installation failed. Please install manually with: brew install vault"
        exit 1
    fi
    echo "Vault is installed via Homebrew"
fi

echo ""
echo "Step 12: Creating Vault directories..."
sudo mkdir -p /opt/vault/data /opt/vault/config /opt/vault/logs
sudo chown -R $MIRA_USER:$MIRA_GROUP /opt/vault

echo ""
echo "Step 13: Creating Vault configuration..."
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

echo ""
if [ "$OS" = "linux" ]; then
    echo "Step 14: Creating Vault systemd service..."
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

    sudo systemctl daemon-reload
    sudo systemctl enable vault.service
    sudo systemctl start vault.service
    sleep 3
elif [ "$OS" = "macos" ]; then
    echo "Step 14: Starting Vault service..."
    # Start Vault in the background
    vault server -config=/opt/vault/config/vault.hcl > /opt/vault/logs/vault.log 2>&1 &
    VAULT_PID=$!
    echo $VAULT_PID > /opt/vault/vault.pid
    sleep 3

    # Verify Vault started
    if ! kill -0 $VAULT_PID 2>/dev/null; then
        echo "ERROR: Vault failed to start. Check /opt/vault/logs/vault.log for details."
        exit 1
    fi
    echo "Vault started with PID $VAULT_PID"
fi

echo ""
echo "Step 15: Initializing Vault..."
export VAULT_ADDR='http://127.0.0.1:8200'
vault operator init -key-shares=1 -key-threshold=1 > /opt/vault/init-keys.txt
chmod 600 /opt/vault/init-keys.txt

echo ""
echo "Step 16: Unsealing Vault..."
UNSEAL_KEY=$(grep 'Unseal Key 1' /opt/vault/init-keys.txt | awk '{print $NF}')
vault operator unseal "$UNSEAL_KEY"

echo ""
echo "Step 17: Logging into Vault..."
ROOT_TOKEN=$(grep 'Initial Root Token' /opt/vault/init-keys.txt | awk '{print $NF}')
vault login "$ROOT_TOKEN"

echo ""
echo "Step 18: Enabling KV2 and AppRole..."
vault secrets enable -version=2 -path=secret kv
vault auth enable approle

echo ""
echo "Step 19: Creating Vault policy and AppRole..."
cat > /tmp/mira-policy.hcl <<'EOF'
# Allow full access to secret/* path
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/*" {
  capabilities = ["list", "read", "delete"]
}
EOF

vault policy write mira-policy /tmp/mira-policy.hcl
vault write auth/approle/role/mira policies="mira-policy" token_ttl=1h token_max_ttl=4h

echo ""
echo "Step 20: Getting AppRole credentials..."
vault read auth/approle/role/mira/role-id > /opt/vault/role-id.txt
vault write -f auth/approle/role/mira/secret-id > /opt/vault/secret-id.txt

echo ""
echo "Step 21: Creating auto-unseal script..."
cat > /opt/vault/unseal.sh <<'EOF'
#!/bin/bash
export VAULT_ADDR='http://127.0.0.1:8200'
sleep 5
UNSEAL_KEY=$(grep 'Unseal Key 1' /opt/vault/init-keys.txt | awk '{print $NF}')
vault operator unseal "$UNSEAL_KEY"
EOF

chmod +x /opt/vault/unseal.sh

echo ""
if [ "$OS" = "linux" ]; then
    echo "Step 22: Creating auto-unseal systemd service..."
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

    sudo systemctl daemon-reload
    sudo systemctl enable vault-unseal.service
elif [ "$OS" = "macos" ]; then
    echo "Step 22: Auto-unseal script created (manual restart required)..."
    echo "Note: On macOS, you'll need to manually unseal Vault after restart using:"
    echo "  /opt/vault/unseal.sh"
fi

echo ""
if [ "$OS" = "macos" ]; then
    echo "Step 23: Starting services..."
    echo "Starting Valkey..."
    brew services start valkey || true
    echo "Starting PostgreSQL..."
    brew services start postgresql@17 || true
    sleep 3
    echo ""
fi

echo "Step 24: Setting up PostgreSQL..."
if [ "$OS" = "linux" ]; then
    # Linux: use postgres system user
    echo "Creating PostgreSQL database and users..."
    sudo -u postgres psql -c "CREATE DATABASE mira_service;" || true
    sudo -u postgres psql -c "CREATE USER mira_admin WITH PASSWORD 'changethisifdeployingpwd' SUPERUSER;" || true
    sudo -u postgres psql -c "CREATE USER mira_dbuser WITH PASSWORD 'changethisifdeployingpwd';" || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_admin;"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_dbuser;"
elif [ "$OS" = "macos" ]; then
    # macOS: create database and users (services already started above)

    echo "Creating PostgreSQL database and users..."
    # On macOS, run as current user (no postgres system user)
    createdb mira_service 2>/dev/null || true
    psql postgres -c "CREATE USER mira_admin WITH PASSWORD 'changethisifdeployingpwd' SUPERUSER;" || true
    psql postgres -c "CREATE USER mira_dbuser WITH PASSWORD 'changethisifdeployingpwd';" || true
    psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_admin;"
    psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE mira_service TO mira_dbuser;"
fi

echo ""
echo "Step 25: Enabling pgvector extension..."
if [ "$OS" = "linux" ]; then
    sudo -u postgres psql -d mira_service -c "CREATE EXTENSION IF NOT EXISTS vector;"
elif [ "$OS" = "macos" ]; then
    psql -d mira_service -c "CREATE EXTENSION IF NOT EXISTS vector;"
fi

echo ""
echo "Step 26: Storing configuration in Vault..."
echo "Storing API keys..."
vault kv put secret/mira/api_keys \
    anthropic_key="$ANTHROPIC_KEY" \
    groq_key="$GROQ_KEY"

echo "Storing database credentials..."

vault kv put secret/mira/database \
    admin_url="postgresql://mira_admin:changethisifdeployingpwd@localhost:5432/mira_service" \
    password="changethisifdeployingpwd" \
    username="mira_dbuser" \
    service_url="postgresql://mira_dbuser:changethisifdeployingpwd@localhost:5432/mira_service"

vault kv put secret/mira/services \
    app_url="http://localhost:1993" \
    valkey_url="valkey://localhost:6379"

echo ""
echo "Step 27: Creating mira shell alias..."
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

if ! grep -q "alias mira=" "$SHELL_RC" 2>/dev/null; then
    echo "alias mira='/opt/mira/app/venv/bin/python3 /opt/mira/app/talkto_mira.py'" >> "$SHELL_RC"
    echo "Alias added to $SHELL_RC"
fi

echo ""
echo "Step 28: Cleaning up installation files..."
echo "Flushing pip cache..."
venv/bin/pip3 cache purge

echo "Removing temporary files..."
# Remove downloaded tarball
rm -f "$DOWNLOAD_DIR/mira-0.94.tar.gz"

# Remove extracted source directory (both OS)
rm -rf "$DOWNLOAD_DIR/mira-OSS-0.94"

# Remove Vault policy temp file
rm -f /tmp/mira-policy.hcl

# Remove Vault installer files (Linux)
if [ "$OS" = "linux" ]; then
    rm -f /tmp/vault_1.18.3_linux_amd64.zip
    rm -f /tmp/vault
fi

# Rename deploy script to archive it
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
    echo "Deploy script archived as: $NEW_NAME"
fi

echo "✓ Cleanup complete - all temporary files removed"

echo ""
echo "================================"
echo "Deployment Complete!"
echo "================================"
echo ""
echo "✓ MIRA installed to: /opt/mira/app"
echo "✓ All temporary files cleaned up"
if [ -n "$SCRIPT_ARCHIVED" ]; then
    echo "✓ Deploy script archived as: $SCRIPT_ARCHIVED"
fi
echo ""
echo "Important files saved to /opt/vault/:"
echo "  - init-keys.txt (Vault unseal key and root token)"
echo "  - role-id.txt (AppRole role ID)"
echo "  - secret-id.txt (AppRole secret ID)"
if [ "$OS" = "macos" ]; then
    echo "  - vault.pid (Vault process ID)"
fi
echo ""
echo "API Key Configuration:"
echo "  Anthropic: $ANTHROPIC_STATUS"
echo "  Groq:      $GROQ_STATUS"
if [ "$ANTHROPIC_KEY" = "PLACEHOLDER_SET_THIS_LATER" ] || [ "$GROQ_KEY" = "PLACEHOLDER_SET_THIS_LATER" ]; then
    echo ""
    echo "⚠️  WARNING: Required API keys not configured!"
    echo "   MIRA will not work until you set both API keys."
    echo "   To configure later, use Vault CLI:"
    echo "     export VAULT_ADDR='http://127.0.0.1:8200'"
    echo "     vault login <root-token-from-init-keys.txt>"
    echo "     vault kv put secret/mira/api_keys \\"
    echo "       anthropic_key=\"sk-ant-your-key\" \\"
    echo "       groq_key=\"gsk_your-key\""
fi
echo ""
echo "Services running:"
if [ "$OS" = "linux" ]; then
    echo "  - Valkey: localhost:6379"
    echo "  - Vault: http://localhost:8200 (systemd service)"
    echo "  - PostgreSQL: localhost:5432 (systemd service)"
elif [ "$OS" = "macos" ]; then
    echo "  - Valkey: localhost:6379 (brew services)"
    echo "  - Vault: http://localhost:8200 (background process)"
    echo "  - PostgreSQL: localhost:5432 (brew services)"
fi
echo ""
if [ "$OS" = "linux" ]; then
    echo "To use MIRA CLI, run: source ~/.bashrc && mira"
elif [ "$OS" = "macos" ]; then
    echo "To use MIRA CLI, run: source $SHELL_RC && mira"
fi
echo ""
echo "IMPORTANT: Secure the /opt/vault directory - it contains sensitive credentials!"
if [ "$OS" = "macos" ]; then
    echo ""
    echo "macOS Notes:"
    echo "  - Vault is running as a background process. To stop: kill \$(cat /opt/vault/vault.pid)"
    echo "  - After system restart, manually start Vault and unseal using /opt/vault/unseal.sh"
    echo "  - PostgreSQL and Valkey are managed by brew services"
fi
