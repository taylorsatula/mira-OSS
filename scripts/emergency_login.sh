#!/bin/bash
# Emergency script to create a user account and generate a magic link when email is down

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/emergency_login.sh <email>"
    echo ""
    echo "This script creates a new user account and generates a magic link token"
    echo "that can be used to log in when the email service is down."
    exit 1
fi

EMAIL=$1

# Get environment from running main.py process
VAULT_ADDR=$(cat /proc/$(pgrep -f "main.py$")/environ | tr '\0' '\n' | grep "^VAULT_ADDR=" | cut -d= -f2)
VAULT_ROLE_ID=$(cat /proc/$(pgrep -f "main.py$")/environ | tr '\0' '\n' | grep "^VAULT_ROLE_ID=" | cut -d= -f2)
VAULT_SECRET_ID=$(cat /proc/$(pgrep -f "main.py$")/environ | tr '\0' '\n' | grep "^VAULT_SECRET_ID=" | cut -d= -f2)

if [ -z "$VAULT_ADDR" ]; then
    echo "Error: Could not find main.py process or VAULT_ADDR environment variable"
    exit 1
fi

# Run the Python script with the environment variables
VAULT_ADDR="$VAULT_ADDR" VAULT_ROLE_ID="$VAULT_ROLE_ID" VAULT_SECRET_ID="$VAULT_SECRET_ID" \
    /opt/mira/venv/bin/python3 scripts/create_user_with_magic_link.py "$EMAIL"
