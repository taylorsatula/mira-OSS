# deploy/finalize.sh
# Systemd service, cleanup, and success message
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: OS, DISTRO, MIRA_USER, MIRA_GROUP, CONFIG_*, STATUS_*, LOUD_MODE variables set

# Validate required variables
: "${OS:?Error: OS must be set}"
: "${MIRA_USER:?Error: MIRA_USER must be set}"

# Systemd service installation (Linux only, if user opted in)
if [ "${CONFIG_INSTALL_SYSTEMD}" = "yes" ] && [ "$OS" = "linux" ]; then
    print_header "Step 15: Systemd Service Configuration"

    # Extract Vault credentials from files
    echo -ne "${DIM}${ARROW}${RESET} Reading Vault credentials... "
    VAULT_ROLE_ID=$(cat /opt/vault/role-id.txt)
    VAULT_SECRET_ID=$(cat /opt/vault/secret-id.txt)

    if [ -z "$VAULT_ROLE_ID" ] || [ -z "$VAULT_SECRET_ID" ]; then
        echo -e "${ERROR}"
        print_error "Failed to read Vault credentials from /opt/vault/"
        print_info "Skipping systemd service creation"
        CONFIG_INSTALL_SYSTEMD="failed"
        STATUS_MIRA_SERVICE="${ERROR} Configuration failed"
    else
        echo -e "${CHECKMARK}"

        # Create systemd service file
        echo -ne "${DIM}${ARROW}${RESET} Creating systemd service file... "

        # Set correct PostgreSQL service name based on distro
        if [ "$DISTRO" = "fedora" ]; then
            PG_SERVICE="postgresql-17.service"
        else
            PG_SERVICE="postgresql.service"
        fi

        sudo tee /etc/systemd/system/mira.service > /dev/null <<EOF
[Unit]
Description=MIRA - AI Assistant with Persistent Memory
Documentation=https://github.com/taylorsatula/mira-OSS
Requires=vault.service ${PG_SERVICE} valkey.service
After=vault.service ${PG_SERVICE} valkey.service vault-unseal.service
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
        if [ "${CONFIG_START_MIRA_NOW}" = "yes" ]; then
            echo ""
            start_service mira.service systemctl

            # Give service a moment to start
            sleep 2

            # Check if service started successfully
            if sudo systemctl is-active --quiet mira.service; then
                print_success "MIRA service is running"
                print_info "View logs: journalctl -u mira -f"
                STATUS_MIRA_SERVICE="${CHECKMARK} Running"
            else
                print_warning "MIRA service may have failed to start"
                print_info "Check status: systemctl status mira"
                print_info "View logs: journalctl -u mira -n 50"
                STATUS_MIRA_SERVICE="${ERROR} Start failed"
            fi
        else
            print_info "To start later: sudo systemctl start mira"
            print_info "To view logs: journalctl -u mira -f"
            STATUS_MIRA_SERVICE="${DIM}Not started${RESET}"
        fi
    fi
elif [ "${CONFIG_INSTALL_SYSTEMD}" = "no" ]; then
    print_header "Step 15: Systemd Service Configuration"
    print_info "Skipping systemd service installation (user opted out)"
fi

# macOS: write a launcher that exports Vault env vars before starting MIRA.
# On Linux these vars are baked into the systemd unit; macOS has no equivalent
# so without this wrapper `main.py` crashes at import with
# `ValueError: VAULT_ADDR environment variable is required`.
if [ "$OS" = "macos" ]; then
    print_header "Step 15b: MIRA Launcher Script"

    RUN_SH="/opt/mira/app/run.sh"
    echo -ne "${DIM}${ARROW}${RESET} Writing $RUN_SH... "
    cat > "$RUN_SH" <<'LAUNCHER'
#!/bin/bash
# MIRA launcher — exports Vault env vars and starts the server.
set -e
cd "$(dirname "$0")"
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_ROLE_ID=$(cat /opt/vault/role-id.txt)
export VAULT_SECRET_ID=$(cat /opt/vault/secret-id.txt)
exec venv/bin/python3 main.py "$@"
LAUNCHER
    chmod +x "$RUN_SH"
    echo -e "${CHECKMARK}"
    print_info "Start MIRA with: $RUN_SH"
fi

print_header "Step 16: Cleanup"

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

echo ""
echo -e "${BOLD}${BLUE}Important Files${RESET} ${DIM}(/opt/vault/)${RESET}"
print_info "init-keys.txt (Vault unseal key and root token)"
print_info "role-id.txt (AppRole role ID)"
print_info "secret-id.txt (AppRole secret ID)"
if [ "$OS" = "macos" ]; then
    print_info "vault.pid (Vault process ID)"
fi

echo ""
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    echo -e "${BOLD}${BLUE}LLM Provider${RESET}"
    echo -e "  Provider:     ${CYAN}Local llama-server${RESET}"
    if [ "$CONFIG_LOCAL_MODEL_CHOICE" = "auto" ]; then
        echo -e "  Main Model:   ${CYAN}Qwopus3.6-27B-v2-MTP-Q5_K_M${RESET} ${DIM}(port 3090)${RESET}"
        echo -e "  Small Model:  ${CYAN}Qwen3.5-9B-UD-Q3_K_XL${RESET} ${DIM}(port 3092)${RESET}"
        echo -e "  VRAM Target:  ${DIM}Dual RTX 3090 / 48GB total${RESET}"
        echo ""
        print_info "Before first MIRA startup, download models & start servers:"
        print_info "  scripts/download_models_for_offline_install.sh"
        print_info ""
        print_info "Models stored at: /opt/mira/models/"
        print_info "Server logs at:   /opt/mira/logs/llama-{main,small}.log"
        print_info "Health check:     curl http://localhost:3090/health"
    else
        echo -e "  Mode:         ${CYAN}Custom (bring your own GGUF)${RESET}"
        echo ""
        print_info "Place your GGUF files in /opt/mira/models/ and configure llama-server manually."
    fi
else
    echo -e "${BOLD}${BLUE}Provider Configuration${RESET}"
    echo -e "  Chat Provider:   ${STATUS_CHAT_PROVIDER}"
    echo -e "  Chat Model:      ${CYAN}${CONFIG_CHAT_MODEL}${RESET}"
    echo -e "  Chat Key:        ${STATUS_CHAT_KEY}"
    if [ "$CONFIG_CHAT_PROVIDER_TYPE" = "anthropic" ]; then
        if [ "$CONFIG_ANTHROPIC_BATCH_KEY" = "$CONFIG_ANTHROPIC_KEY" ]; then
            echo -e "  Batch Key:       ${DIM}Using main key${RESET}"
        else
            echo -e "  Batch Key:       ${CHECKMARK} Separate key"
        fi
    else
        echo -e "  Batch Key:       ${DIM}Not set (generic chat mode)${RESET}"
    fi
    echo -e "  Subcortical:     ${STATUS_SUBCORTICAL}"
    echo -e "  Subcortical Mdl: ${CYAN}${CONFIG_SUBCORTICAL_MODEL}${RESET}"
    echo -e "  Subcortical Key: ${STATUS_SUBCORTICAL_KEY}"
    echo -e "  Kagi:            ${STATUS_KAGI}"

    if [ "${CONFIG_CHAT_API_KEY}" = "PLACEHOLDER_SET_THIS_LATER" ] || [ "${CONFIG_CHAT_API_KEY}" = "PLACEHOLDER_NOT_CONFIGURED" ] || [ "${CONFIG_SUBCORTICAL_API_KEY}" = "PLACEHOLDER_SET_THIS_LATER" ]; then
        echo ""
        print_warning "Required API keys not configured!"
        print_info "MIRA will not work until you set the missing API keys."
        print_info "To configure later, use Vault CLI:"
        echo -e "${DIM}    export VAULT_ADDR='http://127.0.0.1:8200'${RESET}"
        echo -e "${DIM}    vault login <root-token-from-init-keys.txt>${RESET}"
        echo -e "${DIM}    vault kv put secret/mira/api_keys \\${RESET}"
        echo -e "${DIM}      anthropic_key=\"sk-ant-your-key\" \\${RESET}"
        echo -e "${DIM}      anthropic_batch_key=\"sk-ant-your-key\" \\${RESET}"
        echo -e "${DIM}      subcortical_key=\"gsk_your-groq-key\" \\${RESET}"
        echo -e "${DIM}      provider_key=\"your-chat-provider-key\" \\${RESET}"
        echo -e "${DIM}      kagi_api_key=\"your-kagi-key\"${RESET}"
    fi
fi

echo ""
echo -e "${BOLD}${BLUE}Services Running${RESET}"
if [ "$OS" = "linux" ]; then
    print_info "Valkey: localhost:6379"
    print_info "Vault: http://localhost:8200 (systemd service)"
    print_info "PostgreSQL: localhost:5432 (systemd service)"
    if [ "${CONFIG_INSTALL_SYSTEMD}" = "yes" ]; then
        print_info "MIRA: http://localhost:1993 (systemd service - ${STATUS_MIRA_SERVICE})"
    fi
elif [ "$OS" = "macos" ]; then
    print_info "Valkey: localhost:6379 (brew services)"
    print_info "Vault: http://localhost:8200 (background process)"
    print_info "PostgreSQL: localhost:5432 (brew services)"
fi

echo ""
echo -e "${BOLD}${GREEN}Next Steps${RESET}"
if [ "${CONFIG_INSTALL_SYSTEMD}" = "yes" ] && [ "$OS" = "linux" ]; then
    if [[ "${STATUS_MIRA_SERVICE}" == *"Running"* ]]; then
        echo -e "  ${CYAN}→${RESET} MIRA is running at: ${BOLD}http://localhost:1993${RESET}"
        echo -e "  ${CYAN}→${RESET} Open the web UI: ${BOLD}http://localhost:1993/chat${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} View logs: ${BOLD}journalctl -u mira -f${RESET}"
        echo -e "  ${CYAN}→${RESET} Stop MIRA: ${BOLD}sudo systemctl stop mira${RESET}"
    elif [[ "${STATUS_MIRA_SERVICE}" == *"failed"* ]]; then
        echo -e "  ${CYAN}→${RESET} Check logs: ${BOLD}journalctl -u mira -n 50${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} Try starting: ${BOLD}sudo systemctl start mira${RESET}"
    else
        echo -e "  ${CYAN}→${RESET} Start MIRA: ${BOLD}sudo systemctl start mira${RESET}"
        echo -e "  ${CYAN}→${RESET} Open the web UI: ${BOLD}http://localhost:1993/chat${RESET}"
        echo -e "  ${CYAN}→${RESET} View logs: ${BOLD}journalctl -u mira -f${RESET}"
    fi
    echo ""
    print_info "MIRA will auto-start on system boot (systemd enabled)"
else
    echo -e "  ${CYAN}→${RESET} Start MIRA: ${BOLD}/opt/mira/app/run.sh${RESET}"
    echo -e "  ${CYAN}→${RESET} Open the web UI: ${BOLD}http://localhost:1993/chat${RESET}"
fi

echo ""
print_warning "IMPORTANT: Secure /opt/vault/ - it contains sensitive credentials!"

if [ "$OS" = "macos" ]; then
    echo ""
    echo -e "${BOLD}${YELLOW}macOS Notes${RESET}"
    print_info "Start MIRA with /opt/mira/app/run.sh (exports Vault env vars)"
    print_info "Vault is running as a background process"
    print_info "To stop: kill \$(cat /opt/vault/vault.pid)"
    print_info "After system restart, manually start Vault and unseal:"
    echo -e "${DIM}    /opt/vault/unseal.sh${RESET}"
    print_info "PostgreSQL and Valkey are managed by brew services"
fi

# Degraded-mode banner: subcortical runs on every user turn for entity
# extraction / passage filtering / query expansion. Without a key, those calls
# silently no-op, which looks like working MIRA until a user wonders why
# recall is poor. Announce the state loudly at the end so it can't be missed.
if [ "${CONFIG_SUBCORTICAL_API_KEY}" = "PLACEHOLDER_SET_THIS_LATER" ]; then
    echo ""
    echo -e "${BOLD}${YELLOW}╔════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${YELLOW}║  MIRA installed, but running in DEGRADED MODE          ║${RESET}"
    echo -e "${BOLD}${YELLOW}╚════════════════════════════════════════════════════════╝${RESET}"
    print_warning "No subcortical API key was configured."
    print_info "Entity extraction, passage filtering, and query expansion"
    print_info "will silently no-op on every conversation turn — memory"
    print_info "retrieval will be noticeably less accurate."
    echo ""
    print_info "To enable, store your Groq key in Vault:"
    echo -e "${DIM}    export VAULT_ADDR='http://127.0.0.1:8200'${RESET}"
    echo -e "${DIM}    vault login \$(grep 'Initial Root Token' /opt/vault/init-keys.txt | awk '{print \$NF}')${RESET}"
    echo -e "${DIM}    vault kv patch secret/mira/api_keys subcortical_key=\"gsk_...\"${RESET}"
    echo ""
    print_info "Then restart MIRA to pick up the new key."
fi

echo ""
