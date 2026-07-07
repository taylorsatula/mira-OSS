/**
 * TOOL-CONFIG.JS - Tool Configuration Management
 *
 * PURPOSE:
 * Manages user-specific tool configurations through the /actions/tools API.
 * Auto-generates forms from JSON Schema for configurable tools.
 *
 * RESPONSIBILITIES:
 * - Listing all configurable tools
 * - Fetching/updating tool configurations
 * - Generating forms from JSON Schema
 * - Rendering the Tools settings page
 *
 * API INTEGRATION:
 * - GET /v0/api/actions/tools - List configurable tools
 * - GET /v0/api/actions/tools/{tool} - Get tool config
 * - GET /v0/api/actions/tools/{tool}/schema - Get JSON Schema
 * - PUT /v0/api/actions/tools/{tool} - Update config
 * - DELETE /v0/api/actions/tools/{tool} - Reset to defaults
 */

class ToolConfigManager {
    constructor() {
        this.tools = [];
        this.currentTool = null;
        this.currentSchema = null;
    }

    async csrfHeaders() {
        if (!window.miraAPI?._ensureCsrfToken) {
            return {};
        }
        const csrfToken = await window.miraAPI._ensureCsrfToken();
        return { 'X-CSRF-Token': csrfToken };
    }

    /**
     * Fetch list of all configurable tools
     */
    async fetchTools() {
        try {
            const response = await fetch('/v0/api/actions/tools', {
                method: 'GET',
                credentials: 'include'
            });
            const data = await response.json();
            if (data.success) {
                this.tools = data.data.tools;
                return this.tools;
            }
            throw new Error(data.error?.message || 'Failed to fetch tools');
        } catch (error) {
            console.error('[ToolConfig] Failed to fetch tools:', error);
            return [];
        }
    }

    /**
     * Fetch current config for a specific tool
     */
    async getToolConfig(toolName) {
        try {
            const response = await fetch(`/v0/api/actions/tools/${toolName}`, {
                method: 'GET',
                credentials: 'include'
            });
            const data = await response.json();
            if (data.success) {
                return data.data;
            }
            throw new Error(data.error?.message || 'Failed to fetch config');
        } catch (error) {
            console.error(`[ToolConfig] Failed to fetch config for ${toolName}:`, error);
            return null;
        }
    }

    /**
     * Fetch JSON Schema for a tool's config
     */
    async getToolSchema(toolName) {
        try {
            const response = await fetch(`/v0/api/actions/tools/${toolName}/schema`, {
                method: 'GET',
                credentials: 'include'
            });
            const data = await response.json();
            if (data.success) {
                return data.data.schema;
            }
            throw new Error(data.error?.message || 'Failed to fetch schema');
        } catch (error) {
            console.error(`[ToolConfig] Failed to fetch schema for ${toolName}:`, error);
            return null;
        }
    }

    /**
     * Update config for a tool
     */
    async updateToolConfig(toolName, config) {
        try {
            const response = await fetch(`/v0/api/actions/tools/${toolName}`, {
                method: 'PUT',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                    ...(await this.csrfHeaders())
                },
                body: JSON.stringify({ config })
            });
            const data = await response.json();
            if (data.success) {
                return data.data;
            }
            throw new Error(data.error?.message || 'Failed to update config');
        } catch (error) {
            console.error(`[ToolConfig] Failed to update config for ${toolName}:`, error);
            throw error;
        }
    }

    /**
     * Validate config for a tool (tests connection, discovers folders, etc.)
     */
    async validateToolConfig(toolName, config) {
        try {
            const response = await fetch(`/v0/api/actions/tools/${toolName}/validate`, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                    ...(await this.csrfHeaders())
                },
                body: JSON.stringify({ config })
            });
            const data = await response.json();
            if (data.success) {
                return data.data;
            }
            throw new Error(data.error?.message || 'Validation failed');
        } catch (error) {
            console.error(`[ToolConfig] Failed to validate config for ${toolName}:`, error);
            throw error;
        }
    }

    /**
     * Reset tool config to defaults
     */
    async resetToolConfig(toolName) {
        try {
            const response = await fetch(`/v0/api/actions/tools/${toolName}`, {
                method: 'DELETE',
                credentials: 'include',
                headers: await this.csrfHeaders()
            });
            const data = await response.json();
            if (data.success) {
                return data.data;
            }
            throw new Error(data.error?.message || 'Failed to reset config');
        } catch (error) {
            console.error(`[ToolConfig] Failed to reset config for ${toolName}:`, error);
            throw error;
        }
    }

    /**
     * Generate form HTML from JSON Schema
     * @param {string} toolName - Tool name for special handling
     */
    generateFormFromSchema(schema, currentConfig, toolName = '') {
        if (!schema || !schema.properties) {
            return '<p>No configuration options available.</p>';
        }

        // Fields to hide (managed in folder selection section for email_tool)
        // Also hide oauth_status as it's managed via OAuth button
        const hiddenFields = toolName === 'email_tool'
            ? ['inbox_folder', 'sent_folder', 'drafts_folder', 'trash_folder']
            : [];

        // Fields that trigger OAuth flow instead of showing input
        const oauthFields = ['oauth_status'];

        let formHtml = '<div class="tool-config-form">';
        const properties = schema.properties;
        const required = schema.required || [];

        // Track if this tool has OAuth
        let hasOAuth = false;
        let oauthStatus = 'disconnected';

        for (const [fieldName, fieldSchema] of Object.entries(properties)) {
            // Skip auto-discovered fields
            if (hiddenFields.includes(fieldName)) continue;

            const value = currentConfig[fieldName];
            const description = fieldSchema.description || '';
            const isRequired = required.includes(fieldName);
            const fieldType = fieldSchema.type;

            // Handle OAuth status field specially - render OAuth section instead
            if (oauthFields.includes(fieldName)) {
                hasOAuth = true;
                oauthStatus = value || 'disconnected';
                continue; // OAuth section rendered separately below
            }

            const fieldId = `field-${this._escapeAttribute(fieldName)}`;
            const fieldNameAttr = this._escapeAttribute(fieldName);
            const fieldLabel = this._escapeHtml(this._formatFieldName(fieldName));

            formHtml += `<div class="form-group">`;
            formHtml += `<label for="${fieldId}">${fieldLabel}${isRequired ? ' *' : ''}</label>`;

            if (fieldType === 'boolean') {
                const checked = value ? 'checked' : '';
                formHtml += `<input type="checkbox" id="${fieldId}" name="${fieldNameAttr}" ${checked}>`;
            } else if (fieldType === 'integer' || fieldType === 'number') {
                const min = fieldSchema.minimum !== undefined ? `min="${this._escapeAttribute(fieldSchema.minimum)}"` : '';
                const max = fieldSchema.maximum !== undefined ? `max="${this._escapeAttribute(fieldSchema.maximum)}"` : '';
                formHtml += `<input type="number" id="${fieldId}" name="${fieldNameAttr}" value="${this._escapeAttribute(value ?? '')}" ${min} ${max}>`;
            } else if (fieldSchema.enum) {
                formHtml += `<select id="${fieldId}" name="${fieldNameAttr}">`;
                for (const option of fieldSchema.enum) {
                    const selected = value === option ? 'selected' : '';
                    formHtml += `<option value="${this._escapeAttribute(option)}" ${selected}>${this._escapeHtml(option)}</option>`;
                }
                formHtml += '</select>';
            } else if (fieldType === 'object') {
                const jsonValue = JSON.stringify(value || {}, null, 2);
                formHtml += `<textarea id="${fieldId}" name="${fieldNameAttr}" rows="4">${this._escapeHtml(jsonValue)}</textarea>`;
            } else {
                // String type - check for password/secret fields
                const inputType = (fieldName.toLowerCase().includes('password') || fieldName.toLowerCase().includes('secret'))
                    ? 'password' : 'text';
                formHtml += `<input type="${inputType}" id="${fieldId}" name="${fieldNameAttr}" value="${this._escapeAttribute(value ?? '')}">`;
            }

            if (description) {
                formHtml += `<span class="field-description">${this._escapeHtml(description)}</span>`;
            }

            formHtml += '</div>';
        }

        // Add OAuth section if this tool has oauth_status field
        if (hasOAuth) {
            // Determine OAuth provider from tool name (e.g., "square_tool" -> "square")
            const provider = toolName.replace('_tool', '');
            const isConnected = oauthStatus === 'connected';
            const buttonClass = isConnected ? 'oauth-btn oauth-connected' : 'oauth-btn';
            const providerLabel = this._escapeHtml(this._formatFieldName(provider));
            const providerAttr = this._escapeAttribute(provider);
            const buttonText = isConnected ? `✓ Connected to ${providerLabel}` : `Connect with ${providerLabel}`;
            const merchantId = currentConfig.merchant_id ? `<span class="oauth-merchant-id">Merchant: ${this._escapeHtml(currentConfig.merchant_id)}</span>` : '';

            formHtml += `
                <div class="oauth-section">
                    <div class="oauth-header">
                        <h4>Account Connection</h4>
                    </div>
                    <div class="oauth-content">
                        <p class="oauth-hint">
                            ${isConnected
                                ? 'Your account is connected. You can disconnect and reconnect at any time.'
                                : 'Enter your Application ID and Secret above, then click Connect to authorize access.'
                            }
                        </p>
                        ${merchantId}
                        <div class="oauth-buttons">
                            <button type="button" class="${buttonClass}" data-provider="${providerAttr}" ${!currentConfig.client_id ? 'disabled' : ''}>
                                ${buttonText}
                            </button>
                            ${isConnected ? `<button type="button" class="oauth-disconnect-btn" data-provider="${providerAttr}">Disconnect</button>` : ''}
                        </div>
                    </div>
                    <input type="hidden" name="oauth_status" value="${oauthStatus}">
                </div>
            `;
        }

        // Add folder discovery section for email_tool
        if (toolName === 'email_tool') {
            formHtml += `
                <div class="folder-discovery-section">
                    <div class="folder-header">
                        <h4>Email Folders</h4>
                        <button type="button" class="btn-test-connection">Test Connection & Discover Folders</button>
                    </div>
                    <div id="discovered-folders" class="discovered-folders">
                        <p class="folder-hint">Click "Test Connection" to discover available folders</p>
                    </div>
                </div>
            `;
        }

        formHtml += '</div>';
        return formHtml;
    }

    /**
     * Render discovered folders section with selection dropdowns
     */
    renderDiscoveredFolders(discovered, container) {
        if (!discovered || !discovered.folders) {
            container.innerHTML = '<p class="folder-error">Failed to discover folders</p>';
            return;
        }

        const folders = discovered.folders;
        const mapping = discovered.discovered_folders || {};

        // Helper to create a folder select with optional "not set" option
        const folderSelect = (name, label, currentValue, allowEmpty = false) => {
            const emptyOption = allowEmpty ? '<option value="">(not set)</option>' : '';
            const options = folders.map(f => {
                const selected = f.name === currentValue ? 'selected' : '';
                return `<option value="${this._escapeAttribute(f.name)}" ${selected}>${this._escapeHtml(f.name)}</option>`;
            }).join('');

            return `
                <div class="folder-select-group">
                    <label for="folder-${this._escapeAttribute(name)}">${this._escapeHtml(label)}</label>
                    <select id="folder-${this._escapeAttribute(name)}" name="${this._escapeAttribute(name)}" class="folder-select">
                        ${emptyOption}
                        ${options}
                    </select>
                    ${!currentValue && !allowEmpty ? '<span class="folder-warning">⚠ Not detected</span>' : ''}
                </div>
            `;
        };

        let html = '<div class="folder-config">';
        html += `<p class="folder-success">Connection successful! Found ${folders.length} folders.</p>`;
        html += '<div class="folder-selects">';
        html += folderSelect('inbox_folder', 'Inbox', mapping.inbox_folder, false);
        html += folderSelect('sent_folder', 'Sent', mapping.sent_folder, true);
        html += folderSelect('drafts_folder', 'Drafts', mapping.drafts_folder, true);
        html += folderSelect('trash_folder', 'Trash', mapping.trash_folder, true);
        html += '</div>';
        html += '</div>';

        container.innerHTML = html;
    }

    /**
     * Initiate OAuth flow for a provider
     * @param {string} provider - OAuth provider name (e.g., 'square')
     * @param {HTMLElement} statusEl - Status element for feedback
     */
    async initiateOAuth(provider, statusEl) {
        try {
            statusEl.textContent = `Initiating ${provider} authorization...`;
            statusEl.className = 'status-info';

            const response = await fetch(`/v0/auth/oauth/${provider}/init`, {
                method: 'POST',
                credentials: 'include',
                headers: await this.csrfHeaders()
            });

            const data = await response.json();

            if (data.success && data.data?.auth_url) {
                statusEl.textContent = 'Redirecting to authorization...';
                // Redirect to OAuth provider
                window.location.href = data.data.auth_url;
            } else {
                throw new Error(data.error?.message || 'Failed to initiate OAuth');
            }
        } catch (error) {
            statusEl.textContent = `OAuth error: ${error.message}`;
            statusEl.className = 'status-error';
            throw error;
        }
    }

    /**
     * Check for OAuth callback results in URL parameters
     * Called on page load to show success/error messages
     */
    checkOAuthCallback() {
        const urlParams = new URLSearchParams(window.location.search);

        const oauthSuccess = urlParams.get('oauth_success');
        const oauthError = urlParams.get('oauth_error');

        if (oauthSuccess || oauthError) {
            // Clean up URL
            const cleanUrl = window.location.pathname + window.location.hash;
            window.history.replaceState({}, document.title, cleanUrl);

            // Show notification
            const container = document.querySelector('.settings-content') || document.body;
            const notification = document.createElement('div');
            notification.className = oauthSuccess ? 'settings-notification success' : 'settings-notification error';
            const message = document.createElement('span');
            message.textContent = oauthSuccess
                ? `✓ Successfully connected to ${oauthSuccess}!`
                : `OAuth error: ${decodeURIComponent(oauthError)}`;

            const closeButton = document.createElement('button');
            closeButton.className = 'notification-close';
            closeButton.textContent = '×';

            notification.appendChild(message);
            notification.appendChild(closeButton);

            container.insertBefore(notification, container.firstChild);

            // Auto-dismiss after 5 seconds
            setTimeout(() => notification.remove(), 5000);

            // Close button
            closeButton.addEventListener('click', () => notification.remove());
        }
    }

    /**
     * Extract form values from the generated form
     */
    extractFormValues(formContainer, schema) {
        const config = {};
        const properties = schema.properties || {};

        for (const [fieldName, fieldSchema] of Object.entries(properties)) {
            const input = formContainer.querySelector(`[name="${fieldName}"]`);
            if (!input) continue;

            const fieldType = fieldSchema.type;

            if (fieldType === 'boolean') {
                config[fieldName] = input.checked;
            } else if (fieldType === 'integer') {
                config[fieldName] = parseInt(input.value, 10) || 0;
            } else if (fieldType === 'number') {
                config[fieldName] = parseFloat(input.value) || 0;
            } else if (fieldType === 'object') {
                try {
                    config[fieldName] = JSON.parse(input.value);
                } catch {
                    config[fieldName] = {};
                }
            } else {
                config[fieldName] = input.value;
            }
        }

        return config;
    }

    /**
     * Render the tools list page
     */
    async renderToolsPage(container) {
        container.innerHTML = '<p class="loading">Loading tools...</p>';

        await this.fetchTools();

        if (this.tools.length === 0) {
            container.innerHTML = '<p>No configurable tools found.</p>';
            return;
        }

        let html = '<div class="tools-grid">';
        for (const tool of this.tools) {
            const statusClass = tool.has_user_config ? 'configured' : 'not-configured';
            const statusText = tool.has_user_config ? 'Configured' : 'Not configured';
            html += `
                <div class="tool-card" data-tool="${this._escapeAttribute(tool.name)}">
                    <div class="tool-header">
                        <h3>${this._escapeHtml(this._formatFieldName(tool.name))}</h3>
                        <span class="tool-status ${statusClass}">${statusText}</span>
                    </div>
                    <p class="tool-config-class">${this._escapeHtml(tool.config_class)}</p>
                    <button class="configure-btn" data-tool="${this._escapeAttribute(tool.name)}">Configure</button>
                </div>
            `;
        }
        html += '</div>';

        container.innerHTML = html;

        // Bind click handlers
        container.querySelectorAll('.configure-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.showConfigDialog(btn.dataset.tool, container);
            });
        });
    }

    /**
     * Show configuration dialog for a tool
     */
    async showConfigDialog(toolName, parentContainer) {
        this.currentTool = toolName;
        this.discoveredFolders = null; // Store discovered folders for merge on save

        // Fetch schema and current config in parallel
        const [schema, configData] = await Promise.all([
            this.getToolSchema(toolName),
            this.getToolConfig(toolName)
        ]);

        if (!schema || !configData) {
            alert('Failed to load tool configuration.');
            return;
        }

        this.currentSchema = schema;

        // Create modal
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.style.display = 'flex';
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog--form">
                <div class="modal-header">
                    <h2>Configure ${this._escapeHtml(this._formatFieldName(toolName))}</h2>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div id="tool-config-status"></div>
                    ${this.generateFormFromSchema(schema, configData.config, toolName)}
                </div>
                <div class="modal-actions">
                    <button class="btn btn-ghost btn-reset">Reset to Defaults</button>
                    <button class="btn btn-ghost btn-cancel">Cancel</button>
                    <button class="btn btn-primary btn-save">Save Configuration</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Status element for feedback
        const statusEl = modal.querySelector('#tool-config-status');

        // Wire up test connection button for email_tool
        const testBtn = modal.querySelector('.btn-test-connection');
        if (testBtn) {
            testBtn.addEventListener('click', async () => {
                const foldersContainer = modal.querySelector('#discovered-folders');
                try {
                    testBtn.disabled = true;
                    testBtn.textContent = 'Testing...';
                    foldersContainer.innerHTML = '<p class="folder-hint">Connecting to server...</p>';

                    const config = this.extractFormValues(modal.querySelector('.tool-config-form'), schema);
                    const result = await this.validateToolConfig(toolName, config);

                    // Store discovered folders for merge on save
                    this.discoveredFolders = result.discovered?.discovered_folders || null;
                    this.renderDiscoveredFolders(result.discovered, foldersContainer);

                    statusEl.textContent = 'Connection successful!';
                    statusEl.className = 'status-success';
                } catch (error) {
                    foldersContainer.innerHTML = `<p class="folder-error">Connection failed: ${this._escapeHtml(error.message)}</p>`;
                    statusEl.textContent = error.message;
                    statusEl.className = 'status-error';
                } finally {
                    testBtn.disabled = false;
                    testBtn.textContent = 'Test Connection & Discover Folders';
                }
            });
        }

        // Wire up OAuth connect button
        const oauthBtn = modal.querySelector('.oauth-btn');
        if (oauthBtn) {
            oauthBtn.addEventListener('click', async () => {
                const provider = oauthBtn.dataset.provider;

                // First save the current config (client_id/secret) before initiating OAuth
                try {
                    statusEl.textContent = 'Saving configuration before OAuth...';
                    statusEl.className = 'status-info';

                    const config = this.extractFormValues(modal.querySelector('.tool-config-form'), schema);
                    await this.updateToolConfig(toolName, config);

                    // Now initiate OAuth
                    await this.initiateOAuth(provider, statusEl);
                } catch (error) {
                    statusEl.textContent = `Error: ${error.message}`;
                    statusEl.className = 'status-error';
                }
            });
        }

        // Wire up OAuth disconnect button
        const disconnectBtn = modal.querySelector('.oauth-disconnect-btn');
        if (disconnectBtn) {
            disconnectBtn.addEventListener('click', async () => {
                if (!confirm('Disconnect from this service? You will need to reconnect to use the tool.')) {
                    return;
                }

                const provider = disconnectBtn.dataset.provider;
                try {
                    statusEl.textContent = 'Disconnecting...';
                    statusEl.className = 'status-info';

                    const response = await fetch(`/v0/auth/oauth/${provider}/disconnect`, {
                        method: 'POST',
                        credentials: 'include',
                        headers: await this.csrfHeaders()
                    });

                    const data = await response.json();
                    if (data.success) {
                        statusEl.textContent = 'Disconnected successfully!';
                        statusEl.className = 'status-success';

                        // Refresh the modal to show updated status
                        setTimeout(() => {
                            modal.remove();
                            this.showConfigDialog(toolName, parentContainer);
                        }, 1000);
                    } else {
                        throw new Error(data.error?.message || 'Failed to disconnect');
                    }
                } catch (error) {
                    statusEl.textContent = `Error: ${error.message}`;
                    statusEl.className = 'status-error';
                }
            });
        }

        // Event handlers
        modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
        modal.querySelector('.btn-cancel').addEventListener('click', () => modal.remove());

        modal.querySelector('.btn-save').addEventListener('click', async () => {
            // Helper to show status and scroll it into view
            const showStatus = (message, type) => {
                statusEl.textContent = message;
                statusEl.className = `status-${type}`;
                statusEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            };

            try {
                let config = this.extractFormValues(modal.querySelector('.tool-config-form'), schema);

                // For email_tool, handle folder discovery and selection
                if (toolName === 'email_tool') {
                    // Auto-validate if not already done
                    if (!this.discoveredFolders) {
                        showStatus('Testing connection and discovering folders...', 'info');
                        try {
                            const result = await this.validateToolConfig(toolName, config);
                            this.discoveredFolders = result.discovered?.discovered_folders || null;

                            // Update folder display with selection dropdowns
                            const foldersContainer = modal.querySelector('#discovered-folders');
                            if (foldersContainer && result.discovered) {
                                this.renderDiscoveredFolders(result.discovered, foldersContainer);
                                showStatus('Folders discovered. Review selections and click Save again.', 'info');
                                return; // Let user review folder selections
                            }
                        } catch (validationError) {
                            showStatus(`Connection failed: ${validationError.message}`, 'error');
                            return;
                        }
                    }

                    // Extract folder selections from dropdowns
                    const folderSelects = modal.querySelectorAll('.folder-select');
                    folderSelects.forEach(select => {
                        if (select.name && select.value) {
                            config[select.name] = select.value;
                        }
                    });
                }

                showStatus('Saving...', 'info');

                await this.updateToolConfig(toolName, config);

                statusEl.textContent = 'Configuration saved successfully!';
                statusEl.className = 'status-success';

                // Refresh tools list after saving
                setTimeout(() => {
                    modal.remove();
                    this.renderToolsPage(parentContainer);
                }, 1000);
            } catch (error) {
                statusEl.textContent = `Error: ${error.message}`;
                statusEl.className = 'status-error';
            }
        });

        modal.querySelector('.btn-reset').addEventListener('click', async () => {
            if (!confirm('Reset all settings to defaults?')) return;

            try {
                statusEl.textContent = 'Resetting...';
                statusEl.className = 'status-info';

                const result = await this.resetToolConfig(toolName);

                // Refresh form with defaults
                const formContainer = modal.querySelector('.modal-body');
                formContainer.innerHTML = `
                    <div id="tool-config-status" class="status-success">Configuration reset to defaults!</div>
                    ${this.generateFormFromSchema(schema, result.config, toolName)}
                `;

                setTimeout(() => {
                    modal.remove();
                    this.renderToolsPage(parentContainer);
                }, 1000);
            } catch (error) {
                statusEl.textContent = `Error: ${error.message}`;
                statusEl.className = 'status-error';
            }
        });

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    /**
     * Format field name for display (snake_case to Title Case)
     */
    _formatFieldName(name) {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Escape HTML for safe rendering
     */
    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Escape values before placing them inside quoted HTML attributes.
     */
    _escapeAttribute(value) {
        return this._escapeHtml(String(value))
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }
}

// Export for use in settings page
window.ToolConfigManager = ToolConfigManager;
