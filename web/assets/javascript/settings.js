/**
 * SETTINGS.JS - Settings Modal Feature Module
 *
 * PURPOSE:
 * Manages the settings modal UI and all configuration operations for the MIRA application.
 * Provides a centralized interface for users to configure API endpoints,
 * calendar connections, WebAuthn devices, and account management. This is a self-contained
 * feature module that handles both UI rendering and backend persistence.
 *
 * RESPONSIBILITIES:
 * - Settings modal lifecycle (open, close, navigation)
 * - API endpoint configuration (baseURL, streaming preferences)
 * - Connection testing (health check, connectivity validation)
 * - Calendar settings (calendar URL configuration)
 * - WebAuthn device management (list, add, remove credentials)
 * - Raw conversation data export (JSON download)
 * - Logout functionality
 * - Settings persistence via backend actions API
 * - Form validation and user feedback (status messages)
 *
 * WHAT GOES HERE:
 * - New settings sections or configuration categories
 * - Additional integration settings (Slack, Discord, etc.)
 * - Settings import/export functionality
 * - Settings validation and migration logic
 * - User preference management (notifications, etc.)
 * - Account management features (password change, data deletion)
 *
 * WHAT DOESN'T GO HERE:
 * - Application state management → core.js
 * - API communication protocol → api-client.js
 * - WebAuthn protocol details → webauthn-client.js
 * - Main application UI → ui.js, messaging.js
 * - Domain knowledge management → domain-knowledge.js
 *
 * DEPENDENCIES:
 * - api-client.js (MiraAPIClient for backend operations)
 * - webauthn-client.js (WebAuthnClient for credential management)
 *
 * DEPENDENTS:
 * - core.js (instantiates SettingsManager and makes it globally available)
 *
 * KEY PATTERNS:
 * - Class-based module with dependency injection (apiClient, webAuthnClient)
 * - Element caching in constructor for performance
 * - Event listener registration in _initializeEventListeners
 * - Async settings loading on modal open
 * - Status message system with auto-clear timers
 * - Modal overlay pattern (click outside to close)
 *
 * UI SECTIONS:
 * - API Settings: Endpoint configuration and connection testing
 * - Calendar Settings: Calendar URL configuration
 * - WebAuthn Settings: Biometric device registration and management
 * - Account Actions: Data export and logout
 *
 * SECURITY CONSIDERATIONS:
 * - WebAuthn credential operations require user interaction
 * - Logout invalidates server session before redirect
 * - Settings API calls include CSRF tokens for write operations
 *
 * LOAD ORDER:
 * After api-client.js and webauthn-client.js, before core.js instantiates it.
 */

class SettingsManager {
    constructor(apiClient, webAuthnClient) {
        this.apiClient = apiClient;
        this.webAuthnClient = webAuthnClient;
        this.elements = this._cacheElements();
        this._initializeEventListeners();
    }
    
    _cacheElements() {
        return {
            modal: document.getElementById('settings-modal'),
            button: document.getElementById('settings-button'),
            
            // API Settings
            apiEndpoint: document.getElementById('api-endpoint'),
            streamingEnabled: document.getElementById('streaming-enabled'),
            testConnection: document.getElementById('test-connection'),
            connectionStatus: document.getElementById('connection-status'),
            
            // Calendar Settings
            calendarUrl: document.getElementById('calendar-url'),
            saveCalendarSettings: document.getElementById('save-calendar-settings'),
            calendarConnectionStatus: document.getElementById('calendar-connection-status'),
            
            // WebAuthn
            webAuthnContent: document.getElementById('webauthn-content'),
            
            // Other
            rawConversationBtn: document.getElementById('raw-conversation-btn'),
            logoutBtn: document.getElementById('logout-btn')
        };
    }
    
    _initializeEventListeners() {
        // Modal control
        if (this.elements.button) {
            this.elements.button.addEventListener('click', () => this.openSettings());
        }
        
        if (this.elements.modal) {
            this.elements.modal.addEventListener('click', (e) => {
                if (e.target === this.elements.modal || e.target.closest('.modal-close')) {
                    this.closeSettings();
                }
            });
        }
        
        // API Settings
        if (this.elements.testConnection) {
            this.elements.testConnection.addEventListener('click', () => this.testConnection());
        }
        
        if (this.elements.apiEndpoint) {
            this.elements.apiEndpoint.addEventListener('change', () => this.saveApiSettings());
        }
        
        if (this.elements.streamingEnabled) {
            this.elements.streamingEnabled.addEventListener('change', () => this.saveApiSettings());
        }
        
        // Calendar Settings
        if (this.elements.saveCalendarSettings) {
            this.elements.saveCalendarSettings.addEventListener('click', () => this.saveCalendarSettings());
        }
        
        
        // Other buttons
        if (this.elements.rawConversationBtn) {
            this.elements.rawConversationBtn.addEventListener('click', () => this.downloadRawConversationData());
        }
        
        if (this.elements.logoutBtn) {
            this.elements.logoutBtn.addEventListener('click', () => this.logout());
        }
    }
    
    async openSettings() {
        if (this.elements.modal) {
            this.elements.modal.classList.remove('hidden');
            await this.loadSettings();
        }
    }
    
    closeSettings() {
        if (this.elements.modal) {
            this.elements.modal.classList.add('hidden');
        }
    }
    
    async loadSettings() {
        // Load API settings
        this._loadApiSettings();
        
        // Load calendar settings
        await this._loadCalendarSettings();
        
        // Load WebAuthn settings
        await this._loadWebAuthnSettings();
    }
    
    _loadApiSettings() {
        const savedEndpoint = localStorage.getItem('mira-api-endpoint');
        const savedStreaming = localStorage.getItem('mira-streaming-enabled') === 'true';
        
        if (this.elements.apiEndpoint && savedEndpoint) {
            this.elements.apiEndpoint.value = savedEndpoint;
        }
        
        if (this.elements.streamingEnabled) {
            this.elements.streamingEnabled.checked = savedStreaming;
        }
    }
    
    async _loadWebAuthnSettings() {
        if (!this.elements.webAuthnContent || !this.webAuthnClient) return;
        
        // Check if WebAuthn is supported
        if (!this.webAuthnClient.isSupported()) {
            this.elements.webAuthnContent.innerHTML = `
                <p class="setting-description">
                    Biometric authentication is not supported on this device or browser.
                    Please use a modern browser with WebAuthn support.
                </p>
            `;
            return;
        }
        
        // Show loading state
        this.elements.webAuthnContent.innerHTML = '<p class="setting-description">Loading biometric settings...</p>';
        
        try {
            // Get credentials
            const credentials = await this.webAuthnClient.getCredentials();
            
            let html = '<div class="webauthn-settings">';
            
            if (credentials.length === 0) {
                html += `
                    <p class="setting-description">No biometric devices registered.</p>
                    <button class="btn btn-sm" id="add-webauthn-device">Add Biometric Device</button>
                `;
            } else {
                html += '<h4>Registered Devices:</h4>';
                credentials.forEach(cred => {
                    html += `
                        <div class="webauthn-device">
                            <span class="device-name">${cred.name || 'Biometric Device'}</span>
                            <span class="device-date">${new Date(cred.created_at).toLocaleDateString()}</span>
                            <button class="btn btn-sm btn-danger" onclick="window.settingsManager.removeWebAuthnDevice('${cred.id}')">Remove</button>
                        </div>
                    `;
                });
                html += '<button class="btn btn-sm" id="add-webauthn-device">Add Another Device</button>';
            }
            
            html += '</div>';
            this.elements.webAuthnContent.innerHTML = html;
            
            // Add event listener for the add button
            const addBtn = document.getElementById('add-webauthn-device');
            if (addBtn) {
                addBtn.addEventListener('click', () => this.registerWebAuthn());
            }
            
        } catch (error) {
            console.error('Failed to load WebAuthn settings:', error);
            this.elements.webAuthnContent.innerHTML = `
                <p class="setting-description error">Failed to load biometric settings. Please try again.</p>
            `;
        }
    }
    
    saveApiSettings() {
        if (this.elements.apiEndpoint) {
            localStorage.setItem('mira-api-endpoint', this.elements.apiEndpoint.value);
            if (this.apiClient) {
                this.apiClient.baseURL = this.elements.apiEndpoint.value;
            }
        }
        
        if (this.elements.streamingEnabled) {
            localStorage.setItem('mira-streaming-enabled', this.elements.streamingEnabled.checked);
        }
    }
    
    async testConnection() {
        if (!this.apiClient) return;
        
        try {
            this._showConnectionStatus('Testing connection...', 'info');
            const response = await this.apiClient.health.checkHealth();
            
            if (response.status === 'healthy') {
                this._showConnectionStatus('Connection successful', 'success');
            } else {
                throw new Error(response.error || 'Unhealthy response');
            }
        } catch (error) {
            console.error('Connection test failed:', error);
            this._showConnectionStatus(`Connection failed: ${error.message}`, 'error');
        }
    }
    
    async registerWebAuthn() {
        if (!this.webAuthnClient) return;
        
        try {
            const result = await this.webAuthnClient.register();
            
            if (result.success) {
                const webAuthnContent = this.elements.webAuthnContent;
                webAuthnContent.innerHTML = `
                    <p class="setting-description success">Biometric authentication enabled successfully!</p>
                `;
                
                // Reload settings after a moment
                setTimeout(() => {
                    this._loadWebAuthnSettings();
                }, 2000);
            } else {
                throw new Error(result.error?.message || 'Registration failed');
            }
            
        } catch (error) {
            console.error('WebAuthn registration failed:', error);
            alert(`Failed to register biometric device: ${error.message}`);
        }
    }
    
    async removeWebAuthnDevice(credentialId) {
        if (!this.webAuthnClient) return;
        
        if (!confirm('Are you sure you want to remove this biometric device?')) {
            return;
        }
        
        try {
            const result = await this.webAuthnClient.removeCredential(credentialId);
            
            if (result.success) {
                // Reload settings
                await this._loadWebAuthnSettings();
            } else {
                throw new Error(result.error?.message || 'Failed to remove device');
            }
            
        } catch (error) {
            console.error('Failed to remove WebAuthn device:', error);
            alert(`Failed to remove device: ${error.message}`);
        }
    }
    
    async downloadRawConversationData() {
        if (!this.apiClient) return;
        
        try {
            const data = await this.apiClient.data.getRawConversationData();
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mira-conversation-data-${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
        } catch (error) {
            console.error('Failed to download conversation data:', error);
            alert('Failed to download conversation data. Please try again.');
        }
    }
    
    async logout() {
        try {
            // Call the logout endpoint to invalidate server session
            if (this.apiClient && this.apiClient.auth && this.apiClient.auth.token) {
                await this.apiClient.http.request('/v0/auth/logout', {
                    method: 'POST'
                });
            }
        } catch (error) {
            console.error('Logout request failed:', error);
            // Continue with local cleanup even if server logout fails
        } finally {
            // Clear local token and redirect
            if (this.apiClient && this.apiClient.auth) {
                this.apiClient.auth.clearToken();
            }
            window.location.href = '/login';
        }
    }
    
    _showConnectionStatus(message, type) {
        if (!this.elements.connectionStatus) return;
        
        this.elements.connectionStatus.textContent = message;
        this.elements.connectionStatus.className = `connection-status ${type}`;
        
        // Clear status after 3 seconds for success/error
        if (type !== 'info') {
            setTimeout(() => {
                this.elements.connectionStatus.textContent = '';
                this.elements.connectionStatus.className = 'connection-status';
            }, 3000);
        }
    }
    
    async _loadCalendarSettings() {
        if (!this.apiClient) return;
        
        try {
            // Load existing calendar settings via actions endpoint
            const response = await this.apiClient.actions.executeAction('user', 'get_calendar_config', {});
            
            if (response && response.calendar_url) {
                if (this.elements.calendarUrl) {
                    this.elements.calendarUrl.value = response.calendar_url;
                }
            }
        } catch (error) {
            console.log('No existing calendar configuration found:', error.message);
        }
    }
    
    async saveCalendarSettings() {
        if (!this.apiClient) {
            this._showCalendarStatus('API client not available', 'error');
            return;
        }
        
        const calendarUrl = this.elements.calendarUrl?.value || '';
        
        // Validate URL format
        if (!calendarUrl) {
            this._showCalendarStatus('Please enter a calendar URL', 'error');
            return;
        }
        
        // Basic URL validation
        try {
            new URL(calendarUrl);
        } catch {
            this._showCalendarStatus('Please enter a valid URL', 'error');
            return;
        }
        
        try {
            this._showCalendarStatus('Saving calendar settings...', 'info');
            
            // Save calendar URL via actions endpoint
            const response = await this.apiClient.actions.executeAction('user', 'store_calendar_config', {
                calendar_url: calendarUrl
            });
            
            if (response.success) {
                this._showCalendarStatus('Calendar settings saved successfully', 'success');
            } else {
                throw new Error(response.error || 'Failed to save settings');
            }
        } catch (error) {
            console.error('Failed to save calendar settings:', error);
            this._showCalendarStatus(`Failed to save settings: ${error.message}`, 'error');
        }
    }
    
    _showCalendarStatus(message, type) {
        if (!this.elements.calendarConnectionStatus) return;
        
        this.elements.calendarConnectionStatus.textContent = message;
        this.elements.calendarConnectionStatus.className = `connection-status ${type}`;
        
        // Clear status after 3 seconds for success/error
        if (type !== 'info') {
            setTimeout(() => {
                this.elements.calendarConnectionStatus.textContent = '';
                this.elements.calendarConnectionStatus.className = 'connection-status';
            }, 3000);
        }
    }
}

// Export for global use
window.SettingsManager = SettingsManager;