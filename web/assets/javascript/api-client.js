/**
 * API-CLIENT.JS - WebSocket Communication & HTTP API Wrapper
 *
 * PURPOSE:
 * Provides the complete network communication layer for the MIRA application. Manages WebSocket
 * connections for real-time chat streaming, handles automatic reconnection with exponential
 * backoff, and wraps HTTP endpoints for authentication, history, actions, and data retrieval.
 * This is the single source of truth for all backend communication.
 *
 * RESPONSIBILITIES:
 * - WebSocket lifecycle management (connect, disconnect, reconnect with backoff)
 * - WebSocket authentication via httpOnly cookies
 * - Message queuing for offline/reconnecting scenarios
 * - Streaming chat message handling with chunk accumulation
 * - HTTP API wrappers (auth, chat, history, actions, data, health)
 * - CSRF token management for write operations
 * - Event emission system (onConnect, onDisconnect, onMessage, onError, onAuthChange)
 * - Tool event normalization and propagation
 * - Session validation and token management
 * - Magic link authentication flow
 * - WebAuthn integration points (begin/complete registration/login)
 *
 * WHAT GOES HERE:
 * - New backend API endpoint wrappers
 * - WebSocket protocol changes or extensions
 * - Authentication method additions (OAuth, SAML, etc.)
 * - Reconnection strategy improvements
 * - Message queuing enhancements (persistence, priority)
 * - Network error handling and retry logic
 * - API client configuration options
 *
 * WHAT DOESN'T GO HERE:
 * - UI rendering or DOM manipulation → messaging.js, ui.js
 * - WebAuthn protocol details → webauthn-client.js
 * - Settings management UI → settings.js
 * - Application state management → core.js
 * - Feature-specific business logic (keep this a dumb transport layer)
 *
 * DEPENDENCIES:
 * - None (this is a foundational infrastructure module)
 * - Browser APIs: WebSocket, fetch, localStorage, document.cookie
 *
 * DEPENDENTS:
 * - core.js (initializes and configures the client)
 * - messaging.js (uses chat API for sending messages)
 * - history.js (uses history API for loading conversations)
 * - settings.js (uses actions API for settings operations)
 * - domain-knowledge.js (uses actions/data APIs for domain operations)
 *
 * KEY PATTERNS:
 * - Singleton instance stored in AppState.apiClient
 * - Event-driven architecture (register handlers, emit events)
 * - Promise-based async API (all methods return promises)
 * - Automatic httpOnly cookie credential inclusion
 * - WebSocket ready state checks before send operations
 * - Exponential backoff for reconnection attempts
 *
 * ARCHITECTURE NOTES:
 * - This is a reusable class that could be extracted to an npm package
 * - Zero DOM dependencies - works in any JavaScript environment
 * - Server-agnostic protocol handling (just JSON over WebSocket)
 * - Designed for httpOnly cookie authentication (more secure than localStorage tokens)
 *
 * LOAD ORDER:
 * Must be loaded BEFORE core.js (which instantiates it).
 */

class MiraAPIClient {
    constructor(config = {}) {
        this.baseURL = config.baseURL || window.location.origin;
        this.wsURL = this.baseURL.replace(/^http/, 'ws') + '/v0/ws/chat';
        this.token = null; // OSS: single-user API key; hosted: httpOnly cookie (placeholder)
        this.csrfToken = null; // Populated lazily for cookie-based writes (hosted only)
        this.ossMode = false; // True when /oss-auth/token succeeds (OSS single-user build)
        this._ossTokenPromise = null; // Memoized /oss-auth/token fetch

        // Eagerly fetch the OSS single-user API key (no-op in hosted builds where
        // /oss-auth/token 404s). Populates this.token so isAuthenticated() is true
        // by the time the user interacts, and onAuthChange fires for reactive UI.
        this._ensureOssToken();
        
        // Connection state
        this.ws = null;
        this.connectionState = 'disconnected'; // disconnected, connecting, connected, authenticated
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        
        // Message handling
        this.messageQueue = [];
        this.activeConversationId = null;
        this.messageCallbacks = new Map(); // For tracking pending messages
        
        // Event handlers
        this.eventHandlers = {
            onConnect: [],
            onDisconnect: [],
            onMessage: [],
            onError: [],
            onAuthChange: []
        };
        
        // Auth module
        this.auth = {
            isAuthenticated: () => {
                // For httpOnly cookies, we can't read them directly
                // Instead, we'll rely on the server to validate via API calls
                // If we have a token in memory, we're likely authenticated
                return !!this.token;
            },

            getToken: () => this.token,

            setToken: (token) => {
                // Note: Server sets the cookie, we just update our in-memory token
                this.token = token;
                this._emit('onAuthChange', { type: 'login', token });

                // Reconnect with new token if needed
                if (this.ws && this.connectionState !== 'authenticated') {
                    this.connect();
                }
            },

            clearToken: () => {
                // Server clears the cookie on logout, we just clear our in-memory token
                this.token = null;
                this._emit('onAuthChange', { type: 'logout' });
                this.disconnect();
            },
            
            validateToken: async () => {
                // For httpOnly cookies, validate via API and mark authenticated in-memory
                try {
                    const user = await this._httpRequest('/v0/auth/session');
                    // Mark as authenticated for client-side gating
                    this.token = 'httponly';
                    this._emit('onAuthChange', { type: 'login', user });
                    return true;
                } catch (error) {
                    throw new Error('No valid session');
                }
            },
            
            requestMagicLink: async (email) => {
                const response = await fetch(`${this.baseURL}/v0/auth/magic-link`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw { response: { data: error } };
                }
                
                return response.json();
            },
            
            verifyMagicLink: async (token) => {
                const response = await fetch(`${this.baseURL}/v0/auth/verify`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw { response: { data: error } };
                }

                const data = await response.json();
                // Server sets the session cookie automatically (httpOnly)
                // Mark as authenticated
                this.token = 'httponly'; // Placeholder to indicate we're authenticated
                this._emit('onAuthChange', { type: 'login', user: data.data.user });
                return data;
            },
            
            signup: async (email, firstName, lastName, timezone, currentFocus) => {
                const response = await fetch(`${this.baseURL}/v0/auth/signup`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email,
                        first_name: firstName,
                        last_name: lastName,
                        timezone,
                        current_focus: currentFocus
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw { response: { data: error } };
                }

                return response.json();
            },
            
            onAuthChange: (handler) => {
                this.eventHandlers.onAuthChange.push(handler);
            },
            
            // WebAuthn methods
            beginWebAuthnRegistration: async () => {
                return await this._httpRequest('/v0/auth/webauthn/register/begin', {
                    method: 'POST'
                });
            },
            
            completeWebAuthnRegistration: async (credential) => {
                return await this._httpRequest('/v0/auth/webauthn/register/complete', {
                    method: 'POST',
                    body: JSON.stringify({ credential })
                });
            },
            
            beginWebAuthnLogin: async (email) => {
                const response = await fetch(`${this.baseURL}/v0/auth/webauthn/login/begin`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw { response: { data: error } };
                }
                
                return response.json();
            },
            
            completeWebAuthnLogin: async (data) => {
                const response = await fetch(`${this.baseURL}/v0/auth/webauthn/login/complete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw { response: { data: error } };
                }
                
                const result = await response.json();

                // Server sets the session cookie automatically (httpOnly)
                if (result.success) {
                    // Mark as authenticated
                    this.token = 'httponly'; // Placeholder to indicate we're authenticated
                    this._emit('onAuthChange', { type: 'login', user: result.user });
                }

                return result;
            },
            
            removeWebAuthnCredential: async (credentialId) => {
                return await this._httpRequest(`/v0/auth/webauthn/credential/${credentialId}`, {
                    method: 'DELETE'
                });
            },
            
            getWebAuthnCredentials: async () => {
                return await this._httpRequest('/v0/auth/webauthn/credentials');
            },
            
            getCurrentUser: () => {
                // Return basic user info from localStorage if available
                try {
                    const userData = localStorage.getItem('mira-user-data');
                    return userData ? JSON.parse(userData) : null;
                } catch {
                    return null;
                }
            }
        };
        
        // Chat module
        this.chat = {
            sendMessage: async (message, stream = true, imageData = null) => {
                console.log('[SEND] sendMessage called, connectionState:', this.connectionState);
                if (!this.auth.isAuthenticated()) {
                    throw new Error('Not authenticated');
                }

                // Ensure connection
                if (this.connectionState !== 'authenticated') {
                    console.log('[SEND] Not authenticated, connecting...');
                    await this.connect();
                }

                // Send message
                const messageId = this._generateId();
                const messageData = {
                    type: 'message',
                    content: message,
                    // Always stream on server; ignore client flag
                    stream: true,
                    id: messageId,
                    include_thinking: true
                };

                // Add image or document data if provided
                if (imageData && imageData.base64 && imageData.mimeType) {
                    const isImage = imageData.mimeType.startsWith('image/');
                    if (isImage) {
                        messageData.image = imageData.base64;
                        messageData.image_type = imageData.mimeType;
                    } else {
                        messageData.document = imageData.base64;
                        messageData.document_type = imageData.mimeType;
                    }
                }

                return new Promise((resolve, reject) => {
                    // Always treat as streaming for client-side handling
                    this.messageCallbacks.set(messageId, { resolve, reject, stream: true });

                    if (this.connectionState === 'authenticated') {
                        console.log('[SEND] Calling _sendMessage for id:', messageId);
                        this._sendMessage(messageData);
                    } else {
                        console.log('[SEND] Queueing message, state:', this.connectionState);
                        // Queue if not connected
                        this.messageQueue.push(messageData);
                    }
                });
            },
            
            streamChat: async (message, imageData = null) => {
                // Returns an async generator for streaming responses
                const messageId = this._generateId();
                const chunks = [];
                let resolver, rejecter;
                
                const streamPromise = new Promise((resolve, reject) => {
                    resolver = resolve;
                    rejecter = reject;
                });
                
                // Set up streaming callback
                this.messageCallbacks.set(messageId, {
                    stream: true,
                    chunks: chunks,
                    resolve: resolver,
                    reject: rejecter
                });
                
                // Send message with optional image data
                await this.chat.sendMessage(message, true, imageData);
                
                // Return async generator
                return {
                    chunks: chunks,
                    promise: streamPromise,
                    [Symbol.asyncIterator]: async function* () {
                        let index = 0;
                        while (true) {
                            // Wait for new chunks
                            while (index >= chunks.length) {
                                // Check if stream is complete
                                try {
                                    const result = await Promise.race([
                                        streamPromise,
                                        new Promise(resolve => setTimeout(resolve, 100))
                                    ]);
                                    if (result) return; // Stream complete
                                } catch (error) {
                                    throw error;
                                }
                            }
                            
                            // Yield available chunks
                            while (index < chunks.length) {
                                yield chunks[index++];
                            }
                        }
                    }
                };
            }
        };
        
        // Don't auto-connect - wait until actually needed
        // This prevents connection attempts with invalid tokens on page load
        
        // History service (HTTP)
        this.history = {
            getHistory: async (params = {}) => {
                const queryParams = new URLSearchParams();
                queryParams.append('type', 'history');
                if (params.offset !== undefined) queryParams.append('offset', params.offset);
                if (params.limit !== undefined) queryParams.append('limit', params.limit);
                if (params.date) queryParams.append('date', params.date);
                if (params.search) queryParams.append('search', params.search);

                const response = await this._httpRequest(`/v0/api/data?${queryParams.toString()}`);
                return response;
            }
        };
        
        // Actions service (HTTP)
        this.actions = {
            executeAction: async (domain, action, data = {}) => {
                const response = await this._httpRequest('/v0/api/actions', {
                    method: 'POST',
                    body: JSON.stringify({ domain, action, data })
                });
                return response;
            },
            
            // Convenience methods
            createReminder: async (reminderData) => {
                return this.actions.executeAction('reminder', 'create', reminderData);
            },
            
            listReminders: async () => {
                return this.actions.executeAction('reminder', 'list');
            },
            
            linkTemporalDay: async (date) => {
                return this.actions.executeAction('conversation', 'link_day', { date });
            },
            
            unlinkTemporalDay: async (archiveId) => {
                return this.actions.executeAction('conversation', 'unlink_day', { archive_id: archiveId });
            }
        };
        
        // Data service (HTTP)
        this.data = {
            getData: async (dataType, params = {}) => {
                const queryParams = new URLSearchParams(params);
                queryParams.append('type', dataType);
                const response = await this._httpRequest(`/v0/api/data?${queryParams.toString()}`);
                return response;
            },
            
            getLinkedDays: async () => {
                return this.data.getData('linked_days');
            }
        };
        
        // Health service (HTTP)
        this.health = {
            checkHealth: async () => {
                try {
                    const response = await this._httpRequest('/v0/api/health');
                    return response;
                } catch (error) {
                    return { status: 'unhealthy', error: error.message };
                }
            }
        };

        // Billing service removed for OSS (endpoints are hosted-only; chat/settings
        // billing UIs guard on window.miraAPI?.billing?.X and early-return when undefined)
    }
    
    // Generation control
    cancelGeneration() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'cancel' }));
        }
    }

    // Connection management
    async connect() {
        if (this.connectionState !== 'disconnected') return;

        // Ensure OSS API key is loaded (no-op in hosted cookie mode)
        await this._ensureOssToken();

        this.connectionState = 'connecting';
        
        try {
            this.ws = new WebSocket(this.wsURL);
            
            this.ws.onopen = () => this._handleOpen();
            this.ws.onclose = (event) => this._handleClose(event);
            this.ws.onerror = (error) => this._handleError(error);
            this.ws.onmessage = (event) => this._handleMessage(event);
            
        } catch (error) {
            this._handleError(error);
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.connectionState = 'disconnected';
        this.reconnectAttempts = 0;
    }
    
    // Event handlers
    _handleOpen() {
        console.log('WebSocket connected, authenticating...');
        this.connectionState = 'connected';

        // OSS sends the fetched API key; hosted sends empty (server validates cookie)
        this.ws.send(JSON.stringify({
            type: 'auth',
            token: this.token || ''
        }));
    }
    
    _handleClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);

        // Reject pending message callbacks so callers clean up their onMessage
        // handlers. Without this, handlers accumulate across reconnections and
        // each streaming chunk gets processed multiple times (doubled text).
        this._rejectPendingCallbacks('WebSocket closed');

        // Check if this was an auth failure (closed immediately after opening)
        // Must check before changing connectionState
        if (event.code === 1000 && event.reason === '' && this.connectionState === 'connected') {
            console.log('WebSocket closed immediately after connect - likely auth failure');
            this.connectionState = 'disconnected';
            this._emit('onDisconnect', event);
            this.auth.clearToken();
            return;
        }

        this.connectionState = 'disconnected';
        this._emit('onDisconnect', event);

        // Attempt reconnection if not deliberate
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this._scheduleReconnect();
        }
    }
    
    _handleError(error) {
        console.error('WebSocket error:', error);
        this._emit('onError', error);
    }
    
    _handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'auth_success':
                    this._handleAuthSuccess(data);
                    break;
                    
                case 'error':
                    this._handleServerError(data);
                    break;
                    
                case 'text':
                    this._handleTextChunk(data);
                    break;
                    
                case 'tool':
                    // Debug: surface incoming tool events in console
                    try { console.log('[WS] tool event received:', data); } catch (e) {}
                    this._handleToolEvent(data);
                    break;
                    
                case 'response':
                    this._handleCompleteResponse(data);
                    break;
                    
                case 'complete':
                    this._handleMessageComplete(data);
                    break;

                case 'cancelled':
                    this._handleMessageComplete(data);
                    break;

                case 'interrupted':
                    this._handleMessageInterrupted(data);
                    break;

                case 'pong':
                    // Keepalive response
                    break;

                case 'thinking':
                    // Thinking events forwarded to onMessage listeners for progressive display
                    break;

                case 'provider_switch':
                    this._handleProviderSwitch(data);
                    break;

                default:
                    console.warn('Unknown message type:', data.type);
            }
            
            this._emit('onMessage', data);
            
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }
    
    _handleAuthSuccess(data) {
        console.log('WebSocket authenticated');
        this.connectionState = 'authenticated';
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this._emit('onConnect', data);
        
        // Process queued messages
        this._processMessageQueue();
        
        // Start keepalive
        this._startKeepalive();
    }
    
    _handleServerError(data) {
        // Check if this is a USER authentication error (not LLM provider auth errors)
        const message = (data.message || '');
        // Only logout for specific user session authentication failures
        if (message === 'Invalid or expired session' || 
            message === 'Authentication timeout' ||
            message.startsWith('Authentication failed:')) {
            console.log('User authentication error:', data.message);
            // Clear token and emit logout
            this.auth.clearToken();
            // Prevent reconnection attempts
            this.reconnectAttempts = this.maxReconnectAttempts;
        }
        
        const activeCallback = this._getActiveMessageCallback();
        if (activeCallback) {
            activeCallback.reject(new Error(data.message));
            this.messageCallbacks.delete(activeCallback.id);
        }
    }
    
    _handleTextChunk(data) {
        const activeCallback = this._getActiveMessageCallback();
        if (activeCallback && activeCallback.stream) {
            activeCallback.chunks?.push(data);
        }
    }
    
    _handleToolEvent(data) {
        // Tool events are informational, emit a normalized duplicate event
        // Ensure our explicit type isn't overridden by spread order
        this._emit('onMessage', { ...data, type: 'tool_event' });
    }

    _handleProviderSwitch(data) {
        const activeCallback = this._getActiveMessageCallback();
        if (activeCallback && activeCallback.stream) {
            activeCallback.chunks = [];
        }
    }
    
    _handleCompleteResponse(data) {
        const activeCallback = this._getActiveMessageCallback();
        if (activeCallback && !activeCallback.stream) {
            activeCallback.resolve({ response: data.content });
            this.messageCallbacks.delete(activeCallback.id);
        }
    }
    
    _handleMessageComplete(data) {
        this.activeConversationId = data.continuum_id;

        const activeCallback = this._getActiveMessageCallback();
        if (activeCallback) {
            const response = {
                continuum_id: data.continuum_id,
                metadata: data.metadata
            };

            if (activeCallback.stream) {
                // Build response from streamed chunks for progressive rendering
                response.response = activeCallback.chunks
                    ?.filter(chunk => chunk.type === 'text')
                    .map(chunk => chunk.content)
                    .join('');

                // Extract emotion from server's complete response (has preserved tags)
                if (data.response && window.extractEmotionEmoji) {
                    const emoji = window.extractEmotionEmoji(data.response);
                    if (emoji && window.streamingState) {
                        window.streamingState.currentEmotion = emoji;
                    }
                }
            }

            activeCallback.resolve(response);
            this.messageCallbacks.delete(activeCallback.id);
        }
    }

    _handleMessageInterrupted(data) {
        this.activeConversationId = data.continuum_id;

        const activeCallback = this._getActiveMessageCallback();
        if (!activeCallback) {
            return;
        }

        const response = {
            response: data.response || '',
            interrupted: true,
            message: data.message,
            error_type: data.error_type,
            balance: data.balance,
            next_drip_at: data.next_drip_at,
            seconds_until_drip: data.seconds_until_drip
        };

        activeCallback.resolve(response);
        this.messageCallbacks.delete(activeCallback.id);
    }
    
    // Helper methods
    _sendMessage(messageData) {
        console.log('[SEND] _sendMessage called, ws exists:', !!this.ws, 'readyState:', this.ws?.readyState, 'OPEN:', WebSocket.OPEN);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('[SEND] Sending message:', messageData.type, messageData.id);
            this.ws.send(JSON.stringify(messageData));
        } else {
            console.log('[SEND] WebSocket not ready, message NOT sent');
        }
    }
    
    _processMessageQueue() {
        while (this.messageQueue.length > 0 && this.connectionState === 'authenticated') {
            const message = this.messageQueue.shift();
            this._sendMessage(message);
        }
    }
    
    _scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            if (this.connectionState === 'disconnected') {
                this.connect();
            }
        }, delay);
    }
    
    _startKeepalive() {
        // Send ping every 30 seconds
        this.keepaliveInterval = setInterval(() => {
            if (this.connectionState === 'authenticated') {
                this._sendMessage({ type: 'ping' });
            }
        }, 30000);
    }
    
    _rejectPendingCallbacks(reason) {
        if (this.messageCallbacks.size === 0) return;
        console.log(`Rejecting ${this.messageCallbacks.size} pending message callback(s): ${reason}`);
        for (const [id, callback] of this.messageCallbacks) {
            callback.reject(new Error(reason));
        }
        this.messageCallbacks.clear();
    }

    _getActiveMessageCallback() {
        // Get the first callback (FIFO order)
        const [id, callback] = this.messageCallbacks.entries().next().value || [];
        return callback ? { ...callback, id } : null;
    }
    
    _generateId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    _getCookie(name) {
        const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
        return match ? match[2] : null;
    }

    async _ensureOssToken() {
        // OSS single-user mode: fetch the API key from /oss-auth/token.
        // Idempotent and memoized. In hosted builds the endpoint 404s and we
        // stay in cookie-based mode (ossMode stays false).
        if (this._ossTokenPromise) return this._ossTokenPromise;
        this._ossTokenPromise = (async () => {
            try {
                const response = await fetch(`${this.baseURL}/oss-auth/token`);
                if (!response.ok) return;
                const data = await response.json();
                if (data && data.token) {
                    this.token = data.token;
                    this.ossMode = true;
                    this._emit('onAuthChange', { type: 'login', token: this.token });
                }
            } catch (e) {
                // Hosted build or endpoint unavailable — fall back to cookie auth
            }
        })();
        return this._ossTokenPromise;
    }

    async _ensureCsrfToken() {
        if (this.csrfToken) return this.csrfToken;
        try {
            const response = await fetch(`${this.baseURL}/v0/auth/csrf`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include'
            });
            if (!response.ok) throw new Error('Failed to obtain CSRF token');
            const data = await response.json();
            const payload = data.data || data;
            this.csrfToken = payload.csrf_token || payload.csrfToken;
            if (!this.csrfToken) throw new Error('Invalid CSRF response');
            return this.csrfToken;
        } catch (e) {
            console.error('CSRF token fetch failed:', e);
            throw e;
        }
    }

    _emit(event, data) {
        const handlers = this.eventHandlers[event] || [];
        handlers.forEach(handler => {
            try {
                handler(data);
            } catch (error) {
                console.error(`Error in ${event} handler:`, error);
            }
        });
    }
    
    // HTTP request helper for non-WebSocket endpoints
    async _httpRequest(endpoint, options = {}) {
        // Ensure OSS API key is loaded (no-op in hosted cookie mode)
        await this._ensureOssToken();

        const url = `${this.baseURL}${endpoint}`;
        const config = {
            method: options.method || 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            // Include cookies in requests (hosted cookie auth; harmless in OSS)
            credentials: 'include'
        };

        if (this.ossMode) {
            // OSS single-user: authenticate via Bearer API key, no CSRF
            config.headers['Authorization'] = `Bearer ${this.token}`;
        } else {
            // Hosted: httpOnly cookies sent automatically with credentials: 'include'.
            // Attach CSRF token for cookie-based write operations to authenticated endpoints.
            const method = (config.method || 'GET').toUpperCase();
            const endpointPath = endpoint.split('?')[0];
            const csrfProtectedEndpoint = (
                endpointPath.startsWith('/v0/api') ||
                (endpointPath.startsWith('/v0/auth') && endpointPath !== '/v0/auth/csrf')
            );
            if (csrfProtectedEndpoint && method !== 'GET' && method !== 'HEAD') {
                try {
                    const csrf = await this._ensureCsrfToken();
                    config.headers['X-CSRF-Token'] = csrf;
                } catch (e) {
                    // Let the request fail with the error for visibility
                }
            }
        }

        // Add body if provided
        if (options.body) {
            config.body = options.body;
        }
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                // Handle 401 Unauthorized by clearing invalid token
                if (response.status === 401 && this.token) {
                    console.log('Invalid token detected, clearing authentication');
                    this.auth.clearToken();
                }
                
                const error = await response.json();
                throw { response: { data: error, status: response.status } };
            }
            
            const data = await response.json();
            return data.data || data;
            
        } catch (error) {
            console.error(`HTTP request failed: ${endpoint}`, error);
            throw error;
        }
    }
}

// Export for use
window.MiraAPIClient = MiraAPIClient;
