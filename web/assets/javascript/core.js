// Register service worker for PWA cache management
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js');
}

/**
 * CORE.JS - Application Foundation
 *
 * PURPOSE:
 * Provides the foundational infrastructure for the MIRA web application. This is the root module
 * that all other modules depend on. It establishes application state, caches DOM elements,
 * initializes external services, and orchestrates the application startup sequence.
 *
 * RESPONSIBILITIES:
 * - Define and maintain global application state (AppState object)
 * - Cache DOM element references for performance (elements object)
 * - Initialize and configure the API client with authentication and WebSocket handlers
 * - Set up WebAuthn client for biometric authentication
 * - Coordinate application startup sequence (auth, session restoration, event setup)
 * - Manage authentication lifecycle (login/logout event handling)
 * - Provide utility functions that don't fit elsewhere (feedback email)
 *
 * WHAT GOES HERE:
 * - New application-level state properties (add to AppState)
 * - New DOM element references needed across modules (add to elements)
 * - API client configuration and global event handlers
 * - Initialization logic that must run before other modules
 * - Application lifecycle hooks (startup, shutdown)
 * - Core utility functions used across multiple modules
 *
 * WHAT DOESN'T GO HERE:
 * - UI-specific logic (animations, visual effects) → ui.js
 * - Message handling and rendering → messaging.js
 * - History and calendar features → history.js
 * - Event listeners and handlers → events.js
 * - Feature-specific state (use AppState but define logic in feature modules)
 *
 * DEPENDENCIES:
 * - External: MiraAPIClient (api-client.js), WebAuthnClient (webauthn-client.js), SettingsManager (settings.js)
 * - None (this is the foundation module)
 *
 * DEPENDENTS:
 * - ui.js, messaging.js, history.js, events.js (all depend on AppState and elements)
 *
 * LOAD ORDER:
 * Must be loaded FIRST in HTML, before any other application modules.
 */

// ========================================
// APPLICATION STATE & INITIALIZATION
// ========================================

const AppState = {
	// UI State
	historyOpen: false,
	responseActive: false,
	historyDrawer: {
		currentOffset: 0,
		currentScope: 'today',
		isLoading: false,
		hasMore: false
	},

	// Calendar
	currentCalendarDate: new Date(),

	// Message Queue
	messageQueue: JSON.parse(localStorage.getItem('mira-queue') || '[]'),

	// API Client instance
	apiClient: null,

	// WebAuthn client instance
	webAuthnClient: null,

	// Set when a streaming response was interrupted by WS disconnect.
	// Cleared after recovery fetch on reconnect/tab-return.
	responseInterrupted: false,

	// Attached files for multimodal messages
	attachedFiles: [],

	// Intersection Observer for infinite scroll
	scrollObserver: null,

	// Settings manager instance
	settingsManager: null
};

// ========================================
// DOM ELEMENTS CACHE
// ========================================

const elements = {
	// Loading
	loadingScreen: document.getElementById('loading-screen'),
	asciiContainer: document.getElementById('ascii-container'),

	// Response - responseBox is scroll container, responseContent is inner wrapper
	responseBox: document.getElementById('response_box'),
	responseContent: document.getElementById('response_content'),

	// Input
	inputSection: document.getElementById('compose_bar'),
	inputContainer: document.querySelector('#compose_bar form'),
	messageInput: document.getElementById('chat_field'),
	ghostText: document.getElementById('ghost-text'),
	sendButton: document.getElementById('send-button'),
	fileInput: document.getElementById('file-input'),
	attachmentPopover: document.getElementById('attachment-popover'),
	attachmentList: document.getElementById('attachment-list'),
	attachmentAddBtn: document.getElementById('attachment-add-btn'),
	preserveResolution: document.getElementById('preserve-resolution'),
	queueIndicator: document.getElementById('queue-indicator'),
	queueCount: document.querySelector('.retry-count'),
	queuePopover: document.getElementById('queue-popover'),
	queueMessages: document.getElementById('queue-messages'),

	// Tool indicators from toolbar
	toolBadge: document.querySelector('[data-indicator="toolcall_btn"]'),
	workflowBadge: null,
	attachmentButton: document.querySelector('[data-indicator="attachment_btn"]'),
	toolbarRight: document.querySelector('#toolbar .rightside'),

	// Inline history elements
	inlineHistoryContainer: document.getElementById('inline-history-container'),
	inlineHistorySentinel: document.getElementById('inline-history-sentinel'),
	newMessageToast: document.getElementById('new-message-toast'),
	toastScrollBtn: document.getElementById('toast-scroll-btn'),

	// These don't exist in new UI - set to null to avoid errors
	historyDrawer: null,
	historyContent: null,
	historySearch: null,
	datePicker: null,
	calendarPopup: null,
	responseContainer: document.getElementById('response_box'),
	workflowPopover: null,
	workflowSteps: null,
	settingsModal: null,
	rawConversationBtn: null
};

// ========================================
// API CLIENT INITIALIZATION
// ========================================

function initializeAPIClient() {
	const savedEndpoint = localStorage.getItem('mira-api-endpoint');

	let baseURL = savedEndpoint;
	if (!baseURL) {
		baseURL = window.location.origin;
	}

	AppState.apiClient = new window.MiraAPIClient({
		baseURL: baseURL
	});

	AppState.apiClient.auth.onAuthChange((event) => {
		if (event.type === 'logout') {
			window.location.href = '/';
		} else if (event.type === 'login') {
			console.log('Authentication successful:', event.user);
		}
	});

	AppState.apiClient.eventHandlers.onConnect.push(async () => {
		console.log('WebSocket connected');
		if (AppState.messageQueue.length > 0) {
			window.updateQueueIndicator();
		}

		// Recover response that completed while client was disconnected.
		// Brief delay: the orchestrator may still be finishing when we reconnect.
		if (AppState.responseInterrupted) {
			AppState.responseInterrupted = false;
			setTimeout(async () => {
				try {
					const history = await AppState.apiClient.history.getHistory({ limit: 1 });
					const lastMsg = history.messages?.find(m => m.role === 'assistant');
					if (lastMsg && lastMsg.content) {
						window.showResponse?.(lastMsg.content);
					}
				} catch (e) {
					console.warn('Failed to recover interrupted response:', e);
				}
			}, 2000);
		}
	});

	AppState.apiClient.eventHandlers.onDisconnect.push((event) => {
		console.log('WebSocket disconnected:', event);
		if (AppState.messageQueue.length > 0) {
			window.updateQueueIndicator();
		}
	});

	AppState.apiClient.eventHandlers.onError.push((error) => {
		console.error('WebSocket error:', error);
	});

	const globalToolListener = (data) => {
		let isToolEvent = false;
		let toolName = null;
		let phase = null;
		if (data && (data.type === 'tool' || data.type === 'tool_event')) {
			isToolEvent = true;
			toolName = data.tool_name || data.name || data.tool || null;
			phase = data.event || data.status || data.state || null;
		} else if (data && typeof data.type === 'string' && data.type.startsWith('tool_')) {
			isToolEvent = true;
			toolName = data.tool_name || data.name || data.tool || null;
			if (data.type === 'tool_detected') phase = 'detected';
			else if (data.type === 'tool_executing') phase = 'executing';
			else if (data.type === 'tool_completed') phase = 'completed';
			else if (data.type === 'tool_error') phase = 'failed';
		}

		if (!isToolEvent || !toolName) {
			return;
		}
		const normalized = window.normalizeToolPhase(phase);
		if (normalized === 'detected' || normalized === 'executing') {
			window.updateToolIndicator?.(toolName, normalized);
		} else if (normalized === 'completed') {
			window.updateToolIndicator?.(toolName, 'completed');
		} else if (normalized === 'failed' || normalized === 'error') {
			window.updateToolIndicator?.(toolName, 'error');
		}
	};
	AppState.apiClient.eventHandlers.onMessage.push(globalToolListener);

	window.miraAPI = AppState.apiClient;

	if (window.WebAuthnClient) {
		AppState.webAuthnClient = new window.WebAuthnClient(AppState.apiClient);

		const originalClearToken = AppState.apiClient.auth.clearToken;
		AppState.apiClient.auth.clearToken = function() {
			originalClearToken.call(this);
			const lastUser = AppState.webAuthnClient.getLastUser();
			if (lastUser) {
				AppState.webAuthnClient.saveLastUser(lastUser.email, false);
			}
		};
	}
}

// ========================================
// APPLICATION INITIALIZATION
// ========================================

async function initializeMira() {
	initializeAPIClient();

	try {
		await AppState.apiClient.auth.validateToken();
	} catch (e) {
		// If not authenticated, server will redirect on protected routes
	}

	if (window.SettingsManager) {
		AppState.settingsManager = new window.SettingsManager(AppState.apiClient, AppState.webAuthnClient);
		window.settingsManager = AppState.settingsManager;
	}

	if (window.initDomainKnowledge) {
		window.initDomainKnowledge(AppState.apiClient);
	}

	// Update UI indicators
	window.updateQueueIndicator();

	// Load last session response
	await window.loadLastSessionResponse();

	// Setup event listeners (from events.js)
	window.setupEventListeners();
	window.setupDelegatedEvents();
}

// Start when DOM is ready
if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', initializeMira);
} else {
	initializeMira();
}

// ========================================
// FEEDBACK FUNCTIONS
// ========================================

const Functions = {
	openFeedbackEmail: async function() {
		try {
			const timestamp = new Date().toISOString();

			let userId = 'unknown';
			if (AppState.apiClient && AppState.apiClient.auth) {
				try {
					const response = await AppState.apiClient._httpRequest('/v0/auth/session');
					if (response && response.data && response.data.user_id) {
						userId = response.data.user_id;
					}
				} catch (error) {
					console.error('Could not fetch user ID:', error);
				}
			}

			const subject = encodeURIComponent('MIRA Feedback');
			const body = encodeURIComponent(
				`Timestamp: ${timestamp}\n` +
				`User ID: ${userId}\n\n` +
				`My feedback is:\n\n\n` +
				`What I expected:\n\n\n` +
				`What happened:\n\n`
			);

			window.location.href = `mailto:taylor@rocketcitywindowcleaning.com?subject=${subject}&body=${body}`;
		} catch (error) {
			console.error('Error opening feedback email:', error);
			window.location.href = 'mailto:taylor@rocketcitywindowcleaning.com?subject=MIRA%20Feedback';
		}
	},

	collapseSegment: function() {
		if (!AppState.apiClient) {
			console.error('API client not initialized');
			return;
		}

		const responseBox = document.getElementById('response_box');
		if (!responseBox) {
			console.error('Response box not found');
			return;
		}

		// Track API result for the completion message
		let apiSucceeded = null;
		let apiError = null;

		// Fire API call immediately (don't await - animation starts in parallel)
		AppState.apiClient._httpRequest('/v0/api/actions', {
			method: 'POST',
			body: JSON.stringify({
				domain: 'continuum',
				action: 'collapse_segment',
				data: {}
			})
		}).then(response => {
			if (response && response.collapsed) {
				console.log('Segment collapsed:', response);
				apiSucceeded = true;
			} else {
				// Extract message from API error response structure
				const errorMsg = response?.error?.message || response?.message || 'No active segment to collapse';
				console.error('Collapse failed:', errorMsg);
				apiSucceeded = false;
				apiError = errorMsg;
			}
		}).catch(error => {
			console.error('Error collapsing segment:', error);
			apiSucceeded = false;
			// Extract message from thrown error structure: { response: { data: { error: { message } } } }
			apiError = error.response?.data?.error?.message
				|| error.response?.data?.message
				|| error.message
				|| 'Network error';
		});

		// Start animation immediately - don't wait for API
		collapseAnimation(responseBox, {
			scrollContainer: responseBox,  // Hide scrollbar during fall
			onFallComplete: () => {
				setTimeout(() => {
					responseBox.className = 'active';
					if (apiSucceeded === false) {
						responseBox.innerHTML = `<span style="color: #ff6b6b;">Failed to collapse segment: ${apiError}</span><br><br>Please try again.`;
					} else {
						responseBox.innerHTML = `The previous conversation segment has been collapsed as-per your request. Feel free to continue the conversation in a new direction or just come back later. MIRA will be ready when you return. <br><br><br><span style="opacity: 0.4; font-family: 'Minni', serif; letter-spacing: -1px">Dhu preeveeus kanverseishun segment haz bin kulapst eisper yor rikwest. Feel free too kuntinyoo dhu kanverseishun in u noo derekshun or just kum bak leiter. Miru wil bee redee wen yoo ritern.</span>`;
					}
				}, 500);
			}
		});
	}
};

window.Functions = Functions;
