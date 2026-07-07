/**
 * EVENTS.JS - Event Handlers & User Interaction
 *
 * PURPOSE:
 * Centralizes all DOM event handling and user interaction logic. This module wires up the
 * interface, connecting user actions (clicks, touches, keyboard input, paste) to the appropriate
 * application functions. It's the "glue" between the UI and the business logic.
 *
 * RESPONSIBILITIES:
 * - Event listener registration (setupEventListeners, setupDelegatedEvents)
 * - Touch/gesture handling
 * - Click handlers (buttons, popovers, outside clicks)
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 * - Paste event handling (image pasting into compose field)
 * - Form submission handling
 * - Delegated event handling for dynamic content (queue messages)
 * - UI toggle functions (history, calendar, popovers)
 * - Utility functions (mobile detection, haptic feedback)
 * - Settings updates (API endpoint changes)
 *
 * WHAT GOES HERE:
 * - New event listeners for new UI elements
 * - New gesture recognizers (long press, pinch, double-tap)
 * - Keyboard shortcut expansions
 * - Click-outside detection for new popovers/modals
 * - Drag-and-drop event handling
 * - New utility functions for event-related concerns
 *
 * WHAT DOESN'T GO HERE:
 * - Message content processing → messaging.js
 * - Visual effects and animations → ui.js
 * - API calls → messaging.js or history.js
 * - State initialization → core.js
 * - Business logic (events trigger actions, don't implement them)
 *
 * DEPENDENCIES:
 * - core.js: AppState, elements
 * - messaging.js: sendMessage, handleFileSelect, queue functions
 * - ui.js: updateAttachmentPopover, updateAttachmentCount, haptic feedback
 * - history.js: toggleHistory (if history drawer exists)
 *
 * DEPENDENTS:
 * - None (events are the leaf nodes of the dependency graph)
 *
 * KEY PATTERNS:
 * - Event delegation for dynamic content (single listener on parent)
 * - Passive event listeners for scrolling performance
 * - StopPropagation to prevent bubbling where needed
 * - Feature detection (touch support, clipboard API availability)
 *
 * LOAD ORDER:
 * Last (after core.js, ui.js, messaging.js, history.js) since it calls into all of them.
 */

// ========================================
// UTILITY FUNCTIONS
// ========================================

function isMobile() {
	const isSmallScreen = window.innerWidth <= 768;
	const isTouchFirst = window.matchMedia('(pointer: coarse)').matches;
	return isSmallScreen || isTouchFirst;
}

// ========================================
// UI TOGGLE FUNCTIONS
// ========================================

function toggleHistory() {
	console.log('History drawer not available in simplified UI');
}

async function toggleCalendar() {
	console.log('Calendar not available in simplified UI');
}

function toggleWorkflowPopover() {
	console.log('Workflow popover not available in simplified UI');
}

function toggleQueuePopover() {
	if (!elements.queuePopover) return;
	window.hapticFeedback(100);
	const isActive = elements.queuePopover.classList.contains('active');
	if (!isActive) window.renderQueuedMessages();
	elements.queuePopover.classList.toggle('active');
}

function toggleAttachmentPopover() {
	if (!elements.attachmentPopover) return;
	elements.attachmentPopover.classList.toggle('active');
}

function closeAttachmentPopover() {
	if (!elements.attachmentPopover) return;
	elements.attachmentPopover.classList.remove('active');
}

// ========================================
// SETTINGS MANAGEMENT
// ========================================

function updateAPIEndpoint() {
	const endpointInput = document.getElementById('api-endpoint');
	if (endpointInput && AppState.apiClient) {
		AppState.apiClient.baseURL = endpointInput.value;
		localStorage.setItem('mira-api-endpoint', endpointInput.value);
	}
}

// ========================================
// EVENT HANDLING
// ========================================

function handleClickOutside(e) {
	const queueContainer = document.querySelector('.queue-container');
	if (queueContainer && elements.queuePopover &&
		!queueContainer.contains(e.target) &&
		!elements.queuePopover.contains(e.target)) {
		elements.queuePopover.classList.remove('active');
	}
}

// ========================================
// EVENT LISTENER SETUP
// ========================================

function setupEventListeners() {
	document.addEventListener('click', handleClickOutside);

	elements.inputContainer.addEventListener('submit', (e) => {
		e.preventDefault();
		if (elements.sendButton.classList.contains('stop-mode')) {
			window.cancelGeneration();
		} else if (!elements.sendButton.disabled) {
			window.sendMessage();
		}
	});

	// iOS Safari: tapping the send button while the textarea is focused steals focus,
	// which dismisses the virtual keyboard and resizes the visual viewport mid-tap. The
	// synthetic click then lands at the original coordinates — where the button no longer
	// is — and the tap is lost. Preventing pointerdown's default keeps focus on the
	// textarea, the keyboard stays open, and the click fires on the intended target.
	elements.sendButton.addEventListener('pointerdown', (e) => {
		e.preventDefault();
	});

	elements.sendButton.addEventListener('click', (e) => {
		e.preventDefault();
		if (elements.sendButton.classList.contains('stop-mode')) {
			window.cancelGeneration();
		} else if (!elements.sendButton.disabled) {
			window.sendMessage();
		}
	});

	elements.messageInput.addEventListener('keydown', (e) => {
		// On mobile, Enter adds newline (send via button); on desktop, Enter sends (Shift+Enter for newline)
		if (e.key === 'Enter' && !e.shiftKey && !isMobile()) {
			e.preventDefault();
			if (!elements.sendButton.disabled) window.sendMessage();
		}
	});

	if (elements.attachmentButton) {
		elements.attachmentButton.addEventListener('click', (e) => {
			e.stopPropagation();
			if (AppState.attachedFiles.length > 0) {
				window.toggleAttachmentPopover();
			} else {
				elements.fileInput.click();
			}
		});
	}

	if (elements.attachmentAddBtn) {
		elements.attachmentAddBtn.addEventListener('click', () => {
			elements.fileInput.click();
		});
	}

	elements.fileInput.addEventListener('change', window.handleFileSelect);

	const attachmentPopoverClose = elements.attachmentPopover?.querySelector('.popover-close');
	if (attachmentPopoverClose) {
		attachmentPopoverClose.addEventListener('click', (e) => {
			e.stopPropagation();
			window.closeAttachmentPopover();
		});
	}

	document.addEventListener('click', (e) => {
		if (elements.attachmentPopover &&
			elements.attachmentPopover.classList.contains('active') &&
			!elements.attachmentPopover.contains(e.target) &&
			!elements.attachmentButton.contains(e.target)) {
			window.closeAttachmentPopover();
		}
	});

	document.addEventListener('paste', async (e) => {
		const items = e.clipboardData?.items;
		if (!items) return;

		for (const item of items) {
			if (item.type.startsWith('image/')) {
				e.preventDefault();
				const file = item.getAsFile();
				if (file) {
					const maxSize = 5 * 1024 * 1024;
					if (file.size > maxSize) {
						alert('Image must be less than 5MB');
						return;
					}

					try {
						const resizedFile = await window.downscaleImage(file, 540, 120, 1080, 120);

						AppState.attachedFiles.push(resizedFile);

						window.updateAttachmentPopover();
						elements.attachmentButton.classList.add('has-attachment');
						window.updateAttachmentCount();
					} catch (error) {
						console.error('Failed to process pasted image:', error);
						alert('Failed to process pasted image. Please try again.');
					}
				}
				break;
			}
		}
	});

	if (elements.queueIndicator) {
		elements.queueIndicator.addEventListener('click', (e) => {
			e.stopPropagation();
			toggleQueuePopover();
		});
	}

}

function setupDelegatedEvents() {
	if (elements.queueMessages) {
		elements.queueMessages.addEventListener('click', async (e) => {
			const index = parseInt(e.target.dataset.index);
			if (e.target.classList.contains('queue-send-btn')) {
				await window.sendSingleQueuedMessage(index);
			} else if (e.target.classList.contains('queue-remove-btn')) {
				await window.removeQueuedMessage(index);
			} else if (e.target.classList.contains('queue-edit-btn')) {
				window.editQueuedMessage(index);
			} else if (e.target.classList.contains('queue-save-btn')) {
				window.saveQueuedMessageEdit(index);
			} else if (e.target.classList.contains('queue-cancel-btn')) {
				window.cancelQueuedMessageEdit();
			}
		});
	}

	document.querySelectorAll('.popover-close').forEach(btn => {
		btn.addEventListener('click', (e) => {
			e.stopPropagation();
			const popover = btn.closest('.retry-popover');
			if (popover) {
				popover.classList.remove('active');
			}
		});
	});
}

// ========================================
// MODULE EXPORTS
// ========================================

// Export setup functions needed by core.js
window.setupEventListeners = setupEventListeners;
window.setupDelegatedEvents = setupDelegatedEvents;
