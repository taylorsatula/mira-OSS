/**
 * UI.JS - Visual Effects & User Interface Components
 *
 * PURPOSE:
 * Manages all visual effects, animations, and UI component states for the MIRA interface.
 * This module handles the "look and feel" of the application - everything the user sees
 * moving, animating, or changing appearance in response to system events.
 *
 * RESPONSIBILITIES:
 * - Badge system (tool indicators, workflow badges, status indicators)
 * - Badge animations and state transitions (expanding labels, fade effects)
 * - Visual feedback animations (plunger launch, text dissolve/projectile)
 * - Loading screens and ASCII character animations
 * - Attachment popover UI state
 * - Queue indicator UI state
 * - Textarea auto-expansion and placeholder management
 * - Response status badge lifecycle (in-progress → complete)
 *
 * WHAT GOES HERE:
 * - New visual indicators or badges
 * - Animation sequences for user actions
 * - Visual state transitions (fade in/out, expand/collapse)
 * - Loading and progress indicators
 * - UI polish and micro-interactions
 * - Visual feedback for system events
 * - CSS-driven animations coordinated from JavaScript
 *
 * WHAT DOESN'T GO HERE:
 * - Message content handling → messaging.js
 * - Event listener setup → events.js
 * - Network communication → core.js or messaging.js
 * - History/calendar UI → history.js
 * - Business logic (keep this module focused on visual presentation)
 *
 * DEPENDENCIES:
 * - core.js: AppState, elements
 * - messaging.js: streamingState (for badge timing coordination)
 *
 * DEPENDENTS:
 * - messaging.js (calls badge/animation functions during message lifecycle)
 * - events.js (triggers animations based on user interactions)
 *
 * KEY PATTERNS:
 * - Badge state managed through CSS classes (active, deactivating, error)
 * - Timers for animation coordination (cleared on reset to prevent memory leaks)
 * - RequestAnimationFrame for smooth 60fps animations
 * - Promise-based animation sequences for complex choreography
 *
 * LOAD ORDER:
 * After core.js, before or alongside other feature modules.
 */

// ========================================
// BADGE MANAGEMENT
// ========================================

function updateToolBadge(toolName) {
	if (!elements.toolBadge) return;

	const toolIcon = elements.toolBadge.querySelector('img');

	if (toolIcon && toolName) {
		const iconName = toolName.replace('_tool', '').replace('reminder', 'reminders');
		toolIcon.src = `/assets/images/icons/toolicons/${iconName}.png`;
		toolIcon.alt = toolName;
	}

	if (toolName) {
		applyToolBadgeLabel(elements.toolBadge, toolName);
	}

	elements.toolBadge.title = toolName || 'Tool Used';
	elements.toolBadge.classList.add('active');

	clearToolIndicatorTimers();
	scheduleToolIndicatorReset(elements.toolBadge, TOOL_INDICATOR_COMPLETED_HOLD_MS);

	if (toolName) {
		toolIndicatorState.toolName = toolName;
		toolIndicatorState.phase = 'completed';
	}
}

function activateWorkflowBadge(workflowInfo) {
	console.log('Workflow badge not available, workflow info:', workflowInfo);
}

function resetBadges() {
	if (elements.toolBadge) {
		clearToolIndicatorTimers();
		elements.toolBadge.classList.remove('active', 'deactivating', 'error', 'label-expanding', 'label-visible');
		elements.toolBadge.style.removeProperty('--tool-indicator-target-width');
		const span = elements.toolBadge.querySelector('span.tool-indicator-label');
		if (span) {
			span.remove();
		}
	}

	toolIndicatorState.toolName = null;
	toolIndicatorState.phase = null;
}

window.updateToolBadge = updateToolBadge;
window.activateWorkflowBadge = activateWorkflowBadge;
window.resetBadges = resetBadges;

// Tool indicator state
const toolIndicatorState = {
	toolName: null,
	phase: null
};
let toolIndicatorTimeout = null;
let toolIndicatorDeactivateTimeout = null;
let toolIndicatorLabelRevealTimeout = null;

const TOOL_INDICATOR_COMPLETED_HOLD_MS = 5000;
const TOOL_INDICATOR_ERROR_HOLD_MS = 1200;
const TOOL_INDICATOR_DEACTIVATE_ANIMATION_MS = 900;
const TOOL_INDICATOR_LABEL_REVEAL_DELAY_MS = 260;

function clearToolIndicatorTimers() {
	if (toolIndicatorTimeout) {
		clearTimeout(toolIndicatorTimeout);
		toolIndicatorTimeout = null;
	}
	if (toolIndicatorDeactivateTimeout) {
		clearTimeout(toolIndicatorDeactivateTimeout);
		toolIndicatorDeactivateTimeout = null;
	}
	clearToolIndicatorLabelReveal();
}

function getOrCreateToolBadgeLabel(toolButton) {
	let span = toolButton.querySelector('span.tool-indicator-label');
	if (!span) {
		span = document.createElement('span');
		span.className = 'tool-indicator-label';
		const img = toolButton.querySelector('img');
		toolButton.insertBefore(span, img || null);
	}
	return span;
}

function clearToolIndicatorLabelReveal() {
	if (toolIndicatorLabelRevealTimeout) {
		clearTimeout(toolIndicatorLabelRevealTimeout);
		toolIndicatorLabelRevealTimeout = null;
	}
}

function scheduleToolIndicatorReset(toolButton, initialDelay = 0, expectedToolName = null) {
	const span = toolButton.querySelector('span.tool-indicator-label');
	toolIndicatorTimeout = setTimeout(() => {
		clearToolIndicatorLabelReveal();
		toolButton.classList.add('deactivating');
		toolButton.classList.remove('label-visible');
		toolIndicatorDeactivateTimeout = setTimeout(() => {
			toolButton.classList.remove('active', 'deactivating', 'label-expanding', 'label-visible');
			toolButton.style.removeProperty('--tool-indicator-target-width');
			if (span) {
				span.remove();
			}
			const img = toolButton.querySelector('img');
			if (img) {
				img.src = '/assets/images/icons/toolicons/default.png';
				img.alt = 'Tool';
			}
			if (!expectedToolName || toolIndicatorState.toolName === expectedToolName) {
				toolIndicatorState.toolName = null;
				toolIndicatorState.phase = null;
			}
		}, TOOL_INDICATOR_DEACTIVATE_ANIMATION_MS);
	}, initialDelay);
}

function applyToolBadgeLabel(toolButton, toolName) {
	if (!toolButton || !toolName) return;

	const span = getOrCreateToolBadgeLabel(toolButton);
	const displayName = toolName.replace('_tool', '').replace(/_/g, ' ');

	toolButton.classList.remove('label-expanding', 'label-visible');
	clearToolIndicatorLabelReveal();

	span.textContent = displayName.split(' ').map(word =>
		word.charAt(0).toUpperCase() + word.slice(1)
	).join(' ');

	span.style.display = 'inline-block';
	span.style.visibility = 'hidden';
	span.style.maxWidth = 'none';

	const targetWidth = span.scrollWidth;

	span.style.removeProperty('max-width');
	span.style.removeProperty('visibility');
	span.style.removeProperty('display');

	toolButton.style.setProperty('--tool-indicator-target-width', `${targetWidth}px`);

	span.offsetWidth;

	requestAnimationFrame(() => {
		toolButton.classList.add('label-expanding');
		toolIndicatorLabelRevealTimeout = setTimeout(() => {
			toolButton.classList.add('label-visible');
		}, TOOL_INDICATOR_LABEL_REVEAL_DELAY_MS);
	});
}

function normalizeToolPhase(phase) {
	if (!phase) return null;

	let lower = String(phase).toLowerCase();
	if (lower.startsWith('tool_')) {
		lower = lower.slice(5);
	}

	const aliases = {
		complete: 'completed',
		success: 'completed',
		succeeded: 'completed',
		fail: 'error',
		failed: 'error',
		failure: 'error',
		error: 'error',
		starting: 'detected',
		started: 'detected',
		queued: 'detected',
		running: 'executing',
		in_progress: 'executing'
	};

	return aliases[lower] || lower;
}

function updateToolIndicator(toolName, state) {
	const toolButton = elements.toolBadge;
	if (!toolButton) {
		console.warn('[ToolIndicator] No tool button element found when updating indicator.', { toolName, state, elements });
		return;
	}

	if (!toolName) {
		return;
	}

	const previousTool = toolIndicatorState.toolName;
	const previousPhase = toolIndicatorState.phase;
	const sameTool = previousTool === toolName;
	const samePhase = previousPhase === state;

	if (sameTool && samePhase) {
		return;
	}

	const transitioningToExecuting = sameTool && previousPhase === 'detected' && state === 'executing';

	if (!transitioningToExecuting) {
		clearToolIndicatorTimers();
	} else if (toolIndicatorDeactivateTimeout) {
		clearTimeout(toolIndicatorDeactivateTimeout);
		toolIndicatorDeactivateTimeout = null;
	}

	toolButton.classList.remove('error', 'deactivating');

	if (state === 'detected' || state === 'executing') {
		toolButton.classList.add('active');

		const img = toolButton.querySelector('img');
		const shouldUpdateLabel = !sameTool || state === 'detected' || !previousPhase;

		if (shouldUpdateLabel && toolName) {
			applyToolBadgeLabel(toolButton, toolName);
		}

		if (img && (!sameTool || state === 'detected' || !previousPhase)) {
			const iconName = toolName.replace('_tool', '').replace('reminder', 'reminders');
			img.src = `/assets/images/icons/toolicons/${iconName}.png`;
			img.alt = toolName;
		}

	} else if (state === 'completed') {
		toolButton.classList.add('active');
		scheduleToolIndicatorReset(toolButton, TOOL_INDICATOR_COMPLETED_HOLD_MS, toolName);

	} else if (state === 'error') {
		toolButton.classList.add('error', 'active');
		scheduleToolIndicatorReset(toolButton, TOOL_INDICATOR_ERROR_HOLD_MS, toolName);
	}

	toolIndicatorState.toolName = toolName;
	toolIndicatorState.phase = state;
}

window.updateToolIndicator = updateToolIndicator;

// ========================================
// UTILITY FUNCTIONS
// ========================================

function hapticFeedback(pattern = 100) {
	if ('vibrate' in navigator) {
		navigator.vibrate(pattern);
	}
}

// Export hapticFeedback for use in other modules
window.hapticFeedback = hapticFeedback;

// ========================================
// STATUS BADGE MANAGEMENT
// ========================================

let completionBadgeTimer = null;
let completionBadgeDelayedShowTimeout = null;
let initialStatusBadgeTimer = null;
let statusBadgeInitialized = false;
let toolbarStatusBadge = null;
let toolbarBadgeFadeTimeout = null;

const STATUS_BADGE_DECISION_DELAY_MS = 150;
const TOOLBAR_BADGE_FADE_DURATION_MS = 200;
const COMPLETE_BADGE_FADE_DELAY_MS = 100;
const COMPLETE_BADGE_FADE_DURATION_MS = 200;

function getResponseStatusBadge() {
	if (!elements.responseContent) return null;
	return elements.responseContent.querySelector('.response-status-indicator');
}

function ensureResponseStatusBadgePosition() {
	const container = elements.responseContent;
	const badge = getResponseStatusBadge();
	if (!container || !badge) return;
	container.appendChild(badge);
}

function clearCompletionBadgeTimer() {
	clearCompletionBadgeDelayedShow();
	if (completionBadgeTimer) {
		clearTimeout(completionBadgeTimer);
		completionBadgeTimer = null;
	}
}

function clearCompletionBadgeDelayedShow() {
	if (completionBadgeDelayedShowTimeout) {
		clearTimeout(completionBadgeDelayedShowTimeout);
		completionBadgeDelayedShowTimeout = null;
	}
}

function removeExistingCompletionBadge() {
	const existingBadge = getResponseStatusBadge();
	if (existingBadge) existingBadge.remove();
	statusBadgeInitialized = false;
	clearInitialStatusBadgeTimer();
	destroyToolbarStatusBadge(true);
}

const RESPONSE_STATUS_STYLES = {
	'in-progress': {
		color: '#ffd166'
	},
	complete: {
		color: 'lime'
	}
};

function createStatusBadgeElement(context = 'response') {
	const badge = document.createElement('span');
	badge.className = 'response-status-indicator';
	badge.style.backgroundColor = 'black';
	badge.style.padding = '5px 20px';
	badge.style.fontWeight = 'bold';
	badge.style.fontSize = '0.7em';
	badge.style.display = 'inline-flex';
	badge.style.alignItems = 'center';
	badge.style.position = 'relative';
	badge.style.pointerEvents = 'none';
	badge.style.whiteSpace = 'nowrap';
	badge.style.border = '1px solid transparent';

	if (context === 'response') {
		badge.dataset.context = 'response';
		badge.style.display = 'inline-block';
		badge.style.alignItems = '';
		badge.style.top = '-2px';
	} else if (context === 'toolbar') {
		badge.dataset.context = 'toolbar';
		badge.classList.add('toolbar-status-indicator');
		badge.style.top = '0';
		badge.style.alignSelf = 'center';
	}

	return badge;
}

function insertCompletionBadge(label = '/complete', status = 'complete', options = {}) {
	const container = elements.responseContent;
	if (!container) return;

	const { fade = false, fadeDuration = COMPLETE_BADGE_FADE_DURATION_MS } = options;

	let badge = getResponseStatusBadge();
	if (!badge) {
		badge = createStatusBadgeElement('response');
	}

	const style = RESPONSE_STATUS_STYLES[status] || RESPONSE_STATUS_STYLES.complete;
	badge.style.color = style.color;
	badge.style.border = `1px solid ${style.color}`;
	badge.dataset.status = status;
	badge.textContent = label;

	if (fade) {
		badge.style.transition = `opacity ${fadeDuration}ms ease`;
		badge.style.opacity = '0';
	} else {
		badge.style.transition = 'none';
		badge.style.opacity = '1';
	}

	container.appendChild(badge);

	if (fade) {
		requestAnimationFrame(() => {
			requestAnimationFrame(() => {
				badge.style.opacity = '1';
			});
		});
	}

	statusBadgeInitialized = true;
}

function clearInitialStatusBadgeTimer() {
	if (initialStatusBadgeTimer) {
		clearTimeout(initialStatusBadgeTimer);
		initialStatusBadgeTimer = null;
	}
}

function showToolbarStatusBadge(label = '/in-progress') {
	const toolbar = elements.toolbarRight;
	if (!toolbar) return;

	if (toolbarBadgeFadeTimeout) {
		clearTimeout(toolbarBadgeFadeTimeout);
		toolbarBadgeFadeTimeout = null;
	}

	if (!toolbarStatusBadge) {
		toolbarStatusBadge = createStatusBadgeElement('toolbar');
		toolbarStatusBadge.style.transition = 'none';
		const referenceNode = (elements.toolBadge && elements.toolBadge.parentElement === toolbar)
			? elements.toolBadge
			: toolbar.firstChild;
		if (referenceNode) {
			toolbar.insertBefore(toolbarStatusBadge, referenceNode);
		} else {
			toolbar.appendChild(toolbarStatusBadge);
		}
	}

	const style = RESPONSE_STATUS_STYLES['in-progress'];
	toolbarStatusBadge.textContent = label;
	toolbarStatusBadge.dataset.status = 'in-progress';
	toolbarStatusBadge.style.border = `1px solid ${style.color}`;
	toolbarStatusBadge.style.color = style.color;
	toolbarStatusBadge.style.transition = 'none';
	toolbarStatusBadge.style.opacity = '1';
}

function fadeOutToolbarStatusBadge(callback) {
	if (!toolbarStatusBadge) {
		if (typeof callback === 'function') callback();
		return;
	}

	toolbarStatusBadge.style.transition = `opacity ${TOOLBAR_BADGE_FADE_DURATION_MS}ms ease`;
	void toolbarStatusBadge.offsetWidth;
	toolbarStatusBadge.style.opacity = '0';

	toolbarBadgeFadeTimeout = setTimeout(() => {
		toolbarBadgeFadeTimeout = null;
		if (toolbarStatusBadge) {
			toolbarStatusBadge.remove();
			toolbarStatusBadge = null;
		}
		if (typeof callback === 'function') callback();
	}, TOOLBAR_BADGE_FADE_DURATION_MS);
}

function destroyToolbarStatusBadge(immediate = false) {
	if (!toolbarStatusBadge) return;
	if (toolbarBadgeFadeTimeout) {
		clearTimeout(toolbarBadgeFadeTimeout);
		toolbarBadgeFadeTimeout = null;
	}

	if (immediate) {
		toolbarStatusBadge.remove();
		toolbarStatusBadge = null;
		return;
	}

	fadeOutToolbarStatusBadge();
}

function transitionStatusToComplete() {
	const showComplete = () => insertCompletionBadge('/complete', 'complete', {
		fade: true,
		fadeDuration: COMPLETE_BADGE_FADE_DURATION_MS
	});

	clearCompletionBadgeDelayedShow();

	const scheduleShow = (delayMs) => {
		if (delayMs > 0) {
			completionBadgeDelayedShowTimeout = setTimeout(() => {
				completionBadgeDelayedShowTimeout = null;
				showComplete();
			}, delayMs);
		} else {
			showComplete();
		}
	};

	if (toolbarStatusBadge) {
		fadeOutToolbarStatusBadge(() => {
			scheduleShow(COMPLETE_BADGE_FADE_DELAY_MS);
		});
	} else {
		scheduleShow(0);
	}
}

function scheduleCompletionBadge() {
	clearCompletionBadgeTimer();
	clearInitialStatusBadgeTimer();
	const existing = getResponseStatusBadge();
	if (existing && existing.dataset.status === 'complete') {
		return;
	}
	completionBadgeTimer = setTimeout(() => {
		completionBadgeTimer = null;
		transitionStatusToComplete();
	}, 2000);
}

function scheduleInitialStatusBadge() {
	if (statusBadgeInitialized || initialStatusBadgeTimer) return;

	initialStatusBadgeTimer = setTimeout(() => {
		initialStatusBadgeTimer = null;
		if (statusBadgeInitialized) return;
		if (streamingState.streamFinished) {
			statusBadgeInitialized = true;
			return;
		}
		statusBadgeInitialized = true;
		showToolbarStatusBadge('/in-progress');
	}, STATUS_BADGE_DECISION_DELAY_MS);
}

// ========================================
// ATTACHMENT POPOVER
// ========================================

function updateAttachmentPopover() {
	if (AppState.attachedFiles.length === 0) {
		elements.attachmentList.innerHTML = '';
		return;
	}

	elements.attachmentList.innerHTML = '';

	AppState.attachedFiles.forEach((file, index) => {
		const isImage = file.type.startsWith('image/');

		if (isImage) {
			const reader = new FileReader();
			reader.onload = (e) => {
				const item = document.createElement('div');
				item.className = 'attachment-item';
				item.innerHTML = `
					<img src="${e.target.result}" alt="Preview">
					<span class="attachment-item-name">${file.name}</span>
					<button type="button" class="attachment-item-remove" data-action="remove" data-index="${index}">×</button>
				`;

				elements.attachmentList.appendChild(item);

				item.querySelector('[data-action="remove"]').addEventListener('click', (e) => {
					e.stopPropagation();
					const idx = parseInt(e.target.dataset.index);
					clearAttachment(idx);
				});
			};
			reader.readAsDataURL(file);
		} else {
			const item = document.createElement('div');
			item.className = 'attachment-item';
			const fileIcon = getFileIcon(file.type);
			item.innerHTML = `
				<span class="attachment-file-icon">${fileIcon}</span>
				<span class="attachment-item-name">${file.name}</span>
				<button type="button" class="attachment-item-remove" data-action="remove" data-index="${index}">×</button>
			`;

			elements.attachmentList.appendChild(item);

			item.querySelector('[data-action="remove"]').addEventListener('click', (e) => {
				e.stopPropagation();
				const idx = parseInt(e.target.dataset.index);
				clearAttachment(idx);
			});
		}
	});
}

function getFileIcon(mimeType) {
	if (mimeType === 'application/pdf') return '📄';
	if (mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') return '📝';
	if (mimeType === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') return '📊';
	if (mimeType === 'text/csv') return '📊';
	if (mimeType === 'text/plain') return '📝';
	return '📎';
}

function toggleAttachmentPopover() {
	if (!elements.attachmentPopover) return;
	elements.attachmentPopover.classList.toggle('active');
}

function closeAttachmentPopover() {
	if (!elements.attachmentPopover) return;
	elements.attachmentPopover.classList.remove('active');
}

function updateAttachmentCount() {
	if (!elements.attachmentButton) return;

	const existingSpan = elements.attachmentButton.querySelector('span');
	if (existingSpan) {
		existingSpan.remove();
	}

	const count = AppState.attachedFiles.length;

	if (count > 0) {
		const countSpan = document.createElement('span');
		countSpan.textContent = count.toString();
		elements.attachmentButton.appendChild(countSpan);
	}
}

// ========================================
// QUEUE INDICATOR & POPOVER
// ========================================

function updateQueueIndicator() {
	if (!elements.queueIndicator) return;
	if (AppState.messageQueue.length > 0) {
		elements.queueIndicator.classList.remove('hidden');
		if (elements.queueCount) {
			elements.queueCount.textContent = `Retry Send: ${AppState.messageQueue.length}`;
		}
	} else {
		elements.queueIndicator.classList.add('hidden');
		if (elements.queuePopover) {
			elements.queuePopover.classList.remove('active');
		}
	}
}

function toggleQueuePopover() {
	if (!elements.queuePopover) return;
	window.hapticFeedback(100);
	const isActive = elements.queuePopover.classList.contains('active');
	if (!isActive) renderQueuedMessages();
	elements.queuePopover.classList.toggle('active');
}

// ========================================
// ANIMATIONS
// ========================================

// Handle returning from a hidden/background tab.
document.addEventListener('visibilitychange', () => {
	if (document.hidden) return;

	// Snap stale CSS keyframe animations to end state. Browsers pause/defer
	// animations in background tabs; without this, returning causes a burst
	// of simultaneous animations or silently skipped entrance effects.
	for (const anim of document.getAnimations()) {
		if (!(anim instanceof CSSAnimation)) continue;
		const timing = anim.effect?.getComputedTiming();
		if (timing && isFinite(timing.endTime)) anim.finish();
	}

	// If WebSocket died while the tab was hidden, reconnect immediately
	// instead of waiting for the exponential backoff timer.
	if (window.miraAPI && window.miraAPI.connectionState === 'disconnected') {
		window.miraAPI.reconnectAttempts = 0;
		window.miraAPI.connect();
	}
});

async function runPlungerAnimation(messageText) {
	const totalDuration = 1700;
	const pullbackDistance = 7;
	const launchDistance = -2;

	// Run text dissolve first to clear textarea before projectile appears
	await animateTextDissolve(totalDuration);

	// Then run plunger and projectile in parallel
	const animationPromise = animatePlunger(pullbackDistance, launchDistance, totalDuration);
	const projectilePromise = animateTextProjectile(messageText, totalDuration);

	await Promise.all([animationPromise, projectilePromise]);
}

function computeImpactDelay() {
	const inputRect = elements.inputContainer.getBoundingClientRect();
	const responseRect = elements.responseBox.getBoundingClientRect();
	const distance = Math.max(0, inputRect.top - responseRect.top);
	// Projectile velocity: 150px / 544ms ≈ 0.276 px/ms
	// Air resistance: slight drag proportional to distance
	const effectiveSpeed = Math.max(0.276 - (distance * 0.0003), 0.1);
	return Math.round(Math.max(80, Math.min(distance / effectiveSpeed, 200)));
}

window.computeImpactDelay = computeImpactDelay;

function animateTextDissolve(totalDuration) {
	return new Promise((resolve) => {
		const startTime = performance.now();
		const duration = totalDuration * 0.10;

		const animate = (currentTime) => {
			const elapsed = currentTime - startTime;
			const progress = Math.min(elapsed / duration, 1);

			const alpha = 1 - progress;
			elements.messageInput.style.color = `rgba(255, 255, 255, ${alpha})`;

			if (progress < 1) {
				requestAnimationFrame(animate);
			} else {
				elements.messageInput.value = '';
				elements.messageInput.style.color = '';
				elements.messageInput.style.height = '';
				elements.messageInput.focus();
				resolve();
			}
		};

		requestAnimationFrame(animate);
	});
}

function animateTextProjectile(messageText, totalDuration) {
	return new Promise((resolve) => {
		const projectile = document.createElement('div');
		projectile.className = 'message-projectile';
		projectile.textContent = messageText;

		elements.inputContainer.insertBefore(projectile, elements.inputContainer.firstChild);

		const strokeDistance = 150;
		const startTime = performance.now();
		const duration = totalDuration * 0.5;
		const launchStart = totalDuration * 0.18;

		const animate = (currentTime) => {
			const elapsed = currentTime - startTime;
			const progress = Math.min(elapsed / duration, 1);

			let distance = 0;
			let opacity = 0;

			if (elapsed < launchStart) {
				distance = 0;
				opacity = 0;
			} else {
				const activeElapsed = elapsed - launchStart;
				const activeDuration = duration - launchStart;
				const activeProgress = Math.min(activeElapsed / activeDuration, 1);

				const eased = 1 - Math.pow(1 - activeProgress, 2.5);
				distance = eased * strokeDistance;
				opacity = 1 - activeProgress;
			}

			projectile.style.transform = `translateY(-${distance}px)`;
			projectile.style.opacity = opacity;

			if (progress < 1) {
				requestAnimationFrame(animate);
			} else {
				if (projectile.parentNode) {
					projectile.parentNode.removeChild(projectile);
				}
				resolve();
			}
		};

		requestAnimationFrame(animate);
	});
}

function animatePlunger(pullbackDistance, launchDistance, totalDuration) {
	return new Promise((resolve) => {
		const startTime = performance.now();

		const pullbackEnd = 0.12;
		const launchEnd = 0.40;
		const returnEnd = 0.55;
		const overshootEnd = 0.60;
		const settleEnd = 0.65;

		// Synchronized haptic thumps at key animation moments
		// First thump: when plunger reaches highest point (pullback complete)
		setTimeout(() => window.hapticFeedback?.(50), totalDuration * pullbackEnd);
		// Second thump: when plunger hits lowest point (launch complete)
		setTimeout(() => window.hapticFeedback?.(50), totalDuration * launchEnd);

		const animate = (currentTime) => {
			const elapsed = currentTime - startTime;
			const progress = Math.min(elapsed / totalDuration, 1);

			let position = 0;

			if (progress <= pullbackEnd) {
				const phaseProgress = progress / pullbackEnd;
				const eased = 1 - Math.pow(1 - phaseProgress, 2);
				position = eased * pullbackDistance;

			} else if (progress <= launchEnd) {
				const phaseProgress = (progress - pullbackEnd) / (launchEnd - pullbackEnd);
				const eased = 1 - Math.exp(-8 * phaseProgress);
				position = pullbackDistance + eased * (launchDistance - pullbackDistance);

			} else if (progress <= returnEnd) {
				const phaseProgress = (progress - launchEnd) / (returnEnd - launchEnd);
				const eased = phaseProgress * phaseProgress;
				position = launchDistance + eased * (0 - launchDistance);

			} else if (progress <= overshootEnd) {
				const phaseProgress = (progress - returnEnd) / (overshootEnd - returnEnd);
				const eased = Math.sin(phaseProgress * Math.PI / 2);
				position = eased * 1;

			} else if (progress <= settleEnd) {
				const phaseProgress = (progress - overshootEnd) / (settleEnd - overshootEnd);
				const eased = 1 - Math.pow(1 - phaseProgress, 2);
				position = 1 - eased;
			} else {
				position = 0;
			}

			elements.inputContainer.style.transform = `translateY(${position}px)`;

			if (progress < settleEnd) {
				requestAnimationFrame(animate);
			} else {
				elements.inputContainer.style.transform = '';
				resolve();
			}
		};

		requestAnimationFrame(animate);
	});
}

const waveChars = [
	'\u223F', '\u223E', '~', '\u222B', '\u03BB', '\u03C8', '\u03C9',
	'/', '\\', '|', '\u2219', '\u25E6', '\u25D8', 'v', '^',
	'a', 'e', 'i', 'o', 'u', 'm', 'n', 's', 'r', 'h',
	'A', 'E', 'I', 'O', 'U',
	'0', '3', '8'
];
const WAVE_GAP_CHAR = '\u00A0';
const WAVE_VISIBLE_SLOT_RATIO = 0.2;
const WAVE_ENTRY_DURATION_MS = 300;
const WAVE_AMPLITUDE_PX = 12;
const WAVE_SPEED = 'fast';
const WAVE_PHASE_SPEED_SCALE = 0.12;

function computeWaveFrequency() {
	const norm = (Math.random() + Math.random() + Math.random()) / 3;
	return 10 + norm * 40;
}

function phaseDeltaForElapsedSeconds(hz, elapsedSeconds, speedMultiplier = 1) {
	return hz * speedMultiplier * WAVE_PHASE_SPEED_SCALE * Math.PI * 2 * elapsedSeconds;
}

function pickWaveChar() {
	return waveChars[Math.floor(Math.random() * waveChars.length)];
}

function cancelLoadingAnimationFrame() {
	if (elements.loadingScreen._animFrame) {
		cancelAnimationFrame(elements.loadingScreen._animFrame);
		delete elements.loadingScreen._animFrame;
	}
}

function showLoadingScreen(callback = () => {}) {
	if (elements.loadingScreen._stopLoading) {
		elements.loadingScreen._stopLoading({ immediate: true, runCallback: false });
	}
	cancelLoadingAnimationFrame();

	const waveState = prepareWaveChars();
	const runId = Symbol('loading-wave');
	elements.loadingScreen._loadingRunId = runId;
	elements.loadingScreen.classList.add('active');

	const fallbackTimeout = setTimeout(() => {
		console.warn('Loading animation timeout reached, showing slow-response message');
		cancelLoadingAnimationFrame();
		elements.asciiContainer.innerHTML =
			'<p style="color: var(--color-text-secondary, #aaa); font-size: 0.9rem; text-align: center; padding: 2rem; line-height: 1.6;">' +
			'Mira is running slow or generating a super long response. ' +
			'The request has <em>not</em> timed out yet. If it doesn\'t show up in a moment ' +
			'please try refreshing the page (sometimes the UI gets stuck).</p>';
	}, 45000);

	initWaveAnimation(waveState);

	elements.loadingScreen._stopLoading = ({ immediate = false, runCallback = true } = {}) => {
		clearTimeout(fallbackTimeout);
		cancelLoadingAnimationFrame();
		despawnWaveChars(waveState.slots);

		if (runCallback) {
			callback();
		}

		const finish = () => {
			if (elements.loadingScreen._loadingRunId !== runId) return;
			elements.loadingScreen.classList.remove('active');
			elements.asciiContainer.innerHTML = '';
			delete elements.loadingScreen._loadingRunId;
			delete elements.loadingScreen._stopLoading;
		};

		if (immediate) {
			finish();
		} else {
			setTimeout(finish, 500);
		}
	};
}

function prepareWaveChars() {
	elements.asciiContainer.innerHTML = '';
	const slotCount = Math.max(24, Math.min(Math.floor(window.innerWidth / 12), 96));
	const slots = [];
	const visibleSlots = [];

	for (let i = 0; i < slotCount; i++) {
		const span = document.createElement('span');
		span.className = 'ascii-char';
		span.style.opacity = '0';

		const isVisibleSlot = i === 0 || i === slotCount - 1 || Math.random() < WAVE_VISIBLE_SLOT_RATIO;
		if (isVisibleSlot) {
			span.textContent = pickWaveChar();
			span.dataset.waveSlot = 'visible';
			span.dataset.visibleIndex = String(visibleSlots.length);
			visibleSlots.push(span);
		} else {
			span.textContent = WAVE_GAP_CHAR;
			span.dataset.waveSlot = 'gap';
		}

		elements.asciiContainer.appendChild(span);
		slots.push(span);
	}

	return {
		slots,
		visibleSlots,
		phase: 0,
		hz: computeWaveFrequency(),
		speedMultiplier: WAVE_SPEED === 'fast' ? 0.7 : 0.25
	};
}

function initWaveAnimation(waveState) {
	let startTime = null;
	let lastTime = null;
	const slotCount = waveState.slots.length;
	const visibleCount = waveState.visibleSlots.length;
	const horizontalCycles = 1 + ((waveState.hz - 10) / 40) * 2.5;

	function animate(now) {
		if (startTime === null) {
			startTime = now;
			lastTime = now;
		}
		const elapsed = now - startTime;
		const elapsedSeconds = (now - lastTime) / 1000;
		lastTime = now;

		const entryProgress = Math.min(1, elapsed / WAVE_ENTRY_DURATION_MS);
		if (entryProgress >= 1) {
			waveState.phase += phaseDeltaForElapsedSeconds(waveState.hz, elapsedSeconds, waveState.speedMultiplier);
		}

		for (let i = 0; i < slotCount; i++) {
			const char = waveState.slots[i];
			if (char.dataset.waveSlot !== 'visible') {
				char.style.transform = 'none';
				char.style.opacity = '0';
				continue;
			}

			const x = slotCount > 1 ? i / (slotCount - 1) : 0;
			const y1 = Math.sin(waveState.phase + x * Math.PI * 2 * horizontalCycles) * WAVE_AMPLITUDE_PX;
			const y2 = Math.sin(waveState.phase * 0.65 + x * Math.PI * 2 * horizontalCycles * 1.8) * WAVE_AMPLITUDE_PX * 0.25;
			const offset = y1 + y2;

			const visibleIndex = Number(char.dataset.visibleIndex);
			const revealDelay = visibleCount > 1
				? (visibleIndex / (visibleCount - 1)) * WAVE_ENTRY_DURATION_MS * 0.65
				: 0;
			const revealDuration = WAVE_ENTRY_DURATION_MS * 0.35;
			const revealProgress = Math.max(0, Math.min(1, (elapsed - revealDelay) / revealDuration));
			const crestStrength = Math.min(1, Math.abs(y1) / WAVE_AMPLITUDE_PX);

			char.style.transform = `translateY(${offset.toFixed(2)}px)`;
			char.style.opacity = (revealProgress * (0.55 + crestStrength * 0.45)).toFixed(3);

			if (entryProgress >= 1 && crestStrength > 0.92 && Math.random() < 0.025) {
				char.textContent = pickWaveChar();
			}
		}

		elements.loadingScreen._animFrame = requestAnimationFrame(animate);
	}

	elements.loadingScreen._animFrame = requestAnimationFrame(animate);
}

function despawnWaveChars(slots) {
	slots.forEach(char => {
		char.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
		char.style.opacity = '0';
		char.style.transform = 'translateY(0)';
	});
}
