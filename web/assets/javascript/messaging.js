/**
 * MESSAGING.JS - Message Handling & Response Rendering
 *
 * PURPOSE:
 * Orchestrates the complete message lifecycle from user input to rendered response. This module
 * handles message sending, streaming response processing, content rendering, and the message queue
 * for failed sends. It's the central nervous system for all communication with the backend.
 *
 * RESPONSIBILITIES:
 * - Message queue (persist failed messages, allow retry/edit/remove)
 * - File attachment handling (image downscaling, base64 conversion)
 * - Message sending with error handling and retry logic
 * - Streaming response management (chunked content accumulation)
 * - Response rendering pipeline (markdown → HTML, sanitization)
 * - Block-based streaming display (paragraphs and code fences)
 * - System tag filtering (<think>, <mira:*> tags)
 * - Code block copy-to-clipboard functionality
 * - Session restoration (load last assistant message on page load)
 * - Emotion emoji extraction and thinking indicator management
 * - Response container lifecycle (show/hide, animations, scroll control)
 *
 * WHAT GOES HERE:
 * - New message types or formats
 * - Additional content filtering or sanitization rules
 * - Response rendering enhancements (new markdown features, media types)
 * - Message queue operations (bulk actions, priority handling)
 * - Streaming protocol changes or optimizations
 * - New attachment types beyond images
 *
 * WHAT DOESN'T GO HERE:
 * - WebSocket connection management → core.js (API client initialization)
 * - Visual animations for messages → ui.js
 * - Event handlers for send button → events.js
 * - Tool/workflow badge updates → ui.js
 *
 * DEPENDENCIES:
 * - core.js: AppState, elements
 * - ui.js: badge functions, animation functions (runPlungerAnimation, showLoadingScreen)
 * - External: marked.js (markdown parser), DOMPurify (HTML sanitizer)
 *
 * DEPENDENTS:
 * - events.js (calls sendMessage from event handlers)
 * - core.js (calls loadLastSessionResponse during initialization)
 *
 * KEY PATTERNS:
 * - Streaming state machine (buffer → extract blocks → render → repeat)
 * - Two-phase rendering (streaming incremental, then force complete on finish)
 * - Promise-based response container lifecycle management
 * - window.* exports for functions called from WebSocket event handlers
 *
 * LOAD ORDER:
 * After core.js and ui.js (depends on both).
 */

// ========================================
// MESSAGE QUEUE MANAGEMENT
// ========================================

function queueMessage(text) {
	AppState.messageQueue.push({
		text: text,
		timestamp: Date.now()
	});
	localStorage.setItem('mira-queue', JSON.stringify(AppState.messageQueue));
	updateQueueIndicator();
}

function renderQueuedMessages() {
	if (!elements.queueMessages) return;
	if (AppState.messageQueue.length === 0) {
		elements.queueMessages.innerHTML = '<div class="empty-state">No queued messages</div>';
		return;
	}

	let html = '';
	AppState.messageQueue.forEach((msg, index) => {
		const time = new Date(msg.timestamp).toLocaleString();
		html += `
			<div class="queue-message" data-index="${index}">
				<div class="queue-message-text">${msg.text}</div>
				<div class="queue-message-time">${time}</div>
				<div class="queue-message-actions">
					<button class="queue-edit-btn" data-index="${index}">Edit</button>
					<button class="queue-send-btn" data-index="${index}">Send</button>
					<button class="queue-remove-btn" data-index="${index}">Remove</button>
				</div>
			</div>
		`;
	});

	elements.queueMessages.innerHTML = html;
}

async function sendSingleQueuedMessage(index) {
	const msg = AppState.messageQueue[index];
	if (!msg) return;

	elements.messageInput.value = msg.text;
	AppState.messageQueue.splice(index, 1);
	localStorage.setItem('mira-queue', JSON.stringify(AppState.messageQueue));
	updateQueueIndicator();
	renderQueuedMessages();

	await sendMessage();
}

function removeQueuedMessage(index) {
	AppState.messageQueue.splice(index, 1);
	localStorage.setItem('mira-queue', JSON.stringify(AppState.messageQueue));
	updateQueueIndicator();
	renderQueuedMessages();
}

function editQueuedMessage(index) {
	const msg = AppState.messageQueue[index];
	if (!msg) return;

	const messageEl = elements.queueMessages.querySelector(`[data-index="${index}"]`);
	if (!messageEl) return;

	const textEl = messageEl.querySelector('.queue-message-text');
	const actionsEl = messageEl.querySelector('.queue-message-actions');

	const currentText = msg.text;

	textEl.classList.add('editing');
	textEl.innerHTML = `<textarea class="queue-edit-textarea" rows="3">${currentText}</textarea>`;
	actionsEl.innerHTML = `
		<button class="queue-save-btn" data-index="${index}">Save</button>
		<button class="queue-cancel-btn" data-index="${index}">Cancel</button>
	`;

	const textarea = textEl.querySelector('.queue-edit-textarea');
	textarea.focus();
	textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function saveQueuedMessageEdit(index) {
	const messageEl = elements.queueMessages.querySelector(`[data-index="${index}"]`);
	if (!messageEl) return;

	const textarea = messageEl.querySelector('.queue-edit-textarea');
	if (!textarea) return;

	const newText = textarea.value.trim();
	if (!newText) {
		alert('Message cannot be empty');
		return;
	}

	AppState.messageQueue[index].text = newText;
	localStorage.setItem('mira-queue', JSON.stringify(AppState.messageQueue));
	renderQueuedMessages();
}

function cancelQueuedMessageEdit() {
	renderQueuedMessages();
}

window.miraQueue = {
	add: queueMessage
};

// ========================================
// FILE ATTACHMENT HANDLING
// ========================================

async function downscaleImage(file, maxLongestAxis = 512, minShortestAxis = 64) {
	let bitmap = await createImageBitmap(file);
	const { width, height } = bitmap;

	// Already within bounds — return original (no re-encoding)
	if (Math.max(width, height) <= maxLongestAxis) {
		bitmap.close();
		return file;
	}

	// Scale longest axis to target
	let scale = maxLongestAxis / Math.max(width, height);

	// Narrow content protection: if shortest axis would collapse below floor,
	// scale to the floor instead (longest axis will exceed max)
	if (Math.min(width, height) * scale < minShortestAxis) {
		scale = minShortestAxis / Math.min(width, height);
	}

	const targetW = Math.round(width * scale);
	const targetH = Math.round(height * scale);

	// Multi-step halving — each step is ≤2:1, optimal for any resampling algorithm
	let curW = width;
	let curH = height;
	while (curW / 2 >= targetW && curH / 2 >= targetH) {
		curW = Math.round(curW / 2);
		curH = Math.round(curH / 2);
		const stepped = await createImageBitmap(bitmap, { resizeWidth: curW, resizeHeight: curH, resizeQuality: 'high' });
		bitmap.close();
		bitmap = stepped;
	}

	// Final resize to exact target
	const final = await createImageBitmap(bitmap, { resizeWidth: targetW, resizeHeight: targetH, resizeQuality: 'high' });
	bitmap.close();

	const canvas = new OffscreenCanvas(targetW, targetH);
	canvas.getContext('2d').drawImage(final, 0, 0);
	final.close();

	const mimeType = file.type || 'image/jpeg';
	const blob = await canvas.convertToBlob({ type: mimeType, quality: 0.9 });
	return new File([blob], file.name, { type: mimeType });
}

async function handleFileSelect(event) {
	const file = event.target.files[0];
	if (!file) return;

	const supportedTypes = [
		'image/jpeg', 'image/png', 'image/gif', 'image/webp',
		'application/pdf',
		'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
		'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
		'text/plain',
		'text/csv'
	];

	if (!supportedTypes.includes(file.type) && !file.type.startsWith('image/')) {
		alert('Unsupported file type. Supported: Images, PDF, DOCX, XLSX, TXT, CSV');
		return;
	}

	const isImage = file.type.startsWith('image/');
	const isText = file.type === 'text/plain' || file.type === 'text/csv';

	// Size limits: text 100KB, non-image 32MB. Images handled below per toggle.
	if (!isImage) {
		const maxSize = isText ? 100 * 1024 : 32 * 1024 * 1024;
		const maxSizeLabel = isText ? '100KB' : '32MB';
		if (file.size > maxSize) {
			alert(`File must be less than ${maxSizeLabel}`);
			return;
		}
	}

	try {
		let processedFile;

		if (isImage) {
			if (elements.preserveResolution?.checked) {
				// Preserve resolution: send original, backend compresses.
				// 5MB limit applies since we're not downscaling client-side.
				if (file.size > 5 * 1024 * 1024) {
					alert('File must be less than 5MB when preserving resolution');
					return;
				}
				processedFile = file;
			} else {
				// Default: downscale to 512px longest axis.
				// No size check — output is always well under 5MB.
				processedFile = await downscaleImage(file);
			}
		} else {
			processedFile = file;
		}

		AppState.attachedFiles.push(processedFile);

		updateAttachmentPopover();
		elements.attachmentButton.classList.add('has-attachment');
		updateAttachmentCount();

		event.target.value = '';
	} catch (error) {
		console.error('Failed to process file:', error);
		alert('Failed to process file. Please try another file.');
		event.target.value = '';
	}
}

function clearAttachment(index) {
	if (index !== undefined) {
		AppState.attachedFiles.splice(index, 1);
	} else {
		AppState.attachedFiles = [];
	}

	elements.fileInput.value = '';

	if (AppState.attachedFiles.length === 0) {
		elements.attachmentButton.classList.remove('has-attachment');
	}

	updateAttachmentPopover();
	updateAttachmentCount();
}

// ========================================
// MESSAGE SENDING
// ========================================

async function sendMessage(messageText) {
	const message = messageText || elements.messageInput.value.trim();
	if (!message) return;

	// Track user message for current exchange (paired with response when streaming completes)
	streamingState.pendingUserMessage = message;

	// Track activity — hides the status bar optimistically, refreshes from server
	window.inactivityWarningManager?.trackActivity('message_sent');

	window.hapticFeedback(100);

	if (elements.toolBadge) {
		elements.toolBadge.classList.remove('error');
	}

	// Don't clear textarea here — animateTextDissolve fades it out then clears
	if (!messageText && document.activeElement === elements.messageInput) {
		elements.messageInput.blur();
	}

	const attachedFiles = [...AppState.attachedFiles];

	if (AppState.attachedFiles.length > 0) {
		clearAttachment();
	}

	elements.sendButton.disabled = true;
	elements.inputContainer.classList.add('firing');

	resetBadges();

	// Knock gate: prepareResponseContainer awaits this before proceeding.
	// Resolved after plunger completes + impact delay, so the woosh-out
	// fires at the right moment instead of simultaneously with send.
	let resolveKnockGate;
	AppState.knockGate = new Promise((resolve) => { resolveKnockGate = resolve; });

	// Start API call immediately (don't wait for animation)
	handleSendMessage(message, attachedFiles);

	// Dissolve → plunger pullback → projectile launch (adaptive duration)
	await runPlungerAnimation(message);

	// 500ms settle pause after plunger completes
	await new Promise((resolve) => setTimeout(resolve, 500));

	if (AppState.responseActive) {
		// Compute travel delay based on actual distance between elements
		const impactDelay = computeImpactDelay();
		await new Promise((resolve) => setTimeout(resolve, impactDelay));

		// Impact: knock the previous response off screen
		elements.responseContent.classList.add('exiting');
		resolveKnockGate();

		// Crossfade loading screen in 400ms after impact (during woosh)
		scheduleLoadingScreen(400);
	} else {
		// No previous response to knock — just show loading screen
		resolveKnockGate();
		scheduleLoadingScreen();
	}

	elements.sendButton.disabled = false;
	elements.inputContainer.classList.remove('firing');
}

function prepareResponseContainer(callback) {
	// Complete the previous history entry with its assistant response
	if (streamingState.lastAssistantResponse && window.InlineHistory) {
		window.InlineHistory.completeLastEntry(streamingState.lastAssistantResponse);
		streamingState.lastAssistantResponse = null;
	}

	const doTransition = () => {
		if (AppState.responseActive) {
			const box = elements.responseBox;
			const content = elements.responseContent;

			const proceed = () => {
				box.style.display = 'none';
				content.classList.remove('exiting');
				box.classList.remove('no-scroll');
				AppState.responseActive = false;
				if (callback) callback();
			};

			if (content.classList.contains('exiting')) {
				let done = false;
				const onEnd = (e) => {
					if (e && e.animationName && !/woosh-out/.test(e.animationName)) return;
					if (done) return;
					done = true;
					content.removeEventListener('animationend', onEnd);
					proceed();
				};
				content.addEventListener('animationend', onEnd);
				setTimeout(() => {
					if (!done) {
						content.removeEventListener('animationend', onEnd);
						proceed();
					}
				}, 1100);
			} else {
				setTimeout(proceed, 50);
			}
		} else {
			if (callback) callback();
		}
	};

	// If a knock gate exists, wait for the plunger sequence to complete
	// before starting the exit transition
	if (AppState.knockGate) {
		AppState.knockGate.then(() => {
			AppState.knockGate = null;
			doTransition();
		});
	} else {
		doTransition();
	}
}

// ========================================
// STREAMING RESPONSE HANDLING
// ========================================

const streamingState = {
	buffer: '',
	inCodeFence: false,
	fenceLanguage: '',
	renderedLength: 0,
	firstBlockRendered: false,
	pendingUserMessage: null,    // User message for current streaming exchange
	lastAssistantResponse: null, // Previous assistant response (to complete history pair)
	activeAnimations: 0,
	streamFinished: false,
	currentEmotion: '🤖',
	containerReady: false,
	completionPending: false,
	pendingError: null,
	thinkingActive: false,       // true while receiving thinking tokens
	thinkingBuffer: '',          // accumulated thinking text
	thinkingElement: null,       // reference to .thinking-stream-content DOM node
	renderBlocked: false,        // true during transition sequence (text accumulates but doesn't render)
	isGenerating: false,         // true while LLM is generating (controls stop button)
	providerSwitchAlert: null    // reference to the persistent provider-switch alert element
};

let loadingScreenTimer = null;

function clearPendingLoadingScreen() {
	if (!loadingScreenTimer) return;
	clearTimeout(loadingScreenTimer);
	loadingScreenTimer = null;
}

function showLoadingScreenIfStillPending() {
	if (
		streamingState.firstBlockRendered ||
		streamingState.thinkingElement ||
		streamingState.completionPending ||
		streamingState.streamFinished
	) {
		return;
	}

	showLoadingScreen(() => {});
}

function scheduleLoadingScreen(delayMs = 0) {
	clearPendingLoadingScreen();

	if (delayMs > 0) {
		loadingScreenTimer = setTimeout(() => {
			loadingScreenTimer = null;
			showLoadingScreenIfStillPending();
		}, delayMs);
		return;
	}

	showLoadingScreenIfStillPending();
}

// ========================================
// GENERATION CANCEL
// ========================================

window.cancelGeneration = function() {
	if (!streamingState.isGenerating) return;
	window.miraAPI?.cancelGeneration();
};

function setGenerating(active) {
	streamingState.isGenerating = active;
	const btn = elements.sendButton;
	if (!btn) return;

	if (active) {
		btn.classList.add('stop-mode');
		btn.setAttribute('aria-label', 'Stop generation');
	} else {
		btn.classList.remove('stop-mode');
		btn.setAttribute('aria-label', 'Send message');
	}
}

function extractCompleteBlocks(text, state) {
	const blocks = [];
	let currentPos = 0;
	let workingText = text;

	if (!state.inCodeFence) {
		while (currentPos < workingText.length && workingText[currentPos] === '\n') {
			currentPos++;
		}
	}

	while (currentPos < workingText.length) {
		if (state.inCodeFence) {
			// Look for closing fence - must be at line start or buffer start
			const fenceEnd = workingText.indexOf('```', currentPos);

			if (fenceEnd !== -1) {
				// Verify fence is at proper line position
				const beforeFence = fenceEnd > 0 ? workingText[fenceEnd - 1] : '\n';
				if (beforeFence === '\n' || fenceEnd === 0) {
					const blockEnd = fenceEnd + 3;
					// Find newline after fence (if exists)
					const nextNewline = workingText.indexOf('\n', blockEnd);
					const actualEnd = nextNewline !== -1 ? nextNewline + 1 : blockEnd;

					blocks.push({
						type: 'code',
						content: workingText.substring(currentPos, actualEnd),
						language: state.fenceLanguage
					});
					currentPos = actualEnd;
					state.inCodeFence = false;
					state.fenceLanguage = '';
				} else {
					// Not a real fence at line start, wait for more data
					break;
				}
			} else {
				break;
			}
		} else {
			while (currentPos < workingText.length && workingText[currentPos] === '\n') {
				currentPos++;
			}
			if (currentPos >= workingText.length) break;

			const fenceMatch = workingText.substring(currentPos).match(/^```(\w*)\n/);
			if (fenceMatch) {
				state.inCodeFence = true;
				state.fenceLanguage = fenceMatch[1] || '';
				currentPos += fenceMatch[0].length;
				continue;
			}

			const paraEnd = workingText.indexOf('\n\n', currentPos);

			if (paraEnd !== -1) {
				const nextFence = workingText.indexOf('\n```', currentPos);

				if (nextFence === -1 || nextFence > paraEnd) {
					const paraContent = workingText.substring(currentPos, paraEnd).trim();
					if (paraContent.length > 0) {
						blocks.push({
							type: 'paragraph',
							content: paraContent
						});
					}
					currentPos = paraEnd + 2;
				} else {
					break;
				}
			} else {
				break;
			}
		}
	}

	return { blocks, consumedLength: currentPos };
}

function renderBlock(block) {
	let content = block.content;

	content = window.filterSystemTags(content);

	// Detect and transform tool indicator pattern [used: tool1, tool2]
	const toolMatch = content.match(/^\[used: ([^\]]+)\]\n*/);
	let toolPillHtml = '';
	if (toolMatch) {
		const tools = toolMatch[1];
		content = content.replace(toolMatch[0], '');
		toolPillHtml = `<span class="tool-indicator">${tools}</span>`;
	}

	if (!content.trim() && !toolPillHtml) return null;

	// After filterSystemTags, any remaining <mira: or </mira: are orphaned fragments
	// from tags that were split across paragraph boundaries. Skip these blocks.
	if (/<\/?mira:/i.test(content) || /<\/?think>/i.test(content)) {
		console.log('[STREAM] Skipping block with orphaned tag fragment');
		return null;
	}

	let html = content;
	if (typeof marked !== 'undefined') {
		marked.setOptions({
			breaks: true,
			gfm: true,
			headerIds: false,
			mangle: false,
			sanitize: false
		});
		html = marked.parse(content);
	}

	if (typeof DOMPurify !== 'undefined') {
		html = DOMPurify.sanitize(html, {
			ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
						   'blockquote', 'ul', 'ol', 'li', 'a', 'h1', 'h2',
						   'h3', 'h4', 'h5', 'h6', 'hr', 'table', 'thead',
						   'tbody', 'tr', 'th', 'td', 'img', 'span', 'div'],
			ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'id', 'src', 'alt',
						   'title', 'data-language', 'data-copy'],
			ALLOW_DATA_ATTR: true
		});
	}

	// Prepend tool pill if present
	if (toolPillHtml) {
		html = toolPillHtml + html;
	}

	return html;
}

function appendToResponse(html) {
	if (!html) return;

	const wrapper = document.createElement('div');
	wrapper.className = 'response-block';
	wrapper.innerHTML = html;

	streamingState.activeAnimations++;
	elements.responseBox.classList.add('no-scroll');

	// Haptic feedback when message block appears
	window.hapticFeedback?.(30);

	const onAnimEnd = (e) => {
		if (e && e.animationName && !/bubble-up/.test(e.animationName)) return;
		wrapper.removeEventListener('animationend', onAnimEnd);
		streamingState.activeAnimations = Math.max(0, streamingState.activeAnimations - 1);
		if (streamingState.streamFinished && streamingState.activeAnimations === 0) {
			elements.responseBox.classList.remove('no-scroll');
		}
	};
	wrapper.addEventListener('animationend', onAnimEnd);

	elements.responseContent.appendChild(wrapper);
	ensureResponseStatusBadgePosition();
	scheduleInitialStatusBadge();
}

function renderStreamingBlocks(options = {}) {
	const { force = false } = options;
	const pendingText = streamingState.buffer.substring(streamingState.renderedLength);
	if (!pendingText) return;

	const { blocks, consumedLength } = extractCompleteBlocks(pendingText, streamingState);

	if (blocks.length > 0) {
		console.log('[STREAM] Rendering', blocks.length, 'blocks');
		blocks.forEach(block => {
			const html = renderBlock(block);
			if (html) {
				appendToResponse(html);
				if (!streamingState.firstBlockRendered) {
					streamingState.firstBlockRendered = true;
					try { stopLoadingAnimation(); } catch (e) { /* noop */ }
				}
			}
		});
	}

	if (consumedLength > 0) {
		streamingState.renderedLength += consumedLength;
	}

	if (force) {
		const leftover = streamingState.buffer.substring(streamingState.renderedLength);
		if (leftover.trim()) {
			const fallbackBlock = {
				type: streamingState.inCodeFence ? 'code' : 'paragraph',
				content: leftover.trim()
			};
			const html = renderBlock(fallbackBlock);
			if (html) {
				appendToResponse(html);
				if (!streamingState.firstBlockRendered) {
					streamingState.firstBlockRendered = true;
					try { stopLoadingAnimation(); } catch (e) { /* noop */ }
				}
			}
			streamingState.renderedLength = streamingState.buffer.length;
			streamingState.inCodeFence = false;
			streamingState.fenceLanguage = '';
		}
	}
}

function finalizeStreamingScroll() {
	streamingState.streamFinished = true;
	elements.responseBox.classList.remove('no-scroll');
	streamingState.activeAnimations = 0;

	// Add user message to history immediately (assistant response visible in response_content)
	// Store the assistant response so we can complete the pair on next turn
	if (streamingState.pendingUserMessage && streamingState.buffer && window.InlineHistory) {
		window.InlineHistory.addUserMessage(streamingState.pendingUserMessage);
		// Store assistant response to complete the pair later
		streamingState.lastAssistantResponse = streamingState.buffer;
		streamingState.pendingUserMessage = null;
	}

	// Notify inline history of new message (shows toast if user is scrolled up)
	if (window.InlineHistory) {
		window.InlineHistory.notifyNewMessage();
	}
}

window.showStreamingResponse = function(onChunk, onComplete, onError) {
	console.log('showStreamingResponse called, AppState.responseActive:', AppState.responseActive);

	streamingState.buffer = '';
	streamingState.inCodeFence = false;
	streamingState.fenceLanguage = '';
	streamingState.renderedLength = 0;
	streamingState.firstBlockRendered = false;
	streamingState.activeAnimations = 0;
	streamingState.streamFinished = false;
	streamingState.containerReady = false;
	streamingState.completionPending = false;
	streamingState.pendingError = null;
	streamingState.thinkingActive = false;
	streamingState.thinkingBuffer = '';
	streamingState.thinkingElement = null;
	streamingState.renderBlocked = false;

	clearCompletionBadgeTimer();
	removeExistingCompletionBadge();

	prepareResponseContainer(() => {
		console.log('prepareResponseContainer callback executing');
		streamingState.containerReady = true;
		elements.responseContent.innerHTML = '';
		elements.responseBox.style.display = 'block';
		elements.responseBox.classList.add('no-scroll');
		AppState.responseActive = true;
		console.log('Response box should now be visible');

		const originalOnChunk = onChunk;
		const originalOnComplete = onComplete;
		const originalOnError = onError;
		window._streamHandlers = {
			onChunk: (data) => {
				if (data.content) {
					streamingState.buffer += data.content;
					if (!streamingState.renderBlocked) {
						renderStreamingBlocks();
					}
				}

				if (originalOnChunk) originalOnChunk(data);
			},
			onComplete: () => {
				finalizeStreamingScroll();

				// Emotion is now handled via metadata in the completion handler

				if (originalOnComplete) originalOnComplete();
			},
			onError: (err) => {
				finalizeStreamingScroll();
				if (originalOnError) originalOnError(err);
			}
		};

		if (streamingState.thinkingActive && streamingState.thinkingBuffer) {
			renderThinkingBuffer();
		}

		if (streamingState.buffer && !streamingState.thinkingActive && !streamingState.renderBlocked) {
			renderStreamingBlocks();
		}

		if (streamingState.pendingError) {
			const pendingError = streamingState.pendingError;
			streamingState.pendingError = null;
			window._streamHandlers.onError(pendingError);
			return;
		}

		if (streamingState.completionPending) {
			const handlers = window._streamHandlers;
			streamingState.completionPending = false;
			window.completeStreamingResponse(streamingState.buffer);
			if (handlers?.onComplete) {
				handlers.onComplete();
			}
		}
	});
}

window.updateStreamingResponse = function(text) {
	// No-op for backward compatibility
}

window.completeStreamingResponse = function(text) {
	// Emotion is now handled via metadata in the completion handler

	// If still in thinking phase (no text arrived), transition now
	if (streamingState.thinkingActive) {
		transitionThinkingToText();
	}
	streamingState.renderBlocked = false;  // Ensure gate is open for force-render

	renderStreamingBlocks({ force: true });

	addCodeBlockButtons();

	stopLoadingAnimation();
	finalizeStreamingScroll();
	scheduleCompletionBadge();
	elements.responseBox.style.opacity = '';
	elements.responseBox.style.transform = '';
	elements.responseBox.style.filter = '';
	elements.responseBox.style.animation = '';

	delete window._streamHandlers;

	streamingState.buffer = '';
	streamingState.inCodeFence = false;
	streamingState.fenceLanguage = '';
	streamingState.renderedLength = 0;
	streamingState.firstBlockRendered = false;
	streamingState.activeAnimations = 0;
	streamingState.streamFinished = false;
	streamingState.containerReady = false;
	streamingState.completionPending = false;
	streamingState.pendingError = null;
	streamingState.thinkingActive = false;
	streamingState.thinkingBuffer = '';
	streamingState.thinkingElement = null;
	streamingState.renderBlocked = false;
}

function hasStreamingTextContent() {
	if (streamingState.buffer.trim()) return true;
	if (streamingState.renderedLength > 0) return true;
	return Boolean(elements.responseContent?.textContent?.trim());
}

function finalizeBufferedStreamingResponse(text) {
	if (window._streamHandlers) {
		window.completeStreamingResponse(text);
		return;
	}

	streamingState.buffer = text;
	streamingState.completionPending = true;
}

window.stopLoadingAnimation = function() {
	clearPendingLoadingScreen();
	if (elements.loadingScreen._stopLoading) {
		elements.loadingScreen._stopLoading();
	}
}

window.extractEmotionEmoji = function(text) {
	if (!text) return null;

	const emotionMatch = text.match(/<mira:my_emotion>\s*([^\s<]+)\s*<\/mira:my_emotion>/i);
	return emotionMatch ? emotionMatch[1] : null;
};

window.updateThinkingIndicator = function(isThinking, emotionOverride = null) {
	const thinkingIndicator = document.getElementById('thinking-indicator');
	if (!thinkingIndicator) return;

	// Preserve the existing indicator-label (contains tier model name like "Balanced")
	const existingLabel = thinkingIndicator.querySelector('.indicator-label');
	const labelText = existingLabel ? existingLabel.textContent : '';

	if (isThinking) {
		thinkingIndicator.classList.remove('standby');
		thinkingIndicator.classList.add('active');
		// Restore thinking icon during active thinking
		thinkingIndicator.innerHTML = '<img src="../assets/images/icons/think.png">';
	} else {
		thinkingIndicator.classList.remove('active');
		thinkingIndicator.classList.add('standby');
		// Display emotion emoji after response completes
		const emoji = emotionOverride || streamingState.currentEmotion;
		thinkingIndicator.innerHTML = `<span class="emotion-emoji" style="padding-left: 0">${emoji}</span>`;
	}

	// Restore the indicator-label span with tier model name
	if (labelText) {
		const label = document.createElement('span');
		label.className = 'indicator-label';
		label.textContent = labelText;
		thinkingIndicator.appendChild(label);
	}
};

/**
 * Detect JSON content-block arrays (tool_use, text, tool_result) stored as
 * stringified JSON in the messages table and convert them into readable text.
 * Returns the original string unchanged if it isn't a content-block array.
 *
 * Optional toolResultMessages: adjacent role="tool" messages whose content
 * should be rendered as the tool output section.
 *
 * Frontend-only — never mutates the DB or segment cache.
 */
window.formatContentBlocks = function(content, toolResultMessages) {
	if (!content || typeof content !== 'string') return content;

	// Quick guard: content blocks are always a JSON array
	const trimmed = content.trimStart();
	if (trimmed[0] !== '[') return content;

	let blocks;
	try {
		blocks = JSON.parse(trimmed);
	} catch {
		return content;
	}

	if (!Array.isArray(blocks)) return content;

	// Ordered sequence preserving interleaving of text and tool blocks
	const sequence = []; // {type: 'text', text} | {type: 'tool', name, inputEntries, result}
	let toolIndex = 0;

	for (const block of blocks) {
		if (!block || typeof block !== 'object') continue;
		const type = block.type;

		if (type === 'text' && block.text) {
			sequence.push({ type: 'text', text: block.text });
		} else if (type === 'tool_use' && block.name) {
			const inputEntries = block.input && typeof block.input === 'object'
				? Object.entries(block.input).map(([k, v]) =>
					[k, typeof v === 'string' ? v : JSON.stringify(v)]
				)
				: [];
			sequence.push({ type: 'tool', name: block.name, inputEntries, result: null, _idx: toolIndex++ });
		}
		// Skip thinking / redacted_thinking / image / document / tool_result blocks
	}

	// Merge in adjacent role="tool" messages (matched by position to tool_use order)
	if (toolResultMessages && toolResultMessages.length) {
		const toolItems = sequence.filter(s => s.type === 'tool');
		for (let i = 0; i < toolResultMessages.length && i < toolItems.length; i++) {
			const raw = toolResultMessages[i].content || '';
			let summary = raw;
			if (typeof raw === 'string') {
				try {
					const parsed = JSON.parse(raw);
					if (parsed && typeof parsed === 'object') {
						summary = parsed.message || parsed.error || parsed.status || raw;
					}
				} catch { /* leave as-is */ }
			}
			toolItems[i].result = summary;
		}
	}

	// Build readable output in original block order
	const parts = [];

	for (const item of sequence) {
		if (item.type === 'text') {
			parts.push(item.text);
		} else {
			const lines = [`Mira used **${item.name}**`];
			const entries = item.inputEntries;
			const hasResult = !!item.result;

			for (let i = 0; i < entries.length; i++) {
				const isLast = !hasResult && i === entries.length - 1;
				const elbow = isLast ? '└─' : '├─';
				lines.push(`${elbow} ${entries[i][0]}: ${entries[i][1]}`);
			}
			if (hasResult) {
				lines.push(`└─ result: ${item.result}`);
			}

			parts.push(lines.join('\n'));
		}
	}

	return parts.join('\n\n') || content;
};

/**
 * Render an entire assistant turn — every assistant + tool message produced
 * since the most recent user message — as one interleaved sequence. The
 * orchestrator persists tool_use blocks and the final text reply as separate
 * Message rows; rendering only one of them drops the other from the display.
 *
 * `turnMessages` must be in chronological order (oldest first). Each item is
 * a raw history message dict with `role` and `content`.
 */
window.assembleTurnContent = function(turnMessages) {
	if (!Array.isArray(turnMessages) || turnMessages.length === 0) return '';

	const sequence = []; // same shape as formatContentBlocks's internal sequence

	const parseToolResult = (raw) => {
		if (typeof raw !== 'string') return raw;
		try {
			const parsed = JSON.parse(raw);
			if (parsed && typeof parsed === 'object') {
				return parsed.message || parsed.error || parsed.status || raw;
			}
		} catch { /* leave as-is */ }
		return raw;
	};

	const attachToolResult = (resultText) => {
		// Attach to the most recent tool entry still missing a result
		for (let i = sequence.length - 1; i >= 0; i--) {
			if (sequence[i].type === 'tool' && sequence[i].result === null) {
				sequence[i].result = resultText;
				return;
			}
		}
	};

	for (const msg of turnMessages) {
		if (!msg) continue;

		if (msg.role === 'tool') {
			attachToolResult(parseToolResult(msg.content || ''));
			continue;
		}

		if (msg.role !== 'assistant') continue;

		const content = msg.content;
		if (!content) continue;

		// Try to parse as JSON content-block array; fall back to plain text.
		let parsedBlocks = null;
		if (typeof content === 'string') {
			const trimmed = content.trimStart();
			if (trimmed[0] === '[') {
				try {
					const blocks = JSON.parse(trimmed);
					if (Array.isArray(blocks)) parsedBlocks = blocks;
				} catch { /* fall through */ }
			}
		} else if (Array.isArray(content)) {
			parsedBlocks = content;
		}

		if (parsedBlocks) {
			for (const block of parsedBlocks) {
				if (!block || typeof block !== 'object') continue;
				if (block.type === 'text' && block.text) {
					sequence.push({ type: 'text', text: block.text });
				} else if (block.type === 'tool_use' && block.name) {
					const inputEntries = block.input && typeof block.input === 'object'
						? Object.entries(block.input).map(([k, v]) =>
							[k, typeof v === 'string' ? v : JSON.stringify(v)]
						)
						: [];
					sequence.push({ type: 'tool', name: block.name, inputEntries, result: null });
				}
			}
		} else if (typeof content === 'string') {
			sequence.push({ type: 'text', text: content });
		}
	}

	const parts = [];
	for (const item of sequence) {
		if (item.type === 'text') {
			parts.push(item.text);
		} else {
			const lines = [`Mira used **${item.name}**`];
			const entries = item.inputEntries;
			const hasResult = !!item.result;
			for (let i = 0; i < entries.length; i++) {
				const isLast = !hasResult && i === entries.length - 1;
				const elbow = isLast ? '└─' : '├─';
				lines.push(`${elbow} ${entries[i][0]}: ${entries[i][1]}`);
			}
			if (hasResult) {
				lines.push(`└─ result: ${item.result}`);
			}
			parts.push(lines.join('\n'));
		}
	}

	return parts.join('\n\n');
};

window.filterSystemTags = function(text) {
	if (!text) return text;

	return text
		.replace(/<think>[\s\S]*?<\/think>/gi, '')
		// All mira: namespaced tags (internal_monologue, memory_refs, my_emotion, etc.)
		.replace(/<mira:([^>\/\s]+)(?:\s[^>]*)?>[\s\S]*?<\/mira:\1>|<mira:[^>]*\/>/gi, '')
		// Strip ephemeral timestamps injected for LLM context (e.g., [5:47pm])
		.replace(/^\[\d{1,2}:\d{2}[ap]m\]\s*/i, '')
		.trim();
};

window.filterStreamingText = function(text) {
	if (!text) return text;

	let safeEndIndex = text.length;

	const incompleteTagPatterns = [
		/<think$/,
		/<mira:[^>]*$/,  // Handles all mira: tags including internal_monologue
		/<$/
	];

	for (const pattern of incompleteTagPatterns) {
		const match = text.match(pattern);
		if (match) {
			safeEndIndex = match.index;
			break;
		}
	}

	const safeText = text.substring(0, safeEndIndex);
	return window.filterSystemTags(safeText);
};

function addCodeBlockButtons() {
	const codeBlocks = elements.responseContent.querySelectorAll('pre code');

	codeBlocks.forEach((block, index) => {
		const wrapper = document.createElement('div');
		wrapper.className = 'code-block-wrapper';

		const preElement = block.parentElement;
		preElement.parentElement.insertBefore(wrapper, preElement);
		wrapper.appendChild(preElement);

		const copyBtn = document.createElement('button');
		copyBtn.className = 'code-copy-btn';
		copyBtn.innerHTML = 'Copy';
		copyBtn.setAttribute('data-code-index', index);

		copyBtn.addEventListener('click', async () => {
			const codeText = block.textContent;

			try {
				if (navigator.clipboard && window.isSecureContext) {
					await navigator.clipboard.writeText(codeText);
				} else {
					const textarea = document.createElement('textarea');
					textarea.value = codeText;
					textarea.style.position = 'absolute';
					textarea.style.left = '-9999px';
					document.body.appendChild(textarea);
					textarea.select();
					document.execCommand('copy');
					document.body.removeChild(textarea);
				}

				copyBtn.innerHTML = 'Copied!';
				copyBtn.classList.add('copied');

				setTimeout(() => {
					copyBtn.innerHTML = 'Copy';
					copyBtn.classList.remove('copied');
				}, 2000);
			} catch (err) {
				console.error('Failed to copy code:', err);
				copyBtn.innerHTML = 'Failed';
				setTimeout(() => {
					copyBtn.innerHTML = 'Copy';
				}, 2000);
			}
		});

		wrapper.appendChild(copyBtn);
	});
}

window.showResponse = function(message = null) {
	const response = message || "Hello! I'm MIRA, your AI assistant.";
	const filteredResponse = window.filterSystemTags(response);

	clearCompletionBadgeTimer();
	removeExistingCompletionBadge();

	prepareResponseContainer(() => {
		let htmlContent = filteredResponse;

		if (typeof marked !== 'undefined') {
			marked.setOptions({
				breaks: true,
				gfm: true,
				headerIds: false,
				mangle: false,
				sanitize: false
			});

			htmlContent = marked.parse(filteredResponse);
		}

		if (typeof DOMPurify !== 'undefined') {
			htmlContent = DOMPurify.sanitize(htmlContent, {
				ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
							   'blockquote', 'ul', 'ol', 'li', 'a', 'h1', 'h2',
							   'h3', 'h4', 'h5', 'h6', 'hr', 'table', 'thead',
							   'tbody', 'tr', 'th', 'td', 'img', 'span', 'div'],
				ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'id', 'src', 'alt',
							   'title', 'data-language', 'data-copy'],
				ALLOW_DATA_ATTR: true
			});
		}

		elements.responseContent.innerHTML = htmlContent;

		addCodeBlockButtons();

		elements.responseBox.style.display = 'block';
		AppState.responseActive = true;
		scheduleCompletionBadge();
	});
}

// ========================================
// API MESSAGE HANDLING
// ========================================

async function loadLastSessionResponse() {
	if (AppState.responseActive) {
		return;
	}

	try {
		console.log('Loading history - Auth state:', {
			isAuthenticated: AppState.apiClient.auth.isAuthenticated(),
			token: AppState.apiClient.token ? 'Present' : 'Missing',
			cookie: document.cookie
		});

		// Pull enough history to cover a full last turn (multi-round tool use can
		// produce many assistant + tool messages between two user messages).
		const response = await AppState.apiClient.history.getHistory({ limit: 20 });

		const allMessages = response.messages || []; // newest-first

		// Walk back from newest to find the most recent user message; everything
		// at a lower index is part of the last assistant turn.
		const userIdx = allMessages.findIndex(msg => msg.role === 'user');
		const turnSlice = userIdx >= 0 ? allMessages.slice(0, userIdx) : allMessages.slice();

		const turnMessagesNewestFirst = turnSlice.filter(msg =>
			(msg.role === 'assistant' || msg.role === 'tool') &&
			(!msg.metadata || !msg.metadata.is_segment_boundary)
		);

		// Reverse to chronological order for assembly
		const turnMessages = turnMessagesNewestFirst.slice().reverse();

		// Newest assistant in the turn anchors the timestamp/blur calculation
		const assistantMsg = turnMessagesNewestFirst.find(msg => msg.role === 'assistant');

		if (assistantMsg && turnMessages.length > 0) {
			const readable = window.assembleTurnContent(turnMessages);
			const filteredResponse = window.filterSystemTags(readable);

			let htmlContent = filteredResponse;
			if (typeof marked !== 'undefined') {
				marked.setOptions({
					breaks: true,
					gfm: true,
					headerIds: false,
					mangle: false,
					sanitize: false
				});
				htmlContent = marked.parse(filteredResponse);

				if (typeof DOMPurify !== 'undefined') {
					htmlContent = DOMPurify.sanitize(htmlContent, {
						ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
									   'blockquote', 'ul', 'ol', 'li', 'a', 'h1', 'h2',
									   'h3', 'h4', 'h5', 'h6', 'hr', 'table', 'thead',
									   'tbody', 'tr', 'th', 'td', 'img', 'span', 'div'],
						ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'id', 'src', 'alt',
									   'title', 'data-language', 'data-copy'],
						ALLOW_DATA_ATTR: true
					});
				}
			}

			// Create header element
			const headerDiv = document.createElement('div');
			headerDiv.id = 'session-header';
			headerDiv.textContent = 'Where we left off last session:';

			// Create content element
			const contentDiv = document.createElement('div');

			// Calculate progressive blur class based on message age
			let blurClass = 'blur-stale';

			if (assistantMsg.timestamp) {
				const messageTimestamp = new Date(assistantMsg.timestamp);
				const now = new Date();
				const minutesAway = (now - messageTimestamp) / (1000 * 60);

				if (minutesAway < 5) {
					blurClass = 'blur-fresh';
				} else if (minutesAway < 60) {
					blurClass = 'blur-recent';
				} else if (minutesAway < 120) {
					blurClass = 'blur-older';
				} else {
					blurClass = 'blur-stale';
				}
			}

			// Set INITIAL state (fully blurred) to enable transition
			contentDiv.classList.add('blur-initial');

			// NOW set the content
			contentDiv.innerHTML = htmlContent;

			// Clear and append
			elements.responseContent.innerHTML = '';
			elements.responseContent.appendChild(headerDiv);
			elements.responseContent.appendChild(contentDiv);

			addCodeBlockButtons();

			elements.responseContainer.classList.add('active');
			AppState.responseActive = true;

			// Show announcement banner if backend has one and user hasn't dismissed it
			const announcementBanner = document.getElementById('announcement-banner');
			if (announcementBanner) {
				try {
					const userData = await AppState.apiClient.data.getData('user');
					const announcement = userData?.preferences?.announcement;
					if (announcement && announcement.id && announcement.message) {
						const dismissed = localStorage.getItem('dismissed-announcement');
						if (dismissed !== announcement.id) {
							const textEl = document.getElementById('announcement-text');
							if (textEl) textEl.innerHTML = announcement.message;
							announcementBanner.style.display = '';
							// Store announcement id for dismiss handler
							announcementBanner.dataset.announcementId = announcement.id;
							// Wire up dismiss to persist in localStorage
							const dismissBtn = announcementBanner.querySelector('.system-alert-dismiss');
							if (dismissBtn) {
								dismissBtn.addEventListener('click', () => {
									localStorage.setItem('dismissed-announcement', announcement.id);
								}, { once: true });
							}
						}
					}
				} catch {
					// If user data fetch fails, don't show banner
				}
			}

			// Force Chrome to commit the blur-initial style before class change
			// Use setTimeout to give compositor time to paint initial state
			setTimeout(() => {
				contentDiv.classList.remove('blur-initial');
				contentDiv.classList.add(blurClass);
			}, 50);
		} else {
			elements.responseContent.innerHTML = `
				<div id="welcome-message">
					<p>Welcome to MIRA! I'm ready to assist you.</p>
					<p>Start by typing a message below.</p>
				</div>
			`;
			elements.responseContainer.classList.add('active');
			AppState.responseActive = true;
		}
	} catch (error) {
		console.error('Failed to load last session response:', error);
		console.error('Error details:', {
			message: error.message,
			response: error.response,
			stack: error.stack
		});
		elements.responseContent.innerHTML = `
			<div id="welcome-message">
				<p>Welcome to MIRA! I'm ready to assist you.</p>
				<p>Start by typing a message below.</p>
			</div>
		`;
		elements.responseContainer.classList.add('active');
		AppState.responseActive = true;
	}
}

async function handleSendMessage(message, attachedFiles = []) {
	try {
		const streamingEnabled = true;

		let imageData = null;
		if (attachedFiles && attachedFiles.length > 0) {
			try {
				const file = attachedFiles[0];
				const base64String = await new Promise((resolve, reject) => {
					const reader = new FileReader();
					reader.onload = (e) => {
						const dataUrl = e.target.result;
						const base64 = dataUrl.split(',')[1];
						resolve(base64);
					};
					reader.onerror = reject;
					reader.readAsDataURL(file);
				});

				imageData = {
					base64: base64String,
					mimeType: file.type
				};
			} catch (error) {
				console.error('Failed to convert file to base64:', error);
				showResponse('Error: Failed to process image file');
				return;
			}
		}

		if (streamingEnabled) {
			const handleStreamingFailure = (error) => {
				console.error('Streaming error:', error);
				stopLoadingAnimation();

				if (error.message === 'WebSocket closed') {
					AppState.responseInterrupted = true;
					if (hasStreamingTextContent()) {
						finalizeBufferedStreamingResponse(streamingState.buffer);
						window.updateThinkingIndicator(false, streamingState.currentEmotion);
					} else {
						showResponse('Connection lost — reconnecting. Your response will appear shortly.');
					}
					return;
				}

				showResponse(`Error: ${error.message}`);
				queueMessage(message);
			};

			setGenerating(true);
			showStreamingResponse(
				(data) => {
					if (data.content) {
						const currentText = elements.responseContent.textContent;
						updateStreamingResponse(currentText + data.content);
					}
				},
				() => {
					stopLoadingAnimation();
				},
				(error) => {
					handleStreamingFailure(error);
				}
			);

			const messageHandler = (data) => {
				if (data.type === 'provider_switch') {
					// Provider stalled — clear accumulated text and show persistent alert
					console.warn(`Provider switch: ${data.reason}`);
					streamingState.buffer = '';
					streamingState.thinkingBuffer = '';
					streamingState.renderedLength = 0;
					streamingState.firstBlockRendered = false;
					if (streamingState.thinkingElement) {
						streamingState.thinkingElement.remove();
						streamingState.thinkingElement = null;
					}
					// Remove any partial response content from the dead provider
					if (elements.responseContent) {
						elements.responseContent.textContent = '';
					}
					// Show persistent alert that fades when the new generation completes
					if (streamingState.providerSwitchAlert) {
						streamingState.providerSwitchAlert.remove();
					}
					const alertEl = document.createElement('div');
					alertEl.className = 'system-alert alert';
					alertEl.style.cssText = 'border-color: orange; margin: 8px 16px; padding: 8px 12px; border-radius: 6px; background: rgba(255,165,0,0.08); font-size: 13px; color: #e67e22;';
					alertEl.textContent = `⚠ Generation stalled — retrying with ${data.backup_model || 'backup'}`;
					const nc = document.getElementById('notifications-center');
					if (nc) nc.appendChild(alertEl);
					streamingState.providerSwitchAlert = alertEl;
					return;
				}

				if (data.type === 'text') {
					if (streamingState.thinkingActive && streamingState.containerReady) {
						transitionThinkingToText();
					} else if (streamingState.thinkingActive) {
						streamingState.thinkingActive = false;
						streamingState.thinkingBuffer = '';
						streamingState.thinkingElement = null;
					}
					if (window._streamHandlers) {
						window._streamHandlers.onChunk(data);
					} else if (data.content) {
						// Buffer early text arriving before prepareResponseContainer completes
						streamingState.buffer += data.content;
					}
					return;
				}

				if (data.type === 'thinking') {
					window.updateThinkingIndicator(true);
					if (data.content) {
						handleThinkingToken(data.content);
					}
					return;
				}

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

				if (isToolEvent && toolName) {
					const normalized = normalizeToolPhase(phase);
					if (normalized === 'detected' || normalized === 'executing') {
						window.updateToolIndicator?.(toolName, normalized);
					} else if (normalized === 'completed') {
						window.updateToolIndicator?.(toolName, 'completed');
					} else if (normalized === 'failed' || normalized === 'error') {
						window.updateToolIndicator?.(toolName, 'error');
					}
				}
			};

			AppState.apiClient.eventHandlers.onMessage.push(messageHandler);

			try {
				const response = await AppState.apiClient.chat.sendMessage(message, true, imageData);

				setGenerating(false);

				const index = AppState.apiClient.eventHandlers.onMessage.indexOf(messageHandler);
				if (index > -1) {
					AppState.apiClient.eventHandlers.onMessage.splice(index, 1);
				}

				// Remove provider-switch alert when generation completes
				if (streamingState.providerSwitchAlert) {
					streamingState.providerSwitchAlert.remove();
					streamingState.providerSwitchAlert = null;
				}

				if (window._streamHandlers) {
					const handlers = window._streamHandlers;  // Save reference before it gets deleted
					window.completeStreamingResponse(response.response);
					if (handlers && handlers.onComplete) {
						handlers.onComplete();
					}
				} else {
					if (response.response && response.response.length > streamingState.buffer.length) {
						streamingState.buffer = response.response;
					}
					streamingState.completionPending = true;
				}

				if (response.metadata) {
					// Update thinking indicator with emotion from metadata (or default to robot)
					const emotion = response.metadata.emotion || streamingState.currentEmotion;
					window.updateThinkingIndicator(false, emotion);

					if (response.metadata.tools_used && response.metadata.tools_used.length > 0) {
						const lastTool = response.metadata.tools_used[response.metadata.tools_used.length - 1];
						window.updateToolBadge?.(lastTool);

						if (response.metadata.tools_used.includes('domaindoc_tool')) {
							window.domainManager?.fetchDomains().catch(() => {});
						}
					}
					if (response.metadata.workflow_detected) {
						window.updateWorkflowBadge?.(response.metadata.workflow_detected);
					}
				}
			} catch (error) {
				setGenerating(false);

				const index = AppState.apiClient.eventHandlers.onMessage.indexOf(messageHandler);
				if (index > -1) {
					AppState.apiClient.eventHandlers.onMessage.splice(index, 1);
				}

				// Remove provider-switch alert on error too
				if (streamingState.providerSwitchAlert) {
					streamingState.providerSwitchAlert.remove();
					streamingState.providerSwitchAlert = null;
				}

				if (window._streamHandlers) {
					window._streamHandlers.onError(error);
				} else {
					streamingState.pendingError = error;
				}
			}
		}
	} catch (error) {
		console.error('Send message error:', error);
		setGenerating(false);
		stopLoadingAnimation();

		// WS disconnect mid-stream: server is still processing, response will
		// be committed to DB. Flag for recovery on reconnect / tab return.
		if (error.message === 'WebSocket closed') {
			AppState.responseInterrupted = true;
			showResponse('Connection lost — reconnecting. Your response will appear shortly.');
		} else {
			showResponse(`Error: ${error.message}`);
			queueMessage(message);
		}
	}
}

// ========================================
// THINKING STREAM & TRANSITION
// ========================================

function handleThinkingToken(content) {
	streamingState.thinkingActive = true;
	streamingState.thinkingBuffer += content;

	if (!streamingState.containerReady) {
		return;
	}

	renderThinkingBuffer();
}

function renderThinkingBuffer() {
	if (!streamingState.thinkingElement) {
		// First thinking token — dismiss loading screen, show response area
		try { stopLoadingAnimation(); } catch (e) { /* noop */ }

		const el = document.createElement('div');
		el.className = 'thinking-stream-content';
		if (!streamingState.firstBlockRendered) {
			elements.responseContent.innerHTML = '';
		}
		elements.responseContent.appendChild(el);
		elements.responseBox.style.display = 'block';
		elements.responseBox.classList.add('no-scroll');
		AppState.responseActive = true;
		streamingState.thinkingElement = el;
	}

	streamingState.thinkingElement.textContent = streamingState.thinkingBuffer;
	elements.responseBox.scrollTop = elements.responseBox.scrollHeight;
}

function transitionThinkingToText() {
	if (!streamingState.thinkingActive) return;
	streamingState.thinkingActive = false;

	const thinkingEl = streamingState.thinkingElement;
	const thinkingText = streamingState.thinkingBuffer;
	streamingState.thinkingElement = null;

	if (!thinkingEl || !thinkingText) {
		return;
	}

	// Gate output blocks for the full transition: 200ms pause + 300ms collapse
	streamingState.renderBlocked = true;

	setTimeout(() => {
		// Measure streaming text height as the collapse start point
		const startHeight = thinkingEl.offsetHeight;

		// Create drawer at that height so the swap is visually seamless
		const drawer = createThinkingDrawer(thinkingText);
		drawer.classList.add('active');
		drawer.style.height = startHeight + 'px';
		thinkingEl.replaceWith(drawer);

		// Zip closed: collapse to 28px (CSS transition: height 0.3s ease)
		requestAnimationFrame(() => {
			drawer.style.height = '';
		});

		// Unblock once the collapse finishes
		drawer.addEventListener('transitionend', function handler(e) {
			if (e.propertyName === 'height') {
				drawer.removeEventListener('transitionend', handler);
				streamingState.renderBlocked = false;
				renderStreamingBlocks();
			}
		});

		// Safety: unblock even if transitionend doesn't fire
		setTimeout(() => {
			if (streamingState.renderBlocked) {
				streamingState.renderBlocked = false;
				renderStreamingBlocks();
			}
		}, 500);
	}, 200);
}

function createThinkingDrawer(thinkingText) {
	const drawer = document.createElement('div');
	drawer.className = 'thinking-drawer';

	const toggle = document.createElement('div');
	toggle.className = 'thinking-drawer-toggle';
	toggle.innerHTML = '<span class="thinking-drawer-arrow">&#x25B8;</span> Thinking';

	const body = document.createElement('div');
	body.className = 'thinking-drawer-body';
	body.textContent = thinkingText;

	toggle.addEventListener('click', () => {
		const isExpanded = drawer.classList.toggle('expanded');
		toggle.querySelector('.thinking-drawer-arrow').innerHTML =
			isExpanded ? '&#x25BE;' : '&#x25B8;';
	});

	drawer.appendChild(toggle);
	drawer.appendChild(body);
	return drawer;
}

// ========================================
// MODULE EXPORTS
// ========================================

// Export functions needed by other modules
window.sendMessage = sendMessage;
window.handleFileSelect = handleFileSelect;
window.downscaleImage = downscaleImage;
window.clearAttachment = clearAttachment;
window.queueMessage = queueMessage;
window.renderQueuedMessages = renderQueuedMessages;
window.sendSingleQueuedMessage = sendSingleQueuedMessage;
window.removeQueuedMessage = removeQueuedMessage;
window.editQueuedMessage = editQueuedMessage;
window.saveQueuedMessageEdit = saveQueuedMessageEdit;
window.cancelQueuedMessageEdit = cancelQueuedMessageEdit;
window.loadLastSessionResponse = loadLastSessionResponse;
