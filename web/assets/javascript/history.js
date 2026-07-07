/**
 * HISTORY.JS - Conversation History & Calendar
 *
 * PURPOSE:
 * Manages the conversation history system including loading, displaying, searching, and organizing
 * past conversations. Implements the calendar view for date-based navigation and temporal linking
 * functionality that allows linking specific days to the current conversation context.
 *
 * RESPONSIBILITIES:
 * - Conversation history loading with infinite scroll pagination
 * - Date-based grouping (today, yesterday, specific dates)
 * - Search with text highlighting across conversations
 * - Calendar rendering with activity indicators
 * - Month-based conversation activity detection
 * - Temporal day linking/unlinking (attach historical context to current session)
 * - Conversation item creation and rendering
 * - Empty state handling (no conversations, no results)
 * - Scroll trigger management for lazy loading
 *
 * WHAT GOES HERE:
 * - New history filtering or sorting options
 * - Additional calendar views (week view, year view)
 * - History export functionality
 * - Advanced search features (filters, date ranges, metadata)
 * - Conversation tagging or categorization
 * - History management operations (delete, archive)
 * - Temporal linking enhancements (link multiple days, link date ranges)
 *
 * WHAT DOESN'T GO HERE:
 * - Message sending or receiving → messaging.js
 * - Response rendering → messaging.js
 * - Visual animations → ui.js
 * - Event listener setup → events.js
 * - API client configuration → core.js
 *
 * DEPENDENCIES:
 * - core.js: AppState, elements, AppState.apiClient
 * - ui.js: toggleHistory, showLoadingScreen (if history drawer animated)
 *
 * DEPENDENTS:
 * - events.js (calls history functions from UI interactions)
 *
 * KEY PATTERNS:
 * - Pagination state in AppState.historyDrawer (offset, hasMore, isLoading)
 * - IntersectionObserver for infinite scroll triggering
 * - Date key normalization (today/yesterday → YYYY-MM-DD)
 * - Temporal link state stored in button data attributes
 *
 * LOAD ORDER:
 * After core.js, can load alongside ui.js and messaging.js.
 */

// ========================================
// CONVERSATION HISTORY SYSTEM
// ========================================

async function renderConversations() {
	try {
		AppState.historyDrawer.currentOffset = 0;
		AppState.historyDrawer.hasMore = false;
		AppState.historyDrawer.isLoading = false;

		elements.historyContent.innerHTML = '<div class="loading-history">Loading conversations...</div>';

		await loadConversations(0, 20);

		await updateTemporalLinkStates();

	} catch (error) {
		console.error('Failed to load conversations:', error);
		elements.historyContent.innerHTML = '<div class="empty-state">Failed to load conversations</div>';
	}
}

async function loadConversations(offset = 0, limit = 20) {
	if (AppState.historyDrawer.isLoading) return;

	AppState.historyDrawer.isLoading = true;

	try {
		const response = await AppState.apiClient.history.getHistory({ offset, limit });

		if (offset === 0) {
			elements.historyContent.innerHTML = '';
		}

		if (response.messages.length === 0) {
			if (offset === 0) {
				elements.historyContent.innerHTML = '<div class="empty-state">No conversations found</div>';
			}
			AppState.historyDrawer.hasMore = false;
			return;
		}

		const messageGroups = groupMessagesByDate(response.messages);
		renderMessageGroups(messageGroups, offset === 0);

		AppState.historyDrawer.currentOffset = response.next_offset || 0;
		AppState.historyDrawer.hasMore = response.has_more;

		if (AppState.historyDrawer.hasMore) {
			addInfiniteScrollTrigger();
		}

	} catch (error) {
		console.error('Failed to load conversations:', error);
		if (offset === 0) {
			elements.historyContent.innerHTML = '<div class="empty-state">Failed to load conversations</div>';
		}
	} finally {
		AppState.historyDrawer.isLoading = false;
	}
}

function groupMessagesByDate(messages) {
	const groups = {};
	const today = new Date();
	const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);

	messages.forEach(message => {
		if (!message.timestamp) return;

		const msgDate = new Date(message.timestamp);
		let dateKey;

		if (msgDate.toDateString() === today.toDateString()) {
			dateKey = 'today';
		} else if (msgDate.toDateString() === yesterday.toDateString()) {
			dateKey = 'yesterday';
		} else {
			dateKey = msgDate.toLocaleDateString();
		}

		if (!groups[dateKey]) {
			groups[dateKey] = [];
		}
		groups[dateKey].push(message);
	});

	return groups;
}

function renderMessageGroups(messageGroups, clearExisting = false) {
	if (clearExisting) {
		elements.historyContent.innerHTML = '';
	}

	Object.entries(messageGroups).forEach(([dateKey, messages]) => {
		let dateGroup = elements.historyContent.querySelector(`[data-date="${dateKey}"]`);

		if (!dateGroup) {
			dateGroup = document.createElement('div');
			dateGroup.className = 'date-group';
			dateGroup.dataset.date = dateKey;

			const dateHeader = document.createElement('div');
			dateHeader.className = 'date-header';

			const dateText = document.createElement('span');
			dateText.textContent = dateKey === 'today' ? 'Today' :
								  dateKey === 'yesterday' ? 'Yesterday' : dateKey;

			const linkButton = document.createElement('button');
			linkButton.className = 'temporal-link-button';
			linkButton.innerHTML = '<img src="/assets/images/icons/link.png" alt="Link day">';
			linkButton.title = 'Link this day to current conversation';

			linkButton.style.display = dateKey === 'today' ? 'none' : 'flex';

			linkButton.dataset.dateKey = dateKey;
			linkButton.dataset.archiveId = '';

			linkButton.addEventListener('click', async (e) => {
				e.stopPropagation();
				await handleTemporalLink(linkButton, dateKey);
			});

			dateHeader.appendChild(dateText);
			dateHeader.appendChild(linkButton);
			dateGroup.appendChild(dateHeader);

			elements.historyContent.appendChild(dateGroup);
		}

		for (let i = 0; i < messages.length; i += 2) {
			if (i + 1 < messages.length) {
				const msg1 = messages[i];
				const msg2 = messages[i + 1];

				let userMsg, assistantMsg;
				if (msg1.role === 'user') {
					userMsg = msg1;
					assistantMsg = msg2;
				} else if (msg1.role === 'assistant') {
					userMsg = msg2;
					assistantMsg = msg1;
				} else {
					continue;
				}

				if (userMsg && assistantMsg) {
					const conversationItem = createConversationItem(userMsg, assistantMsg);
					dateGroup.appendChild(conversationItem);
				}
			}
		}
	});
}

function createConversationItem(userMsg, assistantMsg) {
	const item = document.createElement('article');
	item.className = 'conversation-item';

	const userText = userMsg.content;
	const assistantText = assistantMsg.content;

	item.innerHTML = `
		<div class="conversation-user" role="log">Me: ${userText}</div>
		<div class="conversation-mira" role="log">MIRA: ${assistantText}</div>
	`;

	return item;
}

function addInfiniteScrollTrigger() {
	if (AppState.scrollObserver) {
		AppState.scrollObserver.disconnect();
		AppState.scrollObserver = null;
	}

	const existingTrigger = elements.historyContent.querySelector('.infinite-scroll-trigger');
	if (existingTrigger) {
		existingTrigger.remove();
	}

	const trigger = document.createElement('div');
	trigger.className = 'infinite-scroll-trigger';
	trigger.innerHTML = '<div class="loading-more">Loading more...</div>';
	elements.historyContent.appendChild(trigger);

	AppState.scrollObserver = new IntersectionObserver(
		(entries) => {
			if (entries[0].isIntersecting && AppState.historyDrawer.hasMore && !AppState.historyDrawer.isLoading) {
				loadConversations(AppState.historyDrawer.currentOffset, 20);
			}
		},
		{ threshold: 0.1 }
	);

	AppState.scrollObserver.observe(trigger);
}

function switchTab(scope) {
	document.querySelectorAll('.history-tab').forEach(tab => {
		tab.classList.toggle('active', tab.dataset.scope === scope);
	});
	AppState.historyDrawer.currentScope = scope;
	elements.datePicker.querySelector('span').textContent = 'Select date';

	loadConversations(0, 20);
}

function filterHistory(scope) {
	const groups = document.querySelectorAll('.date-group');
	const emptyState = document.querySelector('.empty-state');
	let hasVisibleItems = false;

	document.querySelectorAll('.conversation-item').forEach(item => {
		item.style.display = '';
	});

	groups.forEach(group => {
		const shouldShow =
			scope === 'all' ||
			(scope === 'today' && group.dataset.date === 'today') ||
			(scope === 'recent' && ['today', 'yesterday'].includes(group.dataset.date));

		group.classList.toggle('hidden', !shouldShow);
		if (shouldShow) hasVisibleItems = true;
	});

	emptyState.classList.toggle('hidden', hasVisibleItems);
}

function searchHistory(query) {
	const items = document.querySelectorAll('.conversation-item');
	const groups = document.querySelectorAll('.date-group');
	const emptyState = document.querySelector('.empty-state');
	let hasResults = false;

	elements.datePicker.querySelector('span').textContent = 'Select date';

	if (!query) {
		items.forEach(item => {
			const userEl = item.querySelector('.conversation-user');
			const miraEl = item.querySelector('.conversation-mira');
			userEl.innerHTML = userEl.textContent;
			miraEl.innerHTML = miraEl.textContent;
		});
		filterHistory(AppState.historyDrawer.currentScope);
		return;
	}

	const lowerQuery = query.toLowerCase();

	groups.forEach(group => {
		let groupHasMatch = false;
		const groupItems = group.querySelectorAll('.conversation-item');

		groupItems.forEach(item => {
			const userEl = item.querySelector('.conversation-user');
			const miraEl = item.querySelector('.conversation-mira');
			const userText = userEl.textContent;
			const miraText = miraEl.textContent;
			const matches = userText.toLowerCase().includes(lowerQuery) ||
						  miraText.toLowerCase().includes(lowerQuery);

			item.style.display = matches ? 'block' : 'none';
			if (matches) {
				groupHasMatch = true;
				hasResults = true;
				userEl.innerHTML = highlightText(userText, query);
				miraEl.innerHTML = highlightText(miraText, query);
			} else {
				userEl.innerHTML = userText;
				miraEl.innerHTML = miraText;
			}
		});

		group.classList.toggle('hidden', !groupHasMatch);
	});

	emptyState.classList.toggle('hidden', hasResults);
}

function highlightText(text, query) {
	if (!query) return text;

	const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
	return text.replace(regex, '<span class="search-highlight">$1</span>');
}

function loadContinuum(item) {
	const miraText = item.querySelector('.conversation-mira').textContent;
	toggleHistory();
	showLoadingScreen(() => {
		elements.responseContent.textContent = miraText;
		elements.responseContainer.classList.add('active');
		AppState.responseActive = true;
	});
}

// ========================================
// CALENDAR FUNCTIONALITY
// ========================================

async function renderCalendar(date = new Date()) {
	AppState.currentCalendarDate = date;
	const year = date.getFullYear();
	const month = date.getMonth();
	const firstDay = new Date(year, month, 1);
	const lastDay = new Date(year, month + 1, 0);
	const daysInMonth = lastDay.getDate();
	const startingDayOfWeek = firstDay.getDay();

	const hasDataDays = await getConversationDaysForMonth(year, month);

	let html = `
		<div class="calendar-header">
			<div class="calendar-month">${date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</div>
			<div class="calendar-nav">
				<button class="calendar-nav-prev">‹</button>
				<button class="calendar-nav-next">›</button>
			</div>
		</div>
		<div class="calendar-grid">
	`;

	['S', 'M', 'T', 'W', 'T', 'F', 'S'].forEach(day => {
		html += `<div class="calendar-day-header">${day}</div>`;
	});

	for (let i = 0; i < startingDayOfWeek; i++) {
		html += '<div class="calendar-day disabled"></div>';
	}

	const today = new Date();

	for (let day = 1; day <= daysInMonth; day++) {
		const cellDate = new Date(year, month, day);
		const isToday = cellDate.toDateString() === today.toDateString();
		const hasData = hasDataDays.includes(day);
		const isFuture = cellDate > today;

		let classes = 'calendar-day';
		if (isToday) classes += ' today';
		if (hasData && !isFuture) classes += ' has-data';
		if (isFuture) classes += ' disabled';

		html += `<div class="${classes}" ${!isFuture ? `data-date="${year}-${month}-${day}"` : ''}>${day}</div>`;
	}

	html += '</div>';
	elements.calendarPopup.innerHTML = html;
}

async function getConversationDaysForMonth(year, month) {
	try {
		const activeDays = new Set();
		let offset = 0;
		const limit = 100;
		let hasMore = true;

		console.log('Calendar activity debug - fetching for year:', year, 'month:', month);

		while (hasMore && offset < 500) {
			const response = await AppState.apiClient.history.getHistory({ offset, limit });

			console.log(`API response for offset ${offset}:`, response);

			if (response.messages && Array.isArray(response.messages) && response.messages.length > 0) {
				response.messages.forEach(message => {
					if (message.timestamp) {
						const msgDate = new Date(message.timestamp);
						if (msgDate.getFullYear() === year && msgDate.getMonth() === month) {
							activeDays.add(msgDate.getDate());
							console.log('Added active day:', msgDate.getDate(), 'from message:', message.timestamp);
						}
					}
				});

				hasMore = response.meta?.has_more && response.messages.length === limit;
				offset += limit;
			} else {
				hasMore = false;
			}
		}

		const result = Array.from(activeDays);
		console.log('Final active days:', result);
		return result;
	} catch (error) {
		console.error('Failed to get conversation activity:', error);
		return [];
	}
}

async function selectDate(date) {
	const dateStr = date.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });
	elements.datePicker.querySelector('span').textContent = dateStr;
	elements.calendarPopup.classList.remove('active');

	document.querySelectorAll('.history-tab').forEach(tab => tab.classList.remove('active'));

	elements.historyContent.innerHTML = '<div class="loading-history">Loading conversations...</div>';

	try {
		const apiDate = date.toISOString().split('T')[0];

		const response = await AppState.apiClient.history.getHistory({
			offset: 0,
			limit: 100,
			date: apiDate
		});

		if (response.messages && response.messages.length > 0) {
			const messageGroups = groupMessagesByDate(response.messages);
			renderMessageGroups(messageGroups, true);
		} else {
			elements.historyContent.innerHTML = `<div class="empty-state">No conversations on ${dateStr}</div>`;
		}

	} catch (error) {
		console.error('Failed to load conversations for date:', error);
		elements.historyContent.innerHTML = `<div class="empty-state">Failed to load conversations for ${dateStr}</div>`;
	}
}

// ========================================
// TEMPORAL LINKING
// ========================================

function convertDateKeyToDateStr(dateKey) {
	const now = new Date();

	switch (dateKey) {
		case 'today':
			return now.toISOString().split('T')[0];
		case 'yesterday':
			const yesterday = new Date(now);
			yesterday.setDate(yesterday.getDate() - 1);
			return yesterday.toISOString().split('T')[0];
		default:
			if (/^\d{4}-\d{2}-\d{2}$/.test(dateKey)) {
				return dateKey;
			}
			const parsedDate = new Date(dateKey);
			if (!isNaN(parsedDate.getTime())) {
				return parsedDate.toISOString().split('T')[0];
			}
			return null;
	}
}

async function handleTemporalLink(button, dateKey) {
	const isLinked = button.classList.contains('linked');

	if (isLinked) {
		await unlinkTemporalDay(button.dataset.archiveId);
		button.classList.remove('linked');
		button.dataset.archiveId = '';
		button.title = 'Link this day to current conversation';
	} else {
		const linkedDays = await getLinkedDays();
		if (linkedDays.linked_days.length >= linkedDays.max_allowed) {
			alert(`You can only link up to ${linkedDays.max_allowed} days at a time.`);
			return;
		}

		const dateStr = convertDateKeyToDateStr(dateKey);
		if (!dateStr) {
			console.error('Cannot determine date string from dateKey:', dateKey);
			return;
		}

		const archiveId = await linkTemporalDay(dateStr);
		if (archiveId) {
			button.classList.add('linked');
			button.dataset.archiveId = archiveId;
			button.title = 'Unlink this day from current conversation';
		}
	}
}

async function linkTemporalDay(dateStr) {
	try {
		const response = await AppState.apiClient.executeAction(
			'conversation',
			'link_day',
			{ date: dateStr }
		);

		if (response.linked) {
			console.log('Day linked successfully:', response.date);
			return response.archive_id;
		}
		return null;
	} catch (error) {
		console.error('Failed to link day:', error);
		alert('Failed to link day. Please try again.');
		return null;
	}
}

async function unlinkTemporalDay(archiveId) {
	try {
		const response = await AppState.apiClient.executeAction(
			'conversation',
			'unlink_day',
			{ archive_id: archiveId }
		);

		if (response.unlinked) {
			console.log('Day unlinked successfully');
		}
	} catch (error) {
		console.error('Failed to unlink day:', error);
		alert('Failed to unlink day. Please try again.');
	}
}

async function getLinkedDays() {
	try {
		return await AppState.apiClient.data.getData('linked_days');
	} catch (error) {
		console.error('Failed to get linked days:', error);
		return { linked_days: [], max_allowed: 0 };
	}
}

async function updateTemporalLinkStates() {
	const linkedDays = await getLinkedDays();
	const linkedArchiveIds = linkedDays.linked_days.map(day => day.archive_id);

	document.querySelectorAll('.temporal-link-button').forEach(button => {
		if (button.dataset.archiveId && linkedArchiveIds.includes(button.dataset.archiveId)) {
			button.classList.add('linked');
			button.title = 'Unlink this day from current conversation';
		} else {
			button.classList.remove('linked');
			button.dataset.archiveId = '';
			button.title = 'Link this day to current conversation';
		}
	});
}

// ========================================
// INLINE HISTORY VIEW
// ========================================

/**
 * State management for inline conversation history.
 * History is displayed within #response_box, above the current response.
 */
const InlineHistoryState = {
	isExpanded: false,
	loadedSegments: [],     // Cached segment data [{sentinel, pairs}]
	currentOffset: 0,       // API pagination cursor
	hasMore: true,
	isLoading: false,
	savedScrollPosition: null,
	scrollObserver: null
};

const INLINE_PAIR_LIMIT = 20;  // Messages to fetch per batch (10 pairs)

/**
 * Toggle inline history visibility.
 */
function toggleInlineHistory() {
	if (InlineHistoryState.isExpanded) {
		collapseInlineHistory();
	} else {
		expandInlineHistory();
	}
}

/**
 * Expand inline history view.
 * Shows the container, restores scroll position, loads history if needed.
 */
async function expandInlineHistory() {
	InlineHistoryState.isExpanded = true;
	elements.inlineHistoryContainer.classList.remove('hidden');

	// Restore scroll position if previously saved
	if (InlineHistoryState.savedScrollPosition !== null) {
		elements.responseBox.scrollTop = InlineHistoryState.savedScrollPosition;
	}

	// Load initial history if not already loaded
	if (InlineHistoryState.loadedSegments.length === 0) {
		await loadInlineHistory();
	}

	// Setup infinite scroll observer
	setupInlineScrollObserver();

	window.hapticFeedback?.(30);
}

/**
 * Collapse inline history view.
 * Saves scroll position, hides container, disconnects observer.
 */
function collapseInlineHistory() {
	// Save scroll position before collapsing
	InlineHistoryState.savedScrollPosition = elements.responseBox.scrollTop;

	InlineHistoryState.isExpanded = false;
	elements.inlineHistoryContainer.classList.add('hidden');

	// Disconnect observer when collapsed
	if (InlineHistoryState.scrollObserver) {
		InlineHistoryState.scrollObserver.disconnect();
		InlineHistoryState.scrollObserver = null;
	}

	// Hide toast if visible
	hideNewMessageToast();

	window.hapticFeedback?.(30);
}

/**
 * Load more history from the API.
 * Groups messages into segments and prepends to the container.
 */
async function loadInlineHistory() {
	if (InlineHistoryState.isLoading || !InlineHistoryState.hasMore) return;

	InlineHistoryState.isLoading = true;
	showInlineHistoryLoading();

	// Save scroll position for anchoring
	const scrollHeightBefore = elements.responseBox.scrollHeight;
	const scrollTopBefore = elements.responseBox.scrollTop;

	try {
		// Skip first 2 messages (current turn) on initial load
		const skipCurrentTurn = InlineHistoryState.loadedSegments.length === 0 ? 2 : 0;
		const apiOffset = InlineHistoryState.currentOffset + skipCurrentTurn;

		const response = await AppState.apiClient.history.getHistory({
			offset: apiOffset,
			limit: INLINE_PAIR_LIMIT
		});

		if (response.messages && response.messages.length > 0) {
			const segments = groupMessagesIntoSegments(response.messages);

			// Render each segment
			segments.forEach(segment => {
				const segmentElement = renderInlineSegment(segment);
				// Insert after scroll sentinel (which stays at top)
				elements.inlineHistorySentinel.after(segmentElement);
			});

			InlineHistoryState.loadedSegments = [...segments, ...InlineHistoryState.loadedSegments];
			InlineHistoryState.currentOffset += response.messages.length;
			InlineHistoryState.hasMore = response.has_more ||
				(response.meta && response.meta.has_more);
		} else {
			InlineHistoryState.hasMore = false;
		}
	} catch (error) {
		console.error('Failed to load inline history:', error);
	} finally {
		InlineHistoryState.isLoading = false;
		hideInlineHistoryLoading();

		// Scroll anchoring: restore visual position after prepending
		const scrollHeightAfter = elements.responseBox.scrollHeight;
		const heightDifference = scrollHeightAfter - scrollHeightBefore;
		elements.responseBox.scrollTop = scrollTopBefore + heightDifference;
	}
}

/**
 * Group messages into segments based on segment boundary sentinels.
 * @param {Array} messages - Array of message objects from API
 * @returns {Array} Array of {sentinel, pairs} objects
 */
function groupMessagesIntoSegments(messages) {
	const segments = [];
	let currentSegment = null;
	let currentPairs = [];

	// Debug: log segment boundaries found
	const boundaryMessages = messages.filter(m => m.metadata?.is_segment_boundary);
	if (boundaryMessages.length > 0) {
		console.log('[InlineHistory] Found segment boundaries:', boundaryMessages.map(m => ({
			status: m.metadata?.status,
			title: m.metadata?.display_title
		})));
	}

	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i];

		// Check if this is a segment sentinel
		if (msg.metadata?.is_segment_boundary) {
			// Save previous segment if exists
			if (currentSegment || currentPairs.length > 0) {
				segments.push({
					sentinel: currentSegment,
					pairs: currentPairs
				});
			}
			// Start new segment
			currentSegment = msg;
			currentPairs = [];
		} else {
			// Regular message - try to pair with next
			const nextMsg = messages[i + 1];
			if (nextMsg && !nextMsg.metadata?.is_segment_boundary) {
				let userMsg, assistantMsg;
				if (msg.role === 'user' && nextMsg.role === 'assistant') {
					userMsg = msg;
					assistantMsg = nextMsg;
				} else if (msg.role === 'assistant' && nextMsg.role === 'user') {
					userMsg = nextMsg;
					assistantMsg = msg;
				}

				if (userMsg && assistantMsg) {
					currentPairs.push({ user: userMsg, assistant: assistantMsg });
					i++; // Skip the paired message
				}
			}
		}
	}

	// Don't forget the last segment
	if (currentSegment || currentPairs.length > 0) {
		segments.push({
			sentinel: currentSegment,
			pairs: currentPairs
		});
	}

	return segments;
}

/**
 * Render a segment with its title and conversation pairs.
 * @param {Object} segment - {sentinel, pairs} object
 * @returns {HTMLElement} The segment DOM element
 */
function renderInlineSegment(segment) {
	const segmentDiv = document.createElement('div');
	segmentDiv.className = 'inline-history-segment';

	// Add segment summary block if sentinel exists and is collapsed
	if (segment.sentinel && segment.sentinel.metadata?.status === 'collapsed') {
		const summaryBlock = document.createElement('div');
		summaryBlock.className = 'inline-history-segment-summary';
		summaryBlock.id = `segment-${segment.sentinel.id || Date.now()}`;

		// Title
		const titleDiv = document.createElement('div');
		titleDiv.className = 'inline-history-segment-title';
		titleDiv.textContent = segment.sentinel.metadata?.display_title ||
			formatSegmentFallbackTitle(segment.sentinel);
		summaryBlock.appendChild(titleDiv);

		// Full summary content
		if (segment.sentinel.content && segment.sentinel.content !== '[Segment in progress]') {
			const contentDiv = document.createElement('div');
			contentDiv.className = 'inline-history-segment-content';

			// Render with markdown
			let htmlContent = segment.sentinel.content;
			if (typeof marked !== 'undefined') {
				marked.setOptions({
					breaks: true,
					gfm: true,
					headerIds: false,
					mangle: false,
					sanitize: false
				});
				htmlContent = marked.parse(segment.sentinel.content);
			}
			if (typeof DOMPurify !== 'undefined') {
				htmlContent = DOMPurify.sanitize(htmlContent, {
					ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'code', 'ul', 'ol', 'li'],
					ALLOWED_ATTR: []
				});
			}
			contentDiv.innerHTML = htmlContent;
			summaryBlock.appendChild(contentDiv);
		}

		segmentDiv.appendChild(summaryBlock);
	}

	// Render pairs in reverse order (API returns newest-first, we want oldest-first)
	const reversedPairs = [...segment.pairs].reverse();
	reversedPairs.forEach(pair => {
		const pairElement = createInlineHistoryPair(pair.user, pair.assistant);
		segmentDiv.appendChild(pairElement);
	});

	return segmentDiv;
}

/**
 * Create a fallback title for segments without display_title.
 * @param {Object} sentinel - Segment sentinel message
 * @returns {string} Formatted title
 */
function formatSegmentFallbackTitle(sentinel) {
	if (sentinel.timestamp) {
		const date = new Date(sentinel.timestamp);
		return date.toLocaleDateString('en-US', {
			weekday: 'short',
			month: 'short',
			day: 'numeric',
			hour: 'numeric',
			minute: '2-digit'
		});
	}
	return 'Earlier conversation';
}

/**
 * Create a conversation pair element (user + assistant messages).
 * Renders assistant content with markdown.
 * @param {Object} userMsg - User message object
 * @param {Object} assistantMsg - Assistant message object
 * @returns {HTMLElement} The pair DOM element
 */
function createInlineHistoryPair(userMsg, assistantMsg) {
	const pairDiv = document.createElement('div');
	pairDiv.className = 'inline-history-pair';

	// User message (plain text, full content)
	const userDiv = document.createElement('div');
	userDiv.className = 'inline-history-user';
	userDiv.textContent = userMsg.content || '';

	// Assistant message (markdown rendered)
	const assistantDiv = document.createElement('div');
	assistantDiv.className = 'inline-history-assistant';

	// Format content blocks (tool calls stored as JSON arrays) into readable text
	const readable = window.formatContentBlocks ?
		window.formatContentBlocks(assistantMsg.content) :
		assistantMsg.content;

	// Filter and render assistant content
	const filteredContent = window.filterSystemTags ?
		window.filterSystemTags(readable) :
		readable;

	let htmlContent = filteredContent;

	if (typeof marked !== 'undefined') {
		marked.setOptions({
			breaks: true,
			gfm: true,
			headerIds: false,
			mangle: false,
			sanitize: false
		});
		htmlContent = marked.parse(filteredContent);
	}

	if (typeof DOMPurify !== 'undefined') {
		htmlContent = DOMPurify.sanitize(htmlContent, {
			ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
				'blockquote', 'ul', 'ol', 'li', 'a', 'h1', 'h2',
				'h3', 'h4', 'h5', 'h6', 'hr'],
			ALLOWED_ATTR: ['href', 'target', 'rel', 'class']
		});
	}

	assistantDiv.innerHTML = htmlContent;

	pairDiv.appendChild(userDiv);
	pairDiv.appendChild(assistantDiv);

	return pairDiv;
}

/**
 * Setup IntersectionObserver for upward infinite scroll.
 */
function setupInlineScrollObserver() {
	if (InlineHistoryState.scrollObserver) {
		InlineHistoryState.scrollObserver.disconnect();
	}

	InlineHistoryState.scrollObserver = new IntersectionObserver(
		(entries) => {
			if (entries[0].isIntersecting &&
				!InlineHistoryState.isLoading &&
				InlineHistoryState.hasMore) {
				loadInlineHistory();
			}
		},
		{
			root: elements.responseBox,
			threshold: 0.1
		}
	);

	InlineHistoryState.scrollObserver.observe(elements.inlineHistorySentinel);
}

/**
 * Show loading indicator at top of history.
 */
function showInlineHistoryLoading() {
	let loader = elements.inlineHistoryContainer.querySelector('.inline-history-loading');
	if (!loader) {
		loader = document.createElement('div');
		loader.className = 'inline-history-loading';
		loader.textContent = 'Loading older conversations...';
		elements.inlineHistorySentinel.after(loader);
	}
}

/**
 * Hide loading indicator.
 */
function hideInlineHistoryLoading() {
	const loader = elements.inlineHistoryContainer.querySelector('.inline-history-loading');
	if (loader) loader.remove();
}

/**
 * Add only the user message to inline history.
 * The assistant response is visible in response_content, will be added on next turn.
 * @param {string} userMessage - The user's message text
 */
function addUserMessageToHistory(userMessage) {
	if (!InlineHistoryState.isExpanded) return;
	if (!userMessage) return;

	// Create a pair container that will hold user message now, assistant response later
	const pairDiv = document.createElement('div');
	pairDiv.className = 'inline-history-pair inline-history-pair-pending';

	// Create just the user message element
	const userDiv = document.createElement('div');
	userDiv.className = 'inline-history-user';
	userDiv.textContent = userMessage;
	pairDiv.appendChild(userDiv);

	// Find the last segment or create a "Current session" segment
	let currentSegment = elements.inlineHistoryContainer.querySelector('.inline-history-segment-current');

	if (!currentSegment) {
		currentSegment = document.createElement('div');
		currentSegment.className = 'inline-history-segment inline-history-segment-current';

		// Add a title for the current session
		const titleDiv = document.createElement('div');
		titleDiv.className = 'inline-history-segment-title';
		titleDiv.textContent = 'This session';
		currentSegment.appendChild(titleDiv);

		// Append at the end (bottom) of the history container
		elements.inlineHistoryContainer.appendChild(currentSegment);
	}

	// Append the pair (with just user message for now)
	currentSegment.appendChild(pairDiv);
}

/**
 * Complete the last pending history entry by adding its assistant response.
 * Called when the next message is sent.
 * @param {string} assistantResponse - The assistant's response text
 */
function completeLastHistoryEntry(assistantResponse) {
	if (!assistantResponse) return;

	// Find the pending pair (last one without assistant response)
	const pendingPair = elements.inlineHistoryContainer?.querySelector('.inline-history-pair-pending');
	if (!pendingPair) return;

	// Create and append the assistant response
	const assistantDiv = document.createElement('div');
	assistantDiv.className = 'inline-history-assistant';

	// Filter and render with markdown
	const filteredContent = window.filterSystemTags ?
		window.filterSystemTags(assistantResponse) :
		assistantResponse;

	let htmlContent = filteredContent;

	if (typeof marked !== 'undefined') {
		marked.setOptions({
			breaks: true,
			gfm: true,
			headerIds: false,
			mangle: false,
			sanitize: false
		});
		htmlContent = marked.parse(filteredContent);
	}

	if (typeof DOMPurify !== 'undefined') {
		htmlContent = DOMPurify.sanitize(htmlContent, {
			ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
				'blockquote', 'ul', 'ol', 'li', 'a', 'h1', 'h2',
				'h3', 'h4', 'h5', 'h6', 'hr'],
			ALLOWED_ATTR: ['href', 'target', 'rel', 'class']
		});
	}

	assistantDiv.innerHTML = htmlContent;
	pendingPair.appendChild(assistantDiv);

	// Mark as complete
	pendingPair.classList.remove('inline-history-pair-pending');
}

/**
 * Show the new message toast notification or auto-scroll.
 * Called when a new message arrives while user is in history view.
 * - Within 1 viewport of bottom: auto-scroll silently
 * - More than 1 viewport up: show toast
 */
function showNewMessageToast() {
	if (!InlineHistoryState.isExpanded) return;

	const viewportHeight = elements.responseBox.clientHeight;
	const scrollBottom = elements.responseBox.scrollHeight - elements.responseBox.scrollTop - viewportHeight;

	// If within 1 viewport of bottom, auto-scroll silently
	if (scrollBottom <= viewportHeight) {
		elements.responseBox.scrollTo({
			top: elements.responseBox.scrollHeight,
			behavior: 'smooth'
		});
		return;
	}

	// More than 1 viewport up - show toast
	elements.newMessageToast.classList.remove('hidden');
	window.hapticFeedback?.(50);
}

/**
 * Hide the new message toast notification.
 */
function hideNewMessageToast() {
	elements.newMessageToast.classList.add('hidden');
}

/**
 * Scroll to the latest message (bottom of response_box).
 */
function scrollToLatest() {
	elements.responseBox.scrollTo({
		top: elements.responseBox.scrollHeight,
		behavior: 'smooth'
	});
	hideNewMessageToast();
}

/**
 * Initialize inline history event listeners.
 */
function initInlineHistory() {
	// Toast scroll button click
	elements.toastScrollBtn?.addEventListener('click', scrollToLatest);

	// Note: Sidebar history button is handled by sidebar.js which calls InlineHistory.toggle()
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', initInlineHistory);
} else {
	initInlineHistory();
}

// Export for external access
window.InlineHistory = {
	get isExpanded() { return InlineHistoryState.isExpanded; },
	toggle: toggleInlineHistory,
	expand: expandInlineHistory,
	collapse: collapseInlineHistory,
	notifyNewMessage: showNewMessageToast,
	addUserMessage: addUserMessageToHistory,
	completeLastEntry: completeLastHistoryEntry
};
