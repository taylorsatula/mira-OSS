/**
 * SIDEBAR.JS - App Sidebar Navigation for All Pages
 *
 * Handles desktop collapse/expand (persisted via localStorage),
 * mobile toggle/close, swipe gestures, keyboard navigation,
 * focus trapping, and action buttons (History, Collapse Segment).
 */
(function() {
	'use strict';

	const sidebar = document.getElementById('app-sidebar');
	const toggle = document.getElementById('sidebar-toggle');
	const backdrop = document.getElementById('sidebar-backdrop');
	const collapseToggle = document.getElementById('sidebar-collapse-toggle');
	const expandToggle = document.getElementById('sidebar-expand-toggle');
	if (!sidebar || !toggle) return;

	const MOBILE_BP = 768;
	const STORAGE_KEY = 'mira-sidebar-collapsed';
	const SIDEBAR_WIDTH = 250;

	function isMobile() {
		return window.innerWidth <= MOBILE_BP;
	}

	function isOnChatPage() {
		const path = window.location.pathname;
		return path === '/' || path === '/chat' || path === '/chat/' || path === '/chat/index.html';
	}

	// --- Desktop collapse/expand ---

	function setCollapsed(collapsed, animate) {
		if (!animate) {
			sidebar.style.transition = 'none';
			document.body.style.transition = 'none';
		}

		if (collapsed) {
			sidebar.classList.add('collapsed');
			document.body.style.marginLeft = '0px';
			if (expandToggle) expandToggle.classList.add('visible');
		} else {
			sidebar.classList.remove('collapsed');
			document.body.style.marginLeft = SIDEBAR_WIDTH + 'px';
			if (expandToggle) expandToggle.classList.remove('visible');
		}

		if (!animate) {
			sidebar.offsetHeight;
			sidebar.style.transition = '';
			document.body.style.transition = '';
		}

		try {
			localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
		} catch (e) { /* storage unavailable */ }
	}

	function collapseSidebar() {
		if (isMobile()) { closeSidebar(); return; }
		setCollapsed(true, true);
	}

	function expandSidebar() {
		if (isMobile()) return;
		setCollapsed(false, true);
	}

	// Restore collapsed state on load (desktop only, no animation)
	// Applies visual state directly to avoid redundant localStorage write
	function restoreCollapseState() {
		if (isMobile()) {
			document.body.style.marginLeft = '';
			return;
		}
		try {
			if (localStorage.getItem(STORAGE_KEY) !== '1') return;
		} catch (e) { return; }

		sidebar.style.transition = 'none';
		document.body.style.transition = 'none';

		sidebar.classList.add('collapsed');
		document.body.style.marginLeft = '0px';
		if (expandToggle) expandToggle.classList.add('visible');

		sidebar.offsetHeight;
		sidebar.style.transition = '';
		document.body.style.transition = '';
	}

	restoreCollapseState();

	if (collapseToggle) {
		collapseToggle.addEventListener('click', collapseSidebar);
	}

	if (expandToggle) {
		expandToggle.addEventListener('click', expandSidebar);
	}

	// --- Mobile open/close ---

	function openSidebar() {
		sidebar.classList.add('open');
		sidebar.setAttribute('aria-hidden', 'false');
		backdrop.classList.add('open');
		toggle.setAttribute('aria-expanded', 'true');
		toggle.setAttribute('aria-label', 'Close navigation');
		toggle.style.display = 'none';
		const first = sidebar.querySelector('.sidebar-nav-item');
		if (first) first.focus();
	}

	function closeSidebar() {
		sidebar.classList.remove('open');
		sidebar.setAttribute('aria-hidden', 'true');
		backdrop.classList.remove('open');
		toggle.setAttribute('aria-expanded', 'false');
		toggle.setAttribute('aria-label', 'Open navigation');
		toggle.style.display = '';
		toggle.focus();
	}

	toggle.addEventListener('click', function() {
		if (sidebar.classList.contains('open')) {
			closeSidebar();
		} else {
			openSidebar();
		}
	});

	if (backdrop) {
		backdrop.addEventListener('click', closeSidebar);
	}

	// --- Keyboard: Escape closes, focus trapping ---

	document.addEventListener('keydown', function(e) {
		if (!isMobile() || !sidebar.classList.contains('open')) return;

		if (e.key === 'Escape') {
			e.preventDefault();
			closeSidebar();
			return;
		}

		if (e.key === 'Tab') {
			const focusable = sidebar.querySelectorAll(
				'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
			);
			if (focusable.length === 0) return;

			const first = focusable[0];
			const last = focusable[focusable.length - 1];

			if (e.shiftKey && document.activeElement === first) {
				e.preventDefault();
				last.focus();
			} else if (!e.shiftKey && document.activeElement === last) {
				e.preventDefault();
				first.focus();
			}
		}
	});

	// --- Swipe gestures (mobile) ---

	let touchStartX = 0;
	let touchStartY = 0;
	const SWIPE_THRESHOLD = 50;
	const EDGE_ZONE = 30;

	document.addEventListener('touchstart', function(e) {
		if (!isMobile()) return;
		touchStartX = e.touches[0].clientX;
		touchStartY = e.touches[0].clientY;
	}, { passive: true });

	document.addEventListener('touchend', function(e) {
		if (!isMobile()) return;
		const dx = e.changedTouches[0].clientX - touchStartX;
		const dy = e.changedTouches[0].clientY - touchStartY;

		if (Math.abs(dx) < SWIPE_THRESHOLD || Math.abs(dy) > Math.abs(dx)) return;

		if (dx > 0 && touchStartX < EDGE_ZONE && !sidebar.classList.contains('open')) {
			openSidebar();
		} else if (dx < 0 && sidebar.classList.contains('open')) {
			closeSidebar();
		}
	}, { passive: true });

	// --- Aria state + resize handling ---

	function handleResize() {
		if (isMobile()) {
			sidebar.setAttribute('aria-hidden', sidebar.classList.contains('open') ? 'false' : 'true');
			document.body.style.marginLeft = '';
		} else {
			sidebar.removeAttribute('aria-hidden');
			sidebar.classList.remove('open');
			backdrop.classList.remove('open');
			var isCollapsed = sidebar.classList.contains('collapsed');
			document.body.style.marginLeft = (isCollapsed ? 0 : SIDEBAR_WIDTH) + 'px';
		}
	}

	handleResize();
	var resizeRaf;
	window.addEventListener('resize', function() {
		if (resizeRaf) return;
		resizeRaf = requestAnimationFrame(function() {
			resizeRaf = null;
			handleResize();
		});
	});

	// --- Close sidebar on nav link click (mobile) ---

	sidebar.addEventListener('click', function(e) {
		if (!isMobile()) return;
		var link = e.target.closest('a.sidebar-nav-item');
		if (link) closeSidebar();
	});

	// --- History button ---

	var historyBtn = document.getElementById('sidebar-history-btn');
	if (historyBtn) {
		historyBtn.addEventListener('click', function() {
			if (isOnChatPage()) {
				window.InlineHistory?.toggle();
				historyBtn.textContent = window.InlineHistory?.isExpanded
					? 'Hide Chat History'
					: 'Show Chat History';
			} else {
				window.location.href = '/chat?history=1';
			}
		});
	}

	// --- Pause/Unpause button ---

	var pauseBtn = document.getElementById('sidebar-pause-btn');
	var pauseLabel = document.getElementById('sidebar-pause-label');
	var sidebarPaused = false;

	function updateSidebarPauseUI(isPaused, hasSegment) {
		if (!pauseBtn) return;
		sidebarPaused = isPaused;
		pauseBtn.style.display = hasSegment ? '' : 'none';
		if (isPaused) {
			pauseBtn.classList.add('paused');
			if (pauseLabel) pauseLabel.textContent = 'Unpause';
		} else {
			pauseBtn.classList.remove('paused');
			if (pauseLabel) pauseLabel.textContent = 'Pause';
		}
	}

	if (pauseBtn && isOnChatPage()) {
		pauseBtn.addEventListener('click', async function() {
			var manager = window.inactivityWarningManager;
			if (manager) {
				await manager.togglePause();
				return;
			}

			// Fallback: call API directly if manager isn't ready yet
			var api = window.miraAPI || window.AppState?.apiClient;
			if (!api) return;

			var action = sidebarPaused ? 'resume_session' : 'pause_session';
			pauseBtn.disabled = true;

			try {
				await api._httpRequest('/v0/api/actions', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						domain: 'continuum',
						action: action,
						data: {}
					})
				});
				window.hapticFeedback?.(50);
				updateSidebarPauseUI(!sidebarPaused, true);
			} catch (e) {
				console.error('[Sidebar] Pause toggle failed:', e);
			} finally {
				pauseBtn.disabled = false;
			}
		});

		// Subscribe to session status bar for state sync
		function trySubscribe() {
			var manager = window.inactivityWarningManager;
			if (manager) {
				manager.onStatusChange(function(state) {
					updateSidebarPauseUI(state.isPaused, state.hasActiveSegment);
				});
				updateSidebarPauseUI(manager.isPaused, manager.hasActiveSegment);
			} else {
				setTimeout(trySubscribe, 200);
			}
		}
		trySubscribe();
	}

	// --- Collapse Segment button ---

	var collapseBtn = document.getElementById('sidebar-collapse-btn');
	if (collapseBtn) {
		collapseBtn.addEventListener('click', function() {
			if (isOnChatPage() && window.Functions?.collapseSegment) {
				window.Functions.collapseSegment();
			}
		});
	}

	// --- ?history=1 param auto-expand ---

	if (isOnChatPage()) {
		var params = new URLSearchParams(window.location.search);
		if (params.get('history') === '1') {
			var url = new URL(window.location.href);
			url.searchParams.delete('history');
			window.history.replaceState({}, '', url.pathname);

			var retries = 0;
			var waitForHistory = function() {
				if (window.InlineHistory?.expand) {
					window.InlineHistory.expand();
					if (historyBtn) historyBtn.textContent = 'Hide Chat History';
				} else if (++retries < 50) {
					setTimeout(waitForHistory, 100);
				}
			};
			waitForHistory();
		}
	}
})();
