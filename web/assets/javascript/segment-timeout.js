/**
 * SEGMENT-TIMEOUT.JS - Session Timer Control
 *
 * PURPOSE:
 * Displays countdown until current conversation segment auto-collapses.
 * Allows user to pause/resume sessions via pause_session/resume_session actions.
 *
 * RESPONSIBILITIES:
 * - Polling segment status for collapse time
 * - Rendering countdown timer in toolbar indicator
 * - Popover UI for pause/resume actions
 * - Visual urgency indicators (color changes as time decreases)
 *
 * API INTEGRATION:
 * - POST: /actions with domain=continuum, action=get_segment_status
 * - POST: /actions with domain=continuum, action=pause_session
 * - POST: /actions with domain=continuum, action=resume_session
 *
 * DEPENDENCIES:
 * - api-client.js (MiraAPIClient for backend communication)
 * - core.js (AppState.apiClient)
 */

class SegmentTimeoutManager {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.collapseAt = null;
        this.isPaused = false;
        this.pollInterval = null;
        this.countdownInterval = null;
        this.hasActiveSegment = false;

        // Callback subscribers for external listeners (e.g., InactivityWarningManager)
        this.statusCallbacks = [];

        // DOM elements
        this.btn = document.getElementById('segment-timeout-btn');
        this.popover = document.getElementById('segment-timeout-popover');
        this.label = document.getElementById('segment-timeout-label');
        this.countdownTime = document.getElementById('countdown-time');
        this.countdownLabel = document.getElementById('countdown-label');
        this.timeoutStatus = document.getElementById('timeout-status');
        this.header = document.getElementById('segment-timeout-header');
    }

    /**
     * Register a callback to receive status updates
     * @param {Function} callback - Called with {hasActiveSegment, collapseAt, isPaused}
     */
    onStatusChange(callback) {
        this.statusCallbacks.push(callback);
    }

    /**
     * Notify all subscribers of status change
     */
    notifyStatusChange() {
        const state = {
            hasActiveSegment: this.hasActiveSegment,
            collapseAt: this.collapseAt,
            isPaused: this.isPaused
        };
        this.statusCallbacks.forEach(cb => cb(state));
    }

    /**
     * Initialize the manager - fetch status and start polling
     */
    async init() {
        if (!this.btn || !this.popover) {
            console.warn('[SegmentTimeout] Required DOM elements not found');
            return;
        }

        // Set up event listeners
        this.setupEventListeners();

        // Initial fetch
        await this.fetchStatus();

        // Start polling every 60 seconds
        this.pollInterval = setInterval(() => this.fetchStatus(), 60000);

        // Start countdown update every second
        this.countdownInterval = setInterval(() => this.updateCountdown(), 1000);

        console.log('[SegmentTimeout] Initialized');
    }

    /**
     * Set up DOM event listeners
     */
    setupEventListeners() {
        // Toggle popover on button click
        this.btn.addEventListener('click', (e) => {
            e.stopPropagation();
            window.hapticFeedback?.(100);
            this.popover.classList.toggle('active');
        });

        // Close button
        const closeBtn = this.popover.querySelector('.popover-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.popover.classList.remove('active');
            });
        }

        // Pause/Resume button
        const pauseBtn = document.getElementById('pause-session-btn');
        if (pauseBtn) {
            pauseBtn.addEventListener('click', async () => {
                await this.togglePause();
            });
        }

        // Collapse (Wrap Up Now) button
        const collapseBtn = document.getElementById('collapse-segment-btn');
        if (collapseBtn) {
            collapseBtn.addEventListener('click', async () => {
                await this.collapseSegment();
            });
        }

        // Close popover when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.popover.contains(e.target) && !this.btn.contains(e.target)) {
                this.popover.classList.remove('active');
            }
        });
    }

    /**
     * Fetch segment status from API
     */
    async fetchStatus() {
        try {
            const response = await this.apiClient._httpRequest('/v0/api/actions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: 'continuum',
                    action: 'get_segment_status',
                    data: {}
                })
            });

            if (response.success && response.data) {
                const data = response.data;
                this.hasActiveSegment = data.has_active_segment;
                this.isPaused = data.is_paused || false;

                if (data.collapse_at) {
                    this.collapseAt = new Date(data.collapse_at);
                } else {
                    this.collapseAt = null;
                }

                this.updateCountdown();
                this.updateStatus();
                this.updatePauseButton();

                // Notify subscribers (e.g., InactivityWarningManager)
                this.notifyStatusChange();
            }
        } catch (error) {
            console.error('[SegmentTimeout] Failed to fetch status:', error);
        }
    }

    /**
     * Toggle session pause state
     */
    async togglePause() {
        const action = this.isPaused ? 'resume_session' : 'pause_session';
        const pauseBtn = document.getElementById('pause-session-btn');

        try {
            if (pauseBtn) {
                pauseBtn.disabled = true;
                pauseBtn.textContent = '...';
            }

            const response = await this.apiClient._httpRequest('/v0/api/actions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: 'continuum',
                    action: action,
                    data: {}
                })
            });

            if (response.success || response.paused || response.resumed) {
                window.hapticFeedback?.(50);

                // Refresh status from server
                await this.fetchStatus();

                // Keep popover open so user sees the state change
                this.popover.classList.add('active');

                // Brief success feedback
                if (this.timeoutStatus) {
                    this.timeoutStatus.textContent = this.isPaused ? 'Session paused' : 'Session resumed';
                    this.timeoutStatus.className = 'timeout-status success';
                    setTimeout(() => this.updateStatus(), 2000);
                }
            } else {
                throw new Error(response.error?.message || `Failed to ${action}`);
            }
        } catch (error) {
            console.error(`[SegmentTimeout] Failed to ${action}:`, error);
            if (this.timeoutStatus) {
                this.timeoutStatus.textContent = `Failed to ${this.isPaused ? 'resume' : 'pause'}`;
                this.timeoutStatus.className = 'timeout-status error';
            }
        } finally {
            this.updatePauseButton();
        }
    }

    /**
     * Trigger immediate segment collapse
     */
    async collapseSegment() {
        const collapseBtn = document.getElementById('collapse-segment-btn');

        try {
            if (collapseBtn) {
                collapseBtn.disabled = true;
                collapseBtn.textContent = 'Wrapping up...';
            }

            const response = await this.apiClient._httpRequest('/v0/api/actions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: 'continuum',
                    action: 'collapse_segment',
                    data: {}
                })
            });

            if (response.success || response.collapsed) {
                window.hapticFeedback?.(50);

                if (this.timeoutStatus) {
                    this.timeoutStatus.textContent = 'Session wrapped up';
                    this.timeoutStatus.className = 'timeout-status success';
                }

                // Refresh status — segment is now collapsed
                await this.fetchStatus();

                // Close popover after successful collapse
                setTimeout(() => {
                    this.popover.classList.remove('active');
                }, 1500);
            } else {
                throw new Error(response.error?.message || 'Failed to collapse segment');
            }
        } catch (error) {
            console.error('[SegmentTimeout] Failed to collapse segment:', error);
            if (this.timeoutStatus) {
                this.timeoutStatus.textContent = 'Failed to wrap up';
                this.timeoutStatus.className = 'timeout-status error';
            }
        } finally {
            if (collapseBtn) {
                collapseBtn.disabled = false;
                collapseBtn.textContent = 'Wrap Up Now';
            }
        }
    }

    /**
     * Update pause button text based on current state
     */
    updatePauseButton() {
        const pauseBtn = document.getElementById('pause-session-btn');
        const collapseBtn = document.getElementById('collapse-segment-btn');
        const actionsContainer = pauseBtn?.closest('.timeout-actions');

        // Hide all actions when idle — nothing to pause or collapse
        if (actionsContainer) {
            actionsContainer.style.display = this.hasActiveSegment ? '' : 'none';
        }

        if (!pauseBtn) return;

        pauseBtn.disabled = false;
        if (this.isPaused) {
            pauseBtn.textContent = 'Unpause';
            pauseBtn.classList.add('paused');
        } else {
            pauseBtn.textContent = 'Pause';
            pauseBtn.classList.remove('paused');
        }

        // Show collapse button when paused, hide otherwise
        if (collapseBtn) {
            collapseBtn.style.display = this.isPaused ? '' : 'none';
        }
    }

    /**
     * Update the countdown display
     */
    updateCountdown() {
        // Idle state — no active segment, show sleeping indicator
        if (!this.hasActiveSegment) {
            if (this.btn) this.btn.style.display = '';
            if (this.label) this.label.textContent = '';
            if (this.countdownTime) this.countdownTime.textContent = '○';
            if (this.countdownLabel) this.countdownLabel.textContent = 'no active session';
            this.updateIndicatorColor('idle');
            return;
        }

        // Paused state — show pause indicator
        if (this.isPaused) {
            if (this.btn) this.btn.style.display = '';
            if (this.label) this.label.textContent = 'Paused';
            if (this.countdownTime) this.countdownTime.textContent = '⏸';
            if (this.countdownLabel) this.countdownLabel.textContent = 'session paused';
            this.updateIndicatorColor('paused');
            return;
        }

        // Active segment — hide indicator (you're using it)
        if (this.btn) this.btn.style.display = 'none';

        if (!this.collapseAt) return;

        const now = new Date();
        const diff = this.collapseAt - now;

        if (diff <= 0) {
            this.fetchStatus();
            return;
        }

        // Still update popover countdown content even though indicator is hidden,
        // in case the popover was left open
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);

        let timeStr;
        if (hours > 0) {
            timeStr = `${hours}:${minutes.toString().padStart(2, '0')}`;
        } else {
            timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }

        if (this.countdownTime) this.countdownTime.textContent = timeStr;
        if (this.countdownLabel) this.countdownLabel.textContent = 'until session wraps up';
    }

    /**
     * Update the status message in popover
     */
    updateStatus() {
        if (!this.timeoutStatus) return;

        if (!this.hasActiveSegment) {
            if (this.header) this.header.textContent = 'Session';
            this.timeoutStatus.textContent = 'Type a message to begin';
            this.timeoutStatus.className = 'timeout-status idle';
            return;
        }

        if (this.header) this.header.textContent = 'Session Timer';

        if (this.isPaused) {
            this.timeoutStatus.textContent = 'Resumes on next message';
            this.timeoutStatus.className = 'timeout-status paused';
        } else {
            this.timeoutStatus.textContent = '';
            this.timeoutStatus.className = 'timeout-status';
        }
    }

    /**
     * Update indicator button color based on urgency
     */
    updateIndicatorColor(level) {
        if (!this.btn) return;

        // Remove existing color classes
        this.btn.classList.remove('red', 'yellow', 'lime', 'cyan', 'idle');

        switch (level) {
            case 'urgent':
                this.btn.classList.add('red');
                break;
            case 'warning':
                this.btn.classList.add('yellow');
                break;
            case 'paused':
                this.btn.classList.add('lime');
                break;
            case 'idle':
                this.btn.classList.add('idle');
                break;
            case 'default':
                this.btn.classList.add('cyan');
                break;
        }
    }

    /**
     * Clean up intervals on destroy
     */
    destroy() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
            this.countdownInterval = null;
        }
    }
}

// Initialize when DOM is ready and API client is available
(function initSegmentTimeout() {
    function tryInit() {
        if (window.AppState?.apiClient) {
            window.segmentTimeoutManager = new SegmentTimeoutManager(window.AppState.apiClient);
            window.segmentTimeoutManager.init();
        } else {
            // Retry after a short delay
            setTimeout(tryInit, 100);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryInit);
    } else {
        tryInit();
    }
})();
