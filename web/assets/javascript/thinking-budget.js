/**
 * THINKING-BUDGET.JS - LLM Tier Selection Control
 *
 * PURPOSE:
 * Manages user's LLM tier preference.
 * Tier selection persists in user preferences (database-backed).
 *
 * RESPONSIBILITIES:
 * - Fetching/setting LLM tier preference
 * - Popover UI rendering and interaction
 * - Radio button state management
 *
 * API INTEGRATION:
 * - GET/POST: /actions with domain=continuum, action=get_conversation_llm / set_conversation_llm
 *
 * DEPENDENCIES:
 * - api-client.js (MiraAPIClient for backend communication)
 * - core.js (AppState.apiClient)
 */

class TierManager {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.currentTier = 'minimax';
        this.allTiers = [];
    }

    /**
     * Fetch current tier preference from user preferences
     */
    async fetchTier() {
        try {
            const response = await this.apiClient._httpRequest('/v0/api/actions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: 'continuum',
                    action: 'get_conversation_llm',
                    data: {}
                })
            });

            if (response.success) {
                this.currentTier = response.name;
                this.allTiers = response.available || [];
            }
            return this.currentTier;
        } catch (error) {
            console.error('[TierManager] Failed to fetch tier preference:', error);
            return this.currentTier;
        }
    }

    /**
     * Set tier preference
     */
    async setTier(tier) {
        try {
            const response = await this.apiClient._httpRequest('/v0/api/actions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: 'continuum',
                    action: 'set_conversation_llm',
                    data: { name: tier }
                })
            });

            if (response.success === false) {
                console.error('[TierManager] API returned error:', response.error);
                throw new Error(`Failed to set tier: ${JSON.stringify(response.error)}`);
            }

            this.currentTier = tier;
            this.updateIndicatorLabel();
            return response;
        } catch (error) {
            console.error('[TierManager] Failed to set tier preference:', error);
            throw error;
        }
    }

    /**
     * Update the tier indicator label in the toolbar
     */
    updateIndicatorLabel() {
        if (this.currentTier && window.ToolbarPriorityManager) {
            const currentTierData = this.allTiers.find(t => t.name === this.currentTier);
            if (currentTierData) {
                window.ToolbarPriorityManager.setIndicatorLabel('tier_btn', currentTierData.description);
            }
        }
    }

    /**
     * Render popover radio buttons dynamically from tier data
     */
    renderPopover() {
        const tierOptionsContainer = document.getElementById('tier-options');
        const upgradeMsg = document.getElementById('tier-upgrade-message');

        if (!tierOptionsContainer) return;

        // Clear existing options and rebuild dynamically
        tierOptionsContainer.innerHTML = '';

        for (const tier of this.allTiers) {
            const label = document.createElement('label');
            label.className = 'tier-option';

            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'llm-tier';
            radio.value = tier.name;
            radio.checked = (tier.name === this.currentTier);

            radio.addEventListener('change', async (e) => {
                if (e.target.checked) {
                    await this.setTier(e.target.value);
                }
            });

            const nameSpan = document.createElement('span');
            nameSpan.className = 'tier-name';
            nameSpan.textContent = tier.description;  // Show model name as primary label

            label.appendChild(radio);
            label.appendChild(nameSpan);
            tierOptionsContainer.appendChild(label);
        }

        // Hide upgrade message (no longer applicable)
        if (upgradeMsg) {
            upgradeMsg.style.display = 'none';
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    if (AppState && AppState.apiClient) {
        const tierManager = new TierManager(AppState.apiClient);
        window.tierManager = tierManager;

        // Fetch initial tier preference
        await tierManager.fetchTier();
        tierManager.updateIndicatorLabel();

        // Setup tier button and popover
        const tierBtn = document.querySelector('[data-indicator="tier_btn"]');
        const tierPopover = document.getElementById('tier-popover');
        const tierPopoverClose = tierPopover?.querySelector('.popover-close');

        if (tierBtn && tierPopover) {
            // Open popover on button click
            tierBtn.addEventListener('click', async () => {
                tierPopover.classList.toggle('active');
                if (tierPopover.classList.contains('active')) {
                    await tierManager.fetchTier();
                    tierManager.renderPopover();
                }
            });

            // Close popover on close button
            if (tierPopoverClose) {
                tierPopoverClose.addEventListener('click', () => {
                    tierPopover.classList.remove('active');
                });
            }

            // Close popover on click outside
            document.addEventListener('click', (e) => {
                if (tierPopover.classList.contains('active') &&
                    !tierPopover.contains(e.target) &&
                    !tierBtn.contains(e.target)) {
                    tierPopover.classList.remove('active');
                }
            });
        }
    }
});
