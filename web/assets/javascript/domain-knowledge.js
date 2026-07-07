/**
 * DOMAIN-KNOWLEDGE.JS - Domain Knowledge Block Management Feature
 *
 * PURPOSE:
 * Manages domain knowledge blocks - user-defined context that MIRA loads into conversations
 * when enabled. Each domain block contains custom instructions, facts, or context specific to
 * a topic (e.g., "Work", "Michigan Trip", "Project X"). This module handles CRUD operations
 * and provides UI rendering for both the chat interface popover and settings page.
 *
 * RESPONSIBILITIES:
 * - Domain block CRUD operations (create, enable, disable, delete, update)
 * - Domain list fetching and state management
 * - Enabled domain tracking (Set-based for O(1) lookups)
 * - Chat popover rendering (domain toggle interface with single-domain constraint)
 * - Settings page rendering (domain cards with edit/delete actions)
 * - Domain content editing (modal dialog with textarea)
 * - Domain creation dialog (modal with name/description inputs)
 * - Single-domain constraint enforcement (only one active at a time)
 * - Backend API integration via actions endpoint
 *
 * WHAT GOES HERE:
 * - New domain operations (duplicate, import/export, share)
 * - Domain templates or presets
 * - Domain categorization or tagging
 * - Domain search and filtering
 * - Multi-domain support (if constraint is relaxed)
 * - Domain version history or rollback
 * - Collaborative domain editing features
 *
 * WHAT DOESN'T GO HERE:
 * - API communication protocol → api-client.js
 * - Main application state → core.js
 * - General settings management → settings.js
 * - Message handling → messaging.js
 * - UI animations → ui.js
 *
 * DEPENDENCIES:
 * - api-client.js (MiraAPIClient for backend communication)
 * - Uses apiClient.data.getData() and apiClient.actions.executeAction()
 *
 * DEPENDENTS:
 * - Chat interface (domain popover toggle)
 * - Settings page (domain management UI)
 *
 * KEY PATTERNS:
 * - Class-based module with dependency injection (apiClient)
 * - State cached in memory (domains array, enabledDomains Set)
 * - Imperative DOM rendering (createElement, appendChild)
 * - Inline styles for modal dialogs (self-contained styling)
 * - Async/await for all API operations
 * - Re-fetch after mutations to ensure consistency
 *
 * DOMAIN BLOCK CONCEPTS:
 * - Domain Label: Unique identifier (technical name)
 * - Domain Name: User-facing display name
 * - Block Description: Short summary of what the domain contains
 * - Content: The actual markdown/text loaded into conversation context
 * - Enabled: Boolean flag, only one domain can be enabled at a time
 *
 * SINGLE-DOMAIN CONSTRAINT:
 * Backend enforces only one domain active at a time. UI reflects this by:
 * - Disabling checkboxes for inactive domains when one is enabled
 * - Showing constraint notice in both popover and settings
 * - Requiring disable before enable of another domain
 *
 * LOAD ORDER:
 * After api-client.js, alongside other feature modules.
 */

class DomainKnowledgeManager {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.domains = [];
        this.enabledDomains = new Set();
    }

    /**
     * Fetch all domains for current user
     */
    async fetchDomains() {
        try {
            const response = await this.apiClient.data.getData('domaindocs');

            this.domains = response.domaindocs || [];
            this.enabledDomains = new Set(
                this.domains.filter(d => d.enabled).map(d => d.label)
            );

            this.updateIndicatorStatus();
            return this.domains;
        } catch (error) {
            console.error('Failed to fetch domains:', error);
            // Re-throw to let callers handle the failure
            // Previously this silently returned [] without updating this.domains,
            // causing stale data to persist in the UI
            throw error;
        }
    }

    /**
     * Create a new domaindoc
     */
    async createDomain(label, description) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'create', { label, description }
            );

            await this.fetchDomains();
            return response;
        } catch (error) {
            console.error('Failed to create domain:', error);
            throw error;
        }
    }

    /**
     * Enable a domaindoc
     */
    async enableDomain(label) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'enable', { label }
            );

            this.enabledDomains.add(label);
            await this.fetchDomains();
            return response;
        } catch (error) {
            console.error('Failed to enable domain:', error);
            throw error;
        }
    }

    /**
     * Disable a domaindoc
     */
    async disableDomain(label) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'disable', { label }
            );

            this.enabledDomains.delete(label);
            await this.fetchDomains();
            return response;
        } catch (error) {
            console.error('Failed to disable domain:', error);
            throw error;
        }
    }

    /**
     * Delete a domaindoc
     */
    async deleteDomain(label) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'delete', { label }
            );

            this.enabledDomains.delete(label);
            await this.fetchDomains();
            return response;
        } catch (error) {
            console.error('Failed to delete domain:', error);
            throw error;
        }
    }

    /**
     * Archive a domaindoc
     */
    async archiveDomain(label) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'archive', { label }
            );

            this.enabledDomains.delete(label);
            await this.fetchDomains();
            return response;
        } catch (error) {
            console.error('Failed to archive domain:', error);
            throw error;
        }
    }

    /**
     * Unarchive a domaindoc
     */
    async unarchiveDomain(label) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'unarchive', { label }
            );

            await this.fetchDomains();
            return response;
        } catch (error) {
            console.error('Failed to unarchive domain:', error);
            throw error;
        }
    }

    /**
     * Get domaindoc content
     */
    async getDomainContent(label) {
        try {
            const response = await this.apiClient.data.getData('domaindocs', { label });

            return response.content || '';
        } catch (error) {
            console.error('Failed to get domain content:', error);
            throw error;
        }
    }

    /**
     * Update domaindoc content directly (for settings page editing)
     * Note: During conversation, MIRA uses domaindoc_tool for edits.
     */
    async updateDomainContent(label, content) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'rewrite', { label, content }
            );

            return response;
        } catch (error) {
            console.error('Failed to update domain content:', error);
            throw error;
        }
    }

    /**
     * Modify domaindoc metadata (new_label, description)
     */
    async modifyMetadata(label, newLabel, description) {
        try {
            const params = { label };
            if (newLabel) params.new_label = newLabel;
            if (description) params.description = description;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'modify_metadata', params
            );

            await this.fetchDomains();
            return response;
        } catch (error) {
            console.error('Failed to modify domain metadata:', error);
            throw error;
        }
    }

    // =========================================================================
    // Section Management Methods
    // =========================================================================

    /**
     * Get domain with all sections
     */
    async getDomainWithSections(label) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'get', { label }
            );
            return response;
        } catch (error) {
            console.error('Failed to get domain with sections:', error);
            throw error;
        }
    }

    /**
     * Get section content
     * @param {string} parent - Optional parent section header for subsections
     */
    async getSection(label, section, parent = null) {
        try {
            const params = { label, section };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'get_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to get section:', error);
            throw error;
        }
    }

    /**
     * Update section content
     * @param {string} parent - Optional parent section header for subsections
     */
    async updateSection(label, section, content, parent = null) {
        try {
            const params = { label, section, content };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'update_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to update section:', error);
            throw error;
        }
    }

    /**
     * Create new section or subsection
     * @param {string} parent - Optional parent section header to create subsection
     */
    async createSection(label, section, content, after = null, parent = null) {
        try {
            const params = { label, section, content };
            if (after) params.after = after;
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'create_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to create section:', error);
            throw error;
        }
    }

    /**
     * Rename section
     * @param {string} parent - Optional parent section header for subsections
     */
    async renameSection(label, section, newName, parent = null) {
        try {
            const params = { label, section, new_name: newName };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'rename_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to rename section:', error);
            throw error;
        }
    }

    /**
     * Delete section
     * @param {string} parent - Optional parent section header for subsections
     */
    async deleteSection(label, section, parent = null) {
        try {
            const params = { label, section };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'delete_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to delete section:', error);
            throw error;
        }
    }

    /**
     * Reorder sections at a level
     * @param {string} parent - Optional parent to reorder subsections within
     */
    async reorderSections(label, order, parent = null) {
        try {
            const params = { label, order };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'reorder_sections', params
            );
            return response;
        } catch (error) {
            console.error('Failed to reorder sections:', error);
            throw error;
        }
    }

    /**
     * Expand section
     * @param {string} parent - Optional parent section header for subsections
     */
    async expandSection(label, section, parent = null) {
        try {
            const params = { label, section };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'expand_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to expand section:', error);
            throw error;
        }
    }

    /**
     * Collapse section
     * @param {string} parent - Optional parent section header for subsections
     */
    async collapseSection(label, section, parent = null) {
        try {
            const params = { label, section };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'collapse_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to collapse section:', error);
            throw error;
        }
    }

    /**
     * Get section version history
     * @param {string} parent - Optional parent section header for subsections
     */
    async getSectionHistory(label, section, parent = null) {
        try {
            const params = { label, section };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'get_section_history', params
            );
            return response;
        } catch (error) {
            console.error('Failed to get section history:', error);
            throw error;
        }
    }

    /**
     * Rollback section to a previous version
     * @param {string} parent - Optional parent section header for subsections
     */
    async rollbackSection(label, section, versionNum, parent = null) {
        try {
            const params = { label, section, version_num: versionNum };
            if (parent) params.parent = parent;

            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'rollback_section', params
            );
            return response;
        } catch (error) {
            console.error('Failed to rollback section:', error);
            throw error;
        }
    }

    // =========================================================================
    // Sharing
    // =========================================================================

    async shareDomain(label, email) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'share', { label, email }
            );
            return response;
        } catch (error) {
            console.error('Failed to share domain:', error);
            throw error;
        }
    }

    async unshareDomain(label, email) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'unshare', { label, email }
            );
            return response;
        } catch (error) {
            console.error('Failed to unshare domain:', error);
            throw error;
        }
    }

    async listShares(label) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'list_shares', { label }
            );
            return response;
        } catch (error) {
            console.error('Failed to list shares:', error);
            throw error;
        }
    }

    async acceptShare(shareId) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'accept_share', { share_id: shareId }
            );
            return response;
        } catch (error) {
            console.error('Failed to accept share:', error);
            throw error;
        }
    }

    async rejectShare(shareId) {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'reject_share', { share_id: shareId }
            );
            return response;
        } catch (error) {
            console.error('Failed to reject share:', error);
            throw error;
        }
    }

    async listPendingShares() {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'list_pending_shares', {}
            );
            return response;
        } catch (error) {
            console.error('Failed to list pending shares:', error);
            throw error;
        }
    }

    async listSharedWithMe() {
        try {
            const response = await this.apiClient.actions.executeAction(
                'domain_knowledge', 'list_shared_with_me', {}
            );
            return response;
        } catch (error) {
            console.error('Failed to list shared with me:', error);
            throw error;
        }
    }

    /**
     * Update indicator status based on enabled domains
     */
    updateIndicatorStatus() {
        const domainBtn = document.querySelector('[data-indicator="domain_btn"]');
        if (!domainBtn) return;

        const count = this.enabledDomains.size;

        if (count > 0) {
            domainBtn.classList.remove('standby');
            domainBtn.classList.add('active');
        } else {
            domainBtn.classList.remove('active');
            domainBtn.classList.add('standby');
        }

        // Update the indicator label with count or name
        if (window.ToolbarPriorityManager) {
            let labelText = '';
            if (count === 1) {
                // Show the name of the single enabled domain
                const [singleDomain] = this.enabledDomains;
                labelText = singleDomain;
            } else if (count > 1) {
                labelText = `${count} Active`;
            }
            window.ToolbarPriorityManager.setIndicatorLabel('domain_btn', labelText);
        }
    }

    /**
     * Render domain list in popover (chat interface)
     */
    renderDomainPopover(containerElement) {
        if (!containerElement) return;

        containerElement.innerHTML = '';

        if (this.domains.length === 0) {
            containerElement.innerHTML = '<div class="domain-item empty">No domains created yet</div>';
            return;
        }

        this.domains.forEach(domain => {
            const domainItem = document.createElement('div');
            domainItem.className = 'domain-item';
            if (domain.enabled) {
                domainItem.classList.add('enabled');
            }

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = domain.enabled;

            checkbox.addEventListener('change', async () => {
                try {
                    if (checkbox.checked) {
                        await this.enableDomain(domain.label);
                    } else {
                        await this.disableDomain(domain.label);
                    }
                    this.renderDomainPopover(containerElement);
                } catch (error) {
                    const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                    alert(`Failed to ${checkbox.checked ? 'enable' : 'disable'} domain: ${errorMsg}`);
                    checkbox.checked = !checkbox.checked;
                }
            });

            const labelEl = document.createElement('label');
            labelEl.textContent = domain.label;
            if (domain.shared) {
                const sharedTag = document.createElement('span');
                sharedTag.textContent = ` (shared by ${domain.shared_by || 'another user'})`;
                sharedTag.style.cssText = 'color: #0af; font-size: 11px;';
                labelEl.appendChild(sharedTag);
            }

            domainItem.appendChild(checkbox);
            domainItem.appendChild(labelEl);
            containerElement.appendChild(domainItem);
        });

        // Show "View Archived" link only when archived docs exist
        this.apiClient.data.getData('domaindocs', { archived: true }).then(data => {
            const archivedCount = (data.domaindocs || []).length;
            if (archivedCount === 0) return;

            const archivedLink = document.createElement('a');
            archivedLink.href = '/domaindocs?archived=true';
            archivedLink.textContent = `View Archived (${archivedCount})`;
            archivedLink.style.cssText = `
                display: block;
                text-align: center;
                padding: 8px;
                margin-top: 8px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                color: var(--text-inline-muted);
                font-size: 12px;
                text-decoration: none;
            `;
            archivedLink.addEventListener('mouseenter', () => { archivedLink.style.color = '#ffa500'; });
            archivedLink.addEventListener('mouseleave', () => { archivedLink.style.color = '#888'; });
            containerElement.appendChild(archivedLink);
        }).catch(() => {});
    }

    /**
     * Render domain list in settings page with section management
     */
    async renderSettingsPage(containerElement) {
        if (!containerElement) return;

        containerElement.innerHTML = '';

        // Add create button at top
        const createSection = document.createElement('div');
        createSection.style.cssText = 'margin-bottom: 24px;';

        const createBtn = document.createElement('button');
        createBtn.textContent = '+ Create New Domain';
        createBtn.style.cssText = `
            background: rgba(0, 255, 0, 0.15);
            border-color: rgba(0, 255, 0, 0.5);
        `;
        createBtn.addEventListener('click', () => {
            this.showCreateDialog(() => {
                this.fetchDomains().then(() => this.renderSettingsPage(containerElement));
            });
        });
        createSection.appendChild(createBtn);

        // Show Archived toggle
        const archiveToggle = document.createElement('label');
        archiveToggle.style.cssText = `
            display: inline-flex; align-items: center; gap: 8px;
            margin-left: 16px; color: var(--text-inline-muted); font-size: 13px; cursor: pointer;
        `;
        const archiveCheckbox = document.createElement('input');
        archiveCheckbox.type = 'checkbox';
        archiveCheckbox.checked = this._showingArchived || false;
        archiveCheckbox.addEventListener('change', async () => {
            this._showingArchived = archiveCheckbox.checked;
            this.renderSettingsPage(containerElement);
        });
        archiveToggle.appendChild(archiveCheckbox);
        archiveToggle.appendChild(document.createTextNode('Show Archived'));
        createSection.appendChild(archiveToggle);

        containerElement.appendChild(createSection);

        // Pending invitations section
        const invitesSection = document.createElement('div');
        invitesSection.id = 'pending-invitations';
        invitesSection.style.cssText = 'margin-bottom: 24px;';
        containerElement.appendChild(invitesSection);
        this.renderPendingInvitations(invitesSection);

        if (this._showingArchived) {
            // Fetch and render archived docs
            try {
                const archivedData = await this.apiClient.data.getData('domaindocs', { archived: true });
                const archivedDocs = archivedData.domaindocs || [];

                if (archivedDocs.length === 0) {
                    const emptyMsg = document.createElement('p');
                    emptyMsg.style.cssText = 'color: var(--text-inline-muted);';
                    emptyMsg.textContent = 'No archived domains.';
                    containerElement.appendChild(emptyMsg);
                } else {
                    archivedDocs.forEach(domain => {
                        domain.archived = true;
                        this._renderDomainCard(domain, containerElement);
                    });
                }
            } catch (error) {
                console.error('Failed to fetch archived domains:', error);
            }
            return;
        }

        if (this.domains.length === 0) {
            const emptyMsg = document.createElement('p');
            emptyMsg.style.cssText = 'color: var(--text-inline-muted);';
            emptyMsg.textContent = 'No domains created yet. Click the button above to create one.';
            containerElement.appendChild(emptyMsg);
            return;
        }

        // Render each domain card (async to load sections)
        this.domains.forEach(domain => {
            this._renderDomainCard(domain, containerElement);
        });
    }

    /**
     * Render a single domain card with sections
     */
    async _renderDomainCard(domain, containerElement) {
        const domainCard = document.createElement('div');
        domainCard.className = 'domain-card';
        domainCard.style.cssText = `
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        `;

        // Header
        const header = document.createElement('div');
        header.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;';

        const titleContainer = document.createElement('div');

        const title = document.createElement('h3');
        title.textContent = domain.label;
        title.style.cssText = 'margin: 0; color: var(--text-primary); font-weight: 300;';

        titleContainer.appendChild(title);

        const statusBadge = document.createElement('span');
        const isArchived = domain.archived;
        if (isArchived) {
            statusBadge.textContent = 'Archived';
            statusBadge.style.cssText = `
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                background: rgba(255, 165, 0, 0.1);
                color: #ffa500;
                border: 1px solid rgba(255, 165, 0, 0.3);
            `;
        } else {
            statusBadge.textContent = domain.enabled ? 'Enabled' : 'Disabled';
            statusBadge.style.cssText = `
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                background: ${domain.enabled ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 255, 255, 0.05)'};
                color: ${domain.enabled ? '#0f0' : '#888'};
                border: 1px solid ${domain.enabled ? 'rgba(0, 255, 0, 0.3)' : 'rgba(255, 255, 255, 0.1)'};
            `;
        }

        header.appendChild(titleContainer);
        header.appendChild(statusBadge);

        // Description
        const description = document.createElement('p');
        description.textContent = domain.description;
        description.style.cssText = 'color: var(--text-inline-muted); font-size: 14px; margin-bottom: 16px;';

        // Sections area
        const sectionsArea = document.createElement('div');
        sectionsArea.style.cssText = `
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 16px;
        `;

        const sectionsHeader = document.createElement('div');
        sectionsHeader.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;';

        const sectionsTitle = document.createElement('span');
        sectionsTitle.textContent = 'Sections';
        sectionsTitle.style.cssText = 'color: var(--text-primary); font-size: 13px; font-weight: 500;';

        // Button container for Add and Reorder
        const headerButtons = document.createElement('div');
        headerButtons.style.cssText = 'display: flex; gap: 8px;';

        // Reorder mode state
        let isReorderMode = false;

        const reorderBtn = document.createElement('button');
        reorderBtn.textContent = 'Reorder';
        reorderBtn.style.cssText = 'font-size: 11px; padding: 4px 8px;';

        const addSectionBtn = document.createElement('button');
        addSectionBtn.textContent = '+ Add';
        addSectionBtn.style.cssText = 'font-size: 11px; padding: 4px 8px;';
        addSectionBtn.addEventListener('click', () => {
            this.showCreateSectionDialog(domain.label, () => {
                this._refreshDomainCard(domain.label, domainCard, containerElement);
            });
        });

        headerButtons.appendChild(reorderBtn);
        headerButtons.appendChild(addSectionBtn);
        sectionsHeader.appendChild(sectionsTitle);
        sectionsHeader.appendChild(headerButtons);
        sectionsArea.appendChild(sectionsHeader);

        // Section list (will be populated async)
        const sectionsList = document.createElement('div');
        sectionsList.style.cssText = 'font-size: 13px;';
        sectionsList.innerHTML = '<span style="color: var(--text-inline-dim);">Loading sections...</span>';
        sectionsArea.appendChild(sectionsList);

        // Actions
        const actions = document.createElement('div');
        actions.style.cssText = 'display: flex; gap: 8px;';

        if (isArchived) {
            // Archived docs: Unarchive + Edit Details + Delete
            const unarchiveBtn = document.createElement('button');
            unarchiveBtn.textContent = 'Unarchive';
            unarchiveBtn.style.cssText = 'background: rgba(255, 165, 0, 0.15); border-color: rgba(255, 165, 0, 0.5);';
            unarchiveBtn.addEventListener('click', async () => {
                try {
                    await this.unarchiveDomain(domain.label);
                    this.renderSettingsPage(containerElement);
                } catch (error) {
                    const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                    alert(`Failed to unarchive domain: ${errorMsg}`);
                }
            });
            actions.appendChild(unarchiveBtn);
        } else {
            // Non-archived docs: Enable/Disable + Archive
            const toggleBtn = document.createElement('button');
            toggleBtn.textContent = domain.enabled ? 'Disable' : 'Enable';
            toggleBtn.addEventListener('click', async () => {
                try {
                    if (domain.enabled) {
                        await this.disableDomain(domain.label);
                    } else {
                        await this.enableDomain(domain.label);
                    }
                    this.renderSettingsPage(containerElement);
                } catch (error) {
                    const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                    alert(`Failed to ${domain.enabled ? 'disable' : 'enable'} domain: ${errorMsg}`);
                }
            });
            actions.appendChild(toggleBtn);

            const archiveBtn = document.createElement('button');
            archiveBtn.textContent = 'Archive';
            archiveBtn.addEventListener('click', async () => {
                if (confirm(`Archive domain "${domain.label}"? It will be disabled and hidden from normal views.`)) {
                    try {
                        await this.archiveDomain(domain.label);
                        this.renderSettingsPage(containerElement);
                    } catch (error) {
                        const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                        alert(`Failed to archive domain: ${errorMsg}`);
                    }
                }
            });
            actions.appendChild(archiveBtn);
        }

        const editDetailsBtn = document.createElement('button');
        editDetailsBtn.textContent = 'Edit Details';
        editDetailsBtn.addEventListener('click', () => {
            this.showEditMetadataDialog(domain, () => {
                this.renderSettingsPage(containerElement);
            });
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.classList.add('delete-btn');
        deleteBtn.addEventListener('click', async () => {
            if (confirm(`Delete domain "${domain.label}"? This cannot be undone.`)) {
                try {
                    await this.deleteDomain(domain.label);
                    this.renderSettingsPage(containerElement);
                } catch (error) {
                    const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                    alert(`Failed to delete domain: ${errorMsg}`);
                }
            }
        });

        actions.appendChild(editDetailsBtn);

        if (!domain.shared) {
            const shareBtn = document.createElement('button');
            shareBtn.textContent = 'Share';
            shareBtn.addEventListener('click', () => {
                this.showShareDialog(domain.label, () => {
                    this.renderSettingsPage(containerElement);
                });
            });
            actions.appendChild(shareBtn);
        } else {
            const sharedBadge = document.createElement('span');
            sharedBadge.textContent = `Shared by ${domain.shared_by || 'another user'}`;
            sharedBadge.style.cssText = 'color: #0af; font-size: 12px; padding: 4px 8px; background: rgba(0,170,255,0.1); border: 1px solid rgba(0,170,255,0.3); border-radius: 4px;';
            actions.appendChild(sharedBadge);
        }

        if (!domain.shared) {
            actions.appendChild(deleteBtn);
        }

        domainCard.appendChild(header);
        domainCard.appendChild(description);
        domainCard.appendChild(sectionsArea);
        domainCard.appendChild(actions);

        containerElement.appendChild(domainCard);

        // Load sections async
        try {
            const domainData = await this.getDomainWithSections(domain.label);
            const sections = domainData.sections || [];

            // Callback for after reorder - re-renders list while staying in reorder mode
            const onReorderComplete = async () => {
                const updatedData = await this.getDomainWithSections(domain.label);
                const updatedSections = updatedData.sections || [];
                this._renderReorderableSectionsList(sectionsList, domain.label, updatedSections, domainCard, containerElement, onReorderComplete);
            };

            // Render sections (normal mode initially)
            const renderSections = async () => {
                if (isReorderMode) {
                    // Fetch fresh section data for reorder mode
                    const freshData = await this.getDomainWithSections(domain.label);
                    const freshSections = freshData.sections || [];
                    this._renderReorderableSectionsList(sectionsList, domain.label, freshSections, domainCard, containerElement, onReorderComplete);
                } else {
                    this._renderSectionsList(sectionsList, domain.label, sections, domainCard, containerElement);
                }
            };

            // Reorder button toggles mode
            reorderBtn.addEventListener('click', async () => {
                isReorderMode = !isReorderMode;
                reorderBtn.textContent = isReorderMode ? 'Done' : 'Reorder';
                reorderBtn.style.background = isReorderMode ? 'rgba(0, 255, 0, 0.15)' : '';
                reorderBtn.style.borderColor = isReorderMode ? 'rgba(0, 255, 0, 0.5)' : '';
                addSectionBtn.style.display = isReorderMode ? 'none' : '';

                if (!isReorderMode) {
                    // Exiting reorder mode - do full refresh to sync with server
                    this._refreshDomainCard(domain.label, domainCard, containerElement);
                } else {
                    await renderSections();
                }
            });

            renderSections();
        } catch (error) {
            sectionsList.innerHTML = '<span style="color: #f66;">Failed to load sections</span>';
        }
    }

    /**
     * Render section list within a domain card (supports one level of nesting)
     */
    _renderSectionsList(container, label, sections, domainCard, pageContainer) {
        container.innerHTML = '';

        if (sections.length === 0) {
            container.innerHTML = '<span style="color: var(--text-inline-dim);">No sections yet</span>';
            return;
        }

        // Separate top-level sections and subsections
        const topLevel = sections.filter(s => !s.parent_section_id);
        const subsectionsByParent = {};
        sections.forEach(s => {
            if (s.parent_section_id) {
                if (!subsectionsByParent[s.parent_section_id]) {
                    subsectionsByParent[s.parent_section_id] = [];
                }
                subsectionsByParent[s.parent_section_id].push(s);
            }
        });

        topLevel.forEach((section, index) => {
            const subsections = subsectionsByParent[section.id] || [];
            const hasChildren = subsections.length > 0;

            this._renderSectionRow(container, label, section, index, null, hasChildren, subsections.length, domainCard, pageContainer);

            // If parent is expanded, render subsections indented
            if (!section.collapsed && hasChildren) {
                subsections.forEach((subsec, subIndex) => {
                    this._renderSectionRow(container, label, subsec, subIndex, section.header, false, 0, domainCard, pageContainer);
                });
            }
        });
    }

    /**
     * Render sections in reorder mode with drag-and-drop
     */
    _renderReorderableSectionsList(container, label, sections, domainCard, pageContainer, onReorder) {
        container.innerHTML = '';

        if (sections.length === 0) {
            container.innerHTML = '<span style="color: var(--text-inline-dim);">No sections to reorder</span>';
            return;
        }

        // Separate top-level sections and subsections
        const topLevel = sections.filter(s => !s.parent_section_id);
        const subsectionsByParent = {};
        sections.forEach(s => {
            if (s.parent_section_id) {
                if (!subsectionsByParent[s.parent_section_id]) {
                    subsectionsByParent[s.parent_section_id] = [];
                }
                subsectionsByParent[s.parent_section_id].push(s);
            }
        });

        // Create drag-drop context for top-level sections
        let draggedIndex = null;
        let draggedLevel = null; // 'top' or parent section id

        const createDraggableRow = (section, index, isSubsection, parentHeader, sectionsAtLevel) => {
            const isOverview = index === 0 && !isSubsection;
            const isDraggable = !isOverview;

            const row = document.createElement('div');
            row.style.cssText = `
                display: flex;
                align-items: center;
                padding: 8px 12px;
                margin: 2px 0;
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                cursor: ${isDraggable ? 'grab' : 'default'};
                transition: all 0.15s ease;
                ${isSubsection ? 'margin-left: 24px;' : ''}
                ${!isDraggable ? 'opacity: 0.6;' : ''}
            `;

            // Drag handle
            const handle = document.createElement('span');
            handle.textContent = isDraggable ? '⠿' : '○';
            handle.style.cssText = `
                margin-right: 12px;
                color: ${isDraggable ? '#888' : '#444'};
                font-size: 14px;
                user-select: none;
            `;

            // Section name
            const name = document.createElement('span');
            name.textContent = section.header;
            name.style.cssText = `
                color: var(--text-primary);
                font-size: 13px;
                flex: 1;
            `;

            // Badges
            if (isOverview) {
                const badge = document.createElement('span');
                badge.textContent = 'pinned';
                badge.style.cssText = `
                    font-size: 10px;
                    padding: 2px 6px;
                    background: rgba(255, 255, 255, 0.05);
                    color: var(--text-inline-dim);
                    border-radius: 3px;
                    margin-left: 8px;
                `;
                name.appendChild(badge);
            }

            row.appendChild(handle);
            row.appendChild(name);

            if (isDraggable) {
                row.draggable = true;
                const levelId = isSubsection ? parentHeader : 'top';

                row.addEventListener('dragstart', (e) => {
                    draggedIndex = index;
                    draggedLevel = levelId;
                    row.style.opacity = '0.5';
                    e.dataTransfer.effectAllowed = 'move';
                    e.dataTransfer.setData('text/plain', index.toString());
                });

                row.addEventListener('dragend', () => {
                    row.style.opacity = '1';
                    draggedIndex = null;
                    draggedLevel = null;
                    // Remove all drag-over styles
                    container.querySelectorAll('[data-drag-over]').forEach(el => {
                        el.style.borderTop = '';
                        el.style.borderBottom = '';
                        el.removeAttribute('data-drag-over');
                    });
                });

                row.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    // Only allow drop on same level
                    const rowLevel = isSubsection ? parentHeader : 'top';
                    if (draggedLevel !== rowLevel) return;
                    if (draggedIndex === index) return;

                    e.dataTransfer.dropEffect = 'move';

                    // Visual indicator
                    row.setAttribute('data-drag-over', 'true');
                    if (draggedIndex < index) {
                        row.style.borderBottom = '2px solid #c0f';
                        row.style.borderTop = '';
                    } else {
                        row.style.borderTop = '2px solid #c0f';
                        row.style.borderBottom = '';
                    }
                });

                row.addEventListener('dragleave', () => {
                    row.style.borderTop = '';
                    row.style.borderBottom = '';
                    row.removeAttribute('data-drag-over');
                });

                row.addEventListener('drop', async (e) => {
                    e.preventDefault();
                    row.style.borderTop = '';
                    row.style.borderBottom = '';

                    const fromIndex = draggedIndex;
                    const toIndex = index;
                    const rowLevel = isSubsection ? parentHeader : 'top';

                    if (draggedLevel !== rowLevel || fromIndex === toIndex) return;

                    // Build new order
                    const headers = sectionsAtLevel.map(s => s.header);
                    const [moved] = headers.splice(fromIndex, 1);
                    headers.splice(toIndex, 0, moved);

                    try {
                        await this.reorderSections(label, headers, isSubsection ? parentHeader : null);
                        if (onReorder) onReorder();
                    } catch (error) {
                        const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                        alert(`Failed to reorder: ${errorMsg}`);
                    }
                });
            }

            return row;
        };

        // Render top-level sections
        topLevel.forEach((section, index) => {
            container.appendChild(createDraggableRow(section, index, false, null, topLevel));

            // Render subsections (always visible in reorder mode)
            const subsections = subsectionsByParent[section.id] || [];
            if (subsections.length > 0) {
                subsections.forEach((subsec, subIndex) => {
                    container.appendChild(createDraggableRow(subsec, subIndex, true, section.header, subsections));
                });
            }
        });
    }

    /**
     * Render a single section row (top-level or subsection)
     */
    _renderSectionRow(container, label, section, index, parentHeader, hasChildren, childCount, domainCard, pageContainer) {
        const isSubsection = parentHeader !== null;
        const isOverview = index === 0 && !isSubsection;

        // UI-only expand state (doesn't affect what MIRA sees)
        let uiExpanded = !section.collapsed;

        // Wrapper for entire section (header row + content preview)
        const sectionWrapper = document.createElement('div');
        sectionWrapper.style.cssText = `
            border-bottom: 1px solid rgba(255,255,255,0.05);
            ${isSubsection ? 'padding-left: 20px; background: rgba(0,0,0,0.1);' : ''}
        `;

        const sectionRow = document.createElement('div');
        sectionRow.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 0;
        `;

        const leftSide = document.createElement('div');
        leftSide.style.cssText = 'display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;';

        // Subsection indicator
        if (isSubsection) {
            const subIndicator = document.createElement('span');
            subIndicator.textContent = '└';
            subIndicator.style.cssText = 'color: #444; font-size: 12px;';
            leftSide.appendChild(subIndicator);
        }

        // Section header
        const headerText = document.createElement('span');
        headerText.textContent = section.header;
        headerText.style.cssText = `
            color: ${section.collapsed ? '#666' : '#c1d1d6'};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;

        // State badges
        const badges = document.createElement('span');
        badges.style.cssText = 'display: flex; gap: 4px; flex-shrink: 0;';

        if (isOverview) {
            const firstBadge = document.createElement('span');
            firstBadge.textContent = 'overview';
            firstBadge.style.cssText = `
                font-size: 10px;
                padding: 1px 4px;
                background: rgba(0, 191, 255, 0.1);
                color: #00bfff;
                border-radius: 3px;
            `;
            badges.appendChild(firstBadge);
        } else if (section.collapsed) {
            const collapsedBadge = document.createElement('span');
            collapsedBadge.textContent = 'collapsed';
            collapsedBadge.style.cssText = `
                font-size: 10px;
                padding: 1px 4px;
                background: rgba(255, 255, 255, 0.05);
                color: var(--text-inline-dim);
                border-radius: 3px;
            `;
            badges.appendChild(collapsedBadge);

            // Show subsection count for collapsed parents
            if (hasChildren) {
                collapsedBadge.textContent = `collapsed (${childCount})`;
            }
        }

        // Has children badge (for expanded parents)
        if (hasChildren && !section.collapsed) {
            const childBadge = document.createElement('span');
            childBadge.textContent = `${childCount} subs`;
            childBadge.style.cssText = `
                font-size: 10px;
                padding: 1px 4px;
                background: rgba(128, 0, 255, 0.1);
                color: #a0a;
                border-radius: 3px;
            `;
            badges.appendChild(childBadge);
        }

        // Large section warning
        const charCount = section.content ? section.content.length : 0;
        if (charCount > 5000) {
            const largeBadge = document.createElement('span');
            largeBadge.textContent = 'large';
            largeBadge.style.cssText = `
                font-size: 10px;
                padding: 1px 4px;
                background: rgba(255, 165, 0, 0.1);
                color: #ffa500;
                border-radius: 3px;
            `;
            badges.appendChild(largeBadge);
        }

        leftSide.appendChild(headerText);
        leftSide.appendChild(badges);

        // Action buttons - kept minimal: Edit + overflow menu
        const rightSide = document.createElement('div');
        rightSide.style.cssText = 'display: flex; gap: 4px; flex-shrink: 0; align-items: center;';

        // Edit button (primary action)
        const editBtn = document.createElement('button');
        editBtn.textContent = 'Edit';
        editBtn.style.cssText = 'font-size: 10px; padding: 2px 6px; min-width: auto;';
        editBtn.addEventListener('click', () => {
            this.showSectionEditDialog(label, section.header, () => {
                this._refreshDomainCard(label, domainCard, pageContainer);
            }, parentHeader);
        });
        rightSide.appendChild(editBtn);

        // Overflow menu for secondary actions
        const menuContainer = document.createElement('div');
        menuContainer.style.cssText = 'position: relative; display: flex; align-items: center;';

        const menuBtn = document.createElement('button');
        menuBtn.textContent = '⋮';
        menuBtn.title = 'More actions';
        menuBtn.style.cssText = 'font-size: 12px; padding: 2px 6px; min-width: auto; line-height: 1;';

        const menuDropdown = document.createElement('div');
        menuDropdown.style.cssText = `
            display: none;
            position: absolute;
            right: 0;
            top: 100%;
            background: #0a0f14;
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 4px 0;
            min-width: 140px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        `;

        const createMenuItem = (text, onClick, disabled = false, isDanger = false) => {
            const item = document.createElement('div');
            item.textContent = text;
            item.style.cssText = `
                padding: 8px 12px;
                cursor: ${disabled ? 'not-allowed' : 'pointer'};
                color: ${disabled ? '#444' : (isDanger ? '#f66' : '#c1d1d6')};
                font-size: 12px;
                ${disabled ? '' : 'transition: background 0.1s;'}
            `;
            if (!disabled) {
                item.addEventListener('mouseenter', () => item.style.background = 'rgba(204, 0, 255, 0.1)');
                item.addEventListener('mouseleave', () => item.style.background = 'none');
                item.addEventListener('click', () => {
                    menuDropdown.style.display = 'none';
                    onClick();
                });
            }
            return item;
        };

        // Add subsection (only for non-overview top-level sections)
        if (!isSubsection && !isOverview) {
            menuDropdown.appendChild(createMenuItem('Add subsection', () => {
                this.showCreateSectionDialog(label, () => {
                    this._refreshDomainCard(label, domainCard, pageContainer);
                }, section.header);
            }));
        }

        // History
        menuDropdown.appendChild(createMenuItem('History', () => {
            this.showSectionHistoryDialog(label, section.header, () => {
                this._refreshDomainCard(label, domainCard, pageContainer);
            }, parentHeader);
        }));

        // Delete (not for overview)
        if (!isOverview) {
            menuDropdown.appendChild(createMenuItem('Delete', async () => {
                const confirmMsg = hasChildren
                    ? `Delete section "${section.header}" and its ${childCount} subsection(s)?`
                    : `Delete section "${section.header}"?`;
                if (confirm(confirmMsg)) {
                    try {
                        await this.deleteSection(label, section.header, parentHeader);
                        this._refreshDomainCard(label, domainCard, pageContainer);
                    } catch (error) {
                        const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                        alert(`Failed to delete: ${errorMsg}`);
                    }
                }
            }, false, true));
        }

        menuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = menuDropdown.style.display === 'block';
            menuDropdown.style.display = isOpen ? 'none' : 'block';
        });

        // Close menu on outside click
        document.addEventListener('click', () => {
            menuDropdown.style.display = 'none';
        });

        menuContainer.appendChild(menuBtn);
        menuContainer.appendChild(menuDropdown);
        rightSide.appendChild(menuContainer);

        sectionRow.appendChild(leftSide);
        sectionRow.appendChild(rightSide);
        sectionWrapper.appendChild(sectionRow);

        // Content preview - UI-only, independent of MIRA's collapsed state
        // User can preview any section's content for viewing purposes
        const content = section.content || '';
        if (content.length > 0) {
            const previewContainer = document.createElement('div');
            previewContainer.style.cssText = `
                padding: 8px 12px;
                margin: 0 0 8px ${isSubsection ? '28px' : '0'};
                background: rgba(0, 0, 0, 0.2);
                border-radius: 4px;
                border-left: 2px solid rgba(204, 0, 255, 0.2);
                display: ${uiExpanded ? 'block' : 'none'};
            `;

            // Track text expansion state (show more/less within preview)
            let isTextExpanded = false;
            const PREVIEW_LENGTH = 150;
            const needsTruncation = content.length > PREVIEW_LENGTH;

            const contentPreview = document.createElement('div');
            contentPreview.style.cssText = `
                font-family: 'Roboto', sans-serif;
                font-size: 15px;
                line-height: 1.5;
                letter-spacing: 0.2px;
                color: #a0a8ab;
                white-space: pre-wrap;
                word-break: break-word;
            `;

            const updatePreview = () => {
                if (isTextExpanded || !needsTruncation) {
                    contentPreview.textContent = content;
                    contentPreview.style.maxHeight = '300px';
                    contentPreview.style.overflow = 'auto';
                } else {
                    contentPreview.textContent = content.slice(0, PREVIEW_LENGTH).trim() + '...';
                    contentPreview.style.maxHeight = 'none';
                    contentPreview.style.overflow = 'hidden';
                }
            };
            updatePreview();

            previewContainer.appendChild(contentPreview);

            // Expand/collapse toggle for long content
            if (needsTruncation) {
                const toggleLink = document.createElement('button');
                toggleLink.style.cssText = `
                    background: none;
                    border: none;
                    color: #c0f;
                    font-size: 11px;
                    padding: 4px 0;
                    cursor: pointer;
                    margin-top: 4px;
                `;
                toggleLink.textContent = 'Show more';

                toggleLink.addEventListener('click', () => {
                    isTextExpanded = !isTextExpanded;
                    updatePreview();
                    toggleLink.textContent = isTextExpanded ? 'Show less' : 'Show more';
                });

                previewContainer.appendChild(toggleLink);
            }

            sectionWrapper.appendChild(previewContainer);

            // Add preview toggle button to header row (inserted before Edit button)
            const previewToggleBtn = document.createElement('button');
            previewToggleBtn.textContent = uiExpanded ? '👁' : '👁‍🗨';
            previewToggleBtn.title = uiExpanded ? 'Hide preview' : 'Show preview';
            previewToggleBtn.style.cssText = 'font-size: 10px; padding: 2px 6px; min-width: auto;';
            previewToggleBtn.addEventListener('click', () => {
                uiExpanded = !uiExpanded;
                previewContainer.style.display = uiExpanded ? 'block' : 'none';
                previewToggleBtn.textContent = uiExpanded ? '👁' : '👁‍🗨';
                previewToggleBtn.title = uiExpanded ? 'Hide preview' : 'Show preview';
            });
            // Insert at beginning of rightSide (before reorder buttons)
            rightSide.insertBefore(previewToggleBtn, rightSide.firstChild);
        }

        container.appendChild(sectionWrapper);
    }

    /**
     * Refresh a single domain card after section changes
     */
    async _refreshDomainCard(label, domainCard, containerElement) {
        const domain = this.domains.find(d => d.label === label);
        if (!domain) return;

        const newCard = document.createElement('div');
        await this._renderDomainCard(domain, { appendChild: (el) => newCard.appendChild(el) });
        domainCard.replaceWith(newCard.firstChild);
    }

    /**
     * Show section edit dialog
     * @param {string} parentHeader - Optional parent section header for subsections
     */
    async showSectionEditDialog(label, sectionHeader, onSaved, parentHeader = null) {
        let sectionData;
        try {
            sectionData = await this.getSection(label, sectionHeader, parentHeader);
        } catch (error) {
            alert(`Failed to load section: ${error.message}`);
            return;
        }

        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const dialogContent = document.createElement('div');
        dialogContent.style.cssText = `
            background: #0a0f14;
            border: 1px solid #c0f;
            border-radius: 12px;
            padding: 24px;
            max-width: 700px;
            width: 90%;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
        `;

        const dialogHeader = document.createElement('h2');
        dialogHeader.textContent = 'Edit Section';
        dialogHeader.style.cssText = 'margin: 0 0 16px 0; color: var(--text-primary);';

        // Section name input
        const nameLabel = document.createElement('label');
        nameLabel.textContent = 'Section Name:';
        nameLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.value = sectionHeader;
        nameInput.style.cssText = `
            width: 100%;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            box-sizing: border-box;
        `;

        // Content textarea
        const contentLabel = document.createElement('label');
        contentLabel.textContent = 'Content:';
        contentLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const textarea = document.createElement('textarea');
        textarea.value = sectionData.content || '';
        textarea.style.cssText = `
            flex: 1;
            min-height: 350px;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 12px;
            font-family: 'Roboto', sans-serif;
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
            margin-bottom: 16px;
        `;

        const buttonRow = document.createElement('div');
        buttonRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => document.body.removeChild(dialog));

        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Save';
        saveBtn.style.cssText = 'background: rgba(0, 255, 0, 0.15); border-color: rgba(0, 255, 0, 0.5);';
        saveBtn.addEventListener('click', async () => {
            try {
                saveBtn.disabled = true;
                saveBtn.textContent = 'Saving...';

                const newName = nameInput.value.trim();
                let currentName = sectionHeader;

                // Rename first if name changed
                if (newName && newName !== sectionHeader) {
                    await this.renameSection(label, sectionHeader, newName, parentHeader);
                    currentName = newName;
                }

                // Update content with current name
                await this.updateSection(label, currentName, textarea.value, parentHeader);
                document.body.removeChild(dialog);
                if (onSaved) onSaved();
            } catch (error) {
                saveBtn.disabled = false;
                saveBtn.textContent = 'Save';
                const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                alert(`Failed to save: ${errorMsg}`);
            }
        });

        buttonRow.appendChild(cancelBtn);
        buttonRow.appendChild(saveBtn);

        dialogContent.appendChild(dialogHeader);
        dialogContent.appendChild(nameLabel);
        dialogContent.appendChild(nameInput);
        dialogContent.appendChild(contentLabel);
        dialogContent.appendChild(textarea);
        dialogContent.appendChild(buttonRow);
        dialog.appendChild(dialogContent);

        document.body.appendChild(dialog);
        nameInput.focus();
        nameInput.select();
    }

    /**
     * Show create section dialog
     * @param {string} parentHeader - Optional parent to create subsection under
     */
    showCreateSectionDialog(label, onCreated, parentHeader = null) {
        const isSubsection = parentHeader !== null;

        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const dialogContent = document.createElement('div');
        dialogContent.style.cssText = `
            background: #0a0f14;
            border: 1px solid #c0f;
            border-radius: 12px;
            padding: 24px;
            max-width: 500px;
            width: 90%;
        `;

        const dialogHeader = document.createElement('h2');
        dialogHeader.textContent = isSubsection ? `Add Subsection to "${parentHeader}"` : 'Create New Section';
        dialogHeader.style.cssText = 'margin: 0 0 16px 0; color: var(--text-primary);';

        const headerLabel = document.createElement('label');
        headerLabel.textContent = 'Section Header:';
        headerLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const headerInput = document.createElement('input');
        headerInput.type = 'text';
        headerInput.placeholder = 'e.g., RESEARCH FINDINGS';
        headerInput.style.cssText = `
            width: 100%;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            box-sizing: border-box;
        `;

        const contentLabel = document.createElement('label');
        contentLabel.textContent = 'Initial Content:';
        contentLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const contentInput = document.createElement('textarea');
        contentInput.placeholder = 'Section content...';
        contentInput.style.cssText = `
            width: 100%;
            min-height: 120px;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            box-sizing: border-box;
        `;

        const buttonRow = document.createElement('div');
        buttonRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => document.body.removeChild(dialog));

        const createBtn = document.createElement('button');
        createBtn.textContent = 'Create';
        createBtn.style.cssText = 'background: rgba(0, 255, 0, 0.15); border-color: rgba(0, 255, 0, 0.5);';
        createBtn.addEventListener('click', async () => {
            const header = headerInput.value.trim();
            const content = contentInput.value;

            if (!header) {
                alert('Please provide a section header');
                return;
            }

            try {
                createBtn.disabled = true;
                createBtn.textContent = 'Creating...';
                await this.createSection(label, header, content, null, parentHeader);
                document.body.removeChild(dialog);
                if (onCreated) onCreated();
            } catch (error) {
                createBtn.disabled = false;
                createBtn.textContent = 'Create';
                const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                alert(`Failed to create section: ${errorMsg}`);
            }
        });

        buttonRow.appendChild(cancelBtn);
        buttonRow.appendChild(createBtn);

        dialogContent.appendChild(dialogHeader);
        dialogContent.appendChild(headerLabel);
        dialogContent.appendChild(headerInput);
        dialogContent.appendChild(contentLabel);
        dialogContent.appendChild(contentInput);
        dialogContent.appendChild(buttonRow);
        dialog.appendChild(dialogContent);

        document.body.appendChild(dialog);
        headerInput.focus();
    }

    /**
     * Show section version history dialog with rollback capability
     * @param {string} parentHeader - Optional parent section header for subsections
     */
    async showSectionHistoryDialog(label, sectionHeader, onRollback, parentHeader = null) {
        let historyData;
        try {
            historyData = await this.getSectionHistory(label, sectionHeader, parentHeader);
        } catch (error) {
            alert(`Failed to load history: ${error.message}`);
            return;
        }

        const versions = historyData.versions || [];

        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const dialogContent = document.createElement('div');
        dialogContent.style.cssText = `
            background: #0a0f14;
            border: 1px solid #c0f;
            border-radius: 12px;
            padding: 24px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
        `;

        const dialogHeader = document.createElement('h2');
        dialogHeader.textContent = `History: ${sectionHeader}`;
        dialogHeader.style.cssText = 'margin: 0 0 16px 0; color: var(--text-primary);';

        const versionsList = document.createElement('div');
        versionsList.style.cssText = `
            flex: 1;
            overflow-y: auto;
            margin-bottom: 16px;
        `;

        if (versions.length === 0) {
            versionsList.innerHTML = '<p style="color: var(--text-inline-dim);">No version history available for this section.</p>';
        } else {
            versions.forEach(version => {
                const versionRow = document.createElement('div');
                versionRow.style.cssText = `
                    padding: 12px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                `;

                const versionInfo = document.createElement('div');
                versionInfo.style.cssText = 'flex: 1;';

                const versionHeader = document.createElement('div');
                versionHeader.style.cssText = 'display: flex; align-items: center; gap: 8px; margin-bottom: 4px;';

                const versionNum = document.createElement('span');
                versionNum.textContent = `v${version.version_num}`;
                versionNum.style.cssText = 'color: var(--text-primary); font-weight: 500;';

                const operationBadge = document.createElement('span');
                operationBadge.textContent = version.operation;
                operationBadge.style.cssText = `
                    font-size: 11px;
                    padding: 2px 6px;
                    background: rgba(204, 0, 255, 0.1);
                    color: #c0f;
                    border-radius: 3px;
                `;

                versionHeader.appendChild(versionNum);
                versionHeader.appendChild(operationBadge);

                const timestamp = document.createElement('div');
                timestamp.textContent = version.created_at ? new Date(version.created_at).toLocaleString() : 'Unknown time';
                timestamp.style.cssText = 'color: var(--text-inline-dim); font-size: 12px;';

                // Diff preview
                const diffPreview = document.createElement('div');
                diffPreview.style.cssText = 'color: var(--text-inline-muted); font-size: 12px; margin-top: 4px;';
                if (version.diff_data) {
                    const data = version.diff_data;
                    if (version.operation === 'append') {
                        diffPreview.textContent = `Added ${data.appended_length || '?'} chars`;
                    } else if (version.operation === 'sed' || version.operation === 'sed_all') {
                        diffPreview.textContent = `Replaced "${data.find || '?'}" → "${data.replace || ''}" (${data.replacements || 1}x)`;
                    } else if (version.operation === 'replace_section') {
                        diffPreview.textContent = `Full content replaced (${data.old_length || '?'} → ${data.new_length || '?'} chars)`;
                    } else if (version.operation === 'rollback') {
                        diffPreview.textContent = `Restored from v${data.rolled_back_to || '?'}`;
                    } else if (version.operation === 'create_section') {
                        diffPreview.textContent = `Section created (${data.content_length || '?'} chars)`;
                    }
                }

                versionInfo.appendChild(versionHeader);
                versionInfo.appendChild(timestamp);
                if (diffPreview.textContent) {
                    versionInfo.appendChild(diffPreview);
                }

                // Rollback button (only if version has previous_content)
                const actions = document.createElement('div');
                if (version.diff_data && version.diff_data.previous_content !== undefined) {
                    const rollbackBtn = document.createElement('button');
                    rollbackBtn.textContent = 'Restore';
                    rollbackBtn.style.cssText = 'font-size: 11px; padding: 4px 8px;';
                    rollbackBtn.addEventListener('click', async () => {
                        if (confirm(`Restore section to the state before v${version.version_num}?`)) {
                            try {
                                rollbackBtn.disabled = true;
                                rollbackBtn.textContent = 'Restoring...';
                                await this.rollbackSection(label, sectionHeader, version.version_num, parentHeader);
                                document.body.removeChild(dialog);
                                alert('Section restored successfully');
                                if (onRollback) onRollback();
                            } catch (error) {
                                rollbackBtn.disabled = false;
                                rollbackBtn.textContent = 'Restore';
                                const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                                alert(`Failed to restore: ${errorMsg}`);
                            }
                        }
                    });
                    actions.appendChild(rollbackBtn);
                }

                versionRow.appendChild(versionInfo);
                versionRow.appendChild(actions);
                versionsList.appendChild(versionRow);
            });
        }

        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Close';
        closeBtn.addEventListener('click', () => document.body.removeChild(dialog));

        dialogContent.appendChild(dialogHeader);
        dialogContent.appendChild(versionsList);
        dialogContent.appendChild(closeBtn);
        dialog.appendChild(dialogContent);

        document.body.appendChild(dialog);
    }

    /**
     * Show edit metadata dialog for domain name/description
     */
    showEditMetadataDialog(domain, onSaved) {
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const dialogContent = document.createElement('div');
        dialogContent.style.cssText = `
            background: #0a0f14;
            border: 1px solid #c0f;
            border-radius: 12px;
            padding: 24px;
            max-width: 500px;
            width: 90%;
        `;

        const dialogHeader = document.createElement('h2');
        dialogHeader.textContent = `Edit Details: ${domain.label}`;
        dialogHeader.style.cssText = 'margin: 0 0 16px 0; color: var(--text-primary);';

        const labelLabel = document.createElement('label');
        labelLabel.textContent = 'Label:';
        labelLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const labelInput = document.createElement('input');
        labelInput.type = 'text';
        labelInput.value = domain.label;
        labelInput.style.cssText = `
            width: 100%;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            box-sizing: border-box;
        `;

        const descLabel = document.createElement('label');
        descLabel.textContent = 'Description (guidance for MIRA):';
        descLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const descInput = document.createElement('textarea');
        descInput.value = domain.description;
        descInput.style.cssText = `
            width: 100%;
            min-height: 120px;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            font-family: inherit;
            box-sizing: border-box;
        `;

        const buttonRow = document.createElement('div');
        buttonRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => document.body.removeChild(dialog));

        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Save';
        saveBtn.style.cssText = `
            background: rgba(0, 255, 0, 0.15);
            border-color: rgba(0, 255, 0, 0.5);
        `;
        saveBtn.addEventListener('click', async () => {
            const rawLabel = labelInput.value.trim();
            const newDescription = descInput.value.trim();

            if (!rawLabel && !newDescription) {
                alert('Please provide at least a label or description');
                return;
            }

            // Normalize label: convert spaces to underscores, lowercase
            const newLabel = rawLabel ? rawLabel.replace(/\s+/g, '_').toLowerCase() : '';

            // Update input to show normalized value
            if (rawLabel) {
                labelInput.value = newLabel;
            }

            // Validate label format if changing
            if (newLabel && !/^[a-z][a-z0-9_]*$/.test(newLabel)) {
                alert('Label must start with a letter and contain only lowercase letters, numbers, and underscores');
                return;
            }

            // Only pass newLabel if it changed
            const labelChanged = newLabel && newLabel !== domain.label;

            try {
                saveBtn.disabled = true;
                saveBtn.textContent = 'Saving...';
                await this.modifyMetadata(
                    domain.label,
                    labelChanged ? newLabel : null,
                    newDescription || null
                );
                document.body.removeChild(dialog);
                if (onSaved) onSaved();
            } catch (error) {
                saveBtn.disabled = false;
                saveBtn.textContent = 'Save';
                const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                alert(`Failed to update: ${errorMsg}`);
            }
        });

        buttonRow.appendChild(cancelBtn);
        buttonRow.appendChild(saveBtn);

        dialogContent.appendChild(dialogHeader);
        dialogContent.appendChild(labelLabel);
        dialogContent.appendChild(labelInput);
        dialogContent.appendChild(descLabel);
        dialogContent.appendChild(descInput);
        dialogContent.appendChild(buttonRow);
        dialog.appendChild(dialogContent);

        document.body.appendChild(dialog);
        labelInput.focus();
    }

    /**
     * Show create domain dialog
     */
    showCreateDialog(onCreated) {
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const dialogContent = document.createElement('div');
        dialogContent.style.cssText = `
            background: #0a0f14;
            border: 1px solid #c0f;
            border-radius: 12px;
            padding: 24px;
            max-width: 500px;
            width: 90%;
        `;

        const dialogHeader = document.createElement('h2');
        dialogHeader.textContent = 'Create New Domain';
        dialogHeader.style.cssText = 'margin: 0 0 16px 0; color: var(--text-primary);';

        const labelLabel = document.createElement('label');
        labelLabel.textContent = 'Label:';
        labelLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const labelInput = document.createElement('input');
        labelInput.type = 'text';
        labelInput.placeholder = 'e.g., garden, work_notes';
        labelInput.style.cssText = `
            width: 100%;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            box-sizing: border-box;
        `;

        const descLabel = document.createElement('label');
        descLabel.textContent = 'Description (brief - will be expanded by AI):';
        descLabel.style.cssText = 'display: block; margin-bottom: 8px; color: var(--text-primary);';

        const descInput = document.createElement('textarea');
        descInput.placeholder = 'e.g., plants, pests, where I buy stuff';
        descInput.style.cssText = `
            width: 100%;
            min-height: 80px;
            background: #000;
            color: var(--text-primary);
            border: 1px solid rgba(204, 0, 255, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            font-family: inherit;
            box-sizing: border-box;
        `;

        const buttonRow = document.createElement('div');
        buttonRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => document.body.removeChild(dialog));

        const createBtn = document.createElement('button');
        createBtn.textContent = 'Create';
        createBtn.style.cssText = `
            background: rgba(0, 255, 0, 0.15);
            border-color: rgba(0, 255, 0, 0.5);
        `;
        createBtn.addEventListener('click', async () => {
            const rawLabel = labelInput.value.trim();
            const description = descInput.value.trim();

            if (!rawLabel || !description) {
                alert('Please provide both label and description');
                return;
            }

            // Normalize label: convert spaces to underscores, lowercase
            const label = rawLabel.replace(/\s+/g, '_').toLowerCase();

            // Update input to show normalized value
            labelInput.value = label;

            // Validate label format (alphanumeric and underscores)
            if (!/^[a-z][a-z0-9_]*$/.test(label)) {
                alert('Label must start with a letter and contain only lowercase letters, numbers, and underscores');
                return;
            }

            // Check for collision with existing domains
            const existingLabels = this.domains.map(d => d.label);
            if (existingLabels.includes(label)) {
                alert(`A domain with label "${label}" already exists.`);
                return;
            }

            try {
                createBtn.disabled = true;
                createBtn.textContent = 'Creating...';
                await this.createDomain(label, description);
                document.body.removeChild(dialog);
                if (onCreated) onCreated();
            } catch (error) {
                createBtn.disabled = false;
                createBtn.textContent = 'Create';
                const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                alert(`Failed to create domain: ${errorMsg}`);
            }
        });

        buttonRow.appendChild(cancelBtn);
        buttonRow.appendChild(createBtn);

        dialogContent.appendChild(dialogHeader);
        dialogContent.appendChild(labelLabel);
        dialogContent.appendChild(labelInput);
        dialogContent.appendChild(descLabel);
        dialogContent.appendChild(descInput);
        dialogContent.appendChild(buttonRow);
        dialog.appendChild(dialogContent);

        document.body.appendChild(dialog);
        labelInput.focus();
    }

    /**
     * Show share dialog for a domaindoc
     */
    showShareDialog(label, onShared) {
        const dialog = document.createElement('div');
        dialog.className = 'modal-overlay';
        dialog.style.cssText = 'display: flex; position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 10000; align-items: center; justify-content: center;';

        const dialogContent = document.createElement('div');
        dialogContent.className = 'modal-dialog modal-dialog--sm';
        dialogContent.style.cssText = 'background: #1a1a2e; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 24px; width: 420px; max-height: 80vh; overflow-y: auto;';

        const header = document.createElement('h3');
        header.textContent = `Share "${label}"`;
        header.style.cssText = 'margin: 0 0 16px 0; color: var(--text-primary); font-weight: 300;';

        const inviteSection = document.createElement('div');
        inviteSection.style.cssText = 'margin-bottom: 20px;';

        const emailLabel = document.createElement('label');
        emailLabel.textContent = "Collaborator's email address";
        emailLabel.style.cssText = 'display: block; color: var(--text-inline-muted); font-size: 13px; margin-bottom: 6px;';

        const emailInput = document.createElement('input');
        emailInput.type = 'email';
        emailInput.placeholder = 'collaborator@example.com';
        emailInput.style.cssText = 'width: 100%; padding: 8px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: var(--text-primary); font-size: 14px; box-sizing: border-box;';

        const inviteBtn = document.createElement('button');
        inviteBtn.textContent = 'Send Invitation';
        inviteBtn.style.cssText = 'margin-top: 8px; padding: 8px 16px; background: rgba(0,170,255,0.15); border: 1px solid rgba(0,170,255,0.5); border-radius: 4px; color: #0af; cursor: pointer;';

        const inviteStatus = document.createElement('div');
        inviteStatus.style.cssText = 'margin-top: 8px; font-size: 12px;';

        inviteBtn.addEventListener('click', async () => {
            const email = emailInput.value.trim();
            if (!email) {
                inviteStatus.textContent = 'Please enter an email address.';
                inviteStatus.style.color = '#f44';
                return;
            }
            try {
                inviteBtn.disabled = true;
                inviteBtn.textContent = 'Sending...';
                const result = await this.shareDomain(label, email);
                inviteStatus.textContent = `Invitation sent to ${email} (status: ${result.status})`;
                inviteStatus.style.color = '#0f0';
                emailInput.value = '';
                this._loadShareList(shareList, label);
            } catch (error) {
                const errorMsg = error.response?.data?.error?.message || error.message || 'Unknown error';
                inviteStatus.textContent = errorMsg;
                inviteStatus.style.color = '#f44';
                inviteBtn.disabled = false;
                inviteBtn.textContent = 'Send Invitation';
            }
        });

        inviteSection.appendChild(emailLabel);
        inviteSection.appendChild(emailInput);
        inviteSection.appendChild(inviteBtn);
        inviteSection.appendChild(inviteStatus);

        const shareHeader = document.createElement('h4');
        shareHeader.textContent = 'Current collaborators';
        shareHeader.style.cssText = 'margin: 20px 0 8px 0; color: var(--text-primary); font-size: 14px; font-weight: 500;';

        const shareList = document.createElement('div');
        shareList.style.cssText = 'font-size: 13px;';
        shareList.innerHTML = '<span style="color: var(--text-inline-dim);">Loading...</span>';

        const cancelButton = document.createElement('button');
        cancelButton.textContent = 'Close';
        cancelButton.style.cssText = 'margin-top: 16px; padding: 8px 16px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: var(--text-inline-muted); cursor: pointer;';
        cancelButton.addEventListener('click', () => {
            document.body.removeChild(dialog);
            if (onShared) onShared();
        });

        dialogContent.appendChild(header);
        dialogContent.appendChild(inviteSection);
        dialogContent.appendChild(shareHeader);
        dialogContent.appendChild(shareList);
        dialogContent.appendChild(cancelButton);
        dialog.appendChild(dialogContent);
        document.body.appendChild(dialog);

        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) {
                document.body.removeChild(dialog);
                if (onShared) onShared();
            }
        });

        this._loadShareList(shareList, label);
    }

    async _loadShareList(container, label) {
        try {
            const result = await this.listShares(label);
            const shares = result.shares || [];
            if (shares.length === 0) {
                container.innerHTML = '<span style="color: var(--text-inline-dim);">No collaborators yet.</span>';
                return;
            }
            container.innerHTML = '';
            shares.forEach(share => {
                const row = document.createElement('div');
                row.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);';

                const info = document.createElement('div');
                const emailSpan = document.createElement('span');
                emailSpan.textContent = share.email || '';
                emailSpan.style.cssText = 'color: var(--text-primary);';
                const statusSpan = document.createElement('span');
                statusSpan.textContent = ` (${share.status || 'unknown'})`;
                statusSpan.style.cssText = 'color: var(--text-inline-muted); font-size: 11px;';
                info.appendChild(emailSpan);
                info.appendChild(statusSpan);

                const actions = document.createElement('div');
                if (share.status === 'pending' || share.status === 'accepted') {
                    const revokeBtn = document.createElement('button');
                    revokeBtn.textContent = 'Revoke';
                    revokeBtn.style.cssText = 'font-size: 11px; padding: 2px 8px; background: rgba(255,0,0,0.1); border: 1px solid rgba(255,0,0,0.3); border-radius: 3px; color: #f66; cursor: pointer;';
                    revokeBtn.addEventListener('click', async () => {
                        try {
                            await this.unshareDomain(label, share.email);
                            this._loadShareList(container, label);
                        } catch (error) {
                            alert(`Failed to revoke: ${error.response?.data?.error?.message || error.message}`);
                        }
                    });
                    actions.appendChild(revokeBtn);
                }

                row.appendChild(info);
                row.appendChild(actions);
                container.appendChild(row);
            });
        } catch (error) {
            container.innerHTML = '<span style="color: #f44;">Failed to load collaborators.</span>';
        }
    }

    /**
     * Show pending share invitations
     */
    async renderPendingInvitations(containerElement) {
        containerElement.innerHTML = '';

        const header = document.createElement('h3');
        header.textContent = 'Pending Invitations';
        header.style.cssText = 'color: var(--text-primary); font-weight: 300; margin-bottom: 12px;';

        containerElement.appendChild(header);

        try {
            const result = await this.listPendingShares();
            const pending = result.pending_shares || [];

            if (pending.length === 0) {
                const emptyMsg = document.createElement('p');
                emptyMsg.textContent = 'No pending invitations.';
                emptyMsg.style.cssText = 'color: var(--text-inline-dim); font-size: 14px;';
                containerElement.appendChild(emptyMsg);
                return;
            }

            pending.forEach(share => {
                const card = document.createElement('div');
                card.style.cssText = `
                    background: rgba(0, 170, 255, 0.05);
                    border: 1px solid rgba(0, 170, 255, 0.2);
                    border-radius: 6px;
                    padding: 12px;
                    margin-bottom: 8px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                `;

                const info = document.createElement('div');
                const labelEl = document.createElement('div');
                labelEl.textContent = share.label || '';
                labelEl.style.cssText = 'color: var(--text-primary); font-weight: 500;';
                const invitedByEl = document.createElement('div');
                invitedByEl.textContent = `Invited by ${share.from_name || share.from_email || 'another user'}`;
                invitedByEl.style.cssText = 'color: var(--text-inline-muted); font-size: 12px;';
                info.appendChild(labelEl);
                info.appendChild(invitedByEl);

                const actions = document.createElement('div');
                actions.style.cssText = 'display: flex; gap: 8px;';

                const acceptBtn = document.createElement('button');
                acceptBtn.textContent = 'Accept';
                acceptBtn.style.cssText = 'background: rgba(0,255,0,0.15); border-color: rgba(0,255,0,0.5); font-size: 12px; padding: 4px 12px;';
                acceptBtn.addEventListener('click', async () => {
                    try {
                        await this.acceptShare(share.id);
                        card.remove();
                        await this.fetchDomains();
                        this.renderSettingsPage(containerElement.closest('.page-content') || containerElement.parentElement);
                    } catch (error) {
                        alert(`Failed to accept: ${error.response?.data?.error?.message || error.message}`);
                    }
                });

                const rejectBtn = document.createElement('button');
                rejectBtn.textContent = 'Reject';
                rejectBtn.style.cssText = 'background: rgba(255,0,0,0.1); border-color: rgba(255,0,0,0.3); color: #f66; font-size: 12px; padding: 4px 12px;';
                rejectBtn.addEventListener('click', async () => {
                    try {
                        await this.rejectShare(share.id);
                        card.remove();
                    } catch (error) {
                        alert(`Failed to reject: ${error.response?.data?.error?.message || error.message}`);
                    }
                });

                actions.appendChild(acceptBtn);
                actions.appendChild(rejectBtn);

                card.appendChild(info);
                card.appendChild(actions);
                containerElement.appendChild(card);
            });
        } catch (error) {
            const errorMsg = document.createElement('p');
            errorMsg.textContent = 'Failed to load invitations.';
            errorMsg.style.cssText = 'color: #f44;';
            containerElement.appendChild(errorMsg);
        }
    }
}

/**
 * Initialize domain knowledge UI components.
 * Called from core.js after apiClient is ready.
 */
function initDomainKnowledge(apiClient) {
    const domainManager = new DomainKnowledgeManager(apiClient);
    window.domainManager = domainManager;

    const domainBtn = document.querySelector('[data-indicator="domain_btn"]');
    const domainPopover = document.getElementById('domain-popover');
    const domainList = document.getElementById('domain-list');
    const domainCreateBtn = document.getElementById('domain-create-btn');
    const domainPopoverClose = domainPopover?.querySelector('.popover-close');

    // Fetch domains on init to set button indicator state correctly
    // (shows active state if any domain is enabled)
    domainManager.fetchDomains().catch(error => {
        console.error('Failed to fetch initial domain state:', error);
    });

    if (domainBtn && domainPopover) {
        domainBtn.addEventListener('click', async () => {
            domainPopover.classList.toggle('active');
            if (domainPopover.classList.contains('active')) {
                try {
                    await domainManager.fetchDomains();
                    domainManager.renderDomainPopover(domainList);
                } catch (error) {
                    console.error('Failed to load domains:', error);
                    if (domainList) {
                        domainList.innerHTML = '<div class="domain-item error">Failed to load domains</div>';
                    }
                }
            }
        });

        if (domainPopoverClose) {
            domainPopoverClose.addEventListener('click', () => {
                domainPopover.classList.remove('active');
            });
        }

        // Close popover on click outside
        document.addEventListener('click', (e) => {
            if (domainPopover.classList.contains('active') &&
                !domainPopover.contains(e.target) &&
                !domainBtn.contains(e.target)) {
                domainPopover.classList.remove('active');
            }
        });

        if (domainCreateBtn) {
            domainCreateBtn.addEventListener('click', () => {
                domainPopover.classList.remove('active');
                domainManager.showCreateDialog(async () => {
                    try {
                        await domainManager.fetchDomains();
                        domainManager.renderDomainPopover(domainList);
                    } catch (error) {
                        console.error('Failed to refresh domains after create:', error);
                    }
                });
            });
        }
    }

    return domainManager;
}

// Make globally available
if (typeof window !== 'undefined') {
    window.DomainKnowledgeManager = DomainKnowledgeManager;
    window.initDomainKnowledge = initDomainKnowledge;
}
