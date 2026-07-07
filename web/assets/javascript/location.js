/**
 * LOCATION.JS - Location Context Feature Module
 *
 * PURPOSE:
 * Manages location consent and geolocation submission for the MIRA application.
 * On page load, checks for prior consent in localStorage. First-time users see
 * a consent modal explaining what location access does. On consent, queries
 * browser geolocation and POSTs coordinates to the backend.
 *
 * DEPENDENCIES:
 * - core.js (window.miraAPI for HTTP calls)
 * - api-client.js (MiraAPIClient)
 *
 * LOAD ORDER:
 * After core.js and api-client.js.
 */

(function() {
    const CONSENT_KEY = 'mira-location-consent';

    function init() {
        const consent = localStorage.getItem(CONSENT_KEY);

        if (consent === 'true') {
            // Previously consented — query location directly
            queryLocation();
        } else if (consent === null) {
            // First visit — show consent modal
            showConsentModal();
        }
        // consent === 'false' — user declined, do nothing
    }

    function showConsentModal() {
        const modal = document.getElementById('location-consent-modal');
        const allowBtn = document.getElementById('location-consent-allow');
        const denyBtn = document.getElementById('location-consent-deny');
        const closeBtn = document.getElementById('location-consent-close');

        if (!modal || !allowBtn || !denyBtn) return;

        modal.style.display = 'flex';

        allowBtn.addEventListener('click', () => {
            localStorage.setItem(CONSENT_KEY, 'true');
            modal.style.display = 'none';
            queryLocation(true);
        }, { once: true });

        denyBtn.addEventListener('click', () => {
            localStorage.setItem(CONSENT_KEY, 'false');
            modal.style.display = 'none';
        }, { once: true });

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                localStorage.setItem(CONSENT_KEY, 'false');
                modal.style.display = 'none';
            }, { once: true });
        }
    }

    async function queryLocation(fromUserGesture) {
        if (!navigator.geolocation) return;

        // On automatic page-load calls, check browser permission state
        // first so we only call getCurrentPosition() when already granted.
        // This prevents the native browser prompt from appearing on every
        // load. The one-time browser prompt only fires from the explicit
        // MIRA consent click (fromUserGesture = true).
        if (!fromUserGesture && navigator.permissions) {
            try {
                const status = await navigator.permissions.query({ name: 'geolocation' });
                if (status.state !== 'granted') return;
            } catch (_) {
                // permissions API unsupported — fall through to getCurrentPosition
            }
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                submitLocation(position.coords.latitude, position.coords.longitude);
            },
            () => {},
            { timeout: 10000, maximumAge: 300000 }
        );
    }

    async function submitLocation(latitude, longitude) {
        try {
            // Use window.miraAPI (set by core.js at line 212) — consistent with
            // how inactivity-warning.js and other feature modules access the client.
            // By the time the geolocation callback fires (async), core.js init
            // will have completed, but guard defensively.
            const apiClient = window.miraAPI;
            if (!apiClient) return;

            await apiClient._httpRequest('/v0/api/location', {
                method: 'POST',
                body: JSON.stringify({ latitude, longitude })
            });
        } catch (e) {
            // Location enrichment is optional — don't surface errors
            console.debug('Location submission failed:', e);
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
