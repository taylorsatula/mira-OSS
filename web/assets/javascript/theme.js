/**
 * THEME.JS — Appearance preference (Auto / Light / Dark).
 *
 * Pairs with the inline pre-paint boot script in each page <head>. The boot
 * script does the initial sync resolve (read localStorage, resolve auto via
 * matchMedia, set <html data-theme>) to avoid FOUC. This file owns:
 *
 *   - matchMedia subscription while the user is in Auto, so the page
 *     re-themes when the OS preference flips.
 *   - window.miraSetTheme(mode) — called by the Settings Appearance picker.
 *   - window.miraGetTheme() — returns the stored mode ('auto'|'light'|'dark').
 *
 * The token block in style.css consumes <html data-theme="light"> as the only
 * light-mode trigger. Auto is resolved here, never in CSS, so there is no
 * @media interaction to reason about.
 */
(function() {
	'use strict';

	const STORAGE_KEY = 'mira-theme';
	const VALID = ['auto', 'light', 'dark'];
	const lightQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;

	function readMode() {
		try {
			const v = localStorage.getItem(STORAGE_KEY);
			return VALID.includes(v) ? v : 'auto';
		} catch (e) {
			return 'auto';
		}
	}

	function resolve(mode) {
		if (mode === 'light') return 'light';
		if (mode === 'dark') return 'dark';
		return lightQuery && lightQuery.matches ? 'light' : 'dark';
	}

	function apply(mode) {
		document.documentElement.setAttribute('data-theme', resolve(mode));
	}

	// Re-resolve when the OS preference flips, but only while user is in Auto.
	if (lightQuery) {
		const onSystemChange = () => {
			if (readMode() === 'auto') apply('auto');
		};
		if (lightQuery.addEventListener) {
			lightQuery.addEventListener('change', onSystemChange);
		} else if (lightQuery.addListener) {
			lightQuery.addListener(onSystemChange);
		}
	}

	window.miraSetTheme = function(mode) {
		const next = VALID.includes(mode) ? mode : 'auto';
		try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* storage unavailable */ }
		apply(next);
	};

	window.miraGetTheme = readMode;
})();
