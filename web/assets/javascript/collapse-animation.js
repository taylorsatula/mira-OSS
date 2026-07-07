/**
 * Collapse Animation
 *
 * Converts text to alternate font character-by-character, then drops with gravity.
 * Prevents reflow by measuring and locking character widths before animation.
 *
 * Required CSS classes:
 *   .converted { font-family: 'YourFont', monospace; }
 *   .falling { pointer-events: none; }
 *
 * Usage:
 *   collapseAnimation(element, {
 *     conversionDurationMs: 5000,
 *     fallDurationMs: 5000,
 *     maxMsPerChar: 30,
 *     gravity: 980 * 1.9,
 *     onConversionComplete: () => {},
 *     onFallComplete: () => {}
 *   });
 */

const COLLAPSE_DEFAULTS = {
    conversionDurationMs: 5000,  // Total time to convert all chars
    fallDurationMs: 5000,        // Total time to stagger all falls
    maxMsPerChar: 30,            // Cap on per-character timing
    gravity: 980 * 1.9,          // Gravity acceleration (px/s²)
    convertOnFall: false,        // Swap font when each char starts falling (skip conversion phase)
};

/**
 * Fisher-Yates shuffle
 */
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

/**
 * Main entry point - converts text then drops with gravity
 *
 * @param {HTMLElement} element - Element containing text to animate
 * @param {Object} options - Configuration options
 * @param {number} options.conversionDurationMs - Total time for font conversion phase
 * @param {number} options.fallDurationMs - Total time for gravity fall phase
 * @param {number} options.maxMsPerChar - Maximum ms per character (caps speed for short text)
 * @param {number} options.gravity - Gravity acceleration in px/s²
 * @param {HTMLElement} options.scrollContainer - Element to hide overflow on during fall (prevents scrollbar)
 * @param {Function} options.onConversionComplete - Callback when font conversion finishes
 * @param {Function} options.onFallComplete - Callback when all chars have fallen off screen
 */
function collapseAnimation(element, options = {}) {
    const config = {
        conversionDurationMs: options.conversionDurationMs ?? COLLAPSE_DEFAULTS.conversionDurationMs,
        fallDurationMs: options.fallDurationMs ?? COLLAPSE_DEFAULTS.fallDurationMs,
        maxMsPerChar: options.maxMsPerChar ?? COLLAPSE_DEFAULTS.maxMsPerChar,
        gravity: options.gravity ?? COLLAPSE_DEFAULTS.gravity,
        convertOnFall: options.convertOnFall ?? COLLAPSE_DEFAULTS.convertOnFall,
        scrollContainer: options.scrollContainer ?? null,  // Element to hide overflow on during fall
        onConversionComplete: options.onConversionComplete ?? (() => {}),
        onFallComplete: options.onFallComplete ?? (() => {})
    };

    // Step 1: Wrap non-space characters in spans while preserving DOM structure
    // Walk the tree and replace text nodes with character-wrapped versions
    const spans = [];

    function wrapTextNode(textNode) {
        const text = textNode.textContent;
        const fragment = document.createDocumentFragment();

        for (const char of text) {
            if (char === ' ' || char === '\n' || char === '\t') {
                fragment.appendChild(document.createTextNode(char));
            } else {
                const span = document.createElement('span');
                span.textContent = char;
                fragment.appendChild(span);
                spans.push(span);
            }
        }

        textNode.parentNode.replaceChild(fragment, textNode);
    }

    // Recursively walk all text nodes in the tree
    function walkAndWrap(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            wrapTextNode(node);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            // Process children (use array copy since we're modifying the tree)
            Array.from(node.childNodes).forEach(walkAndWrap);
        }
    }

    walkAndWrap(element);

    // Step 2: Measure widths (forces layout)
    const widths = spans.map(s => s.getBoundingClientRect().width);

    // Step 3: Lock widths to prevent reflow during font change
    spans.forEach((span, i) => {
        span.style.display = 'inline-block';
        span.style.width = widths[i] + 'px';
        span.style.textAlign = 'center';
        span.style.position = 'relative';
        span.style.verticalAlign = 'text-top';
    });

    // Step 4: Animate font conversion (or skip to fall-first mode)
    if (config.convertOnFall) {
        // Fall-first mode: skip conversion phase, font swaps per-character during fall
        config.onConversionComplete();
        startGravityFall(spans, config);
    } else {
        // Original: convert all characters first, then fall.
        // Time-based: each tick batch-converts all chars due by elapsed time,
        // so hidden-tab throttling (1Hz) doesn't slow the conversion phase.
        const msPerChar = Math.min(config.maxMsPerChar, config.conversionDurationMs / spans.length);
        let index = 0;
        const conversionStart = performance.now();

        const interval = setInterval(() => {
            const target = Math.min(
                Math.floor((performance.now() - conversionStart) / msPerChar),
                spans.length
            );
            while (index < target) {
                spans[index].classList.add('converted');
                index++;
            }
            if (index >= spans.length) {
                clearInterval(interval);
                config.onConversionComplete();
                startGravityFall(spans, config);
            }
        }, msPerChar);
    }
}

/**
 * Starts the gravity fall phase
 */
function startGravityFall(spans, config) {
    // Hide scrollbar during fall to prevent scrollbar flash as chars drop
    const scrollContainer = config.scrollContainer;
    let originalOverflow = null;
    if (scrollContainer) {
        originalOverflow = scrollContainer.style.overflow;
        scrollContainer.style.overflow = 'hidden';
    }

    const staggerMs = Math.min(config.maxMsPerChar, config.fallDurationMs / spans.length);
    const shuffled = shuffle([...spans]);

    const items = shuffled.map((span, i) => ({
        el: span,
        delay: i * staggerMs,
        y: 0,
        vy: 0,
        // Random gravity multiplier: 0.85 to 1.15 (±15%)
        gravity: config.gravity * (0.85 + Math.random() * 0.3),
        active: false
    }));

    let startTime = 0;
    let lastTime = 0;

    function tick(now) {
        if (!startTime) startTime = now;
        if (!lastTime) lastTime = now;

        const elapsed = now - startTime;
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;

        for (let i = items.length - 1; i >= 0; i--) {
            const item = items[i];
            if (elapsed < item.delay) continue;

            if (!item.active) {
                item.active = true;
                item.el.classList.add('falling');
                if (config.convertOnFall) {
                    item.el.classList.add('converted');
                }
            }

            item.vy += item.gravity * dt;
            item.y += item.vy * dt;
            item.el.style.transform = `translateY(${item.y}px)`;

            if (item.y > window.innerHeight) {
                item.el.style.visibility = 'hidden';
                items.splice(i, 1);
            }
        }

        if (items.length > 0) {
            requestAnimationFrame(tick);
        } else {
            // Restore overflow before callback
            if (scrollContainer && originalOverflow !== null) {
                scrollContainer.style.overflow = originalOverflow;
            }
            config.onFallComplete();
        }
    }

    requestAnimationFrame(tick);
}

// Export for browser global usage
if (typeof window !== 'undefined') {
    window.collapseAnimation = collapseAnimation;
    window.COLLAPSE_DEFAULTS = COLLAPSE_DEFAULTS;
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { collapseAnimation, COLLAPSE_DEFAULTS };
}
