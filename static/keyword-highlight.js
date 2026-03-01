/**
 * Keyword Highlighting for CVtailro Resume Preview
 *
 * Usage:
 *   highlightKeywords(containerElement, keywords, options)
 *   removeHighlights(containerElement)
 */

(function() {
    'use strict';

    /**
     * Highlight keywords in a container element's text content.
     * @param {HTMLElement} container - The element containing resume text
     * @param {Object} keywords - { matched: string[], missing: string[], added: string[] }
     */
    window.highlightKeywords = function(container, keywords) {
        if (!container || !keywords) return;

        // Store original content for toggle
        if (!container.dataset.originalHtml) {
            container.dataset.originalHtml = container.innerHTML;
        }

        var html = container.dataset.originalHtml;

        // Highlight matched keywords (green)
        if (keywords.matched && keywords.matched.length) {
            keywords.matched.forEach(function(kw) {
                var regex = new RegExp('\\b(' + escapeRegex(kw) + ')\\b', 'gi');
                html = html.replace(regex, '<mark class="kw-highlight kw-hl-matched">$1</mark>');
            });
        }

        // Highlight added keywords (blue - added by tailoring)
        if (keywords.added && keywords.added.length) {
            keywords.added.forEach(function(kw) {
                var regex = new RegExp('\\b(' + escapeRegex(kw) + ')\\b', 'gi');
                html = html.replace(regex, '<mark class="kw-highlight kw-hl-added">$1</mark>');
            });
        }

        container.innerHTML = html;
    };

    /**
     * Remove all keyword highlighting from a container.
     * @param {HTMLElement} container
     */
    window.removeHighlights = function(container) {
        if (!container) return;
        if (container.dataset.originalHtml) {
            container.innerHTML = container.dataset.originalHtml;
        }
    };

    /**
     * Toggle keyword highlighting on/off.
     * @param {HTMLElement} container
     * @param {Object} keywords
     * @param {boolean} enabled
     */
    window.toggleKeywordHighlights = function(container, keywords, enabled) {
        if (enabled) {
            highlightKeywords(container, keywords);
        } else {
            removeHighlights(container);
        }
    };

    /**
     * Escape special regex characters in a string.
     */
    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // CSS for highlights (injected once)
    var style = document.createElement('style');
    style.textContent =
        '.kw-highlight { padding: 1px 3px; border-radius: 3px; font-weight: inherit; }' +
        '.kw-hl-matched { background: rgba(0, 184, 148, 0.15); color: inherit; border-bottom: 2px solid rgba(0, 184, 148, 0.4); }' +
        '.kw-hl-added { background: rgba(108, 92, 231, 0.12); color: inherit; border-bottom: 2px solid rgba(108, 92, 231, 0.4); }' +
        '.kw-hl-missing { background: rgba(225, 112, 85, 0.12); color: inherit; border-bottom: 2px solid rgba(225, 112, 85, 0.4); }' +
        '.kw-highlight-legend { display: flex; gap: 16px; font-size: 12px; margin-top: 8px; color: #64748b; }' +
        '.kw-highlight-legend span { display: flex; align-items: center; gap: 4px; }' +
        '.kw-legend-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }' +
        '.kw-legend-dot.matched { background: rgba(0, 184, 148, 0.3); }' +
        '.kw-legend-dot.added { background: rgba(108, 92, 231, 0.3); }';
    document.head.appendChild(style);
})();
