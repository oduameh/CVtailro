# CVtailro — Theme Variables

## Overview

CVtailro uses CSS custom properties (variables) defined on `:root` and `html.dark` for theming. Dark mode is toggled via a class on `<html>` and persisted in `localStorage` (`cvtailro-theme`). The two main templates (`index.html` and `admin.html`) each define their own variable sets with significant overlap.

Font: `'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`

---

## index.html — Public App

### `:root` (Light Mode)

```css
:root {
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --primary-bg: #eff6ff;
    --accent: #2563eb;
    --accent-light: rgba(37, 99, 235, 0.08);
    --accent-glow: rgba(37, 99, 235, 0.12);
    --score-high: #059669;
    --score-mid: #d97706;
    --score-low: #dc2626;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    --text: #0f172a;
    --text-secondary: #64748b;
    --text-tertiary: #94a3b8;
    --error: #dc2626;
    --radius: 10px;
    --radius-lg: 14px;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
}
```

### `html.dark` (Dark Mode)

```css
html.dark {
    --primary-bg: rgba(37, 99, 235, 0.15);
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --border-light: #1e293b;
    --text: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-tertiary: #94a3b8;
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    --accent-light: rgba(37, 99, 235, 0.2);
    --accent-glow: rgba(37, 99, 235, 0.25);
}
```

**Note:** `--primary`, `--primary-hover`, `--accent`, `--score-*`, `--error`, `--radius`, `--radius-lg` are NOT overridden in dark mode — they stay the same.

---

## admin.html — Admin Dashboard

### `:root` (Light Mode)

```css
:root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --text-secondary: #64748b;
    --text-tertiary: #94a3b8;
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
    --accent-light: rgba(37,99,235,0.08);
    --accent-light-border: rgba(37,99,235,0.2);
    --success: #059669;
    --error: #dc2626;
    --warning: #d97706;
    --sidebar-bg: #0f172a;
    --sidebar-text: #94a3b8;
    --sidebar-active: rgba(37,99,235,0.15);
    --sidebar-width: 220px;
    --header-height: 56px;
    --radius: 10px;
    --radius-lg: 14px;
}
```

### `html.dark` (Dark Mode)

```css
html.dark {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --text-secondary: #94a3b8;
    --text-tertiary: #64748b;
    --accent-light: rgba(37,99,235,0.2);
    --accent-light-border: rgba(37,99,235,0.3);
    --sidebar-bg: #020617;
}
```

---

## Color Palette Summary

| Token             | Light          | Dark           | Usage                              |
|-------------------|----------------|----------------|------------------------------------|
| `--primary`       | `#2563eb`      | (same)         | Brand blue, CTAs, links            |
| `--primary-hover` | `#1d4ed8`      | (same)         | Button hover states                |
| `--primary-bg`    | `#eff6ff`      | `rgba(37,99,235,0.15)` | Subtle blue backgrounds |
| `--bg`            | `#f8fafc`      | `#0f172a`      | Page background                    |
| `--card`          | `#ffffff`      | `#1e293b`      | Card/panel background              |
| `--border`        | `#e2e8f0`      | `#334155`      | All borders                        |
| `--text`          | `#0f172a`      | `#f8fafc`      | Primary text                       |
| `--text-secondary`| `#64748b`      | `#cbd5e1` (index) / `#94a3b8` (admin) | Labels, hints |
| `--text-tertiary` | `#94a3b8`      | `#94a3b8` (index) / `#64748b` (admin) | Subtle text    |
| `--error`         | `#dc2626`      | (same)         | Error states                       |
| `--score-high`    | `#059669`      | (same)         | High scores, success               |
| `--score-mid`     | `#d97706`      | (same)         | Medium scores, warnings            |
| `--score-low`     | `#dc2626`      | (same)         | Low scores, errors                 |
| `--sidebar-bg`    | `#0f172a`      | `#020617`      | Admin sidebar (always dark)        |

---

## Sizing Tokens

| Token              | Value    | Used In       |
|--------------------|----------|---------------|
| `--radius`         | `10px`   | Buttons, inputs, small cards |
| `--radius-lg`      | `14px`   | Cards, panels |
| `--sidebar-width`  | `220px`  | Admin sidebar |
| `--header-height`  | `56px`   | Admin top bar, also site-header height in index |

---

## Dark Mode Implementation

Both pages use the same pattern:

```html
<script>
(function() {
    var saved = localStorage.getItem('cvtailro-theme');
    var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (saved === 'dark' || (!saved && prefersDark)) {
        document.documentElement.classList.add('dark');
    }
})();
</script>
```

Toggle is in `index.html` header via `.theme-toggle` button. Admin inherits the same `localStorage` key.
