# CVtailro Design System

## Product Context

CVtailro is an AI-powered resume tailoring SaaS application. Users sign in with Google, upload a PDF resume and paste a job description, then receive an AI-optimized resume in 3 PDF templates (Modern / Executive / Minimal) plus DOCX, a match report with keyword analysis, and interview talking points.

**Live URL:** https://cvtailro-production.up.railway.app

**Tech stack:** Python 3.13, Flask (Jinja2 templates), all CSS is inline `<style>` blocks — no external stylesheets or CSS framework.

### Key Pages

| Page | File | Description |
|------|------|-------------|
| Landing / App | `templates/index.html` | Main user-facing page: hero, upload form, progress tracker, results (score ring, downloads, talking points, keyword analysis) |
| Admin Dashboard | `templates/admin.html` | Sidebar layout with KPI cards, charts (Chart.js), user management, system config, analytics tables |
| Contact | `templates/contact.html` | Static contact page |
| Privacy Policy | `templates/privacy.html` | Static legal page |
| Terms of Service | `templates/terms.html` | Static legal page |

### Dark Mode

Implemented via CSS variables and the `html.dark` class. Theme is persisted in `localStorage` key `cvtailro-theme`. On load, a blocking `<script>` in `<head>` adds the `.dark` class before first paint to prevent flash. The system preference (`prefers-color-scheme: dark`) is respected as fallback.

---

## Typography

| Property | Value |
|----------|-------|
| **Primary font** | `'DM Sans'` (Google Fonts, variable, opsz 9–40, weights 400–700, italic 400) |
| **Fallback stack** | `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif` |
| **Base font size** | `16px` |
| **Base line height** | `1.6` |
| **Rendering** | `-webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale` |
| **iOS zoom prevention** | `input, textarea, select { font-size: 16px !important; }` |

### Type Scale

| Element | Size | Weight | Letter-spacing | Line-height | Notes |
|---------|------|--------|----------------|-------------|-------|
| Hero h1 | 44px (30px mobile) | 700 | -1px (-0.5px mobile) | 1.15 | Largest text on the page |
| Hero subtitle | 17px (16px mobile) | 400 | — | 1.7 | `var(--text-secondary)` |
| Hero stat value | 26px (22px mobile) | 700 | -0.5px | 1.2 | `var(--primary)` |
| Hero stat label | 12px | 600 | 0.5px | — | Uppercase |
| Section headings (card h2) | 16px | 700 | — | — | Inside `.card` |
| Downloads heading | 18px | 700 | -0.02em | — | |
| Tab content h2 | 18px | 700 | — | — | |
| Body / paragraph | 16px | 400 | — | 1.6 | |
| Small text / hints | 13px | 500 | — | — | |
| Tiny text / labels | 12px | 600 | 0.5px | — | Uppercase labels |
| Section titles (admin) | 11px | 600 | 0.8px | — | Uppercase, `var(--text-tertiary)` |
| Nav labels (admin sidebar) | 10px | 700 | 1px | — | Uppercase |

### Hero Gradient Text

```css
.hero-highlight {
    background: linear-gradient(135deg, var(--primary), #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
html.dark .hero-highlight {
    background: linear-gradient(135deg, #60a5fa, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
```

---

## Color Tokens

### Light Mode (`:root`)

| Token | Value | Usage |
|-------|-------|-------|
| `--primary` | `#2563eb` | Primary brand blue (blue-600) |
| `--primary-hover` | `#1d4ed8` | Hover state for primary buttons (blue-700) |
| `--primary-bg` | `#eff6ff` | Light blue tinted backgrounds (blue-50) |
| `--accent` | `#2563eb` | Alias of primary |
| `--accent-light` | `rgba(37, 99, 235, 0.08)` | Very subtle blue tint for hover states |
| `--accent-glow` | `rgba(37, 99, 235, 0.12)` | Focus ring glow |
| `--score-high` | `#059669` | Green — high match scores (emerald-600) |
| `--score-mid` | `#d97706` | Amber — medium match scores (amber-600) |
| `--score-low` | `#dc2626` | Red — low match scores (red-600) |
| `--bg` | `#f8fafc` | Page background (slate-50) |
| `--card` | `#ffffff` | Card / surface background |
| `--border` | `#e2e8f0` | Default border color (slate-200) |
| `--border-light` | `#f1f5f9` | Lighter border variant (slate-100) |
| `--text` | `#0f172a` | Primary text (slate-900) |
| `--text-secondary` | `#64748b` | Secondary text (slate-500) |
| `--text-tertiary` | `#94a3b8` | Tertiary / muted text (slate-400) |
| `--error` | `#dc2626` | Error state (red-600) |
| `--radius` | `10px` | Default border radius |
| `--radius-lg` | `14px` | Large border radius |
| `--shadow-sm` | `0 1px 2px rgba(0, 0, 0, 0.05)` | Subtle shadow |
| `--shadow-md` | `0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05)` | Medium shadow |
| `--shadow-lg` | `0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.05)` | Large shadow |

### Dark Mode (`html.dark`)

| Token | Value | Notes |
|-------|-------|-------|
| `--primary-bg` | `rgba(37, 99, 235, 0.15)` | Darker blue tint |
| `--bg` | `#0f172a` | Dark background (slate-900) |
| `--card` | `#1e293b` | Dark card surface (slate-800) |
| `--border` | `#334155` | Dark border (slate-700) |
| `--border-light` | `#1e293b` | Dark lighter border (slate-800) |
| `--text` | `#f8fafc` | Light text (slate-50) |
| `--text-secondary` | `#cbd5e1` | Light secondary text (slate-300) |
| `--text-tertiary` | `#94a3b8` | Unchanged (slate-400) |
| `--shadow-sm` | `0 1px 3px rgba(0, 0, 0, 0.3)` | Heavier shadow |
| `--shadow-md` | `0 4px 6px -1px rgba(0, 0, 0, 0.4)` | Heavier shadow |
| `--shadow-lg` | `0 10px 15px -3px rgba(0, 0, 0, 0.5)` | Heavier shadow |
| `--accent-light` | `rgba(37, 99, 235, 0.2)` | Stronger accent tint |
| `--accent-glow` | `rgba(37, 99, 235, 0.25)` | Stronger focus glow |

### Admin-Only Tokens

| Token | Value (Light) | Value (Dark) | Usage |
|-------|---------------|--------------|-------|
| `--accent-hover` | `#1d4ed8` | (same) | Admin button hover |
| `--accent-light-border` | `rgba(37, 99, 235, 0.2)` | `rgba(37, 99, 235, 0.3)` | Active border highlight |
| `--success` | `#059669` | (same) | Success states |
| `--warning` | `#d97706` | (same) | Warning states |
| `--sidebar-bg` | `#0f172a` | `#020617` | Admin sidebar background |
| `--sidebar-text` | `#94a3b8` | (same) | Sidebar nav text |
| `--sidebar-active` | `rgba(37, 99, 235, 0.15)` | (same) | Active nav item bg |
| `--sidebar-width` | `220px` | (same) | Fixed sidebar width |
| `--header-height` | `56px` | (same) | Fixed header height |

### Semantic Color Usage

```
Status colors:
  Matched keyword pill:  bg rgba(5, 150, 105, 0.1), color var(--score-high), border rgba(5, 150, 105, 0.2)
  Missing keyword pill:  bg rgba(220, 38, 38, 0.06), color var(--score-low), border rgba(220, 38, 38, 0.15)
  Added keyword pill:    bg var(--accent-light), color var(--primary), border var(--border)
  Score improvement:     bg rgba(5, 150, 105, 0.1), color var(--score-high)
  Error banner:          bg rgba(225, 112, 85, 0.06), color #C0392B, border rgba(225, 112, 85, 0.15)
  Alert success (admin): bg rgba(0, 184, 148, 0.08), color var(--success), border rgba(0, 184, 148, 0.2)
  Alert error (admin):   bg rgba(225, 112, 85, 0.08), color var(--error), border rgba(225, 112, 85, 0.2)
  Danger button (admin): bg rgba(225, 112, 85, 0.1), color var(--error), border rgba(225, 112, 85, 0.2)
```

---

## Spacing & Layout

### Main App Page (index.html)

| Element | Value |
|---------|-------|
| Container max-width | `800px` |
| Container padding | `32px 24px 64px` (mobile: `16px 12px 32px`) |
| Header height | `56px`, sticky `top: 0`, `z-index: 100` |
| Header padding | `0 24px` (mobile: `0 16px`) |
| Hero padding | `72px 24px 48px` (mobile: `40px 20px 32px`) |
| Hero inner max-width | `640px` |
| Card padding | `32px` (mobile: `20px`) |
| Card margin-bottom | `24px` (mobile: `16px`) |
| Card border-radius | `16px` (mobile: `14px`) |
| Form columns gap | `24px` (mobile: single column, `16px` gap) |
| Standard gap (controls) | `12px` |
| Section border dividers | `1px solid var(--border)` |

### Admin Page (admin.html)

| Element | Value |
|---------|-------|
| Sidebar width | `220px` (collapses to full-width horizontal on mobile ≤768px) |
| Header height | `56px`, fixed `top: 0` |
| Main content padding | `24px 28px 48px` |
| Main content margin-left | `var(--sidebar-width)` |
| Dashboard grid | `1fr 340px` (single column on ≤1024px) |
| Chart row grid | `2fr 1fr` (single column on ≤900px) |
| KPI row grid | `repeat(auto-fill, minmax(160px, 1fr))`, gap `14px` |
| Metrics grid | `repeat(auto-fill, minmax(140px, 1fr))`, gap `12px` |
| System stats grid | `repeat(3, 1fr)`, gap `14px` |

---

## Component Patterns

### Favicon / Logo

SVG favicon — blue rounded rectangle with white "CV" text:

```html
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect fill='%232563eb' width='100' height='100' rx='12'/><text x='50' y='68' font-size='60' font-weight='bold' text-anchor='middle' fill='%23FFFFFF' font-family='system-ui'>CV</text></svg>">
```

Logo text: "CV" in `var(--primary)`, "tailro" in `var(--text)`. Font-size `22px`, weight `700`, letter-spacing `-0.5px`. No gap between parts.

### Site Header (index.html)

```css
.site-header {
    position: sticky;
    top: 0;
    z-index: 100;
    height: 56px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
}
```

Contains: logo (left), theme toggle + CTA button (right). The CTA button (`header-cta`) uses primary fill style.

### Theme Toggle

A bordered icon button that toggles between sun/moon SVG icons:

```css
.theme-toggle {
    background: none;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 8px;
    color: var(--text-secondary);
    transition: background 0.2s, color 0.2s;
}
.theme-toggle:hover {
    background: var(--accent-light);
    color: var(--primary);
}
```

### Cards

The base card pattern used across the entire application:

```css
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
}
.card h2 {
    font-size: 16px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 20px;
}
```

Admin cards use `border-radius: var(--radius-lg)` (14px) and `padding: 24px`.

No hover effects on base cards. Cards are purely containers.

**Accent variant — What Changed Card:**

```css
.what-changed-card {
    border-left: 3px solid var(--primary);
}
```

### Buttons

#### Primary Button (main CTA)

```css
.btn-run {
    width: 100%;
    max-width: 440px;
    padding: 16px 32px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    font-family: inherit;
    min-height: 52px;
    letter-spacing: -0.01em;
    transition: background 0.2s, box-shadow 0.2s;
}
.btn-run:hover:not(:disabled) {
    background: var(--primary-hover);
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);
}
.btn-run:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    background: var(--text-tertiary);
}
```

#### Header CTA Button

```css
.header-cta {
    padding: 10px 22px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.header-cta:hover { background: var(--primary-hover); }
```

#### Copy Button

```css
.copy-btn {
    padding: 8px 16px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
}
```

#### Try Again Button

```css
.btn-again {
    padding: 14px 32px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
}
.btn-again:hover {
    background: var(--primary-hover);
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);
}
```

#### Admin Buttons

```css
.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
    transition: all 0.2s;
}
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: var(--accent-hover); box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25); }
.btn-secondary { background: var(--bg); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }
.btn-danger { background: rgba(225, 112, 85, 0.1); color: var(--error); border: 1px solid rgba(225, 112, 85, 0.2); }
.btn-danger:hover { background: rgba(225, 112, 85, 0.18); }
.btn-sm { padding: 7px 14px; font-size: 12px; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
```

#### Secondary Download Button

```css
.dl-btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 18px;
    background: var(--primary-bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.dl-btn-secondary:hover {
    border-color: var(--primary);
    transform: translateY(-1px);
}
```

### Upload Zone

Dashed-border drop zone for PDF upload:

```css
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 48px 28px;
    text-align: center;
    cursor: pointer;
    background: var(--bg);
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: var(--primary);
    background: var(--primary-bg);
}
.upload-zone.dragover {
    border-color: var(--primary);
    background: var(--primary-bg);
    border-style: solid;
}
.upload-zone.has-file {
    border-style: solid;
    border-color: var(--primary);
    background: var(--primary-bg);
}
```

**Upload icon container:**

```css
.upload-icon {
    width: 56px;
    height: 56px;
    margin: 0 auto 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--primary-bg);
    border-radius: 14px;
}
.upload-icon svg {
    width: 28px;
    height: 28px;
    color: var(--primary);
    stroke-width: 1.5;
}
```

### Textarea

```css
textarea {
    width: 100%;
    min-height: 200px;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    font-family: inherit;
    font-size: 15px;
    line-height: 1.7;
    resize: vertical;
    color: var(--text);
    background: var(--card);
    transition: all 0.3s ease;
}
textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--accent-glow);
}
textarea.input-error {
    border-color: var(--error);
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
}
```

### Toggle Group (Segmented Control)

Pill-shaped segmented control for mode selection:

```css
.toggle-group {
    display: inline-flex;
    background: var(--primary-bg);
    border-radius: 14px;
    padding: 4px;
    gap: 2px;
}
.toggle-group label {
    padding: 10px 22px;
    font-size: 14px;
    font-weight: 600;
    border-radius: 11px;
    color: var(--text-secondary);
    background: transparent;
    min-height: 42px;
    transition: all 0.25s ease;
}
.toggle-group input:checked + label {
    background: var(--card);
    color: var(--primary);
    box-shadow: var(--shadow-sm);
}
```

### Score Ring

Circular SVG progress ring showing match score:

```
Dimensions: 140px × 140px
SVG circle: stroke-width 8, rotated -90deg
Background ring: stroke var(--primary-bg), fill none
Fill ring: stroke-linecap round, animated stroke-dashoffset (1.5s cubic-bezier)
Center value: font-size 32px, weight 700, positioned absolute center
Label below: font-size 11px, weight 600, uppercase, letter-spacing 1px
Score thresholds: ≥70 var(--score-high), ≥40 var(--score-mid), <40 var(--score-low)
```

### Score Stats Row

```css
.score-stats {
    display: flex;
    justify-content: center;
    gap: 48px;
    margin-top: 24px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    font-size: 14px;
    color: var(--text-secondary);
}
.score-stats strong {
    color: var(--text);
    font-weight: 700;
    font-size: 15px;
}
```

### Score Improvement Banner

```css
.score-improvement-inner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 12px 20px;
    background: var(--primary-bg);
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
```

Shows: label, before score (red) → arrow → after score (green), delta badge (green bg).

### Keyword Pills

```css
.kw {
    font-size: 12px;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.15s;
}
.kw:hover { transform: scale(1.05); }
.kw-matched {
    background: rgba(5, 150, 105, 0.1);
    color: var(--score-high);
    border: 1px solid rgba(5, 150, 105, 0.2);
}
.kw-missing {
    background: rgba(220, 38, 38, 0.06);
    color: var(--score-low);
    border: 1px solid rgba(220, 38, 38, 0.15);
}
```

### Template Cards (Download Section)

3-column grid of selectable template previews:

```css
.template-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 20px 0;
}
.template-card {
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 24px 16px;
    text-align: center;
    background: var(--card);
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
}
.template-card:hover {
    border-color: var(--primary);
    background: var(--primary-bg);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
}
```

Each card has a `.template-preview` (100px height, 12px radius) with unique styling:
- **Modern:** `linear-gradient(180deg, var(--primary-bg) 0%, var(--card) 100%)`, `border-left: 4px solid #1a3a5c`
- **Executive:** `linear-gradient(180deg, var(--bg) 0%, var(--card) 100%)`, `border-bottom: 2px solid #8B7355`
- **Minimal:** `background: var(--bg)` (clean, no accents)

Stacks to single column on mobile (≤768px).

### Progress Section

```css
.progress-wrap {
    max-width: 540px;
    margin: 0 auto;
    text-align: center;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 40px 32px;
    box-shadow: var(--shadow-md);
}
```

**Progress bar:**

```css
.progress-bar-track {
    width: 100%;
    height: 10px;
    background: var(--primary-bg);
    border-radius: 5px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    background: var(--primary);
    border-radius: 5px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
}
```

**Stage list items:**

```css
.stage {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    font-size: 13px;
    color: var(--text-secondary);
    border-radius: 10px;
}
.stage.is-running {
    background: var(--primary-bg);
}
.stage.is-running .stage-name {
    color: var(--primary);
    font-weight: 700;
}
```

### Tabs

```css
.tabs {
    display: flex;
    border-bottom: 1.5px solid var(--border);
    gap: 4px;
}
.tab {
    padding: 12px 22px;
    font-size: 14px;
    font-weight: 600;
    border-bottom: 2px solid transparent;
    margin-bottom: -1.5px;
    color: var(--text-secondary);
    background: none;
}
.tab.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
}
.tab-content {
    max-height: 500px;
    overflow-y: auto;
    padding: 24px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 0 0 var(--radius) var(--radius);
    font-size: 14px;
    line-height: 1.8;
}
```

### Hero Stats Bar

Connected stat boxes with shared borders:

```css
.hero-stat {
    flex: 1;
    min-width: 120px;
    text-align: center;
    padding: 22px 24px;
    background: var(--card);
    border: 1px solid var(--border);
    border-right: none;
    box-shadow: var(--shadow-sm);
}
.hero-stat:first-child { border-radius: var(--radius) 0 0 var(--radius); }
.hero-stat:last-child { border-radius: 0 var(--radius) var(--radius) 0; border-right: 1px solid var(--border); }
```

Stacks vertically on mobile (≤480px).

### Overlays / Modals

Full-screen overlay with centered panel:

```css
.history-overlay {
    position: fixed;
    inset: 0;
    z-index: 300;
    background: rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
}
html.dark .history-overlay {
    background: rgba(0, 0, 0, 0.6);
}
.history-overlay.active {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 80px 24px 24px;
}
.history-panel {
    background: var(--card);
    border-radius: 12px;
    width: 100%;
    max-width: 640px;
    max-height: calc(100vh - 120px);
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    border: 1px solid var(--border);
    overflow: hidden;
}
```

Panel has: header (20px 24px padding, bottom border), scrollable body (16px 24px padding).

Close button: 36px × 36px, 8px radius, icon button.

### Trust Bar

```css
.trust-bar {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    text-align: center;
    padding: 16px 24px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    letter-spacing: 0.02em;
}
```

### Footer

```css
.site-footer {
    background: var(--card);
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    text-align: center;
    padding: 32px 24px;
    font-size: 13px;
}
```

Contains: brand name (14px, 600 weight), link row (flex, centered, gap 20px), copyright (12px, tertiary color).

---

## Admin-Specific Components

### Admin Header

```css
.top-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    background: var(--sidebar-bg);  /* #0f172a */
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 100;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
```

Logo: 20px, weight 700, white. Badge: `var(--accent)` bg, 10px font, 700 weight, pill-shaped (20px radius), uppercase.

### Admin Sidebar

```css
.sidebar {
    position: fixed;
    top: 56px;
    left: 0;
    bottom: 0;
    width: 220px;
    background: var(--sidebar-bg);  /* #0f172a, dark: #020617 */
    display: flex;
    flex-direction: column;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 20px;
    color: var(--sidebar-text);
    font-size: 14px;
    font-weight: 500;
    border-left: 3px solid transparent;
}
.nav-item:hover {
    color: #e2e8f0;
    background: rgba(255, 255, 255, 0.04);
}
.nav-item.active {
    color: #fff;
    background: var(--sidebar-active);
    border-left-color: var(--accent);
}
```

Footer nav item (Logout): red (#ef4444), hover bg rgba(239, 68, 68, 0.1).

On mobile (≤768px), sidebar becomes a horizontal scrollable row below the header.

### KPI Cards (Admin Dashboard)

```css
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 20px 16px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: transparent;
    transition: background 0.2s;
}
.kpi-card:hover::before {
    background: var(--accent);
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    font-variant-numeric: tabular-nums;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    margin-top: 6px;
}
```

Value color variants: `.accent` → `var(--accent)`, `.success` → `var(--success)`, `.warning` → `var(--warning)`.

### System Stats Cards (Admin)

Similar to KPI but with centered layout and accent bar on hover:

```css
.sys-stat {
    text-align: center;
    padding: 20px 16px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.sys-stat::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
    opacity: 0;
    transition: opacity 0.2s;
}
.sys-stat:hover::before { opacity: 1; }
.sys-val {
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
}
```

### Metric Icon Badges (Admin)

36px × 36px rounded squares (8px radius) with tinted backgrounds:

```
Purple/Blue: rgba(37, 99, 235, 0.1)
Green:       rgba(5, 150, 105, 0.1)
Orange/Red:  rgba(220, 38, 38, 0.08)
Blue:        rgba(59, 130, 246, 0.1)
Yellow:      rgba(217, 119, 6, 0.1)
```

### Analytics Table (Admin)

```css
.analytics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.analytics-table th {
    text-align: left;
    padding: 10px 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    border-bottom: 2px solid var(--border);
    background: var(--bg);
}
.analytics-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
}
.analytics-table tr:hover td {
    background: var(--accent-light);
}
```

### Activity Items (Admin)

```css
.activity-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 8px;
}
.activity-item:hover { background: var(--accent-light); }
.activity-avatar {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: var(--accent-light);
    color: var(--accent);
    font-weight: 600;
    font-size: 14px;
}
```

### User Cards (Admin)

```css
.user-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--card);
    cursor: pointer;
}
.user-card:hover {
    background: var(--accent-light);
    border-color: var(--accent-light-border);
}
```

### Admin Forms

```css
.form-group label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.form-group input, .form-group select {
    width: 100%;
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 13px;
    background: var(--card);
    color: var(--text);
}
.form-group input:focus, .form-group select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-light);
}
```

### Chart Cards (Admin)

```css
.chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
}
.chart-container {
    position: relative;
    height: 220px;
    width: 100%;
}
```

### Live Indicator

```css
.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
}
```

---

## Animations & Transitions

| Name | Duration | Easing | Usage |
|------|----------|--------|-------|
| `fadeInUp` | 0.4–0.5s | `ease-out` | Results cards appearing |
| `slideInUp` | 0.5s | `ease-out` | Staggered card entrance (delays: 0, 0.1, 0.2, 0.3, 0.35s) |
| `pulse` | 2s | infinite | Live indicator dot, skeleton loading |
| Progress bar fill | 1.2s | `cubic-bezier(0.4, 0, 0.2, 1)` | Progress bar width transition |
| Score ring | 1.5s | `cubic-bezier(0.4, 0, 0.2, 1)` | SVG stroke-dashoffset animation |
| General transitions | 0.2–0.3s | `ease` | Hover states, color changes |

```css
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slideInUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
```

Respects `prefers-reduced-motion: reduce` — all animations/transitions reduced to `0.01ms`.

---

## Responsive Breakpoints

| Breakpoint | Target | Key changes |
|------------|--------|-------------|
| ≤480px | Small phones | Hero stats stack vertically |
| ≤768px | Tablets / phones | Form columns stack, cards tighter padding, template grid single column, admin sidebar becomes horizontal scroll bar |
| ≤900px | Small laptops | Admin chart row stacks to single column |
| ≤1024px | Tablets landscape | Admin dashboard grid stacks to single column |

### Mobile-Specific Patterns

- Minimum touch target: `44px` (toggle buttons), `52px` (CTA button)
- Safe area insets respected via `env(safe-area-inset-*)` on body and overlays
- Progress bar becomes sticky on mobile (below 56px header)
- iOS zoom prevention: all inputs forced to `16px` font size

---

## Icon System

All icons are inline SVGs (no icon library). Common patterns:

- Stroke-based icons, `stroke-width: 1.5` or `2`
- Standard sizes: `16px`, `20px`, `24px`, `28px`
- Color inherits from parent via `currentColor`
- Upload icon: 28px, inside 56px × 56px colored container (14px radius)
- Stage checks: 18px × 18px flex container
- Nav icons (admin): 16px, fixed 24px width column

---

## Z-Index Scale

| Layer | Z-Index | Element |
|-------|---------|---------|
| Base content | auto | Cards, sections |
| Sticky progress | 50 | Mobile progress bar |
| Site header | 100 | Sticky header, admin fixed header |
| Sidebar | 90 | Admin sidebar |
| Overlays | 300 | History/profile panels |

---

## Design Principles

1. **Clean & minimal** — White/dark cards on subtle tinted backgrounds, no visual clutter
2. **Consistent borders** — All containers have `1px solid var(--border)`, never bare edges
3. **Blue as the only accent** — `#2563eb` is used for all interactive elements and highlights
4. **Generous whitespace** — 32px card padding, 24px gaps, 48–72px hero padding
5. **Subtle hover states** — Light tint changes, no dramatic transforms (exception: keyword pills scale 1.05)
6. **Font weight hierarchy** — 700 for headings/values, 600 for labels/buttons, 500 for nav, 400 for body
7. **Uppercase micro-labels** — 10–12px, 600–700 weight, letter-spacing 0.5–1px for section dividers and stat labels
8. **No gradients** (except hero text highlight and template previews) — flat colors throughout
9. **Accessible focus states** — 3px outline rings using `box-shadow: 0 0 0 3px var(--accent-glow)`
10. **Dark mode parity** — Every light-mode component has a dark equivalent; shadows get heavier, tints get more opaque
