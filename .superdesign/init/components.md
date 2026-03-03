# CVtailro — Reusable CSS Components

All styles are inline `<style>` blocks within `templates/index.html` and `templates/admin.html`. No external CSS framework. No Tailwind. No React.

---

## 1. Cards

### `.card` (index.html — Public)

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

### `.card` (admin.html — Dashboard)

```css
.card {
    background: var(--card);
    border-radius: var(--radius-lg);    /* 14px */
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid var(--border);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

.card-header h2 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
}
```

---

## 2. Buttons

### `.btn` Family (admin.html)

```css
.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
}

.btn-primary {
    background: var(--accent);
    color: white;
}
.btn-primary:hover {
    background: var(--accent-hover);
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}

.btn-secondary {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
}
.btn-secondary:hover { background: var(--border); }

.btn-danger {
    background: rgba(225,112,85,0.1);
    color: var(--error);
    border: 1px solid rgba(225,112,85,0.2);
}
.btn-danger:hover { background: rgba(225,112,85,0.18); }

.btn-sm { padding: 7px 14px; font-size: 12px; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
```

### `.btn-run` (index.html — Main CTA)

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
    cursor: pointer;
    transition: background 0.2s, box-shadow 0.2s;
    min-height: 52px;
    letter-spacing: -0.01em;
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

### `.header-cta` (index.html — Header button)

```css
.header-cta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 22px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.3s ease;
}
```

### `.btn-google-signin` (index.html)

```css
.btn-google-signin {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    font-family: inherit;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.2s, box-shadow 0.2s;
    min-height: 40px;
    white-space: nowrap;
}
```

### `.copy-btn` (index.html)

```css
.copy-btn {
    padding: 8px 16px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    font-family: inherit;
}
```

---

## 3. Upload Zone (index.html)

```css
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 48px 28px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    background: var(--bg);
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

.upload-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
}

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
```

---

## 4. Score Ring (index.html)

```css
.score-ring-wrap {
    display: inline-block;
    position: relative;
    width: 140px;
    height: 140px;
    margin-bottom: 16px;
}

.score-ring-wrap svg {
    width: 140px;
    height: 140px;
    transform: rotate(-90deg);
}

.score-ring-bg {
    fill: none;
    stroke: var(--primary-bg);
    stroke-width: 8;
}

.score-ring-fill {
    fill: none;
    stroke-width: 8;
    stroke-linecap: round;
    transition: stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.3s;
}

.score-ring-value {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 32px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
}

.score-ring-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
}
```

---

## 5. Template Cards (index.html — Download section)

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
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
    background: var(--card);
}

.template-card:hover {
    border-color: var(--primary);
    background: var(--primary-bg);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
}

.template-name {
    font-weight: 700;
    font-size: 15px;
    margin-top: 14px;
    transition: color 0.2s;
    letter-spacing: -0.01em;
}

.template-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 6px;
}

.template-preview {
    height: 100px;
    border-radius: 12px;
    border: 1px solid var(--border);
}

/* Preview style variants */
.modern-preview {
    background: linear-gradient(180deg, var(--primary-bg) 0%, var(--card) 100%);
    border-left: 4px solid #1a3a5c;
}
.exec-preview {
    background: linear-gradient(180deg, var(--bg) 0%, var(--card) 100%);
    border-bottom: 2px solid #8B7355;
}
.min-preview {
    background: var(--bg);
}
```

---

## 6. Progress Bar & Stages (index.html)

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

.progress-bar-track {
    width: 100%;
    height: 10px;
    background: var(--primary-bg);
    border-radius: 5px;
    overflow: hidden;
    margin-bottom: 12px;
}

.progress-bar-fill {
    width: 0%;
    height: 100%;
    background: var(--primary);
    border-radius: 5px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.stage {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    font-size: 13px;
    color: var(--text-secondary);
    border-radius: 10px;
    transition: background 0.2s;
}

.stage.is-running {
    background: var(--primary-bg);
}
.stage.is-running .stage-name {
    color: var(--primary);
    font-weight: 700;
}
```

---

## 7. Toggle Group (index.html — Mode selector)

```css
.toggle-group {
    display: inline-flex;
    background: var(--primary-bg);
    border-radius: 14px;
    padding: 4px;
    gap: 2px;
    border: none;
}

.toggle-group label {
    padding: 10px 22px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
    user-select: none;
    white-space: nowrap;
    min-height: 42px;
    display: flex;
    align-items: center;
    color: var(--text-secondary);
    background: transparent;
    border-radius: 11px;
    border: none;
}

.toggle-group input { display: none; }

.toggle-group input:checked + label {
    background: var(--card);
    color: var(--primary);
    box-shadow: var(--shadow-sm);
}
```

---

## 8. Keyword Tags (index.html)

```css
.kw {
    font-size: 12px;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: transform 0.15s;
}

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

---

## 9. Tabs (index.html — Preview)

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
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1.5px;
    color: var(--text-secondary);
    transition: all 0.2s ease;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
    font-family: inherit;
}

.tab.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
}
```

---

## 10. KPI Cards (admin.html)

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
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: transparent;
    transition: background 0.2s;
}

.kpi-card:hover::before {
    background: var(--accent);
}

.kpi-card .kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.2;
    font-variant-numeric: tabular-nums;
}

.kpi-card .kpi-value.accent { color: var(--accent); }
.kpi-card .kpi-value.success { color: var(--success); }
.kpi-card .kpi-value.warning { color: var(--warning); }

.kpi-card .kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-card .kpi-sublabel {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 2px;
}
```

---

## 11. User Menu / Dropdown (index.html)

```css
.user-menu-trigger {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px 4px 4px;
    background: none;
    border: 1px solid var(--border);
    border-radius: 24px;
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    transition: background 0.2s, border-color 0.2s;
    min-height: 40px;
}

.user-dropdown {
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    min-width: 200px;
    background: var(--card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    z-index: 200;
    overflow: hidden;
}

.user-menu-wrap.open .user-dropdown { display: block; }

.dropdown-item {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 10px 16px;
    background: none;
    border: none;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    cursor: pointer;
    transition: background 0.15s;
    min-height: 44px;
}

.dropdown-item:hover { background: var(--primary-bg); }
```

---

## 12. Toast Notifications (index.html)

```css
.toast {
    position: fixed;
    top: 80px;
    right: 24px;
    z-index: 1000;
    max-width: 400px;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    transform: translateX(calc(100% + 24px));
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    pointer-events: auto;
}

.toast.visible { transform: translateX(0); }
.toast.toast-error { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
.toast.toast-info { background: #DBEAFE; color: #1E40AF; border: 1px solid #BFDBFE; }
.toast.toast-success { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
```

---

## 13. History Card (index.html overlay)

```css
.history-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.15s;
    min-height: 44px;
    background: var(--bg);
}

.history-card:hover {
    border-color: var(--primary);
    box-shadow: 0 2px 8px var(--accent-glow);
    background: var(--primary-bg);
}

.history-card-score {
    width: 44px;
    height: 44px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    font-weight: 700;
    flex-shrink: 0;
}
.history-card-score.high { background: rgba(5,150,105,0.15); color: var(--score-high); }
.history-card-score.mid { background: rgba(217,119,6,0.15); color: var(--score-mid); }
.history-card-score.low { background: rgba(220,38,38,0.15); color: var(--score-low); }

.history-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    line-height: 1.4;
}
.history-badge.template { background: var(--accent-light); color: var(--primary); }
.history-badge.mode-conservative { background: rgba(59,130,246,0.15); color: #2563eb; }
.history-badge.mode-aggressive { background: rgba(220,38,38,0.15); color: var(--score-low); }
```

---

## 14. Alerts (admin.html)

```css
.alert {
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 13px;
    margin-bottom: 16px;
    display: none;
}
.alert.success {
    background: rgba(0,184,148,0.08);
    color: var(--success);
    border: 1px solid rgba(0,184,148,0.2);
    display: block;
}
.alert.error {
    background: rgba(225,112,85,0.08);
    color: var(--error);
    border: 1px solid rgba(225,112,85,0.2);
    display: block;
}
```

---

## 15. Forms (admin.html)

```css
.form-group { margin-bottom: 18px; }
.form-group label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    display: block;
    margin-bottom: 6px;
}
.form-group input[type="text"],
.form-group input[type="password"],
.form-group input[type="number"],
.form-group select {
    width: 100%;
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    font-family: inherit;
    background: var(--card);
    color: var(--text);
}
.form-group input:focus, .form-group select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-light);
}
```

---

## 16. Analytics Table (admin.html)

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
.analytics-table tr:hover td { background: var(--accent-light); }
```

---

## 17. Animations

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

/* Used on results cards */
.results-section.active .card {
    animation: slideInUp 0.5s ease-out both;
}
.results-section.active .card:nth-child(2) { animation-delay: 0.1s; }
.results-section.active .card:nth-child(3) { animation-delay: 0.2s; }
/* etc. */
```
