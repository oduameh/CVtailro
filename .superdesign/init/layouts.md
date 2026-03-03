# CVtailro — Layout Structures

---

## 1. index.html — Public App Layout

### Overall Structure

```
<body>
  <a.skip-link>                 (accessibility)
  <noscript>                    (fallback banner)
  <header.site-header>          (sticky, 56px)
  <section.hero>                (centered hero)
  <div.trust-bar>               (hidden by default)
  <div>                         (demo/free banner)
  <div.container#main-content>
    <div.error-banner>          (hidden, shown on error)
    <div#configBanner>          (hidden, service unavailable)
    <div#formSection>
      <div.form-columns>        (2-col grid)
        <div.card#uploadCard>   (left: upload zone)
        <div.card>              (right: job description textarea)
      </div>
      <div.controls-row>        (mode toggle + model select)
      <div.cta-row>             (tailoring button)
    </div>
    <div.progress-section>      (hidden, shown during pipeline)
    <div.results-section>       (hidden, shown on completion)
  </div>
  <footer.site-footer>
  <div.history-overlay>         (full-screen overlay, hidden)
  <div.auth-modal-overlay>      (auth modal, hidden)
  <div.reset-pw-overlay>        (password reset overlay, hidden)
  <div.toast#toastEl>           (fixed-position toast)
</body>
```

### Sticky Header

```html
<header class="site-header" role="banner">
    <a href="/" class="site-logo">
        <span class="cv">CV</span><span class="tailro">tailro</span>
    </a>
    <div style="display:flex;align-items:center;gap:12px;">
        <button class="theme-toggle" onclick="toggleDarkMode()">
            <!-- moon/sun SVG icons -->
        </button>
        <div class="header-auth" id="headerAuth">
            <!-- Signed out: btn-google-signin -->
            <!-- Signed in: user-menu-wrap with dropdown -->
        </div>
    </div>
</header>
```

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

### Hero Section

```html
<section class="hero">
    <div class="hero-inner">
        <h1>AI Resume Tailoring.<br><span class="hero-highlight">Getting you hired.</span></h1>
        <p class="hero-sub">Upload your resume, paste a job description...</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">3</div>
                <div class="hero-stat-label">PDF Templates</div>
            </div>
            <!-- 2 more hero-stat items -->
        </div>
    </div>
</section>
```

```css
.hero {
    background: var(--bg);
    color: var(--text);
    text-align: center;
    padding: 72px 24px 48px;
}

.hero-inner {
    max-width: 640px;
    margin: 0 auto;
}

.hero h1 {
    font-size: 44px;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1.15;
    margin-bottom: 20px;
}

.hero-stats {
    display: flex;
    align-items: stretch;
    justify-content: center;
    gap: 0;
    flex-wrap: wrap;
    max-width: 520px;
    margin: 0 auto;
}
```

### Form Columns (2-column grid)

```html
<div class="form-columns">
    <div class="card" id="uploadCard">
        <h2>Upload Your Resume</h2>
        <div class="upload-zone" id="uploadZone">
            <input type="file" id="resumeFile" accept=".pdf,.md,.txt">
            <!-- upload icon, label, privacy note -->
        </div>
    </div>
    <div class="card">
        <h2>Job Description</h2>
        <textarea id="jobDescription" placeholder="Paste the job description here..."></textarea>
        <!-- char count, inline error -->
    </div>
</div>
```

```css
.form-columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 0;
}

/* Mobile: stack */
@media (max-width: 768px) {
    .form-columns {
        grid-template-columns: 1fr;
        gap: 16px;
    }
}
```

### Container

```css
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 32px 24px 64px;
}

@media (max-width: 768px) {
    .container {
        padding: 16px 12px 32px;
    }
}
```

### Footer

```html
<footer class="site-footer">
    <div class="footer-brand">CVtailro</div>
    <div class="footer-links">
        <a href="/privacy">Privacy Policy</a>
        <a href="/terms">Terms of Service</a>
        <a href="/contact">Contact</a>
    </div>
    <div class="footer-copy">&copy; 2025 CVtailro. All rights reserved.</div>
</footer>
```

```css
.site-footer {
    background: var(--card);
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    text-align: center;
    padding: 32px 24px;
    font-size: 13px;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 12px;
}
```

### History Overlay

```html
<div class="history-overlay" id="historyOverlay">
    <div class="history-panel">
        <div class="history-header">
            <h2>Job History</h2>
            <button class="history-close" onclick="closeHistory()">&times;</button>
        </div>
        <div class="history-body" id="historyBody">
            <!-- .history-card items or .history-empty or .history-loading -->
        </div>
    </div>
</div>
```

```css
.history-overlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 300;
    background: rgba(0,0,0,0.4);
    backdrop-filter: blur(4px);
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
    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    border: 1px solid var(--border);
    overflow: hidden;
}
```

### Auth Modal Overlay

```css
.auth-modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 400;
    background: rgba(0,0,0,0.45);
    backdrop-filter: blur(4px);
    justify-content: center;
    align-items: center;
}

.auth-modal {
    background: var(--card);
    border-radius: 14px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.18);
    border: 1px solid var(--border);
    overflow: hidden;
    animation: authModalIn 0.2s ease-out;
    margin: 16px;
}
```

---

## 2. admin.html — Admin Dashboard Layout

### Overall Structure

```
<body>
  <div#authSection.auth-wrapper>         (login/set-password, full-viewport centered)
    <div.auth-card#setPasswordCard>
    <div.auth-card#loginCard>
  </div>

  <div#configSection.app-layout>         (hidden initially, display:none)
    <div.top-header>                     (fixed header bar)
    <nav.sidebar#sidebarNav>             (fixed sidebar)
      <div.sidebar-nav>                  (nav items)
      <div.sidebar-footer>              (logout)
    </nav>
    <div.main-content>                   (offset by sidebar + header)
      <div.tab-content#tab-dashboard>
      <div.tab-content#tab-config>
      <div.tab-content#tab-users>
      <div.tab-content#tab-diagnostics>
      <div.tab-content#tab-analytics>
      <div.tab-content#tab-errors>
      <div.tab-content#tab-system>
    </div>
  </div>
</body>
```

### Auth Wrapper (Login screen)

```css
.auth-wrapper {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg);
    padding: 24px;
}

.auth-card {
    background: var(--card);
    border-radius: var(--radius);
    padding: 40px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
```

### Fixed Top Header

```html
<div class="top-header">
    <div class="header-left">
        <button class="mobile-toggle" onclick="toggleMobileMenu()" id="mobileToggleBtn">&#9776;</button>
        <span class="header-logo">CVtailro</span>
        <span class="header-badge">Admin</span>
    </div>
    <div class="header-right">
        <a href="/">&larr; Back to App</a>
    </div>
</div>
```

```css
.top-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: var(--header-height);   /* 56px */
    background: var(--sidebar-bg);  /* #0f172a (always dark) */
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 100;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
```

### Fixed Sidebar

```html
<nav class="sidebar" id="sidebarNav">
    <div class="sidebar-nav">
        <div class="nav-label">Main</div>
        <a class="nav-item active" data-tab="dashboard" onclick="switchTab('dashboard')">
            <span class="nav-icon">&#128202;</span>
            <span class="nav-text">Dashboard</span>
        </a>
        <!-- more nav-items -->
        <div class="nav-label">Monitoring</div>
        <!-- more nav-items -->
    </div>
    <div class="sidebar-footer">
        <a class="nav-item" onclick="logout()">
            <span class="nav-icon">&#9211;</span>
            <span class="nav-text">Logout</span>
        </a>
    </div>
</nav>
```

```css
.sidebar {
    position: fixed;
    top: var(--header-height);
    left: 0;
    bottom: 0;
    width: var(--sidebar-width);   /* 220px */
    background: var(--sidebar-bg); /* #0f172a always dark */
    display: flex;
    flex-direction: column;
    z-index: 90;
    overflow-y: auto;
    border-right: 1px solid rgba(255,255,255,0.06);
}
```

### Main Content Area

```css
.main-content {
    margin-left: var(--sidebar-width);
    margin-top: var(--header-height);
    padding: 24px 28px 48px;
    min-height: calc(100vh - var(--header-height));
}

/* Mobile: no sidebar offset */
@media (max-width: 768px) {
    .main-content {
        margin-left: 0;
        padding: 20px 16px 60px;
    }
}
```

### Dashboard Tab — 2-Column Grid

```css
.dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 20px;
    align-items: start;
}

.dashboard-sidebar {
    position: sticky;
    top: calc(var(--header-height) + 24px);
}

@media (max-width: 1024px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }
    .dashboard-sidebar {
        position: static;
    }
}
```

### KPI Row (reusable metric strip)

```css
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}
```

### Dashboard Chart Row

```css
.dashboard-chart-row {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

@media (max-width: 900px) {
    .dashboard-chart-row { grid-template-columns: 1fr; }
}
```

### System Stats Grid

```css
.sys-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

@media (max-width: 768px) {
    .sys-grid { grid-template-columns: repeat(2, 1fr); }
}
```

### Tab Switching (JS-driven)

Only one `.tab-content` is `.active` at a time. JavaScript `switchTab()` adds/removes the `.active` class and updates the sidebar `.nav-item.active`.

```css
.tab-content { display: none; }
.tab-content.active { display: block; }
```

---

## 3. Mobile Responsive Breakpoints

| Breakpoint      | Changes                                    |
|-----------------|--------------------------------------------|
| `max-width: 1024px` | Dashboard grid → single column        |
| `max-width: 900px`  | Chart row → single column             |
| `max-width: 768px`  | Sidebar → horizontal (mobile menu toggle), form columns stack, hero tighter padding, main-content full width |
| `max-width: 480px`  | Metrics/stats grids → 1-2 columns, auth modal max width, container padding tighter |
| `max-width: 375px`  | Very small phones: smaller fonts, tighter padding throughout |
