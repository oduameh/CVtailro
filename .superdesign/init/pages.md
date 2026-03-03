# CVtailro — Pages & Section Dependency Trees

---

## Page 1: `templates/index.html` — Public App

**Route:** `GET /`
**Size:** ~5200 lines (HTML + inline CSS + inline JS)
**Role:** The entire user-facing SPA. Resume upload, tailoring, results, history, profile, auth — all in one page.

### Section Dependency Tree

```
index.html
├── <head>
│   ├── Meta tags (SEO, Open Graph, Twitter Card)
│   ├── Google Fonts (DM Sans)
│   ├── Inline dark mode init script
│   ├── <style> — ALL CSS (~2500 lines)
│   │   ├── :root + html.dark variables
│   │   ├── Global resets
│   │   ├── .site-header (sticky header)
│   │   ├── .theme-toggle
│   │   ├── .hero + .hero-stats
│   │   ├── .trust-bar
│   │   ├── .container
│   │   ├── .card (generic)
│   │   ├── .form-columns (2-col layout)
│   │   ├── .upload-zone + .upload-icon
│   │   ├── textarea styles
│   │   ├── .controls-row + .toggle-group
│   │   ├── .btn-run + .cta-row
│   │   ├── .error-banner
│   │   ├── .progress-section + .progress-wrap + .stage
│   │   ├── .score-section + .score-ring-*
│   │   ├── .keywords + .kw-matched + .kw-missing
│   │   ├── .template-grid + .template-card
│   │   ├── .dl-btn-secondary + .additional-files
│   │   ├── .talking-points-card
│   │   ├── .tabs + .tab + .tab-content
│   │   ├── .copy-btn
│   │   ├── .btn-again + .again-wrap
│   │   ├── .site-footer
│   │   ├── .score-improvement-banner
│   │   ├── .what-changed-card + .change-stats
│   │   ├── .collapsible-header + .collapsible-content
│   │   ├── Animations (@keyframes fadeInUp, slideInUp, pulse)
│   │   ├── Mobile responsive (@media breakpoints)
│   │   ├── .header-auth + .btn-google-signin
│   │   ├── .user-menu-wrap + .user-dropdown + .dropdown-item
│   │   ├── .toast (notification)
│   │   ├── .history-overlay + .history-panel + .history-card
│   │   ├── .profile-* (profile overlay)
│   │   ├── .elapsed-time + .reconnecting-badge
│   │   ├── .detail-tab
│   │   ├── .auth-modal-overlay + .auth-modal + .auth-form
│   │   ├── .auth-field + .auth-btn + .auth-divider + .auth-google-btn
│   │   ├── .password-strength + .password-match
│   │   ├── .profile-edit-section + .profile-field
│   │   ├── .reset-pw-overlay
│   │   └── Accessibility (.skip-link, :focus-visible)
│   └── JSON-LD structured data
│
├── <body>
│   ├── .skip-link (a11y)
│   ├── <noscript> fallback
│   │
│   ├── HEADER (.site-header)
│   │   ├── .site-logo (CVtailro)
│   │   ├── .theme-toggle (dark mode button)
│   │   └── .header-auth
│   │       ├── .btn-google-signin (signed out state)
│   │       └── .user-menu-wrap (signed in state)
│   │           ├── .user-menu-trigger (avatar + name + chevron)
│   │           └── .user-dropdown
│   │               ├── .dropdown-user-info (name + email)
│   │               ├── .dropdown-item — History
│   │               ├── .dropdown-item — Profile
│   │               ├── .dropdown-item — Admin Panel (admin only)
│   │               └── .dropdown-item.sign-out — Sign Out
│   │
│   ├── HERO (.hero)
│   │   ├── h1 with .hero-highlight
│   │   ├── .hero-sub (description)
│   │   └── .hero-stats (3 stat cards in a row)
│   │
│   ├── TRUST BAR (.trust-bar, hidden)
│   ├── DEMO BANNER (inline "Free & unlimited")
│   │
│   ├── MAIN CONTENT (.container#main-content)
│   │   ├── .error-banner (hidden)
│   │   ├── #configBanner (hidden)
│   │   │
│   │   ├── FORM SECTION (#formSection)
│   │   │   ├── .form-columns
│   │   │   │   ├── .card#uploadCard — Upload Zone
│   │   │   │   │   ├── h2 "Upload Your Resume"
│   │   │   │   │   ├── .upload-zone#uploadZone
│   │   │   │   │   │   ├── input[type=file]
│   │   │   │   │   │   ├── .upload-icon (SVG)
│   │   │   │   │   │   ├── .upload-label
│   │   │   │   │   │   └── .upload-privacy
│   │   │   │   │   └── .file-name#fileName
│   │   │   │   │
│   │   │   │   └── .card — Job Description
│   │   │   │       ├── h2 "Job Description"
│   │   │   │       ├── textarea#jobDescription
│   │   │   │       ├── .char-count
│   │   │   │       └── .inline-error
│   │   │   │
│   │   │   ├── .controls-row
│   │   │   │   ├── .toggle-group (Conservative / Aggressive mode)
│   │   │   │   └── #modelSelectWrap (model dropdown, optional)
│   │   │   │
│   │   │   └── .cta-row
│   │   │       ├── .btn-run#tailorBtn ("Tailor My Resume")
│   │   │       └── .cta-hint
│   │   │
│   │   ├── PROGRESS SECTION (.progress-section#progressSection)
│   │   │   └── .progress-wrap
│   │   │       ├── .progress-current-stage
│   │   │       ├── .progress-bar-track + .progress-bar-fill
│   │   │       ├── .progress-pct
│   │   │       ├── .elapsed-time
│   │   │       ├── .time-estimate
│   │   │       ├── .reconnecting-badge
│   │   │       ├── .tip-container + .tip-text
│   │   │       └── .stage-list (6 stages)
│   │   │
│   │   └── RESULTS SECTION (.results-section#resultsSection)
│   │       ├── .card — Score Section (.score-section)
│   │       │   ├── .score-ring-wrap (SVG circle + value)
│   │       │   ├── .score-ring-label
│   │       │   ├── .score-stats (cosine sim + missing keywords)
│   │       │   ├── .score-improvement-banner (before → after)
│   │       │   └── .keywords-section (.kw-matched + .kw-missing)
│   │       │
│   │       ├── .card — What Changed (.what-changed-card)
│   │       │   ├── .change-stats (3-col grid of stat metrics)
│   │       │   └── .changes-panel (collapsible sections)
│   │       │
│   │       ├── .card — Downloads
│   │       │   ├── .downloads-heading
│   │       │   ├── .dl-usage-hint
│   │       │   ├── .template-grid (3 template cards: Modern/Executive/Minimal)
│   │       │   │   └── .template-card (preview + name + desc)
│   │       │   └── .additional-toggle + .additional-files
│   │       │       └── .additional-files-grid (.dl-btn-secondary links)
│   │       │
│   │       ├── .card — Resume Preview
│   │       │   ├── .preview-header + .copy-btn
│   │       │   └── #previewContent (rendered markdown)
│   │       │
│   │       ├── .card.talking-points-card — Talking Points
│   │       │   ├── .collapsible-header
│   │       │   └── .talking-points-content (rendered markdown)
│   │       │
│   │       ├── .card — Tabs (Cover Letter / Email Templates / More)
│   │       │   ├── .tabs (.tab buttons)
│   │       │   └── .tab-content panels
│   │       │
│   │       └── .again-wrap (.btn-again "Tailor Another Resume")
│   │
│   ├── FOOTER (.site-footer)
│   │   ├── .footer-brand
│   │   ├── .footer-links (Privacy, Terms, Contact)
│   │   └── .footer-copy
│   │
│   ├── HISTORY OVERLAY (.history-overlay#historyOverlay)
│   │   └── .history-panel
│   │       ├── .history-header (h2 + close button)
│   │       └── .history-body (list of .history-card items)
│   │
│   ├── AUTH MODAL (.auth-modal-overlay#authModalOverlay)
│   │   └── .auth-modal
│   │       ├── .auth-modal-header (title + close)
│   │       ├── .auth-tabs (Sign In / Create Account)
│   │       └── .auth-modal-body
│   │           ├── .auth-form#authLoginForm (email + password + forgot link)
│   │           ├── .auth-form#authRegisterForm (name + email + password + confirm)
│   │           ├── .auth-divider ("or")
│   │           └── .auth-google-btn (Continue with Google)
│   │
│   ├── RESET PASSWORD OVERLAY (.reset-pw-overlay#resetPwOverlay)
│   │
│   └── TOAST (.toast#toastEl)
│
└── <script> — ALL JavaScript (~2500 lines)
    ├── Dark mode toggle
    ├── Auth management (checkAuth, signOut, openAuthModal)
    ├── Email/password auth (register, login, verify, reset)
    ├── User menu toggle
    ├── File upload handling (drag & drop)
    ├── Form validation + submission
    ├── SSE progress stream handling
    ├── Results rendering (score ring, keywords, downloads, preview)
    ├── Markdown → HTML rendering (custom parser)
    ├── History overlay (load, render, detail view)
    ├── Profile overlay (load, edit, password change)
    ├── Toast notification system
    └── Utility functions
```

---

## Page 2: `templates/admin.html` — Admin Dashboard

**Route:** `GET /admin`
**Size:** ~2188 lines (HTML + inline CSS + inline JS)
**Role:** Admin-only dashboard for configuration, user management, analytics, monitoring.

### Section Dependency Tree

```
admin.html
├── <head>
│   ├── Google Fonts (DM Sans)
│   ├── Chart.js CDN
│   ├── Inline dark mode init script
│   └── <style> — ALL CSS (~820 lines)
│       ├── :root + html.dark variables
│       ├── .auth-wrapper + .auth-card (login screen)
│       ├── .app-layout
│       ├── .top-header + .header-badge
│       ├── .sidebar + .nav-item
│       ├── .main-content
│       ├── .dashboard-grid + .dashboard-sidebar
│       ├── .tab-content
│       ├── .tab-header
│       ├── .card + .card-header
│       ├── .section-title
│       ├── .metrics-grid + .metric-card
│       ├── .form-group + .checkbox-group
│       ├── .btn + variants (primary, secondary, danger, sm)
│       ├── .alert (success, error)
│       ├── .analytics-table + .analytics-bar
│       ├── .search-input
│       ├── .user-card
│       ├── .sys-grid + .sys-stat
│       ├── .activity-item + .activity-avatar
│       ├── .resume-tabs + .resume-tab-btn
│       ├── .live-dot + @keyframes pulse
│       ├── .empty-state
│       ├── .chart-card + .chart-container
│       ├── .kpi-row + .kpi-card
│       └── Mobile responsive (@media)
│
├── <body>
│   ├── AUTH SECTION (#authSection.auth-wrapper)
│   │   ├── .auth-card#setPasswordCard (first-time setup)
│   │   │   ├── .auth-logo (CVtailro + "Admin Dashboard")
│   │   │   ├── h2 "Set Admin Password"
│   │   │   ├── .form-group — New Password
│   │   │   ├── .form-group — Confirm Password
│   │   │   └── .btn.btn-primary "Set Password & Login"
│   │   │
│   │   └── .auth-card#loginCard (return login)
│   │       ├── .auth-logo
│   │       ├── h2 "Admin Login"
│   │       ├── .form-group — Password
│   │       └── .btn.btn-primary "Login"
│   │
│   └── DASHBOARD (#configSection.app-layout)
│       ├── .top-header
│       │   ├── .header-left (hamburger + logo + "Admin" badge)
│       │   └── .header-right (← Back to App link)
│       │
│       ├── .sidebar#sidebarNav
│       │   ├── .sidebar-nav
│       │   │   ├── nav-label "Main"
│       │   │   ├── nav-item — Dashboard (📊)
│       │   │   ├── nav-item — Configuration (⚙)
│       │   │   ├── nav-item — Users & Resumes (👥)
│       │   │   ├── nav-label "Monitoring"
│       │   │   ├── nav-item — Diagnostics (🔍)
│       │   │   ├── nav-item — Analytics (📈)
│       │   │   ├── nav-item — Errors (🔴)
│       │   │   └── nav-item — System (💻)
│       │   └── .sidebar-footer
│       │       └── nav-item — Logout (⏻)
│       │
│       └── .main-content
│           ├── TAB: Dashboard (#tab-dashboard)
│           │   ├── .tab-header (h1 + refresh button)
│           │   ├── .kpi-row (4 KPIs: Users, Jobs, API Key, Jobs Today)
│           │   └── .dashboard-grid
│           │       ├── .dashboard-main
│           │       │   ├── .kpi-row (Success Rate, Cost, Active Pipelines, Match Δ)
│           │       │   ├── .dashboard-chart-row
│           │       │   │   ├── .chart-card — Jobs Over Time (Chart.js bar)
│           │       │   │   └── .chart-card — Jobs by Status (Chart.js doughnut)
│           │       │   └── .kpi-row (Saved Resumes, Jobs This Week, Jobs This Month)
│           │       └── .dashboard-sidebar
│           │           └── .card — Recent Jobs
│           │               └── #recentActivity (.activity-item list)
│           │
│           ├── TAB: Configuration (#tab-config)
│           │   ├── .card — API Key
│           │   │   ├── .form-group (password input + Show/Hide button)
│           │   │   └── Test Key button + result
│           │   ├── .card — Model & Rate Limiting
│           │   │   ├── .form-group — Default Model (select)
│           │   │   ├── .checkbox-group — Allow user model selection
│           │   │   └── .form-group — Rate Limit (number input)
│           │   └── .card — Save Configuration
│           │       ├── #saveAlert
│           │       └── .btn-primary "Save Configuration"
│           │
│           ├── TAB: Users & Resumes (#tab-users)
│           │   ├── .search-input
│           │   ├── .card — All Users (#usersList)
│           │   │   └── .user-card items (click to view jobs)
│           │   ├── .card#userJobsPanel (hidden, shows user's jobs)
│           │   └── .card#resumeViewerPanel (hidden, shows resume content)
│           │       ├── .resume-tabs (Tailored / Talking Points / Original / JD)
│           │       └── #resumeContent (pre-formatted text)
│           │
│           ├── TAB: Diagnostics (#tab-diagnostics)
│           │   └── .card — Health Checks
│           │       ├── Run Diagnostics + Copy buttons
│           │       └── <pre> output
│           │
│           ├── TAB: Analytics (#tab-analytics)
│           │   ├── .kpi-row (Total Jobs, Total Tokens, Est. Cost, Avg Cost/Job)
│           │   ├── .dashboard-chart-row (Jobs Over Time + Status charts)
│           │   └── .card — Usage by Model (.analytics-table)
│           │
│           ├── TAB: Errors (#tab-errors)
│           │   └── .card — Error Log
│           │       └── #errorLog (styled error items)
│           │
│           └── TAB: System (#tab-system)
│               ├── .sys-grid (6 stats: Active Pipelines, Queue, Memory, Threads, etc.)
│               └── .card — Service Info
│
└── <script> — ALL JavaScript (~870 lines)
    ├── Tab management (switchTab)
    ├── Mobile menu toggle
    ├── System auto-refresh (10s interval)
    ├── User search/filter
    ├── Auth (checkAuth, login, setPassword, logout)
    ├── Config panel (showConfigPanel, saveConfig)
    ├── Model loading
    ├── Stats loading (loadStats, loadDashboardData)
    ├── Chart rendering (Chart.js: bar + doughnut)
    ├── Analytics loading
    ├── Error log loading
    ├── User management (loadUsersList, loadUserJobs)
    ├── Resume viewer (viewResume, showResumeTab)
    ├── Diagnostics (runDiagnostics, copyDiagOutput)
    └── Utilities (escapeHtml)
```

---

## Additional Pages

### `templates/privacy.html`
**Route:** `GET /privacy`
Simple static content page.

### `templates/terms.html`
**Route:** `GET /terms`
Simple static content page.

### `templates/contact.html`
**Route:** `GET /contact`
Simple static content page.
