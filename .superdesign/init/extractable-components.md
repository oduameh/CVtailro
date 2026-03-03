# CVtailro — Extractable Components

Components that could be extracted as reusable design units for Superdesign. Each entry documents what it is, where it lives, its props/variants, and key CSS.

---

## 1. Site Header / Navigation Bar

**Source:** `index.html` lines ~2594–2642
**Used on:** index.html only (admin has its own `.top-header`)

### Variants
- **Signed Out:** Shows `.btn-google-signin` button
- **Signed In:** Shows `.user-menu-wrap` with avatar, name, chevron, dropdown
- **Admin User:** Dropdown includes "Admin Panel" link

### Key Elements
- `.site-header` — sticky bar, 56px height
- `.site-logo` — "CV" (blue) + "tailro" (text color)
- `.theme-toggle` — dark mode button with moon/sun SVG toggle
- `.header-auth` — auth controls container
- `.user-menu-trigger` — avatar pill button
- `.user-dropdown` — absolute dropdown menu

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `isAuthenticated` | true/false | Show sign-in button vs user menu |
| `userName` | string | Display in dropdown + trigger |
| `userEmail` | string | Display in dropdown info |
| `userPicture` | URL or null | Avatar image or initials |
| `isAdmin` | true/false | Show/hide Admin Panel link |
| `darkMode` | true/false | Moon/sun icon visibility |

---

## 2. Hero Section

**Source:** `index.html` lines ~2645–2664
**Used on:** index.html only

### Key Elements
- `.hero` — full-width section with centered content
- `.hero-inner` — max-width 640px container
- `h1` with `.hero-highlight` (gradient text)
- `.hero-sub` — subtitle paragraph
- `.hero-stats` — 3-card stat row (joined borders)
  - `.hero-stat` → `.hero-stat-value` + `.hero-stat-label`

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `title` | string | Main heading |
| `subtitle` | string | Sub-description |
| `stats` | array of {value, label} | Stat cards |

---

## 3. Upload Zone

**Source:** `index.html` lines ~2696–2714
**Used on:** index.html (form section)

### States
- **Default:** Dashed border, upload icon, "Drop your PDF here" text
- **Dragover:** `.dragover` — solid border, blue background
- **Has File:** `.has-file` — solid border, blue background, shows filename

### Key Elements
- `.upload-zone` — dashed-border container with hidden file input
- `.upload-icon` — 56px icon container
- `.upload-label` — instructional text
- `.upload-privacy` — small privacy note
- `.file-name` — displays selected filename

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `state` | default / dragover / has-file | Visual state |
| `fileName` | string or null | Show filename when file selected |
| `accept` | string | File input accept attribute |

---

## 4. Score Ring

**Source:** `index.html` results section
**Used on:** index.html (results)

### Key Elements
- `.score-section` — centered container
- `.score-ring-wrap` — 140x140 SVG ring
  - SVG `circle.score-ring-bg` (background track)
  - SVG `circle.score-ring-fill` (animated fill, color varies by score)
  - `.score-ring-value` — centered text overlay ("85%")
- `.score-ring-label` — "MATCH SCORE" uppercase label
- `.score-stats` — flex row (cosine similarity + missing keywords count)
- `.score-improvement-banner` — before/after comparison
- `.keywords-section` — keyword tags (.kw-matched + .kw-missing)

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `score` | 0–100 | Ring fill amount + color (green/yellow/red) |
| `originalScore` | 0–100 or null | Before score for improvement banner |
| `cosineSimilarity` | 0–1 | Displayed in stats |
| `missingKeywords` | number | Displayed in stats |
| `matchedKeywords` | string[] | Green tags |
| `missingKeywordsList` | string[] | Red tags |

---

## 5. Template Cards (Download Grid)

**Source:** `index.html` results section
**Used on:** index.html (results downloads)

### Key Elements
- `.template-grid` — 3-column grid
- `.template-card` — clickable card per template
  - `.template-preview` — colored preview rectangle
    - `.modern-preview` — blue gradient + left border
    - `.exec-preview` — neutral gradient + bottom border
    - `.min-preview` — plain background
  - `.template-name` — template name
  - `.template-desc` — short description

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `templates` | array of {name, desc, previewClass, downloadUrl} | Cards to render |
| `jobId` | string | Used to construct download URLs |

---

## 6. Progress Section

**Source:** `index.html` (progress section)
**Used on:** index.html (during pipeline)

### Key Elements
- `.progress-wrap` — centered card container
- `.progress-current-stage` — current stage label
- `.progress-bar-track` + `.progress-bar-fill` — animated progress bar
- `.progress-pct` — percentage text
- `.elapsed-time` — running timer
- `.time-estimate` — estimated time remaining
- `.reconnecting-badge` — SSE reconnect indicator
- `.tip-container` + `.tip-text` — rotating tips
- `.stage-list` — list of 6 pipeline stages
  - `.stage` — each stage row
    - `.stage-check` (pending / running / done / error icons)
    - `.stage-name`

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `percentage` | 0–100 | Bar fill + text |
| `currentStage` | string | Stage name display |
| `stages` | array of {name, status} | Stage list rendering |
| `elapsed` | seconds | Timer display |
| `tip` | string | Rotating tip text |

---

## 7. History Overlay

**Source:** `index.html` (history overlay)
**Used on:** index.html (logged-in users)

### Key Elements
- `.history-overlay` — fixed full-screen backdrop (blur)
- `.history-panel` — centered floating panel (640px max)
  - `.history-header` — title + close button
  - `.history-body` — scrollable list
    - `.history-card` — job entry
      - `.history-card-score` (colored badge: high/mid/low)
      - `.history-card-info` (title + meta + snippet)
      - `.history-card-badges` (template + mode badges)
      - `.history-card-arrow` (chevron)
    - `.history-empty` — empty state
    - `.history-loading` — loading state

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `isOpen` | true/false | Overlay visibility |
| `jobs` | array of job objects | List items |
| `isLoading` | true/false | Show loading indicator |
| `isEmpty` | true/false | Show empty state |

---

## 8. Profile Overlay

**Source:** `index.html` (reuses history overlay structure)
**Used on:** index.html (logged-in users)

### Key Elements
- Reuses `.history-overlay` + `.history-panel` structure
- `.profile-header` — avatar + name + email + provider badge
  - `.profile-avatar` — 80px circle (image or initials)
  - `.profile-name` / `.profile-email`
  - `.profile-provider-badge` (.google / .email)
- `.profile-stats` — 3-column grid
  - `.profile-stat` → `.profile-stat-value` + `.profile-stat-label`
- `.profile-section-title` — section heading
- `.profile-job-card` — recent job cards
- `.profile-edit-section` — name/password editing
  - `.profile-field` — form field
  - `.profile-save-btn`
- `.profile-empty` — empty state

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `user` | {name, email, picture, provider, isAdmin} | Profile info |
| `stats` | {totalJobs, avgScore, bestScore} | Stat cards |
| `recentJobs` | array | Job list |

---

## 9. Footer

**Source:** `index.html` (bottom)
**Used on:** index.html only

### Key Elements
- `.site-footer` — full-width section
- `.footer-brand` — "CVtailro"
- `.footer-links` — flex row of links (Privacy, Terms, Contact)
- `.footer-copy` — copyright text

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `links` | array of {label, href} | Footer navigation |
| `year` | number | Copyright year |

---

## 10. Admin Sidebar

**Source:** `admin.html` lines ~886–925
**Used on:** admin.html only

### Key Elements
- `.sidebar` — fixed left panel, 220px wide, dark background
- `.sidebar-nav` — scrollable nav section
  - `.nav-label` — section header ("Main", "Monitoring")
  - `.nav-item` — navigation link
    - `.nav-icon` — emoji icon
    - `.nav-text` — label text
    - `.active` state — white text, blue left border, blue bg
- `.sidebar-footer` — bottom section with Logout

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `activeTab` | string | Which nav-item has .active |
| `items` | array of {icon, label, tab} | Nav items |

---

## 11. Admin KPI Cards

**Source:** `admin.html` lines ~947–965, 971–989, etc.
**Used on:** admin.html (dashboard, analytics tabs)

### Key Elements
- `.kpi-row` — responsive grid container
- `.kpi-card` — individual metric card
  - `.kpi-value` — large number (with .accent / .success / .warning variants)
  - `.kpi-label` — uppercase label
  - `.kpi-sublabel` — optional secondary text
  - `::before` pseudo-element — 3px top accent on hover

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `value` | string | Main display value |
| `label` | string | Metric name |
| `sublabel` | string or null | Optional secondary label |
| `colorClass` | accent / success / warning / (none) | Value color |

---

## 12. Admin Top Header

**Source:** `admin.html` lines ~874–883
**Used on:** admin.html only

### Key Elements
- `.top-header` — fixed bar, always dark (#0f172a)
- `.header-left` — hamburger + logo + badge
  - `.mobile-toggle` — hidden on desktop
  - `.header-logo` — "CVtailro" white text
  - `.header-badge` — "Admin" blue pill
- `.header-right` — "← Back to App" link

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `showMobileToggle` | true/false | Hamburger visibility |

---

## 13. Auth Modal (index.html)

**Source:** `index.html` (auth modal overlay)
**Used on:** index.html

### Key Elements
- `.auth-modal-overlay` — fixed backdrop
- `.auth-modal` — centered card (420px max)
  - `.auth-modal-header` — title + close button
  - `.auth-tabs` — Sign In / Create Account tab switcher
  - `.auth-modal-body`
    - `.auth-form` — login or register form
    - `.auth-field` — label + input + error
    - `.auth-btn` — full-width submit
    - `.auth-divider` — "or" separator
    - `.auth-google-btn` — Google OAuth button
    - `.auth-error-msg` / `.auth-success-msg` — feedback messages
    - `.auth-footer` — helper text + link

### Variants
- **Sign In:** Email + password + forgot password link
- **Create Account:** Name + email + password + confirm + strength meter
- **Forgot Password:** Email-only form
- **Reset Password:** New password + confirm

---

## 14. Toast Notification

**Source:** `index.html`
**Used on:** index.html

### Key Elements
- `.toast` — fixed position (top-right), slides in
- `.toast.visible` — visible state
- Variants: `.toast-error`, `.toast-info`, `.toast-success`

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `message` | string | Toast text |
| `type` | error / info / success | Color variant |
| `visible` | true/false | Show/hide with animation |

---

## 15. Chart Card (admin.html)

**Source:** `admin.html` lines ~667–700
**Used on:** admin.html (dashboard, analytics)

### Key Elements
- `.chart-card` — card container with border
- `.chart-card-header` — title + optional subtitle
- `.chart-container` — relative container with fixed height (220px)
  - `<canvas>` — Chart.js canvas

### Props
| Prop | Values | Effect |
|------|--------|--------|
| `title` | string | Header text |
| `subtitle` | string or null | Optional right-side text |
| `height` | string | Container height override |
| `chartType` | bar / doughnut | Chart.js type |

---

## Extraction Priority

| Priority | Component | Reason |
|----------|-----------|--------|
| High | Site Header | Used on every page, complex states |
| High | Score Ring | Core feature, visually distinctive |
| High | Upload Zone | Key interaction point |
| High | Template Cards | Core download experience |
| High | Auth Modal | Complex multi-state component |
| Medium | Hero Section | Marketing-critical, easy to extract |
| Medium | Progress Section | Complex state machine |
| Medium | History Overlay | Reusable overlay pattern |
| Medium | Profile Overlay | Reusable overlay pattern |
| Medium | Toast | Utility component |
| Medium | Admin KPI Cards | Repeated pattern in admin |
| Medium | Admin Sidebar | Admin navigation |
| Low | Footer | Simple, rarely changes |
| Low | Admin Top Header | Simple bar |
| Low | Chart Card | Wrapper for Chart.js |
