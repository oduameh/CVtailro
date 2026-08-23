# CVtailro Design System — "Precision Instrument" (v3, approved 2026-08)

## Product Context
AI resume tailoring SaaS. Users sign in with Google, upload a PDF resume, paste a job description, get an AI-tailored resume (8 PDF templates + DOCX), match report, cover letter, talking points. Free while in beta.
Live: https://cvtailro-production.up.railway.app
Stack: Flask/Jinja2, all CSS inline in `<style>` blocks, vanilla JS. Dark mode via `html.dark` class + localStorage key, blocking init script in `<head>` — preserve this mechanism exactly.

## Design Language
A technical power-tool aesthetic on the level of Linear or Vercel. Sharp geometry, crisp 1px borders, monospace micro-labels, one signal color used with discipline. Data-dense but calm. The opposite of generic AI-SaaS: no gradients, no glows, no soft shadows, no rounded-pill everything.

### Fonts (Google Fonts)
```
https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap
```
- **Display / headings**: `'Space Grotesk', system-ui, sans-serif` — hero h1 `clamp(38px, 5.5vw, 64px)`, weight 600, line-height 1.08, letter-spacing -0.02em.
- **Body / UI**: `'Inter', -apple-system, sans-serif` — 15px body (14px dense areas), line-height 1.6.
- **Mono micro-copy**: `'JetBrains Mono', ui-monospace, monospace` — ALL labels, kickers, field labels, numbers, timestamps, stats captions, nav items. Pattern: uppercase, 11px, weight 500-700, letter-spacing 0.08em. Kickers prefixed with `// ` (e.g. `// RESUME → JOB DESCRIPTION`). Section markers numbered: `01 · RESUME FILE`, `02 · JOB DESCRIPTION`.

### Color Tokens (KEEP existing variable NAMES in :root / html.dark; change values)
| Variable | Light | Dark |
|---|---|---|
| `--bg` | `#FAFAFA` | `#09090B` true black |
| `--card` | `#FFFFFF` | `#18181B` zinc-900 |
| `--primary` | `#EA580C` international orange | `#FF4D00` |
| `--primary-hover` | `#C2410C` | `#FF6A33` |
| `--primary-bg` | `rgba(234,88,12,0.07)` | `rgba(255,77,0,0.10)` |
| `--accent` = primary | same | same |
| `--border` | `#E4E4E7` zinc-200 | `#27272A` zinc-800 |
| `--border-light` | `#F4F4F5` | `#1F1F23` |
| `--text` | `#09090B` | `#F4F4F5` |
| `--text-secondary` | `#52525B` zinc-600 | `#A1A1AA` |
| `--text-tertiary` | `#A1A1AA` | `#71717A` |
| `--score-high` | `#15803D` | `#4ADE80` |
| `--score-mid` | `#B45309` | `#FBBF24` |
| `--score-low` | `#DC2626` | `#F87171` |
| `--error` | `#DC2626` | `#F87171` |

### Geometry & Elevation
- `--radius: 3px`, `--radius-lg: 6px`. Buttons/inputs radius 3-4px max.
- NO decorative shadows anywhere; flat surfaces + 1px borders. Focus: `outline: 2px solid var(--primary); outline-offset: 2px`.
- Hero-only texture: faint dotted grid (`background-image: radial-gradient(var(--border) 1px, transparent 1px); background-size: 24px 24px;`) at low opacity behind the hero section only.

### Components
- **Primary button** ("Run tailoring", sign-in): solid `var(--primary)`, white text, MONO uppercase label 12px ls .08em (e.g. `RUN TAILORING →`), padding 13px 26px, radius 3px; hover `--primary-hover`.
- **Secondary button**: 1px `var(--border)` border, transparent bg, mono uppercase label; hover: border/text → ink.
- **Header**: bg blur-less solid `--bg`, hairline bottom border, wordmark "CVtailro" in Space Grotesk 700 with orange period ("CVtailro."), nav links mono uppercase `// BLOG` style, ghost auth buttons.
- **Hero**: left-aligned compact grotesque headline; mono kicker above; supporting paragraph max-width 52ch; stats = three bordered instrument cells in a row (hairline borders, big Space Grotesk or mono numerals ~28px, tiny mono captions underneath).
- **Form = control panel**: each fieldset headed by mono marker label (`01 · RESUME FILE`), fields with card bg + hairline border radius 3px; upload dropzone dashed hairline; segmented Conservative/Aggressive control styled as a physical toggle group (square corners, active = ink fill); textarea with mono char counter top-right.
- **Progress stages**: vertical list with status dots (done = filled accent square/dot, running = pulsing opacity dot, todo = hollow), stage names Inter 500, mono timestamps/percent right-aligned.
- **Score ring & improvement banner**: keep mechanics; numerals Space Grotesk 700; banner flat `--primary-bg` tint + hairline border, mono caption.
- **Keyword chips**: square-ish (radius 2px), hairline border, mono 11px lowercase/verbatim keywords; matched = subtle green tint border-left, missing = red tint border-left.
- **Cards/overlays/modals/tabs/toasts/footer**: flat card bg, hairline borders, mono uppercase section titles; footer multi-column minimal with hairline top rule and mono column labels.
- **Admin-grade details welcome where cheap**: thin scrollbars, tabular numbers (`font-variant-numeric: tabular-nums`) for scores/counters.

### Motion
150-200ms ease on colors/borders/opacity. Results reveal: fadeInUp 250ms. Running-dot pulse via opacity keyframes only. No scale/bounce/glow.

### Hard Rules
1. NEVER gradients (except the dotted-grid radial-gradient texture trick above), never glows/colored shadows, never neon beyond the single accent.
2. Keep ALL existing element IDs, JS-referenced classes, data-* attributes, handlers, aria labels, form names — reskin, not rebuild.
3. Dark mode keeps working (same variable names, values swapped).
4. Fully responsive at 768/640/480 breakpoints; hero cells stack, control panel single-column on mobile.
