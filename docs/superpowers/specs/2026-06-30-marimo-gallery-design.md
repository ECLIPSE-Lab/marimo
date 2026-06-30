# Pelz Lab marimo gallery — landing page redesign

**Date:** 2026-06-30
**Status:** Approved
**Scope:** Replace the bare `generate_index()` output in `scripts/build.py` with a
polished, Pelz-Lab-branded gallery that shows a real thumbnail, description, and tags
for each marimo app deployed at https://pelzlab.science/marimo/.

## Goal

The current landing page is an unstyled Tailwind grid of cards, each holding only a
titleized filename and an "Open Notebook" link. Make it a presentable lab tools
gallery: hero header, responsive card grid, real screenshot thumbnails, short
descriptions and tags.

## Components

### 1. App metadata table (`scripts/build.py`)
A module-level `APPS` dict keyed by file **stem** → `{title, description, tags, order}`.
- Drives card copy and ordering.
- Apps not present fall back to a titleized stem with no description/tags, so the build
  never breaks when a new app is added.

Initial entries:

| stem | title | description | tags |
|---|---|---|---|
| ptychographic_ctf | Ptychographic CTF | Contrast-transfer function for ptychographic phase imaging | ptychography, CTF |
| ptychographic_sampling | Ptychographic Sampling | Real- vs reciprocal-space sampling trade-offs | ptychography, sampling |
| ptychographic_coherence | Ptychographic Coherence | Partial-coherence effects on reconstruction | ptychography, coherence |
| ff_stem_ssnr | FF-STEM SSNR | Spectral SNR / DQE: ptychography vs ADF vs full-field STEM | STEM, SSNR, DQE |
| probe | STEM Probe | Interactive STEM probe former — aperture & aberrations | probe, aberrations |

(Descriptions refined against each app's source during implementation.)

### 2. Rewritten `generate_index()` (`scripts/build.py`)
Emits a self-contained HTML page:
- **Hero**: "Pelz Lab" wordmark + tagline "Interactive ptychography & electron-microscopy
  tools", link to pelzlab.science, slate/teal gradient background.
- **Card grid**: responsive 1 / 2 / 3 columns. Each card = thumbnail (4:3, `object-cover`)
  → title → one-line description → tag chips → "Open app →" button. Whole card clickable,
  hover lift/shadow.
- **Footer**: "Built with marimo · runs in your browser via WebAssembly" + GitHub link.
- **Palette**: `slate-50` page, `slate-900` hero, teal `#1f7a6b` accent, gray borders.
- Tailwind via CDN (matches existing approach) + minimal inline CSS for the gradient/cards.

### 3. Thumbnails
- Stored in repo at `thumbnails/<stem>.png` (4:3).
- `build.py` copies `thumbnails/` → `_site/thumbnails/` at build time (new step).
- Cards reference `thumbnails/<stem>.png`.
- **Missing thumbnail → CSS gradient placeholder** with the app's initial. CI never
  depends on the images existing.

### 4. One-time screenshot generator (`scripts/screenshots.py`)
- **Not part of CI.** Run locally to (re)generate thumbnails.
- Uses Playwright (installed locally only). Builds the site, serves `_site/` on a local
  port, loads each app HTML, waits for the marimo/WASM render to settle, captures a
  cropped 4:3 PNG into `thumbnails/`.
- Output PNGs are committed to the repo.

## Data flow

```
scripts/screenshots.py  (local, one-time)
   build _site → serve → headless render each app → crop → thumbnails/<stem>.png  → git commit

scripts/build.py  (CI, every deploy)
   export apps to _site/apps/*.html
   copy thumbnails/ → _site/thumbnails/
   generate_index(): APPS metadata + thumbnails → _site/index.html
```

## Error handling / robustness
- Build works with no `thumbnails/` dir (placeholder rendered).
- Unknown app → titleized fallback, no crash.
- `is_marimo_notebook()` filtering unchanged.
- Screenshot script failures are isolated per-app and never block the build.

## Testing
- `tests/test_build_index.py`: `generate_index()` over a sample app list produces HTML
  containing the hero text, each app's card (title + open link + thumbnail-or-placeholder),
  and references the right `apps/*.html` paths.
- Manual: `python scripts/build.py` then open `_site/index.html`.

## Out of scope
- CI-side screenshotting.
- Per-app deep redesign; only the gallery/index changes.
