# Staged Floor Plan Recognition Pipeline

**Date:** 2026-02-16
**Status:** Approved
**Problem:** Gemini returns inaccurate polygon coordinates when processing floor plans with furniture, text, and dimension labels.

## Approach

Replace the single-pass Gemini call with a 3-stage pipeline:

1. **Sharp preprocessing** — grayscale + threshold + contrast enhancement
2. **Gemini Image cleaning** — AI removes furniture, text, dimensions; outputs clean line drawing
3. **Gemini coordinate extraction** — extracts precise polygon coordinates from cleaned image

## Elements (5 active, rest commented out)

**Active:**
- `wall` — array of polygons (inner + outer edges showing thickness)
- `door` — array of rectangular polygons
- `window` — array of rectangular polygons
- `balcony` — polygon
- `balcony_window` — polygon/array

**Commented out:**
- apartments, bedroom, living_room, other_room, kitchen_zone, kitchen_table, sink, cooker

## Models

| Stage | Model | Purpose |
|-------|-------|---------|
| Preprocessing | Sharp (Node.js) | Binarization, contrast |
| Cleaning | `gemini-2.5-flash-image` (GA) | Remove non-structural elements |
| Extraction | `gemini-3-flash-preview` | Coordinate extraction |

## File Changes

### `src/gemini.js`
- Add `preprocessImage(imageBuffer)` — Sharp grayscale + threshold
- Add `cleanFloorplan(imageBuffer, mimeType)` — Gemini Image cleaning call
- Update `analyzeFloorplan()` — orchestrate 3 stages sequentially

### `src/prompt.js`
- Comment out schema fields: apartments, bedroom, living_room, other_room, kitchen_zone, kitchen_table, sink, cooker
- Comment out prompt instructions for those elements
- Add cleaning prompt for Gemini Image stage

### `public/app.js`
- Comment out LAYER_CONFIG entries for removed elements
- Comment out rendering logic for removed elements

### `package.json`
- Add `sharp` dependency

## Pipeline Flow

```
Image (user upload)
  │
  ├─ Stage 1: Sharp preprocessing
  │   └─ grayscale → threshold(180) → contrast boost
  │   └─ Output: high-contrast B&W PNG buffer
  │
  ├─ Stage 2: Gemini Image cleaning
  │   └─ Model: gemini-2.5-flash-image
  │   └─ Prompt: "Remove furniture, text, dimensions. Keep walls, doors, windows, balcony only"
  │   └─ Output: cleaned PNG image (base64 → buffer)
  │
  └─ Stage 3: Gemini coordinate extraction
      └─ Model: gemini-3-flash-preview
      └─ Prompt: updated prompt.js (5 elements only)
      └─ Output: JSON with normalized 0-1000 coordinates
```

## Expected Impact

- Reduced noise in input image → better coordinate precision
- Fewer elements to detect → more focused AI attention
- Sharp preprocessing is deterministic and fast (ms)
- Total API cost ~2x current (two Gemini calls instead of one)

## Risks

- Gemini Image may distort wall geometry during cleaning
- Sharp threshold value (180) may need tuning per floor plan style
- Mitigation: save intermediate images for debugging, make Sharp optional
