# Floorplan Web Viewer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a KonvaJS-based web viewer that lets users upload floor plan images, analyze them via the API, and view colored polygon overlays with layer toggles.

**Architecture:** Single-page vanilla JS app served as static files from `public/` by Fastify via `@fastify/static`. KonvaJS loaded from CDN. No build step.

**Tech Stack:** HTML, CSS, vanilla JS, KonvaJS (CDN), @fastify/static

---

### Task 1: Install @fastify/static and register in server

**Files:**
- Modify: `index.js`

**Step 1: Install dependency**

Run: `npm install @fastify/static`

**Step 2: Register static file serving in index.js**

Add import at top of `index.js`:
```javascript
import fastifyStatic from '@fastify/static'
import { dirname, join } from 'node:path'
```

Note: `fileURLToPath` is already imported. Add `dirname` and `join` to the path import.

Inside `buildApp()`, after multipart registration, add:
```javascript
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

await app.register(fastifyStatic, {
  root: join(__dirname, 'public'),
  prefix: '/',
})
```

**Important:** Remove or update the existing `GET /` route since `@fastify/static` will serve `public/index.html` at `/`. Change the health check to `GET /api/health`:
```javascript
app.get('/api/health', async () => {
  return { status: 'ok', service: 'floorplan-service' }
})
```

Also update `test/api.test.js` to use `/api/health` instead of `/`.

**Step 3: Create empty public directory with minimal index.html**

Run: `mkdir -p public`

Create `public/index.html`:
```html
<!DOCTYPE html>
<html><body><h1>Floorplan Viewer</h1></body></html>
```

**Step 4: Run tests, fix api.test.js**

Run: `npm test`
Fix the GET / test to use `/api/health`.

**Step 5: Commit**

```bash
git add index.js public/ package.json package-lock.json test/api.test.js
git commit -m "feat: serve static files from public/ via @fastify/static"
```

---

### Task 2: Create the HTML layout and CSS

**Files:**
- Create: `public/index.html`
- Create: `public/style.css`

**Step 1: Write public/style.css**

Dark theme UI with sidebar layout. Key classes:
- `.app` — full-height flex column
- `header` — top bar with title
- `.main` — flex row: sidebar + canvas
- `.sidebar` — 220px left panel with upload button, layer checkboxes, info
- `.canvas-container` — flex-grow area for KonvaJS
- `.status-bar` — bottom info bar
- `.loading-overlay` — absolute overlay with spinner
- `.upload-btn` — styled red button
- `.layer-item` — checkbox + color dot + label row

Color scheme: background `#1a1a2e`, sidebar `#16213e`, accent `#e94560`.

**Step 2: Write public/index.html**

Structure:
```
div.app
  header > h1 "Floorplan Analyzer"
  div.main
    aside.sidebar
      div > h3 "Upload" + hidden file input + upload button
      div > h3 "Layers" + #layers-panel
      div > h3 "Info" + #info-panel
    div.canvas-container
      div#konva-container
      div.loading-overlay.hidden#loading > spinner + text
  div.status-bar > #status-text + #hover-info
  script src="https://unpkg.com/konva@9/konva.min.js"
  script src="app.js"
```

**Step 3: Verify it loads in browser**

**Step 4: Commit**

```bash
git add public/index.html public/style.css
git commit -m "feat: add HTML layout and CSS for floorplan viewer"
```

---

### Task 3: Create app.js — core KonvaJS viewer with upload and rendering

**Files:**
- Create: `public/app.js`

This is the main task. The file has these sections:

**1. LAYER_CONFIG** — color/opacity/label mapping for each element type:
```
apartments: #264653, 0.15 | wall: #555555, 0.3 | door: #FF6B35, 0.4
window: #00B4D8, 0.4 | balcony_window: #48CAE4, 0.3 | balcony: #90E0EF, 0.2
bedroom: #7B2D8E, 0.25 | living_room: #2D6A4F, 0.25 | other_room: #E9C46A, 0.25
kitchen_table: #E76F51, 0.3 | kitchen_zone: #F4A261, 0.2
sink: #219EBC, 0.5 | cooker: #FB8500, 0.5
```

**2. KonvaJS Setup** — Stage (draggable, fills container), imageLayer, one Layer per element type in `konvaLayers` map.

**3. Zoom** — Mouse wheel handler on stage using pointer-relative zoom (scaleBy = 1.05). Standard KonvaJS pattern: calculate mousePointTo, apply new scale, recalculate position.

**4. Resize** — Window resize listener updates stage dimensions.

**5. Upload flow** — `handleFileUpload(e)`:
- Get file from input
- Show image immediately via `URL.createObjectURL`
- Show loading overlay
- POST `/analyze` with FormData
- On success: call `renderPolygons(result)`, `updateInfo(result)`
- On error: show error in status bar
- Finally: hide overlay, re-enable button

**6. loadImage(url)** — Returns promise. Creates `new Image()`, on load creates `Konva.Image`, adds to imageLayer. Auto-fits: calculates scale to fit canvas with 0.9 padding, centers.

**7. renderPolygons(data)** — For each key in LAYER_CONFIG, get data[key], normalize (single polygon vs array of polygons), create `Konva.Line` with `closed: true`, fill, stroke. Add hover events: mouseenter increases opacity and shows label in status bar (use `textContent`), mouseleave restores.

**8. normalizePolygons(key, data)** — If `data[0][0]` is a number, it's a single polygon `[[x,y],...]` → wrap in array. Otherwise it's already `[[[x,y],...],...]`.

**9. buildLayerToggles()** — Creates checkbox + color dot + label for each layer. Checkbox change toggles `konvaLayers[key].visible()`. Uses DOM creation methods (createElement, appendChild, createTextNode) — NO innerHTML.

**10. updateInfo(data)** — Updates #info-panel with size, scale, counts. Uses `textContent` with line breaks via createElement('br') — NO innerHTML.

**11. Init** — Call `buildLayerToggles()`, set cursor to 'grab'.

**Step 2: Test in browser**

Upload test image, verify polygons, layers, zoom, hover.

**Step 3: Commit**

```bash
git add public/app.js
git commit -m "feat: add KonvaJS floor plan viewer with polygon overlays and layer toggles"
```

---

### Task 4: Manual smoke test and final polish

**Step 1: Start server and test all 4 images from test-data/**

**Step 2: Fix any visual issues (colors, opacities, layout)**

**Step 3: Final commit**

```bash
git add -A
git commit -m "fix: polish floorplan viewer"
```
