# Floorplan Web Viewer — Design Document

## Overview

Simple web interface for uploading floor plan images, sending them to the analysis API, and visualizing the result as colored polygons overlaid on the original image using KonvaJS.

## Architecture

- **Approach**: Static HTML + vanilla JS + KonvaJS via CDN
- **Hosting**: Fastify serves `public/` via `@fastify/static`
- **No build step**, no framework

### Files

```
public/
  index.html    — HTML layout
  app.js        — upload logic, API call, KonvaJS rendering
  style.css     — styles
```

### New dependency

- `@fastify/static` — serve static files

### Data Flow

```
User selects file → JS sends POST /analyze (multipart) →
receives JSON → draws polygons over image on KonvaJS canvas
```

## UI Layout

- **Left panel** (~220px): upload button, layer checkboxes, status message
- **Main area**: KonvaJS Stage (image + polygon overlays), fills remaining space
- **Bottom bar**: hover info (element type name)

## Color Scheme

| Type | Color | Fill Opacity | Stroke Opacity |
|---|---|---|---|
| wall | #555555 | 0.3 | 0.8 |
| door | #FF6B35 | 0.4 | 0.8 |
| window | #00B4D8 | 0.4 | 0.8 |
| balcony_window | #48CAE4 | 0.3 | 0.8 |
| balcony | #90E0EF | 0.2 | 0.8 |
| bedroom | #7B2D8E | 0.25 | 0.8 |
| living_room | #2D6A4F | 0.25 | 0.8 |
| other_room | #E9C46A | 0.25 | 0.8 |
| kitchen_table | #E76F51 | 0.3 | 0.8 |
| kitchen_zone | #F4A261 | 0.2 | 0.8 |
| sink | #219EBC | 0.5 | 0.8 |
| cooker | #FB8500 | 0.5 | 0.8 |
| apartments | #264653 | 0.15 | 0.8 |

Stroke width: 2px for all polygons.

## Features

1. **Upload**: file input, sends to POST /analyze, shows loading spinner
2. **Canvas**: KonvaJS Stage with Image layer + one Layer per element type
3. **Zoom/Pan**: mouse wheel zoom, drag to pan
4. **Hover**: highlight polygon + show type name in status bar
5. **Layer toggles**: checkboxes to show/hide each element type
6. **Fit to screen**: auto-fit image to canvas on load
