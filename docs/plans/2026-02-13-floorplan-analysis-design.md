# Floorplan Analysis Service — Design Document

## Overview

API service that accepts a floor plan image and returns structured JSON with polygon coordinates for all architectural elements (walls, doors, windows, rooms, kitchen, etc.) using Gemini 2.5 Flash vision model.

## Architecture

### Approach

Single-call to Gemini 2.5 Flash with structured output (JSON schema). Stateless request-response, no file storage.

### Data Flow

```
Client → POST /analyze (multipart image) → Fastify → Buffer → Gemini 2.5 Flash → JSON → Client
```

### File Structure

```
index.js              — Fastify server, routing
src/
  gemini.js           — Gemini API client wrapper
  prompt.js           — System prompt and JSON schema for structured output
  schema.js           — Fastify request/response validation schemas
```

### Dependencies

- `fastify` — HTTP framework
- `@google/genai` — Google AI SDK for Gemini
- `@fastify/multipart` — multipart file upload support

## API

### POST /analyze

**Request**: `multipart/form-data` with `image` field (JPEG/PNG, max 10MB)

**Response**: JSON with floor plan markup

**Error codes**:
- 400 — not an image / missing file
- 413 — file too large (>10MB)
- 502 — Gemini API error
- 504 — timeout (>60s)

## Output Schema

```json
{
  "version": 2,
  "width": "number (image width in pixels)",
  "height": "number (image height in pixels)",
  "pixels_per_meter": "number (scale from dimension lines)",

  "apartments": "Polygon — outer apartment boundary",
  "wall": "Polygon[] — wall contour polygons",
  "door": "Polygon[] — door rectangles",
  "window": "Polygon[] — window rectangles",
  "balcony_window": "Polygon | Polygon[] — balcony window boundary",
  "balcony": "Polygon | Polygon[] — balcony area",
  "bedroom": "Polygon | Polygon[] — bedroom(s)",
  "living_room": "Polygon — living room area",
  "other_room": "Polygon[] — bathrooms, corridors, WIC, etc.",
  "kitchen_table": "Polygon | Polygon[] — kitchen countertops",
  "kitchen_zone": "Polygon | null — kitchen zone",
  "sink": "Polygon — sink location",
  "cooker": "Polygon — cooker/stove location"
}
```

Where `Polygon = [[x, y], ...]` — array of [x, y] points in pixel coordinates. Polygons are closed (first point = last point).

## Gemini Configuration

- **Model**: `gemini-2.5-flash`
- **Prompts**: English language
- **Response format**: Structured output with JSON schema
- **API key**: via `GEMINI_API_KEY` environment variable

## Prompt Strategy

System prompt describes:
1. Task: analyze architectural floor plan image
2. Each output field and its meaning
3. Rules: pixel coordinates, closed polygons, first point = last point
4. How to determine `pixels_per_meter` from dimension lines on the drawing
5. How to classify rooms (bedroom vs other_room vs living_room)

## Test Data

4 manually annotated examples in `test-data/`:
- `U3laAgYJ.jpg` + `.json` — 2BR apartment with balcony
- `U0HuSArq.jpg` + `.json` — large apartment with long balcony
- `U3D6Agrz.jpg` + `.json` — 2BR with curved balcony
- `U36wysWv.jpg` + `.json` — 1BR with curved balcony
