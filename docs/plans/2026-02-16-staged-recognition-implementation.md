# Staged Recognition Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-pass Gemini recognition with a 3-stage pipeline (Sharp preprocessing → Gemini Image cleaning → Gemini coordinate extraction) and reduce elements from 15 to 5.

**Architecture:** Image flows through Sharp for binarization, then Gemini Image model removes non-structural elements, then Gemini Vision extracts polygon coordinates for 5 element types only (wall, door, window, balcony, balcony_window).

**Tech Stack:** Node.js, Sharp, @google/genai (Gemini 2.5 Flash Image + Gemini 3 Flash Preview), Fastify

---

### Task 1: Add Sharp dependency

**Files:**
- Modify: `package.json`

**Step 1: Install sharp**

Run: `npm install sharp`

**Step 2: Verify installation**

Run: `node -e "import('sharp').then(s => console.log('sharp OK'))"`
Expected: `sharp OK`

**Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "chore: add sharp dependency for image preprocessing"
```

---

### Task 2: Update prompt.js — reduce to 5 elements

**Files:**
- Modify: `src/prompt.js`

**Step 1: Write the failing test**

Update `test/prompt.test.js` to test for the new 5-element schema:

```javascript
import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA, CLEANING_PROMPT } from '../src/prompt.js'

describe('prompt', () => {
  it('exports a non-empty system prompt string', () => {
    assert.equal(typeof SYSTEM_PROMPT, 'string')
    assert.ok(SYSTEM_PROMPT.length > 100)
  })

  it('uses normalized 0-1000 coordinate system', () => {
    assert.ok(SYSTEM_PROMPT.includes('0 to 1000'), 'prompt should mention 0-1000 range')
    assert.ok(SYSTEM_PROMPT.includes('NORMALIZED'), 'prompt should mention normalized coordinates')
  })

  it('exports a valid JSON schema with 5 active element fields', () => {
    assert.equal(RESPONSE_JSON_SCHEMA.type, 'object')
    const props = Object.keys(RESPONSE_JSON_SCHEMA.properties)
    const activeFields = [
      'version', 'image_width_meters',
      'wall', 'door', 'window',
      'balcony_window', 'balcony'
    ]
    for (const key of activeFields) {
      assert.ok(props.includes(key), `missing property: ${key}`)
    }
  })

  it('does not include commented-out elements in schema', () => {
    const props = Object.keys(RESPONSE_JSON_SCHEMA.properties)
    const removedFields = [
      'apartments', 'bedroom', 'living_room', 'other_room',
      'kitchen_table', 'kitchen_zone', 'sink', 'cooker'
    ]
    for (const key of removedFields) {
      assert.ok(!props.includes(key), `should not have property: ${key}`)
    }
  })

  it('requires only active fields', () => {
    const required = RESPONSE_JSON_SCHEMA.required
    assert.ok(required.includes('wall'))
    assert.ok(required.includes('door'))
    assert.ok(required.includes('window'))
    assert.ok(required.includes('balcony'))
    assert.ok(required.includes('balcony_window'))
    assert.ok(!required.includes('apartments'))
    assert.ok(!required.includes('bedroom'))
  })

  it('exports a cleaning prompt string', () => {
    assert.equal(typeof CLEANING_PROMPT, 'string')
    assert.ok(CLEANING_PROMPT.length > 50)
  })
})
```

**Step 2: Run test to verify it fails**

Run: `node --test test/prompt.test.js`
Expected: FAIL — `CLEANING_PROMPT` not exported, old schema still has all 15 fields

**Step 3: Update src/prompt.js**

Replace the full content of `src/prompt.js` with:

```javascript
export const CLEANING_PROMPT = `Clean this floor plan image. Remove ALL of the following:
- Furniture (beds, sofas, tables, chairs, appliances)
- Text labels and room names
- Dimension lines and measurements
- Decorative elements and hatching
- Floor patterns and tiles

KEEP ONLY:
- Wall outlines (preserve exact thickness and position)
- Door openings and door arcs
- Window marks
- Balcony boundaries and balcony window/door marks

Output a clean black-and-white architectural line drawing with ONLY structural elements.
The wall geometry must remain EXACTLY in its original position — do not shift, scale, or distort any lines.`

export const SYSTEM_PROMPT = `You are an expert architectural floor plan analyzer. You receive a cleaned floor plan image (containing only walls, doors, windows, and balcony) and must extract precise polygon coordinates for each structural element.

TASK:
Analyze the floor plan image and return structured JSON with polygon coordinates for structural elements only.

COORDINATE SYSTEM:
- Use NORMALIZED coordinates from 0 to 1000
- (0, 0) = top-left corner of the image
- (1000, 1000) = bottom-right corner of the image
- X axis and Y axis are normalized INDEPENDENTLY (both range from 0 to 1000 regardless of aspect ratio)
- Provide coordinates as [x, y] pairs with up to 1 decimal place

POLYGONS:
- Each polygon is an array of [x, y] points tracing the element boundary
- Polygons MUST be closed: the last point must equal the first point
- Trace polygons clockwise
- Wall polygons must trace both inner and outer edges, forming the wall thickness
- Use enough points to accurately represent the shape (straight walls: vertices at corners; curved shapes: 10+ points along the curve)

FIELDS TO EXTRACT:

1. "version" — always set to 3
2. "image_width_meters" — estimate the physical width of the entire image in meters. Use the dimension labels visible on the plan to calculate this.
3. "wall" — array of polygons for wall contours. Each wall segment is a polygon tracing the wall outline (both inner and outer edges forming the wall thickness).
4. "door" — array of rectangular polygons marking door positions within walls
5. "window" — array of rectangular polygons marking window positions within walls
6. "balcony_window" — polygon(s) for windows/glass doors between apartment and balcony
7. "balcony" — polygon(s) for the balcony area outside the apartment

IMPORTANT:
- ALL coordinates must be in the 0-1000 normalized range
- Walls have thickness — trace both the inner and outer edges
- If a field has multiple instances (e.g. 3 windows), return an array of polygons
- If a field has one instance, it can be a single polygon (not wrapped in array)`

// Commented out elements — to be restored in future phases:
// 8. "apartments" — single polygon tracing the outer boundary of the entire apartment
// 9. "bedroom" — polygon(s) for bedrooms
// 10. "living_room" — polygon for the living/dining room area
// 11. "other_room" — array of polygons for other rooms (bathrooms, WC, corridors, etc.)
// 12. "kitchen_table" — polygon(s) for kitchen countertops
// 13. "kitchen_zone" — polygon for the kitchen area (null if open kitchen)
// 14. "sink" — polygon for kitchen sink location
// 15. "cooker" — polygon for cooker/stove location

export const RESPONSE_JSON_SCHEMA = {
  type: 'object',
  properties: {
    version: { type: 'integer', description: 'Schema version, always 3' },
    image_width_meters: { type: 'number', description: 'Physical width of the entire image in meters' },
    wall: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of wall contour polygons in normalized 0-1000 coordinates'
    },
    door: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of door rectangle polygons in normalized 0-1000 coordinates'
    },
    window: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of window rectangle polygons in normalized 0-1000 coordinates'
    },
    balcony_window: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony window boundary polygon in normalized 0-1000 coordinates'
    },
    balcony: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony area polygon in normalized 0-1000 coordinates'
    },
    // Commented out — future phases:
    // apartments: { ... },
    // bedroom: { ... },
    // living_room: { ... },
    // other_room: { ... },
    // kitchen_table: { ... },
    // kitchen_zone: { ... },
    // sink: { ... },
    // cooker: { ... },
  },
  required: [
    'version', 'image_width_meters',
    'wall', 'door', 'window',
    'balcony_window', 'balcony'
  ]
}
```

**Step 4: Run test to verify it passes**

Run: `node --test test/prompt.test.js`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/prompt.js test/prompt.test.js
git commit -m "feat: reduce recognition to 5 structural elements, add cleaning prompt"
```

---

### Task 3: Update gemini.js — 3-stage pipeline

**Files:**
- Modify: `src/gemini.js`

**Step 1: Write the failing test**

Update `test/gemini.test.js`:

```javascript
import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { analyzeFloorplan, preprocessImage, cleanFloorplan } from '../src/gemini.js'

describe('analyzeFloorplan', () => {
  it('is an async function', () => {
    assert.equal(typeof analyzeFloorplan, 'function')
  })

  it('rejects without GEMINI_API_KEY', async () => {
    const original = process.env.GEMINI_API_KEY
    delete process.env.GEMINI_API_KEY
    try {
      await assert.rejects(
        () => analyzeFloorplan(Buffer.from('fake'), 'image/jpeg'),
        { message: /GEMINI_API_KEY/ }
      )
    } finally {
      if (original) process.env.GEMINI_API_KEY = original
    }
  })
})

describe('preprocessImage', () => {
  it('is an async function', () => {
    assert.equal(typeof preprocessImage, 'function')
  })
})

describe('cleanFloorplan', () => {
  it('is an async function', () => {
    assert.equal(typeof cleanFloorplan, 'function')
  })

  it('rejects without GEMINI_API_KEY', async () => {
    const original = process.env.GEMINI_API_KEY
    delete process.env.GEMINI_API_KEY
    try {
      await assert.rejects(
        () => cleanFloorplan(Buffer.from('fake'), 'image/png'),
        { message: /GEMINI_API_KEY/ }
      )
    } finally {
      if (original) process.env.GEMINI_API_KEY = original
    }
  })
})
```

**Step 2: Run test to verify it fails**

Run: `node --test test/gemini.test.js`
Expected: FAIL — `preprocessImage` and `cleanFloorplan` not exported

**Step 3: Implement src/gemini.js with 3-stage pipeline**

Replace `src/gemini.js` with:

```javascript
import { GoogleGenAI } from '@google/genai'
import sharp from 'sharp'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA, CLEANING_PROMPT } from './prompt.js'

/**
 * Stage 1: Sharp preprocessing — grayscale + threshold + contrast
 * @param {Buffer} imageBuffer - raw image bytes
 * @returns {Promise<Buffer>} preprocessed PNG buffer
 */
export async function preprocessImage(imageBuffer) {
  return sharp(imageBuffer)
    .greyscale()
    .normalize()
    .sharpen()
    .png()
    .toBuffer()
}

/**
 * Stage 2: Gemini Image cleaning — remove furniture, text, dimensions
 * @param {Buffer} imageBuffer - preprocessed image bytes
 * @param {string} mimeType - e.g. 'image/png'
 * @returns {Promise<Buffer>} cleaned image buffer
 */
export async function cleanFloorplan(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required')
  }

  const ai = new GoogleGenAI({ apiKey })

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash-preview-05-20',
    contents: [
      { text: CLEANING_PROMPT },
      { inlineData: { data: imageBuffer.toString('base64'), mimeType } },
    ],
    config: {
      responseModalities: ['image', 'text'],
    },
  })

  // Extract image from response parts
  for (const part of response.candidates[0].content.parts) {
    if (part.inlineData) {
      return Buffer.from(part.inlineData.data, 'base64')
    }
  }

  // If no image returned, fall back to original
  console.warn('Gemini Image did not return an image, using original')
  return imageBuffer
}

/**
 * Stage 3: Gemini coordinate extraction from cleaned image
 * @param {Buffer} imageBuffer - cleaned image bytes
 * @param {string} mimeType - e.g. 'image/png'
 * @returns {Promise<object>} parsed floor plan JSON
 */
async function extractCoordinates(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  const ai = new GoogleGenAI({ apiKey })

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType } },
      SYSTEM_PROMPT,
    ],
    config: {
      responseMimeType: 'application/json',
      responseJsonSchema: RESPONSE_JSON_SCHEMA,
    },
  })

  return JSON.parse(response.text)
}

/**
 * Full pipeline: preprocess → clean → extract
 * @param {Buffer} imageBuffer - raw image bytes
 * @param {string} mimeType - e.g. 'image/jpeg'
 * @returns {Promise<object>} parsed floor plan JSON
 */
export async function analyzeFloorplan(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required')
  }

  // Stage 1: Sharp preprocessing
  const preprocessed = await preprocessImage(imageBuffer)

  // Stage 2: Gemini Image cleaning
  const cleaned = await cleanFloorplan(preprocessed, 'image/png')

  // Stage 3: Coordinate extraction from cleaned image
  return extractCoordinates(cleaned, 'image/png')
}
```

**Step 4: Run test to verify it passes**

Run: `node --test test/gemini.test.js`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/gemini.js test/gemini.test.js
git commit -m "feat: implement 3-stage recognition pipeline (Sharp + Gemini Image + Gemini Vision)"
```

---

### Task 4: Update app.js — reduce to 5 layer types

**Files:**
- Modify: `public/app.js`

**Step 1: Update LAYER_CONFIG**

Replace the LAYER_CONFIG block (lines 1-17) in `public/app.js` with:

```javascript
// --- Layer config ---
// Layers are rendered in this order (first = bottom, last = top)
const LAYER_CONFIG = {
  balcony:        { color: '#90E0EF', opacity: 0.3,  strokeWidth: 1, label: 'Balconies' },
  wall:           { color: '#888888', opacity: 0.5,  strokeWidth: 1, label: 'Walls' },
  door:           { color: '#FF6B35', opacity: 0.5,  strokeWidth: 1, label: 'Doors' },
  window:         { color: '#00B4D8', opacity: 0.5,  strokeWidth: 1, label: 'Windows' },
  balcony_window: { color: '#48CAE4', opacity: 0.4,  strokeWidth: 1, label: 'Balcony Windows' },
  // Commented out — future phases:
  // apartments:     { color: '#264653', opacity: 0.2,  strokeWidth: 2, dash: [8, 4], label: 'Apartments' },
  // bedroom:        { color: '#7B2D8E', opacity: 0.35, strokeWidth: 1, label: 'Bedrooms' },
  // living_room:    { color: '#2D6A4F', opacity: 0.35, strokeWidth: 1, label: 'Living Rooms' },
  // other_room:     { color: '#E9C46A', opacity: 0.35, strokeWidth: 1, label: 'Other Rooms' },
  // kitchen_zone:   { color: '#F4A261', opacity: 0.3,  strokeWidth: 1, label: 'Kitchen Zones' },
  // kitchen_table:  { color: '#E76F51', opacity: 0.5,  strokeWidth: 1, label: 'Kitchen Tables' },
  // sink:           { color: '#219EBC', opacity: 0.6,  strokeWidth: 1, label: 'Sinks' },
  // cooker:         { color: '#FB8500', opacity: 0.6,  strokeWidth: 1, label: 'Cookers' },
}
```

**Step 2: Verify visually**

Run: `npm run dev` and open http://localhost:3000
Expected: Sidebar shows only 5 layer toggles (Balconies, Walls, Doors, Windows, Balcony Windows)

**Step 3: Commit**

```bash
git add public/app.js
git commit -m "feat: reduce viewer to 5 structural element layers"
```

---

### Task 5: Update E2E test for new schema

**Files:**
- Modify: `test/e2e.test.js`

**Step 1: Update E2E test assertions**

Replace `test/e2e.test.js` with:

```javascript
import { describe, it, before, after } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { buildApp } from '../index.js'

describe('E2E: POST /analyze with real image', () => {
  let app

  before(async () => {
    if (!process.env.GEMINI_API_KEY) {
      console.log('Skipping E2E tests: GEMINI_API_KEY not set')
      return
    }
    app = await buildApp({ logger: false })
  })

  after(async () => {
    if (app) await app.close()
  })

  it('analyzes U3laAgYJ.jpg and returns valid v3 structure with 5 elements', { timeout: 180_000 }, async () => {
    if (!process.env.GEMINI_API_KEY) return

    const imagePath = path.join(process.cwd(), 'test-data', 'U3laAgYJ.jpg')
    const imageBuffer = fs.readFileSync(imagePath)

    const boundary = '----TestBoundary'
    const body = Buffer.concat([
      Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="image"; filename="U3laAgYJ.jpg"\r\nContent-Type: image/jpeg\r\n\r\n`),
      imageBuffer,
      Buffer.from(`\r\n--${boundary}--\r\n`),
    ])

    const response = await app.inject({
      method: 'POST',
      url: '/analyze',
      headers: {
        'content-type': `multipart/form-data; boundary=${boundary}`,
      },
      payload: body,
    })

    assert.equal(response.statusCode, 200)

    const result = JSON.parse(response.body)
    assert.equal(result.version, 3)
    assert.equal(typeof result.image_width_meters, 'number')
    assert.ok(result.image_width_meters > 0, 'image_width_meters should be positive')

    // Active elements should be present
    assert.ok(Array.isArray(result.wall), 'wall should be array')
    assert.ok(Array.isArray(result.door), 'door should be array')
    assert.ok(Array.isArray(result.window), 'window should be array')
    assert.ok(Array.isArray(result.balcony), 'balcony should be array')

    // Validate wall polygon structure
    assert.ok(result.wall.length > 0, 'should detect at least 1 wall')
    assert.ok(result.wall[0].length > 2, 'wall polygon should have at least 3 points')
    assert.equal(result.wall[0][0].length, 2, 'each point should be [x, y]')

    // Validate coordinates are in 0-1000 range
    for (const wallPolygon of result.wall) {
      for (const point of wallPolygon) {
        assert.ok(point[0] >= 0 && point[0] <= 1000, `x=${point[0]} should be 0-1000`)
        assert.ok(point[1] >= 0 && point[1] <= 1000, `y=${point[1]} should be 0-1000`)
      }
    }

    // Removed elements should NOT be present
    assert.equal(result.apartments, undefined, 'apartments should not be in response')
    assert.equal(result.bedroom, undefined, 'bedroom should not be in response')
    assert.equal(result.living_room, undefined, 'living_room should not be in response')

    console.log(`Result: image_width=${result.image_width_meters}m`)
    console.log(`Elements: ${result.wall.length} walls, ${result.door.length} doors, ${result.window.length} windows`)
  })
})
```

**Step 2: Run unit tests to make sure nothing is broken**

Run: `node --test test/prompt.test.js test/gemini.test.js test/api.test.js`
Expected: All PASS

**Step 3: Commit**

```bash
git add test/e2e.test.js
git commit -m "test: update E2E test for 5-element schema and 3-stage pipeline"
```

---

### Task 6: Run E2E test and validate

**Step 1: Run E2E test**

Run: `node --test test/e2e.test.js`
Expected: PASS (timeout 180s to account for 2 Gemini calls)

**Step 2: Manual test via browser**

Run: `npm run dev`
- Open http://localhost:3000
- Upload a test image from `test-data/`
- Verify: only 5 layers shown, polygons render correctly
- Check console for any errors

**Step 3: If E2E fails — debug**

Common issues:
- Gemini Image model name changed → check https://ai.google.dev/gemini-api/docs/models
- Image not returned from cleaning step → check fallback path works
- Coordinates out of range → check prompt clarity

---

### Task 7: Final commit and cleanup

**Step 1: Run all tests**

Run: `npm test`
Expected: All unit tests PASS

**Step 2: Verify git status is clean**

Run: `git status`
Expected: Nothing to commit, working tree clean

**Step 3: Summary of changes**

Files modified:
- `package.json` — added `sharp`
- `src/prompt.js` — reduced to 5 elements, added CLEANING_PROMPT
- `src/gemini.js` — 3-stage pipeline (preprocessImage → cleanFloorplan → extractCoordinates)
- `public/app.js` — reduced LAYER_CONFIG to 5 entries
- `test/prompt.test.js` — updated for new schema
- `test/gemini.test.js` — added preprocessImage and cleanFloorplan tests
- `test/e2e.test.js` — updated for 5-element response
