# Floorplan Analysis Service — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Fastify API that accepts a floor plan image and returns structured JSON with polygon coordinates for walls, rooms, doors, windows, and kitchen elements, using Gemini 2.5 Flash.

**Architecture:** Single `POST /analyze` endpoint receives an image via multipart/form-data, converts it to base64, sends it to Gemini 2.5 Flash with a system prompt and JSON schema for structured output, and returns the parsed JSON.

**Tech Stack:** Fastify, @fastify/multipart, @google/genai, node:test (built-in)

---

### Task 1: Install dependencies

**Files:**
- Modify: `package.json`

**Step 1: Install packages**

Run:
```bash
npm install @google/genai @fastify/multipart
```

**Step 2: Verify installation**

Run:
```bash
node -e "import('@google/genai').then(() => console.log('ok'))"
```
Expected: `ok`

**Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "feat: add @google/genai and @fastify/multipart dependencies"
```

---

### Task 2: Create the Gemini prompt and schema module

**Files:**
- Create: `src/prompt.js`

**Step 1: Write the test**

Create `test/prompt.test.js`:

```javascript
import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA } from '../src/prompt.js'

describe('prompt', () => {
  it('exports a non-empty system prompt string', () => {
    assert.equal(typeof SYSTEM_PROMPT, 'string')
    assert.ok(SYSTEM_PROMPT.length > 100)
  })

  it('exports a valid JSON schema object with required fields', () => {
    assert.equal(RESPONSE_JSON_SCHEMA.type, 'object')
    const props = Object.keys(RESPONSE_JSON_SCHEMA.properties)
    const required = [
      'version', 'width', 'height', 'pixels_per_meter',
      'apartments', 'wall', 'door', 'window', 'bedroom',
      'living_room', 'other_room', 'balcony'
    ]
    for (const key of required) {
      assert.ok(props.includes(key), `missing property: ${key}`)
    }
  })
})
```

**Step 2: Run test to verify it fails**

Run: `node --test test/prompt.test.js`
Expected: FAIL — module not found

**Step 3: Write the prompt module**

Create `src/prompt.js`:

```javascript
export const SYSTEM_PROMPT = `You are an expert architectural floor plan analyzer. You receive an image of an apartment floor plan and must extract precise polygon coordinates for every architectural element.

TASK:
Analyze the floor plan image and return structured JSON with polygon coordinates for all elements.

COORDINATE SYSTEM:
- All coordinates are in pixels relative to the image (origin = top-left corner)
- X axis = horizontal (left to right), Y axis = vertical (top to bottom)
- Provide coordinates as [x, y] pairs with up to 2 decimal places

POLYGONS:
- Each polygon is an array of [x, y] points tracing the element boundary
- Polygons MUST be closed: the last point must equal the first point
- Trace polygons clockwise
- Use enough points to accurately represent the shape (rectangles need 5 points, curved shapes need more)

FIELDS TO EXTRACT:

1. "version" — always set to 2
2. "width" — image width in pixels
3. "height" — image height in pixels
4. "pixels_per_meter" — calculate from dimension lines shown on the plan. Find a labeled dimension (e.g. "3.40" meters), measure its length in pixels, then divide pixels by meters.
5. "apartments" — single polygon tracing the outer boundary of the entire apartment (including all walls)
6. "wall" — array of polygons for wall contours. Each wall segment is a polygon tracing the wall outline. Include door/window cutouts in the wall polygons.
7. "door" — array of rectangular polygons marking door positions within walls
8. "window" — array of rectangular polygons marking window positions within walls
9. "balcony_window" — polygon(s) for windows/glass doors between apartment and balcony
10. "balcony" — polygon(s) for the balcony area outside the apartment
11. "bedroom" — polygon(s) for bedrooms (labeled as "BR", "MASTER BR", "Bedroom", or rooms with beds)
12. "living_room" — polygon for the living/dining room area (labeled as "LIVING", "DINING", or the main open area)
13. "other_room" — array of polygons for other rooms: bathrooms, WC, corridors, walk-in closets, laundry, powder rooms
14. "kitchen_table" — polygon(s) for kitchen countertops/work surfaces
15. "kitchen_zone" — polygon for the kitchen area if it's a separate zone, or null if kitchen is part of living room
16. "sink" — polygon for kitchen sink location
17. "cooker" — polygon for cooker/stove location

ROOM CLASSIFICATION RULES:
- Rooms with beds or labeled "BR"/"Bedroom"/"Master" → bedroom
- The largest open area or rooms labeled "Living"/"Dining" → living_room
- Bathrooms, WC, shower rooms, labeled "BATH"/"MBATH"/"PDR" → other_room
- Walk-in closets "WIC", laundry "L/S", corridors → other_room
- Kitchen area with appliances → kitchen_table/kitchen_zone/sink/cooker

IMPORTANT:
- Be precise with wall coordinates — walls have thickness, trace both inner and outer edges
- Doors are typically rectangular cutouts in walls
- Windows are typically rectangular elements on exterior walls
- If a field has multiple instances (e.g. 2 bedrooms), return an array of polygons
- If a field has one instance, it can be a single polygon (not wrapped in array)
- If kitchen_zone is not a separate room (open kitchen), set it to null`

export const RESPONSE_JSON_SCHEMA = {
  type: 'object',
  properties: {
    version: { type: 'integer', description: 'Schema version, always 2' },
    width: { type: 'number', description: 'Image width in pixels' },
    height: { type: 'number', description: 'Image height in pixels' },
    pixels_per_meter: { type: 'number', description: 'Scale: pixels per meter' },
    apartments: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Outer boundary polygon of the apartment'
    },
    wall: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of wall contour polygons'
    },
    door: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of door rectangle polygons'
    },
    window: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of window rectangle polygons'
    },
    balcony_window: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony window boundary polygon'
    },
    balcony: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony area polygon'
    },
    bedroom: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of bedroom polygons'
    },
    living_room: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Living room polygon'
    },
    other_room: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of other room polygons (bathrooms, corridors, etc.)'
    },
    kitchen_table: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Kitchen countertop polygons'
    },
    kitchen_zone: {
      type: ['array', 'null'],
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Kitchen zone polygon or null if open kitchen'
    },
    sink: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Sink location polygon'
    },
    cooker: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Cooker/stove location polygon'
    }
  },
  required: [
    'version', 'width', 'height', 'pixels_per_meter',
    'apartments', 'wall', 'door', 'window',
    'balcony_window', 'balcony', 'bedroom', 'living_room',
    'other_room', 'kitchen_table', 'sink', 'cooker'
  ]
}
```

**Step 4: Run test to verify it passes**

Run: `node --test test/prompt.test.js`
Expected: PASS

**Step 5: Commit**

```bash
git add src/prompt.js test/prompt.test.js
git commit -m "feat: add Gemini system prompt and JSON schema for floor plan analysis"
```

---

### Task 3: Create the Gemini client module

**Files:**
- Create: `src/gemini.js`

**Step 1: Write the test**

Create `test/gemini.test.js`:

```javascript
import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { analyzeFloorplan } from '../src/gemini.js'

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
```

**Step 2: Run test to verify it fails**

Run: `node --test test/gemini.test.js`
Expected: FAIL — module not found

**Step 3: Write the Gemini client**

Create `src/gemini.js`:

```javascript
import { GoogleGenAI } from '@google/genai'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA } from './prompt.js'

/**
 * @param {Buffer} imageBuffer - raw image bytes
 * @param {string} mimeType - e.g. 'image/jpeg'
 * @returns {Promise<object>} parsed floor plan JSON
 */
export async function analyzeFloorplan(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required')
  }

  const ai = new GoogleGenAI({ apiKey })

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
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
```

**Step 4: Run test to verify it passes**

Run: `node --test test/gemini.test.js`
Expected: PASS

**Step 5: Commit**

```bash
git add src/gemini.js test/gemini.test.js
git commit -m "feat: add Gemini client wrapper for floor plan analysis"
```

---

### Task 4: Wire up the POST /analyze endpoint

**Files:**
- Modify: `index.js`

**Step 1: Write the integration test**

Create `test/api.test.js`:

```javascript
import { describe, it, before, after } from 'node:test'
import assert from 'node:assert/strict'
import { buildApp } from '../index.js'

describe('POST /analyze', () => {
  let app

  before(async () => {
    app = await buildApp()
  })

  after(async () => {
    await app.close()
  })

  it('returns 400 when no file is uploaded', async () => {
    const response = await app.inject({
      method: 'POST',
      url: '/analyze',
      headers: { 'content-type': 'application/json' },
      payload: {},
    })
    assert.equal(response.statusCode, 400)
  })

  it('returns 400 when file is not an image', async () => {
    const form = new FormData()
    form.append('image', new Blob(['hello'], { type: 'text/plain' }), 'test.txt')

    // Use inject with multipart
    const response = await app.inject({
      method: 'POST',
      url: '/analyze',
      payload: form,
      headers: form.headers,
    })
    // Accept 400 or 415 — both indicate rejection of non-image
    assert.ok([400, 415].includes(response.statusCode))
  })
})
```

**Step 2: Run test to verify it fails**

Run: `node --test test/api.test.js`
Expected: FAIL — buildApp not exported

**Step 3: Refactor index.js to export buildApp and wire up the endpoint**

Rewrite `index.js`:

```javascript
import Fastify from 'fastify'
import multipart from '@fastify/multipart'
import { analyzeFloorplan } from './src/gemini.js'

export async function buildApp(opts = {}) {
  const fastify = Fastify({ logger: opts.logger ?? true })

  await fastify.register(multipart, {
    attachFieldsToBody: 'keyValues',
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  })

  fastify.get('/', async () => {
    return { status: 'ok', service: 'floorplan-service' }
  })

  fastify.post('/analyze', async (request, reply) => {
    const file = request.body?.image
    if (!file || !Buffer.isBuffer(file)) {
      return reply.code(400).send({ error: 'Missing "image" file field' })
    }

    // Detect mime type from the multipart data
    const mimeType = request.body.image?.mimetype || 'image/jpeg'
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp']
    if (!allowedTypes.includes(mimeType)) {
      return reply.code(400).send({ error: `Unsupported image type: ${mimeType}. Allowed: ${allowedTypes.join(', ')}` })
    }

    try {
      const result = await analyzeFloorplan(file, mimeType)
      return result
    } catch (err) {
      request.log.error(err, 'Gemini API error')
      return reply.code(502).send({ error: 'Floor plan analysis failed', details: err.message })
    }
  })

  return fastify
}

// Start server if run directly
const isMain = !process.argv[1] || import.meta.url === `file://${process.argv[1]}`
if (isMain) {
  const app = await buildApp()
  try {
    await app.listen({ port: process.env.PORT || 3000 })
  } catch (err) {
    app.log.error(err)
    process.exit(1)
  }
}
```

Note: The mime type detection with `attachFieldsToBody: 'keyValues'` may need adjustment after testing. With `'keyValues'` mode, `req.body.image` is a Buffer, but we lose the mimetype metadata. We may need to switch to `attachFieldsToBody: true` to get the full file object with mimetype. Adjust during implementation as needed.

**Step 4: Run test to verify it passes**

Run: `node --test test/api.test.js`
Expected: PASS (at least the 400 test)

**Step 5: Commit**

```bash
git add index.js test/api.test.js
git commit -m "feat: add POST /analyze endpoint with multipart upload"
```

---

### Task 5: End-to-end test with real test data

**Files:**
- Create: `test/e2e.test.js`

**Step 1: Write the E2E test**

This test requires `GEMINI_API_KEY` to be set. It sends a real floor plan image to the endpoint and validates the response structure.

Create `test/e2e.test.js`:

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

  it('analyzes U3laAgYJ.jpg and returns valid structure', async () => {
    if (!process.env.GEMINI_API_KEY) return

    const imagePath = path.join(process.cwd(), 'test-data', 'U3laAgYJ.jpg')
    const imageBuffer = fs.readFileSync(imagePath)

    // Build a multipart body manually for inject
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
    assert.equal(result.version, 2)
    assert.equal(typeof result.width, 'number')
    assert.equal(typeof result.height, 'number')
    assert.equal(typeof result.pixels_per_meter, 'number')
    assert.ok(Array.isArray(result.apartments))
    assert.ok(Array.isArray(result.wall))
    assert.ok(Array.isArray(result.door))
    assert.ok(Array.isArray(result.window))
    assert.ok(Array.isArray(result.bedroom))
    assert.ok(Array.isArray(result.other_room))

    // Validate polygon structure: apartments should be array of [x,y] pairs
    assert.ok(result.apartments.length > 3, 'apartments should have at least 3 points')
    assert.equal(result.apartments[0].length, 2, 'each point should be [x, y]')
  }, { timeout: 120000 }) // 2 min timeout for Gemini API
})
```

**Step 2: Run E2E test**

Run: `GEMINI_API_KEY=your-key-here node --test test/e2e.test.js`
Expected: PASS (with valid API key)

**Step 3: Commit**

```bash
git add test/e2e.test.js
git commit -m "test: add E2E test for floor plan analysis with real image"
```

---

### Task 6: Add npm scripts and .env support

**Files:**
- Modify: `package.json`
- Create: `.env.example`
- Modify: `.gitignore` (create if not exists)

**Step 1: Update package.json scripts**

```json
{
  "scripts": {
    "start": "node index.js",
    "dev": "node --watch index.js",
    "test": "node --test test/prompt.test.js test/gemini.test.js test/api.test.js",
    "test:e2e": "node --test test/e2e.test.js"
  }
}
```

**Step 2: Create .env.example**

```
GEMINI_API_KEY=your-google-ai-api-key-here
PORT=3000
```

**Step 3: Create .gitignore**

```
node_modules/
.env
```

**Step 4: Commit**

```bash
git add package.json .env.example .gitignore
git commit -m "chore: add npm scripts, .env.example, and .gitignore"
```

---

### Task 7: Manual smoke test

**Step 1: Create .env file with real API key**

```bash
cp .env.example .env
# Edit .env with real GEMINI_API_KEY
```

**Step 2: Start the server**

Run: `node -e "import('dotenv').catch(()=>{});import('fs').then(f=>f.readFileSync('.env','utf8').split('\n').filter(l=>l&&!l.startsWith('#')).forEach(l=>{const[k,...v]=l.split('=');process.env[k]=v.join('=')}))" && node index.js`

Or simpler — just export the key and start:
```bash
export GEMINI_API_KEY=your-key
node index.js
```

**Step 3: Test with curl**

```bash
curl -X POST http://localhost:3000/analyze \
  -F "image=@test-data/U3laAgYJ.jpg" \
  | jq .
```

Expected: JSON response with version, width, height, rooms, walls, etc.

**Step 4: Compare output structure with expected**

Visually compare the returned JSON fields against `test-data/U3laAgYJ.jpg.json`. Exact coordinate match is not expected, but all fields should be present with reasonable polygon data.
