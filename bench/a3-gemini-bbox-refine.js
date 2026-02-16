/**
 * A3: Gemini Bounding Boxes → Crop → Refine
 * Pass 1: Detect elements with bounding boxes
 * Pass 2: For each wall bbox, crop region and ask for detailed polygon
 */
import { GoogleGenAI } from '@google/genai'
import sharp from 'sharp'
import { loadTestImage, loadReference, printReport } from './utils.js'
import 'dotenv/config'

const DETECT_PROMPT = `You are an expert floor plan analyzer.

Find ALL structural elements in this floor plan and return their bounding boxes.
Coordinates: normalized 0-1000. (0,0)=top-left.

Return JSON:
{
  "walls": [{"bbox": [x1,y1,x2,y2], "label": "outer wall left"}, ...],
  "doors": [{"bbox": [x1,y1,x2,y2], "label": "bedroom door"}, ...],
  "windows": [{"bbox": [x1,y1,x2,y2], "label": "top window 1"}, ...],
  "balcony": {"bbox": [x1,y1,x2,y2]},
  "balcony_window": {"bbox": [x1,y1,x2,y2]}
}

Be thorough - find ALL walls (including short segments), ALL doors, ALL windows.`

const DETECT_SCHEMA = {
  type: 'object',
  properties: {
    walls: { type: 'array', items: { type: 'object', properties: { bbox: { type: 'array', items: { type: 'number' } }, label: { type: 'string' } }, required: ['bbox'] } },
    doors: { type: 'array', items: { type: 'object', properties: { bbox: { type: 'array', items: { type: 'number' } }, label: { type: 'string' } }, required: ['bbox'] } },
    windows: { type: 'array', items: { type: 'object', properties: { bbox: { type: 'array', items: { type: 'number' } }, label: { type: 'string' } }, required: ['bbox'] } },
    balcony: { type: 'object', properties: { bbox: { type: 'array', items: { type: 'number' } } }, required: ['bbox'] },
    balcony_window: { type: 'object', properties: { bbox: { type: 'array', items: { type: 'number' } } }, required: ['bbox'] },
  },
  required: ['walls', 'doors', 'windows', 'balcony', 'balcony_window'],
}

const REFINE_PROMPT = `Trace the PRECISE polygon contour of the highlighted structural element in this cropped floor plan region.

For WALLS: trace BOTH inner and outer edges, creating a closed polygon showing wall thickness. Use 20-60 points.
For DOORS/WINDOWS: trace the rectangle precisely. 5 points (4 corners + closing).
For BALCONY: trace the outer boundary.

Coordinates: normalized 0-1000 within THIS cropped region. (0,0)=top-left of crop.

Return JSON: {"polygon": [[x1,y1], [x2,y2], ...]}`

const REFINE_SCHEMA = {
  type: 'object',
  properties: {
    polygon: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 } },
  },
  required: ['polygon'],
}

/** Crop image region and return buffer */
async function cropRegion(imageBuffer, bbox, imgWidth, imgHeight) {
  const pad = 20 // pixels padding
  const x1 = Math.max(0, Math.round(bbox[0] / 1000 * imgWidth) - pad)
  const y1 = Math.max(0, Math.round(bbox[1] / 1000 * imgHeight) - pad)
  const x2 = Math.min(imgWidth, Math.round(bbox[2] / 1000 * imgWidth) + pad)
  const y2 = Math.min(imgHeight, Math.round(bbox[3] / 1000 * imgHeight) + pad)

  return sharp(imageBuffer)
    .extract({ left: x1, top: y1, width: x2 - x1, height: y2 - y1 })
    .jpeg()
    .toBuffer()
}

/** Convert local polygon coords back to global 0-1000 */
function localToGlobal(polygon, bbox) {
  const bw = bbox[2] - bbox[0]
  const bh = bbox[3] - bbox[1]
  return polygon.map(([x, y]) => [
    Math.round((bbox[0] + x / 1000 * bw) * 10) / 10,
    Math.round((bbox[1] + y / 1000 * bh) * 10) / 10,
  ])
}

export default async function run() {
  const imageBuffer = loadTestImage()
  const ref = loadReference()
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })
  const metadata = await sharp(imageBuffer).metadata()
  const imgWidth = metadata.width
  const imgHeight = metadata.height

  const start = Date.now()

  // Pass 1: Detect bounding boxes
  const detectResp = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType: 'image/jpeg' } },
      DETECT_PROMPT,
    ],
    config: { responseMimeType: 'application/json', responseJsonSchema: DETECT_SCHEMA },
  })
  const detected = JSON.parse(detectResp.text)
  console.log(`  Pass 1: ${detected.walls.length} walls, ${detected.doors.length} doors, ${detected.windows.length} windows`)

  // Pass 2: Refine each element (limit to avoid too many API calls)
  const result = { version: 3, image_width_meters: 0, wall: [], door: [], window: [], balcony: [], balcony_window: [] }

  // Refine walls (up to 10)
  for (const wall of detected.walls.slice(0, 10)) {
    try {
      const cropped = await cropRegion(imageBuffer, wall.bbox, imgWidth, imgHeight)
      const resp = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [
          { inlineData: { data: cropped.toString('base64'), mimeType: 'image/jpeg' } },
          'This cropped region contains a WALL segment. ' + REFINE_PROMPT,
        ],
        config: { responseMimeType: 'application/json', responseJsonSchema: REFINE_SCHEMA },
      })
      const { polygon } = JSON.parse(resp.text)
      if (polygon?.length >= 3) result.wall.push(localToGlobal(polygon, wall.bbox))
    } catch (e) {
      console.log(`  Warn: wall refine failed: ${e.message}`)
    }
  }

  // Refine doors
  for (const door of detected.doors.slice(0, 10)) {
    try {
      const cropped = await cropRegion(imageBuffer, door.bbox, imgWidth, imgHeight)
      const resp = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [
          { inlineData: { data: cropped.toString('base64'), mimeType: 'image/jpeg' } },
          'This cropped region contains a DOOR. ' + REFINE_PROMPT,
        ],
        config: { responseMimeType: 'application/json', responseJsonSchema: REFINE_SCHEMA },
      })
      const { polygon } = JSON.parse(resp.text)
      if (polygon?.length >= 3) result.door.push(localToGlobal(polygon, door.bbox))
    } catch (e) {
      console.log(`  Warn: door refine failed: ${e.message}`)
    }
  }

  // Refine windows
  for (const win of detected.windows.slice(0, 10)) {
    try {
      const cropped = await cropRegion(imageBuffer, win.bbox, imgWidth, imgHeight)
      const resp = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [
          { inlineData: { data: cropped.toString('base64'), mimeType: 'image/jpeg' } },
          'This cropped region contains a WINDOW. ' + REFINE_PROMPT,
        ],
        config: { responseMimeType: 'application/json', responseJsonSchema: REFINE_SCHEMA },
      })
      const { polygon } = JSON.parse(resp.text)
      if (polygon?.length >= 3) result.window.push(localToGlobal(polygon, win.bbox))
    } catch (e) {
      console.log(`  Warn: window refine failed: ${e.message}`)
    }
  }

  // Balcony + balcony_window - just use bbox as polygon
  if (detected.balcony?.bbox) {
    const b = detected.balcony.bbox
    result.balcony = [[b[0],b[1]], [b[2],b[1]], [b[2],b[3]], [b[0],b[3]], [b[0],b[1]]]
  }
  if (detected.balcony_window?.bbox) {
    const b = detected.balcony_window.bbox
    result.balcony_window = [[b[0],b[1]], [b[2],b[1]], [b[2],b[3]], [b[0],b[3]], [b[0],b[1]]]
  }

  const timeMs = Date.now() - start
  return { name: 'A3: BBox detect → Crop → Refine', result, ref, timeMs }
}

if (process.argv[1]?.endsWith('a3-gemini-bbox-refine.js')) {
  const { name, result, ref, timeMs } = await run()
  printReport(name, result, ref, timeMs)
}
