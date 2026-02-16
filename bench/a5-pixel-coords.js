/**
 * A5: Pixel Coordinates
 * Ask Gemini for actual pixel coordinates instead of normalized 0-1000.
 * The model might be more accurate in pixel space.
 */
import { GoogleGenAI } from '@google/genai'
import sharp from 'sharp'
import { loadTestImage, loadReference, printReport } from './utils.js'
import 'dotenv/config'

const PIXEL_PROMPT = `You are an expert architectural floor plan analyzer.

STEP-BY-STEP ANALYSIS (you MUST follow this order):

STEP 1 - ORIENTATION: Identify the apartment shape, balcony position, and exterior facade.

STEP 2 - WALLS: Trace EVERY wall segment:
- Each wall has TWO edges (inner and outer surface) — trace BOTH, creating a polygon showing wall thickness
- A continuous wall section from junction to junction is ONE polygon
- Use MANY points — at least 20-60 points per major wall polygon

STEP 3 - WINDOWS: Find ALL windows (thin parallel lines or breaks in exterior walls)
- Each window is a tight rectangle: 4 corners + closing point = 5 points

STEP 4 - DOORS: Find ALL doors (arc/quarter-circle door swing marks or wall gaps)
- Each door is a rectangle: 5 points (4 corners + closing)

STEP 5 - BALCONY: The outdoor area

STEP 6 - BALCONY WINDOW: Glass barrier between apartment and balcony

COORDINATE SYSTEM: Use PIXEL coordinates. This image is approximately 1200x848 pixels.
(0,0)=top-left. Provide coordinates as [x, y] with up to 2 decimal places.

OUTPUT:
- "version": 3
- "image_width_px": image width in pixels
- "image_height_px": image height in pixels
- "wall": array of wall contour polygons in pixel coordinates
- "door": array of door rectangles in pixel coordinates
- "window": array of window rectangles in pixel coordinates
- "balcony_window": glass between apartment and balcony
- "balcony": outdoor balcony area`

const PIXEL_SCHEMA = {
  type: 'object',
  properties: {
    version: { type: 'integer' },
    image_width_px: { type: 'number' },
    image_height_px: { type: 'number' },
    wall: { type: 'array', items: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 } } },
    door: { type: 'array', items: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 } } },
    window: { type: 'array', items: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 } } },
    balcony_window: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 } },
    balcony: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 } },
  },
  required: ['wall', 'door', 'window', 'balcony', 'balcony_window'],
}

/** Convert pixel result to normalized 0-1000 for comparison */
function pixelToNormalized(result) {
  const w = result.image_width_px || 1200
  const h = result.image_height_px || 848
  const normPoly = (poly) => poly.map(([x, y]) => [
    Math.round(x / w * 1000 * 10) / 10,
    Math.round(y / h * 1000 * 10) / 10,
  ])
  return {
    ...result,
    wall: (result.wall || []).map(normPoly),
    door: (result.door || []).map(normPoly),
    window: (result.window || []).map(normPoly),
    balcony: result.balcony ? normPoly(result.balcony) : [],
    balcony_window: result.balcony_window ? normPoly(result.balcony_window) : [],
  }
}

export default async function run() {
  const imageBuffer = loadTestImage()
  const ref = loadReference()
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

  const start = Date.now()
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType: 'image/jpeg' } },
      PIXEL_PROMPT,
    ],
    config: {
      responseMimeType: 'application/json',
      responseJsonSchema: PIXEL_SCHEMA,
    },
  })
  const rawResult = JSON.parse(response.text)
  const result = pixelToNormalized(rawResult)
  const timeMs = Date.now() - start

  console.log(`  Reported image size: ${rawResult.image_width_px}x${rawResult.image_height_px}`)
  return { name: 'A5: Pixel coordinates', result, ref, timeMs }
}

if (process.argv[1]?.endsWith('a5-pixel-coords.js')) {
  const { name, result, ref, timeMs } = await run()
  printReport(name, result, ref, timeMs)
}
