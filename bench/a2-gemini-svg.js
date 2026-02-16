/**
 * A2: Gemini SVG Generation
 * Ask Gemini to produce SVG path commands for each element.
 * LLMs are trained on lots of SVG data - may be more accurate than raw coords.
 */
import { GoogleGenAI } from '@google/genai'
import { loadTestImage, loadReference, printReport } from './utils.js'
import 'dotenv/config'

const SVG_PROMPT = `You are an expert architectural floor plan analyzer.

Analyze this floor plan image and output SVG path data for each structural element.

COORDINATE SYSTEM: Use a 1000x1000 viewBox. (0,0)=top-left, (1000,1000)=bottom-right.
X and Y normalized INDEPENDENTLY to the image dimensions.

For WALLS: trace BOTH inner and outer edges creating a closed polygon showing wall thickness.
Use many points (20-60+ per major wall). Output as SVG path "d" attribute using M, L, Z commands.

For DOORS and WINDOWS: rectangles (M x1,y1 L x2,y2 L x3,y3 L x4,y4 Z).

Output JSON with this exact structure:
{
  "wall": ["M 222.5,699.5 L 344.3,699.6 L 344.6,830.3 ... Z", ...],
  "door": ["M x1,y1 L x2,y2 L x3,y3 L x4,y4 Z", ...],
  "window": ["M x1,y1 L x2,y2 L x3,y3 L x4,y4 Z", ...],
  "balcony": "M x1,y1 L x2,y2 ... Z",
  "balcony_window": "M x1,y1 L x2,y2 ... Z"
}`

const SVG_SCHEMA = {
  type: 'object',
  properties: {
    wall: { type: 'array', items: { type: 'string' }, description: 'SVG path d-attributes for wall polygons' },
    door: { type: 'array', items: { type: 'string' }, description: 'SVG path d-attributes for door rectangles' },
    window: { type: 'array', items: { type: 'string' }, description: 'SVG path d-attributes for window rectangles' },
    balcony: { type: 'string', description: 'SVG path d-attribute for balcony polygon' },
    balcony_window: { type: 'string', description: 'SVG path d-attribute for balcony window polygon' },
  },
  required: ['wall', 'door', 'window', 'balcony', 'balcony_window'],
}

/** Parse SVG path d-attribute to array of [x,y] points */
function parseSvgPath(d) {
  if (!d || typeof d !== 'string') return []
  const points = []
  // Match M/L commands followed by coordinates
  const commands = d.match(/[ML]\s*[\d.]+[\s,]+[\d.]+/gi) || []
  for (const cmd of commands) {
    const nums = cmd.match(/[\d.]+/g)
    if (nums && nums.length >= 2) {
      points.push([parseFloat(nums[0]), parseFloat(nums[1])])
    }
  }
  return points
}

/** Convert SVG result to standard polygon format */
function svgToPolygons(svgResult) {
  const result = { version: 3, image_width_meters: 0 }

  for (const key of ['wall', 'door', 'window']) {
    const paths = svgResult[key] || []
    result[key] = paths.map(d => parseSvgPath(d)).filter(p => p.length >= 3)
  }

  result.balcony = parseSvgPath(svgResult.balcony)
  result.balcony_window = parseSvgPath(svgResult.balcony_window)

  return result
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
      SVG_PROMPT,
    ],
    config: {
      responseMimeType: 'application/json',
      responseJsonSchema: SVG_SCHEMA,
    },
  })
  const svgResult = JSON.parse(response.text)
  const result = svgToPolygons(svgResult)
  const timeMs = Date.now() - start

  return { name: 'A2: Gemini SVG paths', result, ref, timeMs }
}

if (process.argv[1]?.endsWith('a2-gemini-svg.js')) {
  const { name, result, ref, timeMs } = await run()
  printReport(name, result, ref, timeMs)
  console.log('\nRaw SVG sample (wall[0]):', JSON.parse((await import('@google/genai')).default ? '' : '{}'))
}
