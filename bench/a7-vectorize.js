/**
 * A7: Image Vectorization
 * Use Sharp to preprocess → convert to bitmap → trace contours algorithmically.
 * Then use Gemini only to classify which contours are walls/doors/windows.
 *
 * Pure algorithmic approach for coordinates, AI only for labeling.
 */
import { GoogleGenAI } from '@google/genai'
import sharp from 'sharp'
import { loadTestImage, loadReference, printReport } from './utils.js'
import { traceContours, simplifyContour, contourBBox } from './contour-utils.js'
import 'dotenv/config'

export default async function run() {
  const imageBuffer = loadTestImage()
  const ref = loadReference()
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })
  const meta = await sharp(imageBuffer).metadata()
  const imgW = meta.width, imgH = meta.height

  const start = Date.now()

  // Step 1: Preprocess image to binary
  const { data, info } = await sharp(imageBuffer)
    .grayscale()
    .normalize()
    .threshold(160)
    .raw()
    .toBuffer({ resolveWithObject: true })

  const binaryData = new Uint8Array(data.length)
  for (let i = 0; i < data.length; i++) {
    binaryData[i] = data[i] < 128 ? 0 : 255
  }

  console.log(`  Binary image: ${info.width}x${info.height}`)

  // Step 2: Trace contours
  const rawContours = traceContours(binaryData, info.width, info.height)
  console.log(`  Found ${rawContours.length} raw contours`)

  // Filter by size (ignore tiny noise and huge background)
  const imgArea = info.width * info.height
  const filtered = rawContours
    .map(c => ({ contour: c, bbox: contourBBox(c) }))
    .filter(c => c.bbox.area > imgArea * 0.001 && c.bbox.area < imgArea * 0.5)
    .sort((a, b) => b.bbox.area - a.bbox.area)
    .slice(0, 30) // top 30 by area

  console.log(`  Filtered to ${filtered.length} significant contours`)

  // Step 3: Convert contours to normalized coordinates and simplify
  const normalizedContours = filtered.map(({ contour, bbox }) => ({
    polygon: simplifyContour(contour, 40).map(([x, y]) => [
      Math.round(x / imgW * 1000 * 10) / 10,
      Math.round(y / imgH * 1000 * 10) / 10,
    ]),
    bbox: {
      x1: Math.round(bbox.minX / imgW * 1000),
      y1: Math.round(bbox.minY / imgH * 1000),
      x2: Math.round(bbox.maxX / imgW * 1000),
      y2: Math.round(bbox.maxY / imgH * 1000),
    },
    area: bbox.area,
  }))

  // Step 4: Use Gemini to classify contours
  const contoursDesc = normalizedContours.map((c, i) =>
    `Contour ${i}: bbox=[${c.bbox.x1},${c.bbox.y1},${c.bbox.x2},${c.bbox.y2}], ${c.polygon.length} points, area=${c.area}`
  ).join('\n')

  const classifyResp = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType: 'image/jpeg' } },
      `Here are contours extracted from this floor plan image:\n${contoursDesc}\n\nClassify each contour. Return JSON:\n{"classifications": [{"index": 0, "type": "wall"}, {"index": 1, "type": "door"}, ...]}\n\nTypes: "wall", "door", "window", "balcony", "balcony_window", "noise" (for irrelevant contours)`,
    ],
    config: {
      responseMimeType: 'application/json',
      responseJsonSchema: {
        type: 'object',
        properties: {
          classifications: { type: 'array', items: { type: 'object', properties: { index: { type: 'integer' }, type: { type: 'string' } }, required: ['index', 'type'] } },
        },
        required: ['classifications'],
      },
    },
  })

  const { classifications } = JSON.parse(classifyResp.text)
  console.log(`  Classifications: ${classifications.length}`)

  // Step 5: Build result
  const result = { version: 3, image_width_meters: 0, wall: [], door: [], window: [], balcony: [], balcony_window: [] }

  for (const cls of classifications) {
    const contour = normalizedContours[cls.index]
    if (!contour) continue
    if (cls.type === 'wall') result.wall.push(contour.polygon)
    else if (cls.type === 'door') result.door.push(contour.polygon)
    else if (cls.type === 'window') result.window.push(contour.polygon)
    else if (cls.type === 'balcony') result.balcony = contour.polygon
    else if (cls.type === 'balcony_window') result.balcony_window = contour.polygon
  }

  const timeMs = Date.now() - start
  return { name: 'A7: Vectorize + AI classify', result, ref, timeMs }
}

if (process.argv[1]?.endsWith('a7-vectorize.js')) {
  const { name, result, ref, timeMs } = await run()
  printReport(name, result, ref, timeMs)
}
