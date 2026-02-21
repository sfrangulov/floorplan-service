/**
 * A8: Grounding DINO + SAM / Gemini + Contour Tracing via Replicate API
 *
 * Strategy: Use AI for semantic detection (bboxes), then algorithmic contour
 * tracing on cropped regions for precise polygon extraction.
 *
 * SAM auto-segmentation proved ineffective for floor plans (segments line
 * fragments and room spaces, not architectural elements). Instead, we
 * preprocess crops to binary and use our contour tracer for pixel-perfect
 * polygon extraction.
 *
 * Variant A: Grounding DINO (Replicate) for bboxes → contour tracing per crop
 * Variant B: Gemini bbox detection → contour tracing per crop (recommended)
 * Variant S: Gemini bbox detection → SAM 2 (for comparison — shows why SAM fails)
 *
 * Usage:
 *   node bench/a8-grounded-sam.js a    # Variant A: DINO + contour
 *   node bench/a8-grounded-sam.js b    # Variant B: Gemini + contour
 *   node bench/a8-grounded-sam.js s    # Variant S: Gemini + SAM (reference)
 *   node bench/a8-grounded-sam.js      # Default: Variant B
 */
import Replicate from 'replicate'
import { GoogleGenAI } from '@google/genai'
import sharp from 'sharp'
import { loadTestImage, loadReference, printReport } from './utils.js'
import { traceContours, simplifyContour, contourBBox } from './contour-utils.js'
import 'dotenv/config'

const SAM_VERSION = 'fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83'

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

// ── Contour extraction from cropped region ─────────────────────────

/**
 * Extract polygon from a cropped image region using contour tracing.
 * 1. Preprocess to binary
 * 2. Trace contours
 * 3. Pick the largest contour (main element)
 * 4. Map coordinates back to global 0-1000 space
 */
async function extractPolygonFromCrop(imageBuffer, bbox, imgW, imgH, targetPoints) {
  const padPx = 15 // pixel padding around bbox
  const x1 = Math.max(0, Math.round(bbox[0] / 1000 * imgW) - padPx)
  const y1 = Math.max(0, Math.round(bbox[1] / 1000 * imgH) - padPx)
  const x2 = Math.min(imgW, Math.round(bbox[2] / 1000 * imgW) + padPx)
  const y2 = Math.min(imgH, Math.round(bbox[3] / 1000 * imgH) + padPx)
  const cropW = x2 - x1, cropH = y2 - y1

  if (cropW < 3 || cropH < 3) return null

  // Preprocess crop to binary
  const { data, info } = await sharp(imageBuffer)
    .extract({ left: x1, top: y1, width: cropW, height: cropH })
    .grayscale()
    .normalize()
    .threshold(160)
    .raw()
    .toBuffer({ resolveWithObject: true })

  // Convert to binary (black = drawn lines)
  const binaryData = new Uint8Array(data.length)
  for (let i = 0; i < data.length; i++) {
    binaryData[i] = data[i] < 128 ? 0 : 255
  }

  // Trace contours
  const contours = traceContours(binaryData, info.width, info.height)
  if (contours.length === 0) return null

  // Filter by area (skip tiny noise)
  const cropArea = info.width * info.height
  const significant = contours
    .map(c => ({ contour: c, bbox: contourBBox(c) }))
    .filter(c => c.bbox.area > cropArea * 0.01) // at least 1% of crop
    .sort((a, b) => b.bbox.area - a.bbox.area)

  if (significant.length === 0) return null

  // Take largest contour
  const best = significant[0].contour
  const simplified = simplifyContour(best, targetPoints)

  // Map crop-local pixel coords back to global 0-1000 space
  return simplified.map(([px, py]) => [
    Math.round((x1 + px) / imgW * 1000 * 10) / 10,
    Math.round((y1 + py) / imgH * 1000 * 10) / 10,
  ])
}

// ── Helpers ────────────────────────────────────────────────────────

function padBbox(bbox, padPct = 0.1) {
  const w = bbox[2] - bbox[0], h = bbox[3] - bbox[1]
  const px = Math.round(w * padPct), py = Math.round(h * padPct)
  return [
    Math.max(0, bbox[0] - px), Math.max(0, bbox[1] - py),
    Math.min(1000, bbox[2] + px), Math.min(1000, bbox[3] + py),
  ]
}

function buildResult(matched) {
  const result = { version: 3, image_width_meters: 0, wall: [], door: [], window: [], balcony: [], balcony_window: [] }
  for (const { type, polygon } of matched) {
    if (!polygon) continue
    if (type === 'wall') result.wall.push(polygon)
    else if (type === 'door') result.door.push(polygon)
    else if (type === 'window') result.window.push(polygon)
    else if (type === 'balcony') result.balcony = polygon
    else if (type === 'balcony_window') result.balcony_window = polygon
  }
  return result
}

// ── Variant A: Grounding DINO → contour tracing ───────────────────

async function variantA(imageBuffer, imgW, imgH) {
  const replicate = new Replicate()
  const imageUri = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`

  console.log('  [A] Running Grounding DINO...')
  const dinoOutput = await replicate.run('adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa', {
    input: {
      image: imageUri,
      query: 'wall . door . window . balcony',
      box_threshold: 0.25,
      text_threshold: 0.25,
    },
  })

  console.log('  [A] DINO raw output:', JSON.stringify(dinoOutput).slice(0, 500))

  const rawDets = dinoOutput?.detections || (Array.isArray(dinoOutput) ? dinoOutput : [])
  const elements = []
  for (const det of rawDets) {
    const label = (det.label || det.class || 'unknown').toLowerCase()
    const box = det.box || det.bbox
    if (!box) continue
    const type = label.includes('wall') ? 'wall' : label.includes('door') ? 'door' :
                 label.includes('window') ? 'window' : label.includes('balcony') ? 'balcony' : 'unknown'
    if (type === 'unknown') continue
    // DINO returns pixel coords → normalize to 0-1000
    elements.push({
      type,
      bbox: padBbox([
        Math.round(box[0] / imgW * 1000),
        Math.round(box[1] / imgH * 1000),
        Math.round(box[2] / imgW * 1000),
        Math.round(box[3] / imgH * 1000),
      ]),
      targetPoints: type === 'wall' ? 60 : 5,
    })
  }

  console.log(`  [A] DINO: ${elements.length} elements (${elements.filter(e => e.type === 'wall').length} walls, ${elements.filter(e => e.type === 'door').length} doors, ${elements.filter(e => e.type === 'window').length} windows)`)

  // Extract polygons from crops
  const matched = []
  for (const el of elements) {
    const polygon = await extractPolygonFromCrop(imageBuffer, el.bbox, imgW, imgH, el.targetPoints)
    matched.push({ type: el.type, polygon })
  }

  return buildResult(matched)
}

// ── Variant B: Gemini → contour tracing (recommended) ─────────────

async function variantB(imageBuffer, imgW, imgH) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

  // Step 1: Gemini detects bounding boxes
  console.log('  [B] Running Gemini bbox detection...')
  const detectResp = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType: 'image/jpeg' } },
      DETECT_PROMPT,
    ],
    config: { responseMimeType: 'application/json', responseJsonSchema: DETECT_SCHEMA },
  })
  const detected = JSON.parse(detectResp.text)
  console.log(`  [B] Gemini: ${detected.walls.length} walls, ${detected.doors.length} doors, ${detected.windows.length} windows`)

  // Build element list
  const elements = []
  for (const wall of detected.walls) elements.push({ type: 'wall', bbox: padBbox(wall.bbox), targetPoints: 60 })
  for (const door of detected.doors) elements.push({ type: 'door', bbox: padBbox(door.bbox), targetPoints: 5 })
  for (const win of detected.windows) elements.push({ type: 'window', bbox: padBbox(win.bbox), targetPoints: 5 })
  if (detected.balcony?.bbox) elements.push({ type: 'balcony', bbox: padBbox(detected.balcony.bbox), targetPoints: 10 })
  if (detected.balcony_window?.bbox) elements.push({ type: 'balcony_window', bbox: padBbox(detected.balcony_window.bbox), targetPoints: 10 })

  console.log(`  [B] Processing ${elements.length} elements with contour tracing...`)

  // Step 2: Extract polygons from each crop (all local, no API calls)
  const matched = []
  for (const el of elements) {
    const polygon = await extractPolygonFromCrop(imageBuffer, el.bbox, imgW, imgH, el.targetPoints)
    matched.push({ type: el.type, polygon })
    if (polygon) {
      const xs = polygon.map(p => p[0]), ys = polygon.map(p => p[1])
      console.log(`    ${el.type}: ${polygon.length} pts, bbox [${Math.min(...xs)|0},${Math.min(...ys)|0},${Math.max(...xs)|0},${Math.max(...ys)|0}]`)
    } else {
      console.log(`    ${el.type}: no contour found`)
    }
  }

  return buildResult(matched)
}

// ── Variant S: Gemini → SAM 2 (for comparison) ────────────────────

async function variantS(imageBuffer, imgW, imgH) {
  const replicate = new Replicate()
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })
  const imageUri = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`

  // Step 1: Gemini detects bounding boxes
  console.log('  [S] Running Gemini bbox detection...')
  const detectResp = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType: 'image/jpeg' } },
      DETECT_PROMPT,
    ],
    config: { responseMimeType: 'application/json', responseJsonSchema: DETECT_SCHEMA },
  })
  const detected = JSON.parse(detectResp.text)
  console.log(`  [S] Gemini: ${detected.walls.length} walls, ${detected.doors.length} doors, ${detected.windows.length} windows`)

  // Step 2: SAM on full image (1 API call)
  console.log('  [S] Running SAM 2 on full image...')
  const samOutput = await replicate.run(`meta/sam-2:${SAM_VERSION}`, {
    input: {
      image: imageUri,
      points_per_side: 32,
      pred_iou_thresh: 0.7,
      stability_score_thresh: 0.85,
      use_m2m: true,
    },
  })

  console.log(`  [S] SAM returned ${samOutput.individual_masks.length} masks`)

  // Read all masks
  async function readStream(stream) {
    const chunks = []
    for await (const chunk of stream) chunks.push(chunk)
    return Buffer.concat(chunks)
  }

  const masks = []
  for (let i = 0; i < samOutput.individual_masks.length; i++) {
    const buf = await readStream(samOutput.individual_masks[i])
    const { data, info } = await sharp(buf).grayscale().raw().toBuffer({ resolveWithObject: true })

    // Compute mask bbox in normalized coords
    let minX = info.width, minY = info.height, maxX = 0, maxY = 0, count = 0
    for (let y = 0; y < info.height; y++) {
      for (let x = 0; x < info.width; x++) {
        if (data[y * info.width + x] > 128) {
          if (x < minX) minX = x; if (y < minY) minY = y
          if (x > maxX) maxX = x; if (y > maxY) maxY = y
          count++
        }
      }
    }
    if (count === 0) continue
    const pct = count / (info.width * info.height)
    if (pct < 0.001 || pct > 0.4) continue

    masks.push({
      bbox: [
        Math.round(minX / imgW * 1000), Math.round(minY / imgH * 1000),
        Math.round(maxX / imgW * 1000), Math.round(maxY / imgH * 1000),
      ],
      pixelData: data, width: info.width, height: info.height,
      area: count, pct,
    })
  }

  console.log(`  [S] ${masks.length} significant masks`)

  // Step 3: Match masks to Gemini elements by IoU
  function bboxIoU(a, b) {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1])
    const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3])
    if (x2 <= x1 || y2 <= y1) return 0
    const inter = (x2 - x1) * (y2 - y1)
    const areaA = (a[2] - a[0]) * (a[3] - a[1])
    const areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)
  }

  const elements = []
  for (const wall of detected.walls) elements.push({ type: 'wall', bbox: padBbox(wall.bbox), targetPoints: 60 })
  for (const door of detected.doors) elements.push({ type: 'door', bbox: padBbox(door.bbox), targetPoints: 5 })
  for (const win of detected.windows) elements.push({ type: 'window', bbox: padBbox(win.bbox), targetPoints: 5 })
  if (detected.balcony?.bbox) elements.push({ type: 'balcony', bbox: padBbox(detected.balcony.bbox), targetPoints: 10 })
  if (detected.balcony_window?.bbox) elements.push({ type: 'balcony_window', bbox: padBbox(detected.balcony_window.bbox), targetPoints: 10 })

  const matched = []
  for (const el of elements) {
    const candidates = masks
      .map(m => ({ mask: m, iou: bboxIoU(el.bbox, m.bbox) }))
      .filter(c => c.iou > 0.05)
      .sort((a, b) => b.iou - a.iou)

    if (candidates.length === 0) {
      matched.push({ type: el.type, polygon: null })
      continue
    }

    const best = candidates[0].mask
    // Invert for contour tracer
    const inverted = new Uint8Array(best.pixelData.length)
    for (let i = 0; i < best.pixelData.length; i++) {
      inverted[i] = best.pixelData[i] < 128 ? 255 : 0
    }
    const contours = traceContours(inverted, best.width, best.height)
    if (contours.length === 0) { matched.push({ type: el.type, polygon: null }); continue }
    const largest = contours.reduce((a, b) => a.length > b.length ? a : b)
    const simplified = simplifyContour(largest, el.targetPoints)
    const polygon = simplified.map(([x, y]) => [
      Math.round(x / imgW * 1000 * 10) / 10,
      Math.round(y / imgH * 1000 * 10) / 10,
    ])
    matched.push({ type: el.type, polygon })
  }

  return buildResult(matched)
}

// ── Main export ────────────────────────────────────────────────────

export default async function run(variant = 'b') {
  const imageBuffer = loadTestImage()
  const ref = loadReference()
  const meta = await sharp(imageBuffer).metadata()
  const imgW = meta.width, imgH = meta.height

  const start = Date.now()
  let result, name

  if (variant === 'a') {
    name = 'A8a: DINO + contour tracing'
    result = await variantA(imageBuffer, imgW, imgH)
  } else if (variant === 's') {
    name = 'A8s: Gemini + SAM (reference)'
    result = await variantS(imageBuffer, imgW, imgH)
  } else {
    name = 'A8b: Gemini bbox + contour'
    result = await variantB(imageBuffer, imgW, imgH)
  }

  const timeMs = Date.now() - start
  console.log(`  Walls: ${result.wall.length}, Doors: ${result.door.length}, Windows: ${result.window.length}`)
  console.log(`  Balcony: ${Array.isArray(result.balcony) && result.balcony.length > 0 ? 'yes' : 'no'}, Balcony window: ${Array.isArray(result.balcony_window) && result.balcony_window.length > 0 ? 'yes' : 'no'}`)

  return { name, result, ref, timeMs }
}

if (process.argv[1]?.endsWith('a8-grounded-sam.js')) {
  const variant = process.argv[2] || 'b'
  const { name, result, ref, timeMs } = await run(variant)
  printReport(name, result, ref, timeMs)

  // Save result
  const fs = await import('fs')
  const outPath = `bench/a8-result.json`
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2))
  console.log(`\nSaved to ${outPath}`)
}
