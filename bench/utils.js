import fs from 'fs'
import path from 'path'

const TEST_IMAGE = 'U3laAgYJ.jpg'
const REF_WIDTH = 1200
const REF_HEIGHT = 848

export function loadTestImage() {
  return fs.readFileSync(path.join(process.cwd(), 'test-data', TEST_IMAGE))
}

export function loadReference() {
  return JSON.parse(fs.readFileSync(path.join(process.cwd(), 'test-data', `${TEST_IMAGE}.json`), 'utf-8'))
}

/** Normalize reference pixel coords to 0-1000 */
export function normalizeReference(ref) {
  const normPoly = (poly) => poly.map(p => [
    Math.round(p[0] / REF_WIDTH * 1000 * 10) / 10,
    Math.round(p[1] / REF_HEIGHT * 1000 * 10) / 10,
  ])
  return {
    wall: ref.wall.map(normPoly),
    door: ref.door.map(normPoly),
    window: ref.window.map(normPoly),
    balcony: [normPoly(ref.balcony)],
    balcony_window: [normPoly(ref.balcony_window)],
  }
}

/** Compute IoU for two bounding boxes */
function bboxFromPoly(poly) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of poly) {
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
  }
  return { minX, minY, maxX, maxY }
}

function bboxIoU(a, b) {
  const x1 = Math.max(a.minX, b.minX)
  const y1 = Math.max(a.minY, b.minY)
  const x2 = Math.min(a.maxX, b.maxX)
  const y2 = Math.min(a.maxY, b.maxY)
  if (x2 <= x1 || y2 <= y1) return 0
  const intersection = (x2 - x1) * (y2 - y1)
  const areaA = (a.maxX - a.minX) * (a.maxY - a.minY)
  const areaB = (b.maxX - b.minX) * (b.maxY - b.minY)
  return intersection / (areaA + areaB - intersection)
}

/** Match result polygons to reference polygons using IoU */
function matchPolygons(resultPolys, refPolys) {
  let matched = 0
  const used = new Set()
  for (const rPoly of resultPolys) {
    const rBbox = bboxFromPoly(rPoly)
    let bestIoU = 0, bestIdx = -1
    for (let i = 0; i < refPolys.length; i++) {
      if (used.has(i)) continue
      const iou = bboxIoU(rBbox, bboxFromPoly(refPolys[i]))
      if (iou > bestIoU) { bestIoU = iou; bestIdx = i }
    }
    if (bestIoU > 0.3) { matched++; used.add(bestIdx) }
  }
  return { matched, total: refPolys.length, extra: resultPolys.length - matched }
}

/** Full comparison report */
export function compareWithReference(result, ref) {
  const normRef = normalizeReference(ref)
  const report = {}

  for (const key of ['wall', 'door', 'window']) {
    const rPolys = result[key] || []
    const refPolys = normRef[key] || []
    const match = matchPolygons(rPolys, refPolys)
    const rPoints = rPolys.reduce((s, p) => s + p.length, 0)
    const refPoints = refPolys.reduce((s, p) => s + p.length, 0)
    report[key] = {
      count: `${rPolys.length}/${refPolys.length}`,
      matched: `${match.matched}/${match.total}`,
      extra: match.extra,
      points: `${rPoints}/${refPoints}`,
    }
  }

  // balcony and balcony_window - single polygon
  for (const key of ['balcony', 'balcony_window']) {
    const val = result[key]
    const hasIt = val && val.length > 0
    const pts = hasIt ? (Array.isArray(val[0]?.[0]) ? val[0].length : val.length) : 0
    const refPts = normRef[key]?.[0]?.length || 0
    report[key] = { present: hasIt ? 'yes' : 'NO', points: `${pts}/${refPts}` }
  }

  return report
}

export function printReport(name, result, ref, timeMs) {
  const report = compareWithReference(result, ref)
  console.log(`\n${'='.repeat(60)}`)
  console.log(`  ${name}  (${(timeMs / 1000).toFixed(1)}s)`)
  console.log(`${'='.repeat(60)}`)
  console.log(`  Element       | Count    | Matched  | Extra | Points`)
  console.log(`  --------------|----------|----------|-------|--------`)
  for (const key of ['wall', 'door', 'window']) {
    const d = report[key]
    console.log(`  ${key.padEnd(14)}| ${d.count.padEnd(9)}| ${d.matched.padEnd(9)}| ${String(d.extra).padEnd(6)}| ${d.points}`)
  }
  for (const key of ['balcony', 'balcony_window']) {
    const d = report[key]
    console.log(`  ${key.padEnd(14)}| ${d.present.padEnd(9)}| -        | -     | ${d.points}`)
  }
  return report
}
