/**
 * Detailed coordinate comparison between an approach result and reference.
 * Computes: bbox overlap, center distance, Hausdorff distance, coordinate-level accuracy.
 */
import fs from 'fs'
import { loadReference } from './utils.js'

const REF_WIDTH = 1200, REF_HEIGHT = 848

function normRef(poly) {
  return poly.map(p => [
    Math.round(p[0] / REF_WIDTH * 1000 * 10) / 10,
    Math.round(p[1] / REF_HEIGHT * 1000 * 10) / 10,
  ])
}

function bbox(poly) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of poly) {
    if (x < minX) minX = x; if (y < minY) minY = y
    if (x > maxX) maxX = x; if (y > maxY) maxY = y
  }
  return { minX, minY, maxX, maxY }
}

function center(poly) {
  const b = bbox(poly)
  return [(b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2]
}

function dist(a, b) {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
}

/** Min distance from point to polygon point set */
function minDistToSet(point, set) {
  let min = Infinity
  for (const p of set) {
    const d = dist(point, p)
    if (d < min) min = d
  }
  return min
}

/** Hausdorff distance between two point sets */
function hausdorff(setA, setB) {
  let maxMinA = 0
  for (const p of setA) {
    const d = minDistToSet(p, setB)
    if (d > maxMinA) maxMinA = d
  }
  let maxMinB = 0
  for (const p of setB) {
    const d = minDistToSet(p, setA)
    if (d > maxMinB) maxMinB = d
  }
  return Math.max(maxMinA, maxMinB)
}

/** Average min distance from A to B */
function avgMinDist(setA, setB) {
  let sum = 0
  for (const p of setA) {
    sum += minDistToSet(p, setB)
  }
  return sum / setA.length
}

/** Match result polygons to reference using center distance */
function matchByCenterDist(resultPolys, refPolys, maxDist = 100) {
  const matches = []
  const usedRef = new Set()

  for (let i = 0; i < resultPolys.length; i++) {
    const rc = center(resultPolys[i])
    let bestDist = Infinity, bestJ = -1
    for (let j = 0; j < refPolys.length; j++) {
      if (usedRef.has(j)) continue
      const d = dist(rc, center(refPolys[j]))
      if (d < bestDist) { bestDist = d; bestJ = j }
    }
    if (bestJ >= 0 && bestDist < maxDist) {
      matches.push({ ri: i, refi: bestJ, centerDist: bestDist })
      usedRef.add(bestJ)
    }
  }
  return matches
}

function analyzeElement(name, resultPolys, refPolys) {
  console.log(`\n--- ${name.toUpperCase()} ---`)
  console.log(`  Result: ${resultPolys.length} polygons, Ref: ${refPolys.length} polygons`)

  const matches = matchByCenterDist(resultPolys, refPolys)
  console.log(`  Matched: ${matches.length}/${refPolys.length}`)

  for (const m of matches) {
    const rPoly = resultPolys[m.ri]
    const refPoly = refPolys[m.refi]
    const h = hausdorff(rPoly, refPoly)
    const avgR2Ref = avgMinDist(rPoly, refPoly)
    const avgRef2R = avgMinDist(refPoly, rPoly)

    const rb = bbox(rPoly)
    const refb = bbox(refPoly)

    console.log(`\n  Match: result[${m.ri}] ↔ ref[${m.refi}]`)
    console.log(`    Center dist:     ${m.centerDist.toFixed(1)} (in 0-1000 space)`)
    console.log(`    Hausdorff dist:  ${h.toFixed(1)}`)
    console.log(`    Avg dist R→Ref:  ${avgR2Ref.toFixed(1)}`)
    console.log(`    Avg dist Ref→R:  ${avgRef2R.toFixed(1)}`)
    console.log(`    Points: result=${rPoly.length}, ref=${refPoly.length}`)
    console.log(`    Result bbox:  [${rb.minX.toFixed(0)},${rb.minY.toFixed(0)}]-[${rb.maxX.toFixed(0)},${rb.maxY.toFixed(0)}]`)
    console.log(`    Ref bbox:     [${refb.minX.toFixed(0)},${refb.minY.toFixed(0)}]-[${refb.maxX.toFixed(0)},${refb.maxY.toFixed(0)}]`)
  }

  // Unmatched result polygons
  const matchedRI = new Set(matches.map(m => m.ri))
  const unmatched = resultPolys.filter((_, i) => !matchedRI.has(i))
  if (unmatched.length > 0) {
    console.log(`\n  Unmatched result polygons: ${unmatched.length}`)
    for (const p of unmatched) {
      const b = bbox(p)
      const c = center(p)
      console.log(`    bbox=[${b.minX.toFixed(0)},${b.minY.toFixed(0)}]-[${b.maxX.toFixed(0)},${b.maxY.toFixed(0)}], center=[${c[0].toFixed(0)},${c[1].toFixed(0)}], ${p.length}pts`)
    }
  }

  // Unmatched ref polygons
  const matchedRefI = new Set(matches.map(m => m.refi))
  const unmatchedRef = refPolys.filter((_, i) => !matchedRefI.has(i))
  if (unmatchedRef.length > 0) {
    console.log(`\n  Missed reference polygons: ${unmatchedRef.length}`)
    for (const p of unmatchedRef) {
      const b = bbox(p)
      const c = center(p)
      console.log(`    bbox=[${b.minX.toFixed(0)},${b.minY.toFixed(0)}]-[${b.maxX.toFixed(0)},${b.maxY.toFixed(0)}], center=[${c[0].toFixed(0)},${c[1].toFixed(0)}], ${p.length}pts`)
    }
  }

  return matches
}

// Load result from file
const resultPath = process.argv[2]
if (!resultPath) {
  console.error('Usage: node bench/compare-coords.js <result.json>')
  process.exit(1)
}

const result = JSON.parse(fs.readFileSync(resultPath, 'utf-8'))
const ref = loadReference()

// Normalize reference
const normRefData = {
  wall: ref.wall.map(normRef),
  door: ref.door.map(normRef),
  window: ref.window.map(normRef),
}

console.log('='.repeat(60))
console.log('  DETAILED COORDINATE COMPARISON')
console.log('  Distances in normalized 0-1000 space')
console.log('  (1 unit ≈ 0.1% of image dimension)')
console.log('='.repeat(60))

let allMatches = []
for (const key of ['wall', 'door', 'window']) {
  const matches = analyzeElement(key, result[key] || [], normRefData[key] || [])
  allMatches.push(...matches.map(m => ({
    ...m,
    hausdorff: hausdorff(result[key][m.ri], normRefData[key][m.refi]),
    avgDist: (avgMinDist(result[key][m.ri], normRefData[key][m.refi]) + avgMinDist(normRefData[key][m.refi], result[key][m.ri])) / 2,
  })))
}

console.log('\n' + '='.repeat(60))
console.log('  OVERALL ACCURACY SUMMARY')
console.log('='.repeat(60))
if (allMatches.length > 0) {
  const avgCenterDist = allMatches.reduce((s, m) => s + m.centerDist, 0) / allMatches.length
  const avgHausdorff = allMatches.reduce((s, m) => s + m.hausdorff, 0) / allMatches.length
  const avgAvgDist = allMatches.reduce((s, m) => s + m.avgDist, 0) / allMatches.length
  console.log(`  Matched elements:    ${allMatches.length}`)
  console.log(`  Avg center dist:     ${avgCenterDist.toFixed(1)} (lower=better, <20 is good)`)
  console.log(`  Avg Hausdorff dist:  ${avgHausdorff.toFixed(1)} (lower=better, <30 is good)`)
  console.log(`  Avg avg-point dist:  ${avgAvgDist.toFixed(1)} (lower=better, <15 is good)`)
  console.log(`\n  Scale: 10 units ≈ 1% of image, 50 units ≈ 5%`)
} else {
  console.log('  No matches found!')
}
