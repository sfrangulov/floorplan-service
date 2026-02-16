/**
 * Run all benchmark approaches and compare results.
 * Usage: node bench/run-all.js [a1 a2 a3 ...]
 */
import { printReport } from './utils.js'
import 'dotenv/config'

const approaches = {
  a1: () => import('./a1-baseline.js'),
  a2: () => import('./a2-gemini-svg.js'),
  a3: () => import('./a3-gemini-bbox-refine.js'),
  a4: () => import('./a4-sharp-preprocess.js'),
  a5: () => import('./a5-pixel-coords.js'),
  a6: () => import('./a6-gemini-pro.js'),
  a7: () => import('./a7-vectorize.js'),
}

const selected = process.argv.slice(2).filter(a => approaches[a])
const toRun = selected.length > 0 ? selected : Object.keys(approaches)

console.log(`\nRunning ${toRun.length} approaches: ${toRun.join(', ')}`)
console.log('Reference: 4 walls (128pts), 7 doors (35pts), 3 windows (15pts), 1 balcony, 1 balcony_window\n')

const results = []

for (const key of toRun) {
  console.log(`\n--- Starting ${key}... ---`)
  try {
    const mod = await approaches[key]()
    const data = await mod.default()
    const report = printReport(data.name, data.result, data.ref, data.timeMs)
    results.push({ key, name: data.name, report, timeMs: data.timeMs, error: null })
  } catch (e) {
    console.log(`\n  ERROR in ${key}: ${e.message}`)
    results.push({ key, name: key, report: null, timeMs: 0, error: e.message })
  }
}

// Summary table
console.log('\n\n' + '='.repeat(90))
console.log('  SUMMARY COMPARISON')
console.log('='.repeat(90))
console.log('  Approach                          | Walls | Doors | Windows | Time   | Status')
console.log('  ----------------------------------|-------|-------|---------|--------|-------')
for (const r of results) {
  if (r.error) {
    console.log(`  ${r.name.padEnd(34)}| ERROR: ${r.error.slice(0, 50)}`)
  } else {
    const rp = r.report
    console.log(`  ${r.name.slice(0, 34).padEnd(34)}| ${rp.wall.count.padEnd(6)}| ${rp.door.count.padEnd(6)}| ${rp.window.count.padEnd(8)}| ${(r.timeMs / 1000).toFixed(1).padStart(5)}s | ${rp.wall.matched} walls matched`)
  }
}
console.log('  ----------------------------------|-------|-------|---------|--------|-------')
console.log('  Reference                         | 4     | 7     | 3       |    -   | gold standard')
