/**
 * Run an approach and save raw JSON result.
 * Usage: node bench/run-and-save.js a1|a2|...|a7
 */
import fs from 'fs'
import 'dotenv/config'

const key = process.argv[2]
if (!key) { console.error('Usage: node bench/run-and-save.js <approach>'); process.exit(1) }

const mod = await import(`./${key.replace('a', 'a')}-${({
  a1: 'baseline', a2: 'gemini-svg', a3: 'gemini-bbox-refine',
  a4: 'sharp-preprocess', a5: 'pixel-coords', a6: 'gemini-pro', a7: 'vectorize', a8: 'grounded-sam',
})[key]}.js`)

const data = await mod.default()
const outPath = `bench/${key}-result.json`
fs.writeFileSync(outPath, JSON.stringify(data.result, null, 2))
console.log(`Saved to ${outPath} (${(data.timeMs / 1000).toFixed(1)}s)`)
