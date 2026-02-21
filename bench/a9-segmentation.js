/**
 * A9: Segmentation model approach
 * Uses the deployed SegFormer model on Replicate for floorplan analysis.
 */

import 'dotenv/config'
import { loadTestImage, loadReference, printReport } from './utils.js'
import { analyzeFloorplanSegmentation } from '../src/segmentation.js'

async function run() {
  const imageBuffer = loadTestImage()
  const reference = loadReference()

  console.log('A9: Segmentation Model (SegFormer on Replicate)')
  console.log('Running inference...')

  const start = Date.now()
  const result = await analyzeFloorplanSegmentation(imageBuffer, 'image/jpeg')
  const elapsed = Date.now() - start

  printReport('A9: Segmentation Model', result, reference, elapsed)

  // Save result
  const fs = await import('fs')
  fs.writeFileSync('bench/a9-result.json', JSON.stringify(result, null, 2))
  console.log('\nResult saved to bench/a9-result.json')
}

run().catch(console.error)
