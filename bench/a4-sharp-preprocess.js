/**
 * A4: Sharp Preprocessing → Gemini
 * Preprocess image with Sharp: grayscale → high contrast → threshold
 * Then send clean B&W image to Gemini for analysis.
 */
import { GoogleGenAI } from '@google/genai'
import sharp from 'sharp'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA } from '../src/prompt.js'
import { loadTestImage, loadReference, printReport } from './utils.js'
import 'dotenv/config'

async function preprocessImage(imageBuffer) {
  // Convert to high-contrast B&W to make walls/doors/windows clearer
  return sharp(imageBuffer)
    .grayscale()
    .normalize()  // maximize contrast
    .threshold(180)  // binary threshold - walls become solid black
    .sharpen({ sigma: 1.5 })
    .jpeg({ quality: 95 })
    .toBuffer()
}

export default async function run() {
  const imageBuffer = loadTestImage()
  const ref = loadReference()
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

  const start = Date.now()

  const processed = await preprocessImage(imageBuffer)
  console.log(`  Preprocessed: ${(imageBuffer.length / 1024).toFixed(0)}KB → ${(processed.length / 1024).toFixed(0)}KB`)

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: processed.toString('base64'), mimeType: 'image/jpeg' } },
      SYSTEM_PROMPT,
    ],
    config: {
      responseMimeType: 'application/json',
      responseJsonSchema: RESPONSE_JSON_SCHEMA,
    },
  })
  const result = JSON.parse(response.text)
  const timeMs = Date.now() - start

  return { name: 'A4: Sharp preprocess → Gemini', result, ref, timeMs }
}

if (process.argv[1]?.endsWith('a4-sharp-preprocess.js')) {
  const { name, result, ref, timeMs } = await run()
  printReport(name, result, ref, timeMs)
}
