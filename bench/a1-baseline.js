/**
 * A1: Baseline - current approach (gemini-3-flash-preview + hybrid CoT prompt)
 */
import { GoogleGenAI } from '@google/genai'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA } from '../src/prompt.js'
import { loadTestImage, loadReference, printReport } from './utils.js'
import 'dotenv/config'

export default async function run() {
  const imageBuffer = loadTestImage()
  const ref = loadReference()
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

  const start = Date.now()
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType: 'image/jpeg' } },
      SYSTEM_PROMPT,
    ],
    config: {
      responseMimeType: 'application/json',
      responseJsonSchema: RESPONSE_JSON_SCHEMA,
    },
  })
  const result = JSON.parse(response.text)
  const timeMs = Date.now() - start

  return { name: 'A1: Baseline (current prompt)', result, ref, timeMs }
}

// Run standalone
if (process.argv[1]?.endsWith('a1-baseline.js')) {
  const { name, result, ref, timeMs } = await run()
  printReport(name, result, ref, timeMs)
}
