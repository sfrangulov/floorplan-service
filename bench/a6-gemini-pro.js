/**
 * A6: Gemini 2.5 Pro with current prompt
 * Test whether the more capable (but slower) model produces better coordinates.
 * Using gemini-2.5-pro instead of gemini-3-flash-preview.
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
    model: 'gemini-3-pro-preview',
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

  return { name: 'A6: Gemini 2.5 Pro', result, ref, timeMs }
}

if (process.argv[1]?.endsWith('a6-gemini-pro.js')) {
  const { name, result, ref, timeMs } = await run()
  printReport(name, result, ref, timeMs)
}
