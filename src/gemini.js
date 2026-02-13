import { GoogleGenAI } from '@google/genai'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA } from './prompt.js'

/**
 * @param {Buffer} imageBuffer - raw image bytes
 * @param {string} mimeType - e.g. 'image/jpeg'
 * @returns {Promise<object>} parsed floor plan JSON
 */
export async function analyzeFloorplan(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required')
  }

  const ai = new GoogleGenAI({ apiKey })

  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: [
      { inlineData: { data: imageBuffer.toString('base64'), mimeType } },
      SYSTEM_PROMPT,
    ],
    config: {
      responseMimeType: 'application/json',
      responseJsonSchema: RESPONSE_JSON_SCHEMA,
    },
  })

  return JSON.parse(response.text)
}
