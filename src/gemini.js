import { GoogleGenAI } from '@google/genai'
import sharp from 'sharp'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA, CLEANING_PROMPT } from './prompt.js'

/**
 * Stage 1: Sharp preprocessing — grayscale + normalize + sharpen
 * @param {Buffer} imageBuffer - raw image bytes
 * @returns {Promise<Buffer>} preprocessed PNG buffer
 */
export async function preprocessImage(imageBuffer) {
  return sharp(imageBuffer)
    .greyscale()
    .normalize()
    .sharpen()
    .png()
    .toBuffer()
}

/**
 * Stage 2: Gemini Image cleaning — remove furniture, text, dimensions
 * @param {Buffer} imageBuffer - preprocessed image bytes
 * @param {string} mimeType - e.g. 'image/png'
 * @returns {Promise<Buffer>} cleaned image buffer
 */
export async function cleanFloorplan(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required')
  }

  const ai = new GoogleGenAI({ apiKey })

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: [
      { text: CLEANING_PROMPT },
      { inlineData: { data: imageBuffer.toString('base64'), mimeType } },
    ],
    config: {
      responseModalities: ['image', 'text'],
    },
  })

  // Extract image from response parts
  for (const part of response.candidates[0].content.parts) {
    if (part.inlineData) {
      return Buffer.from(part.inlineData.data, 'base64')
    }
  }

  // If no image returned, fall back to original
  console.warn('Gemini Image did not return an image, using original')
  return imageBuffer
}

/**
 * Stage 3: Gemini coordinate extraction from cleaned image
 * @param {Buffer} imageBuffer - cleaned image bytes
 * @param {string} mimeType - e.g. 'image/png'
 * @returns {Promise<object>} parsed floor plan JSON
 */
async function extractCoordinates(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  const ai = new GoogleGenAI({ apiKey })

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
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

/**
 * Full pipeline: preprocess → clean → extract
 * @param {Buffer} imageBuffer - raw image bytes
 * @param {string} mimeType - e.g. 'image/jpeg'
 * @returns {Promise<object>} parsed floor plan JSON
 */
export async function analyzeFloorplan(imageBuffer, mimeType) {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required')
  }

  // Stage 1: Sharp preprocessing
  const preprocessed = await preprocessImage(imageBuffer)

  // Stage 2: Gemini Image cleaning
  const cleaned = await cleanFloorplan(preprocessed, 'image/png')

  // Stage 3: Coordinate extraction from cleaned image
  return extractCoordinates(cleaned, 'image/png')
}
