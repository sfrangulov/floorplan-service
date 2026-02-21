import Replicate from 'replicate'

/**
 * Analyze a floor plan image using the segmentation model on Replicate.
 *
 * @param {Buffer} imageBuffer - raw image bytes
 * @param {string} mimeType - e.g. 'image/jpeg'
 * @returns {Promise<object>} parsed floor plan JSON with polygons
 */
export async function analyzeFloorplanSegmentation(imageBuffer, mimeType) {
  const apiToken = process.env.REPLICATE_API_TOKEN
  if (!apiToken) {
    throw new Error('REPLICATE_API_TOKEN environment variable is required')
  }

  const replicate = new Replicate({ auth: apiToken })

  const base64 = imageBuffer.toString('base64')
  const dataUri = `data:${mimeType};base64,${base64}`

  const modelVersion = process.env.SEGMENTATION_MODEL_VERSION
  if (!modelVersion) {
    throw new Error(
      'SEGMENTATION_MODEL_VERSION environment variable is required '
      + '(format: owner/model:version)'
    )
  }

  const output = await replicate.run(modelVersion, {
    input: { image: dataUri },
  })

  const result = typeof output === 'string' ? JSON.parse(output) : output

  delete result._inference_time_ms

  return result
}
