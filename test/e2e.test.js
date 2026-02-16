import { describe, it, before, after } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { buildApp } from '../index.js'

describe('E2E: POST /analyze with real image', () => {
  let app

  before(async () => {
    if (!process.env.GEMINI_API_KEY) {
      console.log('Skipping E2E tests: GEMINI_API_KEY not set')
      return
    }
    app = await buildApp({ logger: false })
  })

  after(async () => {
    if (app) await app.close()
  })

  it('analyzes U3laAgYJ.jpg and returns valid v3 structure with 5 elements', { timeout: 180_000 }, async () => {
    if (!process.env.GEMINI_API_KEY) return

    const imagePath = path.join(process.cwd(), 'test-data', 'U3laAgYJ.jpg')
    const imageBuffer = fs.readFileSync(imagePath)

    const boundary = '----TestBoundary'
    const body = Buffer.concat([
      Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="image"; filename="U3laAgYJ.jpg"\r\nContent-Type: image/jpeg\r\n\r\n`),
      imageBuffer,
      Buffer.from(`\r\n--${boundary}--\r\n`),
    ])

    const response = await app.inject({
      method: 'POST',
      url: '/analyze',
      headers: {
        'content-type': `multipart/form-data; boundary=${boundary}`,
      },
      payload: body,
    })

    assert.equal(response.statusCode, 200)

    const result = JSON.parse(response.body)
    assert.equal(result.version, 3)
    assert.equal(typeof result.image_width_meters, 'number')
    assert.ok(result.image_width_meters > 0, 'image_width_meters should be positive')

    // Active elements should be present
    assert.ok(Array.isArray(result.wall), 'wall should be array')
    assert.ok(Array.isArray(result.door), 'door should be array')
    assert.ok(Array.isArray(result.window), 'window should be array')

    // Validate wall polygon structure
    assert.ok(result.wall.length > 0, 'should detect at least 1 wall')
    assert.ok(result.wall[0].length > 2, 'wall polygon should have at least 3 points')
    assert.equal(result.wall[0][0].length, 2, 'each point should be [x, y]')

    // Validate coordinates are in 0-1000 range
    for (const wallPolygon of result.wall) {
      for (const point of wallPolygon) {
        assert.ok(point[0] >= 0 && point[0] <= 1000, `x=${point[0]} should be 0-1000`)
        assert.ok(point[1] >= 0 && point[1] <= 1000, `y=${point[1]} should be 0-1000`)
      }
    }

    console.log(`Result: image_width=${result.image_width_meters}m`)
    console.log(`Elements: ${result.wall.length} walls, ${result.door.length} doors, ${result.window.length} windows`)
  })
})
