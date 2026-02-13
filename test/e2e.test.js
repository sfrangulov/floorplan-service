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

  it('analyzes U3laAgYJ.jpg and returns valid structure', { timeout: 120_000 }, async () => {
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
    assert.equal(result.version, 2)
    assert.equal(typeof result.width, 'number')
    assert.equal(typeof result.height, 'number')
    assert.equal(typeof result.pixels_per_meter, 'number')
    assert.ok(Array.isArray(result.apartments), 'apartments should be array')
    assert.ok(Array.isArray(result.wall), 'wall should be array')
    assert.ok(Array.isArray(result.door), 'door should be array')
    assert.ok(Array.isArray(result.window), 'window should be array')
    assert.ok(Array.isArray(result.bedroom), 'bedroom should be array')
    assert.ok(Array.isArray(result.other_room), 'other_room should be array')

    // Validate polygon structure
    assert.ok(result.apartments.length > 3, 'apartments should have at least 3 points')
    assert.equal(result.apartments[0].length, 2, 'each point should be [x, y]')

    console.log(`Result: ${result.width}x${result.height}, ${result.pixels_per_meter} px/m`)
    console.log(`Rooms: ${result.bedroom.length} bedrooms, ${result.other_room.length} other rooms`)
    console.log(`Elements: ${result.wall.length} walls, ${result.door.length} doors, ${result.window.length} windows`)
  })
})
