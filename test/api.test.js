import { describe, it, before, after } from 'node:test'
import assert from 'node:assert/strict'
import { buildApp } from '../index.js'

describe('POST /analyze', () => {
  let app

  before(async () => {
    app = await buildApp({ logger: false })
  })

  after(async () => {
    await app.close()
  })

  it('returns 400 when no file is uploaded', async () => {
    const response = await app.inject({
      method: 'POST',
      url: '/analyze',
    })
    assert.equal(response.statusCode, 400)
  })

  it('returns 400 for non-image content type', async () => {
    const boundary = '----TestBoundary123'
    const body = Buffer.concat([
      Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="image"; filename="test.txt"\r\nContent-Type: text/plain\r\n\r\n`),
      Buffer.from('not an image'),
      Buffer.from(`\r\n--${boundary}--\r\n`),
    ])

    const response = await app.inject({
      method: 'POST',
      url: '/analyze',
      headers: { 'content-type': `multipart/form-data; boundary=${boundary}` },
      payload: body,
    })
    assert.equal(response.statusCode, 400)
  })

  it('GET / returns status ok', async () => {
    const response = await app.inject({
      method: 'GET',
      url: '/',
    })
    assert.equal(response.statusCode, 200)
    const body = JSON.parse(response.body)
    assert.equal(body.status, 'ok')
    assert.equal(body.service, 'floorplan-service')
  })
})
