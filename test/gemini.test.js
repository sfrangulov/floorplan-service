import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { analyzeFloorplan, preprocessImage, cleanFloorplan } from '../src/gemini.js'

describe('analyzeFloorplan', () => {
  it('is an async function', () => {
    assert.equal(typeof analyzeFloorplan, 'function')
  })

  it('rejects without GEMINI_API_KEY', async () => {
    const original = process.env.GEMINI_API_KEY
    delete process.env.GEMINI_API_KEY
    try {
      await assert.rejects(
        () => analyzeFloorplan(Buffer.from('fake'), 'image/jpeg'),
        { message: /GEMINI_API_KEY/ }
      )
    } finally {
      if (original) process.env.GEMINI_API_KEY = original
    }
  })
})

describe('preprocessImage', () => {
  it('is an async function', () => {
    assert.equal(typeof preprocessImage, 'function')
  })
})

describe('cleanFloorplan', () => {
  it('is an async function', () => {
    assert.equal(typeof cleanFloorplan, 'function')
  })

  it('rejects without GEMINI_API_KEY', async () => {
    const original = process.env.GEMINI_API_KEY
    delete process.env.GEMINI_API_KEY
    try {
      await assert.rejects(
        () => cleanFloorplan(Buffer.from('fake'), 'image/png'),
        { message: /GEMINI_API_KEY/ }
      )
    } finally {
      if (original) process.env.GEMINI_API_KEY = original
    }
  })
})
