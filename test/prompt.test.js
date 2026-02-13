import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA } from '../src/prompt.js'

describe('prompt', () => {
  it('exports a non-empty system prompt string', () => {
    assert.equal(typeof SYSTEM_PROMPT, 'string')
    assert.ok(SYSTEM_PROMPT.length > 100)
  })

  it('exports a valid JSON schema object with required fields', () => {
    assert.equal(RESPONSE_JSON_SCHEMA.type, 'object')
    const props = Object.keys(RESPONSE_JSON_SCHEMA.properties)
    const required = [
      'version', 'width', 'height', 'pixels_per_meter',
      'apartments', 'wall', 'door', 'window', 'bedroom',
      'living_room', 'other_room', 'balcony'
    ]
    for (const key of required) {
      assert.ok(props.includes(key), `missing property: ${key}`)
    }
  })
})
