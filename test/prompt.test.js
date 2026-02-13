import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA } from '../src/prompt.js'

describe('prompt', () => {
  it('exports a non-empty system prompt string', () => {
    assert.equal(typeof SYSTEM_PROMPT, 'string')
    assert.ok(SYSTEM_PROMPT.length > 100)
  })

  it('uses normalized 0-1000 coordinate system', () => {
    assert.ok(SYSTEM_PROMPT.includes('0 to 1000'), 'prompt should mention 0-1000 range')
    assert.ok(SYSTEM_PROMPT.includes('NORMALIZED'), 'prompt should mention normalized coordinates')
  })

  it('exports a valid JSON schema object with required fields', () => {
    assert.equal(RESPONSE_JSON_SCHEMA.type, 'object')
    const props = Object.keys(RESPONSE_JSON_SCHEMA.properties)
    const allFields = [
      'version', 'image_width_meters',
      'apartments', 'wall', 'door', 'window',
      'balcony_window', 'balcony', 'bedroom', 'living_room',
      'other_room', 'kitchen_table', 'kitchen_zone', 'sink', 'cooker'
    ]
    for (const key of allFields) {
      assert.ok(props.includes(key), `missing property: ${key}`)
    }
  })

  it('has kitchen_zone as optional (not in required)', () => {
    assert.ok(!RESPONSE_JSON_SCHEMA.required.includes('kitchen_zone'))
    assert.ok(RESPONSE_JSON_SCHEMA.required.includes('apartments'))
  })

  it('does not require pixel-based width/height fields', () => {
    assert.ok(!RESPONSE_JSON_SCHEMA.required.includes('width'))
    assert.ok(!RESPONSE_JSON_SCHEMA.required.includes('height'))
    assert.ok(!RESPONSE_JSON_SCHEMA.required.includes('pixels_per_meter'))
  })
})
