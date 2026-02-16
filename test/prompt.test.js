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

  it('includes step-by-step analysis instructions', () => {
    assert.ok(SYSTEM_PROMPT.includes('STEP 1'), 'prompt should have step-by-step guidance')
    assert.ok(SYSTEM_PROMPT.includes('STEP 2'), 'prompt should have wall tracing step')
  })

  it('includes quality reference example', () => {
    assert.ok(SYSTEM_PROMPT.includes('QUALITY REFERENCE'), 'prompt should have quality example')
    assert.ok(SYSTEM_PROMPT.includes('222.5,699.5'), 'prompt should have coordinate example')
  })

  it('exports a valid JSON schema with 5 active element fields', () => {
    assert.equal(RESPONSE_JSON_SCHEMA.type, 'object')
    const props = Object.keys(RESPONSE_JSON_SCHEMA.properties)
    const activeFields = [
      'version', 'image_width_meters',
      'wall', 'door', 'window',
      'balcony_window', 'balcony'
    ]
    for (const key of activeFields) {
      assert.ok(props.includes(key), `missing property: ${key}`)
    }
  })

  it('does not include commented-out elements in schema', () => {
    const props = Object.keys(RESPONSE_JSON_SCHEMA.properties)
    const removedFields = [
      'apartments', 'bedroom', 'living_room', 'other_room',
      'kitchen_table', 'kitchen_zone', 'sink', 'cooker'
    ]
    for (const key of removedFields) {
      assert.ok(!props.includes(key), `should not have property: ${key}`)
    }
  })

  it('requires only active fields', () => {
    const required = RESPONSE_JSON_SCHEMA.required
    assert.ok(required.includes('wall'))
    assert.ok(required.includes('door'))
    assert.ok(required.includes('window'))
    assert.ok(required.includes('balcony'))
    assert.ok(required.includes('balcony_window'))
    assert.ok(!required.includes('apartments'))
    assert.ok(!required.includes('bedroom'))
  })
})
