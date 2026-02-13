import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import Fastify from 'fastify'
import multipart from '@fastify/multipart'
import { analyzeFloorplan } from './src/gemini.js'

const ALLOWED_MIMETYPES = ['image/jpeg', 'image/png', 'image/webp']

export async function buildApp(opts = {}) {
  const app = Fastify(opts)

  await app.register(multipart, {
    limits: { fileSize: 10 * 1024 * 1024 },
  })

  app.get('/', async (request, reply) => {
    return { status: 'ok', service: 'floorplan-service' }
  })

  app.post('/analyze', async (request, reply) => {
    let file
    try {
      file = await request.file()
    } catch {
      return reply.code(400).send({ error: 'Missing file upload' })
    }

    if (!file) {
      return reply.code(400).send({ error: 'Missing file upload' })
    }

    if (!ALLOWED_MIMETYPES.includes(file.mimetype)) {
      return reply.code(400).send({ error: `Unsupported image type: ${file.mimetype}` })
    }

    let buffer
    try {
      buffer = await file.toBuffer()
    } catch {
      return reply.code(400).send({ error: 'Failed to read uploaded file' })
    }

    try {
      const result = await analyzeFloorplan(buffer, file.mimetype)
      return result
    } catch (err) {
      request.log.error({ err }, 'Gemini analysis failed')
      return reply.code(502).send({ error: 'Floor plan analysis failed' })
    }
  })

  return app
}

// Start server when run directly
const currentFile = fileURLToPath(import.meta.url)
const isMainModule = process.argv[1] && resolve(process.argv[1]) === currentFile

if (isMainModule) {
  const app = await buildApp({ logger: true })
  try {
    await app.listen({ port: 3000 })
  } catch (err) {
    app.log.error(err)
    process.exit(1)
  }
}
