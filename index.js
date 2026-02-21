import 'dotenv/config'
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'
import Fastify from 'fastify'
import multipart from '@fastify/multipart'
import fastifyStatic from '@fastify/static'
import { analyzeFloorplan } from './src/gemini.js'
import { analyzeFloorplanSegmentation, analyzeFloorplanSegmentationLocal } from './src/segmentation.js'

const ALLOWED_MIMETYPES = ['image/jpeg', 'image/png', 'image/webp']

export async function buildApp(opts = {}) {
  const app = Fastify(opts)

  const __filename = fileURLToPath(import.meta.url)
  const __dirname = dirname(__filename)

  await app.register(multipart, {
    limits: { fileSize: 10 * 1024 * 1024 },
  })

  await app.register(fastifyStatic, {
    root: join(__dirname, 'public'),
    prefix: '/',
  })

  app.get('/api/health', async () => {
    return { status: 'ok', service: 'floorplan-service' }
  })

  // Alias for demo-frontend compatibility (sends field name "floorplan")
  app.post('/parse-floorplan', async (request, reply) => {
    return handleAnalyze(request, reply)
  })

  app.post('/analyze', async (request, reply) => {
    return handleAnalyze(request, reply)
  })

  async function handleAnalyze(request, reply) {
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
      const backend = process.env.ANALYSIS_BACKEND || 'gemini'
      let result
      if (backend === 'segmentation-local') {
        result = await analyzeFloorplanSegmentationLocal(buffer)
      } else if (backend === 'segmentation') {
        result = await analyzeFloorplanSegmentation(buffer, file.mimetype)
      } else {
        result = await analyzeFloorplan(buffer, file.mimetype)
      }
      return result
    } catch (err) {
      request.log.error({ err }, 'Floor plan analysis failed')
      return reply.code(502).send({ error: 'Floor plan analysis failed' })
    }
  }

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
