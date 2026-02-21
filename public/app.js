// --- Layer config ---
// Layers are rendered in this order (first = bottom, last = top)
const LAYER_CONFIG = {
  balcony:        { color: '#90E0EF', opacity: 0.3,  strokeWidth: 1, label: 'Balconies' },
  wall:           { color: '#888888', opacity: 0.5,  strokeWidth: 1, label: 'Walls' },
  door:           { color: '#FF6B35', opacity: 0.5,  strokeWidth: 1, label: 'Doors' },
  window:         { color: '#00B4D8', opacity: 0.5,  strokeWidth: 1, label: 'Windows' },
  balcony_window: { color: '#48CAE4', opacity: 0.4,  strokeWidth: 1, label: 'Balcony Windows' },
  // Commented out — future phases:
  // apartments:     { color: '#264653', opacity: 0.2,  strokeWidth: 2, dash: [8, 4], label: 'Apartments' },
  // bedroom:        { color: '#7B2D8E', opacity: 0.35, strokeWidth: 1, label: 'Bedrooms' },
  // living_room:    { color: '#2D6A4F', opacity: 0.35, strokeWidth: 1, label: 'Living Rooms' },
  // other_room:     { color: '#E9C46A', opacity: 0.35, strokeWidth: 1, label: 'Other Rooms' },
  // kitchen_zone:   { color: '#F4A261', opacity: 0.3,  strokeWidth: 1, label: 'Kitchen Zones' },
  // kitchen_table:  { color: '#E76F51', opacity: 0.5,  strokeWidth: 1, label: 'Kitchen Tables' },
  // sink:           { color: '#219EBC', opacity: 0.6,  strokeWidth: 1, label: 'Sinks' },
  // cooker:         { color: '#FB8500', opacity: 0.6,  strokeWidth: 1, label: 'Cookers' },
}

const GAP = 40 // pixels gap between image and polygons

// Actual image dimensions (set after image loads)
let actualImageWidth = 0
let actualImageHeight = 0

// --- DOM refs ---
const fileInput = document.getElementById('file-input')
const uploadBtn = document.getElementById('upload-btn')
const loading = document.getElementById('loading')
const statusText = document.getElementById('status-text')
const hoverInfo = document.getElementById('hover-info')
const layersPanel = document.getElementById('layers-panel')
const infoPanel = document.getElementById('info-panel')
const container = document.getElementById('konva-container')

// --- KonvaJS setup ---
const stage = new Konva.Stage({
  container: 'konva-container',
  width: container.offsetWidth,
  height: container.offsetHeight,
  draggable: true,
})

const imageLayer = new Konva.Layer()
stage.add(imageLayer)

const polygonBgLayer = new Konva.Layer()
stage.add(polygonBgLayer)

const konvaLayers = {}
for (const key of Object.keys(LAYER_CONFIG)) {
  const layer = new Konva.Layer()
  stage.add(layer)
  konvaLayers[key] = layer
}

// --- Zoom (mouse wheel) ---
const SCALE_BY = 1.05
stage.on('wheel', (e) => {
  e.evt.preventDefault()
  const oldScale = stage.scaleX()
  const pointer = stage.getPointerPosition()
  const mousePointTo = {
    x: (pointer.x - stage.x()) / oldScale,
    y: (pointer.y - stage.y()) / oldScale,
  }
  const direction = e.evt.deltaY > 0 ? -1 : 1
  const newScale = direction > 0 ? oldScale * SCALE_BY : oldScale / SCALE_BY
  stage.scale({ x: newScale, y: newScale })
  const newPos = {
    x: pointer.x - mousePointTo.x * newScale,
    y: pointer.y - mousePointTo.y * newScale,
  }
  stage.position(newPos)
})

// --- Resize ---
window.addEventListener('resize', () => {
  stage.width(container.offsetWidth)
  stage.height(container.offsetHeight)
})

// --- Upload ---
uploadBtn.addEventListener('click', () => fileInput.click())
fileInput.addEventListener('change', handleFileUpload)

async function handleFileUpload(e) {
  const file = e.target.files[0]
  if (!file) return

  uploadBtn.disabled = true
  statusText.textContent = 'Uploading...'

  // Show image immediately
  const objectUrl = URL.createObjectURL(file)
  try {
    await loadImage(objectUrl)
  } finally {
    URL.revokeObjectURL(objectUrl)
  }

  // Send to API
  loading.classList.remove('hidden')
  statusText.textContent = 'Analyzing...'

  const formData = new FormData()
  formData.append('image', file)

  try {
    const response = await fetch('/analyze', { method: 'POST', body: formData })
    if (!response.ok) {
      const err = await response.json().catch(() => ({}))
      throw new Error(err.error || `Server error: ${response.status}`)
    }
    const result = await response.json()
    renderPolygons(result)
    fitSideBySide()
    updateInfo(result)
    statusText.textContent = 'Analysis complete'
  } catch (err) {
    statusText.textContent = 'Error: ' + err.message
  } finally {
    loading.classList.add('hidden')
    uploadBtn.disabled = false
    fileInput.value = ''
  }
}

// --- Load image onto canvas ---
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      actualImageWidth = img.width
      actualImageHeight = img.height

      imageLayer.destroyChildren()
      const konvaImg = new Konva.Image({ image: img, x: 0, y: 0 })
      imageLayer.add(konvaImg)

      // Fit image only (before analysis)
      fitImageOnly()
      stage.batchDraw()
      resolve()
    }
    img.onerror = reject
    img.src = url
  })
}

// --- Fit just the image to canvas ---
function fitImageOnly() {
  const padding = 0.9
  const scaleX = (stage.width() * padding) / actualImageWidth
  const scaleY = (stage.height() * padding) / actualImageHeight
  const scale = Math.min(scaleX, scaleY)
  stage.scale({ x: scale, y: scale })
  stage.position({
    x: (stage.width() - actualImageWidth * scale) / 2,
    y: (stage.height() - actualImageHeight * scale) / 2,
  })
}

// --- Fit side-by-side view (image + polygons) to canvas ---
function fitSideBySide() {
  const totalWidth = actualImageWidth * 2 + GAP
  const totalHeight = actualImageHeight
  const padding = 0.9
  const scaleX = (stage.width() * padding) / totalWidth
  const scaleY = (stage.height() * padding) / totalHeight
  const scale = Math.min(scaleX, scaleY)
  stage.scale({ x: scale, y: scale })
  stage.position({
    x: (stage.width() - totalWidth * scale) / 2,
    y: (stage.height() - totalHeight * scale) / 2,
  })
  stage.batchDraw()
}

// --- Scale normalized (0-1000) coordinates to actual image pixels, offset to the right ---
function scalePoint(point) {
  const offsetX = actualImageWidth + GAP
  return [
    point[0] * actualImageWidth / 1000 + offsetX,
    point[1] * actualImageHeight / 1000,
  ]
}

// --- Render polygons ---
function renderPolygons(data) {
  // Clear previous
  polygonBgLayer.destroyChildren()
  for (const layer of Object.values(konvaLayers)) {
    layer.destroyChildren()
  }

  // Draw dark background for polygon area
  const offsetX = actualImageWidth + GAP
  polygonBgLayer.add(new Konva.Rect({
    x: offsetX,
    y: 0,
    width: actualImageWidth,
    height: actualImageHeight,
    fill: '#1a1a2e',
    stroke: '#333',
    strokeWidth: 1,
  }))
  polygonBgLayer.batchDraw()

  for (const [key, config] of Object.entries(LAYER_CONFIG)) {
    const polygons = normalizePolygons(key, data)
    if (!polygons) continue

    const layer = konvaLayers[key]
    for (const polygon of polygons) {
      // Scale from 0-1000 to actual image pixels, offset to the right
      const scaledPoints = polygon.map(scalePoint).flat()

      const lineConfig = {
        points: scaledPoints,
        fill: config.color,
        stroke: config.color,
        strokeWidth: config.strokeWidth,
        closed: true,
        opacity: config.opacity,
      }

      if (config.dash) {
        lineConfig.dash = config.dash
      }

      const line = new Konva.Line(lineConfig)

      // Hover effects
      line.on('mouseenter', () => {
        line.opacity(Math.min(config.opacity + 0.25, 1))
        line.strokeWidth(config.strokeWidth + 1)
        layer.batchDraw()
        hoverInfo.textContent = config.label
        stage.container().style.cursor = 'pointer'
      })
      line.on('mouseleave', () => {
        line.opacity(config.opacity)
        line.strokeWidth(config.strokeWidth)
        layer.batchDraw()
        hoverInfo.textContent = ''
        stage.container().style.cursor = 'grab'
      })

      layer.add(line)
    }
    layer.batchDraw()
  }
}

// --- Normalize polygons ---
function normalizePolygons(key, data) {
  // Support both flat format (data.wall) and nested (data.elements.wall)
  const raw = data[key] || (data.elements && data.elements[key])
  if (!raw || !Array.isArray(raw) || raw.length === 0) return null

  // Single polygon: [[x,y],[x,y],...] — first element is [number, number]
  if (typeof raw[0][0] === 'number') {
    return [raw]
  }
  // Array of polygons: [[[x,y],...],[[x,y],...]]
  return raw
}

// --- Build layer toggles ---
function buildLayerToggles() {
  // Toggle all button
  const toggleAllBtn = document.createElement('button')
  toggleAllBtn.className = 'toggle-all-btn'
  toggleAllBtn.textContent = 'Hide All'
  let allVisible = true
  toggleAllBtn.addEventListener('click', () => {
    allVisible = !allVisible
    toggleAllBtn.textContent = allVisible ? 'Hide All' : 'Show All'
    for (const key of Object.keys(LAYER_CONFIG)) {
      konvaLayers[key].visible(allVisible)
    }
    layersPanel.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.checked = allVisible
    })
  })
  layersPanel.appendChild(toggleAllBtn)

  for (const [key, config] of Object.entries(LAYER_CONFIG)) {
    const label = document.createElement('label')
    label.className = 'layer-item'

    const checkbox = document.createElement('input')
    checkbox.type = 'checkbox'
    checkbox.checked = true
    checkbox.addEventListener('change', () => {
      konvaLayers[key].visible(checkbox.checked)
    })

    const dot = document.createElement('span')
    dot.className = 'color-dot'
    dot.style.backgroundColor = config.color

    const text = document.createElement('span')
    text.textContent = config.label

    label.appendChild(checkbox)
    label.appendChild(dot)
    label.appendChild(text)
    layersPanel.appendChild(label)
  }
}

// --- Update info panel ---
function updateInfo(data) {
  while (infoPanel.firstChild) {
    infoPanel.removeChild(infoPanel.firstChild)
  }

  const lines = []
  if (actualImageWidth && actualImageHeight) {
    lines.push('Image: ' + actualImageWidth + ' x ' + actualImageHeight + ' px')
  }
  if (data.image_width_meters) {
    lines.push('Width: ~' + data.image_width_meters.toFixed(1) + ' m')
  }
  for (const [key, config] of Object.entries(LAYER_CONFIG)) {
    const polygons = normalizePolygons(key, data)
    if (polygons) {
      lines.push(config.label + ': ' + polygons.length)
    }
  }

  lines.forEach((line, i) => {
    if (i > 0) infoPanel.appendChild(document.createElement('br'))
    infoPanel.appendChild(document.createTextNode(line))
  })
}

// --- Init ---
buildLayerToggles()
stage.container().style.cursor = 'grab'
