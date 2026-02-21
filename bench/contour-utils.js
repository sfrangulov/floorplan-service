/**
 * Contour tracing utilities extracted from a7-vectorize.js.
 * Shared by a7 and a8 benchmarks.
 */

/**
 * Simple contour tracer for binary images.
 * Traces boundaries of black regions in a binary buffer.
 */
export function traceContours(pixelData, width, height) {
  const contours = []
  const visited = new Uint8Array(width * height)

  function isBlack(x, y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return false
    return pixelData[y * width + x] === 0
  }

  // 8-directional neighbor offsets (clockwise from right)
  const dx = [1, 1, 0, -1, -1, -1, 0, 1]
  const dy = [0, 1, 1, 1, 0, -1, -1, -1]

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x
      if (!isBlack(x, y) || visited[idx]) continue

      // Check if this is a border pixel (has at least one white neighbor)
      let isBorder = false
      for (let d = 0; d < 8; d++) {
        if (!isBlack(x + dx[d], y + dy[d])) { isBorder = true; break }
      }
      if (!isBorder) continue

      // Trace contour using Moore neighborhood tracing
      const contour = []
      let cx = x, cy = y
      let dir = 0
      let steps = 0
      const maxSteps = width * height

      do {
        if (!visited[cy * width + cx]) {
          contour.push([cx, cy])
          visited[cy * width + cx] = 1
        }

        // Find next border pixel
        let found = false
        for (let i = 0; i < 8; i++) {
          const nd = (dir + 5 + i) % 8  // start looking back-left
          const nx = cx + dx[nd]
          const ny = cy + dy[nd]
          if (isBlack(nx, ny)) {
            cx = nx; cy = ny; dir = nd
            found = true
            break
          }
        }
        if (!found) break
        steps++
      } while ((cx !== x || cy !== y) && steps < maxSteps)

      if (contour.length >= 10) {
        contours.push(contour)
      }
    }
  }

  return contours
}

/** Simplify contour by keeping every nth point */
export function simplifyContour(contour, targetPoints = 30) {
  if (contour.length <= targetPoints) return contour
  const step = Math.max(1, Math.floor(contour.length / targetPoints))
  const simplified = []
  for (let i = 0; i < contour.length; i += step) {
    simplified.push(contour[i])
  }
  // Close the polygon
  if (simplified.length > 0 && (simplified[0][0] !== simplified[simplified.length - 1][0] || simplified[0][1] !== simplified[simplified.length - 1][1])) {
    simplified.push([...simplified[0]])
  }
  return simplified
}

/** Get bounding box of contour */
export function contourBBox(contour) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of contour) {
    if (x < minX) minX = x; if (y < minY) minY = y
    if (x > maxX) maxX = x; if (y > maxY) maxY = y
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY, area: (maxX - minX) * (maxY - minY) }
}
