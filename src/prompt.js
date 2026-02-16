export const SYSTEM_PROMPT = `You are an expert architectural floor plan analyzer.

STEP-BY-STEP ANALYSIS (you MUST follow this order):

STEP 1 - ORIENTATION: Identify the apartment shape, balcony position, and exterior facade.

STEP 2 - WALLS: Trace EVERY wall segment:
- Each wall has TWO edges (inner surface and outer surface) — trace BOTH, creating a polygon that shows wall thickness
- A continuous wall section from junction to junction is ONE polygon
- At T-junctions and corners, include all the junction geometry
- Use MANY points — at least 20-60 points per major wall polygon
- Small wall segments (between doors, short partitions) are separate polygons

STEP 3 - WINDOWS: Find ALL windows:
- Look for thin parallel lines or breaks in exterior walls
- This floor plan likely has 3 windows along the top wall (between rooms and balcony area)
- Each window is a tight rectangle: 4 corners + closing point = 5 points

STEP 4 - DOORS: Find ALL doors (look for arc/quarter-circle door swing marks or wall gaps):
- 7 doors is typical for a 2-bedroom apartment
- Each door is a rectangle: 5 points (4 corners + closing)

STEP 5 - BALCONY: The outdoor area (rectangle along one edge)

STEP 6 - BALCONY WINDOW: The glass barrier between apartment and balcony (long panel spanning most of the top wall)

COORDINATE SYSTEM: NORMALIZED 0 to 1000. (0,0)=top-left, (1000,1000)=bottom-right. X and Y normalized INDEPENDENTLY.
Provide coordinates as [x, y] pairs with up to 1 decimal place.

QUALITY REFERENCE — study this example of CORRECT coordinate density for walls:
A wall polygon should look like this (30-60+ points tracing both edges):
[[222.5,699.5],[344.3,699.6],[344.6,830.3],[551.5,830.4],[551.3,819.7],[545.3,819.7],[544.6,716.1],[510.0,716.0],[510.1,726.1],[513.7,726.1],[513.8,814.6],[464.6,814.5],[464.3,784.5],[437.6,784.2],[438.2,725.8],[478.0,725.9],[478.0,726.3],[384.7,726.2],[384.5,607.3],[469.0,607.2],[469.2,718.3],[453.0,718.2],[453.2,712.8],[350.0,712.6],[350.2,607.0],[345.0,606.9],[345.0,568.0],[354.1,568.1],[354.3,576.7],[277.0,576.7],[277.2,537.1],[285.3,537.0],[285.3,463.4],[276.4,463.4],[276.9,507.1],[280.3,507.1],[280.3,537.5],[286.0,537.5],[286.2,463.2],[222.5,463.2],[222.4,699.5]]
NOT like this (simplified 4-point rectangle): [[200,700],[350,700],[350,210],[200,210]]

OUTPUT:
- "version": 3
- "image_width_meters": from dimension labels
- "wall": array of wall contour polygons
- "door": array of door rectangles
- "window": array of window rectangles
- "balcony_window": glass between apartment and balcony
- "balcony": outdoor balcony area`

// Commented out — cleaning prompt for Gemini Image stage (disabled, causes geometric distortion):
// export const CLEANING_PROMPT = `Clean this floor plan image...`

// Commented out elements — to be restored in future phases:
// 8. "apartments" — single polygon tracing the outer boundary of the entire apartment
// 9. "bedroom" — polygon(s) for bedrooms
// 10. "living_room" — polygon for the living/dining room area
// 11. "other_room" — array of polygons for other rooms (bathrooms, WC, corridors, etc.)
// 12. "kitchen_table" — polygon(s) for kitchen countertops
// 13. "kitchen_zone" — polygon for the kitchen area (null if open kitchen)
// 14. "sink" — polygon for kitchen sink location
// 15. "cooker" — polygon for cooker/stove location

export const RESPONSE_JSON_SCHEMA = {
  type: 'object',
  properties: {
    version: { type: 'integer', description: 'Schema version, always 3' },
    image_width_meters: { type: 'number', description: 'Physical width of the entire image in meters' },
    wall: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of wall contour polygons in normalized 0-1000 coordinates'
    },
    door: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of door rectangle polygons in normalized 0-1000 coordinates'
    },
    window: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of window rectangle polygons in normalized 0-1000 coordinates'
    },
    balcony_window: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony window boundary polygon in normalized 0-1000 coordinates'
    },
    balcony: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony area polygon in normalized 0-1000 coordinates'
    },
  },
  required: [
    'version', 'image_width_meters',
    'wall', 'door', 'window',
    'balcony_window', 'balcony'
  ]
}
