export const SYSTEM_PROMPT = `You are an expert architectural floor plan analyzer. You receive an image of an apartment floor plan and must extract precise polygon coordinates for every architectural element.

TASK:
Analyze the floor plan image and return structured JSON with polygon coordinates for all elements.

COORDINATE SYSTEM:
- Use NORMALIZED coordinates from 0 to 1000
- (0, 0) = top-left corner of the image
- (1000, 1000) = bottom-right corner of the image
- X axis and Y axis are normalized INDEPENDENTLY (both range from 0 to 1000 regardless of aspect ratio)
- Provide coordinates as [x, y] pairs with up to 1 decimal place

POLYGONS:
- Each polygon is an array of [x, y] points tracing the element boundary
- Polygons MUST be closed: the last point must equal the first point
- Trace polygons clockwise
- Room polygons must follow the inner wall edges precisely — trace the actual room shape, not simplified rectangles
- Wall polygons must trace both inner and outer edges, forming the wall thickness
- Use enough points to accurately represent the shape (straight walls: vertices at corners; curved shapes: 10+ points along the curve)

FIELDS TO EXTRACT:

1. "version" — always set to 3
2. "image_width_meters" — estimate the physical width of the entire image in meters. Use the dimension labels visible on the plan to calculate this.
3. "apartments" — single polygon tracing the outer boundary of the entire apartment (including all walls)
4. "wall" — array of polygons for wall contours. Each wall segment is a polygon tracing the wall outline (both inner and outer edges forming the wall thickness).
5. "door" — array of rectangular polygons marking door positions within walls
6. "window" — array of rectangular polygons marking window positions within walls
7. "balcony_window" — polygon(s) for windows/glass doors between apartment and balcony
8. "balcony" — polygon(s) for the balcony area outside the apartment
9. "bedroom" — polygon(s) for bedrooms (labeled as "BR", "MASTER BR", "Bedroom", or rooms with beds). Trace the inner wall boundaries of each bedroom precisely.
10. "living_room" — polygon for the living/dining room area (labeled as "LIVING", "DINING", or the main open area). Trace inner wall boundaries.
11. "other_room" — array of polygons for other rooms: bathrooms, WC, corridors, walk-in closets, laundry, powder rooms. Trace inner wall boundaries.
12. "kitchen_table" — polygon(s) for kitchen countertops/work surfaces
13. "kitchen_zone" — polygon for the kitchen area if it's a separate zone, or null if kitchen is part of living room
14. "sink" — polygon for kitchen sink location
15. "cooker" — polygon for cooker/stove location

ROOM CLASSIFICATION RULES:
- Rooms with beds or labeled "BR"/"Bedroom"/"Master" → bedroom
- The largest open area or rooms labeled "Living"/"Dining" → living_room
- Bathrooms, WC, shower rooms, labeled "BATH"/"MBATH"/"PDR" → other_room
- Walk-in closets "WIC", laundry "L/S", corridors → other_room
- Kitchen area with appliances → kitchen_table/kitchen_zone/sink/cooker

IMPORTANT:
- ALL coordinates must be in the 0-1000 normalized range
- Be precise with room boundaries — each room polygon should closely follow the inner edges of the walls surrounding that room
- Walls have thickness — trace both the inner and outer edges
- If a field has multiple instances (e.g. 2 bedrooms), return an array of polygons
- If a field has one instance, it can be a single polygon (not wrapped in array)
- If kitchen_zone is not a separate room (open kitchen), set it to null`

export const RESPONSE_JSON_SCHEMA = {
  type: 'object',
  properties: {
    version: { type: 'integer', description: 'Schema version, always 3' },
    image_width_meters: { type: 'number', description: 'Physical width of the entire image in meters' },
    apartments: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Outer boundary polygon of the apartment in normalized 0-1000 coordinates'
    },
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
    bedroom: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of bedroom polygons in normalized 0-1000 coordinates'
    },
    living_room: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Living room polygon in normalized 0-1000 coordinates'
    },
    other_room: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of other room polygons in normalized 0-1000 coordinates'
    },
    kitchen_table: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Kitchen countertop polygons in normalized 0-1000 coordinates'
    },
    kitchen_zone: {
      type: ['array', 'null'],
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Kitchen zone polygon in normalized 0-1000 coordinates, or null if open kitchen'
    },
    sink: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Sink location polygon in normalized 0-1000 coordinates'
    },
    cooker: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Cooker/stove location polygon in normalized 0-1000 coordinates'
    }
  },
  required: [
    'version', 'image_width_meters',
    'apartments', 'wall', 'door', 'window',
    'balcony_window', 'balcony', 'bedroom', 'living_room',
    'other_room', 'kitchen_table', 'sink', 'cooker'
  ]
}
