export const SYSTEM_PROMPT = `You are an expert architectural floor plan analyzer. You receive an image of an apartment floor plan and must extract precise polygon coordinates for every architectural element.

TASK:
Analyze the floor plan image and return structured JSON with polygon coordinates for all elements.

COORDINATE SYSTEM:
- All coordinates are in pixels relative to the image (origin = top-left corner)
- X axis = horizontal (left to right), Y axis = vertical (top to bottom)
- Provide coordinates as [x, y] pairs with up to 2 decimal places

POLYGONS:
- Each polygon is an array of [x, y] points tracing the element boundary
- Polygons MUST be closed: the last point must equal the first point
- Trace polygons clockwise
- Use enough points to accurately represent the shape (rectangles need 5 points, curved shapes need more)

FIELDS TO EXTRACT:

1. "version" — always set to 2
2. "width" — image width in pixels
3. "height" — image height in pixels
4. "pixels_per_meter" — calculate from dimension lines shown on the plan. Find a labeled dimension (e.g. "3.40" meters), measure its length in pixels, then divide pixels by meters.
5. "apartments" — single polygon tracing the outer boundary of the entire apartment (including all walls)
6. "wall" — array of polygons for wall contours. Each wall segment is a polygon tracing the wall outline. Include door/window cutouts in the wall polygons.
7. "door" — array of rectangular polygons marking door positions within walls
8. "window" — array of rectangular polygons marking window positions within walls
9. "balcony_window" — polygon(s) for windows/glass doors between apartment and balcony
10. "balcony" — polygon(s) for the balcony area outside the apartment
11. "bedroom" — polygon(s) for bedrooms (labeled as "BR", "MASTER BR", "Bedroom", or rooms with beds)
12. "living_room" — polygon for the living/dining room area (labeled as "LIVING", "DINING", or the main open area)
13. "other_room" — array of polygons for other rooms: bathrooms, WC, corridors, walk-in closets, laundry, powder rooms
14. "kitchen_table" — polygon(s) for kitchen countertops/work surfaces
15. "kitchen_zone" — polygon for the kitchen area if it's a separate zone, or null if kitchen is part of living room
16. "sink" — polygon for kitchen sink location
17. "cooker" — polygon for cooker/stove location

ROOM CLASSIFICATION RULES:
- Rooms with beds or labeled "BR"/"Bedroom"/"Master" → bedroom
- The largest open area or rooms labeled "Living"/"Dining" → living_room
- Bathrooms, WC, shower rooms, labeled "BATH"/"MBATH"/"PDR" → other_room
- Walk-in closets "WIC", laundry "L/S", corridors → other_room
- Kitchen area with appliances → kitchen_table/kitchen_zone/sink/cooker

IMPORTANT:
- Be precise with wall coordinates — walls have thickness, trace both inner and outer edges
- Doors are typically rectangular cutouts in walls
- Windows are typically rectangular elements on exterior walls
- If a field has multiple instances (e.g. 2 bedrooms), return an array of polygons
- If a field has one instance, it can be a single polygon (not wrapped in array)
- If kitchen_zone is not a separate room (open kitchen), set it to null`

export const RESPONSE_JSON_SCHEMA = {
  type: 'object',
  properties: {
    version: { type: 'integer', description: 'Schema version, always 2' },
    width: { type: 'number', description: 'Image width in pixels' },
    height: { type: 'number', description: 'Image height in pixels' },
    pixels_per_meter: { type: 'number', description: 'Scale: pixels per meter' },
    apartments: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Outer boundary polygon of the apartment'
    },
    wall: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of wall contour polygons'
    },
    door: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of door rectangle polygons'
    },
    window: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of window rectangle polygons'
    },
    balcony_window: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony window boundary polygon'
    },
    balcony: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Balcony area polygon'
    },
    bedroom: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of bedroom polygons'
    },
    living_room: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Living room polygon'
    },
    other_room: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Array of other room polygons (bathrooms, corridors, etc.)'
    },
    kitchen_table: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }
      },
      description: 'Kitchen countertop polygons'
    },
    kitchen_zone: {
      type: ['array', 'null'],
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Kitchen zone polygon or null if open kitchen'
    },
    sink: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Sink location polygon'
    },
    cooker: {
      type: 'array',
      items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      description: 'Cooker/stove location polygon'
    }
  },
  required: [
    'version', 'width', 'height', 'pixels_per_meter',
    'apartments', 'wall', 'door', 'window',
    'balcony_window', 'balcony', 'bedroom', 'living_room',
    'other_room', 'kitchen_table', 'sink', 'cooker'
  ]
}
