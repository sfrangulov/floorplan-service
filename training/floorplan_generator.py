"""Procedural floorplan layout generator for synthetic training data."""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Room:
    """A rectangular room in the floorplan."""

    room_type: str
    x: float
    y: float
    w: float
    h: float

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        """Return corners as [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]."""
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]

    @property
    def center(self) -> Tuple[float, float]:
        """Return center point of the room."""
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class Wall:
    """A wall segment."""

    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 3.0
    is_exterior: bool = False


@dataclass
class Door:
    """A door placed between adjacent rooms."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class Window:
    """A window on an exterior wall."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class Layout:
    """Complete floorplan layout with all elements."""

    width: int
    height: int
    rooms: List[Room] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    balcony: Optional[Room] = None
    balcony_window: Optional[Window] = None


def _rooms_adjacent_vertical(r1: Room, r2: Room, tol: float = 2.0) -> bool:
    """Check if r1 and r2 share a vertical boundary (side by side)."""
    # r1's right edge touches r2's left edge (or vice versa)
    if abs(r1.x + r1.w - r2.x) < tol or abs(r2.x + r2.w - r1.x) < tol:
        # Check vertical overlap
        overlap_top = max(r1.y, r2.y)
        overlap_bot = min(r1.y + r1.h, r2.y + r2.h)
        return overlap_bot - overlap_top > tol
    return False


def _rooms_adjacent_horizontal(r1: Room, r2: Room, tol: float = 2.0) -> bool:
    """Check if r1 and r2 share a horizontal boundary (stacked)."""
    # r1's bottom edge touches r2's top edge (or vice versa)
    if abs(r1.y + r1.h - r2.y) < tol or abs(r2.y + r2.h - r1.y) < tol:
        # Check horizontal overlap
        overlap_left = max(r1.x, r2.x)
        overlap_right = min(r1.x + r1.w, r2.x + r2.w)
        return overlap_right - overlap_left > tol
    return False


def _place_door_between(r1: Room, r2: Room, wall_t: float) -> Optional[Door]:
    """Place a door on the shared boundary between two adjacent rooms."""
    tol = 2.0
    door_size = random.uniform(20, 30)

    # Check vertical adjacency (shared vertical wall)
    if abs(r1.x + r1.w - r2.x) < tol:
        # r1 is left of r2
        shared_top = max(r1.y, r2.y)
        shared_bot = min(r1.y + r1.h, r2.y + r2.h)
        shared_len = shared_bot - shared_top
        if shared_len < door_size + 10:
            door_size = max(15, shared_len - 10)
        door_y = random.uniform(shared_top + 5, shared_bot - door_size - 5)
        return Door(
            x=r2.x - wall_t / 2,
            y=door_y,
            width=wall_t,
            height=door_size,
        )
    elif abs(r2.x + r2.w - r1.x) < tol:
        # r2 is left of r1
        shared_top = max(r1.y, r2.y)
        shared_bot = min(r1.y + r1.h, r2.y + r2.h)
        shared_len = shared_bot - shared_top
        if shared_len < door_size + 10:
            door_size = max(15, shared_len - 10)
        door_y = random.uniform(shared_top + 5, shared_bot - door_size - 5)
        return Door(
            x=r1.x - wall_t / 2,
            y=door_y,
            width=wall_t,
            height=door_size,
        )

    # Check horizontal adjacency (shared horizontal wall)
    if abs(r1.y + r1.h - r2.y) < tol:
        # r1 is above r2
        shared_left = max(r1.x, r2.x)
        shared_right = min(r1.x + r1.w, r2.x + r2.w)
        shared_len = shared_right - shared_left
        if shared_len < door_size + 10:
            door_size = max(15, shared_len - 10)
        door_x = random.uniform(shared_left + 5, shared_right - door_size - 5)
        return Door(
            x=door_x,
            y=r2.y - wall_t / 2,
            width=door_size,
            height=wall_t,
        )
    elif abs(r2.y + r2.h - r1.y) < tol:
        # r2 is above r1
        shared_left = max(r1.x, r2.x)
        shared_right = min(r1.x + r1.w, r2.x + r2.w)
        shared_len = shared_right - shared_left
        if shared_len < door_size + 10:
            door_size = max(15, shared_len - 10)
        door_x = random.uniform(shared_left + 5, shared_right - door_size - 5)
        return Door(
            x=door_x,
            y=r1.y - wall_t / 2,
            width=door_size,
            height=wall_t,
        )

    return None


def generate_layout(
    canvas_size: int = 512,
    num_bedrooms: Optional[int] = None,
    has_balcony: Optional[bool] = None,
) -> Layout:
    """Generate a random floorplan layout.

    Args:
        canvas_size: Size of the square canvas in pixels.
        num_bedrooms: Number of bedrooms (1-3). Random if None.
        has_balcony: Whether to include a balcony. Random if None.

    Returns:
        A Layout containing all floorplan elements.
    """
    margin = int(canvas_size * 0.08)
    wall_t = random.uniform(2.5, 5.0)

    if num_bedrooms is None:
        num_bedrooms = random.choice([1, 1, 2, 2, 2, 3])

    if has_balcony is None:
        has_balcony = random.random() < 0.5

    # Apartment bounding box
    apt_x = margin
    apt_w = canvas_size - 2 * margin

    if has_balcony:
        balcony_h = canvas_size * random.uniform(0.08, 0.14)
        apt_y = margin + int(canvas_size * 0.12)
    else:
        balcony_h = 0
        apt_y = margin

    apt_h = canvas_size - apt_y - margin

    # Split top/bottom rows
    top_frac = random.uniform(0.45, 0.55)
    top_h = apt_h * top_frac
    bot_h = apt_h - top_h

    rooms: List[Room] = []
    walls: List[Wall] = []
    doors: List[Door] = []
    windows: List[Window] = []

    # --- Top row: bedrooms + living room ---
    top_count = num_bedrooms + 1  # bedrooms + living_room
    # Generate random width fractions for top rooms
    fracs = [random.uniform(0.8, 1.2) for _ in range(top_count)]
    total = sum(fracs)
    fracs = [f / total for f in fracs]

    top_rooms: List[Room] = []
    cur_x = apt_x
    for i in range(top_count):
        room_w = apt_w * fracs[i]
        if i < num_bedrooms:
            rtype = "bedroom"
        else:
            rtype = "living_room"
        room = Room(
            room_type=rtype,
            x=cur_x,
            y=apt_y,
            w=room_w,
            h=top_h,
        )
        top_rooms.append(room)
        rooms.append(room)
        cur_x += room_w

    # --- Bottom row: kitchen + bathroom ---
    kitchen_frac = random.uniform(0.55, 0.75)
    bath_frac = 1.0 - kitchen_frac

    kitchen = Room(
        room_type="kitchen",
        x=apt_x,
        y=apt_y + top_h,
        w=apt_w * kitchen_frac,
        h=bot_h,
    )
    bathroom = Room(
        room_type="bathroom",
        x=apt_x + apt_w * kitchen_frac,
        y=apt_y + top_h,
        w=apt_w * bath_frac,
        h=bot_h,
    )
    rooms.append(kitchen)
    rooms.append(bathroom)
    bot_rooms = [kitchen, bathroom]

    # --- Exterior walls (4 sides of the apartment) ---
    # Top wall
    walls.append(Wall(apt_x, apt_y, apt_x + apt_w, apt_y,
                       thickness=wall_t, is_exterior=True))
    # Bottom wall
    walls.append(Wall(apt_x, apt_y + apt_h, apt_x + apt_w, apt_y + apt_h,
                       thickness=wall_t, is_exterior=True))
    # Left wall
    walls.append(Wall(apt_x, apt_y, apt_x, apt_y + apt_h,
                       thickness=wall_t, is_exterior=True))
    # Right wall
    walls.append(Wall(apt_x + apt_w, apt_y, apt_x + apt_w, apt_y + apt_h,
                       thickness=wall_t, is_exterior=True))

    # --- Interior walls ---
    # Horizontal wall between top and bottom rows
    walls.append(Wall(apt_x, apt_y + top_h, apt_x + apt_w, apt_y + top_h,
                       thickness=wall_t, is_exterior=False))

    # Vertical walls between top-row rooms
    for i in range(len(top_rooms) - 1):
        r = top_rooms[i]
        wx = r.x + r.w
        walls.append(Wall(wx, apt_y, wx, apt_y + top_h,
                           thickness=wall_t, is_exterior=False))

    # Vertical wall between kitchen and bathroom
    walls.append(Wall(
        bathroom.x, apt_y + top_h,
        bathroom.x, apt_y + apt_h,
        thickness=wall_t, is_exterior=False,
    ))

    # --- Doors between adjacent rooms ---
    all_rooms = top_rooms + bot_rooms
    placed_pairs = set()
    for i, r1 in enumerate(all_rooms):
        for j, r2 in enumerate(all_rooms):
            if i >= j:
                continue
            pair = (i, j)
            if pair in placed_pairs:
                continue
            if (_rooms_adjacent_vertical(r1, r2) or
                    _rooms_adjacent_horizontal(r1, r2)):
                door = _place_door_between(r1, r2, wall_t)
                if door is not None:
                    doors.append(door)
                    placed_pairs.add(pair)

    # --- Windows on exterior walls (skip bathrooms) ---
    for room in rooms:
        if room.room_type == "bathroom":
            continue

        win_frac = random.uniform(0.3, 0.6)

        # Top exterior wall - rooms in the top row touching the top edge
        if abs(room.y - apt_y) < 2.0:
            win_w = room.w * win_frac
            win_x = room.x + (room.w - win_w) / 2
            windows.append(Window(
                x=win_x,
                y=apt_y - wall_t / 2,
                width=win_w,
                height=wall_t,
            ))

        # Bottom exterior wall - rooms in the bottom row touching the bottom
        if abs(room.y + room.h - (apt_y + apt_h)) < 2.0:
            win_w = room.w * win_frac
            win_x = room.x + (room.w - win_w) / 2
            windows.append(Window(
                x=win_x,
                y=apt_y + apt_h - wall_t / 2,
                width=win_w,
                height=wall_t,
            ))

        # Left exterior wall
        if abs(room.x - apt_x) < 2.0:
            win_h = room.h * win_frac
            win_y = room.y + (room.h - win_h) / 2
            windows.append(Window(
                x=apt_x - wall_t / 2,
                y=win_y,
                width=wall_t,
                height=win_h,
            ))

        # Right exterior wall
        if abs(room.x + room.w - (apt_x + apt_w)) < 2.0:
            win_h = room.h * win_frac
            win_y = room.y + (room.h - win_h) / 2
            windows.append(Window(
                x=apt_x + apt_w - wall_t / 2,
                y=win_y,
                width=wall_t,
                height=win_h,
            ))

    # --- Balcony ---
    balcony_room = None
    balcony_window = None
    if has_balcony:
        balcony_room = Room(
            room_type="balcony",
            x=apt_x,
            y=apt_y - balcony_h,
            w=apt_w,
            h=balcony_h,
        )
        # Balcony window on the shared wall between balcony and apartment
        bw_thickness = wall_t * 1.5
        bw_width = apt_w * random.uniform(0.3, 0.6)
        bw_x = apt_x + (apt_w - bw_width) / 2
        balcony_window = Window(
            x=bw_x,
            y=apt_y - bw_thickness / 2,
            width=bw_width,
            height=bw_thickness,
        )

    layout = Layout(
        width=canvas_size,
        height=canvas_size,
        rooms=rooms,
        walls=walls,
        doors=doors,
        windows=windows,
        balcony=balcony_room,
        balcony_window=balcony_window,
    )

    return layout
