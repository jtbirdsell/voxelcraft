//! Block registry (M3). Blocks are `u16` ids; properties come from small tables.
//! Grows into a proper data-driven registry in later milestones.

pub type BlockId = u16;

pub const AIR: BlockId = 0;
pub const STONE: BlockId = 1;
pub const DIRT: BlockId = 2;
pub const GRASS: BlockId = 3;
pub const SAND: BlockId = 4;
pub const WOOD: BlockId = 5;
pub const LEAVES: BlockId = 6;
pub const WATER: BlockId = 7;
pub const SNOW: BlockId = 8;
pub const COAL_ORE: BlockId = 9;
pub const IRON_ORE: BlockId = 10;
pub const LAVA: BlockId = 11;
pub const TORCH: BlockId = 12;
pub const GLOWSTONE: BlockId = 13;
pub const COBBLESTONE: BlockId = 14;
pub const PLANKS: BlockId = 15;
pub const BRICKS: BlockId = 16;
pub const BEDROCK: BlockId = 17;
pub const GRAVEL: BlockId = 18;
pub const OBSIDIAN: BlockId = 19;
pub const GOLD_ORE: BlockId = 20;
pub const DIAMOND_ORE: BlockId = 21;
pub const REDSTONE_ORE: BlockId = 22;
pub const LAPIS_ORE: BlockId = 23;
pub const DEEPSLATE: BlockId = 24;
pub const CRAFTING_TABLE: BlockId = 25;
pub const FURNACE: BlockId = 26;
pub const CHEST: BlockId = 27;
pub const POPPY: BlockId = 28;
pub const DANDELION: BlockId = 29;
pub const TALL_GRASS: BlockId = 30;
pub const CACTUS: BlockId = 31;
// M25 decoration.
pub const FERN: BlockId = 32;
pub const RED_MUSHROOM: BlockId = 33;
pub const BROWN_MUSHROOM: BlockId = 34;
pub const SUGAR_CANE: BlockId = 35;
pub const PUMPKIN: BlockId = 36;
pub const ICE: BlockId = 37;
// M26 glass + partial blocks.
pub const GLASS: BlockId = 38;
pub const STONE_SLAB: BlockId = 39;
pub const STONE_STAIRS: BlockId = 40; // orientation (facing + half) lives in the block-state byte
pub const WOOD_SLAB: BlockId = 41;
// (ids 42-44 were the old fixed-orientation stair variants STONE_STAIRS_E/_S/_W; orientation is now a
// block-state byte (P2), so those ids were retired. P9 reuses 42-43 for the new interactive blocks.)
pub const WOODEN_DOOR: BlockId = 42; // 2-tall; state = facing+open+hinge+half
pub const WOODEN_TRAPDOOR: BlockId = 43; // 1x1 thin; state = facing+half+open
// Connection-aware blocks (P10): shape derived from neighbors at mesh time (post + per-side arms).
pub const WOODEN_FENCE: BlockId = 44;
pub const COBBLESTONE_WALL: BlockId = 45;
pub const GLASS_PANE: BlockId = 46;

/// Highest defined block id; bounds save-id validation (keep in sync as blocks are added).
pub const MAX_BLOCK: BlockId = GLASS_PANE;

#[inline]
pub fn is_fence(id: BlockId) -> bool {
    id == WOODEN_FENCE
}
#[inline]
pub fn is_wall(id: BlockId) -> bool {
    id == COBBLESTONE_WALL
}
#[inline]
pub fn is_pane(id: BlockId) -> bool {
    id == GLASS_PANE
}

/// Whether a connection-aware block (`self_id`) grows an arm toward a horizontal neighbor `n`.
/// Every family attaches to a SOLID full cube face; otherwise each connects only within its own
/// family (panes also to glass). `is_cube && is_opaque` excludes doors/trapdoors/slabs/stairs/glass.
pub fn connects(self_id: BlockId, n: BlockId) -> bool {
    if is_cube(n) && is_opaque(n) {
        return true;
    }
    match self_id {
        WOODEN_FENCE => is_fence(n),
        COBBLESTONE_WALL => is_wall(n),
        GLASS_PANE => is_pane(n) || n == GLASS,
        _ => false,
    }
}

/// Render dimensions of a connection block (post + arm rails). Collision is separate (a conservative
/// centered box in `solid_boxes`). Arms run from the post edge to the cell edge along the connection
/// axis, centered `arm_perp` on the other axis, at each `rails` (y_lo, y_hi).
pub struct ConnectDims {
    pub post: Aabb,
    pub arm_perp: (f32, f32),
    pub rails: &'static [(f32, f32)],
}
const FENCE_RAILS: [(f32, f32); 2] = [(0.375, 0.5625), (0.75, 0.9375)];
const WALL_RAILS: [(f32, f32); 1] = [(0.0, 0.8125)];
const PANE_RAILS: [(f32, f32); 1] = [(0.0, 1.0)];
pub fn connect_dims(id: BlockId) -> ConnectDims {
    match id {
        WOODEN_FENCE => ConnectDims {
            post: [0.3125, 0.0, 0.3125, 0.6875, 1.0, 0.6875],
            arm_perp: (0.4375, 0.5625),
            rails: &FENCE_RAILS,
        },
        COBBLESTONE_WALL => ConnectDims {
            post: [0.25, 0.0, 0.25, 0.75, 1.0, 0.75],
            arm_perp: (0.3125, 0.6875),
            rails: &WALL_RAILS,
        },
        _ => ConnectDims {
            // glass pane
            post: [0.4375, 0.0, 0.4375, 0.5625, 1.0, 0.5625],
            arm_perp: (0.4375, 0.5625),
            rails: &PANE_RAILS,
        },
    }
}
/// A lone glass pane (no connections) renders as a full-cell flat sheet (a window), not a nub.
pub const PANE_SHEET: Aabb = [0.0, 0.0, 0.4375, 1.0, 1.0, 0.5625];

/// Block-state layout for stairs: bits 0-1 = facing (0:+z, 1:+x, 2:-z, 3:-x — the direction the high
/// step faces); bit 2 (top half) is reserved for the stairs-polish milestone — placement currently
/// always makes bottom-half stairs, and corner shaping is derived at mesh time. Most blocks ignore
/// the state byte entirely (state 0).
#[inline]
pub fn stair_facing(state: u8) -> u8 {
    state & 0b11
}

/// Pack a stair block-state from a facing (0..3). (Half/shape arrive with the stairs-polish pass.)
#[inline]
pub fn stair_state(facing: u8) -> u8 {
    facing & 0b11
}

/// Slab block-state (bits 0-1): 0 = bottom half, 1 = top half, 2 = double (a full block formed by
/// placing a matching slab into the empty half). A reserved value (3) decodes as bottom.
pub const SLAB_BOTTOM: u8 = 0;
pub const SLAB_TOP: u8 = 1;
pub const SLAB_DOUBLE: u8 = 2;

#[inline]
pub fn slab_half(state: u8) -> u8 {
    match state & 0b11 {
        1 => SLAB_TOP,
        2 => SLAB_DOUBLE,
        _ => SLAB_BOTTOM,
    }
}

#[inline]
pub fn slab_state(half: u8) -> u8 {
    half & 0b11
}

/// Log axis (bits 0-1 of the WOOD state byte): 0 = Y upright (DEFAULT = state 0, so worldgen trunks +
/// legacy saves stay upright with no migration), 1 = X (east-west), 2 = Z (north-south). The end-grain
/// (WOOD_TOP) shows on the two faces perpendicular to the axis; bark (WOOD_SIDE) on the other four.
pub const AXIS_Y: u8 = 0;
pub const AXIS_X: u8 = 1;
pub const AXIS_Z: u8 = 2;

#[inline]
pub fn log_axis(state: u8) -> u8 {
    match state & 0b11 {
        1 => AXIS_X,
        2 => AXIS_Z,
        _ => AXIS_Y,
    }
}

#[inline]
pub fn log_state(axis: u8) -> u8 {
    axis & 0b11
}

// Door state byte: bits 0-1 facing (0:+z 1:+x 2:-z 3:-x — the wall the closed panel sits against),
// bit 2 open, bit 3 hinge (0 left, 1 right), bit 4 half (0 lower, 1 upper). Both stacked cells carry
// the same facing/hinge/open; the half bit distinguishes them so each cell is self-describing.
pub const DOOR_LOWER: u8 = 0;
pub const DOOR_UPPER: u8 = 1;
#[inline]
pub fn door_facing(state: u8) -> u8 {
    state & 0b11
}
#[inline]
pub fn door_open(state: u8) -> bool {
    state & 0b100 != 0
}
#[inline]
pub fn door_hinge(state: u8) -> u8 {
    (state >> 3) & 1
}
#[inline]
pub fn door_half(state: u8) -> u8 {
    (state >> 4) & 1
}
#[inline]
pub fn door_state(facing: u8, open: bool, hinge: u8, half: u8) -> u8 {
    (facing & 0b11) | ((open as u8) << 2) | ((hinge & 1) << 3) | ((half & 1) << 4)
}

// Trapdoor state byte: bits 0-1 facing, bit 2 half (0 bottom, 1 top), bit 3 open.
#[inline]
pub fn trapdoor_facing(state: u8) -> u8 {
    state & 0b11
}
#[inline]
pub fn trapdoor_half(state: u8) -> u8 {
    (state >> 2) & 1
}
#[inline]
pub fn trapdoor_open(state: u8) -> bool {
    state & 0b1000 != 0
}
#[inline]
pub fn trapdoor_state(facing: u8, half: u8, open: bool) -> u8 {
    (facing & 0b11) | ((half & 1) << 2) | ((open as u8) << 3)
}

/// How a block is meshed: a full greedy cube, a cross billboard (plants), or a non-greedy partial
/// shape (slab / stairs) emitted one cell at a time like a billboard.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderKind {
    Cube,
    Cross,
    Slab,
    Stairs,
    /// 2-tall door panel (thin, oriented by facing/hinge/open). Emitted per-cell.
    Door,
    /// 1x1 thin trapdoor flap (oriented by facing/half/open). Emitted per-cell.
    Trapdoor,
    /// Fence / wall / glass-pane: a post + arms toward connecting neighbors (derived at mesh time).
    Connect,
}

#[inline]
pub fn render_kind(id: BlockId) -> RenderKind {
    match id {
        POPPY | DANDELION | TALL_GRASS | FERN | RED_MUSHROOM | BROWN_MUSHROOM | SUGAR_CANE => {
            RenderKind::Cross
        }
        STONE_SLAB | WOOD_SLAB => RenderKind::Slab,
        STONE_STAIRS => RenderKind::Stairs,
        WOODEN_DOOR => RenderKind::Door,
        WOODEN_TRAPDOOR => RenderKind::Trapdoor,
        WOODEN_FENCE | COBBLESTONE_WALL | GLASS_PANE => RenderKind::Connect,
        _ => RenderKind::Cube,
    }
}

/// A non-solid cross-billboard plant (walk-through, casts no shadow, doesn't cull neighbors).
#[inline]
pub fn is_plant(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Cross)
}

/// A partial-geometry block (slab/stairs): solid for collision but emitted per-cell, not greedy.
#[inline]
pub fn is_partial(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Slab | RenderKind::Stairs)
}

/// Translucent glass: a full cube that renders in its own alpha-blended pass and never blocks light.
#[inline]
pub fn is_glass(id: BlockId) -> bool {
    id == GLASS
}

/// A full greedy-meshed cube (cubes incl. glass; excludes cross plants and partials).
#[inline]
pub fn is_cube(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Cube)
}

/// Whether a block occupies the ray-traced voxel volume as a full cube (casts shadows / AO). Glass
/// is see-through (stored as 0); slabs/stairs are approximated as full cubes (a DOUBLE slab genuinely
/// IS a full cube; a single slab is over-conservative here — per-state sky-transparency is a deferred
/// refinement, kept id-only so the volume upload / light flood don't need the state byte).
#[inline]
pub fn is_volume_solid(id: BlockId) -> bool {
    is_opaque(id) || is_partial(id)
}

/// A block participates in collision (fluids and cross-billboard plants are passable).
#[inline]
pub fn is_solid(id: BlockId) -> bool {
    id != AIR && id != WATER && id != LAVA && !is_plant(id)
}

/// A solid sub-box in a block's local 0..1 space: `[minx, miny, minz, maxx, maxy, maxz]`.
pub type Aabb = [f32; 6];
const BOX_FULL: [Aabb; 1] = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]];
const BOX_SLAB: [Aabb; 1] = [[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]]; // bottom half
const BOX_SLAB_TOP: [Aabb; 1] = [[0.0, 0.5, 0.0, 1.0, 1.0, 1.0]]; // top half
const BOX_NONE: [Aabb; 0] = [];
// Stairs: a bottom slab + an upper half-box on the high side, one per facing (0:+z 1:+x 2:-z 3:-x).
const SLAB_HALF: Aabb = [0.0, 0.0, 0.0, 1.0, 0.5, 1.0];
const BOX_STAIRS_N: [Aabb; 2] = [SLAB_HALF, [0.0, 0.5, 0.5, 1.0, 1.0, 1.0]];
const BOX_STAIRS_E: [Aabb; 2] = [SLAB_HALF, [0.5, 0.5, 0.0, 1.0, 1.0, 1.0]];
const BOX_STAIRS_S: [Aabb; 2] = [SLAB_HALF, [0.0, 0.5, 0.0, 1.0, 1.0, 0.5]];
const BOX_STAIRS_W: [Aabb; 2] = [SLAB_HALF, [0.0, 0.5, 0.0, 0.5, 1.0, 1.0]];

// Door / trapdoor panels are 3/16 thick. The same box drives BOTH render (mesher) and collision
// (solid_boxes), so an open door's swung side-panel still blocks while the doorway opens up.
const DT: f32 = 0.1875; // thickness
const DTI: f32 = 0.8125; // 1.0 - DT
// Closed door panel, flush against the facing wall (0:+z 1:+x 2:-z 3:-x); hinge-independent.
const DOOR_CLOSED: [[Aabb; 1]; 4] = [
    [[0.0, 0.0, DTI, 1.0, 1.0, 1.0]], // +Z
    [[DTI, 0.0, 0.0, 1.0, 1.0, 1.0]], // +X
    [[0.0, 0.0, 0.0, 1.0, 1.0, DT]],  // -Z
    [[0.0, 0.0, 0.0, DT, 1.0, 1.0]],  // -X
];
// Open door panel, swung 90° to a perpendicular wall on the hinge side. [facing][hinge].
const DOOR_OPEN: [[[Aabb; 1]; 2]; 4] = [
    [[[0.0, 0.0, 0.0, DT, 1.0, 1.0]], [[DTI, 0.0, 0.0, 1.0, 1.0, 1.0]]], // +Z -> X wall
    [[[0.0, 0.0, 0.0, 1.0, 1.0, DT]], [[0.0, 0.0, DTI, 1.0, 1.0, 1.0]]], // +X -> Z wall
    [[[0.0, 0.0, 0.0, DT, 1.0, 1.0]], [[DTI, 0.0, 0.0, 1.0, 1.0, 1.0]]], // -Z -> X wall
    [[[0.0, 0.0, 0.0, 1.0, 1.0, DT]], [[0.0, 0.0, DTI, 1.0, 1.0, 1.0]]], // -X -> Z wall
];
// Trapdoor: closed flat flap (bottom/top of the cell) + open vertical flap against the facing wall.
const TRAP_BOTTOM: [Aabb; 1] = [[0.0, 0.0, 0.0, 1.0, DT, 1.0]];
const TRAP_TOP: [Aabb; 1] = [[0.0, DTI, 0.0, 1.0, 1.0, 1.0]];
const TRAP_OPEN: [[Aabb; 1]; 4] = [
    [[0.0, 0.0, DTI, 1.0, 1.0, 1.0]], // +Z
    [[DTI, 0.0, 0.0, 1.0, 1.0, 1.0]], // +X
    [[0.0, 0.0, 0.0, 1.0, 1.0, DT]],  // -Z
    [[0.0, 0.0, 0.0, DT, 1.0, 1.0]],  // -X
];
// Connection-block COLLISION (separate from the thin render geometry): a centered 0.5-wide box so a
// fence/wall/pane LINE can't be slipped through (0.5 gap < the 0.6-wide player), 1.5 tall for
// fences/walls (unjumpable) and 1.0 for panes. A lone post is thus slightly over-conservative.
const FENCE_COLLIDE: [Aabb; 1] = [[0.25, 0.0, 0.25, 0.75, 1.5, 0.75]];
const PANE_COLLIDE: [Aabb; 1] = [[0.25, 0.0, 0.25, 0.75, 1.0, 0.75]];

/// The upper-step box (above the bottom slab) of a stair, given its facing.
#[inline]
pub fn stair_upper_box(facing: u8) -> Aabb {
    match facing {
        1 => BOX_STAIRS_E[1],
        2 => BOX_STAIRS_S[1],
        3 => BOX_STAIRS_W[1],
        _ => BOX_STAIRS_N[1],
    }
}

/// The solid collision boxes of a block (empty if passable). Full cubes are a single unit box;
/// slabs/stairs return their true partial shape (stairs per the facing in `state`) so the player
/// can stand at half-height / step up.
#[inline]
pub fn solid_boxes(id: BlockId, state: u8) -> &'static [Aabb] {
    if !is_solid(id) {
        return &BOX_NONE;
    }
    match render_kind(id) {
        RenderKind::Slab => match slab_half(state) {
            SLAB_TOP => &BOX_SLAB_TOP,
            SLAB_DOUBLE => &BOX_FULL,
            _ => &BOX_SLAB,
        },
        RenderKind::Stairs => match stair_facing(state) {
            1 => &BOX_STAIRS_E,
            2 => &BOX_STAIRS_S,
            3 => &BOX_STAIRS_W,
            _ => &BOX_STAIRS_N,
        },
        // Door/trapdoor: thin panel — collision == the visible geometry. An open door's swung panel
        // still blocks its thin side, leaving the doorway walkable.
        RenderKind::Door => {
            if door_open(state) {
                &DOOR_OPEN[door_facing(state) as usize][door_hinge(state) as usize]
            } else {
                &DOOR_CLOSED[door_facing(state) as usize]
            }
        }
        RenderKind::Trapdoor => {
            if trapdoor_open(state) {
                &TRAP_OPEN[trapdoor_facing(state) as usize]
            } else if trapdoor_half(state) == 1 {
                &TRAP_TOP
            } else {
                &TRAP_BOTTOM
            }
        }
        // Fence/wall/pane: a conservative centered collision box (render geometry is the thin
        // post+arms, emitted separately by the mesher).
        RenderKind::Connect => {
            if id == GLASS_PANE {
                &PANE_COLLIDE
            } else {
                &FENCE_COLLIDE
            }
        }
        _ => &BOX_FULL,
    }
}

/// Water or lava — simulated by the flowing-fluid tick and passable to the player.
#[inline]
pub fn is_fluid(id: BlockId) -> bool {
    id == WATER || id == LAVA
}

/// A block fully hides the touching face of an adjacent opaque block.
/// Water and cross-billboard plants are non-opaque; leaves stay opaque (rendered as solid foliage).
#[inline]
pub fn is_opaque(id: BlockId) -> bool {
    id != AIR
        && id != WATER
        && id != GLASS
        && id != WOODEN_DOOR
        && id != WOODEN_TRAPDOOR
        && id != WOODEN_FENCE
        && id != COBBLESTONE_WALL
        && id != GLASS_PANE
        && !is_plant(id)
        && !is_partial(id)
}

/// Whether a block produces any geometry at all.
#[inline]
pub fn renders(id: BlockId) -> bool {
    id != AIR
}

#[inline]
pub fn is_water(id: BlockId) -> bool {
    id == WATER
}

/// Does `neighbor` hide the face of `self_id` that touches it?
/// Opaque neighbors hide everything; a translucent block hides only the same translucent
/// type (so water surfaces show against air/solids but internal water faces are culled).
#[inline]
pub fn occludes(self_id: BlockId, neighbor: BlockId) -> bool {
    if neighbor == AIR {
        false
    } else if is_opaque(neighbor) {
        true
    } else {
        self_id == neighbor
    }
}

/// Human-readable block name (hotbar label, tooltips, F3).
pub fn display_name(id: BlockId) -> &'static str {
    match id {
        AIR => "Air",
        STONE => "Stone",
        DIRT => "Dirt",
        GRASS => "Grass Block",
        SAND => "Sand",
        WOOD => "Wood",
        LEAVES => "Leaves",
        WATER => "Water",
        SNOW => "Snow",
        COAL_ORE => "Coal Ore",
        IRON_ORE => "Iron Ore",
        LAVA => "Lava",
        TORCH => "Torch",
        GLOWSTONE => "Glowstone",
        COBBLESTONE => "Cobblestone",
        PLANKS => "Wood Planks",
        BRICKS => "Bricks",
        BEDROCK => "Bedrock",
        GRAVEL => "Gravel",
        OBSIDIAN => "Obsidian",
        GOLD_ORE => "Gold Ore",
        DIAMOND_ORE => "Diamond Ore",
        REDSTONE_ORE => "Redstone Ore",
        LAPIS_ORE => "Lapis Ore",
        DEEPSLATE => "Deepslate",
        CRAFTING_TABLE => "Crafting Table",
        FURNACE => "Furnace",
        CHEST => "Chest",
        POPPY => "Poppy",
        DANDELION => "Dandelion",
        TALL_GRASS => "Tall Grass",
        CACTUS => "Cactus",
        FERN => "Fern",
        RED_MUSHROOM => "Red Mushroom",
        BROWN_MUSHROOM => "Brown Mushroom",
        SUGAR_CANE => "Sugar Cane",
        PUMPKIN => "Pumpkin",
        ICE => "Ice",
        GLASS => "Glass",
        STONE_SLAB => "Stone Slab",
        STONE_STAIRS => "Stone Stairs",
        WOOD_SLAB => "Wood Slab",
        WOODEN_DOOR => "Wooden Door",
        WOODEN_TRAPDOOR => "Wooden Trapdoor",
        WOODEN_FENCE => "Wooden Fence",
        COBBLESTONE_WALL => "Cobblestone Wall",
        GLASS_PANE => "Glass Pane",
        _ => "Unknown",
    }
}

/// Base albedo for a block face. `face_offset[1] == 1` is the +Y (top) face.
pub fn face_color(id: BlockId, face_offset: [i32; 3]) -> [f32; 3] {
    let top = face_offset[1] == 1;
    let bottom = face_offset[1] == -1;
    match id {
        GRASS => {
            if top {
                [0.36, 0.60, 0.27]
            } else if bottom {
                [0.45, 0.33, 0.21]
            } else {
                // side: dirt with a thin grass lip looks busy; keep dirt-ish with green tint
                [0.42, 0.42, 0.24]
            }
        }
        DIRT => [0.45, 0.33, 0.21],
        STONE => [0.49, 0.49, 0.52],
        SAND => [0.80, 0.75, 0.52],
        WOOD => {
            if top || bottom {
                [0.55, 0.43, 0.27]
            } else {
                [0.40, 0.30, 0.18]
            }
        }
        LEAVES => [0.20, 0.42, 0.18],
        WATER => [0.16, 0.34, 0.62],
        SNOW => [0.92, 0.94, 0.97],
        COAL_ORE => [0.28, 0.28, 0.30],
        IRON_ORE => [0.60, 0.52, 0.45],
        LAVA => [1.0, 0.42, 0.06],
        TORCH => [0.85, 0.62, 0.28],
        GLOWSTONE => [0.95, 0.82, 0.45],
        COBBLESTONE => [0.42, 0.42, 0.44],
        PLANKS => [0.62, 0.48, 0.30],
        BRICKS => [0.55, 0.28, 0.22],
        BEDROCK => [0.20, 0.20, 0.22],
        GRAVEL => [0.50, 0.47, 0.45],
        OBSIDIAN => [0.12, 0.09, 0.18],
        GOLD_ORE => [0.55, 0.50, 0.35],
        DIAMOND_ORE => [0.45, 0.62, 0.62],
        REDSTONE_ORE => [0.45, 0.30, 0.30],
        LAPIS_ORE => [0.30, 0.35, 0.55],
        DEEPSLATE => [0.22, 0.22, 0.25],
        CRAFTING_TABLE => [0.50, 0.36, 0.22],
        FURNACE => [0.38, 0.38, 0.40],
        CHEST => [0.55, 0.42, 0.24],
        POPPY => [0.80, 0.15, 0.12],
        DANDELION => [0.90, 0.82, 0.20],
        TALL_GRASS => [0.34, 0.55, 0.24],
        CACTUS => [0.30, 0.52, 0.24],
        FERN => [0.27, 0.46, 0.20],
        RED_MUSHROOM => [0.80, 0.20, 0.18],
        BROWN_MUSHROOM => [0.55, 0.40, 0.28],
        SUGAR_CANE => [0.55, 0.72, 0.40],
        PUMPKIN => {
            if top || bottom {
                [0.80, 0.52, 0.14] // mirror face_tile (top tile on both caps) for the GI invariant
            } else {
                [0.78, 0.48, 0.12]
            }
        }
        ICE => [0.66, 0.80, 0.92],
        GLASS => [0.82, 0.91, 0.98],
        STONE_SLAB | STONE_STAIRS => [0.49, 0.49, 0.52], // stone
        WOODEN_DOOR | WOODEN_TRAPDOOR | WOODEN_FENCE => [0.60, 0.46, 0.28], // wood
        COBBLESTONE_WALL => [0.42, 0.42, 0.44],                            // cobble
        GLASS_PANE => [0.82, 0.91, 0.98],                                  // glass
        WOOD_SLAB => [0.62, 0.48, 0.30], // planks
        _ => [1.0, 0.0, 1.0],
    }
}

/// Self-emission strength per block (0 = unlit material). Lava glows; the lit shaders add this as
/// extra outgoing radiance, and GI/reflection rays treat an emissive hit as a light source.
pub fn emission(id: BlockId) -> f32 {
    match id {
        LAVA => 1.0,
        TORCH => 0.9,
        GLOWSTONE => 1.0,
        _ => 0.0,
    }
}

/// Block-light emitted (0..15), for the light flood (M14b). Torch/glowstone/lava are sources.
pub fn light_emission(id: BlockId) -> u8 {
    match id {
        GLOWSTONE => 15,
        TORCH => 14,
        LAVA => 11,
        _ => 0,
    }
}

/// Light lost when entering this block (1 for air/transparent; 15 fully blocks). Opaque solids stop
/// light; fluids and (later) glass/foliage let it pass with light attenuation.
/// Reserved for graduated light propagation (the BFS currently uses a uniform step).
#[allow(dead_code)]
pub fn light_attenuation(id: BlockId) -> u8 {
    match id {
        AIR => 1,
        WATER => 2,
        _ if is_opaque(id) => 15,
        _ => 1,
    }
}

/// Whether light can pass through this block at all. Reserved for the graduated light pass.
#[allow(dead_code)]
#[inline]
pub fn transmits_light(id: BlockId) -> bool {
    !is_opaque(id)
}

/// Whether this block stops skylight descending. Solid terrain, roofs, and partial blocks (slabs/
/// stairs) do; glass and foliage (leaves) let it through so windows/tree canopies don't cast
/// pitch-black ground shadows (binary skylight, M14). Uses `is_volume_solid` (= opaque cubes +
/// partials) so slab/stair roofs still darken what's under them (M26 made `is_opaque` exclude them).
#[inline]
pub fn blocks_skylight(id: BlockId) -> bool {
    is_volume_solid(id) && id != LEAVES
}

/// The tool that mines a block fastest (for mining-speed + drop gating from M19).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolClass {
    None,
    Pickaxe,
    Axe,
    Shovel,
    Sword,
    Hoe,
}

/// Whether a block needs the correct tool to yield a drop (pickaxe blocks do; wood/dirt drop by hand).
pub fn requires_tool(id: BlockId) -> bool {
    matches!(tool_class(id), ToolClass::Pickaxe)
}

/// Minimum harvest level needed to get a drop (0 = any pickaxe). Iron/lapis need stone, gold/diamond/
/// redstone need iron, obsidian needs diamond.
pub fn required_harvest(id: BlockId) -> u8 {
    match id {
        IRON_ORE | LAPIS_ORE => 1,
        GOLD_ORE | DIAMOND_ORE | REDSTONE_ORE => 2,
        OBSIDIAN => 3,
        _ => 0,
    }
}

/// Time in seconds to break a block by hand (tools speed this up in M19). INFINITY = unbreakable.
pub fn hardness(id: BlockId) -> f32 {
    match id {
        BEDROCK => f32::INFINITY,
        AIR | WATER | LAVA => 0.0,
        LEAVES | TORCH | GLOWSTONE => 0.3,
        DIRT | GRASS | SAND | GRAVEL | SNOW => 0.6,
        WOOD | PLANKS | CRAFTING_TABLE | CHEST | WOODEN_DOOR | WOODEN_TRAPDOOR | WOODEN_FENCE => 1.2,
        COBBLESTONE_WALL => 2.0,
        STONE | COBBLESTONE | BRICKS | COAL_ORE | IRON_ORE | GOLD_ORE | DIAMOND_ORE
        | REDSTONE_ORE | LAPIS_ORE => 1.5,
        DEEPSLATE | FURNACE => 2.0,
        OBSIDIAN => 8.0,
        ICE => 0.5,
        PUMPKIN => 1.0,
        GLASS | GLASS_PANE => 0.3,
        STONE_SLAB | STONE_STAIRS => 1.5,
        WOOD_SLAB => 1.2,
        _ if is_plant(id) => 0.0,
        _ => 1.0,
    }
}

/// The tool class that mines a block fastest (drives tool speed + drop gating).
pub fn tool_class(id: BlockId) -> ToolClass {
    match id {
        STONE | COBBLESTONE | BRICKS | DEEPSLATE | OBSIDIAN | FURNACE | COAL_ORE | IRON_ORE
        | GOLD_ORE | DIAMOND_ORE | REDSTONE_ORE | LAPIS_ORE => ToolClass::Pickaxe,
        ICE | STONE_SLAB | STONE_STAIRS => ToolClass::Pickaxe,
        WOOD | PLANKS | CRAFTING_TABLE | CHEST | PUMPKIN | WOOD_SLAB | WOODEN_DOOR
        | WOODEN_TRAPDOOR | WOODEN_FENCE => ToolClass::Axe,
        COBBLESTONE_WALL => ToolClass::Pickaxe,
        DIRT | GRASS | SAND | GRAVEL | SNOW => ToolClass::Shovel,
        _ => ToolClass::None,
    }
}

/// Whether a block can be broken at all (bedrock and air can't).
#[inline]
pub fn breakable(id: BlockId) -> bool {
    id != AIR && hardness(id).is_finite()
}

/// What a broken block yields as an item drop + count (None = nothing). Ores drop their MATERIAL
/// item (coal/diamond/redstone/lapis directly; iron/gold as raw ore that smelts to an ingot); stone
/// yields cobblestone; grass yields dirt; leaves/ice/glass drop nothing; most blocks drop themselves.
/// Tool gating (the right pickaxe tier) is enforced by the caller. Note: `ItemId == BlockId == u16`,
/// so block-item and material ids mix freely here.
pub fn drops(id: BlockId, state: u8) -> Option<(crate::item::ItemId, u8)> {
    use crate::item;
    Some(match id {
        AIR => return None,
        STONE => (COBBLESTONE, 1),
        GRASS => (DIRT, 1),
        LEAVES => return None, // saplings/apples arrive with tree variety + farming
        ICE => return None,             // melts away (no silk touch yet)
        GLASS | GLASS_PANE => return None, // shatters
        // A double slab drops two slab items; a single drops one.
        STONE_SLAB | WOOD_SLAB => (id, if slab_half(state) == SLAB_DOUBLE { 2 } else { 1 }),
        COAL_ORE => (item::COAL, 1),
        IRON_ORE => (item::RAW_IRON, 1),
        GOLD_ORE => (item::RAW_GOLD, 1),
        DIAMOND_ORE => (item::DIAMOND, 1),
        REDSTONE_ORE => (item::REDSTONE_DUST, 4),
        LAPIS_ORE => (item::LAPIS, 6),
        _ => (id, 1),
    })
}

/// Experience awarded for mining a block (the ores that drop XP in Minecraft). Iron/gold give none
/// here — their XP comes from smelting the ore in a furnace.
pub fn mining_xp(id: BlockId) -> u32 {
    match id {
        COAL_ORE => 1,
        REDSTONE_ORE | LAPIS_ORE => 2,
        DIAMOND_ORE => 4,
        _ => 0,
    }
}

/// Atlas tile ids (M13). Index = `row*ATLAS_COLS + col` into the procedural texture atlas. Kept
/// beside `face_color`/`face_tile` so the painters (texture.rs) and the WGSL `tile_average()` stay
/// in lockstep. The 0..63 space leaves ample room for the block expansion in M16.
pub mod tile {
    pub const STONE: u32 = 0;
    pub const DIRT: u32 = 1;
    pub const GRASS_TOP: u32 = 2;
    pub const GRASS_SIDE: u32 = 3;
    pub const SAND: u32 = 4;
    pub const WOOD_TOP: u32 = 5;
    pub const WOOD_SIDE: u32 = 6;
    pub const LEAVES: u32 = 7;
    pub const WATER: u32 = 8;
    pub const SNOW: u32 = 9;
    pub const COAL: u32 = 10;
    pub const IRON: u32 = 11;
    pub const LAVA: u32 = 12;
    pub const MOB: u32 = 13;
    pub const MOB_HEAD: u32 = 14;
    pub const TORCH: u32 = 15;
    pub const GLOWSTONE: u32 = 16;
    pub const COBBLE: u32 = 17;
    pub const PLANKS: u32 = 18;
    pub const BRICKS: u32 = 19;
    pub const BEDROCK: u32 = 20;
    pub const GRAVEL: u32 = 21;
    pub const OBSIDIAN: u32 = 22;
    pub const GOLD: u32 = 23;
    pub const DIAMOND: u32 = 24;
    pub const REDSTONE: u32 = 25;
    pub const LAPIS: u32 = 26;
    pub const DEEPSLATE: u32 = 27;
    pub const CRAFTING: u32 = 28;
    pub const FURNACE: u32 = 29;
    pub const CHEST: u32 = 30;
    pub const POPPY: u32 = 31;
    pub const DANDELION: u32 = 32;
    pub const TALL_GRASS: u32 = 33;
    pub const CACTUS: u32 = 34;
    pub const FERN: u32 = 35;
    pub const RED_MUSHROOM: u32 = 36;
    pub const BROWN_MUSHROOM: u32 = 37;
    pub const SUGAR_CANE: u32 = 38;
    pub const PUMPKIN_TOP: u32 = 39;
    pub const PUMPKIN_SIDE: u32 = 40;
    pub const ICE: u32 = 41;
    pub const GLASS: u32 = 42;
    // M27 typed-mob body colors.
    pub const MOB_COW: u32 = 43;
    pub const MOB_PIG: u32 = 44;
    pub const MOB_SHEEP: u32 = 45;
    pub const MOB_CHICKEN: u32 = 46;
    pub const MOB_ZOMBIE: u32 = 47;
    pub const MOB_SKELETON: u32 = 48;
    pub const MOB_CREEPER: u32 = 49;
    pub const MOB_SPIDER: u32 = 50;
    /// Base of 11 flat-color tiles (51..=61) for mob-drop materials (item.rs BEEF..=ROTTEN_FLESH),
    /// so dropped loot renders in its real color instead of a generic wooden box.
    pub const MATERIAL_DROP: u32 = 51;
    pub const MAGENTA: u32 = 63; // missing/unknown sentinel
}

/// Tint class for a face: 0 = use texel as-is, 1 = multiply by foliage (grass/leaves) biome tint,
/// 2 = water tint. Carried in the vertex `shade.y`; applied in-shader from M13d (biome tint) on.
pub fn tint_class(id: BlockId, _face_offset: [i32; 3]) -> f32 {
    match id {
        GRASS | LEAVES => 1.0,
        WATER => 2.0,
        _ => 0.0,
    }
}

/// Atlas tile for a block face. Mirrors `face_color`'s per-face logic (grass top/side/bottom,
/// wood end/side) so `tile_average(face_tile(id, face))` reproduces `face_color(id, face)`.
pub fn face_tile(id: BlockId, face_offset: [i32; 3]) -> u32 {
    let top = face_offset[1] == 1;
    let bottom = face_offset[1] == -1;
    match id {
        GRASS => {
            if top {
                tile::GRASS_TOP
            } else if bottom {
                tile::DIRT
            } else {
                tile::GRASS_SIDE
            }
        }
        DIRT => tile::DIRT,
        STONE => tile::STONE,
        SAND => tile::SAND,
        WOOD => {
            if top || bottom {
                tile::WOOD_TOP
            } else {
                tile::WOOD_SIDE
            }
        }
        LEAVES => tile::LEAVES,
        WATER => tile::WATER,
        SNOW => tile::SNOW,
        COAL_ORE => tile::COAL,
        IRON_ORE => tile::IRON,
        LAVA => tile::LAVA,
        TORCH => tile::TORCH,
        GLOWSTONE => tile::GLOWSTONE,
        COBBLESTONE => tile::COBBLE,
        PLANKS => tile::PLANKS,
        BRICKS => tile::BRICKS,
        BEDROCK => tile::BEDROCK,
        GRAVEL => tile::GRAVEL,
        OBSIDIAN => tile::OBSIDIAN,
        GOLD_ORE => tile::GOLD,
        DIAMOND_ORE => tile::DIAMOND,
        REDSTONE_ORE => tile::REDSTONE,
        LAPIS_ORE => tile::LAPIS,
        DEEPSLATE => tile::DEEPSLATE,
        CRAFTING_TABLE => tile::CRAFTING,
        FURNACE => tile::FURNACE,
        CHEST => tile::CHEST,
        POPPY => tile::POPPY,
        DANDELION => tile::DANDELION,
        TALL_GRASS => tile::TALL_GRASS,
        CACTUS => tile::CACTUS,
        FERN => tile::FERN,
        RED_MUSHROOM => tile::RED_MUSHROOM,
        BROWN_MUSHROOM => tile::BROWN_MUSHROOM,
        SUGAR_CANE => tile::SUGAR_CANE,
        PUMPKIN => {
            if top || bottom {
                tile::PUMPKIN_TOP
            } else {
                tile::PUMPKIN_SIDE
            }
        }
        ICE => tile::ICE,
        GLASS => tile::GLASS,
        STONE_SLAB | STONE_STAIRS => tile::STONE,
        WOOD_SLAB | WOODEN_DOOR | WOODEN_TRAPDOOR | WOODEN_FENCE => tile::PLANKS,
        COBBLESTONE_WALL => tile::COBBLE,
        GLASS_PANE => tile::GLASS,
        _ => tile::MAGENTA,
    }
}

/// Atlas tile for a block face accounting for log axis (WOOD only): the end-grain (WOOD_TOP) lands on
/// the two faces perpendicular to the log's axis, bark (WOOD_SIDE) on the other four. Every other
/// block ignores `axis` and defers to `face_tile`. Called by the mesher; icons use `face_tile`
/// (default axis Y), which keeps the inventory log looking upright.
pub fn log_face_tile(id: BlockId, face_offset: [i32; 3], axis: u8) -> u32 {
    if id == WOOD {
        // Map the axis constant (Y=0, X=1, Z=2) to the face_offset component it runs parallel to.
        let comp = match axis {
            AXIS_X => 0,
            AXIS_Z => 2,
            _ => 1, // AXIS_Y
        };
        return if face_offset[comp] != 0 {
            tile::WOOD_TOP
        } else {
            tile::WOOD_SIDE
        };
    }
    face_tile(id, face_offset)
}
