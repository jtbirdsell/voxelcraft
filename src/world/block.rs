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
// Attachment-face fixtures (P11): small per-cell models mounted on a floor or wall face. Walk-through
// and non-occluding; the torch keeps its block-light glow (id-keyed). state = attach face + on/pressed.
pub const LEVER: BlockId = 47;
pub const BUTTON: BlockId = 48;
// ── Underground overhaul (U2): copper + emerald ores, and a deepslate variant of every ore ──
pub const COPPER_ORE: BlockId = 49;
pub const EMERALD_ORE: BlockId = 50;
pub const DEEPSLATE_COAL_ORE: BlockId = 51;
pub const DEEPSLATE_IRON_ORE: BlockId = 52;
pub const DEEPSLATE_COPPER_ORE: BlockId = 53;
pub const DEEPSLATE_GOLD_ORE: BlockId = 54;
pub const DEEPSLATE_REDSTONE_ORE: BlockId = 55;
pub const DEEPSLATE_EMERALD_ORE: BlockId = 56;
pub const DEEPSLATE_LAPIS_ORE: BlockId = 57;
pub const DEEPSLATE_DIAMOND_ORE: BlockId = 58;
// U3: storage/compaction blocks (9 ingots/gems <-> 1 block) + raw-material blocks.
pub const IRON_BLOCK: BlockId = 59;
pub const GOLD_BLOCK: BlockId = 60;
pub const DIAMOND_BLOCK: BlockId = 61;
pub const EMERALD_BLOCK: BlockId = 62;
pub const LAPIS_BLOCK: BlockId = 63;
pub const REDSTONE_BLOCK: BlockId = 64;
pub const COPPER_BLOCK: BlockId = 65;
pub const COAL_BLOCK: BlockId = 66;
pub const RAW_IRON_BLOCK: BlockId = 67;
pub const RAW_COPPER_BLOCK: BlockId = 68;
pub const RAW_GOLD_BLOCK: BlockId = 69;
// U3: the underground stone family (variants in blobs/geodes + their polished/brick build forms).
pub const TUFF: BlockId = 70;
pub const CALCITE: BlockId = 71;
pub const GRANITE: BlockId = 72;
pub const POLISHED_GRANITE: BlockId = 73;
pub const DIORITE: BlockId = 74;
pub const POLISHED_DIORITE: BlockId = 75;
pub const ANDESITE: BlockId = 76;
pub const POLISHED_ANDESITE: BlockId = 77;
pub const CLAY: BlockId = 78;
pub const DRIPSTONE_BLOCK: BlockId = 79;
pub const SMOOTH_BASALT: BlockId = 80;
pub const COBBLED_DEEPSLATE: BlockId = 81;
pub const POLISHED_DEEPSLATE: BlockId = 82;
pub const DEEPSLATE_BRICKS: BlockId = 83;
pub const DEEPSLATE_TILES: BlockId = 84;
pub const CUT_COPPER: BlockId = 85;
// ── Underground overhaul (U4): the cave-biome decoration set. PURE registry — placement/growth/sculk
// behavior land in later milestones (U8 cave biomes, U9 growth, U10 sculk spread, U11 Warden). The
// crystal/plant/dripleaf/root blocks render as cross-billboard cutouts (RenderKind::Cross); the rest
// are full cubes. ──
pub const AMETHYST_BLOCK: BlockId = 86;
pub const BUDDING_AMETHYST: BlockId = 87;
pub const AMETHYST_CLUSTER: BlockId = 88;
pub const SMALL_AMETHYST_BUD: BlockId = 89;
pub const MEDIUM_AMETHYST_BUD: BlockId = 90;
pub const LARGE_AMETHYST_BUD: BlockId = 91;
pub const POINTED_DRIPSTONE: BlockId = 92;
pub const MOSS_BLOCK: BlockId = 93;
pub const GLOW_LICHEN: BlockId = 94;
pub const CAVE_VINE: BlockId = 95;
pub const CAVE_VINE_BERRIES: BlockId = 96;
pub const AZALEA: BlockId = 97;
pub const FLOWERING_AZALEA: BlockId = 98;
pub const BIG_DRIPLEAF: BlockId = 99;
pub const SMALL_DRIPLEAF: BlockId = 100;
pub const ROOTED_DIRT: BlockId = 101;
pub const HANGING_ROOTS: BlockId = 102;
pub const SPORE_BLOSSOM: BlockId = 103;
pub const AZALEA_LEAVES: BlockId = 104;
pub const SCULK: BlockId = 105;
pub const SCULK_VEIN: BlockId = 106;
pub const SCULK_SENSOR: BlockId = 107;
pub const SCULK_SHRIEKER: BlockId = 108;
pub const SCULK_CATALYST: BlockId = 109;
pub const REINFORCED_DEEPSLATE: BlockId = 110;
// S6 farming: tilled soil + the three crops (growth stage 0-7 in state bits 0-2).
pub const FARMLAND: BlockId = 111;
pub const WHEAT_CROP: BlockId = 112;
pub const CARROT_CROP: BlockId = 113;
pub const POTATO_CROP: BlockId = 114;
// S7: oak sapling (random-ticks into a tree; leaves shed them).
pub const SAPLING: BlockId = 115;

/// Highest defined block id; bounds save-id validation (keep in sync as blocks are added).
pub const MAX_BLOCK: BlockId = SAPLING;

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
    /// Small attached fixture (torch/lever/button): a box (or two) on a floor/wall face. Walk-through,
    /// non-occluding, emitted per-cell; the attach face lives in the block-state byte (P11).
    Attach,
    /// Big dripleaf (F2): a thin standable leaf platform whose collision/render box droops with the
    /// tilt stage in the state byte (stand on it → it tilts down → you fall through → it resets).
    Platform,
}

#[inline]
pub fn render_kind(id: BlockId) -> RenderKind {
    match id {
        POPPY | DANDELION | TALL_GRASS | FERN | RED_MUSHROOM | BROWN_MUSHROOM | SUGAR_CANE
        // U4 cave-biome cross-billboard cutouts (crystals, vines, dripleaves, roots, sculk vein).
        | AMETHYST_CLUSTER | SMALL_AMETHYST_BUD | MEDIUM_AMETHYST_BUD | LARGE_AMETHYST_BUD
        | POINTED_DRIPSTONE | GLOW_LICHEN | CAVE_VINE | CAVE_VINE_BERRIES | AZALEA
        | FLOWERING_AZALEA | SMALL_DRIPLEAF | HANGING_ROOTS | SPORE_BLOSSOM
        | SCULK_VEIN => {
            RenderKind::Cross
        }
        BIG_DRIPLEAF => RenderKind::Platform, // F2: a standable, tilting leaf platform
        STONE_SLAB | WOOD_SLAB => RenderKind::Slab,
        STONE_STAIRS => RenderKind::Stairs,
        WOODEN_DOOR => RenderKind::Door,
        WOODEN_TRAPDOOR => RenderKind::Trapdoor,
        WOODEN_FENCE | COBBLESTONE_WALL | GLASS_PANE => RenderKind::Connect,
        TORCH | LEVER | BUTTON => RenderKind::Attach,
        WHEAT_CROP | CARROT_CROP | POTATO_CROP | SAPLING => RenderKind::Cross, // S6 crops + S7 sapling
        _ => RenderKind::Cube,
    }
}

/// A non-solid cross-billboard plant (walk-through, casts no shadow, doesn't cull neighbors).
#[inline]
pub fn is_plant(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Cross)
}

/// A small attached fixture (torch/lever/button): walk-through and non-occluding (like a plant), but
/// targetable (so it can be broken) and emitted as a 3D box rather than a cross-billboard.
#[inline]
pub fn is_attach(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Attach)
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

/// A block the interaction/mining raycast can hit (S6): everything solid PLUS the walk-through
/// cross plants — without this, aiming at wheat (or tall grass, or a flower) targeted the block
/// BEHIND it, so plants could never be harvested or bonemealed directly. Raycast-only: collision
/// keeps using `is_solid` (plants stay walk-through).
#[inline]
pub fn is_targetable(id: BlockId) -> bool {
    is_solid(id) || is_plant(id)
}

/// S6 crop helpers: the three crops carry a growth stage 0..=7 in state bits 0-2.
pub const CROP_MAX_STAGE: u8 = 7;
#[inline]
pub fn is_crop(id: BlockId) -> bool {
    matches!(id, WHEAT_CROP | CARROT_CROP | POTATO_CROP)
}
#[inline]
pub fn crop_stage(state: u8) -> u8 {
    state & 0b111
}
#[inline]
pub fn crop_mature(state: u8) -> bool {
    crop_stage(state) >= CROP_MAX_STAGE
}

/// The leaf sapling/apple roll (S7), shared by mining and natural decay: sapling 5%, apple 0.5%.
pub fn leaf_drop(r: u64) -> Option<(crate::item::ItemId, u8)> {
    match r % 1000 {
        0..=4 => Some((APPLE_ITEM, 1)),
        5..=54 => Some((SAPLING, 1)), // the block-item plants directly
        _ => None,
    }
}
const APPLE_ITEM: crate::item::ItemId = crate::item::APPLE;

/// Stage-aware crop billboard tile (S6). Mesher-only: `face_tile` stays id+face keyed (the N2 GI
/// lock) and reports the mature tile; crops never enter the voxel volume, so the stage-dependent
/// visual can't diverge GI from raster.
pub fn crop_tile(id: BlockId, state: u8) -> u32 {
    let young = crop_stage(state) < 4;
    match id {
        WHEAT_CROP => {
            if young {
                tile::CROP_WHEAT_YOUNG
            } else {
                tile::CROP_WHEAT_MATURE
            }
        }
        CARROT_CROP => {
            if young {
                tile::CROP_SPROUT
            } else {
                tile::CROP_CARROT_MATURE
            }
        }
        POTATO_CROP => {
            if young {
                tile::CROP_SPROUT
            } else {
                tile::CROP_POTATO_MATURE
            }
        }
        _ => face_tile(id, [0, 1, 0]),
    }
}

/// Random EXTRA drop on breaking (S6), on top of the deterministic `drops()`: mature wheat sheds
/// bonus seeds, mature root crops an extra item, tall grass sometimes seeds. `r` is a fresh random
/// word (the caller steps a game rng); None = no bonus this time.
pub fn bonus_drops(id: BlockId, state: u8, r: u64) -> Option<(crate::item::ItemId, u8)> {
    match id {
        TALL_GRASS if r % 10 < 3 => Some((crate::item::SEEDS, 1)),
        // S7: leaves shed a sapling 5% / an apple 0.5% (same roll decayed leaves use).
        LEAVES | AZALEA_LEAVES => leaf_drop(r),
        WHEAT_CROP if crop_mature(state) => {
            let n = (r % 4) as u8; // 0..=3 bonus seeds
            (n > 0).then_some((crate::item::SEEDS, n))
        }
        CARROT_CROP if crop_mature(state) => {
            let n = (r % 3) as u8;
            (n > 0).then_some((crate::item::CARROT, n))
        }
        POTATO_CROP if crop_mature(state) => {
            let n = (r % 3) as u8;
            (n > 0).then_some((crate::item::POTATO, n))
        }
        _ => None,
    }
}

/// A solid sub-box in a block's local 0..1 space: `[minx, miny, minz, maxx, maxy, maxz]`.
pub type Aabb = [f32; 6];
const BOX_FULL: [Aabb; 1] = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]];
const BOX_SLAB: [Aabb; 1] = [[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]]; // bottom half
const BOX_SLAB_TOP: [Aabb; 1] = [[0.0, 0.5, 0.0, 1.0, 1.0, 1.0]]; // top half
const BOX_NONE: [Aabb; 0] = [];
// Big-dripleaf (F2) leaf-platform boxes by tilt stage: a firm flat leaf near the cell top, drooping as
// it tilts, then nothing (you fall through) at full tilt. Render geometry == collision (mesher emits
// solid_boxes), so the leaf visibly droops + vanishes as it folds.
const DRIPLEAF_STABLE_BOX: [Aabb; 1] = [[0.0, 0.6875, 0.0, 1.0, 0.8125, 1.0]];
const DRIPLEAF_TILT_BOX: [Aabb; 1] = [[0.0, 0.5, 0.0, 1.0, 0.625, 1.0]];
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

// ── Attachment fixtures (torch/lever/button) ──────────────────────────────────────────────────────
// state bits 0-2 select the attach face (the direction from the block's cell toward the support it
// hangs on); bit 3 is on/pressed; bits 4-5 reserved for P31 redstone (powered / is-source).
pub const ATTACH_FLOOR: u8 = 0; // support below (sits on a floor)
pub const ATTACH_PZ: u8 = 1; // support wall at +Z (placed against a block's -Z face)
pub const ATTACH_PX: u8 = 2; // support wall at +X
pub const ATTACH_NZ: u8 = 3; // support wall at -Z
pub const ATTACH_NX: u8 = 4; // support wall at -X

#[inline]
pub fn attach_face(state: u8) -> u8 {
    state & 0b111
}
#[inline]
pub fn attach_on(state: u8) -> bool {
    (state >> 3) & 1 != 0
}
/// Pack an attach state (bits 4-5 left zero — reserved for P31 redstone).
#[inline]
pub fn attach_state(face: u8, on: bool) -> u8 {
    (face & 0b111) | ((on as u8) << 3)
}

// Torch: a thin 2px stick. On the floor it stands centered; on a wall it sits raised against that
// wall (a simple straightened approximation of Minecraft's angled wall torch — reads clearly at range).
const TORCH_FLOOR: [Aabb; 1] = [[0.4375, 0.0, 0.4375, 0.5625, 0.625, 0.5625]];
const TORCH_PZ: [Aabb; 1] = [[0.4375, 0.20, 0.75, 0.5625, 0.825, 1.0]];
const TORCH_NZ: [Aabb; 1] = [[0.4375, 0.20, 0.0, 0.5625, 0.825, 0.25]];
const TORCH_PX: [Aabb; 1] = [[0.75, 0.20, 0.4375, 1.0, 0.825, 0.5625]];
const TORCH_NX: [Aabb; 1] = [[0.0, 0.20, 0.4375, 0.25, 0.825, 0.5625]];
// Lever: a base plate flush to the attach face + a short handle stub rising out of it.
const LEVER_FLOOR: [Aabb; 2] = [
    [0.3125, 0.0, 0.375, 0.6875, 0.1875, 0.625],
    [0.4375, 0.1875, 0.4375, 0.5625, 0.5, 0.5625],
];
const LEVER_PZ: [Aabb; 2] = [
    [0.3125, 0.25, 0.8125, 0.6875, 0.75, 1.0],
    [0.4375, 0.4375, 0.5, 0.5625, 0.5625, 0.8125],
];
const LEVER_NZ: [Aabb; 2] = [
    [0.3125, 0.25, 0.0, 0.6875, 0.75, 0.1875],
    [0.4375, 0.4375, 0.1875, 0.5625, 0.5625, 0.5],
];
const LEVER_PX: [Aabb; 2] = [
    [0.8125, 0.25, 0.3125, 1.0, 0.75, 0.6875],
    [0.5, 0.4375, 0.4375, 0.8125, 0.5625, 0.5625],
];
const LEVER_NX: [Aabb; 2] = [
    [0.0, 0.25, 0.3125, 0.1875, 0.75, 0.6875],
    [0.1875, 0.4375, 0.4375, 0.5, 0.5625, 0.5625],
];
// Button: a shallow pad proud of the attach face.
const BUTTON_FLOOR: [Aabb; 1] = [[0.3125, 0.0, 0.375, 0.6875, 0.125, 0.625]];
const BUTTON_PZ: [Aabb; 1] = [[0.3125, 0.375, 0.875, 0.6875, 0.625, 1.0]];
const BUTTON_NZ: [Aabb; 1] = [[0.3125, 0.375, 0.0, 0.6875, 0.625, 0.125]];
const BUTTON_PX: [Aabb; 1] = [[0.875, 0.375, 0.3125, 1.0, 0.625, 0.6875]];
const BUTTON_NX: [Aabb; 1] = [[0.0, 0.375, 0.3125, 0.125, 0.625, 0.6875]];

/// The render sub-boxes of an attach fixture (torch/lever/button) for its attach face. Render-only:
/// collision is empty (solid_boxes returns BOX_NONE), so these boxes drive the mesher alone.
pub fn attach_boxes(id: BlockId, state: u8) -> &'static [Aabb] {
    let face = attach_face(state);
    match id {
        TORCH => match face {
            ATTACH_PZ => &TORCH_PZ,
            ATTACH_PX => &TORCH_PX,
            ATTACH_NZ => &TORCH_NZ,
            ATTACH_NX => &TORCH_NX,
            _ => &TORCH_FLOOR,
        },
        LEVER => match face {
            ATTACH_PZ => &LEVER_PZ,
            ATTACH_PX => &LEVER_PX,
            ATTACH_NZ => &LEVER_NZ,
            ATTACH_NX => &LEVER_NX,
            _ => &LEVER_FLOOR,
        },
        BUTTON => match face {
            ATTACH_PZ => &BUTTON_PZ,
            ATTACH_PX => &BUTTON_PX,
            ATTACH_NZ => &BUTTON_NZ,
            ATTACH_NX => &BUTTON_NX,
            _ => &BUTTON_FLOOR,
        },
        _ => &BOX_NONE,
    }
}

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
        // Attach fixtures (torch/lever/button): walk-through. Render geometry is the small box(es) in
        // attach_boxes (emitted by the mesher); collision is empty so you can stand in a torch's cell.
        RenderKind::Attach => &BOX_NONE,
        // F2 big dripleaf: a standable leaf that droops with its tilt stage, then drops you at full tilt.
        RenderKind::Platform => match dripleaf_tilt(state) {
            DRIPLEAF_FULL => &BOX_NONE,
            DRIPLEAF_TILTING => &DRIPLEAF_TILT_BOX,
            _ => &DRIPLEAF_STABLE_BOX,
        },
        _ => &BOX_FULL,
    }
}

// Big-dripleaf tilt stage (F2), in the block-state byte: 0 stable (firm), 1 tilting (drooping), 2 full
// (no collision — you fall through). Resets toward 0 when nobody is standing on it.
pub const DRIPLEAF_STABLE: u8 = 0;
pub const DRIPLEAF_TILTING: u8 = 1;
pub const DRIPLEAF_FULL: u8 = 2;
#[inline]
pub fn dripleaf_tilt(state: u8) -> u8 {
    (state & 0b11).min(2)
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
        && id != BIG_DRIPLEAF // F2: a thin leaf platform — doesn't occlude or block light
        && !is_plant(id)
        && !is_attach(id)
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
        LEVER => "Lever",
        BUTTON => "Button",
        COPPER_ORE => "Copper Ore",
        EMERALD_ORE => "Emerald Ore",
        DEEPSLATE_COAL_ORE => "Deepslate Coal Ore",
        DEEPSLATE_IRON_ORE => "Deepslate Iron Ore",
        DEEPSLATE_COPPER_ORE => "Deepslate Copper Ore",
        DEEPSLATE_GOLD_ORE => "Deepslate Gold Ore",
        DEEPSLATE_REDSTONE_ORE => "Deepslate Redstone Ore",
        DEEPSLATE_EMERALD_ORE => "Deepslate Emerald Ore",
        DEEPSLATE_LAPIS_ORE => "Deepslate Lapis Ore",
        DEEPSLATE_DIAMOND_ORE => "Deepslate Diamond Ore",
        IRON_BLOCK => "Block of Iron",
        GOLD_BLOCK => "Block of Gold",
        DIAMOND_BLOCK => "Block of Diamond",
        EMERALD_BLOCK => "Block of Emerald",
        LAPIS_BLOCK => "Lapis Lazuli Block",
        REDSTONE_BLOCK => "Block of Redstone",
        COPPER_BLOCK => "Block of Copper",
        COAL_BLOCK => "Block of Coal",
        RAW_IRON_BLOCK => "Block of Raw Iron",
        RAW_COPPER_BLOCK => "Block of Raw Copper",
        RAW_GOLD_BLOCK => "Block of Raw Gold",
        TUFF => "Tuff",
        CALCITE => "Calcite",
        GRANITE => "Granite",
        POLISHED_GRANITE => "Polished Granite",
        DIORITE => "Diorite",
        POLISHED_DIORITE => "Polished Diorite",
        ANDESITE => "Andesite",
        POLISHED_ANDESITE => "Polished Andesite",
        CLAY => "Clay",
        DRIPSTONE_BLOCK => "Dripstone Block",
        SMOOTH_BASALT => "Smooth Basalt",
        COBBLED_DEEPSLATE => "Cobbled Deepslate",
        POLISHED_DEEPSLATE => "Polished Deepslate",
        DEEPSLATE_BRICKS => "Deepslate Bricks",
        DEEPSLATE_TILES => "Deepslate Tiles",
        CUT_COPPER => "Cut Copper",
        AMETHYST_BLOCK => "Block of Amethyst",
        BUDDING_AMETHYST => "Budding Amethyst",
        AMETHYST_CLUSTER => "Amethyst Cluster",
        SMALL_AMETHYST_BUD => "Small Amethyst Bud",
        MEDIUM_AMETHYST_BUD => "Medium Amethyst Bud",
        LARGE_AMETHYST_BUD => "Large Amethyst Bud",
        POINTED_DRIPSTONE => "Pointed Dripstone",
        MOSS_BLOCK => "Moss Block",
        GLOW_LICHEN => "Glow Lichen",
        CAVE_VINE => "Cave Vine",
        CAVE_VINE_BERRIES => "Cave Vine with Berries",
        AZALEA => "Azalea",
        FLOWERING_AZALEA => "Flowering Azalea",
        BIG_DRIPLEAF => "Big Dripleaf",
        SMALL_DRIPLEAF => "Small Dripleaf",
        ROOTED_DIRT => "Rooted Dirt",
        HANGING_ROOTS => "Hanging Roots",
        SPORE_BLOSSOM => "Spore Blossom",
        AZALEA_LEAVES => "Azalea Leaves",
        SCULK => "Sculk",
        SCULK_VEIN => "Sculk Vein",
        SCULK_SENSOR => "Sculk Sensor",
        SCULK_SHRIEKER => "Sculk Shrieker",
        SCULK_CATALYST => "Sculk Catalyst",
        REINFORCED_DEEPSLATE => "Reinforced Deepslate",
        FARMLAND => "Farmland",
        SAPLING => "Oak Sapling",
        WHEAT_CROP => "Wheat",
        CARROT_CROP => "Carrots",
        POTATO_CROP => "Potatoes",
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
        WOODEN_DOOR | WOODEN_TRAPDOOR | WOODEN_FENCE => [0.62, 0.48, 0.30], // planks (matches tile avg)
        COBBLESTONE_WALL => [0.42, 0.42, 0.44],                            // cobble
        GLASS_PANE => [0.82, 0.91, 0.98],                                  // glass
        WOOD_SLAB => [0.62, 0.48, 0.30], // planks
        LEVER => [0.62, 0.48, 0.30],     // planks (matches tile::PLANKS average)
        BUTTON => [0.49, 0.49, 0.52],    // stone (matches tile::STONE average)
        // U2 ores: tile-average reflectance (stone/deepslate host blended with the ore fleck). Kept in
        // lockstep with the texture.rs base_color + the rtx_common.wgsl voxel_color cases.
        COPPER_ORE => [0.58, 0.49, 0.44],
        EMERALD_ORE => [0.46, 0.58, 0.49],
        DEEPSLATE_COAL_ORE => [0.20, 0.20, 0.22],
        DEEPSLATE_IRON_ORE => [0.34, 0.30, 0.30],
        DEEPSLATE_COPPER_ORE => [0.36, 0.30, 0.29],
        DEEPSLATE_GOLD_ORE => [0.36, 0.33, 0.27],
        DEEPSLATE_REDSTONE_ORE => [0.33, 0.23, 0.25],
        DEEPSLATE_EMERALD_ORE => [0.24, 0.36, 0.30],
        DEEPSLATE_LAPIS_ORE => [0.22, 0.26, 0.38],
        DEEPSLATE_DIAMOND_ORE => [0.27, 0.37, 0.39],
        // U3 storage blocks (solid metal/gem; raw blocks are chunkier nuggets).
        IRON_BLOCK => [0.86, 0.84, 0.80],
        GOLD_BLOCK => [0.96, 0.82, 0.25],
        DIAMOND_BLOCK => [0.40, 0.85, 0.84],
        EMERALD_BLOCK => [0.22, 0.78, 0.42],
        LAPIS_BLOCK => [0.18, 0.32, 0.72],
        REDSTONE_BLOCK => [0.72, 0.12, 0.10],
        COPPER_BLOCK => [0.78, 0.46, 0.34],
        COAL_BLOCK => [0.10, 0.10, 0.12],
        RAW_IRON_BLOCK => [0.72, 0.58, 0.46],
        RAW_COPPER_BLOCK => [0.74, 0.46, 0.30],
        RAW_GOLD_BLOCK => [0.82, 0.66, 0.28],
        // U3 stone family.
        TUFF => [0.42, 0.43, 0.40],
        CALCITE => [0.90, 0.90, 0.88],
        GRANITE => [0.66, 0.45, 0.38],
        POLISHED_GRANITE => [0.68, 0.47, 0.40],
        DIORITE => [0.84, 0.84, 0.85],
        POLISHED_DIORITE => [0.86, 0.86, 0.87],
        ANDESITE => [0.55, 0.56, 0.57],
        POLISHED_ANDESITE => [0.57, 0.58, 0.59],
        CLAY => [0.62, 0.64, 0.70],
        DRIPSTONE_BLOCK => [0.55, 0.42, 0.34],
        SMOOTH_BASALT => [0.28, 0.27, 0.30],
        COBBLED_DEEPSLATE => [0.26, 0.26, 0.29],
        POLISHED_DEEPSLATE => [0.24, 0.24, 0.27],
        DEEPSLATE_BRICKS => [0.23, 0.23, 0.26],
        DEEPSLATE_TILES => [0.22, 0.22, 0.25],
        CUT_COPPER => [0.78, 0.46, 0.34],
        // U4 cave-biome opaque cubes (must match texture.rs base_color + rtx_common.wgsl voxel_color).
        AMETHYST_BLOCK => [0.55, 0.40, 0.78],
        BUDDING_AMETHYST => [0.58, 0.42, 0.80],
        MOSS_BLOCK => [0.30, 0.42, 0.20],
        ROOTED_DIRT => [0.42, 0.32, 0.22],
        AZALEA_LEAVES => [0.30, 0.46, 0.22],
        SCULK => [0.06, 0.09, 0.11],
        SCULK_SENSOR => [0.10, 0.20, 0.24],
        SCULK_SHRIEKER => [0.12, 0.16, 0.18],
        SCULK_CATALYST => [0.10, 0.14, 0.16],
        REINFORCED_DEEPSLATE => [0.20, 0.21, 0.24],
        // S6 farming (FARMLAND must match texture.rs base_color + rtx_common.wgsl voxel_color).
        FARMLAND => [0.35, 0.24, 0.15],
        SAPLING => [0.30, 0.50, 0.20],
        WHEAT_CROP => [0.75, 0.65, 0.30],
        CARROT_CROP => [0.35, 0.55, 0.22],
        POTATO_CROP => [0.38, 0.52, 0.26],
        // U4 cave-biome cross-billboard cutouts: representative UI/icon swatch (texel art is paint_plant).
        AMETHYST_CLUSTER | SMALL_AMETHYST_BUD | MEDIUM_AMETHYST_BUD | LARGE_AMETHYST_BUD => {
            [0.62, 0.45, 0.85]
        }
        POINTED_DRIPSTONE => [0.55, 0.42, 0.34],
        GLOW_LICHEN => [0.55, 0.78, 0.66],
        CAVE_VINE => [0.36, 0.52, 0.22],
        CAVE_VINE_BERRIES => [0.95, 0.65, 0.20],
        AZALEA | FLOWERING_AZALEA => [0.30, 0.50, 0.22],
        BIG_DRIPLEAF | SMALL_DRIPLEAF => [0.34, 0.55, 0.24],
        HANGING_ROOTS => [0.58, 0.44, 0.30],
        SPORE_BLOSSOM => [0.85, 0.40, 0.55],
        SCULK_VEIN => [0.10, 0.16, 0.14],
        _ => [1.0, 0.0, 1.0],
    }
}

/// Self-emission strength per block (0 = unlit material). Lava glows; the lit shaders add this as
/// extra outgoing radiance, and GI/reflection rays treat an emissive hit as a light source.
pub fn emission(id: BlockId) -> f32 {
    match id {
        LAVA => 1.0,
        TORCH => 1.4, // M35-SL: brighter self-glow so a torch reads as a torch in daylight
        GLOWSTONE => 1.0,
        // U4: the sculk catalyst's soul-fire bloom is the only opaque cave-biome self-emitter (the
        // cross-billboard emitters — clusters/buds/glow lichen/berries — aren't in the voxel volume, so
        // they rely on light_emission only). Cross-block emission would have no GI volume to bounce in.
        SCULK_CATALYST => 0.4,
        _ => 0.0,
    }
}

/// Block-light emitted (0..15), for the light flood (M14b). Torch/glowstone/lava are sources.
pub fn light_emission(id: BlockId) -> u8 {
    match id {
        GLOWSTONE => 15,
        TORCH => 14,
        CAVE_VINE_BERRIES => 14,
        LAVA => 11,
        GLOW_LICHEN => 7,
        SCULK_CATALYST => 6,
        AMETHYST_CLUSTER => 5,
        LARGE_AMETHYST_BUD => 4,
        MEDIUM_AMETHYST_BUD => 2,
        SMALL_AMETHYST_BUD => 1,
        SCULK_SENSOR => 1,
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
    is_volume_solid(id) && id != LEAVES && id != AZALEA_LEAVES
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
        IRON_ORE | LAPIS_ORE | COPPER_ORE
        | DEEPSLATE_IRON_ORE | DEEPSLATE_COPPER_ORE | DEEPSLATE_LAPIS_ORE
        | IRON_BLOCK | COPPER_BLOCK | LAPIS_BLOCK | REDSTONE_BLOCK | RAW_IRON_BLOCK
        | RAW_COPPER_BLOCK | CUT_COPPER => 1,
        GOLD_ORE | DIAMOND_ORE | REDSTONE_ORE | EMERALD_ORE
        | DEEPSLATE_GOLD_ORE | DEEPSLATE_DIAMOND_ORE | DEEPSLATE_REDSTONE_ORE
        | DEEPSLATE_EMERALD_ORE
        | GOLD_BLOCK | DIAMOND_BLOCK | EMERALD_BLOCK | RAW_GOLD_BLOCK => 2,
        OBSIDIAN => 3,
        _ => 0,
    }
}

/// Time in seconds to break a block by hand (tools speed this up in M19). INFINITY = unbreakable.
pub fn hardness(id: BlockId) -> f32 {
    match id {
        // Reinforced deepslate is unbreakable like bedrock (breakable() returns false for non-finite).
        BEDROCK | REINFORCED_DEEPSLATE => f32::INFINITY,
        AIR | WATER | LAVA => 0.0,
        LEAVES | TORCH | GLOWSTONE => 0.3,
        // U4 thin foliage: azalea leaves are leaf-soft; glow lichen / sculk vein are cross billboards
        // but slightly tougher than 0 (listed BEFORE the `is_plant => 0.0` fallback so they win).
        AZALEA_LEAVES => 0.2,
        GLOW_LICHEN | SCULK_VEIN => 0.2,
        MOSS_BLOCK => 0.4,
        DIRT | GRASS | SAND | GRAVEL | SNOW | CLAY | FARMLAND => 0.6,
        // U4: amethyst family, pointed dripstone, rooted dirt, sculk sensor — stone-soft (1.5).
        AMETHYST_BLOCK | BUDDING_AMETHYST | AMETHYST_CLUSTER | SMALL_AMETHYST_BUD
        | MEDIUM_AMETHYST_BUD | LARGE_AMETHYST_BUD | POINTED_DRIPSTONE | SCULK_SENSOR => 1.5,
        ROOTED_DIRT => 0.5,
        // U4: sculk core blocks are 0.6; the shrieker/catalyst are tougher (3.0).
        SCULK => 0.6,
        SCULK_SHRIEKER | SCULK_CATALYST => 3.0,
        WOOD | PLANKS | CRAFTING_TABLE | CHEST | WOODEN_DOOR | WOODEN_TRAPDOOR | WOODEN_FENCE => 1.2,
        COBBLESTONE_WALL => 2.0,
        STONE | COBBLESTONE | BRICKS | COAL_ORE | IRON_ORE | GOLD_ORE | DIAMOND_ORE
        | REDSTONE_ORE | LAPIS_ORE | COPPER_ORE | EMERALD_ORE => 1.5,
        TUFF | CALCITE | GRANITE | POLISHED_GRANITE | DIORITE | POLISHED_DIORITE | ANDESITE
        | POLISHED_ANDESITE | DRIPSTONE_BLOCK | SMOOTH_BASALT => 1.5,
        DEEPSLATE | FURNACE => 2.0,
        DEEPSLATE_COAL_ORE | DEEPSLATE_IRON_ORE | DEEPSLATE_COPPER_ORE | DEEPSLATE_GOLD_ORE
        | DEEPSLATE_REDSTONE_ORE | DEEPSLATE_EMERALD_ORE | DEEPSLATE_LAPIS_ORE
        | DEEPSLATE_DIAMOND_ORE => 3.0,
        // Storage/raw blocks + deepslate build variants + cut copper: metal/slate-tough (3.0).
        IRON_BLOCK | GOLD_BLOCK | DIAMOND_BLOCK | EMERALD_BLOCK | LAPIS_BLOCK | REDSTONE_BLOCK
        | COPPER_BLOCK | COAL_BLOCK | RAW_IRON_BLOCK | RAW_COPPER_BLOCK | RAW_GOLD_BLOCK
        | COBBLED_DEEPSLATE | POLISHED_DEEPSLATE | DEEPSLATE_BRICKS | DEEPSLATE_TILES
        | CUT_COPPER => 3.0,
        OBSIDIAN => 8.0,
        ICE => 0.5,
        PUMPKIN => 1.0,
        GLASS | GLASS_PANE => 0.3,
        STONE_SLAB | STONE_STAIRS => 1.5,
        WOOD_SLAB => 1.2,
        LEVER | BUTTON => 0.5,
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
        COPPER_ORE | EMERALD_ORE | DEEPSLATE_COAL_ORE | DEEPSLATE_IRON_ORE | DEEPSLATE_COPPER_ORE
        | DEEPSLATE_GOLD_ORE | DEEPSLATE_REDSTONE_ORE | DEEPSLATE_EMERALD_ORE | DEEPSLATE_LAPIS_ORE
        | DEEPSLATE_DIAMOND_ORE => ToolClass::Pickaxe,
        IRON_BLOCK | GOLD_BLOCK | DIAMOND_BLOCK | EMERALD_BLOCK | LAPIS_BLOCK | REDSTONE_BLOCK
        | COPPER_BLOCK | COAL_BLOCK | RAW_IRON_BLOCK | RAW_COPPER_BLOCK | RAW_GOLD_BLOCK | TUFF
        | CALCITE | GRANITE | POLISHED_GRANITE | DIORITE | POLISHED_DIORITE | ANDESITE
        | POLISHED_ANDESITE | DRIPSTONE_BLOCK | SMOOTH_BASALT | COBBLED_DEEPSLATE
        | POLISHED_DEEPSLATE | DEEPSLATE_BRICKS | DEEPSLATE_TILES | CUT_COPPER => ToolClass::Pickaxe,
        // U4: amethyst family + pointed dripstone are pickaxe-mined (reinforced deepslate is unbreakable
        // so its tool is moot; sculk/moss/foliage are tool::None and fall through below).
        AMETHYST_BLOCK | BUDDING_AMETHYST | AMETHYST_CLUSTER | SMALL_AMETHYST_BUD
        | MEDIUM_AMETHYST_BUD | LARGE_AMETHYST_BUD | POINTED_DRIPSTONE => ToolClass::Pickaxe,
        WOOD | PLANKS | CRAFTING_TABLE | CHEST | PUMPKIN | WOOD_SLAB | WOODEN_DOOR
        | WOODEN_TRAPDOOR | WOODEN_FENCE => ToolClass::Axe,
        COBBLESTONE_WALL | BUTTON => ToolClass::Pickaxe,
        DIRT | GRASS | SAND | GRAVEL | SNOW | CLAY | ROOTED_DIRT | FARMLAND => ToolClass::Shovel,
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
        FARMLAND => (DIRT, 1),
        // S6 crops: the deterministic base drop by stage; the random bonus rides `bonus_drops`.
        WHEAT_CROP => {
            if crop_mature(state) { (crate::item::WHEAT, 1) } else { (crate::item::SEEDS, 1) }
        }
        CARROT_CROP => (crate::item::CARROT, if crop_mature(state) { 2 } else { 1 }),
        POTATO_CROP => (crate::item::POTATO, if crop_mature(state) { 2 } else { 1 }),
        // S6: tall grass drops seeds by chance (bonus_drops), never itself (vanilla).
        TALL_GRASS => return None,
        LEAVES => return None, // saplings/apples arrive with tree variety + farming
        ICE => return None,             // melts away (no silk touch yet)
        GLASS | GLASS_PANE => return None, // shatters
        // A double slab drops two slab items; a single drops one.
        STONE_SLAB | WOOD_SLAB => (id, if slab_half(state) == SLAB_DOUBLE { 2 } else { 1 }),
        COAL_ORE | DEEPSLATE_COAL_ORE => (item::COAL, 1),
        IRON_ORE | DEEPSLATE_IRON_ORE => (item::RAW_IRON, 1),
        GOLD_ORE | DEEPSLATE_GOLD_ORE => (item::RAW_GOLD, 1),
        DIAMOND_ORE | DEEPSLATE_DIAMOND_ORE => (item::DIAMOND, 1),
        REDSTONE_ORE | DEEPSLATE_REDSTONE_ORE => (item::REDSTONE_DUST, 4),
        LAPIS_ORE | DEEPSLATE_LAPIS_ORE => (item::LAPIS, 6),
        COPPER_ORE | DEEPSLATE_COPPER_ORE => (item::RAW_COPPER, 3),
        EMERALD_ORE | DEEPSLATE_EMERALD_ORE => (item::EMERALD, 1),
        // Stone-type drops: deepslate cobbles like stone, clay yields 4 clay balls (no silk touch yet).
        DEEPSLATE => (COBBLED_DEEPSLATE, 1),
        CLAY => (item::CLAY_BALL, 4),
        // U4 cave-biome drops. Amethyst cluster shatters into 4 shards; cave-vine berries yield 1 glow
        // berry. The amethyst buds + budding amethyst drop nothing (only the full cluster is harvestable),
        // azalea leaves drop nothing (like LEAVES), and the sculk family + reinforced deepslate drop
        // nothing (sculk needs silk touch in MC; reinforced deepslate is uncraftable/unbreakable). Every
        // other U4 block (amethyst block, pointed dripstone, moss, glow lichen, cave vine, azaleas,
        // dripleaves, rooted dirt, hanging roots, spore blossom) drops itself via the `_` fallthrough.
        AMETHYST_CLUSTER => (item::AMETHYST_SHARD, 4),
        CAVE_VINE_BERRIES => (item::GLOW_BERRIES, 1),
        BUDDING_AMETHYST | SMALL_AMETHYST_BUD | MEDIUM_AMETHYST_BUD | LARGE_AMETHYST_BUD
        | AZALEA_LEAVES | SCULK | SCULK_VEIN | SCULK_SENSOR | SCULK_SHRIEKER | SCULK_CATALYST
        | REINFORCED_DEEPSLATE => return None,
        _ => (id, 1),
    })
}

/// Experience awarded for mining a block (the ores that drop XP in Minecraft). Iron/gold give none
/// here — their XP comes from smelting the ore in a furnace.
pub fn mining_xp(id: BlockId) -> u32 {
    match id {
        COAL_ORE | DEEPSLATE_COAL_ORE => 1,
        REDSTONE_ORE | LAPIS_ORE | DEEPSLATE_REDSTONE_ORE | DEEPSLATE_LAPIS_ORE => 2,
        EMERALD_ORE | DEEPSLATE_EMERALD_ORE => 3,
        DIAMOND_ORE | DEEPSLATE_DIAMOND_ORE => 4,
        _ => 0,
    }
}

/// Atlas tile ids (M13). Index = `row*ATLAS_COLS + col` into the procedural texture atlas. Kept
/// beside `face_color`/`face_tile` so the painters (texture.rs) and the WGSL `tile_average()` stay
/// in lockstep. P19 grew the atlas to a 16x16=256 grid, so the 0..255 space has ample room for the
/// Phase-5 biome/wood/ore/ground-cover tiles.
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
    // P18 mob tiles. MOB_WOLF took tile 62 (the last slot of the old 8×8 grid); the other three are
    // intentional ALIASES of existing tiles (silhouette distinguishes them), NOT free slots. P19 then
    // grew the grid to 16x16=256, so 64..=255 are now free real estate (all read MAGENTA until painted).
    pub const MOB_WOLF: u32 = 62; // new flat grey
    pub const MOB_ENDERMAN: u32 = OBSIDIAN; // near-black
    pub const MOB_SLIME: u32 = SUGAR_CANE; // gel green
    pub const MOB_VILLAGER: u32 = MOB_COW; // brown robe (biped silhouette ≠ the cow quadruped)
    pub const MAGENTA: u32 = 63; // missing/unknown sentinel
    // Underground overhaul (U2): copper/emerald ore + a deepslate-host variant of every ore. The
    // 16x16 grid has free slots 64..=255; deepslate variants reuse the stone-ore flecks on a slate host.
    pub const COPPER: u32 = 64;
    pub const EMERALD: u32 = 65;
    pub const DS_COAL: u32 = 66;
    pub const DS_IRON: u32 = 67;
    pub const DS_COPPER: u32 = 68;
    pub const DS_GOLD: u32 = 69;
    pub const DS_REDSTONE: u32 = 70;
    pub const DS_EMERALD: u32 = 71;
    pub const DS_LAPIS: u32 = 72;
    pub const DS_DIAMOND: u32 = 73;
    // U3 storage/compaction blocks (metal/gem faces + raw-nugget faces).
    pub const IRON_BLOCK: u32 = 74;
    pub const GOLD_BLOCK: u32 = 75;
    pub const DIAMOND_BLOCK: u32 = 76;
    pub const EMERALD_BLOCK: u32 = 77;
    pub const LAPIS_BLOCK: u32 = 78;
    pub const REDSTONE_BLOCK: u32 = 79;
    pub const COPPER_BLOCK: u32 = 80;
    pub const COAL_BLOCK: u32 = 81;
    pub const RAW_IRON_BLOCK: u32 = 82;
    pub const RAW_COPPER_BLOCK: u32 = 83;
    pub const RAW_GOLD_BLOCK: u32 = 84;
    // U3 underground stone family.
    pub const TUFF: u32 = 85;
    pub const CALCITE: u32 = 86;
    pub const GRANITE: u32 = 87;
    pub const POLISHED_GRANITE: u32 = 88;
    pub const DIORITE: u32 = 89;
    pub const POLISHED_DIORITE: u32 = 90;
    pub const ANDESITE: u32 = 91;
    pub const POLISHED_ANDESITE: u32 = 92;
    pub const CLAY: u32 = 93;
    pub const DRIPSTONE: u32 = 94;
    pub const SMOOTH_BASALT: u32 = 95;
    pub const COBBLED_DEEPSLATE: u32 = 96;
    pub const POLISHED_DEEPSLATE: u32 = 97;
    pub const DEEPSLATE_BRICKS: u32 = 98;
    pub const DEEPSLATE_TILES: u32 = 99;
    pub const CUT_COPPER: u32 = 100;
    // U4 cave-biome set (101..=125): cube faces + cross-billboard plant cutouts (painted by paint_plant).
    pub const AMETHYST_BLOCK: u32 = 101;
    pub const BUDDING_AMETHYST: u32 = 102;
    pub const AMETHYST_CLUSTER: u32 = 103;
    pub const SMALL_AMETHYST_BUD: u32 = 104;
    pub const MEDIUM_AMETHYST_BUD: u32 = 105;
    pub const LARGE_AMETHYST_BUD: u32 = 106;
    pub const POINTED_DRIPSTONE: u32 = 107;
    pub const MOSS_BLOCK: u32 = 108;
    pub const GLOW_LICHEN: u32 = 109;
    pub const CAVE_VINE: u32 = 110;
    pub const CAVE_VINE_BERRIES: u32 = 111;
    pub const AZALEA: u32 = 112;
    pub const FLOWERING_AZALEA: u32 = 113;
    pub const BIG_DRIPLEAF: u32 = 114;
    pub const SMALL_DRIPLEAF: u32 = 115;
    pub const ROOTED_DIRT: u32 = 116;
    pub const HANGING_ROOTS: u32 = 117;
    pub const SPORE_BLOSSOM: u32 = 118;
    pub const AZALEA_LEAVES: u32 = 119;
    pub const SCULK: u32 = 120;
    pub const SCULK_VEIN: u32 = 121;
    pub const SCULK_SENSOR: u32 = 122;
    pub const SCULK_SHRIEKER: u32 = 123;
    pub const SCULK_CATALYST: u32 = 124;
    pub const REINFORCED_DEEPSLATE: u32 = 125;
    // M34-VM2 item sprites (alpha-cutout shapes, painted by `paint_item`). Tools occupy a contiguous
    // 25-tile block matching the tool id layout (tier*5 + class) so `item::item_tile` is a simple add.
    pub const TOOL_BASE: u32 = 126; // 126..=150: 5 tiers × 5 classes (Pick/Axe/Shovel/Sword/Hoe)
    pub const BOW: u32 = 151;
    pub const SHIELD: u32 = 152;
    pub const ARROW: u32 = 153;
    pub const STICK: u32 = 154;
    pub const INGOT_IRON: u32 = 155;
    pub const INGOT_GOLD: u32 = 156;
    pub const INGOT_COPPER: u32 = 157;
    pub const GEM_DIAMOND: u32 = 158;
    pub const GEM_EMERALD: u32 = 159;
    pub const GEM_LAPIS: u32 = 160;
    pub const GEM_AMETHYST: u32 = 161;
    pub const DUST_REDSTONE: u32 = 162;
    pub const ITEM_COAL: u32 = 163;
    pub const ITEM_CHARCOAL: u32 = 164;
    pub const RAW_IRON_ITEM: u32 = 165;
    pub const RAW_GOLD_ITEM: u32 = 166;
    pub const RAW_COPPER_ITEM: u32 = 167;
    pub const ITEM_FLINT: u32 = 168;
    pub const ITEM_BONE: u32 = 169;
    pub const ITEM_FEATHER: u32 = 170;
    pub const ITEM_LEATHER: u32 = 171;
    pub const ITEM_STRING: u32 = 172;
    pub const FOOD_BREAD: u32 = 173;
    pub const FOOD_APPLE: u32 = 174;
    pub const FOOD_CARROT: u32 = 175;
    pub const FOOD_WHEAT: u32 = 176;
    pub const FOOD_SEEDS: u32 = 177;
    pub const FOOD_BERRIES: u32 = 178;
    pub const FOOD_COOKED: u32 = 179; // a cooked steak/chop (shared by the cooked meats)
    pub const FOOD_POTATO: u32 = 180; // S6 (sprite band grows: texture.rs paint_item range)
    pub const FOOD_BAKED_POTATO: u32 = 181;
    /// Last item-sprite tile (texture.rs's `paint_item` dispatch band ends here); block tiles resume after.
    pub const ITEM_SPRITE_END: u32 = FOOD_BAKED_POTATO;
    // S6 farming block tiles.
    pub const FARMLAND_TOP: u32 = 182;
    pub const CROP_WHEAT_YOUNG: u32 = 183;
    pub const CROP_WHEAT_MATURE: u32 = 184;
    pub const CROP_SPROUT: u32 = 185; // shared young-stage sprout (carrots + potatoes)
    pub const CROP_CARROT_MATURE: u32 = 186;
    pub const CROP_POTATO_MATURE: u32 = 187;
    pub const SAPLING: u32 = 188; // S7
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
        LEVER => tile::PLANKS,
        BUTTON => tile::STONE,
        COPPER_ORE => tile::COPPER,
        EMERALD_ORE => tile::EMERALD,
        DEEPSLATE_COAL_ORE => tile::DS_COAL,
        DEEPSLATE_IRON_ORE => tile::DS_IRON,
        DEEPSLATE_COPPER_ORE => tile::DS_COPPER,
        DEEPSLATE_GOLD_ORE => tile::DS_GOLD,
        DEEPSLATE_REDSTONE_ORE => tile::DS_REDSTONE,
        DEEPSLATE_EMERALD_ORE => tile::DS_EMERALD,
        DEEPSLATE_LAPIS_ORE => tile::DS_LAPIS,
        DEEPSLATE_DIAMOND_ORE => tile::DS_DIAMOND,
        IRON_BLOCK => tile::IRON_BLOCK,
        GOLD_BLOCK => tile::GOLD_BLOCK,
        DIAMOND_BLOCK => tile::DIAMOND_BLOCK,
        EMERALD_BLOCK => tile::EMERALD_BLOCK,
        LAPIS_BLOCK => tile::LAPIS_BLOCK,
        REDSTONE_BLOCK => tile::REDSTONE_BLOCK,
        COPPER_BLOCK => tile::COPPER_BLOCK,
        COAL_BLOCK => tile::COAL_BLOCK,
        RAW_IRON_BLOCK => tile::RAW_IRON_BLOCK,
        RAW_COPPER_BLOCK => tile::RAW_COPPER_BLOCK,
        RAW_GOLD_BLOCK => tile::RAW_GOLD_BLOCK,
        TUFF => tile::TUFF,
        CALCITE => tile::CALCITE,
        GRANITE => tile::GRANITE,
        POLISHED_GRANITE => tile::POLISHED_GRANITE,
        DIORITE => tile::DIORITE,
        POLISHED_DIORITE => tile::POLISHED_DIORITE,
        ANDESITE => tile::ANDESITE,
        POLISHED_ANDESITE => tile::POLISHED_ANDESITE,
        CLAY => tile::CLAY,
        DRIPSTONE_BLOCK => tile::DRIPSTONE,
        SMOOTH_BASALT => tile::SMOOTH_BASALT,
        COBBLED_DEEPSLATE => tile::COBBLED_DEEPSLATE,
        POLISHED_DEEPSLATE => tile::POLISHED_DEEPSLATE,
        DEEPSLATE_BRICKS => tile::DEEPSLATE_BRICKS,
        DEEPSLATE_TILES => tile::DEEPSLATE_TILES,
        CUT_COPPER => tile::CUT_COPPER,
        // U4 cave-biome set.
        AMETHYST_BLOCK => tile::AMETHYST_BLOCK,
        BUDDING_AMETHYST => tile::BUDDING_AMETHYST,
        AMETHYST_CLUSTER => tile::AMETHYST_CLUSTER,
        SMALL_AMETHYST_BUD => tile::SMALL_AMETHYST_BUD,
        MEDIUM_AMETHYST_BUD => tile::MEDIUM_AMETHYST_BUD,
        LARGE_AMETHYST_BUD => tile::LARGE_AMETHYST_BUD,
        POINTED_DRIPSTONE => tile::POINTED_DRIPSTONE,
        MOSS_BLOCK => tile::MOSS_BLOCK,
        GLOW_LICHEN => tile::GLOW_LICHEN,
        CAVE_VINE => tile::CAVE_VINE,
        CAVE_VINE_BERRIES => tile::CAVE_VINE_BERRIES,
        AZALEA => tile::AZALEA,
        FLOWERING_AZALEA => tile::FLOWERING_AZALEA,
        BIG_DRIPLEAF => tile::BIG_DRIPLEAF,
        SMALL_DRIPLEAF => tile::SMALL_DRIPLEAF,
        ROOTED_DIRT => tile::ROOTED_DIRT,
        HANGING_ROOTS => tile::HANGING_ROOTS,
        SPORE_BLOSSOM => tile::SPORE_BLOSSOM,
        AZALEA_LEAVES => tile::AZALEA_LEAVES,
        SCULK => tile::SCULK,
        SCULK_VEIN => tile::SCULK_VEIN,
        SCULK_SENSOR => tile::SCULK_SENSOR,
        SCULK_SHRIEKER => tile::SCULK_SHRIEKER,
        SCULK_CATALYST => tile::SCULK_CATALYST,
        REINFORCED_DEEPSLATE => tile::REINFORCED_DEEPSLATE,
        FARMLAND => {
            if top {
                tile::FARMLAND_TOP
            } else {
                tile::DIRT
            }
        }
        SAPLING => tile::SAPLING,
        WHEAT_CROP => tile::CROP_WHEAT_MATURE,
        CARROT_CROP => tile::CROP_CARROT_MATURE,
        POTATO_CROP => tile::CROP_POTATO_MATURE,
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
