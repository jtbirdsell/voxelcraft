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

/// A block participates in collision (fluids are passable).
#[inline]
pub fn is_solid(id: BlockId) -> bool {
    id != AIR && id != WATER && id != LAVA
}

/// Water or lava — simulated by the flowing-fluid tick and passable to the player.
#[inline]
pub fn is_fluid(id: BlockId) -> bool {
    id == WATER || id == LAVA
}

/// A fluid can flow into this cell (only empty air, so floods are monotonic and terminate).
#[inline]
pub fn is_replaceable(id: BlockId) -> bool {
    id == AIR
}

/// A block fully hides the touching face of an adjacent opaque block.
/// Water is non-opaque (translucent); leaves stay opaque (rendered as solid foliage).
#[inline]
pub fn is_opaque(id: BlockId) -> bool {
    id != AIR && id != WATER
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
        _ => [1.0, 0.0, 1.0],
    }
}

/// Self-emission strength per block (0 = unlit material). Lava glows; the lit shaders add this as
/// extra outgoing radiance, and GI/reflection rays treat an emissive hit as a light source.
pub fn emission(id: BlockId) -> f32 {
    match id {
        LAVA => 1.0,
        _ => 0.0,
    }
}
