//! Block registry (M1 minimal). Blocks are `u16` ids; properties come from small tables.
//! This will grow into a proper data-driven registry in later milestones.

pub type BlockId = u16;

pub const AIR: BlockId = 0;
pub const STONE: BlockId = 1;
pub const DIRT: BlockId = 2;
pub const GRASS: BlockId = 3;
pub const SAND: BlockId = 4;
pub const WOOD: BlockId = 5;
pub const LEAVES: BlockId = 6;
pub const WATER: BlockId = 7;

/// A block participates in collision / hides faces of neighbors.
#[inline]
pub fn is_solid(id: BlockId) -> bool {
    id != AIR && id != WATER
}

/// A block fully hides the touching face of an adjacent block (opaque cube).
/// Non-opaque blocks (air, water, leaves) let neighbor faces show through.
#[inline]
pub fn is_opaque(id: BlockId) -> bool {
    !matches!(id, AIR | WATER | LEAVES)
}

/// Base albedo for a block face. `face_offset[1] == 1` means the +Y (top) face,
/// which lets grass be green on top and dirt on the sides.
pub fn face_color(id: BlockId, face_offset: [i32; 3]) -> [f32; 3] {
    let top = face_offset[1] == 1;
    match id {
        GRASS => {
            if top {
                [0.36, 0.60, 0.27]
            } else {
                [0.45, 0.33, 0.21]
            }
        }
        DIRT => [0.45, 0.33, 0.21],
        STONE => [0.49, 0.49, 0.52],
        SAND => [0.80, 0.74, 0.51],
        WOOD => [0.43, 0.31, 0.18],
        LEAVES => [0.26, 0.44, 0.22],
        WATER => [0.20, 0.36, 0.66],
        _ => [1.0, 0.0, 1.0], // magenta = unknown
    }
}
