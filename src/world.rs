//! World model: cubic 32^3 chunks addressed by chunk coordinates, an infinite (in X/Z)
//! procedurally generated world, and a neighborhood view used for cross-chunk meshing.

use std::sync::Arc;

use glam::IVec3;
use rustc_hash::FxHashMap;

use crate::block::{self, BlockId};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE as i32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Vertical extent of the world, in chunks (so the world is 32 * this blocks tall).
pub const WORLD_HEIGHT_CHUNKS: i32 = 8;
pub const WORLD_HEIGHT: i32 = WORLD_HEIGHT_CHUNKS * CHUNK_SIZE_I;

#[inline]
fn local_index(x: usize, y: usize, z: usize) -> usize {
    x + CHUNK_SIZE * (z + CHUNK_SIZE * y)
}

/// One cubic section of blocks.
#[derive(Clone)]
pub struct Chunk {
    pub blocks: Vec<BlockId>,
    pub solid_count: u32,
}

impl Chunk {
    pub fn filled(id: BlockId) -> Self {
        let solid_count = if id == block::AIR { 0 } else { CHUNK_VOLUME as u32 };
        Self {
            blocks: vec![id; CHUNK_VOLUME],
            solid_count,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.blocks[local_index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: BlockId) {
        let i = local_index(x, y, z);
        let was_solid = self.blocks[i] != block::AIR;
        let now_solid = id != block::AIR;
        match (was_solid, now_solid) {
            (false, true) => self.solid_count += 1,
            (true, false) => self.solid_count -= 1,
            _ => {}
        }
        self.blocks[i] = id;
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }
}

/// Convert a world block position to the chunk position containing it.
#[inline]
pub fn chunk_of(world_pos: IVec3) -> IVec3 {
    IVec3::new(
        world_pos.x.div_euclid(CHUNK_SIZE_I),
        world_pos.y.div_euclid(CHUNK_SIZE_I),
        world_pos.z.div_euclid(CHUNK_SIZE_I),
    )
}

/// Block-space origin (minimum corner) of a chunk.
#[inline]
pub fn chunk_origin(pos: IVec3) -> IVec3 {
    pos * CHUNK_SIZE_I
}

#[derive(Default)]
pub struct World {
    pub chunks: FxHashMap<IVec3, Arc<Chunk>>,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, pos: IVec3) -> Option<&Arc<Chunk>> {
        self.chunks.get(&pos)
    }

    pub fn is_generated(&self, pos: IVec3) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// Build a neighborhood (center + 6 face neighbors) for meshing.
    pub fn neighborhood(&self, pos: IVec3) -> Option<Neighborhood> {
        let center = self.chunks.get(&pos)?.clone();
        Some(Neighborhood {
            center,
            neg_x: self.chunks.get(&(pos - IVec3::X)).cloned(),
            pos_x: self.chunks.get(&(pos + IVec3::X)).cloned(),
            neg_y: self.chunks.get(&(pos - IVec3::Y)).cloned(),
            pos_y: self.chunks.get(&(pos + IVec3::Y)).cloned(),
            neg_z: self.chunks.get(&(pos - IVec3::Z)).cloned(),
            pos_z: self.chunks.get(&(pos + IVec3::Z)).cloned(),
        })
    }
}

/// Center chunk plus its 6 face neighbors, providing block lookups that span chunk
/// boundaries (only one axis may be out of [0,32) at a time — which is all the greedy
/// mesher needs).
pub struct Neighborhood {
    pub center: Arc<Chunk>,
    neg_x: Option<Arc<Chunk>>,
    pos_x: Option<Arc<Chunk>>,
    neg_y: Option<Arc<Chunk>>,
    pos_y: Option<Arc<Chunk>>,
    neg_z: Option<Arc<Chunk>>,
    pos_z: Option<Arc<Chunk>>,
}

impl Neighborhood {
    #[inline]
    pub fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        const S: i32 = CHUNK_SIZE_I;
        let get = |c: &Option<Arc<Chunk>>, x: i32, y: i32, z: i32| -> BlockId {
            c.as_ref()
                .map(|c| c.get(x as usize, y as usize, z as usize))
                .unwrap_or(block::AIR)
        };
        if x < 0 {
            return get(&self.neg_x, x + S, y, z);
        }
        if x >= S {
            return get(&self.pos_x, x - S, y, z);
        }
        if y < 0 {
            return get(&self.neg_y, x, y + S, z);
        }
        if y >= S {
            return get(&self.pos_y, x, y - S, z);
        }
        if z < 0 {
            return get(&self.neg_z, x, y, z + S);
        }
        if z >= S {
            return get(&self.pos_z, x, y, z - S);
        }
        self.center.get(x as usize, y as usize, z as usize)
    }

    #[inline]
    pub fn opaque_at(&self, x: i32, y: i32, z: i32) -> bool {
        block::is_opaque(self.block_at(x, y, z))
    }
}
