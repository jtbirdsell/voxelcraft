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

/// One cubic section of blocks. `light` packs sky (hi nibble) + block (lo nibble) light per voxel;
/// it is derived (recomputed on load, never serialized) and filled by the light flood (M14).
#[derive(Clone)]
pub struct Chunk {
    pub blocks: Vec<BlockId>,
    /// Per-cell block-state byte (0 = default). Carries orientation/half/axis/etc. for blocks whose
    /// family interprets it (stairs facing+half, slabs half, logs axis, doors …). Most cells are 0,
    /// so it LZ4-compresses to ~nothing. Baked into geometry at mesh time — never reaches the GPU.
    pub states: Vec<u8>,
    /// Packed sky/block light. Reserved for gameplay light queries (mob spawning, M31); rendering
    /// light is recomputed fresh in the mesher (`light::compute`), so this stays zero for now.
    #[allow(dead_code)]
    pub light: Vec<u8>,
    pub solid_count: u32,
    /// Count of light-emitting blocks, kept current in `set` so the mesher's emitter gate is O(1).
    pub emitter_count: u32,
}

impl Chunk {
    pub fn filled(id: BlockId) -> Self {
        let solid_count = if id == block::AIR { 0 } else { CHUNK_VOLUME as u32 };
        let emitter_count = if block::light_emission(id) > 0 {
            CHUNK_VOLUME as u32
        } else {
            0
        };
        Self {
            blocks: vec![id; CHUNK_VOLUME],
            states: vec![0; CHUNK_VOLUME],
            light: vec![0; CHUNK_VOLUME],
            solid_count,
            emitter_count,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.blocks[local_index(x, y, z)]
    }

    /// The block-state byte at a cell (0 = default). Interpreted per block family by `block.rs`.
    #[inline]
    pub fn state(&self, x: usize, y: usize, z: usize) -> u8 {
        self.states[local_index(x, y, z)]
    }

    // Light accessors — reserved for gameplay light queries (M31). Allowed dead for now.
    #[allow(dead_code)]
    #[inline]
    pub fn sky_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.light[local_index(x, y, z)] >> 4
    }

    #[allow(dead_code)]
    #[inline]
    pub fn block_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.light[local_index(x, y, z)] & 0x0F
    }

    #[allow(dead_code)]
    #[inline]
    pub fn set_light(&mut self, x: usize, y: usize, z: usize, sky: u8, block_l: u8) {
        self.light[local_index(x, y, z)] = (sky << 4) | (block_l & 0x0F);
    }

    /// Set a block with its default state (0). Most blocks (dirt, stone, …) only ever use this.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: BlockId) {
        self.set_state(x, y, z, id, 0);
    }

    /// Set a block id together with its block-state byte, keeping solid/emitter counts current.
    #[inline]
    pub fn set_state(&mut self, x: usize, y: usize, z: usize, id: BlockId, state: u8) {
        let i = local_index(x, y, z);
        let was_solid = self.blocks[i] != block::AIR;
        let now_solid = id != block::AIR;
        match (was_solid, now_solid) {
            (false, true) => self.solid_count += 1,
            (true, false) => self.solid_count -= 1,
            _ => {}
        }
        let was_emit = block::light_emission(self.blocks[i]) > 0;
        let now_emit = block::light_emission(id) > 0;
        match (was_emit, now_emit) {
            (false, true) => self.emitter_count += 1,
            (true, false) => self.emitter_count -= 1,
            _ => {}
        }
        self.blocks[i] = id;
        self.states[i] = state;
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
    /// Panic-safe block lookup: returns AIR for any voxel more than one axis out of the center
    /// (diagonal/corner chunks aren't in the 6-face view). Single-axis-out uses the face neighbor.
    #[inline]
    pub fn block_or_air(&self, x: i32, y: i32, z: i32) -> BlockId {
        const S: i32 = CHUNK_SIZE_I;
        let oob = (x < 0 || x >= S) as i32 + (y < 0 || y >= S) as i32 + (z < 0 || z >= S) as i32;
        if oob == 0 {
            return self.center.get(x as usize, y as usize, z as usize);
        }
        if oob > 1 {
            return block::AIR;
        }
        self.block_at(x, y, z)
    }

    /// True if the center or any of the 6 face neighbors contains a light-emitting block — the
    /// cheap gate that lets the block-light flood skip the vast majority of (sourceless) chunks.
    pub fn any_emitter(&self) -> bool {
        let has = |c: &Option<Arc<Chunk>>| c.as_ref().is_some_and(|c| c.emitter_count > 0);
        self.center.emitter_count > 0
            || has(&self.neg_x)
            || has(&self.pos_x)
            || has(&self.neg_y)
            || has(&self.pos_y)
            || has(&self.neg_z)
            || has(&self.pos_z)
    }

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

    /// Block-state byte at a (possibly cross-chunk) cell — mirrors `block_at`. Used by the mesher to
    /// read oriented-block state (stairs facing now; fence/wall/pane connections + stair corners
    /// will read neighbor states here). Out-of-view neighbors default to 0.
    #[inline]
    pub fn state_at(&self, x: i32, y: i32, z: i32) -> u8 {
        const S: i32 = CHUNK_SIZE_I;
        let get = |c: &Option<Arc<Chunk>>, x: i32, y: i32, z: i32| -> u8 {
            c.as_ref()
                .map(|c| c.state(x as usize, y as usize, z as usize))
                .unwrap_or(0)
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
        self.center.state(x as usize, y as usize, z as usize)
    }
}
