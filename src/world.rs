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

pub struct World {
    pub chunks: FxHashMap<IVec3, Arc<Chunk>>,
    pub seed: u64,
}

impl World {
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: FxHashMap::default(),
            seed,
        }
    }

    pub fn get(&self, pos: IVec3) -> Option<&Arc<Chunk>> {
        self.chunks.get(&pos)
    }

    pub fn is_generated(&self, pos: IVec3) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// Generate the chunk at `pos` if not already present.
    pub fn ensure_generated(&mut self, pos: IVec3) {
        if !self.chunks.contains_key(&pos) {
            let chunk = generate_chunk(pos, self.seed);
            self.chunks.insert(pos, Arc::new(chunk));
        }
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

/// Deterministic terrain height (in blocks) at a world column. Single-octave-ish for M2;
/// real multi-octave noise + biomes/caves arrive in M3.
fn terrain_height(wx: i32, wz: i32, seed: u64) -> i32 {
    let sx = (seed & 0xffff) as f32 * 0.001;
    let sz = ((seed >> 16) & 0xffff) as f32 * 0.001;
    let fx = wx as f32;
    let fz = wz as f32;
    let h = 80.0
        + 22.0 * ((fx * 0.013 + sx).sin() * (fz * 0.011 + sz).cos())
        + 9.0 * (fx * 0.05 + fz * 0.031).sin()
        + 4.0 * (fx * 0.1 - fz * 0.07).cos();
    (h.round() as i32).clamp(1, WORLD_HEIGHT - 2)
}

fn surface_block(wy: i32, height: i32) -> BlockId {
    if wy >= height {
        block::AIR
    } else if wy == height - 1 {
        block::GRASS
    } else if wy + 4 >= height {
        block::DIRT
    } else {
        block::STONE
    }
}

/// Generate the chunk at `pos`. Uses a per-footprint min/max height test to early-out for
/// chunks that are entirely air (above terrain) or entirely stone (deep underground).
pub fn generate_chunk(pos: IVec3, seed: u64) -> Chunk {
    let origin = chunk_origin(pos);
    let (cy0, cy1) = (origin.y, origin.y + CHUNK_SIZE_I);

    // Footprint height range.
    let mut hmin = i32::MAX;
    let mut hmax = i32::MIN;
    let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let h = terrain_height(origin.x + lx as i32, origin.z + lz as i32, seed);
            heights[lz * CHUNK_SIZE + lx] = h;
            hmin = hmin.min(h);
            hmax = hmax.max(h);
        }
    }

    // Entirely above the highest column → all air.
    if cy0 >= hmax {
        return Chunk::filled(block::AIR);
    }
    // Entirely below the lowest surface (minus the 4-block dirt band) → all stone.
    if cy1 <= hmin - 4 {
        return Chunk::filled(block::STONE);
    }

    let mut chunk = Chunk::filled(block::AIR);
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let height = heights[lz * CHUNK_SIZE + lx];
            for ly in 0..CHUNK_SIZE {
                let wy = cy0 + ly as i32;
                let id = surface_block(wy, height);
                if id != block::AIR {
                    chunk.set(lx, ly, lz, id);
                }
            }
        }
    }
    chunk
}
