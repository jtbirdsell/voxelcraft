//! Chunk data model (M1: a single 32^3 chunk). Cubic sections, flat block array for now;
//! palette compression and streaming arrive in later milestones.

use crate::block::{self, BlockId};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE as i32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub struct Chunk {
    pub blocks: Vec<BlockId>,
}

#[inline]
fn index(x: usize, y: usize, z: usize) -> usize {
    // x fastest, then z, then y.
    x + CHUNK_SIZE * (z + CHUNK_SIZE * y)
}

impl Chunk {
    pub fn new_empty() -> Self {
        Self {
            blocks: vec![block::AIR; CHUNK_VOLUME],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.blocks[index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: BlockId) {
        self.blocks[index(x, y, z)] = id;
    }

    #[inline]
    pub fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0 && y >= 0 && z >= 0 && x < CHUNK_SIZE_I && y < CHUNK_SIZE_I && z < CHUNK_SIZE_I
    }

    /// M1 placeholder terrain: a smooth rolling heightmap of grass/dirt/stone within one chunk.
    pub fn generate_demo() -> Self {
        let mut chunk = Self::new_empty();
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let fx = x as f32;
                let fz = z as f32;
                let h = 16.0
                    + 6.0 * ((fx * 0.30).sin() + (fz * 0.30).cos())
                    + 4.0 * ((fx * 0.15 + fz * 0.10).sin());
                let height = (h.round() as i32).clamp(2, CHUNK_SIZE_I - 1) as usize;
                for y in 0..height {
                    let id = if y == height - 1 {
                        block::GRASS
                    } else if y + 4 >= height {
                        block::DIRT
                    } else {
                        block::STONE
                    };
                    chunk.set(x, y, z, id);
                }
            }
        }
        chunk
    }
}
