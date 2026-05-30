//! Block-light + skylight flood (M14). Computed from a chunk's `Neighborhood` (blocks only) during
//! meshing and baked per-face into the vertex `light` field — it is fully derived from blocks, so
//! nothing is stored or serialized. Two channels (0..15): skylight (geometric access to open sky;
//! makes caves dark) and block light (torch/glowstone/lava sources; survives in caves).
//!
//! The result is a padded `LDIM^3` grid in chunk-local coords `[-1, 32]` (one-block border), which is
//! exactly what the greedy mesher needs to read the air voxel just outside each face. The border is
//! solved from the neighborhood blocks (not a neighbor's stored light), which makes the common case
//! consistent across chunks.
//!
//! Known cross-chunk approximations (the `Neighborhood` only exposes the 6 face neighbors, so diagonal
//! chunks and full vertical columns aren't visible). To revisit in a light-engine v2 (column heightmap
//! + diagonal access, or a world-level pass):
//!   - Skylight border columns can't see a roof in a diagonally-above chunk, so an overhang sitting
//!     over a chunk seam can show a faint skylight seam.
//!   - Skylight only scans one chunk up (~64 blocks), so a vertical shaft taller than 32 blocks reads
//!     as fully sky-lit below that.
//!   - Block light treats diagonal/corner-chunk blocks as air, so a torch can leak slightly through a
//!     corner wall near a chunk boundary.

use std::collections::VecDeque;

use crate::block;
use crate::world::{Neighborhood, CHUNK_SIZE_I};

/// One-block border, so face sampling of the immediately-outside voxel is in range.
const LP: i32 = 1;
const LDIM: i32 = CHUNK_SIZE_I + 2 * LP; // 34

/// Block-light flood padding (max reach of a level-15 source).
const BP: i32 = 15;
const BDIM: i32 = CHUNK_SIZE_I + 2 * BP; // 62

const DIRS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

#[inline]
fn lidx(x: i32, y: i32, z: i32) -> usize {
    ((x + LP) + LDIM * ((z + LP) + LDIM * (y + LP))) as usize
}

#[inline]
fn bidx(x: i32, y: i32, z: i32) -> usize {
    ((x + BP) + BDIM * ((z + BP) + BDIM * (y + BP))) as usize
}

/// Packed light (sky in the high nibble, block in the low nibble) at a chunk-local voxel `[-1,32]`.
#[inline]
pub fn at(grid: &[u8], x: i32, y: i32, z: i32) -> u8 {
    grid[lidx(x, y, z)]
}

/// Compute the padded packed-light grid for the center chunk of `neigh`.
pub fn compute(neigh: &Neighborhood) -> Vec<u8> {
    let s = CHUNK_SIZE_I;
    let mut grid = vec![0u8; (LDIM * LDIM * LDIM) as usize];

    // --- Skylight: binary "open to sky straight up" per column, scanned center + up-neighbor. ---
    for z in -LP..=s {
        for x in -LP..=s {
            let mut blocked = false;
            for y in (-LP..2 * s).rev() {
                let id = neigh.block_or_air(x, y, z);
                if y <= s {
                    let sky = if blocked { 0u8 } else { 15u8 };
                    grid[lidx(x, y, z)] = sky << 4;
                }
                if block::blocks_skylight(id) {
                    blocked = true;
                }
            }
        }
    }

    // --- Block light: BFS from emitters, only when an emitter is in reach (skips empty chunks). ---
    if neigh.any_emitter() {
        let mut bl = vec![0u8; (BDIM * BDIM * BDIM) as usize];
        let mut q: VecDeque<(i32, i32, i32)> = VecDeque::new();
        for y in -BP..s + BP {
            for z in -BP..s + BP {
                for x in -BP..s + BP {
                    let e = block::light_emission(neigh.block_or_air(x, y, z));
                    if e > 0 {
                        let i = bidx(x, y, z);
                        if e > bl[i] {
                            bl[i] = e;
                            q.push_back((x, y, z));
                        }
                    }
                }
            }
        }
        while let Some((x, y, z)) = q.pop_front() {
            let l = bl[bidx(x, y, z)];
            if l <= 1 {
                continue;
            }
            for (dx, dy, dz) in DIRS {
                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                if nx < -BP || nx >= s + BP || ny < -BP || ny >= s + BP || nz < -BP || nz >= s + BP {
                    continue;
                }
                if block::is_opaque(neigh.block_or_air(nx, ny, nz)) {
                    continue;
                }
                let nl = l - 1;
                let ni = bidx(nx, ny, nz);
                if nl > bl[ni] {
                    bl[ni] = nl;
                    q.push_back((nx, ny, nz));
                }
            }
        }
        for y in -LP..=s {
            for z in -LP..=s {
                for x in -LP..=s {
                    let b = bl[bidx(x, y, z)] & 0x0F;
                    let gi = lidx(x, y, z);
                    grid[gi] = (grid[gi] & 0xF0) | b;
                }
            }
        }
    }

    grid
}
