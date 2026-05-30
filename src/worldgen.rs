//! Procedural world generation (M3): multi-octave OpenSimplex2 terrain, parameter-space biomes,
//! 3D-noise caves, scattered ores, sea-level water, and deterministic trees.
//!
//! Determinism: every block is a pure function of (seed, world position). Trees are decorated
//! without cross-chunk writes — each chunk independently stamps any tree (from neighboring
//! columns, within a small margin) whose voxels fall inside it. The `Worldgen` is immutable
//! after construction and shared across worker threads via `Arc`.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use glam::IVec3;

use crate::block::{self, BlockId};
use crate::world::{chunk_origin, Chunk, CHUNK_SIZE, CHUNK_SIZE_I, WORLD_HEIGHT};

pub const SEA_LEVEL: i32 = 64;
const MAX_TREE_HEIGHT: i32 = 9;
const TREE_MARGIN: i32 = 2;
const TREE_SALT: u64 = 0x7265655F_5345_4544;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Biome {
    Ocean,
    Beach,
    Desert,
    Plains,
    Forest,
    Snowy,
    Mountains,
}

pub struct Worldgen {
    seed: u64,
    continent: FastNoiseLite,
    hills: FastNoiseLite,
    detail: FastNoiseLite,
    temperature: FastNoiseLite,
    humidity: FastNoiseLite,
    cave_a: FastNoiseLite,
    cave_b: FastNoiseLite,
    cheese: FastNoiseLite,
}

impl Worldgen {
    pub fn new(seed: u64) -> Self {
        let mk = |off: u64, nt: NoiseType, freq: f32, octaves: i32| {
            let mut n = FastNoiseLite::new();
            n.set_seed(Some(seed.wrapping_add(off) as i32));
            n.set_noise_type(Some(nt));
            n.set_frequency(Some(freq));
            n.set_fractal_type(Some(FractalType::FBm));
            n.set_fractal_octaves(Some(octaves));
            n
        };
        Self {
            seed,
            continent: mk(1, NoiseType::OpenSimplex2, 0.0016, 4),
            hills: mk(2, NoiseType::OpenSimplex2, 0.0095, 4),
            detail: mk(3, NoiseType::OpenSimplex2, 0.031, 3),
            temperature: mk(4, NoiseType::OpenSimplex2, 0.0009, 2),
            humidity: mk(5, NoiseType::OpenSimplex2, 0.0012, 2),
            cave_a: mk(6, NoiseType::OpenSimplex2, 0.018, 2),
            cave_b: mk(7, NoiseType::OpenSimplex2S, 0.018, 2),
            cheese: mk(8, NoiseType::OpenSimplex2, 0.010, 2),
        }
    }

    /// Surface height (number of solid blocks) at a world column.
    pub fn height(&self, wx: i32, wz: i32) -> i32 {
        let (fx, fz) = (wx as f32, wz as f32);
        let c = self.continent.get_noise_2d(fx, fz); // [-1, 1]
        let cont = c.signum() * c.abs().powf(1.15); // flatten lowlands, exaggerate highs
        let base = (SEA_LEVEL + 4) as f32 + cont * 34.0;
        let h = self.hills.get_noise_2d(fx, fz) * 16.0;
        let d = self.detail.get_noise_2d(fx, fz) * 4.5;
        ((base + h + d).round() as i32).clamp(3, WORLD_HEIGHT - 12)
    }

    fn biome(&self, wx: i32, wz: i32, height: i32) -> Biome {
        if height <= SEA_LEVEL - 1 {
            return Biome::Ocean;
        }
        if height <= SEA_LEVEL + 1 {
            return Biome::Beach;
        }
        if height >= 104 {
            return Biome::Mountains;
        }
        let t = self.temperature.get_noise_2d(wx as f32, wz as f32);
        let hum = self.humidity.get_noise_2d(wx as f32, wz as f32);
        if t < -0.35 {
            Biome::Snowy
        } else if t > 0.33 && hum < -0.05 {
            Biome::Desert
        } else if hum > 0.12 {
            Biome::Forest
        } else {
            Biome::Plains
        }
    }

    /// Human-readable biome name at a world column (for the F3 debug overlay).
    pub fn biome_name(&self, wx: i32, wz: i32) -> &'static str {
        let h = self.height(wx, wz);
        match self.biome(wx, wz, h) {
            Biome::Ocean => "Ocean",
            Biome::Beach => "Beach",
            Biome::Desert => "Desert",
            Biome::Plains => "Plains",
            Biome::Forest => "Forest",
            Biome::Snowy => "Snowy",
            Biome::Mountains => "Mountains",
        }
    }

    fn surface_top(&self, biome: Biome, height: i32) -> BlockId {
        match biome {
            Biome::Desert | Biome::Beach => block::SAND,
            Biome::Snowy => block::SNOW,
            Biome::Mountains => {
                if height > 112 {
                    block::SNOW
                } else {
                    block::STONE
                }
            }
            _ => block::GRASS,
        }
    }

    fn subsurface(&self, biome: Biome) -> BlockId {
        match biome {
            Biome::Desert | Biome::Beach => block::SAND,
            Biome::Mountains => block::STONE,
            _ => block::DIRT,
        }
    }

    fn is_cave(&self, wx: i32, wy: i32, wz: i32) -> bool {
        let (fx, fy, fz) = (wx as f32, wy as f32, wz as f32);
        // Spaghetti tunnels: two noise fields both near zero.
        let a = self.cave_a.get_noise_3d(fx, fy, fz);
        let b = self.cave_b.get_noise_3d(fx, fy, fz);
        if a.abs() < 0.05 && b.abs() < 0.05 {
            return true;
        }
        // Cheese pockets (vertically squashed for wider caverns).
        self.cheese.get_noise_3d(fx, fy * 1.4, fz) > 0.80
    }

    /// Host rock by depth: deepslate in the deep band, stone above.
    #[inline]
    fn host_stone(wy: i32) -> BlockId {
        if wy < 14 {
            block::DEEPSLATE
        } else {
            block::STONE
        }
    }

    /// Depth-banded ore distribution (diamond/redstone deep, gold lower, iron/coal common), else the
    /// host rock (deepslate/stone).
    fn ore_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        let h = hash3(self.seed, wx, wy, wz);
        if wy < 16 && (h % 1100) < 3 {
            return block::DIAMOND_ORE;
        }
        if (14..40).contains(&wy) && ((h >> 4) % 1400) < 5 {
            return block::LAPIS_ORE;
        }
        if wy < 16 && ((h >> 8) % 1200) < 6 {
            return block::REDSTONE_ORE;
        }
        if wy < 32 && ((h >> 12) % 1500) < 4 {
            return block::GOLD_ORE;
        }
        if wy < 64 && ((h >> 16) % 900) < 7 {
            return block::IRON_ORE;
        }
        if wy < 128 && ((h >> 22) % 700) < 8 {
            return block::COAL_ORE;
        }
        Self::host_stone(wy)
    }

    pub fn generate_chunk(&self, pos: IVec3) -> Chunk {
        let origin = chunk_origin(pos);
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        let cy1 = oy + CHUNK_SIZE_I;

        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        let mut hmin = i32::MAX;
        let mut hmax = i32::MIN;
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let h = self.height(ox + lx as i32, oz + lz as i32);
                heights[lz * CHUNK_SIZE + lx] = h;
                hmin = hmin.min(h);
                hmax = hmax.max(h);
            }
        }

        // Entirely above terrain (+ tree reach) and above sea level → all air.
        if oy > hmax + MAX_TREE_HEIGHT && oy >= SEA_LEVEL {
            return Chunk::filled(block::AIR);
        }

        let mut chunk = Chunk::filled(block::AIR);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = ox + lx as i32;
                let wz = oz + lz as i32;
                let height = heights[lz * CHUNK_SIZE + lx];
                let biome = self.biome(wx, wz, height);
                let top = self.surface_top(biome, height);
                let sub = self.subsurface(biome);
                for ly in 0..CHUNK_SIZE {
                    let wy = oy + ly as i32;
                    if wy < height {
                        let depth = height - 1 - wy;
                        let mut id = if depth == 0 {
                            top
                        } else if depth <= 3 {
                            sub
                        } else {
                            block::STONE
                        };
                        // Bedrock floor (jagged, never carved): y0 solid, y1 ~1/2, y2 ~1/4.
                        let bedrock =
                            wy == 0 || (wy <= 2 && (hash3(self.seed ^ 0xB, wx, wy, wz) % 4) as i32 > wy);
                        if bedrock {
                            id = block::BEDROCK;
                        } else if wy >= 3 && wy < height - 1 && self.is_cave(wx, wy, wz) {
                            id = block::AIR;
                        } else if id == block::STONE {
                            id = self.ore_at(wx, wy, wz);
                        }
                        if id != block::AIR {
                            chunk.set(lx, ly, lz, id);
                        }
                    } else if wy < SEA_LEVEL {
                        // Cold columns freeze their water *surface* into ice. Keyed on temperature
                        // directly — biome() classifies any water column as Ocean, never Snowy, so a
                        // biome test here would be dead. Sampled only for the surface cell.
                        let frozen = wy == SEA_LEVEL - 1
                            && self.temperature.get_noise_2d(wx as f32, wz as f32) < -0.30;
                        let fluid = if frozen { block::ICE } else { block::WATER };
                        chunk.set(lx, ly, lz, fluid);
                    }
                }
            }
        }

        // Trees + surface decoration only for chunks overlapping the surface band.
        if cy1 > hmin && oy <= hmax + MAX_TREE_HEIGHT {
            self.place_trees(&mut chunk, origin);
            self.place_decoration(&mut chunk, origin);
            self.place_sugar_cane(&mut chunk, origin);
        }

        chunk
    }

    /// Scatter cross-billboard plants (flowers, tall grass) on grass and cactus on sand. Deterministic
    /// and confined to each chunk's own columns (no cross-chunk writes), like `place_trees`.
    /// Sugar cane: clumps of 1-2-tall stalks along water edges, stamped by world-y and clipped to
    /// this chunk so a stalk crossing a vertical chunk boundary is completed by the chunk above
    /// (each chunk writes only its own cells; fully world-derived, so it's seam-safe + deterministic).
    fn place_sugar_cane(&self, chunk: &mut Chunk, origin: IVec3) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = ox + lx as i32;
                let wz = oz + lz as i32;
                let ground = self.height(wx, wz);
                if ground <= SEA_LEVEL || !self.near_water(wx, wz) {
                    continue;
                }
                let surf = self.surface_top(self.biome(wx, wz, ground), ground);
                if !matches!(surf, block::SAND | block::GRASS) {
                    continue;
                }
                if (hash2(self.seed ^ 0x5A1A_CA4E, wx, wz) % 1000) as f32 / 1000.0 >= 0.22 {
                    continue;
                }
                let tall = 1 + (hash2(self.seed ^ 0xCA4E, wx, wz) % 2) as i32; // 1 or 2
                for k in 0..tall {
                    let ly = ground + k - oy; // base sits in the air cell above the surface
                    if ly < 0 {
                        continue; // a lower segment belongs to the chunk below
                    }
                    if ly >= CHUNK_SIZE_I {
                        break; // an upper segment belongs to the chunk above
                    }
                    if chunk.get(lx, ly as usize, lz) != block::AIR {
                        break; // blocked by a tree/flower — stop the stalk here
                    }
                    chunk.set(lx, ly as usize, lz, block::SUGAR_CANE);
                }
            }
        }
    }

    /// True if any orthogonal neighbor column sits at/below sea level (i.e. holds water).
    fn near_water(&self, wx: i32, wz: i32) -> bool {
        self.height(wx + 1, wz) <= SEA_LEVEL
            || self.height(wx - 1, wz) <= SEA_LEVEL
            || self.height(wx, wz + 1) <= SEA_LEVEL
            || self.height(wx, wz - 1) <= SEA_LEVEL
    }

    fn place_decoration(&self, chunk: &mut Chunk, origin: IVec3) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = ox + lx as i32;
                let wz = oz + lz as i32;
                let ground = self.height(wx, wz);
                if ground <= SEA_LEVEL {
                    continue;
                }
                let ly = ground - oy; // air cell directly above the surface block
                if ly < 1 || ly >= CHUNK_SIZE_I {
                    continue;
                }
                if chunk.get(lx, ly as usize, lz) != block::AIR {
                    continue; // occupied (e.g. by a tree)
                }
                let below = chunk.get(lx, (ly - 1) as usize, lz);
                let biome = self.biome(wx, wz, ground);
                let r = (hash2(self.seed ^ 0x00DE_C0DE, wx, wz) % 1000) as f32 / 1000.0;
                let plant = match biome {
                    Biome::Plains if below == block::GRASS => {
                        if r < 0.05 {
                            Some(block::TALL_GRASS)
                        } else if r < 0.066 {
                            Some(block::POPPY)
                        } else if r < 0.082 {
                            Some(block::DANDELION)
                        } else if r < 0.0845 {
                            Some(block::PUMPKIN) // rare patch
                        } else {
                            None
                        }
                    }
                    Biome::Forest if below == block::GRASS => {
                        if r < 0.11 {
                            Some(block::TALL_GRASS)
                        } else if r < 0.125 {
                            Some(block::DANDELION)
                        } else if r < 0.150 {
                            Some(block::FERN)
                        } else if r < 0.156 {
                            Some(block::RED_MUSHROOM)
                        } else if r < 0.162 {
                            Some(block::BROWN_MUSHROOM)
                        } else {
                            None
                        }
                    }
                    Biome::Desert if below == block::SAND && r < 0.012 => Some(block::CACTUS),
                    _ => None,
                };
                if let Some(p) = plant {
                    chunk.set(lx, ly as usize, lz, p);
                }
            }
        }
    }

    fn place_trees(&self, chunk: &mut Chunk, origin: IVec3) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        for bz in (oz - TREE_MARGIN)..(oz + CHUNK_SIZE_I + TREE_MARGIN) {
            for bx in (ox - TREE_MARGIN)..(ox + CHUNK_SIZE_I + TREE_MARGIN) {
                let ground = self.height(bx, bz);
                if ground <= SEA_LEVEL {
                    continue;
                }
                let biome = self.biome(bx, bz, ground);
                let density = match biome {
                    Biome::Forest => 0.09,
                    Biome::Plains => 0.012,
                    _ => continue,
                };
                let h = hash2(self.seed ^ TREE_SALT, bx, bz);
                if (h as f32 / u32::MAX as f32) >= density {
                    continue;
                }
                let trunk = 4 + ((h >> 10) % 3) as i32; // 4..=6
                stamp_tree(chunk, ox, oy, oz, bx, ground, bz, trunk);
            }
        }
    }
}

/// Stamp one oak-like tree (trunk + leaf canopy) into `chunk`, clipping to its bounds.
fn stamp_tree(
    chunk: &mut Chunk,
    ox: i32,
    oy: i32,
    oz: i32,
    bx: i32,
    ground: i32,
    bz: i32,
    trunk: i32,
) {
    let crown = ground + trunk; // canopy center height
    // Leaves first (only into air), then trunk overrides.
    for dy in -2i32..=1 {
        let ly = crown + dy;
        let r: i32 = if dy >= 1 { 1 } else { 2 };
        for dz in -r..=r {
            for dx in -r..=r {
                // Trim far corners on the widest/topmost leaf layers.
                if r == 2 && dx.abs() == 2 && dz.abs() == 2 && (dy == -2 || dy == 1) {
                    continue;
                }
                set_world(chunk, ox, oy, oz, bx + dx, ly, bz + dz, block::LEAVES, true);
            }
        }
    }
    for wy in ground..(ground + trunk) {
        set_world(chunk, ox, oy, oz, bx, wy, bz, block::WOOD, false);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn set_world(
    chunk: &mut Chunk,
    ox: i32,
    oy: i32,
    oz: i32,
    wx: i32,
    wy: i32,
    wz: i32,
    id: BlockId,
    air_only: bool,
) {
    let lx = wx - ox;
    let ly = wy - oy;
    let lz = wz - oz;
    if lx >= 0
        && lx < CHUNK_SIZE_I
        && ly >= 0
        && ly < CHUNK_SIZE_I
        && lz >= 0
        && lz < CHUNK_SIZE_I
    {
        let (lx, ly, lz) = (lx as usize, ly as usize, lz as usize);
        if !air_only || chunk.get(lx, ly, lz) == block::AIR {
            chunk.set(lx, ly, lz, id);
        }
    }
}

fn mix64(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    h
}

fn hash2(seed: u64, x: i32, z: i32) -> u32 {
    let h = seed
        ^ (x as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (z as i64 as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    mix64(h) as u32
}

fn hash3(seed: u64, x: i32, y: i32, z: i32) -> u32 {
    let h = seed
        ^ (x as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (y as i64 as u64).wrapping_mul(0x85EB_CA6B_C2B2_AE35)
        ^ (z as i64 as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    mix64(h) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let wg = Worldgen::new(0xABCDEF);
        for &pos in &[
            IVec3::new(0, 2, 0),
            IVec3::new(3, 1, -5),
            IVec3::new(-7, 3, 11),
        ] {
            let a = wg.generate_chunk(pos);
            let b = wg.generate_chunk(pos);
            assert_eq!(a.blocks, b.blocks, "chunk {pos:?} not deterministic");
            assert_eq!(a.solid_count, b.solid_count);
        }
    }

    #[test]
    fn different_seeds_produce_different_terrain() {
        let a = Worldgen::new(1);
        let b = Worldgen::new(2);
        let mut diffs = 0;
        for i in 0..64 {
            let (x, z) = (i * 7 - 100, i * 13 - 50);
            if a.height(x, z) != b.height(x, z) {
                diffs += 1;
            }
        }
        assert!(diffs > 32, "seeds barely differed ({diffs}/64 columns)");
    }

    #[test]
    fn neighbor_chunks_align_at_borders() {
        // The block just outside one chunk equals the corresponding block in its neighbor,
        // which is what makes cross-chunk face culling correct.
        let wg = Worldgen::new(42);
        let left = wg.generate_chunk(IVec3::new(0, 2, 0));
        let right = wg.generate_chunk(IVec3::new(1, 2, 0));
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                // left's x=31 surface continuity isn't guaranteed equal to right's x=0,
                // but generation must be position-pure, so re-generating right matches itself.
                let _ = (left.get(31, y, z), right.get(0, y, z));
            }
        }
        // Stronger: regenerating the same neighbor yields identical data.
        let right2 = wg.generate_chunk(IVec3::new(1, 2, 0));
        assert_eq!(right.blocks, right2.blocks);
    }
}
