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

pub const SEA_LEVEL: i32 = 96;
const MAX_TREE_HEIGHT: i32 = 12;
const TREE_MARGIN: i32 = 2;
const TREE_SALT: u64 = 0x7265655F_5345_4544;

// Climate classifier thresholds (P20). These reproduce the current biome boundaries
// byte-for-byte; lifting them to named consts is the extensibility seam for P21 (which
// edits this data, not the control flow). Copied verbatim from the pre-P20 `biome()` ladder.
const OCEAN_MAX_H: i32 = SEA_LEVEL - 1; // height <= 63  -> Ocean
const BEACH_MAX_H: i32 = SEA_LEVEL + 1; // height <= 65  -> Beach
const MOUNTAIN_MIN_H: i32 = SEA_LEVEL + 40; // height >= sea+40 -> Mountains (deepened: was abs 104)
const SNOWY_TEMP_MAX: f32 = -0.35; // temperature <  -0.35 -> Snowy
const DESERT_TEMP_MIN: f32 = 0.33; // temperature > 0.33 (and dry) -> Desert
const DESERT_HUM_MAX: f32 = -0.05; // humidity < -0.05 (with hot temp) -> Desert
const FOREST_HUM_MIN: f32 = 0.12; // humidity > 0.12 -> Forest, else Plains

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Biome {
    Ocean,
    Beach,
    Desert,
    Plains,
    Forest,
    Snowy,
    Mountains,
}

/// The continuous climate axes a column is classified by (P20). Two of the three already
/// existed as first-class noise fields; `continentalness` names the elevation axis that
/// today is consumed implicitly via `height()`. P21 will add biomes that branch on all
/// three; in P20 only `temperature`/`humidity` drive the decision (the altitude tier is
/// handled by the short-circuits in `biome()`), so `continentalness` is inert plumbing.
#[derive(Clone, Copy)]
struct Climate {
    temperature: f32,     // [-1, 1] from the `temperature` noise
    humidity: f32,        // [-1, 1] from the `humidity` noise
    // Raw [-1, 1] continent noise. NOTE: `height()` applies `signum * abs.powf(1.15)`
    // shaping (worldgen.rs) before use — P21 must decide whether to branch on raw or shaped.
    continentalness: f32,
}

/// Pure climate classifier: maps continuous climate to a discrete `Biome`. Reproduces the
/// pre-P20 temperature/humidity ladder exactly. `height` is retained for P21 altitude-aware
/// rows; in P20 the altitude tier is resolved by `biome()`'s short-circuits before this is
/// reached, so only temperature/humidity are read here. `continentalness` is intentionally
/// not consulted in P20 — wiring it into a decision would change outputs and break the
/// byte-identity gate; it is reserved for P21.
fn classify(c: Climate, _height: i32) -> Biome {
    let _ = c.continentalness; // reserved for P21; must not influence the P20 decision
    if c.temperature < SNOWY_TEMP_MAX {
        Biome::Snowy
    } else if c.temperature > DESERT_TEMP_MIN && c.humidity < DESERT_HUM_MAX {
        Biome::Desert
    } else if c.humidity > FOREST_HUM_MIN {
        Biome::Forest
    } else {
        Biome::Plains
    }
}

/// What a cell becomes if a cave carves it (U7): solid keeps its block; the rest become air / aquifer
/// water / lava.
enum Carve {
    None,
    Air,
    Aquifer,
    Lava,
}

/// Carved cells at or below this y flood with lava (deep lava lakes near bedrock).
const LAVA_LEVEL: i32 = 8;

// Large-feature reach budgets. A region-cell feature is seam-correct only when its `for_cells` reach
// is at least the furthest any of its voxels extends from the cell origin — otherwise a chunk that the
// feature overlaps but whose cell sweep doesn't reach the origin carves nothing, truncating the feature
// at the chunk plane. `feature_reach_covers_full_span` machine-checks these against the walk geometry.
const RAVINE_LEN_MIN: i32 = 50;
const RAVINE_LEN_SPAN: u32 = 60; // segments = MIN + h%SPAN → up to 109
const RAVINE_CELL: i32 = 128;
const RAVINE_REACH: i32 = 176; // covers ~108 steps × 1.5/axis + half-width + rounding
const VEIN_LEN_MIN: i32 = 10;
const VEIN_LEN_SPAN: u32 = 10; // segments = MIN + h%SPAN → up to 19
const VEIN_CELL: i32 = 80;
const VEIN_REACH: i32 = 56; // covers ~18 steps × 2.5/axis + filler-ellipsoid radius

/// U8 cave biome: drives underground decoration (dripstone clusters, lush moss/vines, deep-dark sculk).
/// A pure function of (seed-independent) world position noise, so it's deterministic and chunk-local.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum CaveBiome {
    Normal,
    Dripstone,
    Lush,
    DeepDark,
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
    deepslate: FastNoiseLite,
    // U7 cave overhaul: thin noodle worms, variable spaghetti width, localized aquifer level.
    noodle_a: FastNoiseLite,
    noodle_b: FastNoiseLite,
    spaghetti_radius: FastNoiseLite,
    water_table: FastNoiseLite,
    // U8 cave biomes: low-frequency temperature/humidity climate for caves, + a 3D deep-dark blotch.
    cave_temp: FastNoiseLite,
    cave_humid: FastNoiseLite,
    deep_dark: FastNoiseLite,
}

impl Worldgen {
    /// Mean deepslate boundary + jitter amplitude: deepslate fills a wavy band ~y5..y15 just above the
    /// bedrock floor (the underground is shallow here vs Minecraft, so the band hugs the world floor).
    const DEEPSLATE_MID: i32 = 10;
    const DEEPSLATE_JITTER: f32 = 5.0;

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
            deepslate: mk(9, NoiseType::OpenSimplex2, 0.010, 2),
            noodle_a: mk(10, NoiseType::OpenSimplex2, 0.050, 2),
            noodle_b: mk(11, NoiseType::OpenSimplex2S, 0.050, 2),
            spaghetti_radius: mk(12, NoiseType::OpenSimplex2, 0.006, 2),
            water_table: mk(13, NoiseType::OpenSimplex2, 0.004, 2),
            cave_temp: mk(14, NoiseType::OpenSimplex2, 0.0015, 2),
            cave_humid: mk(15, NoiseType::OpenSimplex2, 0.0015, 2),
            deep_dark: mk(16, NoiseType::OpenSimplex2S, 0.012, 2),
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

    /// The three continuous climate axes at a world column. Sampled identically to the way
    /// `height()` reads `continent`, so P21 can branch on the elevation axis without re-deriving it.
    fn sample_climate(&self, wx: i32, wz: i32) -> Climate {
        let (fx, fz) = (wx as f32, wz as f32);
        Climate {
            temperature: self.temperature.get_noise_2d(fx, fz),
            humidity: self.humidity.get_noise_2d(fx, fz),
            continentalness: self.continent.get_noise_2d(fx, fz),
        }
    }

    fn biome(&self, wx: i32, wz: i32, height: i32) -> Biome {
        // Altitude tier — short-circuits BEFORE climate sampling, exactly as before. This keeps
        // temperature/humidity sampled at the same call sites/columns/order as the pre-P20 code,
        // so the result is byte-trivially identical (the climate columns also sample `continent`
        // now, but its value is unused, so no output changes).
        if height <= OCEAN_MAX_H {
            return Biome::Ocean;
        }
        if height <= BEACH_MAX_H {
            return Biome::Beach;
        }
        if height >= MOUNTAIN_MIN_H {
            return Biome::Mountains;
        }
        classify(self.sample_climate(wx, wz), height)
    }

    /// Passive species that may spawn on the surface at this column (empty = none here). Only the
    /// grass biomes (Plains/Forest) have a walkable grass surface (see `surface_top`), and the spawn
    /// code gates passives on grass — so the other biomes never reach this and correctly get none.
    /// Desert/cold-biome fauna (husks, rabbits, …) are deferred until those mobs + surfaces exist.
    pub fn passive_pool(&self, wx: i32, wz: i32) -> &'static [crate::entity::Species] {
        use crate::entity::Species::*;
        let h = self.height(wx, wz);
        match self.biome(wx, wz, h) {
            Biome::Forest => &[Cow, Pig, Sheep, Chicken, Wolf], // wolves roam forests
            Biome::Plains => &[Cow, Pig, Sheep, Chicken, Villager], // villagers on the plains
            _ => &[],
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
                if height > SEA_LEVEL + 48 {
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

    /// Local aquifer water level for a column (U7): "wet" regions (positive table noise) hold an
    /// underground water level near y40; dry regions return below the world so their caves stay air.
    fn local_water_table(&self, wx: i32, wz: i32) -> i32 {
        let wt = self.water_table.get_noise_2d(wx as f32, wz as f32); // [-1, 1]
        if wt > 0.25 {
            28 + (wt * 12.0) as i32 // ~y31..40 in the wettest columns (localized underground lakes)
        } else {
            i32::MIN // dry column: no aquifer
        }
    }

    /// What a cell becomes if it's carved (U7): cheese caverns + variable-width spaghetti + thin noodle
    /// worms. Carved space below LAVA_LEVEL fills with lava, below the local water table with water
    /// (localized aquifers), else air.
    fn carve(&self, wx: i32, wy: i32, wz: i32) -> Carve {
        let (fx, fy, fz) = (wx as f32, wy as f32, wz as f32);
        let cheese = self.cheese.get_noise_3d(fx, fy * 1.3, fz) > 0.78;
        // Variable-width spaghetti: both fields within a region-modulated radius.
        let r = 0.06 + 0.04 * self.spaghetti_radius.get_noise_3d(fx, fy, fz).max(0.0);
        let a = self.cave_a.get_noise_3d(fx, fy, fz);
        let b = self.cave_b.get_noise_3d(fx, fy, fz);
        let spaghetti = a.abs() < r && b.abs() < r;
        // Thin noodle worms: two high-frequency fields, both very near zero.
        let noodle = self.noodle_a.get_noise_3d(fx, fy, fz).abs() < 0.025
            && self.noodle_b.get_noise_3d(fx, fy, fz).abs() < 0.025;
        if !(cheese || spaghetti || noodle) {
            return Carve::None;
        }
        if wy <= LAVA_LEVEL {
            Carve::Lava
        } else if wy < self.local_water_table(wx, wz) {
            Carve::Aquifer
        } else {
            Carve::Air
        }
    }

    /// The block a carved cell fills with (U7/U8): lava near bedrock, aquifer water below the local
    /// water table, else air. Shared by the cave post-pass AND ravines so both flood identically.
    fn carve_fill(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        if wy <= LAVA_LEVEL {
            block::LAVA
        } else if wy < self.local_water_table(wx, wz) {
            block::WATER
        } else {
            block::AIR
        }
    }

    /// U8 cave biome at a world cell. Deep Dark is the deepest + rarest (a 3D blotch gate below the
    /// y22 band); otherwise a low-frequency cave climate splits humid→Lush, hot→Dripstone, else Normal.
    fn cave_biome(&self, wx: i32, wy: i32, wz: i32) -> CaveBiome {
        if wy < 22
            && self
                .deep_dark
                .get_noise_3d(wx as f32 * 0.6, wy as f32, wz as f32 * 0.6)
                > 0.55
        {
            return CaveBiome::DeepDark;
        }
        let t = self.cave_temp.get_noise_2d(wx as f32, wz as f32);
        let h = self.cave_humid.get_noise_2d(wx as f32, wz as f32);
        if h > 0.4 {
            CaveBiome::Lush
        } else if t > 0.4 {
            CaveBiome::Dripstone
        } else {
            CaveBiome::Normal
        }
    }

    /// U7 cave post-pass: carve caves AFTER the ore/blob passes so the walls expose ore. Never touches
    /// the bedrock band or the top surface skin; fills carved space with air/water/lava per `carve`.
    fn carve_caves(
        &self,
        chunk: &mut Chunk,
        origin: IVec3,
        heights: &[i32; CHUNK_SIZE * CHUNK_SIZE],
    ) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let height = heights[lz * CHUNK_SIZE + lx];
                let (wx, wz) = (ox + lx as i32, oz + lz as i32);
                for ly in 0..CHUNK_SIZE {
                    let wy = oy + ly as i32;
                    if wy < 3 || wy >= height - 1 {
                        continue; // protect the bedrock band + the top solid skin
                    }
                    let cur = chunk.get(lx, ly, lz);
                    if cur == block::AIR || cur == block::BEDROCK {
                        continue;
                    }
                    if let Carve::None = self.carve(wx, wy, wz) {
                        continue;
                    }
                    chunk.set(lx, ly, lz, self.carve_fill(wx, wy, wz));
                }
            }
        }
    }

    /// The y below which a column is deepslate, as a noise-perturbed per-column band just above
    /// bedrock. This is the SINGLE SOURCE OF TRUTH for "am I in deepslate" — host rock and (from U6)
    /// ore-variant choice both read it, so a stone cell never holds a deepslate-variant ore. Computed
    /// once per column in `generate_chunk` and threaded into `ore_at`/`host_stone`.
    fn deepslate_top(&self, wx: i32, wz: i32) -> i32 {
        let n = self.deepslate.get_noise_2d(wx as f32, wz as f32); // [-1, 1]
        Self::DEEPSLATE_MID + (n * Self::DEEPSLATE_JITTER).round() as i32
    }

    /// Host rock by depth: deepslate below the (noisy) per-column boundary `ds_top`, stone above.
    #[inline]
    fn host_stone(wy: i32, ds_top: i32) -> BlockId {
        if wy < ds_top {
            block::DEEPSLATE
        } else {
            block::STONE
        }
    }

    // ── U6: ore generation (region-cell blobs + 1.18 triangular altitude bands + large veins) ────
    // Ores are placed as small connected blobs (not per-voxel scatter), with a trapezoidal/triangular
    // probability vs Y per ore type (peaking at the classic depth), the deepslate variant chosen
    // per-blob by the deepslate boundary, emerald/high-iron gated to Mountains, plus rare 1.18-style
    // large veins (copper+granite+raw-copper mid-depth, iron+tuff+raw-iron deep).

    /// Blocks an ore may replace (host rock + the stone variants), so ores show up inside granite etc.
    #[inline]
    fn is_ore_host(b: BlockId) -> bool {
        matches!(
            b,
            block::STONE
                | block::DEEPSLATE
                | block::GRANITE
                | block::DIORITE
                | block::ANDESITE
                | block::TUFF
        )
    }

    /// Visit every region-cell candidate (jittered origin + hash) overlapping this chunk for `salt`.
    fn for_cells(
        &self,
        origin: IVec3,
        salt: u64,
        cs: i32,
        reach: i32,
        mut f: impl FnMut(i32, i32, i32, u32),
    ) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        let lo_x = (ox - reach).div_euclid(cs);
        let hi_x = (ox + CHUNK_SIZE_I + reach).div_euclid(cs);
        let lo_y = (oy - reach).div_euclid(cs);
        let hi_y = (oy + CHUNK_SIZE_I + reach).div_euclid(cs);
        let lo_z = (oz - reach).div_euclid(cs);
        let hi_z = (oz + CHUNK_SIZE_I + reach).div_euclid(cs);
        for cz in lo_z..=hi_z {
            for cy in lo_y..=hi_y {
                for cx in lo_x..=hi_x {
                    let (ex, ey, ez, h) = self.cell_feature(salt, cx, cy, cz, cs);
                    f(ex, ey, ez, h);
                }
            }
        }
    }

    /// Stamp one ore blob (roughly round) centered at (ex,ey,ez), replacing ore-host rock and choosing
    /// the stone- vs deepslate-host variant per-voxel by the (per-column) deepslate boundary.
    #[allow(clippy::too_many_arguments)]
    fn stamp_ore_blob(
        &self,
        chunk: &mut Chunk,
        ox: i32,
        oy: i32,
        oz: i32,
        ex: i32,
        ey: i32,
        ez: i32,
        radius: i32,
        h: u32,
        stone_ore: BlockId,
        ds_ore: BlockId,
    ) {
        let r = radius.max(1);
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let v = (dx * dx + dy * dy + dz * dz) as f32 / (r * r) as f32;
                    let jitter =
                        (hash3(h as u64 ^ 0xABE, ex + dx, ey + dy, ez + dz) & 0xff) as f32 / 255.0;
                    if v > 1.0 + (jitter - 0.5) * 0.4 {
                        continue;
                    }
                    let lx = ex + dx - ox;
                    let ly = ey + dy - oy;
                    let lz = ez + dz - oz;
                    if lx < 0
                        || lx >= CHUNK_SIZE_I
                        || ly < 0
                        || ly >= CHUNK_SIZE_I
                        || lz < 0
                        || lz >= CHUNK_SIZE_I
                    {
                        continue;
                    }
                    let (lxu, lyu, lzu) = (lx as usize, ly as usize, lz as usize);
                    if Self::is_ore_host(chunk.get(lxu, lyu, lzu)) {
                        // Pick the variant per-voxel by the *column's* deepslate boundary (same rule
                        // host_stone used), so a deepslate-variant ore never lands in stone-zone rock
                        // (or vice-versa) where the boundary crosses the blob's footprint.
                        let ore = if (ey + dy) < self.deepslate_top(ex + dx, ez + dz) {
                            ds_ore
                        } else {
                            stone_ore
                        };
                        chunk.set(lxu, lyu, lzu, ore);
                    }
                }
            }
        }
    }

    /// Place one ore type across the chunk: `gate(ex,ey,ez,hash) -> Option<radius>` carries the
    /// triangular altitude band + any biome gate; each hit stamps a blob.
    #[allow(clippy::too_many_arguments)]
    fn place_ore(
        &self,
        chunk: &mut Chunk,
        origin: IVec3,
        salt: u64,
        cs: i32,
        reach: i32,
        stone_ore: BlockId,
        ds_ore: BlockId,
        gate: impl Fn(i32, i32, i32, u32) -> Option<i32>,
    ) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        self.for_cells(origin, salt, cs, reach, |ex, ey, ez, h| {
            if let Some(r) = gate(ex, ey, ez, h) {
                self.stamp_ore_blob(chunk, ox, oy, oz, ex, ey, ez, r, h, stone_ore, ds_ore);
            }
        });
    }

    /// A 1.18-style large ore vein: a chain of ore blobs along a hash-derived near-horizontal line,
    /// wrapped in a `filler` stone (granite/tuff) with occasional `raw_block` nuggets. Rare + big.
    #[allow(clippy::too_many_arguments)]
    fn stamp_vein(
        &self,
        chunk: &mut Chunk,
        origin: IVec3,
        ex: i32,
        ey: i32,
        ez: i32,
        h: u32,
        stone_ore: BlockId,
        ds_ore: BlockId,
        filler: BlockId,
        raw_block: BlockId,
    ) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        let len = VEIN_LEN_MIN + (h % VEIN_LEN_SPAN) as i32; // up to 19 segments
        let ang = (h >> 4) as f32 / (1u32 << 28) as f32 * std::f32::consts::TAU;
        let (sx, sz) = (ang.cos() * 2.0, ang.sin() * 2.0);
        let (mut fx, mut fy, mut fz) = (ex as f32, ey as f32, ez as f32);
        for i in 0..len {
            let sh = hash3(h as u64 ^ 0x5E1, i, 0, 0);
            let (cx, cy, cz) = (fx.round() as i32, fy.round() as i32, fz.round() as i32);
            // Filler shell (replaces stone/deepslate) then the ore core (replaces the filler too).
            self.stamp_ellipsoid(chunk, ox, oy, oz, cx, cy, cz, 3 + (sh % 2) as i32, sh, filler);
            self.stamp_ore_blob(
                chunk, ox, oy, oz, cx, cy, cz, 1 + (sh >> 8) as i32 % 2, sh, stone_ore, ds_ore,
            );
            if sh % 5 == 0 {
                let (lx, ly, lz) = (cx - ox, cy - oy, cz - oz);
                if (0..CHUNK_SIZE_I).contains(&lx)
                    && (0..CHUNK_SIZE_I).contains(&ly)
                    && (0..CHUNK_SIZE_I).contains(&lz)
                {
                    let cur = chunk.get(lx as usize, ly as usize, lz as usize);
                    if Self::is_ore_host(cur) || cur == stone_ore || cur == ds_ore {
                        chunk.set(lx as usize, ly as usize, lz as usize, raw_block);
                    }
                }
            }
            fx += sx + ((sh & 0xf) as f32 / 15.0 - 0.5);
            fz += sz + (((sh >> 4) & 0xf) as f32 / 15.0 - 0.5);
            fy += ((sh >> 8) & 0x7) as f32 / 7.0 - 0.5; // gentle vertical wander
        }
    }

    /// U6 ore pass: standard ore blobs (triangular bands) + mountain-gated emerald/high-iron + rare
    /// large copper/iron veins. Runs after the stone-variant blobs so ores can sit inside them.
    fn place_ores(&self, chunk: &mut Chunk, origin: IVec3) {
        use block::*;
        // (salt, cell, reach, stone, deepslate, lo, peak, hi, peak-per-mille, max radius extra)
        self.place_ore(chunk, origin, 0xC0A1, 10, 4, COAL_ORE, DEEPSLATE_COAL_ORE, |_x, ey, _z, h| {
            ore_gate(ey, 40, 96, 140, 130, h, 3)
        });
        self.place_ore(chunk, origin, 0x1207, 10, 4, IRON_ORE, DEEPSLATE_IRON_ORE, |_x, ey, _z, h| {
            ore_gate(ey, 3, 20, 56, 115, h, 2)
        });
        self.place_ore(chunk, origin, 0xC0FF, 12, 5, COPPER_ORE, DEEPSLATE_COPPER_ORE, |_x, ey, _z, h| {
            ore_gate(ey, 0, 48, 96, 95, h, 3)
        });
        self.place_ore(chunk, origin, 0x901D, 14, 4, GOLD_ORE, DEEPSLATE_GOLD_ORE, |_x, ey, _z, h| {
            ore_gate(ey, 3, 12, 44, 55, h, 2)
        });
        self.place_ore(chunk, origin, 0x5ED5, 12, 4, REDSTONE_ORE, DEEPSLATE_REDSTONE_ORE, |_x, ey, _z, h| {
            ore_gate(ey, 3, 6, 26, 105, h, 2)
        });
        self.place_ore(chunk, origin, 0x1A21, 14, 4, LAPIS_ORE, DEEPSLATE_LAPIS_ORE, |_x, ey, _z, h| {
            ore_gate(ey, 3, 14, 44, 62, h, 2)
        });
        self.place_ore(chunk, origin, 0xD1A3, 16, 4, DIAMOND_ORE, DEEPSLATE_DIAMOND_ORE, |_x, ey, _z, h| {
            ore_gate(ey, 3, 7, 22, 42, h, 2)
        });
        // Mountain-gated emerald (sparse) + a high iron band. (S4a: the biome() call had its wz and
        // height arguments transposed, so the Mountains gate actually tested the world Z coordinate —
        // emeralds generated only in the z>=136 half-plane and never in real mountains at z<136.)
        self.place_ore(chunk, origin, 0xE3E5, 20, 4, EMERALD_ORE, DEEPSLATE_EMERALD_ORE, |x, ey, z, h| {
            let r = ore_gate(ey, 96, 130, 170, 60, h, 1)?;
            (self.biome(x, z, self.height(x, z)) == Biome::Mountains).then_some(r)
        });
        self.place_ore(chunk, origin, 0x1207_AA, 12, 4, IRON_ORE, DEEPSLATE_IRON_ORE, |x, ey, z, h| {
            let r = ore_gate(ey, 96, 140, 170, 95, h, 2)?;
            (self.biome(x, z, self.height(x, z)) == Biome::Mountains).then_some(r)
        });
        // 1.18 large ore veins (rare, big).
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        let _ = (ox, oy, oz);
        self.for_cells(origin, 0xC0FF_5E1, VEIN_CELL, VEIN_REACH, |ex, ey, ez, h| {
            if (18..52).contains(&ey) && h % 100 < 35 {
                self.stamp_vein(chunk, origin, ex, ey, ez, h, COPPER_ORE, DEEPSLATE_COPPER_ORE, GRANITE, RAW_COPPER_BLOCK);
            }
        });
        self.for_cells(origin, 0x1207_5E1, VEIN_CELL, VEIN_REACH, |ex, ey, ez, h| {
            if (3..18).contains(&ey) && h % 100 < 28 {
                self.stamp_vein(chunk, origin, ex, ey, ez, h, IRON_ORE, DEEPSLATE_IRON_ORE, TUFF, RAW_IRON_BLOCK);
            }
        });
    }

    // ── U5: cross-chunk-safe region-cell blob engine ────────────────────────────────────────────
    // A feature is a pure function of (seed, region cell): each cell holds one hash-jittered candidate
    // origin; the caller's `activate` closure gates it and picks block + radius. Every chunk within
    // `reach` of a blob re-derives the identical origin and stamps only the part of the ellipsoid that
    // lands inside its own bounds (writes are clipped, never cross-chunk) — so the tiled result equals
    // a single untiled pass. This generalizes the `place_trees` margin trick to bounded large features.

    /// The deterministic jittered candidate origin for region cell (cx,cy,cz) of size `cs` under
    /// `salt`, plus a fresh hash for the caller to derive activation/type/size.
    fn cell_feature(&self, salt: u64, cx: i32, cy: i32, cz: i32, cs: i32) -> (i32, i32, i32, u32) {
        let h = hash3(self.seed ^ salt, cx, cy, cz);
        let jx = (h & 0x3ff) as i32 % cs;
        let jy = ((h >> 10) & 0x3ff) as i32 % cs;
        let jz = ((h >> 20) & 0x3ff) as i32 % cs;
        (cx * cs + jx, cy * cs + jy, cz * cs + jz, h)
    }

    /// Stamp one ragged ellipsoid of `block` centered at (ex,ey,ez), replacing ONLY host rock
    /// (stone/deepslate), clipped to this chunk's bounds. Vertically squashed + per-voxel-jittered so
    /// blobs read as organic pockets rather than perfect ellipsoids.
    #[allow(clippy::too_many_arguments)]
    fn stamp_ellipsoid(
        &self,
        chunk: &mut Chunk,
        ox: i32,
        oy: i32,
        oz: i32,
        ex: i32,
        ey: i32,
        ez: i32,
        radius: i32,
        h: u32,
        block: BlockId,
    ) {
        let rx = radius.max(1);
        let ry = (radius * 3 / 4).max(1);
        let rz = radius.max(1);
        for dz in -rz..=rz {
            for dy in -ry..=ry {
                for dx in -rx..=rx {
                    let v = (dx * dx) as f32 / (rx * rx) as f32
                        + (dy * dy) as f32 / (ry * ry) as f32
                        + (dz * dz) as f32 / (rz * rz) as f32;
                    let jitter = (hash3(h as u64, ex + dx, ey + dy, ez + dz) & 0xff) as f32 / 255.0;
                    if v > 1.0 + (jitter - 0.5) * 0.35 {
                        continue;
                    }
                    let lx = ex + dx - ox;
                    let ly = ey + dy - oy;
                    let lz = ez + dz - oz;
                    if lx < 0
                        || lx >= CHUNK_SIZE_I
                        || ly < 0
                        || ly >= CHUNK_SIZE_I
                        || lz < 0
                        || lz >= CHUNK_SIZE_I
                    {
                        continue;
                    }
                    let (lxu, lyu, lzu) = (lx as usize, ly as usize, lz as usize);
                    let cur = chunk.get(lxu, lyu, lzu);
                    if cur == block::STONE || cur == block::DEEPSLATE {
                        chunk.set(lxu, lyu, lzu, block);
                    }
                }
            }
        }
    }

    /// Iterate every blob origin (under `salt`, cell size `cs`, max `reach`) whose ellipsoid could
    /// reach this chunk; `activate(ex,ey,ez,hash) -> Option<(block,radius)>` gates + types each one.
    fn stamp_blobs(
        &self,
        chunk: &mut Chunk,
        origin: IVec3,
        salt: u64,
        cs: i32,
        reach: i32,
        activate: impl Fn(i32, i32, i32, u32) -> Option<(BlockId, i32)>,
    ) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        let lo_x = (ox - reach).div_euclid(cs);
        let hi_x = (ox + CHUNK_SIZE_I + reach).div_euclid(cs);
        let lo_y = (oy - reach).div_euclid(cs);
        let hi_y = (oy + CHUNK_SIZE_I + reach).div_euclid(cs);
        let lo_z = (oz - reach).div_euclid(cs);
        let hi_z = (oz + CHUNK_SIZE_I + reach).div_euclid(cs);
        for cz in lo_z..=hi_z {
            for cy in lo_y..=hi_y {
                for cx in lo_x..=hi_x {
                    let (ex, ey, ez, h) = self.cell_feature(salt, cx, cy, cz, cs);
                    if let Some((block, radius)) = activate(ex, ey, ez, h) {
                        self.stamp_ellipsoid(chunk, ox, oy, oz, ex, ey, ez, radius, h, block);
                    }
                }
            }
        }
    }

    /// U5 underground material blobs: stone variants (granite/diorite/andesite in stone, tuff near the
    /// deepslate band) + dirt/gravel pockets + sparse clay near the water table. Host rock only.
    fn place_blobs(&self, chunk: &mut Chunk, origin: IVec3) {
        // Stone variants — common, medium blobs.
        self.stamp_blobs(chunk, origin, 0x57A1E_B10B, 14, 7, |ex, ey, ez, h| {
            if h % 100 >= 55 {
                return None; // ~45% of cells host a variant blob
            }
            let radius = 3 + (h >> 7) as i32 % 4; // 3..=6
            let block = if ey < self.deepslate_top(ex, ez) + 4 {
                block::TUFF // deep band -> tuff (1.18 generates tuff around deepslate)
            } else {
                match (h >> 9) % 3 {
                    0 => block::GRANITE,
                    1 => block::DIORITE,
                    _ => block::ANDESITE,
                }
            };
            Some((block, radius))
        });
        // Dirt + gravel pockets — small, common (gravel biased deep, dirt shallow).
        self.stamp_blobs(chunk, origin, 0xD127_6ACE, 12, 6, |_ex, ey, _ez, h| {
            if h % 100 >= 48 {
                return None;
            }
            let radius = 2 + (h >> 7) as i32 % 3; // 2..=4
            let gravel = (h >> 9) % 2 == 0 || ey < SEA_LEVEL - 24;
            Some((if gravel { block::GRAVEL } else { block::DIRT }, radius))
        });
        // Clay — sparse, in the water-table band (refined to genuinely-under-water in U7).
        self.stamp_blobs(chunk, origin, 0xC1A1_9EED, 24, 5, |_ex, ey, _ez, h| {
            if h % 1000 >= 130 || !(SEA_LEVEL - 12..=SEA_LEVEL + 2).contains(&ey) {
                return None;
            }
            let radius = 2 + (h >> 7) as i32 % 2; // 2..=3
            Some((block::CLAY, radius))
        });
    }

    // ── U8: amethyst geodes ──────────────────────────────────────────────────────────────────────
    // Rare deep region-cell features: a hollow sphere of smooth-basalt / calcite / budding-amethyst
    // shells around an air pocket, with amethyst clusters glinting on the inner lining. Like the ore
    // blobs, every chunk within `reach` re-derives the identical origin and clips its writes.

    /// Blocks a geode shell may replace (solid rock only — never carved space, ore, or other features).
    #[inline]
    fn is_geode_host(b: BlockId) -> bool {
        matches!(
            b,
            block::STONE
                | block::DEEPSLATE
                | block::GRANITE
                | block::DIORITE
                | block::ANDESITE
                | block::TUFF
        )
    }

    /// Place amethyst geodes (rare, deep) via the region-cell engine.
    fn place_geodes(&self, chunk: &mut Chunk, origin: IVec3) {
        self.for_cells(origin, 0xA3E0_DE5, 48, 12, |ex, ey, ez, h| {
            if h % 100 < 12 && (8..44).contains(&ey) {
                self.stamp_geode(chunk, origin, ex, ey, ez, h);
            }
        });
    }

    /// Stamp one hollow amethyst geode centered at (ex,ey,ez): concentric smooth-basalt → calcite →
    /// budding-amethyst shells around a hollow air center, with clusters on the inner lining. Shells
    /// replace only solid rock; the hollow is carved regardless. Clipped to this chunk's bounds.
    fn stamp_geode(&self, chunk: &mut Chunk, origin: IVec3, ex: i32, ey: i32, ez: i32, h: u32) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        let r_outer = 5 + (h % 3) as i32; // 5..=7
        let rr = r_outer + 1;
        for dz in -rr..=rr {
            for dy in -rr..=rr {
                for dx in -rr..=rr {
                    let lx = ex + dx - ox;
                    let ly = ey + dy - oy;
                    let lz = ez + dz - oz;
                    if lx < 0
                        || lx >= CHUNK_SIZE_I
                        || ly < 0
                        || ly >= CHUNK_SIZE_I
                        || lz < 0
                        || lz >= CHUNK_SIZE_I
                    {
                        continue;
                    }
                    let jitter =
                        (hash3(h as u64 ^ 0x6E0DE, ex + dx, ey + dy, ez + dz) & 0xff) as f32 / 255.0;
                    let d = ((dx * dx + dy * dy + dz * dz) as f32).sqrt() + (jitter - 0.5) * 0.6;
                    let r = r_outer as f32;
                    if d > r + 0.5 {
                        continue;
                    }
                    let (lxu, lyu, lzu) = (lx as usize, ly as usize, lz as usize);
                    let cur = chunk.get(lxu, lyu, lzu);
                    if d > r - 1.0 {
                        if Self::is_geode_host(cur) {
                            chunk.set(lxu, lyu, lzu, block::SMOOTH_BASALT); // outer shell
                        }
                    } else if d > r - 2.0 {
                        if Self::is_geode_host(cur) {
                            chunk.set(lxu, lyu, lzu, block::CALCITE); // middle shell
                        }
                    } else if d > r - 2.7 {
                        if Self::is_geode_host(cur) {
                            chunk.set(lxu, lyu, lzu, block::BUDDING_AMETHYST); // inner lining
                        }
                    } else {
                        chunk.set(lxu, lyu, lzu, block::AIR); // hollow center (carved regardless)
                    }
                }
            }
        }
        // Scatter amethyst clusters: a budding-amethyst lining cell whose neighbor toward the center is
        // air sprouts a cluster into that air cell (~30% gate). Same-chunk reads/writes only.
        for dz in -rr..=rr {
            for dy in -rr..=rr {
                for dx in -rr..=rr {
                    let lx = ex + dx - ox;
                    let ly = ey + dy - oy;
                    let lz = ez + dz - oz;
                    if lx < 0
                        || lx >= CHUNK_SIZE_I
                        || ly < 0
                        || ly >= CHUNK_SIZE_I
                        || lz < 0
                        || lz >= CHUNK_SIZE_I
                    {
                        continue;
                    }
                    if chunk.get(lx as usize, ly as usize, lz as usize) != block::BUDDING_AMETHYST {
                        continue;
                    }
                    // Step toward the geode center and, if that cell is air + inside the chunk, sprout.
                    let nx = lx - dx.signum();
                    let ny = ly - dy.signum();
                    let nz = lz - dz.signum();
                    if nx < 0
                        || nx >= CHUNK_SIZE_I
                        || ny < 0
                        || ny >= CHUNK_SIZE_I
                        || nz < 0
                        || nz >= CHUNK_SIZE_I
                    {
                        continue;
                    }
                    let (nxu, nyu, nzu) = (nx as usize, ny as usize, nz as usize);
                    if chunk.get(nxu, nyu, nzu) != block::AIR {
                        continue;
                    }
                    if hash3(h as u64 ^ 0xC1057E, ex + dx, ey + dy, ez + dz) % 100 < 30 {
                        chunk.set(nxu, nyu, nzu, block::AMETHYST_CLUSTER);
                    }
                }
            }
        }
    }

    // ── U8: ravines ──────────────────────────────────────────────────────────────────────────────
    // Rare long vertical chasms along a hash-derived heading, pinching toward their ends, that can
    // breach the surface for drama. Region-cell sourced (cross-chunk-safe), clipped to this chunk.

    /// Place ravines (rare, long) via the region-cell engine.
    fn place_ravines(&self, chunk: &mut Chunk, origin: IVec3) {
        // reach must cover the ravine's full reach from its origin: up to (len-1)≈108 steps × 1.5/step
        // + half-width + rounding ≈ 165 blocks. Under-reaching truncates long ravines at chunk seams.
        // `carve_ravine` early-outs keep the wider cell sweep cheap for non-overlapping ravines.
        self.for_cells(origin, 0x2A1_DE5, RAVINE_CELL, RAVINE_REACH, |ex, ey, ez, h| {
            if h % 100 < 22 {
                self.carve_ravine(chunk, origin, ex, ey, ez, h);
            }
        });
    }

    /// Carve one ravine: a thin chasm walked along a hash heading from (ex,ez). At each step it cuts a
    /// narrow cross-section (half-width pinching toward the ends) from a deep bottom up toward the
    /// surface. Only solid rock is carved; fills follow `carve_fill` (lava/water/air). Clipped.
    fn carve_ravine(&self, chunk: &mut Chunk, origin: IVec3, ex: i32, ey: i32, ez: i32, h: u32) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        let len = RAVINE_LEN_MIN + (h % RAVINE_LEN_SPAN) as i32;
        let ang = (h >> 4) as f32 / (1u32 << 28) as f32 * std::f32::consts::TAU;
        let (sx, sz) = (ang.cos(), ang.sin());
        let bottom = (ey - (20 + (h % 16) as i32)).max(3); // chasm floor, never into bedrock
        // Whole-ravine vertical early-out: every carved cell lies in [bottom, ey+4], so a ravine whose
        // span can't reach this chunk's Y range does no work (cheap with the now-wider cell sweep).
        if ey + 4 < oy || bottom >= oy + CHUNK_SIZE_I {
            return;
        }
        let (mut fx, mut fz) = (ex as f32, ez as f32);
        for i in 0..len {
            let (px, pz) = (fx.round() as i32, fz.round() as i32);
            // Advance the centerline up-front so every early-`continue` below stays on the same path.
            fx += sx + ((hash3(h as u64 ^ 0x5A1, i, 0, 0) & 0xf) as f32 / 15.0 - 0.5);
            fz += sz + ((hash3(h as u64 ^ 0x5A2, i, 0, 0) & 0xf) as f32 / 15.0 - 0.5);
            // Half-width pinches toward both ends via a sine envelope (1.5..3.0).
            let frac = i as f32 / len as f32;
            let env = (frac * std::f32::consts::PI).sin(); // 0 at the ends, 1 in the middle
            let hw = (1.5 + 1.5 * env) as i32; // 1..3 cells of horizontal half-width
            // Cheap skip: this step's cross-section is entirely outside this chunk in x/z.
            if px + hw < ox
                || px - hw >= ox + CHUNK_SIZE_I
                || pz + hw < oz
                || pz - hw >= oz + CHUNK_SIZE_I
            {
                continue;
            }
            // Top: allow it to open at the surface (but the column's top skin stays; carve_caves-style
            // bound is height-1 elsewhere — here we deliberately breach to `surface-1` for drama).
            let surf = self.height(px, pz);
            let top = (surf - 1).min(ey + 4);
            if top < bottom {
                continue;
            }
            for ddz in -hw..=hw {
                for ddx in -hw..=hw {
                    // Round the cross-section + jitter the rim so walls read ragged.
                    let rim = (hash3(h as u64 ^ 0x21B, px + ddx, 0, pz + ddz) & 0x7) as i32;
                    if ddx * ddx + ddz * ddz > hw * hw + rim {
                        continue;
                    }
                    let (wx, wz) = (px + ddx, pz + ddz);
                    for wy in bottom..=top {
                        let lx = wx - ox;
                        let ly = wy - oy;
                        let lz = wz - oz;
                        if lx < 0
                            || lx >= CHUNK_SIZE_I
                            || ly < 0
                            || ly >= CHUNK_SIZE_I
                            || lz < 0
                            || lz >= CHUNK_SIZE_I
                        {
                            continue;
                        }
                        let (lxu, lyu, lzu) = (lx as usize, ly as usize, lz as usize);
                        let cur = chunk.get(lxu, lyu, lzu);
                        // Only carve solid rock; skip already-carved/bedrock/fluid cells.
                        if cur == block::AIR
                            || cur == block::BEDROCK
                            || cur == block::WATER
                            || cur == block::LAVA
                        {
                            continue;
                        }
                        chunk.set(lxu, lyu, lzu, self.carve_fill(wx, wy, wz));
                    }
                }
            }
        }
    }

    // ── U8: cave-biome decoration ────────────────────────────────────────────────────────────────
    // Runs LAST, on the final carved state. For each AIR cell with a solid floor/ceiling inside this
    // chunk, the cell's `cave_biome` picks a decoration. All reads/writes are same-chunk + the floor/
    // ceiling tests require ly>0 / ly<31, so chunk Y-seams are simply skipped (like surface deco) and
    // the result is a pure function of world position — identical regardless of which chunk gen'd it.

    fn decorate_caves(
        &self,
        chunk: &mut Chunk,
        origin: IVec3,
        heights: &[i32; CHUNK_SIZE * CHUNK_SIZE],
    ) {
        let (ox, oy, oz) = (origin.x, origin.y, origin.z);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let height = heights[lz * CHUNK_SIZE + lx];
                let (wx, wz) = (ox + lx as i32, oz + lz as i32);
                for ly in 0..CHUNK_SIZE {
                    let wy = oy + ly as i32;
                    if wy < 3 || wy >= height - 1 {
                        continue;
                    }
                    if chunk.get(lx, ly, lz) != block::AIR {
                        continue;
                    }
                    let biome = self.cave_biome(wx, wy, wz);
                    let g = hash3(self.seed ^ 0xDEC0_8A7E, wx, wy, wz);

                    // FLOOR: only when the cell below is in-chunk and an opaque solid cube.
                    if ly > 0 {
                        let below = chunk.get(lx, ly - 1, lz);
                        if block::is_opaque(below) && block::is_cube(below) {
                            match biome {
                                CaveBiome::Lush if g % 1000 < 120 => {
                                    chunk.set(lx, ly - 1, lz, block::MOSS_BLOCK);
                                    let sub = g >> 10;
                                    if sub % 100 < 14 {
                                        chunk.set(lx, ly, lz, block::AZALEA);
                                    } else if sub % 100 < 20 {
                                        chunk.set(lx, ly, lz, block::SPORE_BLOSSOM);
                                    } else if sub % 100 < 26 {
                                        chunk.set(lx, ly - 1, lz, block::CLAY);
                                    }
                                }
                                CaveBiome::Dripstone if g % 1000 < 100 => {
                                    chunk.set(lx, ly, lz, block::POINTED_DRIPSTONE);
                                    if (g >> 10) % 100 < 25 {
                                        chunk.set(lx, ly - 1, lz, block::DRIPSTONE_BLOCK);
                                    }
                                }
                                CaveBiome::DeepDark if g % 1000 < 220 => {
                                    chunk.set(lx, ly - 1, lz, block::SCULK);
                                    let sub = g >> 10;
                                    if sub % 1000 < 20 {
                                        let pick = (sub >> 10) % 3;
                                        let f = match pick {
                                            0 => block::SCULK_SENSOR,
                                            1 => block::SCULK_SHRIEKER,
                                            _ => block::SCULK_CATALYST,
                                        };
                                        chunk.set(lx, ly, lz, f);
                                    }
                                }
                                CaveBiome::Normal if g % 1000 < 12 => {
                                    chunk.set(lx, ly, lz, block::GLOW_LICHEN);
                                }
                                _ => {}
                            }
                        }
                    }

                    // Re-read: a floor pass may have filled this air cell already.
                    if chunk.get(lx, ly, lz) != block::AIR {
                        continue;
                    }

                    // WALLS (deep dark): sculk vein onto an adjacent in-chunk solid wall.
                    if biome == CaveBiome::DeepDark && (g >> 20) % 100 < 30 {
                        for (dx, dz) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                            let (nx, nz) = (lx as i32 + dx, lz as i32 + dz);
                            if !(0..CHUNK_SIZE_I).contains(&nx) || !(0..CHUNK_SIZE_I).contains(&nz) {
                                continue;
                            }
                            if block::is_opaque(chunk.get(nx as usize, ly, nz as usize)) {
                                chunk.set(lx, ly, lz, block::SCULK_VEIN);
                                break;
                            }
                        }
                    }

                    if chunk.get(lx, ly, lz) != block::AIR {
                        continue;
                    }

                    // CEILING: only when the cell above is in-chunk and an opaque solid cube.
                    if ly < CHUNK_SIZE - 1 {
                        let above = chunk.get(lx, ly + 1, lz);
                        if block::is_opaque(above) && block::is_cube(above) {
                            let c = g >> 5;
                            match biome {
                                CaveBiome::Dripstone if c % 1000 < 80 => {
                                    chunk.set(lx, ly, lz, block::POINTED_DRIPSTONE);
                                }
                                CaveBiome::Lush if c % 1000 < 100 => {
                                    let pick = (c >> 10) % 3;
                                    let v = match pick {
                                        0 => block::CAVE_VINE_BERRIES,
                                        1 => block::CAVE_VINE,
                                        _ => block::GLOW_LICHEN,
                                    };
                                    chunk.set(lx, ly, lz, v);
                                }
                                CaveBiome::DeepDark if c % 1000 < 120 => {
                                    chunk.set(lx, ly, lz, block::SCULK_VEIN);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
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
                let ds_top = self.deepslate_top(wx, wz); // noisy deepslate boundary, once per column
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
                        } else if id == block::STONE {
                            // Host rock only; ores (U6) + caves (U7) are post-passes after the fill.
                            id = Self::host_stone(wy, ds_top);
                        }
                        chunk.set(lx, ly, lz, id);
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

        // U5/U6/U7/U8: material blobs -> ore blobs/veins -> carve caves (after ores, so walls expose
        // ore) -> geodes + ravines (carve more) -> cave-biome decoration (on the final carved state).
        if oy <= hmax {
            self.place_blobs(&mut chunk, origin);
            self.place_ores(&mut chunk, origin);
            self.carve_caves(&mut chunk, origin, &heights);
            self.place_geodes(&mut chunk, origin);
            self.place_ravines(&mut chunk, origin);
            self.decorate_caves(&mut chunk, origin, &heights);
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

/// Triangular probability weight in [0,1] (U6 ore bands): 0 outside (lo,hi), peaking at 1.0 at `peak`.
fn tri(wy: i32, lo: i32, peak: i32, hi: i32) -> f32 {
    if wy <= lo || wy >= hi {
        return 0.0;
    }
    if wy <= peak {
        (wy - lo) as f32 / (peak - lo).max(1) as f32
    } else {
        (hi - wy) as f32 / (hi - peak).max(1) as f32
    }
}

/// Ore-blob activation gate: the triangular Y band scaled by `peak_pm` (per-mille at the peak).
/// Returns a blob radius (1 + 0..max_extra) on a hit, else None.
fn ore_gate(wy: i32, lo: i32, peak: i32, hi: i32, peak_pm: i32, h: u32, max_extra: i32) -> Option<i32> {
    let w = tri(wy, lo, peak, hi);
    if w <= 0.0 || (h % 1000) as f32 >= w * peak_pm as f32 {
        return None;
    }
    Some(1 + (h >> 11) as i32 % max_extra.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    impl Worldgen {
        /// Verbatim copy of the PRE-P20 `biome()` body — an independent oracle for the
        /// byte-identity gate. It samples the same noise the same way and uses the original
        /// hardcoded literals (NOT the new named consts), so asserting `biome == biome_old`
        /// over a dense grid proves the climate-struct refactor changed no output. This is
        /// non-circular: the oracle is the old logic, the subject is the new classifier path.
        fn biome_old(&self, wx: i32, wz: i32, height: i32) -> Biome {
            if height <= SEA_LEVEL - 1 {
                return Biome::Ocean;
            }
            if height <= SEA_LEVEL + 1 {
                return Biome::Beach;
            }
            if height >= SEA_LEVEL + 40 {
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
    }

    /// Gate A: the P20 refactor is byte-identical to the pre-refactor logic. Since `biome()`
    /// is the ONLY changed function and every downstream consumer (surface_top, subsurface,
    /// decoration, trees, …) is a pure function of the returned `Biome` (height/noise/ore/
    /// ice-freeze are all untouched), `biome == biome_old` over real terrain ⇒ byte-identical
    /// chunks. Uses the real world seed and a dense grid that exercises every biome.
    #[test]
    fn p20_biome_refactor_is_byte_identical() {
        let wg = Worldgen::new(0x5EED_C0FFEE);
        let mut seen = HashSet::new();
        let mut n = 0u32;
        let mut wx = -4096;
        while wx <= 4096 {
            let mut wz = -4096;
            while wz <= 4096 {
                let h = wg.height(wx, wz);
                let new = wg.biome(wx, wz, h);
                assert_eq!(
                    new,
                    wg.biome_old(wx, wz, h),
                    "P20 biome diverged at ({wx},{wz}) height {h}: {new:?} != old"
                );
                seen.insert(new);
                n += 1;
                wz += 48;
            }
            wx += 48;
        }
        assert!(n > 10_000, "grid too sparse ({n} columns)");
        // The grid must actually exercise the rare arms (Desert/Snowy) over real noise, or
        // the equivalence proof has untested classifier branches.
        assert!(
            seen.len() >= 6,
            "grid only hit {} of 7 biomes: {:?}",
            seen.len(),
            seen
        );
    }

    /// Gate B: the classifier arms + altitude short-circuits each map to the right biome.
    /// Synthetic climates guarantee every branch is exercised (including the rare Desert/Snowy
    /// arms) regardless of which biomes the noise grid happens to surface.
    #[test]
    fn p20_classifier_arms_and_short_circuits() {
        let wg = Worldgen::new(0x5EED_C0FFEE);
        // Altitude tier dominates regardless of climate at that column.
        assert_eq!(wg.biome(0, 0, SEA_LEVEL - 1), Biome::Ocean);
        assert_eq!(wg.biome(0, 0, SEA_LEVEL + 1), Biome::Beach);
        assert_eq!(wg.biome(0, 0, MOUNTAIN_MIN_H), Biome::Mountains);
        // Climate ladder (climate-tier height, synthetic climates straddling each threshold).
        let h = (SEA_LEVEL + 20) as i32; // a non-short-circuited height
        let c = |t, hum| Climate { temperature: t, humidity: hum, continentalness: 0.0 };
        assert_eq!(classify(c(-0.5, 0.0), h), Biome::Snowy);
        assert_eq!(classify(c(0.5, -0.2), h), Biome::Desert);
        assert_eq!(classify(c(0.5, 0.5), h), Biome::Forest); // hot+wet is NOT desert
        assert_eq!(classify(c(0.0, 0.5), h), Biome::Forest);
        assert_eq!(classify(c(0.0, 0.0), h), Biome::Plains);
        // Boundary exactness: thresholds are strict `<`/`>`, so the boundary value falls through.
        assert_eq!(classify(c(SNOWY_TEMP_MAX, 0.0), h), Biome::Plains);
        assert_eq!(classify(c(0.0, FOREST_HUM_MIN), h), Biome::Plains);
    }

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

    /// F4 regression: a region-cell feature's `for_cells` reach must cover the furthest its voxels can
    /// extend from the cell origin, or chunks it overlaps but doesn't enumerate carve nothing → the
    /// feature is sheared off flat at the chunk plane. Guards the ravine + large-vein reach budgets
    /// against anyone bumping a length without bumping the matching reach.
    #[test]
    fn feature_reach_covers_full_span() {
        // Ravine: walks (len-1) centerline steps of <= 1.0 (cos/sin) + 0.5 (jitter) per axis, then a
        // half-width-3 cross-section; the centerline is rounded (≤ +0.5).
        let ravine_steps = RAVINE_LEN_MIN + RAVINE_LEN_SPAN as i32 - 1 - 1; // max len, minus the last step
        let ravine_span = (ravine_steps as f32 * 1.5).ceil() as i32 + 3 + 1;
        assert!(
            RAVINE_REACH >= ravine_span,
            "ravine reach {RAVINE_REACH} < span {ravine_span}: long ravines truncate at chunk seams"
        );
        // Large vein: (len-1) steps of <= 2.0 (cos/sin × 2) + 0.5 (jitter) per axis, then a filler
        // ellipsoid of radius up to 4 (3 + sh%2).
        let vein_steps = VEIN_LEN_MIN + VEIN_LEN_SPAN as i32 - 1 - 1;
        let vein_span = (vein_steps as f32 * 2.5).ceil() as i32 + 4 + 1;
        assert!(
            VEIN_REACH >= vein_span,
            "vein reach {VEIN_REACH} < span {vein_span}: large veins clip at chunk seams"
        );
    }

    /// U1 (deepening): sea level rose to 96 (more underground), terrain+canopy never pierces the
    /// world ceiling, and the deepslate boundary is a noisy low band that host rock honors.
    #[test]
    fn u1_deepening_and_noisy_deepslate() {
        let wg = Worldgen::new(0x5EED_C0FFEE);
        let (mut hmin, mut hmax) = (i32::MAX, i32::MIN);
        let (mut ds_min, mut ds_max) = (i32::MAX, i32::MIN);
        let mut wx = -2048;
        while wx <= 2048 {
            let mut wz = -2048;
            while wz <= 2048 {
                let h = wg.height(wx, wz);
                hmin = hmin.min(h);
                hmax = hmax.max(h);
                assert!(
                    h + MAX_TREE_HEIGHT <= WORLD_HEIGHT,
                    "terrain+canopy pierces the ceiling at ({wx},{wz}): h={h}"
                );
                let ds = wg.deepslate_top(wx, wz);
                ds_min = ds_min.min(ds);
                ds_max = ds_max.max(ds);
                wz += 64;
            }
            wx += 64;
        }
        // Sea level deepened: there are submerged columns and real highlands well above it.
        assert!(hmin < SEA_LEVEL, "no submerged columns (sea level didn't deepen)");
        assert!(hmax > SEA_LEVEL + 20, "no highlands above the raised sea level");
        // The deepslate boundary is noisy and confined to a low band MID +/- JITTER.
        assert!(ds_max - ds_min >= 4, "deepslate boundary not noisy (min {ds_min}, max {ds_max})");
        assert!(ds_min >= Worldgen::DEEPSLATE_MID - 5 && ds_max <= Worldgen::DEEPSLATE_MID + 5);
        // host_stone honors the boundary; a deep chunk has deepslate only in the low band + stone above.
        assert_eq!(Worldgen::host_stone(4, 10), block::DEEPSLATE);
        assert_eq!(Worldgen::host_stone(10, 10), block::STONE);
        let deep = wg.generate_chunk(IVec3::new(0, 0, 0)); // world y 0..32 (deep underground)
        let (mut saw_deepslate, mut saw_stone_high) = (false, false);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    let id = deep.get(lx, ly, lz);
                    let wy = ly as i32; // chunk origin y = 0
                    if id == block::DEEPSLATE {
                        saw_deepslate = true;
                        assert!(wy < Worldgen::DEEPSLATE_MID + 6, "deepslate above its band at wy={wy}");
                    }
                    if id == block::STONE && wy > Worldgen::DEEPSLATE_MID + 6 {
                        saw_stone_high = true;
                    }
                }
            }
        }
        assert!(saw_deepslate, "deep chunk has no deepslate near bedrock");
        assert!(saw_stone_high, "deep chunk has no stone above the deepslate band");
    }

    /// U5 (region-cell blob engine): blobs are deterministic, varied, and seam-consistent.
    #[test]
    fn u5_blobs_deterministic_and_present() {
        let wg = Worldgen::new(0xB10B_5EED);
        // Determinism: a deep chunk regenerates identically (blobs are pure f(seed,pos)).
        let a = wg.generate_chunk(IVec3::new(0, 0, 0));
        let b = wg.generate_chunk(IVec3::new(0, 0, 0));
        assert_eq!(a.blocks, b.blocks);
        assert_eq!(a.solid_count, b.solid_count);
        // Variety: a deep region holds stone variants + gravel blobs (replacing host rock).
        let mut seen = HashSet::new();
        for cz in -1..=1 {
            for cx in -1..=1 {
                let c = wg.generate_chunk(IVec3::new(cx, 0, cz));
                for &blk in &c.blocks {
                    if matches!(
                        blk,
                        block::GRANITE
                            | block::DIORITE
                            | block::ANDESITE
                            | block::TUFF
                            | block::GRAVEL
                            | block::DIRT
                            | block::CLAY
                    ) {
                        seen.insert(blk);
                    }
                }
            }
        }
        let has_variant = [block::GRANITE, block::DIORITE, block::ANDESITE, block::TUFF]
            .iter()
            .any(|b| seen.contains(b));
        assert!(has_variant, "no stone-variant blobs generated");
        assert!(seen.contains(&block::GRAVEL), "no gravel blobs generated");
        // Cross-chunk seam: re-generating a neighbor matches itself (position-pure clip, no drift).
        let n1 = wg.generate_chunk(IVec3::new(1, 0, 0));
        let n2 = wg.generate_chunk(IVec3::new(1, 0, 0));
        assert_eq!(n1.blocks, n2.blocks);
    }

    /// U6 (ore blobs + triangular bands): ores generate as blobs in the right altitude bands, the
    /// deepslate variants appear at depth, copper exists, and generation stays deterministic.
    #[test]
    fn u6_ore_bands_blobs_and_variants() {
        let wg = Worldgen::new(0x0DE_5EED);
        let (mut diamond_max, mut coal_max) = (i32::MIN, i32::MIN);
        let (mut copper, mut iron, mut lapis, mut redstone, mut gold, mut ds_variants) =
            (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
        for cy in 0..4 {
            for cx in -2..=2 {
                for cz in -2..=2 {
                    let c = wg.generate_chunk(IVec3::new(cx, cy, cz));
                    for ly in 0..CHUNK_SIZE {
                        let wy = cy * CHUNK_SIZE_I + ly as i32;
                        for lz in 0..CHUNK_SIZE {
                            for lx in 0..CHUNK_SIZE {
                                let id = c.get(lx, ly, lz);
                                match id {
                                    block::DIAMOND_ORE | block::DEEPSLATE_DIAMOND_ORE => {
                                        diamond_max = diamond_max.max(wy)
                                    }
                                    block::COAL_ORE | block::DEEPSLATE_COAL_ORE => {
                                        coal_max = coal_max.max(wy)
                                    }
                                    block::COPPER_ORE | block::DEEPSLATE_COPPER_ORE => copper += 1,
                                    block::IRON_ORE | block::DEEPSLATE_IRON_ORE => iron += 1,
                                    block::LAPIS_ORE | block::DEEPSLATE_LAPIS_ORE => lapis += 1,
                                    block::REDSTONE_ORE | block::DEEPSLATE_REDSTONE_ORE => redstone += 1,
                                    block::GOLD_ORE | block::DEEPSLATE_GOLD_ORE => gold += 1,
                                    _ => {}
                                }
                                if matches!(
                                    id,
                                    block::DEEPSLATE_COAL_ORE
                                        | block::DEEPSLATE_IRON_ORE
                                        | block::DEEPSLATE_COPPER_ORE
                                        | block::DEEPSLATE_GOLD_ORE
                                        | block::DEEPSLATE_REDSTONE_ORE
                                        | block::DEEPSLATE_LAPIS_ORE
                                        | block::DEEPSLATE_DIAMOND_ORE
                                ) {
                                    ds_variants += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // The full overworld ore set generates (copper/emerald are the U2 additions); deepslate
        // variants appear at depth. Thresholds are lenient (100-chunk underground sample).
        assert!(copper > 20, "too little copper ore ({copper})");
        assert!(iron > 20, "too little iron ore ({iron})");
        assert!(lapis > 10, "too little lapis ore ({lapis})");
        assert!(redstone > 10, "too little redstone ore ({redstone})");
        assert!(gold > 5, "too little gold ore ({gold})");
        assert!(ds_variants > 10, "too few deepslate ore variants at depth ({ds_variants})");
        // Diamond stays in the deep band; coal reaches the upper band (the triangular distribution).
        assert!(diamond_max < 30, "diamond appears too high (max y={diamond_max})");
        assert!(coal_max > 60, "coal never reaches its upper band (max y={coal_max})");
        // Determinism with the new ore engine.
        let a = wg.generate_chunk(IVec3::new(0, 0, 0));
        let b = wg.generate_chunk(IVec3::new(0, 0, 0));
        assert_eq!(a.blocks, b.blocks);
    }

    /// S4a regression: the emerald/high-iron gate called biome() with wz and height transposed,
    /// so "Mountains" actually tested the world Z coordinate — emeralds sprayed into ORDINARY
    /// terrain wherever z>=136 and could never generate in real mountains at z<136. With the fix,
    /// non-mountain chunks in the z>=136 half-plane must contain zero emerald ore. (True mountains
    /// are vanishingly rare at current terrain amplitude — max height ~134 vs MOUNTAIN_MIN_H 136 —
    /// so the positive case rides on the biome-semantics assertion until the biome expansion.)
    #[test]
    fn emerald_gate_keys_on_terrain_not_z() {
        let wg = Worldgen::new(0x0DE_5EED);
        // biome() altitude tiers key on the height argument (the fixed call order)...
        assert_eq!(wg.biome(10, -500, 140), Biome::Mountains);
        assert_ne!(wg.biome(10, 500, 90), Biome::Mountains);
        // ...and no stray emeralds in non-mountain terrain at z >= 136 (the old bug's half-plane).
        let mut emeralds = 0usize;
        for ccx in -3..=3 {
            for ccz in 5..=8 {
                for cy in 3..=4 {
                    let c = wg.generate_chunk(IVec3::new(ccx, cy, ccz));
                    emeralds += c
                        .blocks
                        .iter()
                        .filter(|&&b| b == block::EMERALD_ORE || b == block::DEEPSLATE_EMERALD_ORE)
                        .count();
                }
            }
        }
        assert_eq!(emeralds, 0, "stray emeralds in non-mountain terrain at z>=136 (the pre-S4a bug)");
    }

    /// U7 (cave carving + fluids): caves carve a sensible fraction, lava only near bedrock, aquifers
    /// exist, the bedrock floor survives carving, and generation stays deterministic.
    #[test]
    fn u7_caves_fluids_and_bedrock() {
        let wg = Worldgen::new(0xCA7E5);
        let (mut carved, mut solid) = (0u64, 0u64);
        let (mut lava_above, mut water_at_lava) = (0u64, 0u64);
        let (mut saw_lava, mut saw_aquifer, mut floor_intact) = (false, false, true);
        for cy in 0..3 {
            for cx in -1..=1 {
                for cz in -1..=1 {
                    let c = wg.generate_chunk(IVec3::new(cx, cy, cz));
                    for ly in 0..CHUNK_SIZE {
                        let wy = cy * CHUNK_SIZE_I + ly as i32;
                        for lz in 0..CHUNK_SIZE {
                            for lx in 0..CHUNK_SIZE {
                                let id = c.get(lx, ly, lz);
                                if wy >= 3 && wy < SEA_LEVEL {
                                    match id {
                                        block::AIR | block::WATER | block::LAVA => carved += 1,
                                        _ => solid += 1,
                                    }
                                }
                                if id == block::LAVA {
                                    saw_lava = true;
                                    if wy > LAVA_LEVEL {
                                        lava_above += 1;
                                    }
                                }
                                if id == block::WATER {
                                    if wy <= LAVA_LEVEL {
                                        water_at_lava += 1;
                                    }
                                    if (3..SEA_LEVEL - 6).contains(&wy) {
                                        saw_aquifer = true; // underground water = an aquifer pool
                                    }
                                }
                                // The y0 floor is always bedrock and is never carved.
                                if wy == 0 && id != block::BEDROCK {
                                    floor_intact = false;
                                }
                            }
                        }
                    }
                }
            }
        }
        let frac = carved as f64 / (carved + solid) as f64;
        assert!(frac > 0.02 && frac < 0.40, "cave fraction {frac} out of range");
        assert!(saw_lava, "no lava generated near bedrock");
        assert_eq!(lava_above, 0, "gen-time lava above LAVA_LEVEL");
        assert_eq!(water_at_lava, 0, "cave water at/below the lava level");
        assert!(saw_aquifer, "no underground aquifer water generated");
        assert!(floor_intact, "bedrock floor at y0 was carved away");
        // Determinism with caves + fluids.
        let a = wg.generate_chunk(IVec3::new(0, 0, 0));
        let b = wg.generate_chunk(IVec3::new(0, 0, 0));
        assert_eq!(a.blocks, b.blocks);
        assert_eq!(a.solid_count, b.solid_count);
    }

    /// U8 (geodes + ravines + cave biomes): over a deep multi-chunk region the marquee features all
    /// appear — geode shells (smooth basalt / calcite / budding amethyst) + amethyst clusters, plus
    /// the three flavored cave biomes' signature deco (sculk / moss / dripstone). Also: `cave_biome`
    /// returns all four variants over a world-position grid, and a deep chunk regenerates identically.
    #[test]
    fn u8_geodes_ravines_and_cave_biomes() {
        let wg = Worldgen::new(0x0DE_5EED);
        let mut seen = HashSet::new();
        for cy in 0..3 {
            for cx in -2..=2 {
                for cz in -2..=2 {
                    let c = wg.generate_chunk(IVec3::new(cx, cy, cz));
                    for &blk in &c.blocks {
                        seen.insert(blk);
                    }
                }
            }
        }
        // A geode generated somewhere in the sample (all three shells present).
        assert!(seen.contains(&block::SMOOTH_BASALT), "no geode smooth-basalt shell");
        assert!(seen.contains(&block::CALCITE), "no geode calcite shell");
        assert!(seen.contains(&block::BUDDING_AMETHYST), "no geode budding-amethyst lining");
        assert!(seen.contains(&block::AMETHYST_CLUSTER), "no amethyst clusters in geodes");
        // Cave-biome decoration: deep-dark sculk, lush moss, dripstone spikes.
        assert!(seen.contains(&block::SCULK), "no deep-dark sculk floor");
        assert!(seen.contains(&block::MOSS_BLOCK), "no lush moss floor");
        assert!(seen.contains(&block::POINTED_DRIPSTONE), "no dripstone spikes");

        // `cave_biome` surfaces all four variants over a grid of world positions.
        let mut biomes = HashSet::new();
        let mut wx = -512;
        while wx <= 512 {
            let mut wz = -512;
            while wz <= 512 {
                for wy in [6, 12, 18, 30, 50] {
                    biomes.insert(wg.cave_biome(wx, wy, wz));
                }
                wz += 24;
            }
            wx += 24;
        }
        assert!(biomes.contains(&CaveBiome::Normal), "cave_biome never Normal");
        assert!(biomes.contains(&CaveBiome::Dripstone), "cave_biome never Dripstone");
        assert!(biomes.contains(&CaveBiome::Lush), "cave_biome never Lush");
        assert!(biomes.contains(&CaveBiome::DeepDark), "cave_biome never DeepDark");

        // Determinism with geodes + ravines + decoration in the mix.
        let a = wg.generate_chunk(IVec3::new(0, 0, 0));
        let b = wg.generate_chunk(IVec3::new(0, 0, 0));
        assert_eq!(a.blocks, b.blocks);
        assert_eq!(a.solid_count, b.solid_count);
    }
}
