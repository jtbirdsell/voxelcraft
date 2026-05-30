//! Procedural texture atlas (M13). A deterministic 8x8 grid of 16x16-px tiles painted in code at
//! startup (no external assets). Authored as `Rgba8Unorm` with per-tile base colors equal to the
//! WGSL `tile_average()` values, so switching the surface shaders from the flat average to atlas
//! sampling is a detail-only upgrade (same mean brightness, no GI mismatch). Mips arrive in M13c.

use crate::block::tile as T;

pub const TILE_PX: u32 = 16;
pub const ATLAS_COLS: u32 = 8;
pub const ATLAS_ROWS: u32 = 8;
pub const ATLAS_W: u32 = ATLAS_COLS * TILE_PX; // 128
pub const ATLAS_H: u32 = ATLAS_ROWS * TILE_PX; // 128

/// Bake the full atlas to a tightly-packed `Rgba8Unorm` image (`y*ATLAS_W + x`, 4 bytes/texel).
pub fn build_atlas() -> Vec<u8> {
    let mut img = vec![0u8; (ATLAS_W * ATLAS_H * 4) as usize];
    for tile in 0..(ATLAS_COLS * ATLAS_ROWS) {
        let ox = (tile % ATLAS_COLS) * TILE_PX;
        let oy = (tile / ATLAS_COLS) * TILE_PX;
        for ty in 0..TILE_PX {
            for tx in 0..TILE_PX {
                let [r, g, b, a] = paint(tile, tx, ty);
                let o = (((oy + ty) * ATLAS_W + (ox + tx)) * 4) as usize;
                img[o] = r;
                img[o + 1] = g;
                img[o + 2] = b;
                img[o + 3] = a;
            }
        }
    }
    img
}

/// Base color per tile (matches `tile_average()` in rtx_common.wgsl).
fn base_color(tile: u32) -> [f32; 3] {
    match tile {
        T::STONE => [0.49, 0.49, 0.52],
        T::DIRT => [0.45, 0.33, 0.21],
        T::GRASS_TOP => [0.36, 0.60, 0.27],
        T::GRASS_SIDE => [0.42, 0.42, 0.24],
        T::SAND => [0.80, 0.75, 0.52],
        T::WOOD_TOP => [0.55, 0.43, 0.27],
        T::WOOD_SIDE => [0.40, 0.30, 0.18],
        T::LEAVES => [0.20, 0.42, 0.18],
        T::WATER => [0.16, 0.34, 0.62],
        T::SNOW => [0.92, 0.94, 0.97],
        T::COAL => [0.28, 0.28, 0.30],
        T::IRON => [0.60, 0.52, 0.45],
        T::LAVA => [1.0, 0.42, 0.06],
        T::MOB => [0.86, 0.55, 0.58],
        T::MOB_HEAD => [0.80, 0.50, 0.52],
        T::TORCH => [0.85, 0.62, 0.28],
        T::GLOWSTONE => [0.95, 0.82, 0.45],
        T::COBBLE => [0.42, 0.42, 0.44],
        T::PLANKS => [0.62, 0.48, 0.30],
        T::BRICKS => [0.55, 0.28, 0.22],
        T::BEDROCK => [0.20, 0.20, 0.22],
        T::GRAVEL => [0.50, 0.47, 0.45],
        T::OBSIDIAN => [0.12, 0.09, 0.18],
        T::GOLD => [0.55, 0.50, 0.35],
        T::DIAMOND => [0.45, 0.62, 0.62],
        T::REDSTONE => [0.45, 0.30, 0.30],
        T::LAPIS => [0.30, 0.35, 0.55],
        T::DEEPSLATE => [0.22, 0.22, 0.25],
        T::CRAFTING => [0.50, 0.36, 0.22],
        T::FURNACE => [0.38, 0.38, 0.40],
        T::CHEST => [0.55, 0.42, 0.24],
        T::POPPY => [0.80, 0.15, 0.12],
        T::DANDELION => [0.90, 0.82, 0.20],
        T::TALL_GRASS => [0.34, 0.55, 0.24],
        T::CACTUS => [0.30, 0.52, 0.24],
        _ => [1.0, 0.0, 1.0],
    }
}

/// Cross-billboard plant tiles: an RGBA cutout (alpha 0 outside the plant shape) so the X quads
/// read as flowers/grass, not solid squares.
fn paint_plant(tile: u32, x: u32, y: u32) -> [u8; 4] {
    let dx = x as i32 - 8;
    let stem = (7..=8).contains(&x) && y >= 6;
    let leaves = (9..=11).contains(&y) && (1..=2).contains(&dx.abs());
    match tile {
        T::TALL_GRASS => {
            let here = hashf(x, 0, 5) > 0.35;
            let top = 3 + (hashf(x, 1, 9) * 6.0) as u32;
            if here && y >= top {
                let g = hashf(x, y, 2);
                return [to_u8(0.20), to_u8(0.42 + g * 0.20), to_u8(0.15), 255];
            }
            [0, 0, 0, 0]
        }
        T::POPPY => {
            let head = y < 7 && (dx.abs() + (y as i32 - 3).abs()) <= 3;
            if head {
                return [to_u8(0.82), to_u8(0.12), to_u8(0.10), 255];
            }
            if stem || leaves {
                return [to_u8(0.18), to_u8(0.42), to_u8(0.16), 255];
            }
            [0, 0, 0, 0]
        }
        T::DANDELION => {
            let head = y < 6 && (dx.abs() + (y as i32 - 3).abs()) <= 3;
            if head {
                return [to_u8(0.92), to_u8(0.82), to_u8(0.18), 255];
            }
            if stem || leaves {
                return [to_u8(0.18), to_u8(0.42), to_u8(0.16), 255];
            }
            [0, 0, 0, 0]
        }
        _ => [255, 0, 255, 255],
    }
}

/// Stone-host ore tile: speckled stone with colored ore flecks.
fn ore(fleck: [f32; 3], x: u32, y: u32) -> [f32; 3] {
    let host = base_color(T::STONE);
    let n = hashf(x, y, 77);
    if blob(x, y, 3) {
        fleck
    } else {
        shade(host, (n - 0.5) * 0.14)
    }
}

/// Paint one texel of a tile. Detail is brightness variation around the base color (mean-preserving),
/// plus a few per-tile motifs (ore specks, bark columns, wood rings, a grassy top strip).
fn paint(tile: u32, x: u32, y: u32) -> [u8; 4] {
    if matches!(tile, T::POPPY | T::DANDELION | T::TALL_GRASS) {
        return paint_plant(tile, x, y);
    }
    let base = base_color(tile);
    let n = hashf(x, y, tile.wrapping_mul(131) + 7); // 0..1 fine grain
    let m = hashf(x / 2, y / 2, tile.wrapping_mul(977) + 3); // 0..1 coarse grain

    let mut c = base;
    match tile {
        T::STONE => {
            c = shade(base, (n - 0.5) * 0.16);
            if hashf(x, y, 41) > 0.94 {
                c = shade(base, -0.28); // pebble/crack
            }
        }
        T::DIRT => {
            c = shade(base, (n - 0.5) * 0.22);
            if hashf(x, y, 53) > 0.90 {
                c = shade(base, -0.20);
            }
        }
        T::GRASS_TOP => {
            c = shade(base, (n - 0.5) * 0.26);
            if hashf(x, y, 61) > 0.80 {
                c = shade(base, 0.18); // brighter blades
            }
        }
        T::GRASS_SIDE => {
            // Dirt body with a grassy green strip along the top few rows + a ragged fringe.
            let grass = [0.34, 0.58, 0.26];
            let dirt = [0.45, 0.33, 0.21];
            let fringe = 3 + (hashf(x, 0, 71) * 2.0) as u32;
            if y < fringe {
                c = shade(grass, (n - 0.5) * 0.22);
            } else {
                c = shade(dirt, (n - 0.5) * 0.20);
            }
        }
        T::SAND => {
            c = shade(base, (n - 0.5) * 0.12);
        }
        T::WOOD_TOP => {
            // Concentric rings around the tile center.
            let dx = x as f32 - 7.5;
            let dy = y as f32 - 7.5;
            let r = (dx * dx + dy * dy).sqrt();
            let ring = ((r * 1.4).sin() * 0.5 + 0.5) * 0.20 - 0.10;
            c = shade(base, ring + (n - 0.5) * 0.06);
        }
        T::WOOD_SIDE => {
            // Vertical bark: darker grooves on a few columns.
            let groove = matches!(x % 5, 0 | 3);
            let d = if groove { -0.22 } else { 0.0 };
            c = shade(base, d + (m - 0.5) * 0.16);
        }
        T::LEAVES => {
            c = shade(base, (n - 0.5) * 0.34);
            if hashf(x, y, 83) > 0.86 {
                c = shade(base, -0.30); // gaps/shadow
            }
        }
        T::WATER => {
            c = shade(base, (m - 0.5) * 0.10);
        }
        T::SNOW => {
            c = shade(base, (n - 0.5) * 0.06);
        }
        T::COAL => {
            // Stone host with black coal blobs.
            let stone = base_color(T::STONE);
            c = shade(stone, (n - 0.5) * 0.14);
            if blob(x, y, 0) {
                c = [0.07, 0.07, 0.08];
            }
        }
        T::IRON => {
            let stone = base_color(T::STONE);
            c = shade(stone, (n - 0.5) * 0.14);
            if blob(x, y, 9) {
                c = [0.74, 0.58, 0.42]; // ore fleck
            }
        }
        T::LAVA => {
            // Cracked crust: brighter molten cells separated by dark seams.
            let cell = ((x / 4) ^ (y / 4)).wrapping_mul(2654435761);
            let hot = (cell & 7) as f32 / 7.0;
            c = shade(base, hot * 0.25 - 0.05);
            if (x % 4 == 0 || y % 4 == 0) && hashf(x, y, 17) > 0.4 {
                c = [0.45, 0.10, 0.02]; // dark seam
            }
        }
        T::MOB | T::MOB_HEAD => {
            c = shade(base, (n - 0.5) * 0.10);
        }
        T::COBBLE => {
            c = shade(base, (n - 0.5) * 0.30);
            if blob(x, y, 7) {
                c = shade(base, -0.35);
            }
        }
        T::DEEPSLATE => {
            c = shade(base, (n - 0.5) * 0.22);
            if hashf(x, y, 31) > 0.9 {
                c = shade(base, -0.30);
            }
        }
        T::GRAVEL => {
            c = shade(base, (n - 0.5) * 0.28);
        }
        T::BEDROCK => {
            c = shade(base, (n - 0.5) * 0.55);
        }
        T::OBSIDIAN => {
            c = shade(base, (m - 0.5) * 0.16);
            if hashf(x, y, 13) > 0.85 {
                c = [0.22, 0.14, 0.30];
            }
        }
        T::PLANKS => {
            let groove = y % 8 == 0;
            let d = if groove { -0.28 } else { 0.0 };
            c = shade(base, d + (m - 0.5) * 0.14);
        }
        T::BRICKS => {
            let row = y / 4;
            let offset = (row % 2) * 4;
            let bx = (x + offset) % 8;
            if y % 4 == 0 || bx == 0 {
                c = [0.80, 0.76, 0.72]; // mortar
            } else {
                c = shade(base, (n - 0.5) * 0.12);
            }
        }
        T::GOLD => c = ore([0.95, 0.78, 0.25], x, y),
        T::DIAMOND => c = ore([0.45, 0.92, 0.90], x, y),
        T::REDSTONE => c = ore([0.85, 0.12, 0.12], x, y),
        T::LAPIS => c = ore([0.15, 0.28, 0.85], x, y),
        T::CRAFTING => {
            c = shade(base, (m - 0.5) * 0.16);
            if x < 2 || y < 2 || x > 13 {
                c = shade(base, -0.22);
            }
        }
        T::FURNACE => {
            c = shade(base, (m - 0.5) * 0.14);
            if y > 9 && x > 3 && x < 12 {
                c = [0.10, 0.08, 0.08]; // dark front opening
            }
        }
        T::CHEST => {
            c = shade(base, (m - 0.5) * 0.14);
            if (7..=8).contains(&y) {
                c = [0.30, 0.22, 0.10]; // latch band
            }
        }
        T::GLOWSTONE => {
            // Lumpy yellow rock with bright nodules.
            c = shade(base, (m - 0.5) * 0.20);
            if blob(x, y, 4) {
                c = [1.0, 0.95, 0.7];
            }
        }
        T::TORCH => {
            // Brown stick lower, bright flame near the top.
            if y < 5 {
                let flame = [1.0, 0.85, 0.35];
                c = shade(flame, (n - 0.5) * 0.10);
            } else {
                let stick = [0.45, 0.32, 0.16];
                c = shade(stick, (m - 0.5) * 0.18);
            }
        }
        T::CACTUS => {
            c = shade(base, (m - 0.5) * 0.16);
            if x % 7 == 0 {
                c = shade(base, -0.24); // vertical ribs
            }
        }
        _ => {}
    }
    [to_u8(c[0]), to_u8(c[1]), to_u8(c[2]), 255]
}

/// A small clustered "ore blob" mask: true on a few deterministic 2x2-ish clumps.
fn blob(x: u32, y: u32, salt: u32) -> bool {
    // Three clump centers per tile; a texel is in a clump if close to one.
    for i in 0..3u32 {
        let cx = (hashf(i, 0, salt + 1) * 12.0) as i32 + 2;
        let cy = (hashf(i, 1, salt + 5) * 12.0) as i32 + 2;
        let dx = x as i32 - cx;
        let dy = y as i32 - cy;
        if dx * dx + dy * dy <= 2 {
            return true;
        }
    }
    false
}

#[inline]
fn shade(c: [f32; 3], delta: f32) -> [f32; 3] {
    [c[0] * (1.0 + delta), c[1] * (1.0 + delta), c[2] * (1.0 + delta)]
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Integer hash → [0,1).
#[inline]
fn hashf(x: u32, y: u32, salt: u32) -> f32 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(salt.wrapping_mul(2246822519));
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}
