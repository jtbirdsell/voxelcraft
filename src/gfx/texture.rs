//! Procedural texture atlas (M13). A deterministic 16x16 grid of 16x16-px tiles painted in code at
//! startup (no external assets). Authored as `Rgba8Unorm` with per-tile flat base colors (the single
//! source of truth for a tile's color, mirrored only by item::material_color for UI swatches).
//! P19 (Phase 5): grew the grid 8x8=64 → 16x16=256 — the per-tile pixel content is unchanged (TILE_PX
//! and paint() are grid-independent), only the grid gained free slots for new biome/wood/ore tiles.
//! No mip chain (mip_level_count=1); nearest-sampled at full res.

use crate::block::tile as T;

pub const TILE_PX: u32 = 16;
pub const ATLAS_COLS: u32 = 16;
pub const ATLAS_ROWS: u32 = 16;
pub const ATLAS_W: u32 = ATLAS_COLS * TILE_PX; // 256
pub const ATLAS_H: u32 = ATLAS_ROWS * TILE_PX; // 256

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
        T::FERN => [0.27, 0.46, 0.20],
        T::RED_MUSHROOM => [0.80, 0.20, 0.18],
        T::BROWN_MUSHROOM => [0.55, 0.40, 0.28],
        T::SUGAR_CANE => [0.55, 0.72, 0.40],
        T::PUMPKIN_TOP => [0.80, 0.52, 0.14],
        T::PUMPKIN_SIDE => [0.78, 0.48, 0.12],
        T::ICE => [0.66, 0.80, 0.92],
        T::GLASS => [0.82, 0.91, 0.98],
        T::MOB_COW => [0.42, 0.30, 0.20],
        T::MOB_PIG => [0.88, 0.55, 0.58],
        T::MOB_SHEEP => [0.90, 0.90, 0.88],
        T::MOB_CHICKEN => [0.95, 0.95, 0.90],
        T::MOB_ZOMBIE => [0.32, 0.55, 0.34],
        T::MOB_SKELETON => [0.82, 0.82, 0.78],
        T::MOB_CREEPER => [0.36, 0.66, 0.34],
        T::MOB_SPIDER => [0.20, 0.17, 0.20],
        T::MOB_WOLF => [0.72, 0.72, 0.74], // grey wolf fur (P18)
        // Mob-drop material tiles (51..=61) map back to the item's flat color (single source of truth).
        t if (T::MATERIAL_DROP..T::MATERIAL_DROP + 11).contains(&t) => {
            crate::item::material_color(crate::item::BEEF + (t - T::MATERIAL_DROP) as u16)
        }
        // U2 ores: tile averages (must match block::face_color + the rtx_common.wgsl voxel_color cases).
        T::COPPER => [0.58, 0.49, 0.44],
        T::EMERALD => [0.46, 0.58, 0.49],
        T::DS_COAL => [0.20, 0.20, 0.22],
        T::DS_IRON => [0.34, 0.30, 0.30],
        T::DS_COPPER => [0.36, 0.30, 0.29],
        T::DS_GOLD => [0.36, 0.33, 0.27],
        T::DS_REDSTONE => [0.33, 0.23, 0.25],
        T::DS_EMERALD => [0.24, 0.36, 0.30],
        T::DS_LAPIS => [0.22, 0.26, 0.38],
        T::DS_DIAMOND => [0.27, 0.37, 0.39],
        // U3 storage blocks (must match block::face_color + voxel_color).
        T::IRON_BLOCK => [0.86, 0.84, 0.80],
        T::GOLD_BLOCK => [0.96, 0.82, 0.25],
        T::DIAMOND_BLOCK => [0.40, 0.85, 0.84],
        T::EMERALD_BLOCK => [0.22, 0.78, 0.42],
        T::LAPIS_BLOCK => [0.18, 0.32, 0.72],
        T::REDSTONE_BLOCK => [0.72, 0.12, 0.10],
        T::COPPER_BLOCK => [0.78, 0.46, 0.34],
        T::COAL_BLOCK => [0.10, 0.10, 0.12],
        T::RAW_IRON_BLOCK => [0.72, 0.58, 0.46],
        T::RAW_COPPER_BLOCK => [0.74, 0.46, 0.30],
        T::RAW_GOLD_BLOCK => [0.82, 0.66, 0.28],
        // U3 stone family.
        T::TUFF => [0.42, 0.43, 0.40],
        T::CALCITE => [0.90, 0.90, 0.88],
        T::GRANITE => [0.66, 0.45, 0.38],
        T::POLISHED_GRANITE => [0.68, 0.47, 0.40],
        T::DIORITE => [0.84, 0.84, 0.85],
        T::POLISHED_DIORITE => [0.86, 0.86, 0.87],
        T::ANDESITE => [0.55, 0.56, 0.57],
        T::POLISHED_ANDESITE => [0.57, 0.58, 0.59],
        T::CLAY => [0.62, 0.64, 0.70],
        T::DRIPSTONE => [0.55, 0.42, 0.34],
        T::SMOOTH_BASALT => [0.28, 0.27, 0.30],
        T::COBBLED_DEEPSLATE => [0.26, 0.26, 0.29],
        T::POLISHED_DEEPSLATE => [0.24, 0.24, 0.27],
        T::DEEPSLATE_BRICKS => [0.23, 0.23, 0.26],
        T::DEEPSLATE_TILES => [0.22, 0.22, 0.25],
        T::CUT_COPPER => [0.78, 0.46, 0.34],
        _ => [1.0, 0.0, 1.0],
    }
}

/// Glass: a near-transparent pale-blue pane with a more opaque border + the odd glint, so the
/// alpha-blended glass pass reads as a framed sheet of glass you can see through.
fn paint_glass(x: u32, y: u32) -> [u8; 4] {
    let frame = x == 0 || y == 0 || x == 15 || y == 15;
    if frame {
        return [191, 217, 242, 235]; // light, mostly opaque border
    }
    let glint = hashf(x, y, 23) > 0.92;
    let a = if glint { 110 } else { 36 };
    [to_u8(0.82), to_u8(0.91), to_u8(0.98), a]
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
        T::FERN => {
            // A bushy triangular spray of fronds, widest at the top, on a short stem.
            let spread = ((y as i32) - 3).max(0) / 2 + 1;
            let frond = (4..=14).contains(&y) && dx.abs() <= spread && (x ^ y) % 2 == 0;
            let stalk = (7..=8).contains(&x) && y >= 9;
            if frond || stalk {
                let g = hashf(x, y, 3);
                return [to_u8(0.16), to_u8(0.38 + g * 0.18), to_u8(0.14), 255];
            }
            [0, 0, 0, 0]
        }
        T::RED_MUSHROOM => {
            let stalk = (7..=8).contains(&x) && (9..=13).contains(&y);
            let cap = dx * dx + (y as i32 - 8) * (y as i32 - 8) <= 25 && y <= 8;
            if cap {
                if hashf(x, y, 6) > 0.80 {
                    return [to_u8(0.96), to_u8(0.96), to_u8(0.92), 255]; // white spots
                }
                return [to_u8(0.82), to_u8(0.16), to_u8(0.14), 255];
            }
            if stalk {
                return [to_u8(0.93), to_u8(0.91), to_u8(0.83), 255];
            }
            [0, 0, 0, 0]
        }
        T::BROWN_MUSHROOM => {
            let stalk = (7..=8).contains(&x) && (10..=13).contains(&y);
            let cap = dx * dx + (y as i32 - 9) * (y as i32 - 9) * 2 <= 20 && y <= 9;
            if cap {
                let d = hashf(x, y, 8) * 0.12;
                return [to_u8(0.50 + d), to_u8(0.36 + d), to_u8(0.24), 255];
            }
            if stalk {
                return [to_u8(0.85), to_u8(0.80), to_u8(0.70), 255];
            }
            [0, 0, 0, 0]
        }
        T::SUGAR_CANE => {
            // A jointed vertical green stalk filling the tile height.
            if (6..=9).contains(&x) {
                let joint = y % 5 == 0;
                let base = if joint { 0.44 } else { 0.58 };
                let g = hashf(x, y, 4) * 0.10;
                return [to_u8(base * 0.78), to_u8(base + 0.16 + g), to_u8(base * 0.52), 255];
            }
            [0, 0, 0, 0]
        }
        _ => [255, 0, 255, 255],
    }
}

/// Host-parameterized ore tile: chunky colored lumps (a brighter core + a darker rim so the lumps
/// read as faceted gems/metal) on an arbitrary host rock. Stone ores pass STONE; deepslate ores (U2)
/// pass the DEEPSLATE host with the SAME fleck color, so the variant reads as "the same ore in slate".
fn ore_on(host: [f32; 3], fleck: [f32; 3], x: u32, y: u32, salt: u32) -> [f32; 3] {
    let n = hashf(x, y, 77);
    match ore_lump(x, y, salt) {
        2 => shade(fleck, 0.12 + (n - 0.5) * 0.10), // bright core
        1 => shade(fleck, -0.22),                   // darker rim, for depth
        _ => shade(host, (n - 0.5) * 0.14),
    }
}

/// Stone-host ore tile (the common case): chunky colored lumps on a stone background.
fn ore(fleck: [f32; 3], x: u32, y: u32, salt: u32) -> [f32; 3] {
    ore_on(base_color(T::STONE), fleck, x, y, salt)
}

/// Solid metal/gem storage block (U3): flat base, a beveled darker border + a faint inner highlight,
/// and a touch of grain so the face doesn't read as a perfectly flat swatch.
fn metal_block(base: [f32; 3], x: u32, y: u32) -> [f32; 3] {
    if x == 0 || y == 0 || x == 15 || y == 15 {
        return shade(base, -0.28); // beveled edge
    }
    if x == 1 || y == 1 {
        return shade(base, 0.16); // top-left inner highlight
    }
    shade(base, (hashf(x, y, 91) - 0.5) * 0.08)
}

/// Raw-material block (U3): chunky nuggets (ore_lump cores) on a darker matrix of the same metal.
fn raw_block(base: [f32; 3], x: u32, y: u32, salt: u32) -> [f32; 3] {
    match ore_lump(x, y, salt) {
        2 => shade(base, 0.18),
        1 => shade(base, -0.08),
        _ => shade(base, -0.28),
    }
}

/// Two-tone speckled stone (U3 granite/diorite/andesite): base with scattered lighter + darker flecks.
fn speckle(base: [f32; 3], x: u32, y: u32, salt: u32) -> [f32; 3] {
    let h = hashf(x, y, salt);
    if h > 0.86 {
        shade(base, 0.20)
    } else if h < 0.14 {
        shade(base, -0.22)
    } else {
        shade(base, (hashf(x, y, salt + 1) - 0.5) * 0.12)
    }
}

/// Mortar/seam grid (U3): `square` = a square tile grid; else a running-bond brick grid. Returns true
/// on a seam texel (caller draws mortar), false on the face interior.
fn grid(x: u32, y: u32, square: bool, cell: u32) -> bool {
    if square {
        x % cell == 0 || y % cell == 0
    } else {
        let row = y / cell;
        let bx = (x + (row % 2) * (cell / 2)) % cell;
        y % cell == 0 || bx == 0
    }
}

/// Ore-lump mask with a rim: 2 = lump core, 1 = lump edge, 0 = host stone. Chunkier than a single
/// speck so ore blocks read like Minecraft veins.
fn ore_lump(x: u32, y: u32, salt: u32) -> u8 {
    let mut best = 0u8;
    for i in 0..5u32 {
        let cx = (hashf(i, 0, salt + 1) * 11.0) as i32 + 2;
        let cy = (hashf(i, 1, salt + 5) * 11.0) as i32 + 2;
        let dx = x as i32 - cx;
        let dy = y as i32 - cy;
        let d2 = dx * dx + dy * dy;
        if d2 <= 2 {
            return 2; // core
        } else if d2 <= 5 {
            best = best.max(1); // rim
        }
    }
    best
}

/// Paint one texel of a tile. Detail is brightness variation around the base color (mean-preserving),
/// plus a few per-tile motifs (ore specks, bark columns, wood rings, a grassy top strip).
fn paint(tile: u32, x: u32, y: u32) -> [u8; 4] {
    if matches!(
        tile,
        T::POPPY
            | T::DANDELION
            | T::TALL_GRASS
            | T::FERN
            | T::RED_MUSHROOM
            | T::BROWN_MUSHROOM
            | T::SUGAR_CANE
    ) {
        return paint_plant(tile, x, y);
    }
    if tile == T::GLASS {
        return paint_glass(x, y);
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
        T::COAL => c = ore([0.10, 0.10, 0.11], x, y, 0),
        T::IRON => c = ore([0.78, 0.62, 0.45], x, y, 9),
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
        T::MOB_COW | T::MOB_PIG | T::MOB_SHEEP | T::MOB_CHICKEN | T::MOB_ZOMBIE
        | T::MOB_SKELETON | T::MOB_CREEPER | T::MOB_SPIDER | T::MOB_WOLF => {
            c = shade(base, (n - 0.5) * 0.12); // subtle grain on the flat body color
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
        T::GOLD => c = ore([0.95, 0.78, 0.25], x, y, 3),
        T::DIAMOND => c = ore([0.45, 0.92, 0.90], x, y, 14),
        T::REDSTONE => c = ore([0.85, 0.12, 0.12], x, y, 21),
        T::LAPIS => c = ore([0.15, 0.28, 0.85], x, y, 28),
        // U2: copper/emerald on stone, and a deepslate-host variant of every ore (same flecks on slate).
        T::COPPER => c = ore([0.85, 0.45, 0.30], x, y, 33),
        T::EMERALD => c = ore([0.20, 0.80, 0.40], x, y, 37),
        T::DS_COAL => c = ore_on(base_color(T::DEEPSLATE), [0.10, 0.10, 0.11], x, y, 40),
        T::DS_IRON => c = ore_on(base_color(T::DEEPSLATE), [0.78, 0.62, 0.45], x, y, 49),
        T::DS_COPPER => c = ore_on(base_color(T::DEEPSLATE), [0.85, 0.45, 0.30], x, y, 73),
        T::DS_GOLD => c = ore_on(base_color(T::DEEPSLATE), [0.95, 0.78, 0.25], x, y, 43),
        T::DS_REDSTONE => c = ore_on(base_color(T::DEEPSLATE), [0.85, 0.12, 0.12], x, y, 61),
        T::DS_EMERALD => c = ore_on(base_color(T::DEEPSLATE), [0.20, 0.80, 0.40], x, y, 77),
        T::DS_LAPIS => c = ore_on(base_color(T::DEEPSLATE), [0.15, 0.28, 0.85], x, y, 68),
        T::DS_DIAMOND => c = ore_on(base_color(T::DEEPSLATE), [0.45, 0.92, 0.90], x, y, 54),
        // U3 storage blocks: solid metal/gem faces; raw blocks are lumpy nuggets.
        T::IRON_BLOCK | T::GOLD_BLOCK | T::DIAMOND_BLOCK | T::EMERALD_BLOCK | T::LAPIS_BLOCK
        | T::REDSTONE_BLOCK | T::COPPER_BLOCK | T::COAL_BLOCK => c = metal_block(base, x, y),
        T::RAW_IRON_BLOCK => c = raw_block(base, x, y, 12),
        T::RAW_COPPER_BLOCK => c = raw_block(base, x, y, 18),
        T::RAW_GOLD_BLOCK => c = raw_block(base, x, y, 24),
        // U3 stone family.
        T::TUFF => {
            c = shade(base, (n - 0.5) * 0.18);
            if hashf(x, y, 45) > 0.88 {
                c = shade(base, -0.22);
            }
        }
        T::GRANITE => c = speckle(base, x, y, 51),
        T::DIORITE => c = speckle(base, x, y, 57),
        T::ANDESITE => c = speckle(base, x, y, 63),
        // Polished stones + calcite read smooth: low-amplitude grain only.
        T::CALCITE | T::POLISHED_GRANITE | T::POLISHED_DIORITE | T::POLISHED_ANDESITE
        | T::POLISHED_DEEPSLATE | T::SMOOTH_BASALT => c = shade(base, (n - 0.5) * 0.07),
        T::CLAY => c = shade(base, (m - 0.5) * 0.07),
        T::DRIPSTONE => {
            c = shade(base, (m - 0.5) * 0.18);
            if x % 6 == 0 {
                c = shade(base, -0.16); // faint vertical flutes
            }
        }
        T::COBBLED_DEEPSLATE => {
            c = shade(base, (n - 0.5) * 0.30);
            if blob(x, y, 7) {
                c = shade(base, -0.35);
            }
        }
        T::DEEPSLATE_BRICKS => {
            c = if grid(x, y, false, 8) {
                shade(base, -0.45) // dark mortar
            } else {
                shade(base, (n - 0.5) * 0.10)
            };
        }
        T::DEEPSLATE_TILES => {
            c = if grid(x, y, true, 8) {
                shade(base, -0.45)
            } else {
                shade(base, (n - 0.5) * 0.10)
            };
        }
        T::CUT_COPPER => {
            c = if grid(x, y, true, 8) {
                shade(base, -0.28) // copper seam
            } else {
                shade(base, (n - 0.5) * 0.08)
            };
        }
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
        T::PUMPKIN_TOP => {
            c = shade(base, (n - 0.5) * 0.10);
            if (7..=8).contains(&x) && (7..=8).contains(&y) {
                c = [0.40, 0.30, 0.12]; // stem nub
            } else if x % 4 == 0 {
                c = shade(base, -0.16); // ridges
            }
        }
        T::PUMPKIN_SIDE => {
            c = shade(base, (m - 0.5) * 0.10);
            if x % 4 == 0 {
                c = shade(base, -0.20); // vertical ribs
            }
        }
        T::ICE => {
            c = shade(base, (m - 0.5) * 0.08);
            if hashf(x, y, 19) > 0.92 {
                c = [0.86, 0.93, 0.99]; // glint / hairline crack
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_dimensions_power_of_two() {
        // P19: 16x16 grid of 16px tiles = 256x256.
        assert_eq!((ATLAS_COLS, ATLAS_ROWS), (16, 16));
        assert_eq!((ATLAS_W, ATLAS_H), (256, 256));
        assert_eq!(build_atlas().len(), (ATLAS_W * ATLAS_H * 4) as usize);
        assert!(ATLAS_W.is_power_of_two() && ATLAS_H.is_power_of_two(), "PoT for any future mips");
        // An unassigned slot reads the magenta missing-texture color (the base_color `_` fallback).
        // (Tiles 64+ are being filled by the underground overhaul; pick a far slot that stays free.)
        assert_eq!(crate::block::tile::MAGENTA, 63);
        assert_eq!(base_color(200), [1.0, 0.0, 1.0]);
    }

    #[test]
    fn shader_grid_matches_cpu() {
        // The grid dimension is duplicated in atlas.wgsl as a WGSL const (no uniform plumbing). Guard
        // against the two drifting: a resize that updates one file but not the other garbles every tile.
        let wgsl = include_str!("../../assets/shaders/atlas.wgsl");
        assert!(
            wgsl.contains(&format!("ATLAS_COLS: u32 = {}u", ATLAS_COLS)),
            "atlas.wgsl ATLAS_COLS must equal texture.rs ATLAS_COLS ({ATLAS_COLS})"
        );
        assert!(wgsl.contains(&format!("ATLAS_COLS_F: f32 = {:.1}", ATLAS_COLS as f32)));
        assert!(wgsl.contains(&format!("ATLAS_ROWS_F: f32 = {:.1}", ATLAS_ROWS as f32)));
    }

    /// GI-sync guard (U2): every OPAQUE block that lands in the RTX voxel volume must have a `case Nu:`
    /// in rtx_common.wgsl `voxel_color`, or it GI-bounces the default grey. Non-opaque blocks store 0
    /// in the volume and are correctly absent. Fails loudly the moment a new opaque block forgets its
    /// case — closing the long-standing "new ore bounces grey" gap permanently.
    #[test]
    fn voxel_color_covers_every_opaque_block() {
        let wgsl = include_str!("../../assets/shaders/rtx_common.wgsl");
        // Isolate the voxel_color switch body so case ids from other functions aren't counted (notably
        // voxel_emission, which also has `Nu` literals).
        let start = wgsl.find("fn voxel_color").expect("voxel_color present");
        let rest = &wgsl[start..];
        let end = rest[1..].find("\nfn ").map(|e| e + 1).unwrap_or(rest.len());
        let body = rest[..end].as_bytes();
        // Collect every integer N that appears as a `Nu` case label (handles `case 40u, 42u, ...:`).
        let mut covered = std::collections::HashSet::new();
        let mut i = 0;
        while i < body.len() {
            if body[i].is_ascii_digit() {
                let mut j = i;
                while j < body.len() && body[j].is_ascii_digit() {
                    j += 1;
                }
                let is_u = j < body.len() && body[j] == b'u';
                if is_u {
                    if let Ok(n) = std::str::from_utf8(&body[i..j]).unwrap().parse::<u16>() {
                        covered.insert(n);
                    }
                }
                i = if is_u { j + 1 } else { j };
            } else {
                i += 1;
            }
        }
        for id in 1..=crate::block::MAX_BLOCK {
            if crate::block::is_opaque(id) {
                assert!(
                    covered.contains(&id),
                    "rtx_common.wgsl voxel_color is missing opaque block id {} ({})",
                    id,
                    crate::block::display_name(id)
                );
            }
        }
    }
}
