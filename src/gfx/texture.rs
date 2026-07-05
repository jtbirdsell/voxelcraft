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
        // U4 cave-biome opaque cubes (must match block::face_color + voxel_color). The cross tiles
        // (cluster/buds/dripstone/vines/lichen/azalea/dripleaves/roots/spore/sculk-vein) are RGBA
        // cutouts painted by paint_plant and never read base_color, so they're intentionally absent.
        T::AMETHYST_BLOCK => [0.55, 0.40, 0.78],
        T::BUDDING_AMETHYST => [0.58, 0.42, 0.80],
        T::MOSS_BLOCK => [0.30, 0.42, 0.20],
        T::ROOTED_DIRT => [0.42, 0.32, 0.22],
        T::AZALEA_LEAVES => [0.30, 0.46, 0.22],
        T::SCULK => [0.06, 0.09, 0.11],
        T::SCULK_SENSOR => [0.10, 0.20, 0.24],
        T::SCULK_SHRIEKER => [0.12, 0.16, 0.18],
        T::SCULK_CATALYST => [0.10, 0.14, 0.16],
        T::REINFORCED_DEEPSLATE => [0.20, 0.21, 0.24],
        // S6 farming (FARMLAND_TOP must match block::face_color(FARMLAND) + wgsl voxel_color).
        T::FARMLAND_TOP => [0.35, 0.24, 0.15],
        T::CROP_WHEAT_YOUNG => [0.40, 0.62, 0.28],
        T::CROP_WHEAT_MATURE => [0.75, 0.65, 0.30],
        T::CROP_SPROUT => [0.36, 0.58, 0.26],
        T::CROP_CARROT_MATURE => [0.35, 0.55, 0.22],
        T::CROP_POTATO_MATURE => [0.38, 0.52, 0.26],
        T::FOOD_POTATO => [0.85, 0.70, 0.40],
        T::FOOD_BAKED_POTATO => [0.75, 0.55, 0.30],
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
        // S6 crops. Young stages: short green shoots rising from the soil line. Mature wheat: full
        // golden stalks with heads; mature carrots/potatoes: bushy tops (the produce is underground).
        T::CROP_WHEAT_YOUNG | T::CROP_SPROUT => {
            let here = matches!(x % 4, 1 | 3);
            let top = 10 + (hashf(x, 1, 11) * 3.0) as u32;
            if here && y >= top {
                let g = hashf(x, y, 4);
                let warm = if tile == T::CROP_WHEAT_YOUNG { 0.06 } else { 0.0 };
                return [to_u8(0.22 + warm), to_u8(0.50 + g * 0.18), to_u8(0.16), 255];
            }
            [0, 0, 0, 0]
        }
        T::CROP_WHEAT_MATURE => {
            if matches!(x % 4, 1 | 2) && y >= 2 {
                let head = y <= 6 && hashf(x, y, 9) > 0.35;
                return if head {
                    [to_u8(0.92), to_u8(0.80), to_u8(0.34), 255]
                } else {
                    [to_u8(0.74), to_u8(0.62), to_u8(0.26), 255]
                };
            }
            [0, 0, 0, 0]
        }
        T::CROP_CARROT_MATURE | T::CROP_POTATO_MATURE => {
            let carrot = tile == T::CROP_CARROT_MATURE;
            // Bushy foliage rows; a hint of the crop at the soil line.
            let bush = y >= 6 && (x + y) % 2 == 0 && hashf(x, y, 7) > 0.25;
            if bush {
                let g = hashf(x, y, 8);
                return [to_u8(0.18), to_u8(0.44 + g * 0.20), to_u8(0.14), 255];
            }
            if y >= 14 && x % 5 == 2 {
                return if carrot {
                    [to_u8(0.92), to_u8(0.52), to_u8(0.16), 255]
                } else {
                    [to_u8(0.80), to_u8(0.66), to_u8(0.38), 255]
                };
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
        // ── U4 cave-biome cross-billboard cutouts ──────────────────────────────────────────────────
        // Amethyst cluster + the 3 buds: purple crystal spikes rising from the bottom. Taller/more
        // spikes the larger the stage (cluster > large > medium > small). Lighter near the tips.
        T::AMETHYST_CLUSTER | T::LARGE_AMETHYST_BUD | T::MEDIUM_AMETHYST_BUD
        | T::SMALL_AMETHYST_BUD => {
            // Per-stage spike height (rows from the bottom the crystals reach up to) + spike count.
            let (height, spikes): (u32, u32) = match tile {
                T::AMETHYST_CLUSTER => (13, 4),
                T::LARGE_AMETHYST_BUD => (10, 3),
                T::MEDIUM_AMETHYST_BUD => (7, 2),
                _ => (4, 2), // small
            };
            let top = 15u32.saturating_sub(height); // spikes occupy y in top..=15
            // A spike is a centered triangle around each spike center; half-width shrinks with height.
            for s in 0..spikes {
                let cx = 2 + (hashf(s, 0, 90) * 12.0) as i32;
                let h = top + (hashf(s, 1, 90) * 3.0) as u32; // slight per-spike base variation
                if y >= h {
                    let from_base = (15 - y) as i32; // 0 at bottom, grows upward
                    let half = (from_base / 3) + 1; // tapers to a point near the tip
                    if (x as i32 - cx).abs() <= (3 - half).max(0) {
                        // Brighter toward the tip (smaller from_base near top).
                        let tip = (from_base as f32 / height as f32).min(1.0);
                        let r = 0.55 + tip * 0.18;
                        let g = 0.40 + tip * 0.16;
                        let b = 0.80 + tip * 0.15;
                        return [to_u8(r), to_u8(g), to_u8(b), 255];
                    }
                }
            }
            [0, 0, 0, 0]
        }
        T::POINTED_DRIPSTONE => {
            // A centered brown/tan cone spike (drawn pointing down: widest at the top, tip at bottom).
            let half = (y as i32 / 3) + 1; // narrow at the bottom tip, wide at the top
            if (x as i32 - 8).abs() <= half && half <= 6 {
                let d = hashf(x, y, 12) * 0.12;
                return [to_u8(0.55 + d), to_u8(0.42 + d), to_u8(0.34), 255];
            }
            [0, 0, 0, 0]
        }
        T::CAVE_VINE | T::CAVE_VINE_BERRIES => {
            // Thin green strands hanging from the top, with optional orange glowing berry dots.
            let strand = matches!(x, 4 | 8 | 11) && y <= 13;
            if strand {
                let g = hashf(x, y, 4);
                return [to_u8(0.24), to_u8(0.46 + g * 0.16), to_u8(0.16), 255];
            }
            if tile == T::CAVE_VINE_BERRIES {
                // Berry dots clustered along the strands.
                let berry = matches!((x, y), (4, 6) | (8, 9) | (11, 4) | (8, 13));
                if berry || (matches!(x, 3 | 5 | 7 | 9 | 10 | 12) && (y == 7 || y == 10)) {
                    return [to_u8(0.98), to_u8(0.66), to_u8(0.18), 255]; // glowing berry
                }
            }
            [0, 0, 0, 0]
        }
        T::GLOW_LICHEN => {
            // Scattered pale cyan-green lichen blotches (low coverage).
            if blob(x, y, 21) || hashf(x, y, 35) > 0.86 {
                let g = hashf(x, y, 7) * 0.12;
                return [to_u8(0.50), to_u8(0.78 + g), to_u8(0.66), 255];
            }
            [0, 0, 0, 0]
        }
        T::AZALEA | T::FLOWERING_AZALEA => {
            // A bushy green clump filling most of the tile, with pink flower dots when flowering.
            let dx = x as i32 - 8;
            let dy = y as i32 - 7;
            let bush = dx * dx + dy * dy * 2 <= 60 && (x ^ y) % 3 != 0;
            if bush {
                if tile == T::FLOWERING_AZALEA && hashf(x, y, 27) > 0.84 {
                    return [to_u8(0.92), to_u8(0.55), to_u8(0.72), 255]; // pink flower
                }
                let g = hashf(x, y, 17) * 0.16;
                return [to_u8(0.22), to_u8(0.44 + g), to_u8(0.18), 255];
            }
            [0, 0, 0, 0]
        }
        T::BIG_DRIPLEAF | T::SMALL_DRIPLEAF => {
            // A green leaf pad on a central stem. Big = larger pad higher up.
            let stem = (7..=8).contains(&x) && y >= 7;
            let (cy, rad): (i32, i32) = if tile == T::BIG_DRIPLEAF { (4, 6) } else { (5, 4) };
            let dx = x as i32 - 8;
            let dy = y as i32 - cy;
            let pad = dx * dx + dy * dy <= rad * rad && y as i32 <= cy + rad / 2;
            if pad {
                let g = hashf(x, y, 19) * 0.14;
                return [to_u8(0.26), to_u8(0.50 + g), to_u8(0.22), 255];
            }
            if stem {
                return [to_u8(0.34), to_u8(0.48), to_u8(0.24), 255];
            }
            [0, 0, 0, 0]
        }
        T::HANGING_ROOTS => {
            // Thin tan strands dangling from the top.
            let strand = matches!(x, 5 | 8 | 10) && y <= 11;
            if strand {
                let d = hashf(x, y, 9) * 0.12;
                return [to_u8(0.58 + d), to_u8(0.44 + d), to_u8(0.30), 255];
            }
            [0, 0, 0, 0]
        }
        T::SPORE_BLOSSOM => {
            // A pink rosette near the top + a couple of dangling spore strands.
            let dx = x as i32 - 8;
            let dy = y as i32 - 4;
            let rosette = dx * dx + dy * dy <= 16;
            if rosette {
                if dx * dx + dy * dy <= 3 {
                    return [to_u8(0.95), to_u8(0.78), to_u8(0.40), 255]; // pale center
                }
                let d = hashf(x, y, 22) * 0.10;
                return [to_u8(0.86 + d), to_u8(0.38), to_u8(0.56), 255]; // pink petals
            }
            // Dangling spore strands below the bloom.
            if matches!(x, 6 | 10) && (8..=14).contains(&y) {
                return [to_u8(0.80), to_u8(0.50), to_u8(0.62), 255];
            }
            [0, 0, 0, 0]
        }
        T::SCULK_VEIN => {
            // Dark tendrils with faint cyan, low coverage.
            if blob(x, y, 13) {
                return [to_u8(0.10), to_u8(0.45), to_u8(0.46), 255]; // faint cyan node
            }
            if hashf(x, y, 29) > 0.80 {
                return [to_u8(0.05), to_u8(0.08), to_u8(0.10), 255]; // dark tendril
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

/// M34-VM2 item sprites: alpha-cutout shapes (tools, weapons, ingots/gems, materials, food) shared by
/// the held-item view-model and the hotbar/inventory icons. Transparent outside the shape so they read
/// as objects, not tiles. Shapes are deliberately simple but distinct per item.
fn paint_item(tile: u32, x: u32, y: u32) -> [u8; 4] {
    const CLEAR: [u8; 4] = [0, 0, 0, 0];
    let (xi, yi) = (x as i32, y as i32);
    // Fill with a colour + a faint per-texel grain so the sprite isn't flat.
    let fill = |c: [f32; 3]| -> [u8; 4] {
        let n = hashf(x, y, 7);
        let s = shade(c, (n - 0.5) * 0.16);
        [to_u8(s[0]), to_u8(s[1]), to_u8(s[2]), 255]
    };
    let cx = xi - 8;
    let cy = yi - 8;
    let manh = cx.abs() + cy.abs();
    let r2 = cx * cx + cy * cy;

    // Tools: a wood handle + a tier-coloured head, laid out tier*5 + class (Pick/Axe/Shovel/Sword/Hoe).
    if (T::TOOL_BASE..T::TOOL_BASE + 25).contains(&tile) {
        let idx = tile - T::TOOL_BASE;
        let tier = (idx / 5) as usize;
        let class = idx % 5;
        const HEAD: [[f32; 3]; 5] = [
            [0.62, 0.47, 0.28], // wood
            [0.50, 0.50, 0.53], // stone
            [0.80, 0.80, 0.83], // iron
            [0.97, 0.80, 0.28], // gold
            [0.45, 0.86, 0.84], // diamond
        ];
        let head = HEAD[tier];
        let handle = [0.45, 0.32, 0.18];
        if class == 3 {
            // Sword: a vertical blade, a crossguard, and a grip.
            if (7..=8).contains(&xi) && (1..=9).contains(&yi) {
                return fill(head);
            }
            if xi == 7 && yi == 0 {
                return fill(head); // tip
            }
            if (9..=10).contains(&yi) && (5..=10).contains(&xi) {
                return fill([0.50, 0.42, 0.26]); // crossguard
            }
            if (7..=8).contains(&xi) && (10..=14).contains(&yi) {
                return fill(handle); // grip
            }
            return CLEAR;
        }
        // Handle: an anti-diagonal band (bottom-left → upper-mid) for the other four classes.
        if (xi + yi - 16).abs() <= 1 && (3..=12).contains(&xi) && (4..=13).contains(&yi) {
            return fill(handle);
        }
        let in_head = match class {
            0 => {
                // Pickaxe: an arc bar across the top with down-turned tips.
                ((2..=3).contains(&yi) && (3..=12).contains(&xi))
                    || (matches!(xi, 3 | 4 | 11 | 12) && (3..=5).contains(&yi))
            }
            1 => (8..=12).contains(&xi) && yi >= 2 && yi <= 10 - (xi - 8), // axe blade, tapering
            2 => (9..=12).contains(&xi) && (2..=5).contains(&yi),         // shovel blade
            4 => {
                // Hoe: an L (a top bar + a short downward stub at the left).
                ((2..=3).contains(&yi) && (7..=12).contains(&xi))
                    || ((7..=8).contains(&xi) && (2..=5).contains(&yi))
            }
            _ => false,
        };
        if in_head {
            return fill(head);
        }
        return CLEAR;
    }

    match tile {
        T::BOW => {
            // A left-bowed arc (centre off to the left) + a straight bowstring on the right.
            let ax = xi - 2;
            let rr = ax * ax + cy * cy;
            if (64..=100).contains(&rr) && xi >= 2 && xi <= 12 {
                return fill([0.45, 0.30, 0.15]);
            }
            if xi == 12 && (3..=13).contains(&yi) {
                return fill([0.86, 0.86, 0.80]); // string
            }
            CLEAR
        }
        T::SHIELD => {
            // A rounded heater shield: wide top, tapering to a point, with a lighter rim + boss.
            let top = (3..=10).contains(&yi) && (3..=12).contains(&xi);
            let taper = (11..=14).contains(&yi) && (xi - 8).abs() <= (14 - yi);
            if top || taper {
                let rim = xi == 3 || xi == 12 || yi == 3;
                if rim {
                    return fill([0.72, 0.58, 0.30]);
                }
                if r2 <= 4 {
                    return fill([0.85, 0.86, 0.88]); // metal boss
                }
                return fill([0.40, 0.26, 0.50]); // face
            }
            CLEAR
        }
        T::ARROW => {
            // A diagonal shaft, a grey head at the tip, white fletching at the tail.
            if (xi - yi).abs() <= 1 && (2..=13).contains(&xi) {
                return fill([0.50, 0.36, 0.20]); // shaft
            }
            if xi >= 12 && yi <= 4 && (xi - yi) >= 9 {
                return fill([0.78, 0.80, 0.83]); // arrowhead
            }
            if xi <= 4 && yi >= 11 && (yi - xi) >= 8 {
                return fill([0.92, 0.92, 0.95]); // fletching
            }
            CLEAR
        }
        T::STICK => {
            if (xi + yi - 15).abs() <= 1 && (3..=12).contains(&xi) {
                return fill([0.48, 0.34, 0.18]);
            }
            CLEAR
        }
        T::INGOT_IRON => ingot(xi, yi, [0.82, 0.82, 0.85], &fill),
        T::INGOT_GOLD => ingot(xi, yi, [0.97, 0.82, 0.30], &fill),
        T::INGOT_COPPER => ingot(xi, yi, [0.80, 0.46, 0.30], &fill),
        T::GEM_DIAMOND => gem(manh, cx, cy, [0.46, 0.90, 0.88], &fill),
        T::GEM_EMERALD => gem(manh, cx, cy, [0.20, 0.80, 0.42], &fill),
        T::GEM_LAPIS => gem(manh, cx, cy, [0.20, 0.34, 0.74], &fill),
        T::GEM_AMETHYST => gem(manh, cx, cy, [0.64, 0.42, 0.86], &fill),
        T::DUST_REDSTONE => {
            if r2 <= 26 && hashf(x, y, 13) > 0.45 {
                return fill([0.85, 0.12, 0.10]);
            }
            CLEAR
        }
        T::ITEM_COAL => lump(x, y, r2, [0.15, 0.15, 0.17], &fill),
        T::ITEM_CHARCOAL => lump(x, y, r2, [0.26, 0.21, 0.18], &fill),
        T::RAW_IRON_ITEM => lump(x, y, r2, [0.74, 0.60, 0.48], &fill),
        T::RAW_GOLD_ITEM => lump(x, y, r2, [0.88, 0.72, 0.34], &fill),
        T::RAW_COPPER_ITEM => lump(x, y, r2, [0.78, 0.48, 0.32], &fill),
        T::ITEM_FLINT => {
            // An angular dark shard (a clipped triangle).
            if yi >= 4 && yi <= 13 && (xi - 4) >= 0 && (xi - 4) <= (yi - 4) && xi <= 12 {
                return fill([0.26, 0.26, 0.30]);
            }
            CLEAR
        }
        T::ITEM_BONE => {
            let shaft = (7..=8).contains(&xi) && (4..=11).contains(&yi);
            let knobs = (matches!(xi, 5 | 6 | 9 | 10)) && (matches!(yi, 3 | 4 | 11 | 12));
            if shaft || knobs {
                return fill([0.92, 0.91, 0.85]);
            }
            CLEAR
        }
        T::ITEM_FEATHER => {
            if (xi + yi - 15).abs() <= 4 && (xi - yi).abs() <= 6 && (3..=13).contains(&yi) {
                let quill = (xi + yi - 15).abs() <= 1;
                return fill(if quill { [0.80, 0.80, 0.82] } else { [0.93, 0.94, 0.97] });
            }
            CLEAR
        }
        T::ITEM_LEATHER => {
            if (3..=12).contains(&xi) && (4..=12).contains(&yi) && manh <= 9 {
                return fill([0.55, 0.36, 0.20]);
            }
            CLEAR
        }
        T::ITEM_STRING => {
            if (yi - 8 + ((xi as f32 * 0.9).sin() * 2.5) as i32).abs() <= 1 {
                return fill([0.86, 0.86, 0.84]);
            }
            CLEAR
        }
        T::FOOD_BREAD => {
            if cx * cx + cy * cy * 2 <= 34 {
                let crust = cx * cx + cy * cy * 2 >= 22;
                return fill(if crust { [0.55, 0.34, 0.15] } else { [0.80, 0.58, 0.30] });
            }
            CLEAR
        }
        T::FOOD_APPLE => {
            if r2 <= 20 {
                return fill([0.84, 0.16, 0.16]);
            }
            if xi == 8 && (2..=4).contains(&yi) {
                return fill([0.40, 0.26, 0.12]); // stem
            }
            if (9..=10).contains(&xi) && (3..=4).contains(&yi) {
                return fill([0.30, 0.62, 0.24]); // leaf
            }
            CLEAR
        }
        T::FOOD_CARROT => {
            if (4..=13).contains(&yi) && (xi - 8).abs() <= (13 - yi) / 2 {
                return fill([0.92, 0.52, 0.16]);
            }
            if (2..=4).contains(&yi) && (6..=10).contains(&xi) && hashf(x, y, 5) > 0.4 {
                return fill([0.28, 0.60, 0.24]); // greens
            }
            CLEAR
        }
        T::FOOD_WHEAT => {
            if matches!(xi % 4, 1 | 2) && yi >= 3 {
                let head = yi <= 7 && hashf(x, y, 9) > 0.4;
                return fill(if head { [0.92, 0.80, 0.34] } else { [0.74, 0.62, 0.26] });
            }
            CLEAR
        }
        T::FOOD_SEEDS => {
            if r2 <= 30 && hashf(x, y, 3) > 0.78 {
                return fill([0.62, 0.66, 0.30]);
            }
            CLEAR
        }
        T::FOOD_BERRIES => {
            // A small cluster of bright berries.
            for i in 0..3u32 {
                let bxr = (hashf(i, 0, 21) * 8.0) as i32 + 4;
                let byr = (hashf(i, 1, 25) * 8.0) as i32 + 4;
                if (xi - bxr).pow(2) + (yi - byr).pow(2) <= 5 {
                    return fill([0.98, 0.70, 0.18]);
                }
            }
            CLEAR
        }
        T::FOOD_COOKED => {
            // A cooked steak: a rounded brown blob with a darker seared edge.
            if cx * cx * 2 + cy * cy <= 36 {
                let edge = cx * cx * 2 + cy * cy >= 24;
                return fill(if edge { [0.40, 0.22, 0.12] } else { [0.66, 0.40, 0.24] });
            }
            CLEAR
        }
        T::FOOD_POTATO => {
            // A lumpy tan oval with a few eyes.
            if cx * cx + cy * cy * 2 <= 32 {
                if hashf(x, y, 17) > 0.9 {
                    return fill([0.55, 0.42, 0.22]); // eye
                }
                return fill([0.85, 0.70, 0.40]);
            }
            CLEAR
        }
        T::FOOD_BAKED_POTATO => {
            // The same oval, roasted: darker skin with a split showing the pale inside.
            if cx * cx + cy * cy * 2 <= 32 {
                if yi == 8 && (5..=10).contains(&xi) {
                    return fill([0.92, 0.85, 0.62]); // the split
                }
                return fill([0.62, 0.44, 0.24]);
            }
            CLEAR
        }
        _ => CLEAR,
    }
}

/// A rounded metal ingot bar (shared by the iron/gold/copper sprites).
fn ingot(xi: i32, yi: i32, c: [f32; 3], fill: &dyn Fn([f32; 3]) -> [u8; 4]) -> [u8; 4] {
    if (4..=11).contains(&xi) && (6..=10).contains(&yi)
        && !((xi == 4 || xi == 11) && (yi == 6 || yi == 10))
    {
        // A brighter top edge gives the bar a little relief.
        if yi == 6 {
            return fill(shade(c, 0.14));
        }
        return fill(c);
    }
    [0, 0, 0, 0]
}

/// A faceted gem rhombus (shared by diamond/emerald/lapis/amethyst).
fn gem(manh: i32, cx: i32, cy: i32, c: [f32; 3], fill: &dyn Fn([f32; 3]) -> [u8; 4]) -> [u8; 4] {
    if manh <= 5 {
        // The upper-left half catches the light.
        if cx + cy <= 0 {
            return fill(shade(c, 0.18));
        }
        return fill(c);
    }
    [0, 0, 0, 0]
}

/// A lumpy nugget (coal / raw ore), a rough disc with a frayed edge + a couple of dark pits.
fn lump(x: u32, y: u32, r2: i32, c: [f32; 3], fill: &dyn Fn([f32; 3]) -> [u8; 4]) -> [u8; 4] {
    if r2 <= 18 + (hashf(x, y, 31) * 8.0) as i32 {
        if hashf(x, y, 43) > 0.86 {
            return fill(shade(c, -0.28)); // pit
        }
        return fill(c);
    }
    [0, 0, 0, 0]
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
            // U4 cave-biome cross-billboard cutouts.
            | T::AMETHYST_CLUSTER
            | T::SMALL_AMETHYST_BUD
            | T::MEDIUM_AMETHYST_BUD
            | T::LARGE_AMETHYST_BUD
            | T::POINTED_DRIPSTONE
            | T::GLOW_LICHEN
            | T::CAVE_VINE
            | T::CAVE_VINE_BERRIES
            | T::AZALEA
            | T::FLOWERING_AZALEA
            | T::BIG_DRIPLEAF
            | T::SMALL_DRIPLEAF
            | T::HANGING_ROOTS
            | T::SPORE_BLOSSOM
            | T::SCULK_VEIN
            // S6 crop billboards.
            | T::CROP_WHEAT_YOUNG
            | T::CROP_WHEAT_MATURE
            | T::CROP_SPROUT
            | T::CROP_CARROT_MATURE
            | T::CROP_POTATO_MATURE
    ) {
        return paint_plant(tile, x, y);
    }
    if tile == T::GLASS {
        return paint_glass(x, y);
    }
    if (T::TOOL_BASE..=T::ITEM_SPRITE_END).contains(&tile) {
        return paint_item(tile, x, y);
    }
    let base = base_color(tile);
    let n = hashf(x, y, tile.wrapping_mul(131) + 7); // 0..1 fine grain
    let m = hashf(x / 2, y / 2, tile.wrapping_mul(977) + 3); // 0..1 coarse grain

    let mut c = base;
    match tile {
        T::FARMLAND_TOP => {
            // Tilled furrow rows: alternating dark troughs and lighter ridges.
            let trough = y % 4 < 2;
            c = shade(base, if trough { -0.22 } else { 0.10 } + (n - 0.5) * 0.10);
        }
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
        // U4 cave-biome opaque cubes.
        T::AMETHYST_BLOCK => {
            // Purple crystal with a few brighter facets.
            c = shade(base, (n - 0.5) * 0.16);
            if hashf(x, y, 71) > 0.82 {
                c = shade(base, 0.30); // bright facet
            }
        }
        T::BUDDING_AMETHYST => {
            c = shade(base, (n - 0.5) * 0.16);
            if hashf(x, y, 73) > 0.82 {
                c = shade(base, 0.30); // bright facet
            }
            if blob(x, y, 9) {
                c = shade([0.72, 0.55, 0.92], 0.10); // small budding nubs
            }
        }
        T::MOSS_BLOCK => {
            // Green mottle (clones the GRASS_TOP motif on a darker moss base).
            c = shade(base, (n - 0.5) * 0.26);
            if hashf(x, y, 67) > 0.80 {
                c = shade(base, 0.18); // brighter tufts
            }
        }
        T::ROOTED_DIRT => {
            // Dirt body with pale root flecks threading through.
            c = shade(base, (n - 0.5) * 0.22);
            if hashf(x, y, 55) > 0.86 {
                c = shade([0.78, 0.70, 0.52], 0.0); // pale root
            }
        }
        T::AZALEA_LEAVES => {
            // Leafy green (clone of LEAVES on the azalea base).
            c = shade(base, (n - 0.5) * 0.34);
            if hashf(x, y, 85) > 0.86 {
                c = shade(base, -0.30); // gaps/shadow
            }
        }
        T::SCULK => {
            // Near-black base with a scatter of cyan emissive-looking flecks.
            c = shade(base, (n - 0.5) * 0.20);
            if blob(x, y, 6) {
                c = [0.12, 0.55, 0.55]; // cyan node
            } else if hashf(x, y, 39) > 0.90 {
                c = [0.08, 0.30, 0.32]; // dim cyan speck
            }
        }
        T::SCULK_SENSOR => {
            c = shade(base, (n - 0.5) * 0.16);
            // A pair of cyan tendril hints reaching up the middle.
            if (7..=8).contains(&x) || blob(x, y, 11) {
                c = [0.20, 0.70, 0.72];
            }
        }
        T::SCULK_SHRIEKER => {
            c = shade(base, (n - 0.5) * 0.14);
            // A cyan ring around the center mouth.
            let dx = x as i32 - 8;
            let dy = y as i32 - 8;
            let r2 = dx * dx + dy * dy;
            if (16..=30).contains(&r2) {
                c = [0.22, 0.78, 0.78]; // glowing ring
            } else if r2 < 16 {
                c = [0.05, 0.08, 0.09]; // dark maw
            }
        }
        T::SCULK_CATALYST => {
            // Darker base + bone-white bits + more cyan than plain sculk.
            c = shade(base, (n - 0.5) * 0.16);
            if blob(x, y, 6) {
                c = [0.18, 0.65, 0.65]; // cyan bloom
            } else if blob(x, y, 14) {
                c = [0.88, 0.86, 0.80]; // bone-white bits
            } else if hashf(x, y, 47) > 0.88 {
                c = [0.10, 0.40, 0.42];
            }
        }
        T::REINFORCED_DEEPSLATE => {
            // Deepslate base with a subtle metallic frame border.
            c = shade(base, (n - 0.5) * 0.16);
            if x == 0 || y == 0 || x == 15 || y == 15 {
                c = [0.40, 0.41, 0.45]; // metal frame
            } else if x == 1 || y == 1 || x == 14 || y == 14 {
                c = shade(base, 0.18); // inner bevel
            }
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
