//! Crafting recipes (M20): a small data-driven registry matched against a 3x3 grid (the 2x2
//! inventory grid maps into the top-left). Shaped recipes match a trimmed bounding box (so they
//! work anywhere in the grid); shapeless recipes match the ingredient multiset.

use std::sync::OnceLock;

use crate::block;
use crate::item::{self, ItemId, Tier};

const EMPTY: ItemId = 0; // block::AIR

struct Recipe {
    shaped: bool,
    /// Shaped: 3x3 pattern (row-major), recipe placed top-left, 0 = empty.
    /// Shapeless: ingredient ids packed at the front, rest 0.
    cells: [ItemId; 9],
    output: ItemId,
    count: u8,
}

// Shorthand block/material ids used in patterns.
const W: ItemId = block::WOOD;
const P: ItemId = block::PLANKS;
const C: ItemId = block::COBBLESTONE;
const S: ItemId = item::STICK;

/// Build a shaped recipe from up to three ASCII rows + a `(char -> item)` legend; ' ' / '.' = empty.
/// The pattern is placed top-left; `match_grid` trims the bounding box so it's position-invariant.
fn shaped(rows: &[&str], legend: &[(char, ItemId)], output: ItemId, count: u8) -> Recipe {
    let mut cells = [EMPTY; 9];
    for (r, row) in rows.iter().enumerate() {
        for (c, ch) in row.bytes().enumerate() {
            if ch != b' ' && ch != b'.' {
                let id = legend.iter().find(|&&(k, _)| k as u8 == ch).map_or(EMPTY, |&(_, v)| v);
                cells[r * 3 + c] = id;
            }
        }
    }
    Recipe { shaped: true, cells, output, count }
}

/// A shapeless recipe (ingredient multiset; positions don't matter).
fn shapeless(items: &[ItemId], output: ItemId, count: u8) -> Recipe {
    let mut cells = [EMPTY; 9];
    for (i, &it) in items.iter().enumerate().take(9) {
        cells[i] = it;
    }
    Recipe { shaped: false, cells, output, count }
}

/// The 5 tools of a tier from a material `m` + sticks (pickaxe/axe/shovel/sword/hoe).
fn tool_set(r: &mut Vec<Recipe>, m: ItemId, tier: Tier) {
    let leg = &[('M', m), ('S', S)];
    r.push(shaped(&["MMM", " S ", " S "], leg, item::tool_id(tier, 0), 1)); // pickaxe
    r.push(shaped(&["MM ", "MS ", " S "], leg, item::tool_id(tier, 1), 1)); // axe
    r.push(shaped(&["M", "S", "S"], leg, item::tool_id(tier, 2), 1)); // shovel
    r.push(shaped(&["M", "M", "S"], leg, item::tool_id(tier, 3), 1)); // sword
    r.push(shaped(&["MM ", " S ", " S "], leg, item::tool_id(tier, 4), 1)); // hoe
}

/// U3 compaction pair: 9 of `material` -> 1 `block`, and `block` -> 9 `material` (storage blocks).
fn storage(r: &mut Vec<Recipe>, material: ItemId, block: ItemId) {
    r.push(shaped(&["MMM", "MMM", "MMM"], &[('M', material)], block, 1));
    r.push(shapeless(&[block], material, 9));
}

/// U3 2x2 -> 4 conversion (granite->polished, cobbled deepslate->polished->bricks->tiles, copper->cut).
fn square4(r: &mut Vec<Recipe>, input: ItemId, output: ItemId) {
    r.push(shaped(&["II", "II"], &[('I', input)], output, 4));
}

/// The 4 armor pieces of a tier from a material `m` (helmet/chestplate/leggings/boots).
fn armor_set(r: &mut Vec<Recipe>, m: ItemId, tier: u16) {
    let leg = &[('M', m)];
    r.push(shaped(&["MMM", "M M"], leg, item::armor_id(tier, 0), 1)); // helmet
    r.push(shaped(&["M M", "MMM", "MMM"], leg, item::armor_id(tier, 1), 1)); // chestplate
    r.push(shaped(&["MMM", "M M", "M M"], leg, item::armor_id(tier, 2), 1)); // leggings
    r.push(shaped(&["M M", "M M"], leg, item::armor_id(tier, 3), 1)); // boots
}

/// The full crafting registry, built once. Families (tools/armor) are generated; the rest are
/// declared with the `shaped`/`shapeless` helpers. Wood-variant / door / fence / dye families arrive
/// once their block ids land (the recipe builder scales to them then).
fn build_recipes() -> Vec<Recipe> {
    let mut r = Vec::new();
    // Basics.
    r.push(shapeless(&[W], P, 4)); // a log -> 4 planks
    r.push(shaped(&["P", "P"], &[('P', P)], S, 4)); // 2 planks -> 4 sticks
    r.push(shaped(&["PP", "PP"], &[('P', P)], block::CRAFTING_TABLE, 1));
    r.push(shaped(&["CCC", "C C", "CCC"], &[('C', C)], block::FURNACE, 1));
    r.push(shaped(&["PPP", "P P", "PPP"], &[('P', P)], block::CHEST, 1));
    // Torches: coal or charcoal over a stick -> 4 torches.
    r.push(shaped(&["O", "S"], &[('O', item::COAL), ('S', S)], block::TORCH, 4));
    r.push(shaped(&["O", "S"], &[('O', item::CHARCOAL), ('S', S)], block::TORCH, 4));
    // S12: a bucket — three iron ingots in a V.
    r.push(shaped(&["I I", " I "], &[('I', item::IRON_INGOT)], item::BUCKET, 1));
    // S11: a bed — wool over planks.
    r.push(shaped(&["WWW", "PPP"], &[('W', block::WOOL), ('P', P)], block::BED, 1));
    // S6: bread — the farm-to-table staple.
    r.push(shaped(&["WWW"], &[('W', item::WHEAT)], item::BREAD, 1));
    // Slabs + stairs (cobblestone + planks; per-material variants come with wood types).
    r.push(shaped(&["CCC"], &[('C', C)], block::STONE_SLAB, 6));
    r.push(shaped(&["PPP"], &[('P', P)], block::WOOD_SLAB, 6));
    r.push(shaped(&["C  ", "CC ", "CCC"], &[('C', C)], block::STONE_STAIRS, 4));
    // Full tool set.
    tool_set(&mut r, P, Tier::Wood);
    tool_set(&mut r, C, Tier::Stone);
    tool_set(&mut r, item::IRON_INGOT, Tier::Iron);
    tool_set(&mut r, item::GOLD_INGOT, Tier::Gold);
    tool_set(&mut r, item::DIAMOND, Tier::Diamond);
    // Full armor set (tiers: 0 leather, 1 iron, 2 gold, 3 diamond).
    armor_set(&mut r, item::LEATHER, 0);
    armor_set(&mut r, item::IRON_INGOT, 1);
    armor_set(&mut r, item::GOLD_INGOT, 2);
    armor_set(&mut r, item::DIAMOND, 3);
    // P13: bow (3 string + 3 sticks, MC shape) and arrows (flint + stick + feather -> 4).
    r.push(shaped(
        &[" ST", "S T", " ST"],
        &[('S', item::STRING), ('T', S)],
        item::BOW,
        1,
    ));
    r.push(shaped(
        &["F", "S", "E"],
        &[('F', item::FLINT), ('S', S), ('E', item::FEATHER)],
        item::ARROW,
        4,
    ));
    // P14: shield (6 planks + 1 iron ingot, MC shape).
    r.push(shaped(
        &["PIP", "PPP", " P "],
        &[('P', P), ('I', item::IRON_INGOT)],
        item::SHIELD,
        1,
    ));
    // U3: storage/compaction blocks (9 ingots/gems <-> 1 block; raw-material blocks too).
    storage(&mut r, item::IRON_INGOT, block::IRON_BLOCK);
    storage(&mut r, item::GOLD_INGOT, block::GOLD_BLOCK);
    storage(&mut r, item::DIAMOND, block::DIAMOND_BLOCK);
    storage(&mut r, item::EMERALD, block::EMERALD_BLOCK);
    storage(&mut r, item::LAPIS, block::LAPIS_BLOCK);
    storage(&mut r, item::REDSTONE_DUST, block::REDSTONE_BLOCK);
    storage(&mut r, item::COPPER_INGOT, block::COPPER_BLOCK);
    storage(&mut r, item::COAL, block::COAL_BLOCK);
    storage(&mut r, item::RAW_IRON, block::RAW_IRON_BLOCK);
    storage(&mut r, item::RAW_COPPER, block::RAW_COPPER_BLOCK);
    storage(&mut r, item::RAW_GOLD, block::RAW_GOLD_BLOCK);
    // U3: clay block <-> 4 clay balls.
    r.push(shaped(&["CC", "CC"], &[('C', item::CLAY_BALL)], block::CLAY, 1));
    // U3: cut copper + the polished/brick stone chains (2x2 -> 4).
    square4(&mut r, block::COPPER_BLOCK, block::CUT_COPPER);
    square4(&mut r, block::GRANITE, block::POLISHED_GRANITE);
    square4(&mut r, block::DIORITE, block::POLISHED_DIORITE);
    square4(&mut r, block::ANDESITE, block::POLISHED_ANDESITE);
    square4(&mut r, block::COBBLED_DEEPSLATE, block::POLISHED_DEEPSLATE);
    square4(&mut r, block::POLISHED_DEEPSLATE, block::DEEPSLATE_BRICKS);
    square4(&mut r, block::DEEPSLATE_BRICKS, block::DEEPSLATE_TILES);
    // U4: 4 amethyst shards (2x2) -> 1 amethyst block (the only U4 recipe; the rest of the cave-biome
    // set is placement-only decoration with no crafting).
    r.push(shaped(&["AA", "AA"], &[('A', item::AMETHYST_SHARD)], block::AMETHYST_BLOCK, 1));
    r
}

static RECIPES: OnceLock<Vec<Recipe>> = OnceLock::new();

/// Bounding box (min_r, min_c, max_r, max_c) of the non-empty cells, or None if all empty.
fn trim(g: &[ItemId; 9]) -> Option<(usize, usize, usize, usize)> {
    let (mut r0, mut c0, mut r1, mut c1, mut any) = (3, 3, 0, 0, false);
    for r in 0..3 {
        for c in 0..3 {
            if g[r * 3 + c] != EMPTY {
                any = true;
                r0 = r0.min(r);
                c0 = c0.min(c);
                r1 = r1.max(r);
                c1 = c1.max(c);
            }
        }
    }
    any.then_some((r0, c0, r1, c1))
}

fn shaped_match(input: &[ItemId; 9], pattern: &[ItemId; 9]) -> bool {
    let (ir0, ic0, ir1, ic1) = match trim(input) {
        Some(t) => t,
        None => return false,
    };
    let (pr0, pc0, pr1, pc1) = match trim(pattern) {
        Some(t) => t,
        None => return false,
    };
    if ir1 - ir0 != pr1 - pr0 || ic1 - ic0 != pc1 - pc0 {
        return false;
    }
    for dr in 0..=(ir1 - ir0) {
        for dc in 0..=(ic1 - ic0) {
            if input[(ir0 + dr) * 3 + (ic0 + dc)] != pattern[(pr0 + dr) * 3 + (pc0 + dc)] {
                return false;
            }
        }
    }
    true
}

fn shapeless_match(input: &[ItemId; 9], ingredients: &[ItemId; 9]) -> bool {
    let mut a: Vec<ItemId> = input.iter().copied().filter(|&x| x != EMPTY).collect();
    let mut b: Vec<ItemId> = ingredients.iter().copied().filter(|&x| x != EMPTY).collect();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

/// The output (item, count) for a filled 3x3 grid of item ids (0 = empty), if any recipe matches.
pub fn match_grid(grid: &[ItemId; 9]) -> Option<(ItemId, u8)> {
    for r in RECIPES.get_or_init(build_recipes) {
        let hit = if r.shaped {
            shaped_match(grid, &r.cells)
        } else {
            shapeless_match(grid, &r.cells)
        };
        if hit {
            return Some((r.output, r.count));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planks_and_sticks() {
        let mut g = [0; 9];
        g[4] = block::WOOD; // shapeless, position-invariant
        assert_eq!(match_grid(&g), Some((block::PLANKS, 4)));

        let mut g = [0; 9];
        g[1] = block::PLANKS;
        g[4] = block::PLANKS; // 2 planks stacked anywhere
        assert_eq!(match_grid(&g), Some((item::STICK, 4)));
    }

    #[test]
    fn pickaxe_offset_invariant() {
        // Pickaxe shifted to the right columns still matches (shaped bounding-box trim).
        let g = [
            0, block::PLANKS, block::PLANKS,
            0, 0, item::STICK,
            0, 0, item::STICK,
        ];
        // PP / .S / .S trims to a hoe shape (hoes are craftable now), not a pickaxe.
        assert_eq!(match_grid(&g), Some((item::tool_id(Tier::Wood, 4), 1)));
        let g = [
            block::COBBLESTONE, block::COBBLESTONE, block::COBBLESTONE,
            0, item::STICK, 0,
            0, item::STICK, 0,
        ];
        assert_eq!(match_grid(&g), Some((item::STONE_PICKAXE, 1)));
    }

    #[test]
    fn table_and_furnace() {
        let g = [block::PLANKS, block::PLANKS, 0, block::PLANKS, block::PLANKS, 0, 0, 0, 0];
        assert_eq!(match_grid(&g), Some((block::CRAFTING_TABLE, 1)));
        let g = [
            block::COBBLESTONE, block::COBBLESTONE, block::COBBLESTONE,
            block::COBBLESTONE, 0, block::COBBLESTONE,
            block::COBBLESTONE, block::COBBLESTONE, block::COBBLESTONE,
        ];
        assert_eq!(match_grid(&g), Some((block::FURNACE, 1)));
    }

    #[test]
    fn empty_grid_no_recipe() {
        assert_eq!(match_grid(&[0; 9]), None);
    }

    #[test]
    fn u3_storage_polish_and_clay() {
        // 9 iron ingots -> iron block, and a single block back to 9 ingots.
        let i = item::IRON_INGOT;
        assert_eq!(match_grid(&[i, i, i, i, i, i, i, i, i]), Some((block::IRON_BLOCK, 1)));
        let mut g = [0; 9];
        g[4] = block::IRON_BLOCK;
        assert_eq!(match_grid(&g), Some((i, 9)));
        // 4 copper blocks (2x2) -> 4 cut copper; a single copper block -> 9 ingots (distinct grids).
        let b = block::COPPER_BLOCK;
        assert_eq!(match_grid(&[b, b, 0, b, b, 0, 0, 0, 0]), Some((block::CUT_COPPER, 4)));
        let mut g = [0; 9];
        g[0] = b;
        assert_eq!(match_grid(&g), Some((item::COPPER_INGOT, 9)));
        // Deepslate polish chain: cobbled -> polished.
        let cd = block::COBBLED_DEEPSLATE;
        assert_eq!(match_grid(&[cd, cd, 0, cd, cd, 0, 0, 0, 0]), Some((block::POLISHED_DEEPSLATE, 4)));
        // 4 clay balls -> clay block.
        let cb = item::CLAY_BALL;
        assert_eq!(match_grid(&[cb, cb, 0, cb, cb, 0, 0, 0, 0]), Some((block::CLAY, 1)));
    }

    #[test]
    fn registry_is_substantial_and_known() {
        // The recipe set should cover the full tool/armor progression, and every ingredient/output
        // must be a known item (catches id drift as new materials/blocks land).
        let recipes = RECIPES.get_or_init(build_recipes);
        assert!(recipes.len() >= 40, "expected a full recipe set, got {}", recipes.len());
        for r in recipes {
            assert!(item::is_known(r.output), "unknown recipe output {}", r.output);
            for &c in &r.cells {
                assert!(c == EMPTY || item::is_known(c), "unknown ingredient {c}");
            }
        }
    }

    #[test]
    fn diamond_gear_craftable() {
        let (d, s) = (item::DIAMOND, item::STICK);
        // Diamond pickaxe: three diamonds on top, two sticks down the middle.
        let g = [d, d, d, 0, s, 0, 0, s, 0];
        assert_eq!(match_grid(&g), Some((item::DIAMOND_PICKAXE, 1)));
        // Diamond chestplate.
        let g = [d, 0, d, d, d, d, d, d, d];
        assert_eq!(match_grid(&g), Some((item::armor_id(3, 1), 1)));
        // Iron helmet.
        let i = item::IRON_INGOT;
        let g = [i, i, i, i, 0, i, 0, 0, 0];
        assert_eq!(match_grid(&g), Some((item::armor_id(1, 0), 1)));
    }
}
