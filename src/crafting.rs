//! Crafting recipes (M20): a small data-driven registry matched against a 3x3 grid (the 2x2
//! inventory grid maps into the top-left). Shaped recipes match a trimmed bounding box (so they
//! work anywhere in the grid); shapeless recipes match the ingredient multiset.

use crate::block;
use crate::item::{self, ItemId};

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

#[rustfmt::skip]
static RECIPES: &[Recipe] = &[
    // Wood -> 4 planks (shapeless).
    Recipe { shaped: false, cells: [W,0,0, 0,0,0, 0,0,0], output: P, count: 4 },
    // 2 planks stacked -> 4 sticks.
    Recipe { shaped: true,  cells: [P,0,0, P,0,0, 0,0,0], output: S, count: 4 },
    // 2x2 planks -> crafting table.
    Recipe { shaped: true,  cells: [P,P,0, P,P,0, 0,0,0], output: block::CRAFTING_TABLE, count: 1 },
    // 8 cobblestone ring -> furnace.
    Recipe { shaped: true,  cells: [C,C,C, C,0,C, C,C,C], output: block::FURNACE, count: 1 },
    // 8 planks ring -> chest.
    Recipe { shaped: true,  cells: [P,P,P, P,0,P, P,P,P], output: block::CHEST, count: 1 },
    // Tools: top row material, sticks down the middle (pickaxe), L (axe), column (shovel/sword).
    Recipe { shaped: true,  cells: [P,P,P, 0,S,0, 0,S,0], output: item::WOOD_PICKAXE,  count: 1 },
    Recipe { shaped: true,  cells: [C,C,C, 0,S,0, 0,S,0], output: item::STONE_PICKAXE, count: 1 },
    Recipe { shaped: true,  cells: [P,P,0, P,S,0, 0,S,0], output: item::WOOD_AXE,      count: 1 },
    Recipe { shaped: true,  cells: [C,C,0, C,S,0, 0,S,0], output: item::STONE_AXE,     count: 1 },
    Recipe { shaped: true,  cells: [P,0,0, S,0,0, S,0,0], output: item::WOOD_SHOVEL,   count: 1 },
    Recipe { shaped: true,  cells: [C,0,0, S,0,0, S,0,0], output: item::STONE_SHOVEL,  count: 1 },
    Recipe { shaped: true,  cells: [P,0,0, P,0,0, S,0,0], output: item::WOOD_SWORD,    count: 1 },
    Recipe { shaped: true,  cells: [C,0,0, C,0,0, S,0,0], output: item::STONE_SWORD,   count: 1 },
];

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
    for r in RECIPES {
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
        // only 2 wide -> not a pickaxe; should be None
        assert_eq!(match_grid(&g), None);
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
}
