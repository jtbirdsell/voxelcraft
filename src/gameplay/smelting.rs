//! Smelting (M21): the smelt-recipe + fuel tables that drive the furnace tick in `game.rs`.
//!
//! Smelting inputs are *item ids*; for ores `ItemId == BlockId`, so the input keys are the ore
//! block ids and the outputs are ingot materials (or, for cobblestone, the stone block-item).

use crate::block;
use crate::item::{self, ItemId};

/// Seconds of furnace burn it takes to smelt one item.
pub const SMELT_TIME: f32 = 6.0;

/// The item a furnace produces from one unit of `input`, or `None` if it can't be smelted.
pub fn smelt_output(input: ItemId) -> Option<ItemId> {
    Some(match input {
        block::IRON_ORE => item::IRON_INGOT,
        block::GOLD_ORE => item::GOLD_INGOT,
        block::COBBLESTONE => block::STONE, // cobble re-smelts to smooth stone
        // Cooking raw mob drops (P4).
        item::BEEF => item::COOKED_BEEF,
        item::PORK => item::COOKED_PORK,
        item::CHICKEN_MEAT => item::COOKED_CHICKEN,
        item::MUTTON => item::COOKED_MUTTON,
        _ => return None,
    })
}

/// Seconds of burn time one unit of `fuel` provides, or `None` if it isn't a fuel.
pub fn fuel_value(fuel: ItemId) -> Option<f32> {
    Some(match fuel {
        block::PLANKS => 9.0, // 1.5 items
        block::WOOD => 18.0,  // 3 items — a whole log
        item::STICK => 3.0,   // 0.5 items
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ores_smelt_to_ingots() {
        assert_eq!(smelt_output(block::IRON_ORE), Some(item::IRON_INGOT));
        assert_eq!(smelt_output(block::GOLD_ORE), Some(item::GOLD_INGOT));
        assert_eq!(smelt_output(block::COBBLESTONE), Some(block::STONE));
        assert_eq!(smelt_output(block::STONE), None); // already smelted
        assert_eq!(smelt_output(item::STICK), None); // sticks are fuel, not input
        assert_eq!(smelt_output(item::IRON_INGOT), None); // ingots don't re-smelt
    }

    #[test]
    fn raw_meat_cooks() {
        assert_eq!(smelt_output(block::IRON_ORE), Some(item::IRON_INGOT));
        assert_eq!(smelt_output(item::BEEF), Some(item::COOKED_BEEF));
        assert_eq!(smelt_output(item::CHICKEN_MEAT), Some(item::COOKED_CHICKEN));
        assert_eq!(smelt_output(item::COOKED_BEEF), None); // already cooked
    }

    #[test]
    fn fuels_have_positive_burn() {
        assert!(fuel_value(block::PLANKS).unwrap() > 0.0);
        assert!(fuel_value(block::WOOD).unwrap() > fuel_value(block::PLANKS).unwrap());
        assert!(fuel_value(item::STICK).is_some());
        assert_eq!(fuel_value(block::IRON_ORE), None); // ore is input, never fuel
    }

    #[test]
    fn a_log_smelts_at_least_one_item() {
        // A burning unit of fuel must cook a whole number of items cleanly enough to be useful.
        assert!(fuel_value(block::WOOD).unwrap() >= SMELT_TIME);
    }
}
