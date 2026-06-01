//! Food values (P4): how much hunger + saturation each edible item restores. Right-click-hold to
//! eat. Values follow Minecraft (hunger in half-haunches restored; saturation is the hidden reserve).

use crate::item::{self, ItemId};

pub struct Food {
    /// Hunger points restored (1 = half a haunch).
    pub hunger: u32,
    /// Saturation restored (clamped by the player to the hunger level + the reserve cap).
    pub saturation: f32,
    /// Edible even at full hunger (golden apples / milk; eaten for the effect, not the food).
    pub always_edible: bool,
}

/// The food value of an item, or `None` if it isn't edible.
pub fn food(item: ItemId) -> Option<Food> {
    let f = |hunger, saturation| {
        Some(Food {
            hunger,
            saturation,
            always_edible: false,
        })
    };
    match item {
        // Raw mob drops — edible but weak (cooking is far better).
        item::BEEF | item::PORK => f(3, 1.8),
        item::CHICKEN_MEAT | item::MUTTON => f(2, 1.2),
        // Rotten flesh: edible in a pinch (the MC hunger-effect penalty lands with the effects system).
        item::ROTTEN_FLESH => f(4, 0.8),
        // Cooked meats (from smelting).
        item::COOKED_BEEF | item::COOKED_PORK => f(8, 12.8),
        item::COOKED_CHICKEN => f(6, 7.2),
        item::COOKED_MUTTON => f(6, 9.6),
        // Staples.
        item::BREAD => f(5, 6.0),
        item::APPLE => f(4, 2.4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooking_improves_food() {
        let raw = food(item::BEEF).unwrap();
        let cooked = food(item::COOKED_BEEF).unwrap();
        assert!(cooked.hunger > raw.hunger, "steak should restore more than raw beef");
        assert!(food(item::BREAD).is_some() && food(item::APPLE).is_some());
        assert!(food(item::STICK).is_none() && food(item::IRON_INGOT).is_none());
    }
}
