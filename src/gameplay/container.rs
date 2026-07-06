//! Generic block-entity container (P3): a fixed-size slot array with the canonical cursor-vs-slot
//! click behavior (`item::slot_click`). Chests use it (27 slots; a double chest is 54); future
//! storage blocks (barrel, dispenser, hopper) reuse the same type. Contents live in `Game`, keyed by
//! the block position, and are persisted alongside furnaces in `save_state.bin`.

use crate::item::{self, ItemStack};

/// Slots in a single chest.
pub const CHEST_SLOTS: usize = 27;

#[derive(Clone)]
pub struct Container {
    pub slots: Vec<Option<ItemStack>>,
}

impl Container {
    pub fn new(size: usize) -> Self {
        Self {
            slots: vec![None; size],
        }
    }

    /// Standard cursor-vs-slot interaction on slot `i` (left = pick/drop/merge/swap, right = half/one),
    /// delegating to the shared `item::slot_click` so behavior matches the inventory exactly.
    pub fn click(&mut self, i: usize, held: &mut Option<ItemStack>, right: bool) {
        if i >= self.slots.len() {
            return;
        }
        let mut s = self.slots[i];
        item::slot_click(held, &mut s, right);
        self.slots[i] = s;
    }

    /// S14: shift-click destination — merge into matching stacks, then fill empties (durability
    /// preserved). Returns the leftover that did not fit.
    pub fn insert(&mut self, mut stack: ItemStack) -> Option<ItemStack> {
        let cap = item::max_stack(stack.item);
        for s in self.slots.iter_mut().flatten() {
            if s.item == stack.item && s.count < cap {
                let moved = (cap - s.count).min(stack.count);
                s.count += moved;
                stack.count -= moved;
                if stack.count == 0 {
                    return None;
                }
            }
        }
        for slot in self.slots.iter_mut() {
            if slot.is_none() {
                let moved = cap.min(stack.count);
                *slot = Some(ItemStack {
                    item: stack.item,
                    count: moved,
                    durability: stack.durability,
                });
                stack.count -= moved;
                if stack.count == 0 {
                    return None;
                }
            }
        }
        Some(stack)
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// Everything the container holds, for spilling when the block is broken.
    pub fn contents(&self) -> impl Iterator<Item = ItemStack> + '_ {
        self.slots.iter().flatten().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block;

    #[test]
    fn click_picks_up_and_drops() {
        let mut c = Container::new(CHEST_SLOTS);
        c.slots[5] = Some(ItemStack::new(item::item_of_block(block::STONE), 10));
        let mut held = None;
        c.click(5, &mut held, false); // pick up whole stack
        assert_eq!(held.unwrap().count, 10);
        assert!(c.slots[5].is_none());
        c.click(0, &mut held, true); // drop one into empty slot
        assert_eq!(c.slots[0].unwrap().count, 1);
        assert_eq!(held.unwrap().count, 9);
    }

    #[test]
    fn s14_insert_merges_then_fills() {
        let mut c = Container::new(3);
        let stone = item::item_of_block(block::STONE);
        c.slots[1] = Some(ItemStack::new(stone, 60));
        assert!(c.insert(ItemStack::new(stone, 10)).is_none());
        assert_eq!(c.slots[1].unwrap().count, 64, "tops off the partial stack first");
        assert_eq!(c.slots[0].unwrap().count, 6, "remainder starts a new stack");
        // Fill it, then overflow comes back.
        c.slots[0] = Some(ItemStack::new(stone, 64));
        c.slots[2] = Some(ItemStack::new(stone, 64));
        let rest = c.insert(ItemStack::new(stone, 5));
        assert_eq!(rest.unwrap().count, 5, "full container returns the leftover");
    }

    #[test]
    fn contents_and_empty() {
        let mut c = Container::new(CHEST_SLOTS);
        assert!(c.is_empty());
        c.slots[0] = Some(ItemStack::new(item::item_of_block(block::DIRT), 3));
        c.slots[26] = Some(ItemStack::new(item::IRON_INGOT, 2));
        assert!(!c.is_empty());
        assert_eq!(c.contents().count(), 2);
    }
}
