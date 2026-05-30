//! Items + inventory (M15). Every item is currently a placeable block, so `ItemId == BlockId`; the
//! abstraction leaves room for tools/food (M18+) to take ids outside the block range. The player
//! `Inventory` is 9 hotbar + 27 main + 4 armor slots plus a held (cursor) stack.

use crate::block::{self, BlockId};

pub type ItemId = u16;

/// The item that a block stacks as (identity for now).
#[inline]
pub fn item_of_block(b: BlockId) -> ItemId {
    b
}

/// The block an item places, if it is a block-item (all items, for now).
#[inline]
pub fn block_of_item(i: ItemId) -> Option<BlockId> {
    if i == block::AIR {
        None
    } else {
        Some(i)
    }
}

/// Max stack size for an item (64 for blocks; tools/food override later).
#[inline]
pub fn max_stack(_item: ItemId) -> u8 {
    64
}

/// Display name for the tooltip/hotbar label.
#[inline]
pub fn item_name(item: ItemId) -> &'static str {
    block::display_name(item)
}

#[derive(Clone, Copy)]
pub struct ItemStack {
    pub item: ItemId,
    pub count: u8,
}

impl ItemStack {
    pub fn new(item: ItemId, count: u8) -> Self {
        Self { item, count }
    }
    /// The block this stack places (None for non-block items).
    pub fn block(&self) -> Option<BlockId> {
        block_of_item(self.item)
    }
}

pub const HOTBAR: usize = 9;
pub const MAIN: usize = 27;
pub const ARMOR: usize = 4;
pub const SLOTS: usize = HOTBAR + MAIN + ARMOR; // 40: [0,9) hotbar, [9,36) main, [36,40) armor

/// Creative building palette pre-loaded into the hotbar.
const PALETTE: [BlockId; 9] = [
    block::GRASS,
    block::DIRT,
    block::STONE,
    block::SAND,
    block::WOOD,
    block::LEAVES,
    block::GLOWSTONE,
    block::TORCH,
    block::WATER,
];

pub struct Inventory {
    pub slots: [Option<ItemStack>; SLOTS],
    /// Cursor stack while an inventory screen is open (M15b).
    #[allow(dead_code)]
    pub held: Option<ItemStack>,
    pub selected: usize,
    pub creative: bool,
}

impl Inventory {
    pub fn new(creative: bool) -> Self {
        let mut slots = [None; SLOTS];
        if creative {
            for (i, &b) in PALETTE.iter().enumerate() {
                slots[i] = Some(ItemStack::new(item_of_block(b), 1));
            }
        }
        Self {
            slots,
            held: None,
            selected: 0,
            creative,
        }
    }

    /// The block the selected hotbar slot would place (AIR if empty).
    pub fn selected_block(&self) -> BlockId {
        self.slots[self.selected]
            .and_then(|s| s.block())
            .unwrap_or(block::AIR)
    }

    pub fn select(&mut self, index: usize) {
        if index < HOTBAR {
            self.selected = index;
        }
    }

    pub fn scroll(&mut self, delta: i32) {
        let n = HOTBAR as i32;
        self.selected = (((self.selected as i32 + delta) % n + n) % n) as usize;
    }

    /// Consume one of the selected stack after a successful place (no-op in creative). Returns false
    /// if there was nothing to place.
    pub fn consume_selected(&mut self) -> bool {
        if self.creative {
            return self.slots[self.selected].is_some();
        }
        if let Some(stack) = &mut self.slots[self.selected] {
            stack.count -= 1;
            if stack.count == 0 {
                self.slots[self.selected] = None;
            }
            true
        } else {
            false
        }
    }

    /// Try to add a whole stack, merging into existing stacks then empty slots (hotbar before main).
    /// Returns the leftover that did not fit (count 0 means fully absorbed).
    pub fn insert(&mut self, mut stack: ItemStack) -> Option<ItemStack> {
        let cap = max_stack(stack.item);
        // Merge into existing matching stacks.
        for slot in self.slots[0..HOTBAR + MAIN].iter_mut() {
            if let Some(s) = slot {
                if s.item == stack.item && s.count < cap {
                    let room = cap - s.count;
                    let moved = room.min(stack.count);
                    s.count += moved;
                    stack.count -= moved;
                    if stack.count == 0 {
                        return None;
                    }
                }
            }
        }
        // Fill empty slots.
        for slot in self.slots[0..HOTBAR + MAIN].iter_mut() {
            if slot.is_none() {
                let moved = cap.min(stack.count);
                *slot = Some(ItemStack::new(stack.item, moved));
                stack.count -= moved;
                if stack.count == 0 {
                    return None;
                }
            }
        }
        Some(stack)
    }

    /// Add one item of `block` (a pickup). Returns false if the inventory is full.
    pub fn add_block(&mut self, b: BlockId) -> bool {
        self.insert(ItemStack::new(item_of_block(b), 1)).is_none()
    }

    /// Standard inventory-screen click: left = pick up / drop whole stack / merge / swap; right =
    /// pick up half / drop one. Mutates `held` and the clicked `slot`.
    pub fn click_slot(&mut self, slot: usize, right: bool) {
        if slot >= SLOTS {
            return;
        }
        match (self.held, self.slots[slot]) {
            (None, Some(s)) => {
                if right {
                    let half = s.count.div_ceil(2);
                    let rem = s.count - half;
                    self.held = Some(ItemStack::new(s.item, half));
                    self.slots[slot] = (rem > 0).then(|| ItemStack::new(s.item, rem));
                } else {
                    self.held = self.slots[slot].take();
                }
            }
            (Some(h), None) => {
                if right {
                    self.slots[slot] = Some(ItemStack::new(h.item, 1));
                    let rem = h.count - 1;
                    self.held = (rem > 0).then(|| ItemStack::new(h.item, rem));
                } else {
                    self.slots[slot] = self.held.take();
                }
            }
            (Some(h), Some(s)) => {
                if h.item == s.item {
                    let cap = max_stack(s.item);
                    let room = cap.saturating_sub(s.count);
                    let moved = if right { 1.min(h.count) } else { h.count }.min(room);
                    self.slots[slot] = Some(ItemStack::new(s.item, s.count + moved));
                    let rem = h.count - moved;
                    self.held = (rem > 0).then(|| ItemStack::new(h.item, rem));
                } else if !right {
                    self.slots[slot] = Some(h);
                    self.held = Some(s);
                }
            }
            (None, None) => {}
        }
    }

    /// Drop the held stack back into the inventory (called when a screen closes).
    pub fn return_held(&mut self) {
        if let Some(h) = self.held.take() {
            self.held = self.insert(h);
        }
    }
}
