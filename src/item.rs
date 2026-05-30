//! Items + inventory (M15). Every item is currently a placeable block, so `ItemId == BlockId`; the
//! abstraction leaves room for tools/food (M18+) to take ids outside the block range. The player
//! `Inventory` is 9 hotbar + 27 main + 4 armor slots plus a held (cursor) stack.

use crate::block::{self, BlockId, ToolClass};

pub type ItemId = u16;

/// Item ids `< TOOL_BASE` are block-items (id == BlockId). Tools occupy `[TOOL_BASE, TOOL_BASE+25)`,
/// laid out as `TOOL_BASE + tier*5 + class` (Pickaxe0, Axe1, Shovel2, Sword3, Hoe4).
pub const TOOL_BASE: ItemId = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Wood,
    Stone,
    Iron,
    Gold,
    Diamond,
}

const TOOL_CLASSES: [ToolClass; 5] = [
    ToolClass::Pickaxe,
    ToolClass::Axe,
    ToolClass::Shovel,
    ToolClass::Sword,
    ToolClass::Hoe,
];

/// Tool id from a tier + class index (Pickaxe0/Axe1/Shovel2/Sword3/Hoe4).
#[inline]
pub const fn tool_id(tier: Tier, class_idx: u16) -> ItemId {
    TOOL_BASE + (tier as u16) * 5 + class_idx
}

// Named tool ids for the creative palette + recipes.
pub const WOOD_PICKAXE: ItemId = tool_id(Tier::Wood, 0);
pub const WOOD_AXE: ItemId = tool_id(Tier::Wood, 1);
pub const WOOD_SHOVEL: ItemId = tool_id(Tier::Wood, 2);
pub const WOOD_SWORD: ItemId = tool_id(Tier::Wood, 3);
pub const STONE_PICKAXE: ItemId = tool_id(Tier::Stone, 0);
pub const STONE_AXE: ItemId = tool_id(Tier::Stone, 1);
pub const STONE_SHOVEL: ItemId = tool_id(Tier::Stone, 2);
pub const STONE_SWORD: ItemId = tool_id(Tier::Stone, 3);
pub const DIAMOND_PICKAXE: ItemId = tool_id(Tier::Diamond, 0);
pub const DIAMOND_AXE: ItemId = tool_id(Tier::Diamond, 1);
pub const DIAMOND_SHOVEL: ItemId = tool_id(Tier::Diamond, 2);
pub const DIAMOND_SWORD: ItemId = tool_id(Tier::Diamond, 3);

/// Crafting materials (non-block, non-tool items) occupy `[MATERIAL_BASE, MATERIAL_BASE+16)`.
pub const MATERIAL_BASE: ItemId = 512;
pub const STICK: ItemId = MATERIAL_BASE;
pub const IRON_INGOT: ItemId = MATERIAL_BASE + 1;
pub const GOLD_INGOT: ItemId = MATERIAL_BASE + 2;

fn material_name(item: ItemId) -> &'static str {
    match item {
        STICK => "Stick",
        IRON_INGOT => "Iron Ingot",
        GOLD_INGOT => "Gold Ingot",
        _ => "Material",
    }
}

#[inline]
pub fn is_tool(item: ItemId) -> bool {
    (TOOL_BASE..TOOL_BASE + 25).contains(&item)
}

#[inline]
pub fn is_material(item: ItemId) -> bool {
    (MATERIAL_BASE..MATERIAL_BASE + 16).contains(&item)
}

#[inline]
pub fn tool_tier(item: ItemId) -> Tier {
    match (item - TOOL_BASE) / 5 {
        1 => Tier::Stone,
        2 => Tier::Iron,
        3 => Tier::Gold,
        4 => Tier::Diamond,
        _ => Tier::Wood,
    }
}

#[inline]
pub fn tool_class(item: ItemId) -> ToolClass {
    TOOL_CLASSES[((item - TOOL_BASE) % 5) as usize]
}

/// Mining-speed multiplier of a tier (applied when the tool matches the block's tool class).
pub fn tool_speed(tier: Tier) -> f32 {
    match tier {
        Tier::Wood => 2.0,
        Tier::Stone => 4.0,
        Tier::Iron => 6.0,
        Tier::Gold => 12.0,
        Tier::Diamond => 8.0,
    }
}

/// Harvest level (which tiers a block needs to drop): wood/gold 0, stone 1, iron 2, diamond 3.
pub fn harvest_level(tier: Tier) -> u8 {
    match tier {
        Tier::Wood | Tier::Gold => 0,
        Tier::Stone => 1,
        Tier::Iron => 2,
        Tier::Diamond => 3,
    }
}

pub fn tool_max_durability(item: ItemId) -> u16 {
    match tool_tier(item) {
        Tier::Wood => 59,
        Tier::Stone => 131,
        Tier::Iron => 250,
        Tier::Gold => 32,
        Tier::Diamond => 1561,
    }
}

/// Melee damage (used by combat in M21). Swords hit hardest; other tools ~2.
#[allow(dead_code)]
pub fn attack_damage(item: ItemId) -> f32 {
    if !is_tool(item) {
        return 1.0; // bare hand / thrown block
    }
    if tool_class(item) == ToolClass::Sword {
        match tool_tier(item) {
            Tier::Wood | Tier::Gold => 4.0,
            Tier::Stone => 5.0,
            Tier::Iron => 6.0,
            Tier::Diamond => 7.0,
        }
    } else {
        2.0
    }
}

/// The block an item places, if it is a block-item (tools place nothing).
#[inline]
pub fn block_of_item(i: ItemId) -> Option<BlockId> {
    if i == block::AIR || i >= TOOL_BASE {
        None
    } else {
        Some(i)
    }
}

/// The item that a block stacks as (identity).
#[inline]
pub fn item_of_block(b: BlockId) -> ItemId {
    b
}

/// Max stack size: tools don't stack.
#[inline]
pub fn max_stack(item: ItemId) -> u8 {
    if is_tool(item) {
        1
    } else {
        64
    }
}

/// Display name for the tooltip/hotbar label.
pub fn item_name(item: ItemId) -> &'static str {
    if let Some(b) = block_of_item(item) {
        return block::display_name(b);
    }
    if is_material(item) {
        return material_name(item);
    }
    if !is_tool(item) {
        return "Unknown";
    }
    let tier = match tool_tier(item) {
        Tier::Wood => "Wooden",
        Tier::Stone => "Stone",
        Tier::Iron => "Iron",
        Tier::Gold => "Golden",
        Tier::Diamond => "Diamond",
    };
    let class = match tool_class(item) {
        ToolClass::Pickaxe => "Pickaxe",
        ToolClass::Axe => "Axe",
        ToolClass::Shovel => "Shovel",
        ToolClass::Sword => "Sword",
        ToolClass::Hoe => "Hoe",
        ToolClass::None => "Tool",
    };
    // Small static table keyed by tool id (avoids per-call allocation).
    TOOL_NAMES[(item - TOOL_BASE) as usize].get_or_init(|| format!("{tier} {class}"))
}

/// Atlas tile for a dropped item entity: the block's tile, or a generic tool tile.
pub fn item_tile(item: ItemId) -> u32 {
    match block_of_item(item) {
        Some(b) => block::face_tile(b, [0, 1, 0]),
        None => block::tile::PLANKS, // generic handle-colored placeholder for tool drops
    }
}

/// Self-emission of a dropped item (lava block glows; tools don't).
pub fn item_emission(item: ItemId) -> f32 {
    block_of_item(item).map(block::emission).unwrap_or(0.0)
}

/// True if `item` is a registered block, tool, or material (used to reject corrupt save ids).
pub fn is_known(item: ItemId) -> bool {
    is_tool(item) || is_material(item) || (item != block::AIR && item <= block::CACTUS)
}

/// Representative UI color for an item: the block face color, or a tier color for tools.
pub fn item_color(item: ItemId) -> [f32; 3] {
    if let Some(b) = block_of_item(item) {
        return block::face_color(b, [0, 1, 0]);
    }
    if is_material(item) {
        return match item {
            IRON_INGOT => [0.80, 0.78, 0.74],
            GOLD_INGOT => [0.95, 0.80, 0.22],
            _ => [0.55, 0.40, 0.22], // stick / generic wooden
        };
    }
    if is_tool(item) {
        return match tool_tier(item) {
            Tier::Wood => [0.55, 0.40, 0.22],
            Tier::Stone => [0.45, 0.45, 0.47],
            Tier::Iron => [0.80, 0.78, 0.74],
            Tier::Gold => [0.92, 0.80, 0.25],
            Tier::Diamond => [0.40, 0.85, 0.85],
        };
    }
    [0.7, 0.7, 0.7]
}

use std::sync::OnceLock;
static TOOL_NAMES: [OnceLock<String>; 25] = [const { OnceLock::new() }; 25];

/// Standard cursor-vs-slot click: left = pick up / drop / merge / swap; right = half / one. Works on
/// any external `slot` (inventory or craft grid). Preserves tool durability; tools never stack.
pub fn slot_click(held: &mut Option<ItemStack>, slot: &mut Option<ItemStack>, right: bool) {
    let with = |item: ItemId, count: u8, dur: u16| ItemStack { item, count, durability: dur };
    match (*held, *slot) {
        (None, Some(s)) => {
            if right {
                let half = s.count.div_ceil(2);
                let rem = s.count - half;
                *held = Some(with(s.item, half, s.durability));
                *slot = (rem > 0).then(|| with(s.item, rem, s.durability));
            } else {
                *held = slot.take();
            }
        }
        (Some(h), None) => {
            if right {
                *slot = Some(with(h.item, 1, h.durability));
                let rem = h.count - 1;
                *held = (rem > 0).then(|| with(h.item, rem, h.durability));
            } else {
                *slot = held.take();
            }
        }
        (Some(h), Some(s)) => {
            if h.item == s.item && !is_tool(h.item) {
                let room = max_stack(s.item).saturating_sub(s.count);
                let moved = if right { 1.min(h.count) } else { h.count }.min(room);
                *slot = Some(with(s.item, s.count + moved, s.durability));
                let rem = h.count - moved;
                *held = (rem > 0).then(|| with(h.item, rem, h.durability));
            } else if !right {
                *slot = Some(h);
                *held = Some(s);
            }
        }
        (None, None) => {}
    }
}

#[derive(Clone, Copy)]
pub struct ItemStack {
    pub item: ItemId,
    pub count: u8,
    /// Remaining durability for tools (0 for non-tools / unused).
    pub durability: u16,
}

impl ItemStack {
    pub fn new(item: ItemId, count: u8) -> Self {
        let durability = if is_tool(item) {
            tool_max_durability(item)
        } else {
            0
        };
        Self {
            item,
            count,
            durability,
        }
    }
    /// The block this stack places (None for tools / non-block items).
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
            // A starter tool set in the first main row (recipes to craft these arrive in M20).
            for (i, &t) in [
                DIAMOND_PICKAXE,
                DIAMOND_AXE,
                DIAMOND_SHOVEL,
                DIAMOND_SWORD,
                STONE_PICKAXE,
            ]
            .iter()
            .enumerate()
            {
                slots[HOTBAR + i] = Some(ItemStack::new(t, 1));
            }
            // Crafting materials in the next main row.
            for (i, &b) in [
                block::CRAFTING_TABLE,
                block::WOOD,
                block::PLANKS,
                block::COBBLESTONE,
            ]
            .iter()
            .enumerate()
            {
                slots[HOTBAR + 5 + i] = Some(ItemStack::new(item_of_block(b), 64));
            }
        }
        Self {
            slots,
            held: None,
            selected: 0,
            creative,
        }
    }

    /// The block the selected hotbar slot would place (AIR if empty / a tool).
    pub fn selected_block(&self) -> BlockId {
        self.slots[self.selected]
            .and_then(|s| s.block())
            .unwrap_or(block::AIR)
    }

    /// The item id in the selected hotbar slot (AIR if empty).
    pub fn selected_item(&self) -> ItemId {
        self.slots[self.selected].map(|s| s.item).unwrap_or(block::AIR)
    }

    /// Damage the selected tool by `amount` on use; remove it if it breaks. No-op for non-tools.
    pub fn damage_selected(&mut self, amount: u16) {
        if let Some(stack) = &mut self.slots[self.selected] {
            if is_tool(stack.item) {
                stack.durability = stack.durability.saturating_sub(amount);
                if stack.durability == 0 {
                    self.slots[self.selected] = None;
                }
            }
        }
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

    /// Drop one of the selected stack (Q): returns the single-item stack to spawn (preserving tool
    /// durability), or None if empty. Decrements the slot unless in creative.
    pub fn drop_one_selected(&mut self) -> Option<ItemStack> {
        let stack = self.slots[self.selected]?;
        if !self.creative {
            let s = self.slots[self.selected].as_mut().unwrap();
            s.count -= 1;
            if s.count == 0 {
                self.slots[self.selected] = None;
            }
        }
        Some(ItemStack {
            item: stack.item,
            count: 1,
            durability: stack.durability,
        })
    }

    /// Empty every slot (and the cursor), returning the stacks — for the death drop.
    pub fn drain_all(&mut self) -> Vec<ItemStack> {
        let mut out = Vec::new();
        for s in self.slots.iter_mut() {
            if let Some(stack) = s.take() {
                out.push(stack);
            }
        }
        if let Some(h) = self.held.take() {
            out.push(h);
        }
        out
    }

    /// Standard inventory-screen click: left = pick up / drop whole stack / merge / swap; right =
    /// pick up half / drop one. Mutates `held` and the clicked `slot`.
    pub fn click_slot(&mut self, slot: usize, right: bool) {
        if slot >= SLOTS {
            return;
        }
        let mut held = self.held;
        let mut cell = self.slots[slot];
        slot_click(&mut held, &mut cell, right);
        self.held = held;
        self.slots[slot] = cell;
    }

    /// Return the held stack to the inventory when a screen closes; yields any leftover that didn't
    /// fit (the caller drops it as a world item rather than stranding it on the hidden cursor).
    pub fn return_held(&mut self) -> Option<ItemStack> {
        self.held.take().and_then(|h| self.insert(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block;

    #[test]
    fn drop_one_decrements_in_survival() {
        let mut inv = Inventory::new(false);
        inv.slots[0] = Some(ItemStack::new(item_of_block(block::STONE), 3));
        assert_eq!(inv.drop_one_selected().unwrap().item, item_of_block(block::STONE));
        assert_eq!(inv.slots[0].unwrap().count, 2);
        inv.slots[0] = Some(ItemStack::new(item_of_block(block::STONE), 1));
        inv.drop_one_selected();
        assert!(inv.slots[0].is_none());
    }

    #[test]
    fn creative_drop_does_not_decrement() {
        let mut inv = Inventory::new(true); // palette pre-loaded, count 1
        let before = inv.slots[0].unwrap().count;
        let _ = inv.drop_one_selected();
        assert_eq!(inv.slots[0].unwrap().count, before);
    }

    #[test]
    fn drain_all_empties_inventory() {
        let mut inv = Inventory::new(false);
        inv.slots[0] = Some(ItemStack::new(item_of_block(block::DIRT), 10));
        inv.slots[20] = Some(ItemStack::new(item_of_block(block::WOOD), 5));
        assert_eq!(inv.drain_all().len(), 2);
        assert!(inv.slots.iter().all(|s| s.is_none()));
    }

    #[test]
    fn block_drops_table() {
        assert_eq!(block::drops(block::STONE), Some(block::COBBLESTONE));
        assert_eq!(block::drops(block::GRASS), Some(block::DIRT));
        assert_eq!(block::drops(block::LEAVES), None);
        assert_eq!(block::drops(block::DIRT), Some(block::DIRT));
    }

    #[test]
    fn tool_id_layout() {
        assert!(is_tool(DIAMOND_PICKAXE) && !is_tool(block::STONE));
        assert_eq!(tool_class(DIAMOND_PICKAXE), block::ToolClass::Pickaxe);
        assert_eq!(tool_class(DIAMOND_SWORD), block::ToolClass::Sword);
        assert!(matches!(tool_tier(DIAMOND_PICKAXE), Tier::Diamond));
        assert!(matches!(tool_tier(STONE_PICKAXE), Tier::Stone));
        assert_eq!(max_stack(DIAMOND_PICKAXE), 1);
        assert_eq!(max_stack(item_of_block(block::STONE)), 64);
        assert_eq!(block_of_item(DIAMOND_PICKAXE), None); // tools place nothing
        assert_eq!(block_of_item(item_of_block(block::STONE)), Some(block::STONE));
    }

    #[test]
    fn harvest_gating() {
        // Diamond ore (required harvest 2) needs >= iron; a stone pickaxe (level 1) can't.
        assert_eq!(block::required_harvest(block::DIAMOND_ORE), 2);
        assert!(harvest_level(Tier::Diamond) >= block::required_harvest(block::DIAMOND_ORE));
        assert!(harvest_level(Tier::Stone) < block::required_harvest(block::DIAMOND_ORE));
        assert!(block::requires_tool(block::STONE) && !block::requires_tool(block::DIRT));
    }

    #[test]
    fn tool_durability_init() {
        let s = ItemStack::new(DIAMOND_PICKAXE, 1);
        assert_eq!(s.durability, tool_max_durability(DIAMOND_PICKAXE));
        let b = ItemStack::new(item_of_block(block::STONE), 10);
        assert_eq!(b.durability, 0);
    }
}
