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

/// Crafting + mob-drop materials (non-block, non-tool items) occupy `[MATERIAL_BASE, +16)`.
pub const MATERIAL_BASE: ItemId = 512;
pub const STICK: ItemId = MATERIAL_BASE;
pub const IRON_INGOT: ItemId = MATERIAL_BASE + 1;
pub const GOLD_INGOT: ItemId = MATERIAL_BASE + 2;
// Mob drops (M29 combat loot). Foods become edible once a hunger-eating milestone lands; the rest
// are crafting materials (leather→armor, bone→bonemeal, gunpowder→TNT, string→bows, …).
pub const BEEF: ItemId = MATERIAL_BASE + 3;
pub const PORK: ItemId = MATERIAL_BASE + 4;
pub const CHICKEN_MEAT: ItemId = MATERIAL_BASE + 5;
pub const MUTTON: ItemId = MATERIAL_BASE + 6;
pub const LEATHER: ItemId = MATERIAL_BASE + 7;
pub const BONE: ItemId = MATERIAL_BASE + 8;
pub const FEATHER: ItemId = MATERIAL_BASE + 9;
pub const GUNPOWDER: ItemId = MATERIAL_BASE + 10;
pub const STRING: ItemId = MATERIAL_BASE + 11;
pub const SPIDER_EYE: ItemId = MATERIAL_BASE + 12;
pub const ROTTEN_FLESH: ItemId = MATERIAL_BASE + 13;
// Foods (P4): cooked meats come from smelting the raw drops; bread/apple are early-game staples.
pub const COOKED_BEEF: ItemId = MATERIAL_BASE + 14;
pub const COOKED_PORK: ItemId = MATERIAL_BASE + 15;
pub const COOKED_CHICKEN: ItemId = MATERIAL_BASE + 16;
pub const COOKED_MUTTON: ItemId = MATERIAL_BASE + 17;
pub const BREAD: ItemId = MATERIAL_BASE + 18;
pub const APPLE: ItemId = MATERIAL_BASE + 19;
// Ore materials (P5a): mined ores now drop these instead of the ore block, so they feed tools/armor/
// fuel. Iron/gold drop raw ore that smelts to an ingot; coal/diamond/redstone/lapis drop directly.
pub const COAL: ItemId = MATERIAL_BASE + 20;
pub const CHARCOAL: ItemId = MATERIAL_BASE + 21;
pub const DIAMOND: ItemId = MATERIAL_BASE + 22;
pub const REDSTONE_DUST: ItemId = MATERIAL_BASE + 23;
pub const LAPIS: ItemId = MATERIAL_BASE + 24;
pub const RAW_IRON: ItemId = MATERIAL_BASE + 25;
pub const RAW_GOLD: ItemId = MATERIAL_BASE + 26;
pub const FLINT: ItemId = MATERIAL_BASE + 27;

/// Size of the material id window (room for foods, dyes, and other crafting materials to come).
pub const MATERIAL_COUNT: ItemId = 64;

fn material_name(item: ItemId) -> &'static str {
    match item {
        STICK => "Stick",
        IRON_INGOT => "Iron Ingot",
        GOLD_INGOT => "Gold Ingot",
        BEEF => "Raw Beef",
        PORK => "Raw Porkchop",
        CHICKEN_MEAT => "Raw Chicken",
        MUTTON => "Raw Mutton",
        LEATHER => "Leather",
        BONE => "Bone",
        FEATHER => "Feather",
        GUNPOWDER => "Gunpowder",
        STRING => "String",
        SPIDER_EYE => "Spider Eye",
        ROTTEN_FLESH => "Rotten Flesh",
        COOKED_BEEF => "Steak",
        COOKED_PORK => "Cooked Porkchop",
        COOKED_CHICKEN => "Cooked Chicken",
        COOKED_MUTTON => "Cooked Mutton",
        BREAD => "Bread",
        APPLE => "Apple",
        COAL => "Coal",
        CHARCOAL => "Charcoal",
        DIAMOND => "Diamond",
        REDSTONE_DUST => "Redstone Dust",
        LAPIS => "Lapis Lazuli",
        RAW_IRON => "Raw Iron",
        RAW_GOLD => "Raw Gold",
        FLINT => "Flint",
        _ => "Material",
    }
}

/// Flat display color of a material item (UI swatch + the world-drop tile).
pub fn material_color(item: ItemId) -> [f32; 3] {
    match item {
        IRON_INGOT => [0.80, 0.78, 0.74],
        GOLD_INGOT => [0.95, 0.80, 0.22],
        BEEF => [0.78, 0.25, 0.22],
        PORK => [0.92, 0.62, 0.62],
        CHICKEN_MEAT => [0.90, 0.72, 0.55],
        MUTTON => [0.82, 0.40, 0.36],
        LEATHER => [0.62, 0.43, 0.24],
        BONE => [0.92, 0.91, 0.82],
        FEATHER => [0.95, 0.95, 0.96],
        GUNPOWDER => [0.22, 0.22, 0.24],
        STRING => [0.88, 0.88, 0.86],
        SPIDER_EYE => [0.55, 0.18, 0.20],
        ROTTEN_FLESH => [0.52, 0.36, 0.30],
        COOKED_BEEF => [0.55, 0.32, 0.20],
        COOKED_PORK => [0.80, 0.52, 0.45],
        COOKED_CHICKEN => [0.78, 0.60, 0.40],
        COOKED_MUTTON => [0.62, 0.34, 0.28],
        BREAD => [0.75, 0.58, 0.30],
        APPLE => [0.82, 0.18, 0.18],
        COAL => [0.13, 0.13, 0.14],
        CHARCOAL => [0.24, 0.20, 0.17],
        DIAMOND => [0.45, 0.92, 0.90],
        REDSTONE_DUST => [0.80, 0.10, 0.10],
        LAPIS => [0.18, 0.30, 0.78],
        RAW_IRON => [0.78, 0.62, 0.50],
        RAW_GOLD => [0.85, 0.70, 0.30],
        FLINT => [0.30, 0.28, 0.28],
        _ => [0.55, 0.40, 0.22], // stick / generic wooden
    }
}

/// Armor (non-stacking wearables) occupy `[ARMOR_BASE, ARMOR_BASE+16)`, laid out as
/// `ARMOR_BASE + tier*4 + piece` — piece 0 Helmet, 1 Chestplate, 2 Leggings, 3 Boots; tiers
/// 0 Leather, 1 Iron, 2 Gold, 3 Diamond.
pub const ARMOR_BASE: ItemId = 768;

/// Armor id from a tier (0 Leather, 1 Iron, 2 Gold, 3 Diamond) and piece (0 Helmet … 3 Boots).
/// Named per-piece consts are added when armor crafting recipes land (a later milestone).
#[inline]
pub const fn armor_id(tier: u16, piece: u16) -> ItemId {
    ARMOR_BASE + tier * 4 + piece
}

#[inline]
pub fn is_armor(item: ItemId) -> bool {
    (ARMOR_BASE..ARMOR_BASE + 16).contains(&item)
}

/// Inventory armor slot (36..40) a piece equips into: helmet→36, chest→37, legs→38, boots→39.
#[inline]
pub fn armor_slot(item: ItemId) -> usize {
    HOTBAR + MAIN + ((item - ARMOR_BASE) % 4) as usize
}

/// Defense points for one piece (Minecraft values), indexed [tier][piece].
pub fn armor_points(item: ItemId) -> u32 {
    const PTS: [[u32; 4]; 4] = [
        [1, 3, 2, 1], // leather
        [2, 6, 5, 2], // iron
        [2, 5, 3, 1], // gold
        [3, 8, 6, 3], // diamond
    ];
    if !is_armor(item) {
        return 0;
    }
    let tier = ((item - ARMOR_BASE) / 4) as usize;
    let piece = ((item - ARMOR_BASE) % 4) as usize;
    PTS[tier][piece]
}

static ARMOR_NAMES: [&str; 16] = [
    "Leather Helmet", "Leather Chestplate", "Leather Leggings", "Leather Boots",
    "Iron Helmet", "Iron Chestplate", "Iron Leggings", "Iron Boots",
    "Golden Helmet", "Golden Chestplate", "Golden Leggings", "Golden Boots",
    "Diamond Helmet", "Diamond Chestplate", "Diamond Leggings", "Diamond Boots",
];

fn armor_name(item: ItemId) -> &'static str {
    ARMOR_NAMES[(item - ARMOR_BASE) as usize]
}

fn armor_color(item: ItemId) -> [f32; 3] {
    match (item - ARMOR_BASE) / 4 {
        0 => [0.55, 0.36, 0.22], // leather
        1 => [0.80, 0.78, 0.74], // iron
        2 => [0.95, 0.80, 0.22], // gold
        _ => [0.40, 0.85, 0.85], // diamond
    }
}

#[inline]
pub fn is_tool(item: ItemId) -> bool {
    (TOOL_BASE..TOOL_BASE + 25).contains(&item)
}

#[inline]
pub fn is_material(item: ItemId) -> bool {
    (MATERIAL_BASE..MATERIAL_BASE + MATERIAL_COUNT).contains(&item)
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

/// Max stack size: tools and armor don't stack.
#[inline]
pub fn max_stack(item: ItemId) -> u8 {
    if is_tool(item) || is_armor(item) {
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
    if is_armor(item) {
        return armor_name(item);
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
    // Mob-drop materials get a flat-colored atlas tile so a dropped pile reads as its real item
    // (beef/leather/bone/…) instead of a generic wooden box.
    if (BEEF..=ROTTEN_FLESH).contains(&item) {
        return block::tile::MATERIAL_DROP + (item - BEEF) as u32;
    }
    match block_of_item(item) {
        Some(b) => block::face_tile(b, [0, 1, 0]),
        None => block::tile::PLANKS, // generic handle-colored placeholder for tools/sticks/ingots
    }
}

/// Self-emission of a dropped item (lava block glows; tools don't).
pub fn item_emission(item: ItemId) -> f32 {
    block_of_item(item).map(block::emission).unwrap_or(0.0)
}

/// True if `item` is a registered block, tool, or material (used to reject corrupt save ids).
/// Only *defined* material ids count — the unpopulated tail of the material range is rejected — and
/// the block bound tracks `block::MAX_BLOCK` so it stays in sync as new blocks are added.
pub fn is_known(item: ItemId) -> bool {
    is_tool(item)
        || is_armor(item)
        // A *defined* material (material_name returns its real name, not the "Material" fallback) —
        // so the unpopulated tail of the material range is still rejected from corrupt saves.
        || (is_material(item) && material_name(item) != "Material")
        || (item != block::AIR && item <= block::MAX_BLOCK)
}

/// Representative UI color for an item: the block face color, or a tier color for tools.
pub fn item_color(item: ItemId) -> [f32; 3] {
    if let Some(b) = block_of_item(item) {
        return block::face_color(b, [0, 1, 0]);
    }
    if is_material(item) {
        return material_color(item);
    }
    if is_armor(item) {
        return armor_color(item);
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

#[derive(Clone)]
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
            // A diamond armor set equipped in the 4 armor slots.
            for piece in 0..4u16 {
                slots[HOTBAR + MAIN + piece as usize] = Some(ItemStack::new(armor_id(3, piece), 1));
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

    /// Total defense points from the 4 equipped armor pieces (slots `[36,40)`).
    pub fn equipped_armor(&self) -> u32 {
        self.slots[HOTBAR + MAIN..]
            .iter()
            .flatten()
            .map(|s| armor_points(s.item))
            .sum()
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
        // Fill empty slots, preserving the incoming durability (tools must not be silently repaired).
        for slot in self.slots[0..HOTBAR + MAIN].iter_mut() {
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
    fn insert_preserves_tool_durability() {
        // A damaged tool returned to the inventory must keep its wear (no free repair).
        let mut inv = Inventory::new(false);
        let mut pick = ItemStack::new(WOOD_PICKAXE, 1);
        pick.durability = 10;
        assert!(inv.insert(pick).is_none());
        let found = inv.slots.iter().flatten().find(|s| s.item == WOOD_PICKAXE).copied();
        assert_eq!(found.map(|s| s.durability), Some(10), "durability must survive insert");
    }

    #[test]
    fn unknown_material_ids_rejected() {
        assert!(is_known(STICK) && is_known(IRON_INGOT) && is_known(GOLD_INGOT));
        assert!(is_known(LEATHER) && is_known(GUNPOWDER)); // defined mob-drop materials
        assert!(is_known(COOKED_BEEF) && is_known(BREAD)); // defined foods
        // The unpopulated tail of the material range is still rejected (name falls back to "Material").
        assert!(!is_known(MATERIAL_BASE + 40), "undefined material id should be rejected");
    }

    #[test]
    fn held_stack_folds_into_slots_on_save() {
        // The save path clones the inventory and folds the cursor stack back into slots so it
        // isn't lost when saving with a screen open.
        let mut inv = Inventory::new(false);
        inv.held = Some(ItemStack::new(item_of_block(block::STONE), 10));
        let mut saved = inv.clone();
        if let Some(h) = saved.held.take() {
            let _ = saved.insert(h);
        }
        assert!(saved.held.is_none());
        let total: u32 = saved
            .slots
            .iter()
            .flatten()
            .filter(|s| s.item == item_of_block(block::STONE))
            .map(|s| s.count as u32)
            .sum();
        assert_eq!(total, 10, "held stack must survive into slots");
    }

    #[test]
    fn armor_defense_and_slots() {
        // Slot mapping: helmet→36, boots→39.
        assert_eq!(armor_slot(armor_id(0, 0)), HOTBAR + MAIN);
        assert_eq!(armor_slot(armor_id(3, 3)), HOTBAR + MAIN + 3);
        // Minecraft defense values.
        assert_eq!(armor_points(armor_id(3, 1)), 8); // diamond chestplate
        assert_eq!(armor_points(armor_id(0, 3)), 1); // leather boots
        assert_eq!(armor_points(STICK), 0); // non-armor has no defense
        // Armor doesn't stack, and is a known/valid item.
        assert_eq!(max_stack(armor_id(1, 0)), 1);
        assert!(is_armor(armor_id(2, 2)) && is_known(armor_id(2, 2)));
        // A full diamond set sums to 20 points (the reduction cap).
        let total: u32 = (0..4).map(|p| armor_points(armor_id(3, p))).sum();
        assert_eq!(total, 20);
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
        assert_eq!(block::drops(block::STONE), Some((block::COBBLESTONE, 1)));
        assert_eq!(block::drops(block::GRASS), Some((block::DIRT, 1)));
        assert_eq!(block::drops(block::LEAVES), None);
        assert_eq!(block::drops(block::DIRT), Some((block::DIRT, 1)));
        // Ores drop their material item (not the ore block).
        assert_eq!(block::drops(block::COAL_ORE), Some((COAL, 1)));
        assert_eq!(block::drops(block::DIAMOND_ORE), Some((DIAMOND, 1)));
        assert_eq!(block::drops(block::IRON_ORE), Some((RAW_IRON, 1)));
        assert_eq!(block::drops(block::REDSTONE_ORE), Some((REDSTONE_DUST, 4)));
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
