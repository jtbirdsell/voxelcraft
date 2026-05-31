//! Block registry (M3). Blocks are `u16` ids; properties come from small tables.
//! Grows into a proper data-driven registry in later milestones.

pub type BlockId = u16;

pub const AIR: BlockId = 0;
pub const STONE: BlockId = 1;
pub const DIRT: BlockId = 2;
pub const GRASS: BlockId = 3;
pub const SAND: BlockId = 4;
pub const WOOD: BlockId = 5;
pub const LEAVES: BlockId = 6;
pub const WATER: BlockId = 7;
pub const SNOW: BlockId = 8;
pub const COAL_ORE: BlockId = 9;
pub const IRON_ORE: BlockId = 10;
pub const LAVA: BlockId = 11;
pub const TORCH: BlockId = 12;
pub const GLOWSTONE: BlockId = 13;
pub const COBBLESTONE: BlockId = 14;
pub const PLANKS: BlockId = 15;
pub const BRICKS: BlockId = 16;
pub const BEDROCK: BlockId = 17;
pub const GRAVEL: BlockId = 18;
pub const OBSIDIAN: BlockId = 19;
pub const GOLD_ORE: BlockId = 20;
pub const DIAMOND_ORE: BlockId = 21;
pub const REDSTONE_ORE: BlockId = 22;
pub const LAPIS_ORE: BlockId = 23;
pub const DEEPSLATE: BlockId = 24;
pub const CRAFTING_TABLE: BlockId = 25;
pub const FURNACE: BlockId = 26;
pub const CHEST: BlockId = 27;
pub const POPPY: BlockId = 28;
pub const DANDELION: BlockId = 29;
pub const TALL_GRASS: BlockId = 30;
pub const CACTUS: BlockId = 31;
// M25 decoration.
pub const FERN: BlockId = 32;
pub const RED_MUSHROOM: BlockId = 33;
pub const BROWN_MUSHROOM: BlockId = 34;
pub const SUGAR_CANE: BlockId = 35;
pub const PUMPKIN: BlockId = 36;
pub const ICE: BlockId = 37;
// M26 glass + partial blocks.
pub const GLASS: BlockId = 38;
pub const STONE_SLAB: BlockId = 39;
pub const STONE_STAIRS: BlockId = 40;
pub const WOOD_SLAB: BlockId = 41;

/// Highest defined block id; bounds save-id validation (keep in sync as blocks are added).
pub const MAX_BLOCK: BlockId = WOOD_SLAB;

/// How a block is meshed: a full greedy cube, a cross billboard (plants), or a non-greedy partial
/// shape (slab / stairs) emitted one cell at a time like a billboard.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderKind {
    Cube,
    Cross,
    Slab,
    Stairs,
}

#[inline]
pub fn render_kind(id: BlockId) -> RenderKind {
    match id {
        POPPY | DANDELION | TALL_GRASS | FERN | RED_MUSHROOM | BROWN_MUSHROOM | SUGAR_CANE => {
            RenderKind::Cross
        }
        STONE_SLAB | WOOD_SLAB => RenderKind::Slab,
        STONE_STAIRS => RenderKind::Stairs,
        _ => RenderKind::Cube,
    }
}

/// A non-solid cross-billboard plant (walk-through, casts no shadow, doesn't cull neighbors).
#[inline]
pub fn is_plant(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Cross)
}

/// A partial-geometry block (slab/stairs): solid for collision but emitted per-cell, not greedy.
#[inline]
pub fn is_partial(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Slab | RenderKind::Stairs)
}

/// Translucent glass: a full cube that renders in its own alpha-blended pass and never blocks light.
#[inline]
pub fn is_glass(id: BlockId) -> bool {
    id == GLASS
}

/// A full greedy-meshed cube (cubes incl. glass; excludes cross plants and partials).
#[inline]
pub fn is_cube(id: BlockId) -> bool {
    matches!(render_kind(id), RenderKind::Cube)
}

/// Whether a block occupies the ray-traced voxel volume as a full cube (casts shadows / AO). Glass
/// is see-through (stored as 0); slabs/stairs are approximated as full cubes (like leaves today).
#[inline]
pub fn is_volume_solid(id: BlockId) -> bool {
    is_opaque(id) || is_partial(id)
}

/// A block participates in collision (fluids and cross-billboard plants are passable).
#[inline]
pub fn is_solid(id: BlockId) -> bool {
    id != AIR && id != WATER && id != LAVA && !is_plant(id)
}

/// A solid sub-box in a block's local 0..1 space: `[minx, miny, minz, maxx, maxy, maxz]`.
pub type Aabb = [f32; 6];
const BOX_FULL: [Aabb; 1] = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]];
const BOX_SLAB: [Aabb; 1] = [[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]];
// Stairs: a bottom slab plus the upper-back quarter (fixed orientation, stepping toward +z).
const BOX_STAIRS: [Aabb; 2] = [
    [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
    [0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
];
const BOX_NONE: [Aabb; 0] = [];

/// The solid collision boxes of a block (empty if passable). Full cubes are a single unit box;
/// slabs/stairs return their true partial shape so the player can stand at half-height / step up.
#[inline]
pub fn solid_boxes(id: BlockId) -> &'static [Aabb] {
    if !is_solid(id) {
        return &BOX_NONE;
    }
    match render_kind(id) {
        RenderKind::Slab => &BOX_SLAB,
        RenderKind::Stairs => &BOX_STAIRS,
        _ => &BOX_FULL,
    }
}

/// Water or lava — simulated by the flowing-fluid tick and passable to the player.
#[inline]
pub fn is_fluid(id: BlockId) -> bool {
    id == WATER || id == LAVA
}

/// A block fully hides the touching face of an adjacent opaque block.
/// Water and cross-billboard plants are non-opaque; leaves stay opaque (rendered as solid foliage).
#[inline]
pub fn is_opaque(id: BlockId) -> bool {
    id != AIR && id != WATER && id != GLASS && !is_plant(id) && !is_partial(id)
}

/// Whether a block produces any geometry at all.
#[inline]
pub fn renders(id: BlockId) -> bool {
    id != AIR
}

#[inline]
pub fn is_water(id: BlockId) -> bool {
    id == WATER
}

/// Does `neighbor` hide the face of `self_id` that touches it?
/// Opaque neighbors hide everything; a translucent block hides only the same translucent
/// type (so water surfaces show against air/solids but internal water faces are culled).
#[inline]
pub fn occludes(self_id: BlockId, neighbor: BlockId) -> bool {
    if neighbor == AIR {
        false
    } else if is_opaque(neighbor) {
        true
    } else {
        self_id == neighbor
    }
}

/// Human-readable block name (hotbar label, tooltips, F3).
pub fn display_name(id: BlockId) -> &'static str {
    match id {
        AIR => "Air",
        STONE => "Stone",
        DIRT => "Dirt",
        GRASS => "Grass Block",
        SAND => "Sand",
        WOOD => "Wood",
        LEAVES => "Leaves",
        WATER => "Water",
        SNOW => "Snow",
        COAL_ORE => "Coal Ore",
        IRON_ORE => "Iron Ore",
        LAVA => "Lava",
        TORCH => "Torch",
        GLOWSTONE => "Glowstone",
        COBBLESTONE => "Cobblestone",
        PLANKS => "Wood Planks",
        BRICKS => "Bricks",
        BEDROCK => "Bedrock",
        GRAVEL => "Gravel",
        OBSIDIAN => "Obsidian",
        GOLD_ORE => "Gold Ore",
        DIAMOND_ORE => "Diamond Ore",
        REDSTONE_ORE => "Redstone Ore",
        LAPIS_ORE => "Lapis Ore",
        DEEPSLATE => "Deepslate",
        CRAFTING_TABLE => "Crafting Table",
        FURNACE => "Furnace",
        CHEST => "Chest",
        POPPY => "Poppy",
        DANDELION => "Dandelion",
        TALL_GRASS => "Tall Grass",
        CACTUS => "Cactus",
        FERN => "Fern",
        RED_MUSHROOM => "Red Mushroom",
        BROWN_MUSHROOM => "Brown Mushroom",
        SUGAR_CANE => "Sugar Cane",
        PUMPKIN => "Pumpkin",
        ICE => "Ice",
        GLASS => "Glass",
        STONE_SLAB => "Stone Slab",
        STONE_STAIRS => "Stone Stairs",
        WOOD_SLAB => "Wood Slab",
        _ => "Unknown",
    }
}

/// Base albedo for a block face. `face_offset[1] == 1` is the +Y (top) face.
pub fn face_color(id: BlockId, face_offset: [i32; 3]) -> [f32; 3] {
    let top = face_offset[1] == 1;
    let bottom = face_offset[1] == -1;
    match id {
        GRASS => {
            if top {
                [0.36, 0.60, 0.27]
            } else if bottom {
                [0.45, 0.33, 0.21]
            } else {
                // side: dirt with a thin grass lip looks busy; keep dirt-ish with green tint
                [0.42, 0.42, 0.24]
            }
        }
        DIRT => [0.45, 0.33, 0.21],
        STONE => [0.49, 0.49, 0.52],
        SAND => [0.80, 0.75, 0.52],
        WOOD => {
            if top || bottom {
                [0.55, 0.43, 0.27]
            } else {
                [0.40, 0.30, 0.18]
            }
        }
        LEAVES => [0.20, 0.42, 0.18],
        WATER => [0.16, 0.34, 0.62],
        SNOW => [0.92, 0.94, 0.97],
        COAL_ORE => [0.28, 0.28, 0.30],
        IRON_ORE => [0.60, 0.52, 0.45],
        LAVA => [1.0, 0.42, 0.06],
        TORCH => [0.85, 0.62, 0.28],
        GLOWSTONE => [0.95, 0.82, 0.45],
        COBBLESTONE => [0.42, 0.42, 0.44],
        PLANKS => [0.62, 0.48, 0.30],
        BRICKS => [0.55, 0.28, 0.22],
        BEDROCK => [0.20, 0.20, 0.22],
        GRAVEL => [0.50, 0.47, 0.45],
        OBSIDIAN => [0.12, 0.09, 0.18],
        GOLD_ORE => [0.55, 0.50, 0.35],
        DIAMOND_ORE => [0.45, 0.62, 0.62],
        REDSTONE_ORE => [0.45, 0.30, 0.30],
        LAPIS_ORE => [0.30, 0.35, 0.55],
        DEEPSLATE => [0.22, 0.22, 0.25],
        CRAFTING_TABLE => [0.50, 0.36, 0.22],
        FURNACE => [0.38, 0.38, 0.40],
        CHEST => [0.55, 0.42, 0.24],
        POPPY => [0.80, 0.15, 0.12],
        DANDELION => [0.90, 0.82, 0.20],
        TALL_GRASS => [0.34, 0.55, 0.24],
        CACTUS => [0.30, 0.52, 0.24],
        FERN => [0.27, 0.46, 0.20],
        RED_MUSHROOM => [0.80, 0.20, 0.18],
        BROWN_MUSHROOM => [0.55, 0.40, 0.28],
        SUGAR_CANE => [0.55, 0.72, 0.40],
        PUMPKIN => {
            if top || bottom {
                [0.80, 0.52, 0.14] // mirror face_tile (top tile on both caps) for the GI invariant
            } else {
                [0.78, 0.48, 0.12]
            }
        }
        ICE => [0.66, 0.80, 0.92],
        GLASS => [0.82, 0.91, 0.98],
        STONE_SLAB | STONE_STAIRS => [0.49, 0.49, 0.52], // stone
        WOOD_SLAB => [0.62, 0.48, 0.30],                 // planks
        _ => [1.0, 0.0, 1.0],
    }
}

/// Self-emission strength per block (0 = unlit material). Lava glows; the lit shaders add this as
/// extra outgoing radiance, and GI/reflection rays treat an emissive hit as a light source.
pub fn emission(id: BlockId) -> f32 {
    match id {
        LAVA => 1.0,
        TORCH => 0.9,
        GLOWSTONE => 1.0,
        _ => 0.0,
    }
}

/// Block-light emitted (0..15), for the light flood (M14b). Torch/glowstone/lava are sources.
pub fn light_emission(id: BlockId) -> u8 {
    match id {
        GLOWSTONE => 15,
        TORCH => 14,
        LAVA => 11,
        _ => 0,
    }
}

/// Light lost when entering this block (1 for air/transparent; 15 fully blocks). Opaque solids stop
/// light; fluids and (later) glass/foliage let it pass with light attenuation.
/// Reserved for graduated light propagation (the BFS currently uses a uniform step).
#[allow(dead_code)]
pub fn light_attenuation(id: BlockId) -> u8 {
    match id {
        AIR => 1,
        WATER => 2,
        _ if is_opaque(id) => 15,
        _ => 1,
    }
}

/// Whether light can pass through this block at all. Reserved for the graduated light pass.
#[allow(dead_code)]
#[inline]
pub fn transmits_light(id: BlockId) -> bool {
    !is_opaque(id)
}

/// Whether this block stops skylight descending. Solid terrain, roofs, and partial blocks (slabs/
/// stairs) do; glass and foliage (leaves) let it through so windows/tree canopies don't cast
/// pitch-black ground shadows (binary skylight, M14). Uses `is_volume_solid` (= opaque cubes +
/// partials) so slab/stair roofs still darken what's under them (M26 made `is_opaque` exclude them).
#[inline]
pub fn blocks_skylight(id: BlockId) -> bool {
    is_volume_solid(id) && id != LEAVES
}

/// The tool that mines a block fastest (for mining-speed + drop gating from M19).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolClass {
    None,
    Pickaxe,
    Axe,
    Shovel,
    Sword,
    Hoe,
}

/// Whether a block needs the correct tool to yield a drop (pickaxe blocks do; wood/dirt drop by hand).
pub fn requires_tool(id: BlockId) -> bool {
    matches!(tool_class(id), ToolClass::Pickaxe)
}

/// Minimum harvest level needed to get a drop (0 = any pickaxe). Iron/lapis need stone, gold/diamond/
/// redstone need iron, obsidian needs diamond.
pub fn required_harvest(id: BlockId) -> u8 {
    match id {
        IRON_ORE | LAPIS_ORE => 1,
        GOLD_ORE | DIAMOND_ORE | REDSTONE_ORE => 2,
        OBSIDIAN => 3,
        _ => 0,
    }
}

/// Time in seconds to break a block by hand (tools speed this up in M19). INFINITY = unbreakable.
pub fn hardness(id: BlockId) -> f32 {
    match id {
        BEDROCK => f32::INFINITY,
        AIR | WATER | LAVA => 0.0,
        LEAVES | TORCH | GLOWSTONE => 0.3,
        DIRT | GRASS | SAND | GRAVEL | SNOW => 0.6,
        WOOD | PLANKS | CRAFTING_TABLE | CHEST => 1.2,
        STONE | COBBLESTONE | BRICKS | COAL_ORE | IRON_ORE | GOLD_ORE | DIAMOND_ORE
        | REDSTONE_ORE | LAPIS_ORE => 1.5,
        DEEPSLATE | FURNACE => 2.0,
        OBSIDIAN => 8.0,
        ICE => 0.5,
        PUMPKIN => 1.0,
        GLASS => 0.3,
        STONE_SLAB | STONE_STAIRS => 1.5,
        WOOD_SLAB => 1.2,
        _ if is_plant(id) => 0.0,
        _ => 1.0,
    }
}

/// The tool class that mines a block fastest (drives tool speed + drop gating).
pub fn tool_class(id: BlockId) -> ToolClass {
    match id {
        STONE | COBBLESTONE | BRICKS | DEEPSLATE | OBSIDIAN | FURNACE | COAL_ORE | IRON_ORE
        | GOLD_ORE | DIAMOND_ORE | REDSTONE_ORE | LAPIS_ORE => ToolClass::Pickaxe,
        ICE | STONE_SLAB | STONE_STAIRS => ToolClass::Pickaxe,
        WOOD | PLANKS | CRAFTING_TABLE | CHEST | PUMPKIN | WOOD_SLAB => ToolClass::Axe,
        DIRT | GRASS | SAND | GRAVEL | SNOW => ToolClass::Shovel,
        _ => ToolClass::None,
    }
}

/// Whether a block can be broken at all (bedrock and air can't).
#[inline]
pub fn breakable(id: BlockId) -> bool {
    id != AIR && hardness(id).is_finite()
}

/// What a broken block yields as an item drop (None = nothing). Stone yields cobblestone, grass
/// yields dirt, leaves drop nothing; most blocks drop themselves. (Tool gating arrives in M19.)
pub fn drops(id: BlockId) -> Option<BlockId> {
    match id {
        AIR => None,
        STONE => Some(COBBLESTONE),
        GRASS => Some(DIRT),
        LEAVES => None,
        ICE => None,   // melts away (no silk touch yet)
        GLASS => None, // shatters
        _ => Some(id),
    }
}

/// Experience awarded for mining a block (the ores that drop XP in Minecraft). Iron/gold give none
/// here — their XP comes from smelting the ore in a furnace.
pub fn mining_xp(id: BlockId) -> u32 {
    match id {
        COAL_ORE => 1,
        REDSTONE_ORE | LAPIS_ORE => 2,
        DIAMOND_ORE => 4,
        _ => 0,
    }
}

/// Atlas tile ids (M13). Index = `row*ATLAS_COLS + col` into the procedural texture atlas. Kept
/// beside `face_color`/`face_tile` so the painters (texture.rs) and the WGSL `tile_average()` stay
/// in lockstep. The 0..63 space leaves ample room for the block expansion in M16.
pub mod tile {
    pub const STONE: u32 = 0;
    pub const DIRT: u32 = 1;
    pub const GRASS_TOP: u32 = 2;
    pub const GRASS_SIDE: u32 = 3;
    pub const SAND: u32 = 4;
    pub const WOOD_TOP: u32 = 5;
    pub const WOOD_SIDE: u32 = 6;
    pub const LEAVES: u32 = 7;
    pub const WATER: u32 = 8;
    pub const SNOW: u32 = 9;
    pub const COAL: u32 = 10;
    pub const IRON: u32 = 11;
    pub const LAVA: u32 = 12;
    pub const MOB: u32 = 13;
    pub const MOB_HEAD: u32 = 14;
    pub const TORCH: u32 = 15;
    pub const GLOWSTONE: u32 = 16;
    pub const COBBLE: u32 = 17;
    pub const PLANKS: u32 = 18;
    pub const BRICKS: u32 = 19;
    pub const BEDROCK: u32 = 20;
    pub const GRAVEL: u32 = 21;
    pub const OBSIDIAN: u32 = 22;
    pub const GOLD: u32 = 23;
    pub const DIAMOND: u32 = 24;
    pub const REDSTONE: u32 = 25;
    pub const LAPIS: u32 = 26;
    pub const DEEPSLATE: u32 = 27;
    pub const CRAFTING: u32 = 28;
    pub const FURNACE: u32 = 29;
    pub const CHEST: u32 = 30;
    pub const POPPY: u32 = 31;
    pub const DANDELION: u32 = 32;
    pub const TALL_GRASS: u32 = 33;
    pub const CACTUS: u32 = 34;
    pub const FERN: u32 = 35;
    pub const RED_MUSHROOM: u32 = 36;
    pub const BROWN_MUSHROOM: u32 = 37;
    pub const SUGAR_CANE: u32 = 38;
    pub const PUMPKIN_TOP: u32 = 39;
    pub const PUMPKIN_SIDE: u32 = 40;
    pub const ICE: u32 = 41;
    pub const GLASS: u32 = 42;
    // M27 typed-mob body colors.
    pub const MOB_COW: u32 = 43;
    pub const MOB_PIG: u32 = 44;
    pub const MOB_SHEEP: u32 = 45;
    pub const MOB_CHICKEN: u32 = 46;
    pub const MOB_ZOMBIE: u32 = 47;
    pub const MOB_SKELETON: u32 = 48;
    pub const MOB_CREEPER: u32 = 49;
    pub const MOB_SPIDER: u32 = 50;
    pub const MAGENTA: u32 = 63; // missing/unknown sentinel
}

/// Tint class for a face: 0 = use texel as-is, 1 = multiply by foliage (grass/leaves) biome tint,
/// 2 = water tint. Carried in the vertex `shade.y`; applied in-shader from M13d (biome tint) on.
pub fn tint_class(id: BlockId, _face_offset: [i32; 3]) -> f32 {
    match id {
        GRASS | LEAVES => 1.0,
        WATER => 2.0,
        _ => 0.0,
    }
}

/// Atlas tile for a block face. Mirrors `face_color`'s per-face logic (grass top/side/bottom,
/// wood end/side) so `tile_average(face_tile(id, face))` reproduces `face_color(id, face)`.
pub fn face_tile(id: BlockId, face_offset: [i32; 3]) -> u32 {
    let top = face_offset[1] == 1;
    let bottom = face_offset[1] == -1;
    match id {
        GRASS => {
            if top {
                tile::GRASS_TOP
            } else if bottom {
                tile::DIRT
            } else {
                tile::GRASS_SIDE
            }
        }
        DIRT => tile::DIRT,
        STONE => tile::STONE,
        SAND => tile::SAND,
        WOOD => {
            if top || bottom {
                tile::WOOD_TOP
            } else {
                tile::WOOD_SIDE
            }
        }
        LEAVES => tile::LEAVES,
        WATER => tile::WATER,
        SNOW => tile::SNOW,
        COAL_ORE => tile::COAL,
        IRON_ORE => tile::IRON,
        LAVA => tile::LAVA,
        TORCH => tile::TORCH,
        GLOWSTONE => tile::GLOWSTONE,
        COBBLESTONE => tile::COBBLE,
        PLANKS => tile::PLANKS,
        BRICKS => tile::BRICKS,
        BEDROCK => tile::BEDROCK,
        GRAVEL => tile::GRAVEL,
        OBSIDIAN => tile::OBSIDIAN,
        GOLD_ORE => tile::GOLD,
        DIAMOND_ORE => tile::DIAMOND,
        REDSTONE_ORE => tile::REDSTONE,
        LAPIS_ORE => tile::LAPIS,
        DEEPSLATE => tile::DEEPSLATE,
        CRAFTING_TABLE => tile::CRAFTING,
        FURNACE => tile::FURNACE,
        CHEST => tile::CHEST,
        POPPY => tile::POPPY,
        DANDELION => tile::DANDELION,
        TALL_GRASS => tile::TALL_GRASS,
        CACTUS => tile::CACTUS,
        FERN => tile::FERN,
        RED_MUSHROOM => tile::RED_MUSHROOM,
        BROWN_MUSHROOM => tile::BROWN_MUSHROOM,
        SUGAR_CANE => tile::SUGAR_CANE,
        PUMPKIN => {
            if top || bottom {
                tile::PUMPKIN_TOP
            } else {
                tile::PUMPKIN_SIDE
            }
        }
        ICE => tile::ICE,
        GLASS => tile::GLASS,
        STONE_SLAB | STONE_STAIRS => tile::STONE,
        WOOD_SLAB => tile::PLANKS,
        _ => tile::MAGENTA,
    }
}
