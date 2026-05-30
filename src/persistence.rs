//! World persistence. Player-edited chunks are saved (LZ4-compressed) to `chunks.bin`; the
//! seed, spawn, look, time and fly state go in `level.bin`. Unedited chunks regenerate
//! deterministically from the seed, so only modifications need to hit disk.

use std::fs;
use std::path::{Path, PathBuf};

use glam::IVec3;
use rustc_hash::FxHashMap;

use crate::block::BlockId;
use crate::item::{Inventory, ItemStack, HOTBAR, SLOTS};
use crate::world::{Chunk, CHUNK_VOLUME};

const MAGIC: u32 = 0x5643_5231; // "VCR1"
const INV_MAGIC: u32 = 0x5643_4956; // "VCIV" — inventory save_state.bin

pub struct Level {
    pub seed: u64,
    pub spawn: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub time: f32,
    pub flying: bool,
    /// Player survival state (M24). Appended after the legacy 33-byte header; absent in old saves,
    /// where these default to a full/fresh player (identical to a just-spawned one).
    pub health: f32,
    pub hunger: f32,
    pub air: f32,
    pub saturation: f32,
    pub xp: f32,
    pub level: u32,
}

/// One furnace's persisted contents (M24), saved in the container section of `save_state.bin`.
/// Plain primitives + `ItemStack` so persistence stays decoupled from `game::FurnaceState`.
pub struct FurnaceSave {
    pub pos: IVec3,
    pub input: Option<ItemStack>,
    pub fuel: Option<ItemStack>,
    pub output: Option<ItemStack>,
    pub burn_remaining: f32,
    pub burn_max: f32,
    pub cook_progress: f32,
    pub cook_item: u16,
}

pub fn save_dir() -> PathBuf {
    PathBuf::from("saves").join("world")
}

fn rd_u32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn rd_i32(d: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn rd_f32(d: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}

pub fn load_level(dir: &Path) -> Option<Level> {
    let d = fs::read(dir.join("level.bin")).ok()?;
    if d.len() < 33 {
        return None;
    }
    let seed = u64::from_le_bytes(d[0..8].try_into().ok()?);
    let spawn = [rd_f32(&d, 8), rd_f32(&d, 12), rd_f32(&d, 16)];
    // Survival block (24 bytes from offset 33) is present only in M24+ saves; default to full.
    let has_survival = d.len() >= 33 + 24;
    let sv = |o: usize, default: f32| if has_survival { rd_f32(&d, o) } else { default };
    Some(Level {
        seed,
        spawn,
        yaw: rd_f32(&d, 20),
        pitch: rd_f32(&d, 24),
        time: rd_f32(&d, 28),
        flying: d[32] != 0,
        health: sv(33, 20.0),
        hunger: sv(37, 20.0),
        air: sv(41, 11.0),
        saturation: sv(45, 5.0),
        xp: sv(49, 0.0),
        level: if has_survival { rd_u32(&d, 53) } else { 0 },
    })
}

pub fn save_level(dir: &Path, level: &Level) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut b = Vec::with_capacity(57);
    b.extend_from_slice(&level.seed.to_le_bytes());
    for c in level.spawn {
        b.extend_from_slice(&c.to_le_bytes());
    }
    b.extend_from_slice(&level.yaw.to_le_bytes());
    b.extend_from_slice(&level.pitch.to_le_bytes());
    b.extend_from_slice(&level.time.to_le_bytes());
    b.push(level.flying as u8);
    // Survival block (M24).
    b.extend_from_slice(&level.health.to_le_bytes());
    b.extend_from_slice(&level.hunger.to_le_bytes());
    b.extend_from_slice(&level.air.to_le_bytes());
    b.extend_from_slice(&level.saturation.to_le_bytes());
    b.extend_from_slice(&level.xp.to_le_bytes());
    b.extend_from_slice(&level.level.to_le_bytes());
    fs::write(dir.join("level.bin"), b)
}

/// Serialize one inventory slot: a present byte, then item(u16)+count(u8)+durability(u16) if present.
fn wr_slot(b: &mut Vec<u8>, slot: Option<ItemStack>) {
    match slot {
        Some(s) => {
            b.push(1);
            b.extend_from_slice(&s.item.to_le_bytes());
            b.push(s.count);
            b.extend_from_slice(&s.durability.to_le_bytes());
        }
        None => b.push(0),
    }
}

/// Read a slot written by `wr_slot`, advancing the cursor; rejects unknown/corrupt ids.
fn rd_slot(d: &[u8], o: &mut usize) -> Option<ItemStack> {
    if *o >= d.len() {
        return None;
    }
    let present = d[*o];
    *o += 1;
    if present == 0 {
        return None;
    }
    if *o + 5 > d.len() {
        return None;
    }
    let item = u16::from_le_bytes([d[*o], d[*o + 1]]);
    let count = d[*o + 2].min(crate::item::max_stack(item));
    let dur = u16::from_le_bytes([d[*o + 3], d[*o + 4]]);
    *o += 5;
    (count > 0 && crate::item::is_known(item)).then(|| {
        let mut st = ItemStack::new(item, count);
        st.durability = dur;
        st
    })
}

/// Write the player inventory + furnace containers to `save_state.bin` (VCIV v3). Sparse inventory;
/// the furnace section is length-prefixed and appended, so a v2 loader simply ignores it.
pub fn save_state(dir: &Path, inv: &Inventory, furnaces: &[FurnaceSave]) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut b = Vec::new();
    b.extend_from_slice(&INV_MAGIC.to_le_bytes());
    b.push(3); // version 3: v2 inventory + a furnace container section
    b.push(inv.selected as u8);
    b.push(inv.creative as u8);
    let filled: Vec<(usize, ItemStack)> = inv
        .slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|s| (i, s)))
        .collect();
    b.push(filled.len() as u8);
    for (i, s) in filled {
        b.push(i as u8);
        b.extend_from_slice(&s.item.to_le_bytes());
        b.push(s.count);
        b.extend_from_slice(&s.durability.to_le_bytes());
    }
    // Container section: furnaces.
    b.extend_from_slice(&(furnaces.len() as u32).to_le_bytes());
    for f in furnaces {
        b.extend_from_slice(&f.pos.x.to_le_bytes());
        b.extend_from_slice(&f.pos.y.to_le_bytes());
        b.extend_from_slice(&f.pos.z.to_le_bytes());
        b.extend_from_slice(&f.burn_remaining.to_le_bytes());
        b.extend_from_slice(&f.burn_max.to_le_bytes());
        b.extend_from_slice(&f.cook_progress.to_le_bytes());
        b.extend_from_slice(&f.cook_item.to_le_bytes());
        wr_slot(&mut b, f.input);
        wr_slot(&mut b, f.fuel);
        wr_slot(&mut b, f.output);
    }
    fs::write(dir.join("save_state.bin"), b)
}

/// Load the inventory + furnace containers, or a fresh inventory if absent/corrupt.
pub fn load_state(dir: &Path, creative_default: bool) -> (Inventory, Vec<FurnaceSave>) {
    let mut inv = Inventory::new(creative_default);
    let mut furnaces = Vec::new();
    let Ok(d) = fs::read(dir.join("save_state.bin")) else {
        return (inv, furnaces);
    };
    if d.len() < 8 || rd_u32(&d, 0) != INV_MAGIC {
        return (inv, furnaces);
    }
    let version = d[4];
    if !(1..=3).contains(&version) {
        return (inv, furnaces); // unknown version -> fresh inventory
    }
    inv.selected = (d[5] as usize).min(HOTBAR - 1);
    inv.creative = d[6] != 0;
    for s in inv.slots.iter_mut() {
        *s = None; // saved state is authoritative (replaces the default palette)
    }
    let count = d[7] as usize;
    let rec = if version >= 2 { 6 } else { 4 };
    let mut o = 8;
    for _ in 0..count {
        if o + rec > d.len() {
            break;
        }
        let slot = d[o] as usize;
        let item = u16::from_le_bytes([d[o + 1], d[o + 2]]);
        // Clamp the count to the item's max stack and reject unknown ids (corrupt/edited saves).
        let cnt = d[o + 3].min(crate::item::max_stack(item));
        if slot < SLOTS && cnt > 0 && crate::item::is_known(item) {
            let mut stack = ItemStack::new(item, cnt); // durability defaults (max for tools)
            if version >= 2 {
                stack.durability = u16::from_le_bytes([d[o + 4], d[o + 5]]);
            }
            inv.slots[slot] = Some(stack);
        }
        o += rec;
    }
    // Furnace container section (v3+).
    if version >= 3 && o + 4 <= d.len() {
        let fcount = rd_u32(&d, o) as usize;
        o += 4;
        for _ in 0..fcount {
            if o + 26 > d.len() {
                break; // pos(12) + burn/burn_max/cook(12) + cook_item(2)
            }
            let pos = IVec3::new(rd_i32(&d, o), rd_i32(&d, o + 4), rd_i32(&d, o + 8));
            let burn_remaining = rd_f32(&d, o + 12);
            let burn_max = rd_f32(&d, o + 16);
            let cook_progress = rd_f32(&d, o + 20);
            let cook_item = u16::from_le_bytes([d[o + 24], d[o + 25]]);
            o += 26;
            let input = rd_slot(&d, &mut o);
            let fuel = rd_slot(&d, &mut o);
            let output = rd_slot(&d, &mut o);
            furnaces.push(FurnaceSave {
                pos,
                input,
                fuel,
                output,
                burn_remaining,
                burn_max,
                cook_progress,
                cook_item,
            });
        }
    }
    (inv, furnaces)
}

pub fn load_chunks(dir: &Path) -> FxHashMap<IVec3, Chunk> {
    let mut map = FxHashMap::default();
    let Ok(d) = fs::read(dir.join("chunks.bin")) else {
        return map;
    };
    if d.len() < 8 || rd_u32(&d, 0) != MAGIC {
        return map;
    }
    let count = rd_u32(&d, 4);
    let mut o = 8usize;
    for _ in 0..count {
        if o + 16 > d.len() {
            break;
        }
        let pos = IVec3::new(rd_i32(&d, o), rd_i32(&d, o + 4), rd_i32(&d, o + 8));
        let clen = rd_u32(&d, o + 12) as usize;
        o += 16;
        if o + clen > d.len() {
            break;
        }
        if let Ok(raw) = lz4_flex::decompress_size_prepended(&d[o..o + clen]) {
            if raw.len() == CHUNK_VOLUME * 2 {
                let blocks: Vec<BlockId> = raw
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                let solid_count = blocks.iter().filter(|&&b| b != 0).count() as u32;
                let emitter_count =
                    blocks.iter().filter(|&&b| crate::block::light_emission(b) > 0).count() as u32;
                let light = vec![0u8; CHUNK_VOLUME];
                map.insert(pos, Chunk { blocks, light, solid_count, emitter_count });
            }
        }
        o += clen;
    }
    log::info!("Loaded {} edited chunks from save", map.len());
    map
}

pub fn save_chunks(dir: &Path, chunks: &[(IVec3, &Chunk)]) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut b = Vec::new();
    b.extend_from_slice(&MAGIC.to_le_bytes());
    b.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    let mut raw = Vec::with_capacity(CHUNK_VOLUME * 2);
    for (pos, chunk) in chunks {
        b.extend_from_slice(&pos.x.to_le_bytes());
        b.extend_from_slice(&pos.y.to_le_bytes());
        b.extend_from_slice(&pos.z.to_le_bytes());
        raw.clear();
        for &blk in &chunk.blocks {
            raw.extend_from_slice(&blk.to_le_bytes());
        }
        let comp = lz4_flex::compress_prepend_size(&raw);
        b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        b.extend_from_slice(&comp);
    }
    fs::write(dir.join("chunks.bin"), b)?;
    log::info!("Saved {} edited chunks", chunks.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("voxelcraft_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut blocks = vec![0u16; CHUNK_VOLUME];
        blocks[0] = 1;
        blocks[1234] = 5;
        blocks[CHUNK_VOLUME - 1] = 7;
        let chunk = Chunk {
            blocks: blocks.clone(),
            light: vec![0u8; CHUNK_VOLUME],
            solid_count: 3,
            emitter_count: 0,
        };
        let pos = IVec3::new(-3, 2, 7);
        save_chunks(&dir, &[(pos, &chunk)]).unwrap();

        let loaded = load_chunks(&dir);
        assert_eq!(loaded.len(), 1);
        let lc = loaded.get(&pos).unwrap();
        assert_eq!(lc.blocks, blocks);
        assert_eq!(lc.solid_count, 3);

        let level = Level {
            seed: 0xDEAD_BEEF,
            spawn: [1.5, 2.5, 3.5],
            yaw: 0.5,
            pitch: -0.2,
            time: 0.33,
            flying: true,
            health: 12.5,
            hunger: 7.0,
            air: 4.0,
            saturation: 1.5,
            xp: 3.0,
            level: 4,
        };
        save_level(&dir, &level).unwrap();
        let ll = load_level(&dir).unwrap();
        assert_eq!(ll.seed, 0xDEAD_BEEF);
        assert_eq!(ll.spawn, [1.5, 2.5, 3.5]);
        assert!((ll.time - 0.33).abs() < 1e-6);
        assert!(ll.flying);
        // M24 survival fields round-trip.
        assert!((ll.health - 12.5).abs() < 1e-6);
        assert!((ll.air - 4.0).abs() < 1e-6);
        assert_eq!(ll.level, 4);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn state_roundtrip_inventory_and_furnaces() {
        let dir = std::env::temp_dir().join(format!("voxelcraft_inv_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut inv = Inventory::new(false);
        inv.selected = 3;
        inv.slots[0] = Some(ItemStack::new(1, 64));
        inv.slots[10] = Some(ItemStack::new(5, 12));
        let mut pick = ItemStack::new(crate::item::DIAMOND_PICKAXE, 1);
        pick.durability = 1000;
        inv.slots[5] = Some(pick);

        let furnaces = vec![FurnaceSave {
            pos: IVec3::new(4, 70, -9),
            input: Some(ItemStack::new(crate::block::IRON_ORE, 3)),
            fuel: Some(ItemStack::new(crate::block::PLANKS, 5)),
            output: Some(ItemStack::new(crate::item::IRON_INGOT, 2)),
            burn_remaining: 4.5,
            burn_max: 9.0,
            cook_progress: 2.25,
            cook_item: crate::block::IRON_ORE,
        }];
        save_state(&dir, &inv, &furnaces).unwrap();

        let (loaded, lf) = load_state(&dir, true);
        assert_eq!(loaded.selected, 3);
        assert!(!loaded.creative);
        assert_eq!(loaded.slots[0].unwrap().count, 64);
        assert_eq!(loaded.slots[5].unwrap().durability, 1000); // tool durability roundtrips
        assert!(loaded.slots[1].is_none());
        // Furnace container round-trips.
        assert_eq!(lf.len(), 1);
        let f = &lf[0];
        assert_eq!(f.pos, IVec3::new(4, 70, -9));
        assert_eq!(f.input.unwrap().count, 3);
        assert_eq!(f.output.unwrap().item, crate::item::IRON_INGOT);
        assert!((f.cook_progress - 2.25).abs() < 1e-6);
        assert_eq!(f.cook_item, crate::block::IRON_ORE);

        let _ = fs::remove_dir_all(&dir);
    }
}
