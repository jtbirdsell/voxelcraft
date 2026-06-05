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

const MAGIC: u32 = 0x5643_5231; // "VCR1" — block ids only (legacy; still loads)
const MAGIC_V2: u32 = 0x5643_5232; // "VCR2" — adds a per-chunk block-state LZ4 section
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
    /// World difficulty (P6) as a u8 (0 Peaceful … 3 Hard). Appended after the survival block;
    /// absent in older saves → defaults to Normal (2).
    pub difficulty: u8,
}

/// One chest's persisted contents (P3), saved in the chest section of `save_state.bin` (v4+).
pub struct ChestSave {
    pub pos: IVec3,
    pub slots: Vec<Option<ItemStack>>,
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

/// The legacy single-world directory. Post-N4 nothing in the normal play path writes here — it is only
/// the *source* for the one-time `migrate_legacy_world()` step (and the read-only SKIPMENU fallback).
pub fn save_dir() -> PathBuf {
    PathBuf::from("saves").join("world")
}

/// Root holding one subdirectory per world (N4 multi-world). A sibling of the legacy `save_dir()` and
/// of `settings.cfg` — neither of which moves.
pub fn worlds_root() -> PathBuf {
    PathBuf::from("saves").join("worlds")
}

/// One world as shown in the world-select screen. `dir` (the slug directory) is the **sole on-disk
/// identity**; `name` is the human-facing display name (the `name.txt` sidecar, falling back to the dir
/// name); `seed` comes from `level.bin`. Nothing keys load/delete off the display name.
#[derive(Clone, Debug)]
pub struct WorldInfo {
    pub dir: PathBuf,
    pub name: String,
    pub seed: u64,
}

/// Map a display name to a filesystem-safe slug: ASCII alphanumerics and '-' are kept (case preserved),
/// every other char becomes '_', runs of '_' collapse, the result is trimmed of '_' and capped, with a
/// non-empty fallback. The slug is identity only; the verbatim name lives in `name.txt`.
pub fn slugify(name: &str) -> String {
    let mut s = String::new();
    let mut prev_us = false;
    for ch in name.chars() {
        let c = if ch.is_ascii_alphanumeric() || ch == '-' { ch } else { '_' };
        if c == '_' {
            if prev_us {
                continue;
            }
            prev_us = true;
        } else {
            prev_us = false;
        }
        s.push(c);
        if s.len() >= 32 {
            break;
        }
    }
    let s = s.trim_matches('_').to_string();
    if s.is_empty() {
        "world".to_string()
    } else {
        s
    }
}

/// The display name for a world directory: the `name.txt` sidecar (trimmed), else the directory's own
/// name. `name.txt` is display-only and may desync from the slug — the slug stays the identity.
fn world_name(dir: &Path) -> String {
    if let Ok(txt) = fs::read_to_string(dir.join("name.txt")) {
        let t = txt.trim();
        if !t.is_empty() {
            return t.to_string();
        }
    }
    dir.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "world".to_string())
}

/// List the worlds under `root`, most-recently-played first (by `level.bin` mtime, with a stable
/// slug-name tiebreak so equal mtimes — 2 s on some Windows filesystems — sort deterministically). Only
/// directories with a readable `level.bin` count; a dir missing it (a create that crashed before
/// writing the header) is silently skipped, never listed.
pub fn list_worlds_in(root: &Path) -> Vec<WorldInfo> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<(WorldInfo, std::time::SystemTime)> = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let Some(level) = load_level(&dir) else {
            continue;
        };
        let mtime = fs::metadata(dir.join("level.bin"))
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        let name = world_name(&dir);
        out.push((WorldInfo { dir, name, seed: level.seed }, mtime));
    }
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.dir.cmp(&b.0.dir)));
    out.into_iter().map(|(w, _)| w).collect()
}

/// The user-facing world list (`saves/worlds/`). Migration runs once at startup, not here.
pub fn list_worlds() -> Vec<WorldInfo> {
    list_worlds_in(&worlds_root())
}

/// Create a fresh world under `root`, seeding its `level.bin` then its `name.txt`. The directory is a
/// unique slug of `name` (collisions append `-2`, `-3`, …). Creation is **exclusive** (`fs::create_dir`,
/// never `create_dir_all`) so it can never write into an existing world's directory — which also makes
/// it correct on case-insensitive NTFS, where "World" and an existing "world" are the same path.
/// `level.bin` is written first so a crash can't leave a named-but-unlistable dir.
pub fn create_world_in(root: &Path, name: &str, seed: u64) -> std::io::Result<WorldInfo> {
    fs::create_dir_all(root)?;
    let base = slugify(name);
    let mut dir = None;
    for n in 1..=9999u32 {
        let cand = if n == 1 {
            root.join(&base)
        } else {
            root.join(format!("{base}-{n}"))
        };
        match fs::create_dir(&cand) {
            Ok(()) => {
                dir = Some(cand);
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    let dir = dir.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::AlreadyExists, "no free world directory slug")
    })?;
    // level.bin FIRST (this commits the dir as a listable world), then the display-name sidecar.
    let level = Level {
        seed,
        spawn: [8.0, 96.0, 24.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        pitch: -0.30,
        time: 0.34,
        flying: true,
        health: 20.0,
        hunger: 20.0,
        air: 11.0,
        saturation: 5.0,
        xp: 0.0,
        level: 0,
        difficulty: 2, // Normal
    };
    save_level(&dir, &level)?;
    let display = name.trim();
    let _ = fs::write(dir.join("name.txt"), display);
    Ok(WorldInfo { dir, name: display.to_string(), seed })
}

/// Create a world in the standard `saves/worlds/` root.
pub fn create_world(name: &str, seed: u64) -> std::io::Result<WorldInfo> {
    create_world_in(&worlds_root(), name, seed)
}

/// Delete a world directory. Refuses anything that is not a **direct child** of `root`, so a malformed
/// `dir` can never reach `remove_dir_all` on the wrong path. Component-based (not literal-PathBuf)
/// checks, so separators / case don't produce false negatives on Windows.
pub fn delete_world_in(root: &Path, dir: &Path) -> std::io::Result<()> {
    let rel = dir.strip_prefix(root).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "world dir is not under the worlds root")
    })?;
    let mut comps = rel.components();
    match (comps.next(), comps.next()) {
        (Some(std::path::Component::Normal(_)), None) => fs::remove_dir_all(dir),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "world dir must be a direct child of the worlds root",
        )),
    }
}

/// Delete a world from the standard `saves/worlds/` root.
pub fn delete_world(dir: &Path) -> std::io::Result<()> {
    delete_world_in(&worlds_root(), dir)
}

/// One-time migration of the legacy single-world `saves/world/` into `saves/worlds/world/` so an
/// existing save appears in the multi-world list. Idempotent (a no-op once the target exists) and
/// best-effort: a failed rename leaves the legacy world exactly where it is and never crashes the
/// caller. **Call once at startup**, never lazily from a menu / screenshot path.
pub fn migrate_legacy_world() {
    migrate_legacy_world_in(&save_dir(), &worlds_root());
}

fn migrate_legacy_world_in(legacy: &Path, root: &Path) {
    // Only a *real* legacy world (one with a level.bin) migrates — never an empty leftover dir.
    if !legacy.join("level.bin").is_file() {
        return;
    }
    let target = root.join("world");
    if target.exists() {
        // Already migrated (or a name clash). Never auto-delete a world; just note the stale legacy copy.
        log::warn!(
            "legacy world {} left in place: {} already exists",
            legacy.display(),
            target.display()
        );
        return;
    }
    if let Err(e) = fs::create_dir_all(root) {
        log::error!("world migration: cannot create {}: {e}", root.display());
        return;
    }
    // Same-volume rename is atomic and avoids double disk use; on failure (e.g. a briefly-locked file
    // on Windows) leave the legacy world untouched rather than risk data loss or a menu crash.
    match fs::rename(legacy, &target) {
        Ok(()) => {
            let _ = fs::write(target.join("name.txt"), "World");
            log::info!("migrated legacy world {} -> {}", legacy.display(), target.display());
        }
        Err(e) => log::error!(
            "world migration rename failed ({e}); legacy world left at {}",
            legacy.display()
        ),
    }
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
        // Difficulty byte at offset 57 (after the 24-byte survival block); default Normal (2).
        difficulty: if d.len() >= 58 { d[57] } else { 2 },
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
    b.push(level.difficulty); // P6: difficulty byte, after the survival block
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

/// Write the player inventory + furnace + chest containers to `save_state.bin` (VCIV v4). Sparse
/// inventory; each container section is length-prefixed and appended, so an older loader simply
/// stops after the sections it knows.
pub fn save_state(
    dir: &Path,
    inv: &Inventory,
    furnaces: &[FurnaceSave],
    chests: &[ChestSave],
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut b = Vec::new();
    b.extend_from_slice(&INV_MAGIC.to_le_bytes());
    b.push(4); // version 4: v3 (inventory + furnaces) + a chest container section
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
    // Chest container section (v4+): pos + slot count + each slot.
    b.extend_from_slice(&(chests.len() as u32).to_le_bytes());
    for c in chests {
        b.extend_from_slice(&c.pos.x.to_le_bytes());
        b.extend_from_slice(&c.pos.y.to_le_bytes());
        b.extend_from_slice(&c.pos.z.to_le_bytes());
        b.push(c.slots.len() as u8);
        for &s in &c.slots {
            wr_slot(&mut b, s);
        }
    }
    fs::write(dir.join("save_state.bin"), b)
}

/// Load the inventory + furnace + chest containers, or a fresh inventory if absent/corrupt.
pub fn load_state(
    dir: &Path,
    creative_default: bool,
) -> (Inventory, Vec<FurnaceSave>, Vec<ChestSave>) {
    let mut inv = Inventory::new(creative_default);
    let mut furnaces = Vec::new();
    let mut chests = Vec::new();
    let Ok(d) = fs::read(dir.join("save_state.bin")) else {
        return (inv, furnaces, chests);
    };
    if d.len() < 8 || rd_u32(&d, 0) != INV_MAGIC {
        return (inv, furnaces, chests);
    }
    let version = d[4];
    if !(1..=4).contains(&version) {
        return (inv, furnaces, chests); // unknown version -> fresh inventory
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
        // Chest container section (v4+), immediately after the furnaces.
        if version >= 4 && o + 4 <= d.len() {
            let ccount = rd_u32(&d, o) as usize;
            o += 4;
            for _ in 0..ccount {
                if o + 13 > d.len() {
                    break; // pos(12) + nslots(1)
                }
                let pos = IVec3::new(rd_i32(&d, o), rd_i32(&d, o + 4), rd_i32(&d, o + 8));
                let nslots = d[o + 12] as usize;
                o += 13;
                let mut slots = Vec::with_capacity(nslots);
                for _ in 0..nslots {
                    slots.push(rd_slot(&d, &mut o));
                }
                chests.push(ChestSave { pos, slots });
            }
        }
    }
    (inv, furnaces, chests)
}

/// Migrate legacy block ids to the current id + block-state scheme.
/// P2: the old fixed-orientation stair ids (42 = STONE_STAIRS_E, 43 = _S, 44 = _W) become
/// STONE_STAIRS (40) with the facing carried in the state byte (1/2/3); the base STONE_STAIRS(40)
/// was already facing 0 (state 0).
/// **LEGACY (VCR1) ONLY — the caller MUST gate this on `!has_states`.** Ids 42/43/44 were later
/// reused by P9/P10 (WOODEN_DOOR/WOODEN_TRAPDOOR/WOODEN_FENCE), so running this on a VCR2 file (which
/// post-dates the reuse) would silently clobber every placed door/trapdoor/fence into a stone stair.
fn migrate_legacy_blocks(blocks: &mut [BlockId], states: &mut [u8]) {
    for (b, s) in blocks.iter_mut().zip(states.iter_mut()) {
        match *b {
            42 => {
                *b = crate::block::STONE_STAIRS;
                *s = 1;
            }
            43 => {
                *b = crate::block::STONE_STAIRS;
                *s = 2;
            }
            44 => {
                *b = crate::block::STONE_STAIRS;
                *s = 3;
            }
            _ => {}
        }
    }
}

pub fn load_chunks(dir: &Path) -> FxHashMap<IVec3, Chunk> {
    let mut map = FxHashMap::default();
    let Ok(d) = fs::read(dir.join("chunks.bin")) else {
        return map;
    };
    if d.len() < 8 {
        return map;
    }
    let magic = rd_u32(&d, 0);
    let has_states = magic == MAGIC_V2;
    if magic != MAGIC && !has_states {
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
        let blocks_raw = lz4_flex::decompress_size_prepended(&d[o..o + clen]);
        o += clen;
        // Block-state section (VCR2 only): slen(4) + lz4(states). Missing/legacy => all-default 0.
        let mut states = vec![0u8; CHUNK_VOLUME];
        if has_states {
            if o + 4 > d.len() {
                break;
            }
            let slen = rd_u32(&d, o) as usize;
            o += 4;
            if o + slen > d.len() {
                break;
            }
            if let Ok(s) = lz4_flex::decompress_size_prepended(&d[o..o + slen]) {
                if s.len() == CHUNK_VOLUME {
                    states = s;
                }
            }
            o += slen;
        }
        if let Ok(raw) = blocks_raw {
            if raw.len() == CHUNK_VOLUME * 2 {
                let mut blocks: Vec<BlockId> = raw
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                // Migrate ONLY legacy VCR1 files: a VCR2 magic proves the save post-dates P9/P10's
                // reuse of ids 42/43/44, so those are genuine door/trapdoor/fence and must be left as-is.
                if !has_states {
                    migrate_legacy_blocks(&mut blocks, &mut states);
                }
                let solid_count = blocks.iter().filter(|&&b| b != 0).count() as u32;
                let emitter_count =
                    blocks.iter().filter(|&&b| crate::block::light_emission(b) > 0).count() as u32;
                let light = vec![0u8; CHUNK_VOLUME];
                map.insert(pos, Chunk { blocks, states, light, solid_count, emitter_count });
            }
        }
    }
    log::info!("Loaded {} edited chunks from save", map.len());
    map
}

pub fn save_chunks(dir: &Path, chunks: &[(IVec3, &Chunk)]) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut b = Vec::new();
    b.extend_from_slice(&MAGIC_V2.to_le_bytes());
    b.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    let mut raw = Vec::with_capacity(CHUNK_VOLUME * 2);
    for (pos, chunk) in chunks {
        b.extend_from_slice(&pos.x.to_le_bytes());
        b.extend_from_slice(&pos.y.to_le_bytes());
        b.extend_from_slice(&pos.z.to_le_bytes());
        // Block ids (LZ4).
        raw.clear();
        for &blk in &chunk.blocks {
            raw.extend_from_slice(&blk.to_le_bytes());
        }
        let comp = lz4_flex::compress_prepend_size(&raw);
        b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        b.extend_from_slice(&comp);
        // Block states (LZ4) — VCR2. One byte per cell, mostly 0, so this is tiny on disk.
        let scomp = lz4_flex::compress_prepend_size(&chunk.states);
        b.extend_from_slice(&(scomp.len() as u32).to_le_bytes());
        b.extend_from_slice(&scomp);
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
            states: vec![0u8; CHUNK_VOLUME],
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
        assert_eq!(lc.states, vec![0u8; CHUNK_VOLUME]); // default states round-trip

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
            difficulty: 3, // Hard
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
        assert_eq!(ll.difficulty, 3); // P6 difficulty round-trips

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn block_states_roundtrip() {
        // Non-zero block-state bytes (e.g. stair facing/half) must survive save → load (VCR2).
        let dir = std::env::temp_dir().join(format!("voxelcraft_states_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut chunk = Chunk::filled(crate::block::AIR);
        chunk.set_state(1, 2, 3, crate::block::STONE_STAIRS, 0b101);
        chunk.set_state(5, 6, 7, crate::block::WOOD, 2);
        let pos = IVec3::new(2, 1, -4);
        save_chunks(&dir, &[(pos, &chunk)]).unwrap();

        let loaded = load_chunks(&dir);
        let lc = loaded.get(&pos).unwrap();
        assert_eq!(lc.state(1, 2, 3), 0b101);
        assert_eq!(lc.state(5, 6, 7), 2);
        assert_eq!(lc.get(1, 2, 3), crate::block::STONE_STAIRS);
        assert_eq!(lc.states.iter().filter(|&&s| s != 0).count(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn vcr2_reused_ids_not_migrated() {
        // Ids 42/43/44 are WOODEN_DOOR/TRAPDOOR/FENCE since P9/P10 (they were *also* the retired stair
        // ids pre-P2). The legacy-stair migration must run ONLY on VCR1 files — a VCR2 save (every save
        // current code writes) must round-trip these blocks + their state bytes UNCHANGED, never
        // clobbered into STONE_STAIRS.
        let dir = std::env::temp_dir().join(format!("voxelcraft_vcr2reuse_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut chunk = Chunk::filled(crate::block::AIR);
        let door = crate::block::door_state(1, true, 0, crate::block::DOOR_LOWER);
        chunk.set_state(0, 0, 0, crate::block::WOODEN_DOOR, door);
        chunk.set_state(1, 0, 0, crate::block::WOODEN_TRAPDOOR, crate::block::trapdoor_state(2, 1, true));
        chunk.set_state(2, 0, 0, crate::block::WOODEN_FENCE, 0);
        let pos = IVec3::new(7, -2, 3);
        save_chunks(&dir, &[(pos, &chunk)]).unwrap();

        let loaded = load_chunks(&dir);
        let lc = loaded.get(&pos).unwrap();
        assert_eq!(lc.get(0, 0, 0), crate::block::WOODEN_DOOR, "door id survives VCR2 reload");
        assert_eq!(lc.state(0, 0, 0), door, "door state byte not clobbered by migration");
        assert_eq!(lc.get(1, 0, 0), crate::block::WOODEN_TRAPDOOR);
        assert_eq!(lc.get(2, 0, 0), crate::block::WOODEN_FENCE);
        // None became a stone stair.
        assert!(!lc.blocks.iter().any(|&b| b == crate::block::STONE_STAIRS));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_vcr1_loads_with_default_states() {
        // A pre-VCR2 chunks.bin (block ids only, "VCR1" magic) must still load, with all-default
        // states — so existing worlds keep their edits.
        let dir = std::env::temp_dir().join(format!("voxelcraft_vcr1_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mut blocks = vec![0u16; CHUNK_VOLUME];
        blocks[100] = crate::block::COBBLESTONE;
        let pos = IVec3::new(0, 1, 0);
        // Hand-write a legacy VCR1 file: MAGIC + count + [pos + clen + lz4(blocks)] (no state section).
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&pos.x.to_le_bytes());
        b.extend_from_slice(&pos.y.to_le_bytes());
        b.extend_from_slice(&pos.z.to_le_bytes());
        let mut raw = Vec::new();
        for &blk in &blocks {
            raw.extend_from_slice(&blk.to_le_bytes());
        }
        let comp = lz4_flex::compress_prepend_size(&raw);
        b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        b.extend_from_slice(&comp);
        fs::write(dir.join("chunks.bin"), b).unwrap();

        let loaded = load_chunks(&dir);
        let lc = loaded.get(&pos).unwrap();
        assert_eq!(lc.get(4, 0, 3), crate::block::COBBLESTONE); // index 100 = 4 + 32*(3 + 32*0)
        assert_eq!(lc.states, vec![0u8; CHUNK_VOLUME]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_stair_ids_migrate_to_state() {
        // Old fixed-orientation stair ids (42/43/44) must become STONE_STAIRS + facing in the state
        // byte, so worlds built before P2 keep their oriented stairs.
        let dir = std::env::temp_dir().join(format!("voxelcraft_stairmig_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mut blocks = vec![0u16; CHUNK_VOLUME];
        blocks[0] = 42; // old STONE_STAIRS_E -> facing 1
        blocks[1] = 43; // old STONE_STAIRS_S -> facing 2
        blocks[2] = 44; // old STONE_STAIRS_W -> facing 3
        blocks[3] = crate::block::STONE_STAIRS; // 40, already facing 0
        let pos = IVec3::new(0, 0, 0);
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes()); // legacy VCR1 (block ids only)
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&pos.x.to_le_bytes());
        b.extend_from_slice(&pos.y.to_le_bytes());
        b.extend_from_slice(&pos.z.to_le_bytes());
        let mut raw = Vec::new();
        for &blk in &blocks {
            raw.extend_from_slice(&blk.to_le_bytes());
        }
        let comp = lz4_flex::compress_prepend_size(&raw);
        b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        b.extend_from_slice(&comp);
        fs::write(dir.join("chunks.bin"), b).unwrap();

        let loaded = load_chunks(&dir);
        let lc = loaded.get(&pos).unwrap();
        for x in 0..4 {
            assert_eq!(lc.get(x, 0, 0), crate::block::STONE_STAIRS);
        }
        assert_eq!(lc.state(0, 0, 0), 1);
        assert_eq!(lc.state(1, 0, 0), 2);
        assert_eq!(lc.state(2, 0, 0), 3);
        assert_eq!(lc.state(3, 0, 0), 0);
        // No retired stair ids survive migration.
        assert!(lc.blocks.iter().all(|&b| b <= crate::block::MAX_BLOCK));

        let _ = fs::remove_dir_all(&dir);
    }

    fn stub_level(seed: u64) -> Level {
        Level {
            seed,
            spawn: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            time: 0.0,
            flying: false,
            health: 20.0,
            hunger: 20.0,
            air: 11.0,
            saturation: 5.0,
            xp: 0.0,
            level: 0,
            difficulty: 2,
        }
    }

    #[test]
    fn slugify_keeps_safe_chars_and_collapses() {
        assert_eq!(slugify("My World!"), "My_World");
        assert_eq!(slugify("  spaces  "), "spaces");
        assert_eq!(slugify("a///b"), "a_b"); // runs of unsafe chars collapse to one '_'
        assert_eq!(slugify("Home-2"), "Home-2"); // '-' is kept
        assert_eq!(slugify(""), "world"); // empty → fallback
        assert_eq!(slugify("***"), "world"); // all-unsafe trims to empty → fallback
        assert!(slugify(&"x".repeat(100)).len() <= 32); // capped
    }

    #[test]
    fn create_then_list_reports_name_and_seed() {
        let root =
            std::env::temp_dir().join(format!("voxelcraft_wlist_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let w = create_world_in(&root, "Caves", 0x00AB_CDEF).unwrap();
        assert!(w.dir.join("level.bin").is_file());
        assert!(w.dir.join("name.txt").is_file());
        let list = list_worlds_in(&root);
        let found = list.iter().find(|x| x.dir == w.dir).expect("world listed");
        assert_eq!(found.name, "Caves");
        assert_eq!(found.seed, 0x00AB_CDEF);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn create_world_suffixes_on_exact_collision() {
        let root =
            std::env::temp_dir().join(format!("voxelcraft_wcoll_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let a = create_world_in(&root, "Home", 1).unwrap();
        let b = create_world_in(&root, "Home", 2).unwrap();
        assert_eq!(a.dir.file_name().unwrap(), "Home");
        assert_eq!(b.dir.file_name().unwrap(), "Home-2");
        // Neither create clobbered the other's header.
        assert_eq!(load_level(&a.dir).unwrap().seed, 1);
        assert_eq!(load_level(&b.dir).unwrap().seed, 2);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn create_world_never_overwrites_an_existing_world() {
        // On NTFS "World" and "world" are the same path; on a case-sensitive FS they differ. Either way,
        // the exclusive create must never write a stub level.bin over an already-existing world.
        let root =
            std::env::temp_dir().join(format!("voxelcraft_wcase_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let a = create_world_in(&root, "world", 111).unwrap();
        let b = create_world_in(&root, "World", 222).unwrap();
        assert_ne!(a.dir, b.dir);
        assert_eq!(load_level(&a.dir).unwrap().seed, 111); // untouched
        assert_eq!(load_level(&b.dir).unwrap().seed, 222);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_world_only_inside_root() {
        let root =
            std::env::temp_dir().join(format!("voxelcraft_wdel_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let w = create_world_in(&root, "Doomed", 5).unwrap();
        assert!(w.dir.is_dir());
        // Refuse a sibling outside the root.
        let outside = root.parent().unwrap().join(format!("voxelcraft_wdel_evil_{}", std::process::id()));
        fs::create_dir_all(&outside).unwrap();
        assert!(delete_world_in(&root, &outside).is_err());
        assert!(outside.is_dir(), "an outside dir must be left untouched");
        // Refuse the root itself (zero components beyond root).
        assert!(delete_world_in(&root, &root).is_err());
        // The real world deletes.
        delete_world_in(&root, &w.dir).unwrap();
        assert!(!w.dir.exists());
        let _ = fs::remove_dir_all(&outside);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn legacy_world_migrates_once_and_is_idempotent() {
        let base =
            std::env::temp_dir().join(format!("voxelcraft_wmig_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let legacy = base.join("world");
        let root = base.join("worlds");
        fs::create_dir_all(&legacy).unwrap();
        save_level(&legacy, &stub_level(0x1234)).unwrap();
        fs::write(legacy.join("chunks.bin"), b"chunkdata").unwrap();

        migrate_legacy_world_in(&legacy, &root);
        let target = root.join("world");
        assert!(!legacy.exists(), "legacy dir is renamed away");
        assert!(target.join("level.bin").is_file());
        assert!(target.join("chunks.bin").is_file(), "edited chunks moved with the world");
        assert_eq!(load_level(&target).unwrap().seed, 0x1234);

        // A stray legacy reappearing (e.g. a throwaway run) must NOT clobber the migrated world, and the
        // stray is left in place (never auto-deleted).
        fs::create_dir_all(&legacy).unwrap();
        save_level(&legacy, &stub_level(0x9999)).unwrap();
        migrate_legacy_world_in(&legacy, &root);
        assert_eq!(load_level(&target).unwrap().seed, 0x1234, "migrated world unchanged");
        assert!(legacy.exists(), "stray legacy left in place");
        let _ = fs::remove_dir_all(&base);
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
        let mut chest_slots = vec![None; crate::container::CHEST_SLOTS];
        chest_slots[0] = Some(ItemStack::new(crate::block::DIAMOND_ORE, 7));
        chest_slots[26] = Some(ItemStack::new(crate::item::GOLD_INGOT, 12));
        let chests = vec![ChestSave {
            pos: IVec3::new(-8, 64, 3),
            slots: chest_slots,
        }];
        save_state(&dir, &inv, &furnaces, &chests).unwrap();

        let (loaded, lf, lc) = load_state(&dir, true);
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
        // Chest container round-trips.
        assert_eq!(lc.len(), 1);
        let c = &lc[0];
        assert_eq!(c.pos, IVec3::new(-8, 64, 3));
        assert_eq!(c.slots.len(), crate::container::CHEST_SLOTS);
        assert_eq!(c.slots[0].unwrap().item, crate::block::DIAMOND_ORE);
        assert_eq!(c.slots[0].unwrap().count, 7);
        assert_eq!(c.slots[26].unwrap().item, crate::item::GOLD_INGOT);

        let _ = fs::remove_dir_all(&dir);
    }
}
