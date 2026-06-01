//! Streaming manager: keeps chunks generated and meshed around the camera within a render
//! distance using a worker pool, frustum-culls them for drawing, and unloads distant ones.
//!
//! Threading model: workers generate chunks and build meshes from immutable snapshots; the main
//! thread owns the world map (the only mutator) and performs GPU uploads (budgeted per frame).

use std::collections::VecDeque;
use std::sync::Arc;

use glam::{IVec3, Vec3};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::block::{self, BlockId};
use crate::entity::Entities;
use crate::frustum::Frustum;
use crate::gpu::Gpu;
use crate::item::{self, ItemStack};
use crate::mesher::{self, MeshData};
use crate::renderer::{ChunkRenderer, GpuMesh};
use crate::persistence::{self, Level};
use crate::smelting;
use crate::voxel_volume::VoxelVolume;
use crate::worker::{Job, JobResult, WorkerPool};
use crate::worldgen::Worldgen;
use crate::world::{self, Chunk, World, CHUNK_SIZE_I, WORLD_HEIGHT_CHUNKS};

/// Fluid simulation cadence and per-tick work cap, plus how far each fluid spreads horizontally
/// from where it last fell (water reaches further than the more sluggish lava).
const FLUID_INTERVAL: f32 = 0.18;
const FLUID_BUDGET: usize = 96;
const WATER_SPREAD: u8 = 7;
const LAVA_SPREAD: u8 = 3;

/// Random-block-tick tuning (U9): how often a batch runs, how many cells each near chunk samples, and
/// the chunk radius around the player that ticks. Drives amethyst/dripstone/cave-vine growth.
const RANDOM_TICK_INTERVAL: f32 = 0.25;
const RANDOM_TICKS_PER_CHUNK: u32 = 16;
const RANDOM_TICK_RADIUS: i32 = 4;

/// Whether a block participates in U9 random-tick growth.
fn random_tickable(id: BlockId) -> bool {
    matches!(
        id,
        block::BUDDING_AMETHYST
            | block::SMALL_AMETHYST_BUD
            | block::MEDIUM_AMETHYST_BUD
            | block::LARGE_AMETHYST_BUD
            | block::CAVE_VINE
            | block::CAVE_VINE_BERRIES
            | block::POINTED_DRIPSTONE
    )
}

/// Pure growth decision for a non-budding random-tickable block (U9), testable without a world.
/// Returns (relative offset of the edit, new block id) or None. `below`/`above` are neighbor ids;
/// `vine_len` is the cave-vine chain length above this cell (a length cap). Budding amethyst sprouts
/// on a random face and is handled separately (it needs a chosen direction).
fn grow_decision(
    id: BlockId,
    rnd: u32,
    below: BlockId,
    above: BlockId,
    vine_len: i32,
) -> Option<(IVec3, BlockId)> {
    match id {
        block::SMALL_AMETHYST_BUD if rnd % 100 < 16 => Some((IVec3::ZERO, block::MEDIUM_AMETHYST_BUD)),
        block::MEDIUM_AMETHYST_BUD if rnd % 100 < 16 => Some((IVec3::ZERO, block::LARGE_AMETHYST_BUD)),
        block::LARGE_AMETHYST_BUD if rnd % 100 < 16 => Some((IVec3::ZERO, block::AMETHYST_CLUSTER)),
        block::CAVE_VINE | block::CAVE_VINE_BERRIES
            if rnd % 100 < 12 && below == block::AIR && vine_len < 14 =>
        {
            // Grow a new tip below; occasionally it bears (glowing) berries.
            let tip = if (rnd >> 9) % 4 == 0 {
                block::CAVE_VINE_BERRIES
            } else {
                block::CAVE_VINE
            };
            Some((IVec3::NEG_Y, tip))
        }
        block::POINTED_DRIPSTONE
            if rnd % 100 < 5
                && matches!(above, block::POINTED_DRIPSTONE | block::DRIPSTONE_BLOCK)
                && below == block::AIR =>
        {
            // Hanging stalactite slowly extends downward.
            Some((IVec3::NEG_Y, block::POINTED_DRIPSTONE))
        }
        _ => None,
    }
}

/// U10 sculk-spread tuning: conversions per tick, the catalyst detection radius from a mob death.
const SCULK_BUDGET: u32 = 24;
const SCULK_CATALYST_RANGE: i32 = 8;
/// How long a vibration lives so consumers (sensors / the Warden) can react over a short window.
const VIBRATION_TTL: f32 = 0.6;

/// U11 shrieker → Warden tuning. A shrieker only summons in the deep dark; it raises a warning level on
/// a cooldown (so it doesn't tick every frame) and at the summon level spawns one Warden + pulses
/// Darkness onto the player.
const WARDEN_DEEP_Y: f32 = 36.0; // the player must be below this to wake a shrieker
const SHRIEK_COOLDOWN: f32 = 1.5; // min seconds between warning increments
const WARDEN_SUMMON_LEVEL: u8 = 4; // warnings needed to spawn a Warden (MC parity)
const DARKNESS_PULSE: f32 = 8.0; // seconds of Darkness a single shriek applies to the player

/// One in-flight sculk-spread charge (U10): a budget of conversions creeping out from a catalyst bloom.
struct SculkCharge {
    pos: IVec3,
    amount: u16,
}

/// A vibration (game event) the sculk family + Warden listen for (U10): block edits, mob deaths,
/// footsteps. `freq` is the MC frequency class; `ttl` ages it out.
#[derive(Clone, Copy)]
pub struct Vibration {
    pub pos: Vec3,
    pub freq: u8,
    ttl: f32,
}

/// The position of the most-recent live vibration in `vibrations` within `range` of `from` (or None).
/// A free function so both `Game::nearest_vibration` and the entity-tick closure (which can't borrow
/// `self` while `self.entities` is borrowed mut) share one search.
fn nearest_vibration_in(vibrations: &[Vibration], from: Vec3, range: f32) -> Option<Vec3> {
    vibrations
        .iter()
        .rev()
        .find(|v| (v.pos - from).length() <= range)
        .map(|v| v.pos)
}

/// Whether sculk spread may convert this block into sculk (exposed solid rock/ground, never bedrock,
/// reinforced deepslate, or the sculk family itself).
fn sculk_convertible(b: BlockId) -> bool {
    block::is_opaque(b)
        && block::is_cube(b)
        && !matches!(
            b,
            block::BEDROCK
                | block::REINFORCED_DEEPSLATE
                | block::SCULK
                | block::SCULK_SENSOR
                | block::SCULK_SHRIEKER
                | block::SCULK_CATALYST
        )
}

/// Natural-spawn tuning (P16: light + biome-gated, per-category caps, packs).
const SPAWN_INTERVAL: f32 = 2.0; // seconds between spawn attempts
const HOSTILE_CAP: usize = 12; // max live hostile mobs (continuous night spawning is MC-faithful)
const PASSIVE_CAP: usize = 8; // max live passive animals
const PASSIVE_MIN: f32 = 8.0; // animals may spawn close (8..24 ring)
const PASSIVE_MAX: f32 = 24.0;
const HOSTILE_MIN: f32 = 16.0; // a no-spawn bubble so hostiles don't appear point-blank (MC ~24)
const HOSTILE_MAX: f32 = 32.0; // ...but inside DESPAWN_RADIUS(48) so they don't instantly vanish
const PACK_SIZE: usize = 4; // animals spawn in clusters of this size
const PACK_RADIUS: i32 = 3; // pack scatter radius (blocks)

/// Binary sky-light at `wp` (0 or 15): 15 if no skylight-blocking block sits anywhere in the column
/// above, up to `top`. Mirrors the mesher's open-to-sky model (light.rs) but reads live via `block_at`.
/// Pure (closure-parameterized) so it is unit-testable without a GPU-backed Game.
fn sky_light(wp: IVec3, top: i32, block_at: &impl Fn(IVec3) -> crate::block::BlockId) -> u8 {
    let mut y = wp.y + 1;
    while y < top {
        if crate::block::blocks_skylight(block_at(IVec3::new(wp.x, y, wp.z))) {
            return 0;
        }
        y += 1;
    }
    15
}

/// Block-light at `wp` (0..15): a bounded BFS from emitters within radius 15, stepping −1 per cell and
/// stopping at `is_volume_solid` occluders — the same model the mesher's flood uses (light.rs), but
/// wp-centered (so it is cleaner than the mesher's chunk-local approximation near chunk seams). Pure.
fn block_light(wp: IVec3, block_at: &impl Fn(IVec3) -> crate::block::BlockId) -> u8 {
    use crate::block;
    const R: i32 = 15;
    let mut sources: Vec<(IVec3, u8)> = Vec::new();
    for dy in -R..=R {
        for dz in -R..=R {
            for dx in -R..=R {
                let p = IVec3::new(wp.x + dx, wp.y + dy, wp.z + dz);
                let e = block::light_emission(block_at(p));
                if e > 0 {
                    sources.push((p, e));
                }
            }
        }
    }
    if sources.is_empty() {
        return 0;
    }
    const N: i32 = 2 * R + 1;
    let idx = |p: IVec3| (((p.x - wp.x + R)) + N * ((p.z - wp.z + R) + N * (p.y - wp.y + R))) as usize;
    let mut level = vec![0u8; (N * N * N) as usize];
    let mut q = std::collections::VecDeque::new();
    for (p, e) in sources {
        let i = idx(p);
        if e > level[i] {
            level[i] = e;
            q.push_back(p);
        }
    }
    while let Some(p) = q.pop_front() {
        let l = level[idx(p)];
        if l <= 1 {
            continue;
        }
        for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
            let np = IVec3::new(p.x + dx, p.y + dy, p.z + dz);
            if (np.x - wp.x).abs() > R || (np.y - wp.y).abs() > R || (np.z - wp.z).abs() > R {
                continue;
            }
            if block::is_volume_solid(block_at(np)) {
                continue;
            }
            let (nl, ni) = (l - 1, idx(np));
            if nl > level[ni] {
                level[ni] = nl;
                q.push_back(np);
            }
        }
    }
    level[idx(wp)]
}

/// Creeper blast damage at distance `d` from the center: max at the center, linear falloff to 0 at
/// `radius * 1.6`.
fn explosion_damage(d: f32, radius: f32) -> f32 {
    const MAX: f32 = 12.0;
    let reach = radius * 1.6;
    if d >= reach {
        0.0
    } else {
        MAX * (1.0 - d / reach)
    }
}

/// One furnace's contents + burn state (M21). Lives in `Game::furnaces`, keyed by block position;
/// the `step_furnaces` tick consumes fuel to smelt `input` into `output`. Persistence: deferred to
/// M24 (save consolidation) — furnace contents are in-memory only for now.
#[derive(Default, Clone)]
pub struct FurnaceState {
    pub input: Option<ItemStack>,
    pub fuel: Option<ItemStack>,
    pub output: Option<ItemStack>,
    /// Seconds of fuel still burning, and the full burn time of that fuel unit (for the flame gauge).
    pub burn_remaining: f32,
    pub burn_max: f32,
    /// Seconds cooked toward `SMELT_TIME` for the current input.
    pub cook_progress: f32,
    /// The input item id the accumulated `cook_progress` belongs to (so swapping the input can't
    /// finish a different recipe instantly). 0 = nothing being cooked.
    cook_item: item::ItemId,
}

impl FurnaceState {
    /// Whatever the furnace currently holds, for spilling when the block is broken.
    fn contents(&self) -> impl Iterator<Item = ItemStack> {
        [self.input, self.fuel, self.output].into_iter().flatten()
    }

    /// True if the furnace holds nothing and isn't burning (safe to drop on chunk unload).
    fn is_empty(&self) -> bool {
        self.input.is_none()
            && self.fuel.is_none()
            && self.output.is_none()
            && self.burn_remaining <= 0.0
    }
}

/// Advance one furnace by `dt`: keep any lit fuel burning, light fresh fuel when there's something
/// to smelt and room for the result, cook the input, and emit one output item per `SMELT_TIME`.
fn step_furnace(f: &mut FurnaceState, dt: f32) {
    if f.burn_remaining > 0.0 {
        f.burn_remaining = (f.burn_remaining - dt).max(0.0);
    }

    // Swapping the input to a different item discards any progress cooked for the old one — you
    // can't bank ~6s on iron ore, drop in cobblestone, and have stone pop out instantly.
    let input_item = f.input.map(|s| s.item).unwrap_or(0);
    if input_item != f.cook_item {
        f.cook_progress = 0.0;
        f.cook_item = input_item;
    }

    // What (if anything) the current input smelts to, and whether the output slot can take it.
    let target = f.input.and_then(|s| smelting::smelt_output(s.item));
    let can_output = match (target, f.output) {
        (Some(out), None) => Some(out),
        (Some(out), Some(o)) if o.item == out && o.count < item::max_stack(out) => Some(out),
        _ => None,
    };

    // Light a fresh unit of fuel if we want to smelt and the fire has gone out.
    if can_output.is_some() && f.burn_remaining <= 0.0 {
        if let Some(fuel) = f.fuel.as_mut() {
            if let Some(burn) = smelting::fuel_value(fuel.item) {
                fuel.count -= 1;
                f.burn_remaining = burn;
                f.burn_max = burn;
                if fuel.count == 0 {
                    f.fuel = None;
                }
            }
        }
    }

    if f.burn_remaining > 0.0 && can_output.is_some() {
        f.cook_progress += dt;
        if f.cook_progress >= smelting::SMELT_TIME {
            f.cook_progress -= smelting::SMELT_TIME;
            let out = can_output.unwrap();
            if let Some(inp) = f.input.as_mut() {
                inp.count -= 1;
                if inp.count == 0 {
                    f.input = None;
                }
            }
            match f.output.as_mut() {
                Some(o) => o.count += 1,
                None => {
                    f.output = Some(ItemStack {
                        item: out,
                        count: 1,
                        durability: 0,
                    })
                }
            }
        }
    } else {
        // Not actively smelting (no input, output full, or fire out): the progress arrow relaxes.
        f.cook_progress = (f.cook_progress - dt * 2.0).max(0.0);
    }
}

pub struct Game {
    world: World,
    worldgen: Arc<Worldgen>,
    seed: u64,
    /// Player-edited chunks, kept resident (and persisted) regardless of streaming.
    saved: FxHashMap<IVec3, Arc<Chunk>>,
    dirty: bool,
    /// Built meshes per chunk; `None` means "meshed but empty" (no geometry).
    meshes: FxHashMap<IVec3, Option<GpuMesh>>,
    render_distance: i32,
    center: IVec3,
    /// XZ offsets within (render_distance + 1), sorted nearest-first.
    offsets: Vec<(i32, i32, i32)>, // (dx, dz, dist2)

    volume: VoxelVolume,
    pool: WorkerPool,
    pending_gen: FxHashSet<IVec3>,
    pending_mesh: FxHashSet<IVec3>,
    ready_meshes: VecDeque<(IVec3, u32, MeshData)>,
    /// Mesh version per chunk; bumped on edit so stale in-flight mesh jobs are discarded.
    versions: FxHashMap<IVec3, u32>,

    max_inflight_gen: usize,
    max_inflight_mesh: usize,
    upload_budget: usize,

    /// Active fluid cells (kind + remaining horizontal reach) and a frontier of cells still to
    /// evaluate for spreading. Flood is monotonic (fluids only enter air), so it terminates.
    fluid: FxHashMap<IVec3, (BlockId, u8)>,
    fluid_frontier: VecDeque<IVec3>,
    fluid_timer: f32,

    /// Mobs and dropped items.
    entities: Entities,

    /// Furnaces the player has opened, keyed by block position; ticked by `step_furnaces`.
    furnaces: FxHashMap<IVec3, FurnaceState>,

    /// Chests the player has opened/placed, keyed by block position (27-slot containers).
    chests: FxHashMap<IVec3, crate::container::Container>,

    /// World difficulty (P6): gates hostile spawning; mirrored from the app via `set_difficulty`.
    difficulty: crate::rules::Difficulty,

    /// Natural-spawn cadence + a small RNG for spawn placement (M31).
    spawn_timer: f32,
    spawn_rng: u64,

    /// Random-block-tick cadence + RNG (U9): drives amethyst/dripstone/cave-vine growth in near chunks.
    tick_timer: f32,
    tick_rng: u64,

    /// U10 sculk: in-flight catalyst spread charges (a budgeted frontier; transient — dropped on save
    /// like the fluid frontier, but the sculk BLOCKS persist via the normal edited-chunk path).
    sculk_charges: VecDeque<SculkCharge>,
    /// U10 vibrations (game events) emitted recently, aged by TTL; heard by the Warden (U11).
    vibrations: Vec<Vibration>,

    /// U11 footstep emitter: the previous camera position + a stride timer so a moving player emits a
    /// periodic vibration the Warden can track (reduced: no sneak gate — see notes).
    last_cam: Vec3,
    step_timer: f32,
    /// U11 shrieker→Warden warning. A shrieker hearing a vibration while the player is deep raises the
    /// warning level (on a cooldown); at 4 it spawns a Warden near the shrieker, then resets.
    warden_warning: u8,
    warden_warn_pos: Option<Vec3>,
    shriek_cd: f32,
    /// U11 darkness the app drains into the Player after each update (a shrieker pulse). Read + cleared
    /// via `take_pending_darkness` (Game can't reach the Player directly).
    pending_darkness: f32,
}

impl Game {
    pub fn new(
        gpu: &Gpu,
        volume_bgl: &wgpu::BindGroupLayout,
        seed: u64,
        render_distance: i32,
        saved_chunks: FxHashMap<IVec3, Chunk>,
    ) -> Self {
        let saved: FxHashMap<IVec3, Arc<Chunk>> = saved_chunks
            .into_iter()
            .map(|(pos, chunk)| (pos, Arc::new(chunk)))
            .collect();
        let volume = VoxelVolume::new(gpu, volume_bgl);
        let r = render_distance + 1;
        let mut offsets = Vec::new();
        for dz in -r..=r {
            for dx in -r..=r {
                let dist2 = dx * dx + dz * dz;
                if dist2 <= r * r {
                    offsets.push((dx, dz, dist2));
                }
            }
        }
        offsets.sort_by_key(|&(_, _, d2)| d2);

        let workers = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2))
            .unwrap_or(4)
            .max(1);
        let worldgen = Arc::new(Worldgen::new(seed));
        let pool = WorkerPool::new(worldgen.clone(), workers);
        log::info!("Worker pool: {} threads", pool.worker_count());

        Self {
            world: World::new(),
            worldgen,
            seed,
            saved,
            dirty: false,
            volume,
            meshes: FxHashMap::default(),
            render_distance,
            center: IVec3::new(i32::MIN, 0, i32::MIN),
            offsets,
            pool,
            pending_gen: FxHashSet::default(),
            pending_mesh: FxHashSet::default(),
            ready_meshes: VecDeque::new(),
            versions: FxHashMap::default(),
            max_inflight_gen: 1024,
            max_inflight_mesh: 512,
            upload_budget: 64,
            fluid: FxHashMap::default(),
            fluid_frontier: VecDeque::new(),
            fluid_timer: 0.0,
            entities: Entities::new(),
            furnaces: FxHashMap::default(),
            chests: FxHashMap::default(),
            difficulty: crate::rules::Difficulty::Normal,
            spawn_timer: 0.0,
            spawn_rng: seed ^ 0x5FA1_2E37_9B1D_C0DE,
            tick_timer: 0.0,
            tick_rng: seed ^ 0x7A1C_9E37_55C0_1DEF,
            sculk_charges: VecDeque::new(),
            vibrations: Vec::new(),
            last_cam: Vec3::ZERO,
            step_timer: 0.0,
            warden_warning: 0,
            warden_warn_pos: None,
            shriek_cd: 0.0,
            pending_darkness: 0.0,
        }
    }

    fn center_of(camera_pos: Vec3) -> IVec3 {
        world::chunk_of(IVec3::new(
            camera_pos.x.floor() as i32,
            camera_pos.y.floor() as i32,
            camera_pos.z.floor() as i32,
        ))
    }

    #[inline]
    fn within(&self, pos: IVec3, radius: i32) -> bool {
        (pos.x - self.center.x).abs() <= radius && (pos.z - self.center.z).abs() <= radius
    }

    fn neighbors_ready(&self, pos: IVec3) -> bool {
        for off in [IVec3::X, -IVec3::X, IVec3::Z, -IVec3::Z] {
            if !self.world.is_generated(pos + off) {
                return false;
            }
        }
        if pos.y - 1 >= 0 && !self.world.is_generated(pos - IVec3::Y) {
            return false;
        }
        if pos.y + 1 < WORLD_HEIGHT_CHUNKS && !self.world.is_generated(pos + IVec3::Y) {
            return false;
        }
        true
    }

    pub fn update(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        camera_pos: Vec3,
        dt: f32,
        day: f32,
    ) -> crate::entity::Collected {
        self.center = Self::center_of(camera_pos);
        let r = self.render_distance;

        // U10: age out old vibrations so consumers (sculk sensors / the Warden) see only recent events.
        self.vibrations.retain_mut(|v| {
            v.ttl -= dt;
            v.ttl > 0.0
        });

        // U11 footsteps: a moving player periodically emits a vibration the Warden can track. Emitted
        // BEFORE the entity tick so the Warden hears it this frame. (Reduced: no sneak gate — Minecraft
        // mutes sneaking; here every step is heard.) Skip the very first frame (last_cam unset).
        self.shriek_cd = (self.shriek_cd - dt).max(0.0);
        if self.last_cam != Vec3::ZERO {
            let moved = (camera_pos - self.last_cam).length();
            self.step_timer = (self.step_timer - dt).max(0.0);
            if moved > 0.08 && self.step_timer <= 0.0 {
                self.emit_vibration(camera_pos, 1); // freq 1 ≈ a step (MC)
                self.step_timer = 0.3;
            }
        }
        self.last_cam = camera_pos;

        // 1. Drain worker results.
        for result in self.pool.drain() {
            match result {
                JobResult::Generated { pos, chunk } => {
                    self.pending_gen.remove(&pos);
                    if self.within(pos, r + 2) {
                        self.world.chunks.insert(pos, Arc::new(chunk));
                    }
                }
                JobResult::Meshed { pos, version, mesh } => {
                    let current = self.versions.get(&pos).copied().unwrap_or(0);
                    if version == current && self.within(pos, r + 1) {
                        // Keep `pos` in `pending_mesh` until it is actually uploaded, so the
                        // submission loop below doesn't re-submit a chunk awaiting upload.
                        self.ready_meshes.push_back((pos, version, mesh));
                    } else {
                        // Stale (edited since submission) or out of range — discard and allow
                        // re-submission if still needed.
                        self.pending_mesh.remove(&pos);
                    }
                }
            }
        }

        // 2. Upload a budget of finished meshes to the GPU (main thread only).
        let mut uploads = 0;
        while uploads < self.upload_budget {
            let Some((pos, version, data)) = self.ready_meshes.pop_front() else {
                break;
            };
            self.pending_mesh.remove(&pos);
            // Skip if edited since this mesh was built.
            if self.versions.get(&pos).copied().unwrap_or(0) != version {
                continue;
            }
            let gpu_mesh = if data.is_empty() {
                None
            } else {
                Some(renderer.upload_mesh(gpu, &data))
            };
            self.meshes.insert(pos, gpu_mesh);
            uploads += 1;
        }

        let (cx, cz) = (self.center.x, self.center.z);

        // 3. Submit generation jobs (nearest-first), bounded by in-flight cap.
        'generate: for &(dx, dz, _) in &self.offsets {
            if self.pending_gen.len() >= self.max_inflight_gen {
                break;
            }
            for cy in 0..WORLD_HEIGHT_CHUNKS {
                let pos = IVec3::new(cx + dx, cy, cz + dz);
                if !self.world.is_generated(pos) && !self.pending_gen.contains(&pos) {
                    if let Some(c) = self.saved.get(&pos) {
                        // Edited chunk: load from memory instantly, no generation needed.
                        self.world.chunks.insert(pos, c.clone());
                    } else {
                        self.pending_gen.insert(pos);
                        self.pool.submit(Job::Generate { pos });
                        if self.pending_gen.len() >= self.max_inflight_gen {
                            break 'generate;
                        }
                    }
                }
            }
        }

        // 4. Submit mesh jobs for chunks whose neighbors are ready.
        'mesh: for &(dx, dz, dist2) in &self.offsets {
            if dist2 > r * r {
                break;
            }
            if self.pending_mesh.len() >= self.max_inflight_mesh {
                break;
            }
            for cy in 0..WORLD_HEIGHT_CHUNKS {
                let pos = IVec3::new(cx + dx, cy, cz + dz);
                if !self.meshes.contains_key(&pos)
                    && !self.pending_mesh.contains(&pos)
                    && self.neighbors_ready(pos)
                {
                    if let Some(neigh) = self.world.neighborhood(pos) {
                        let origin = world::chunk_origin(pos).to_array();
                        let version = self.versions.get(&pos).copied().unwrap_or(0);
                        self.pending_mesh.insert(pos);
                        self.pool.submit(Job::Mesh {
                            pos,
                            version,
                            neigh,
                            origin,
                        });
                        if self.pending_mesh.len() >= self.max_inflight_mesh {
                            break 'mesh;
                        }
                    }
                }
            }
        }

        // 5. Unload distant chunks/meshes.
        let keep = r + 2;
        let center = self.center;
        let in_keep = |pos: &IVec3| {
            (pos.x - center.x).abs() <= keep && (pos.z - center.z).abs() <= keep
        };
        self.meshes.retain(|pos, _| in_keep(pos));
        self.world
            .chunks
            .retain(|pos, _| (pos.x - center.x).abs() <= keep + 1 && (pos.z - center.z).abs() <= keep + 1);
        self.pending_mesh.retain(in_keep);
        self.pending_gen
            .retain(|pos| (pos.x - center.x).abs() <= keep + 1 && (pos.z - center.z).abs() <= keep + 1);

        // Drop fluid cells whose chunk just unloaded, so the fluid map and frontier can't grow
        // without bound as the player roams (unsimulated fluid blocks are already persisted).
        let loaded = &self.world.chunks;
        self.fluid
            .retain(|pos, _| loaded.contains_key(&world::chunk_of(*pos)));
        self.fluid_frontier
            .retain(|pos| loaded.contains_key(&world::chunk_of(*pos)));
        // Drop *empty* furnaces in unloaded chunks so the map can't grow without bound as the player
        // roams; non-empty ones stay resident so their contents survive until M24 adds persistence.
        self.furnaces
            .retain(|pos, f| loaded.contains_key(&world::chunk_of(*pos)) || !f.is_empty());

        // Advance flowing fluids at a fixed cadence (bounded steps per frame).
        self.fluid_timer += dt;
        let mut steps = 0;
        while self.fluid_timer >= FLUID_INTERVAL && steps < 4 {
            self.fluid_timer -= FLUID_INTERVAL;
            steps += 1;
            if !self.step_fluids(gpu, renderer) {
                self.fluid_timer = 0.0;
                break;
            }
        }

        // Advance any active furnaces (pure item-state tick; no world/GPU touch).
        for f in self.furnaces.values_mut() {
            step_furnace(f, dt);
        }

        // U9: random block ticks (amethyst/dripstone/cave-vine growth) in near chunks.
        self.tick_timer += dt;
        if self.tick_timer >= RANDOM_TICK_INTERVAL {
            self.tick_timer -= RANDOM_TICK_INTERVAL;
            self.step_random_ticks(gpu, renderer);
        }

        // Natural spawning: one gated attempt per interval around the player (M31).
        self.spawn_timer += dt;
        if self.spawn_timer >= SPAWN_INTERVAL {
            self.spawn_timer -= SPAWN_INTERVAL;
            self.try_spawn(day, camera_pos);
        }

        // Update mobs and dropped items (AI + physics). The sampler closures borrow only the chunk
        // map (read-only), leaving `self.entities` free to mutate. `is_hazard` (P15 mob nav) flags
        // lava — non-solid, so the bool `is_solid` can't see it, but a mob must not walk into it.
        let chunks = &self.world.chunks;
        let block_at = |wp: IVec3| -> block::BlockId {
            let cpos = world::chunk_of(wp);
            match chunks.get(&cpos) {
                Some(c) => {
                    let o = world::chunk_origin(cpos);
                    c.get((wp.x - o.x) as usize, (wp.y - o.y) as usize, (wp.z - o.z) as usize)
                }
                None => block::AIR, // unloaded → air (a mob just won't sense a hazard it can't see)
            }
        };
        // P17: undead burn where it's daytime AND the cell is open to the sky (reuses the P16 sky-light
        // free fn against the same block_at; no self-borrow, so it coexists with the &mut entities tick).
        let top = WORLD_HEIGHT_CHUNKS * CHUNK_SIZE_I;
        // U11: the Warden's vibration sense. We can't call `self.nearest_vibration` while `self.entities`
        // is borrowed mut, so borrow the bus read-only up front and inline its body in the closure (it
        // mirrors `nearest_vibration`: the most-recent live vibration within `range` of `from`).
        let vibrations = &self.vibrations;
        let collected = self.entities.update(
            dt,
            camera_pos,
            |wp| block::is_solid(block_at(wp)),
            |wp| block_at(wp) == block::LAVA,
            |wp| day > 0.6 && sky_light(wp, top, &block_at) == 15,
            |from, range| nearest_vibration_in(vibrations, from, range),
        );

        // Apply any creeper explosions: carve the crater, then add radial blast damage to the player.
        let mut collected = collected;
        for (center, radius) in std::mem::take(&mut collected.explosions) {
            let dmg = self.apply_explosion(gpu, renderer, center, radius, camera_pos);
            if dmg > 0.0 {
                collected.player_damage.push((dmg, center)); // source = blast center
            }
        }

        // U10: mob deaths emit a vibration and, near a sculk catalyst, seed a sculk bloom.
        for &death in &collected.deaths {
            self.emit_vibration(death, 13); // freq 13 ≈ mob death (MC)
            let dpos = death.floor().as_ivec3();
            if self.catalyst_near(dpos, SCULK_CATALYST_RANGE) {
                self.sculk_charges.push_back(SculkCharge {
                    pos: dpos,
                    amount: 12,
                });
            }
        }
        self.step_sculk(gpu, renderer);

        // U11: shriekers near a fresh vibration (while the player is deep) build a warning that
        // eventually summons a Warden + inflicts Darkness. Runs AFTER all vibrations are emitted.
        self.step_warden_shriek(camera_pos);

        // Feed the GPU voxel volume (ray-traced lighting) around the player.
        self.volume.update(gpu, &self.world, self.center);

        collected
    }

    /// U11 shrieker → Warden. When the player is deep underground, a sculk shrieker that has a live
    /// vibration nearby "shrieks": on a cooldown it raises a warning level and pulses Darkness onto the
    /// player. At WARDEN_SUMMON_LEVEL it spawns a Warden by the shrieker (unless one is already near) and
    /// resets. Cheap: only runs when deep AND off cooldown, scanning a tiny radius per live vibration.
    fn step_warden_shriek(&mut self, camera_pos: Vec3) {
        // Only the deep dark summons the Warden; and only one shriek per cooldown window. Cheap gate:
        // there must be a live vibration within shrieker-hearing range of the player at all.
        if camera_pos.y >= WARDEN_DEEP_Y
            || self.shriek_cd > 0.0
            || self.nearest_vibration(camera_pos, WARDEN_DEEP_Y).is_none()
        {
            return;
        }
        // Don't pile on a second Warden if one is already prowling near the player.
        if self.entities.warden_count() > 0 {
            return;
        }
        const R: i32 = 6;
        // Snapshot vibration positions (so the &self block_at scan doesn't conflict with the borrow).
        let vib_cells: Vec<IVec3> = self.vibrations.iter().map(|v| v.pos.floor().as_ivec3()).collect();
        let mut shrieker: Option<IVec3> = None;
        'scan: for c in vib_cells {
            for dy in -R..=R {
                for dz in -R..=R {
                    for dx in -R..=R {
                        let p = c + IVec3::new(dx, dy, dz);
                        if self.block_at(p) == block::SCULK_SHRIEKER {
                            shrieker = Some(p);
                            break 'scan;
                        }
                    }
                }
            }
        }
        let Some(sp) = shrieker else { return };
        // A shriek fired: cooldown, raise the warning, pulse Darkness, remember where.
        self.shriek_cd = SHRIEK_COOLDOWN;
        self.warden_warning = self.warden_warning.saturating_add(1);
        self.warden_warn_pos = Some(sp.as_vec3() + Vec3::splat(0.5));
        self.pending_darkness = self.pending_darkness.max(DARKNESS_PULSE);
        if self.warden_warning >= WARDEN_SUMMON_LEVEL {
            // Reduced spawn placement: just above the shrieker (no full 3-air scan).
            let pos = sp.as_vec3() + Vec3::new(0.5, 1.0, 0.5);
            self.entities.spawn_mob(pos, crate::entity::Species::Warden);
            self.warden_warning = 0;
        }
    }

    /// U11: the app drains this (a one-shot Darkness pulse from a shriek) into the Player after update.
    pub fn take_pending_darkness(&mut self) -> f32 {
        std::mem::take(&mut self.pending_darkness)
    }

    /// Carve a spherical crater of AIR (skipping bedrock) and return the radial blast damage the
    /// player at `player_pos` should take (falls off to 0 at ~1.6x the radius).
    fn apply_explosion(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        center: Vec3,
        radius: f32,
        player_pos: Vec3,
    ) -> f32 {
        let r = radius.ceil() as i32;
        let c = IVec3::new(
            center.x.floor() as i32,
            center.y.floor() as i32,
            center.z.floor() as i32,
        );
        // Collect the cells to clear, then carve them in ONE batched pass (apply_fluid_changes
        // re-meshes each affected chunk once — set_block-per-block would remesh 100+ times). Fluids
        // are left intact (Minecraft blasts don't destroy water; carving them leaves dry holes).
        let mut changes: Vec<(IVec3, BlockId)> = Vec::new();
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    let p = c + IVec3::new(dx, dy, dz);
                    if (p.as_vec3() + Vec3::splat(0.5) - center).length() <= radius {
                        let id = self.block_at(p);
                        if id != block::AIR && id != block::BEDROCK && !block::is_fluid(id) {
                            changes.push((p, block::AIR));
                        }
                    }
                }
            }
        }
        self.apply_fluid_changes(gpu, renderer, &changes);
        explosion_damage((player_pos - center).length(), radius)
    }

    /// Debug: detonate at `center` (headless explosion verification).
    pub fn debug_explode(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        center: Vec3,
        radius: f32,
        player_pos: Vec3,
    ) -> f32 {
        self.apply_explosion(gpu, renderer, center, radius, player_pos)
    }

    /// Fully generate and mesh the current radius synchronously (used for headless screenshots).
    pub fn load_all_blocking(&mut self, gpu: &Gpu, renderer: &ChunkRenderer, camera_pos: Vec3) {
        self.center = Self::center_of(camera_pos);
        let (cx, cz) = (self.center.x, self.center.z);
        let r = self.render_distance;

        for &(dx, dz, _) in &self.offsets {
            for cy in 0..WORLD_HEIGHT_CHUNKS {
                let pos = IVec3::new(cx + dx, cy, cz + dz);
                if !self.world.is_generated(pos) {
                    let chunk = match self.saved.get(&pos) {
                        Some(c) => c.clone(),
                        None => Arc::new(self.worldgen.generate_chunk(pos)),
                    };
                    self.world.chunks.insert(pos, chunk);
                }
            }
        }
        let positions: Vec<IVec3> = self
            .offsets
            .iter()
            .filter(|&&(_, _, d2)| d2 <= r * r)
            .flat_map(|&(dx, dz, _)| {
                (0..WORLD_HEIGHT_CHUNKS).map(move |cy| IVec3::new(cx + dx, cy, cz + dz))
            })
            .collect();
        for pos in positions {
            if !self.meshes.contains_key(&pos) {
                if let Some(neigh) = self.world.neighborhood(pos) {
                    let origin = world::chunk_origin(pos).to_array();
                    let data = mesher::build_mesh(&neigh, origin);
                    let gpu_mesh = if data.is_empty() {
                        None
                    } else {
                        Some(renderer.upload_mesh(gpu, &data))
                    };
                    self.meshes.insert(pos, gpu_mesh);
                }
            }
        }
    }

    /// Block id at a world position (air if the chunk isn't loaded).
    pub fn block_at(&self, wp: IVec3) -> crate::block::BlockId {
        let cpos = world::chunk_of(wp);
        let origin = world::chunk_origin(cpos);
        match self.world.get(cpos) {
            Some(chunk) => chunk.get(
                (wp.x - origin.x) as usize,
                (wp.y - origin.y) as usize,
                (wp.z - origin.z) as usize,
            ),
            None => crate::block::AIR,
        }
    }

    pub fn is_solid_at(&self, wp: IVec3) -> bool {
        crate::block::is_solid(self.block_at(wp))
    }

    /// Block id + block-state byte at a world position (air/0 if the chunk isn't loaded). Used by the
    /// player collision closure so stair facing (and future oriented blocks) collide correctly.
    pub fn block_state_at(&self, wp: IVec3) -> (crate::block::BlockId, u8) {
        let cpos = world::chunk_of(wp);
        let origin = world::chunk_origin(cpos);
        match self.world.get(cpos) {
            Some(chunk) => {
                let (x, y, z) = (
                    (wp.x - origin.x) as usize,
                    (wp.y - origin.y) as usize,
                    (wp.z - origin.z) as usize,
                );
                (chunk.get(x, y, z), chunk.state(x, y, z))
            }
            None => (crate::block::AIR, 0),
        }
    }

    /// Set a block with its default state (0). Delegates to `set_block_state`.
    pub fn set_block(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        wp: IVec3,
        id: crate::block::BlockId,
    ) -> bool {
        self.set_block_state(gpu, renderer, wp, id, 0)
    }

    /// Set a block id + block-state byte at a world position and synchronously re-mesh the affected
    /// chunk(s). Returns true if a change was made (a different id OR a different orientation/state).
    pub fn set_block_state(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        wp: IVec3,
        id: crate::block::BlockId,
        state: u8,
    ) -> bool {
        let cpos = world::chunk_of(wp);
        let origin = world::chunk_origin(cpos);
        let (lx, ly, lz) = (
            (wp.x - origin.x) as usize,
            (wp.y - origin.y) as usize,
            (wp.z - origin.z) as usize,
        );

        let (old, old_state) = {
            let Some(arc) = self.world.chunks.get_mut(&cpos) else {
                return false;
            };
            let chunk = Arc::make_mut(arc);
            let old = chunk.get(lx, ly, lz);
            let old_state = chunk.state(lx, ly, lz);
            if old == id && old_state == state {
                return false;
            }
            chunk.set_state(lx, ly, lz, id, state);
            (old, old_state)
        };

        // Breaking/replacing a furnace spills its contents so smelted goods and fuel aren't lost.
        if old == block::FURNACE && id != block::FURNACE {
            if let Some(state) = self.furnaces.remove(&wp) {
                let center = wp.as_vec3() + Vec3::splat(0.5);
                for stack in state.contents() {
                    self.entities.spawn_item(center, stack);
                }
            }
        }
        // Breaking/replacing a chest spills its contents.
        if old == block::CHEST && id != block::CHEST {
            if let Some(c) = self.chests.remove(&wp) {
                let center = wp.as_vec3() + Vec3::splat(0.5);
                for stack in c.contents() {
                    self.entities.spawn_item(center, stack);
                }
            }
        }
        // Breaking either half of a 2-tall door removes its partner half. The drop (one door item)
        // comes from the player-mined cell only; this direct removal bypasses the drop path. The
        // recursion terminates: the partner's own hook finds this cell already non-door.
        if old == block::WOODEN_DOOR && id != block::WOODEN_DOOR {
            let partner = if block::door_half(old_state) == block::DOOR_LOWER {
                wp + IVec3::Y
            } else {
                wp - IVec3::Y
            };
            if self.block_at(partner) == block::WOODEN_DOOR {
                self.set_block_state(gpu, renderer, partner, block::AIR, 0);
            }
        }

        // Persist the edited chunk (kept resident in `saved`) and refresh its volume voxels.
        if let Some(arc) = self.world.chunks.get(&cpos) {
            self.saved.insert(cpos, arc.clone());
            self.dirty = true;
        }
        self.volume.invalidate(cpos);

        // A non-fluid placement (including breaking to air) clears any fluid record here.
        if !block::is_fluid(id) {
            self.fluid.remove(&wp);
        }

        // Edited chunk plus any neighbor sharing the touched boundary must re-mesh.
        let mut affected = vec![cpos];
        if lx == 0 {
            affected.push(cpos - IVec3::X);
        } else if lx == 31 {
            affected.push(cpos + IVec3::X);
        }
        if ly == 0 {
            affected.push(cpos - IVec3::Y);
        } else if ly == 31 {
            affected.push(cpos + IVec3::Y);
        }
        if lz == 0 {
            affected.push(cpos - IVec3::Z);
        } else if lz == 31 {
            affected.push(cpos + IVec3::Z);
        }

        for p in affected {
            *self.versions.entry(p).or_insert(0) += 1;
            self.pending_mesh.remove(&p);
            self.remesh_sync(gpu, renderer, p);
        }
        // U10: a block edit is a vibration the sculk family + Warden hear (break vs place frequency).
        let freq = if id == block::AIR { 6 } else { 9 };
        self.emit_vibration(wp.as_vec3() + Vec3::splat(0.5), freq);
        true
    }

    fn remesh_sync(&mut self, gpu: &Gpu, renderer: &ChunkRenderer, pos: IVec3) {
        if let Some(neigh) = self.world.neighborhood(pos) {
            let origin = world::chunk_origin(pos).to_array();
            let data = mesher::build_mesh(&neigh, origin);
            let gpu_mesh = if data.is_empty() {
                None
            } else {
                Some(renderer.upload_mesh(gpu, &data))
            };
            self.meshes.insert(pos, gpu_mesh);
        }
    }

    /// Register a player-placed fluid block as a flow source (the block itself is already set).
    pub fn add_fluid_source(&mut self, pos: IVec3, kind: BlockId) {
        self.fluid.insert(pos, (kind, 0));
        self.fluid_frontier.push_back(pos);
    }

    /// True if `p` is in a loaded chunk and currently empty air (so a fluid may flow into it).
    fn is_air_loaded(&self, p: IVec3) -> bool {
        let cpos = world::chunk_of(p);
        match self.world.get(cpos) {
            Some(chunk) => {
                let o = world::chunk_origin(cpos);
                chunk.get((p.x - o.x) as usize, (p.y - o.y) as usize, (p.z - o.z) as usize)
                    == block::AIR
            }
            None => false,
        }
    }

    /// One fluid step: drain a budget of frontier cells, flowing each straight down, or — if
    /// blocked below — outward to air neighbors with reduced reach. The flood only enters air, so
    /// it is monotonic and terminates. Returns true while frontier work remains.
    fn step_fluids(&mut self, gpu: &Gpu, renderer: &ChunkRenderer) -> bool {
        if self.fluid_frontier.is_empty() {
            return false;
        }
        let mut changes: Vec<(IVec3, BlockId)> = Vec::new();
        let mut created: Vec<IVec3> = Vec::new();
        let mut budget = FLUID_BUDGET;
        while budget > 0 {
            let Some(pos) = self.fluid_frontier.pop_front() else {
                break;
            };
            let Some(&(kind, level)) = self.fluid.get(&pos) else {
                continue;
            };
            budget -= 1;
            // The player may have removed this fluid since it was queued.
            if self.block_at(pos) != kind {
                self.fluid.remove(&pos);
                continue;
            }
            let below = pos - IVec3::Y;
            if self.is_air_loaded(below) {
                // Fall straight down; landing resets horizontal reach to full.
                self.fluid.insert(below, (kind, 0));
                changes.push((below, kind));
                created.push(below);
            } else {
                let max_spread = if kind == block::LAVA { LAVA_SPREAD } else { WATER_SPREAD };
                if level < max_spread {
                    for d in [IVec3::X, -IVec3::X, IVec3::Z, -IVec3::Z] {
                        let n = pos + d;
                        if !self.fluid.contains_key(&n) && self.is_air_loaded(n) {
                            self.fluid.insert(n, (kind, level + 1));
                            changes.push((n, kind));
                            created.push(n);
                        }
                    }
                }
            }
        }
        for p in created {
            self.fluid_frontier.push_back(p);
        }
        if !changes.is_empty() {
            self.apply_fluid_changes(gpu, renderer, &changes);
        }
        !self.fluid_frontier.is_empty()
    }

    /// Apply a batch of fluid block writes, then re-mesh each touched chunk exactly once (far
    /// cheaper than routing every cell through `set_block`, which re-meshes per call).
    fn apply_fluid_changes(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        changes: &[(IVec3, BlockId)],
    ) {
        let mut mutated: FxHashSet<IVec3> = FxHashSet::default();
        let mut affected: FxHashSet<IVec3> = FxHashSet::default();
        for &(wp, id) in changes {
            let cpos = world::chunk_of(wp);
            let origin = world::chunk_origin(cpos);
            let (lx, ly, lz) = (
                (wp.x - origin.x) as usize,
                (wp.y - origin.y) as usize,
                (wp.z - origin.z) as usize,
            );
            let Some(arc) = self.world.chunks.get_mut(&cpos) else {
                continue;
            };
            let chunk = Arc::make_mut(arc);
            if chunk.get(lx, ly, lz) == id {
                continue;
            }
            chunk.set(lx, ly, lz, id);
            mutated.insert(cpos);
            affected.insert(cpos);
            if lx == 0 {
                affected.insert(cpos - IVec3::X);
            } else if lx == 31 {
                affected.insert(cpos + IVec3::X);
            }
            if ly == 0 {
                affected.insert(cpos - IVec3::Y);
            } else if ly == 31 {
                affected.insert(cpos + IVec3::Y);
            }
            if lz == 0 {
                affected.insert(cpos - IVec3::Z);
            } else if lz == 31 {
                affected.insert(cpos + IVec3::Z);
            }
        }
        for &cpos in &mutated {
            if let Some(arc) = self.world.chunks.get(&cpos) {
                self.saved.insert(cpos, arc.clone());
            }
            self.volume.invalidate(cpos);
        }
        if !mutated.is_empty() {
            self.dirty = true;
        }
        for cpos in affected {
            *self.versions.entry(cpos).or_insert(0) += 1;
            self.pending_mesh.remove(&cpos);
            // Only re-mesh chunks that are still loaded; an unloaded boundary neighbor re-meshes
            // with correct neighbors when streaming brings it back.
            if self.world.is_generated(cpos) {
                self.remesh_sync(gpu, renderer, cpos);
            }
        }
    }

    /// Batched block edits WITH state bytes (U9 growth / U10 sculk spread): mutate, persist, invalidate
    /// the volume, and re-mesh each affected chunk exactly once (never per-cell). Skips no-op writes.
    fn apply_block_changes(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        changes: &[(IVec3, BlockId, u8)],
    ) {
        let mut mutated: FxHashSet<IVec3> = FxHashSet::default();
        let mut affected: FxHashSet<IVec3> = FxHashSet::default();
        for &(wp, id, state) in changes {
            let cpos = world::chunk_of(wp);
            let origin = world::chunk_origin(cpos);
            let (lx, ly, lz) = (
                (wp.x - origin.x) as usize,
                (wp.y - origin.y) as usize,
                (wp.z - origin.z) as usize,
            );
            let Some(arc) = self.world.chunks.get_mut(&cpos) else {
                continue;
            };
            let chunk = Arc::make_mut(arc);
            if chunk.get(lx, ly, lz) == id && chunk.state(lx, ly, lz) == state {
                continue;
            }
            chunk.set_state(lx, ly, lz, id, state);
            mutated.insert(cpos);
            affected.insert(cpos);
            if lx == 0 {
                affected.insert(cpos - IVec3::X);
            } else if lx == 31 {
                affected.insert(cpos + IVec3::X);
            }
            if ly == 0 {
                affected.insert(cpos - IVec3::Y);
            } else if ly == 31 {
                affected.insert(cpos + IVec3::Y);
            }
            if lz == 0 {
                affected.insert(cpos - IVec3::Z);
            } else if lz == 31 {
                affected.insert(cpos + IVec3::Z);
            }
        }
        for &cpos in &mutated {
            if let Some(arc) = self.world.chunks.get(&cpos) {
                self.saved.insert(cpos, arc.clone());
            }
            self.volume.invalidate(cpos);
        }
        if !mutated.is_empty() {
            self.dirty = true;
        }
        for cpos in affected {
            *self.versions.entry(cpos).or_insert(0) += 1;
            self.pending_mesh.remove(&cpos);
            if self.world.is_generated(cpos) {
                self.remesh_sync(gpu, renderer, cpos);
            }
        }
    }

    /// U9 random-tick scheduler: each cadence, sample a few random cells in near-loaded chunks and
    /// dispatch growth (amethyst/dripstone/cave vines). Edits are batched -> one re-mesh per chunk.
    fn step_random_ticks(&mut self, gpu: &Gpu, renderer: &ChunkRenderer) {
        let center = self.center;
        let near: Vec<IVec3> = self
            .world
            .chunks
            .keys()
            .filter(|p| {
                (p.x - center.x).abs() <= RANDOM_TICK_RADIUS
                    && (p.z - center.z).abs() <= RANDOM_TICK_RADIUS
            })
            .copied()
            .collect();
        let mut rng = self.tick_rng;
        // Phase 1: sample candidate (wp, id) of random-tickable blocks (no &mut self while reading).
        let mut candidates: Vec<(IVec3, BlockId)> = Vec::new();
        for cpos in &near {
            let Some(arc) = self.world.chunks.get(cpos) else {
                continue;
            };
            let origin = world::chunk_origin(*cpos);
            for _ in 0..RANDOM_TICKS_PER_CHUNK {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let (lx, ly, lz) =
                    ((rng & 31) as usize, ((rng >> 5) & 31) as usize, ((rng >> 10) & 31) as usize);
                let id = arc.get(lx, ly, lz);
                if random_tickable(id) {
                    candidates.push((origin + IVec3::new(lx as i32, ly as i32, lz as i32), id));
                }
            }
        }
        // Phase 2: dispatch growth into a batch.
        let mut changes: Vec<(IVec3, BlockId, u8)> = Vec::new();
        for (wp, id) in candidates {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            self.random_tick_block(wp, id, rng as u32, &mut changes);
        }
        self.tick_rng = rng;
        if !changes.is_empty() {
            self.apply_block_changes(gpu, renderer, &changes);
        }
    }

    /// Dispatch one random block tick to the right growth handler (U9). Read-only on the world; the
    /// resulting edits go into `changes` (applied as a batch by the caller).
    fn random_tick_block(
        &self,
        wp: IVec3,
        id: BlockId,
        rnd: u32,
        changes: &mut Vec<(IVec3, BlockId, u8)>,
    ) {
        if id == block::BUDDING_AMETHYST {
            // Sprout a small bud on a random air-adjacent face (~18% per tick).
            if rnd % 100 < 18 {
                const DIRS: [IVec3; 5] =
                    [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z, IVec3::Y];
                let np = wp + DIRS[((rnd >> 7) % 5) as usize];
                if self.block_at(np) == block::AIR {
                    changes.push((np, block::SMALL_AMETHYST_BUD, 0));
                }
            }
            return;
        }
        let below = self.block_at(wp - IVec3::Y);
        let above = self.block_at(wp + IVec3::Y);
        let vine_len = if matches!(id, block::CAVE_VINE | block::CAVE_VINE_BERRIES) {
            self.vine_length(wp)
        } else {
            0
        };
        if let Some((off, new)) = grow_decision(id, rnd, below, above, vine_len) {
            changes.push((wp + off, new, 0));
        }
    }

    /// Cave-vine chain length directly above `wp` (a growth length cap so vines don't reach forever).
    fn vine_length(&self, wp: IVec3) -> i32 {
        let mut n = 0;
        let mut p = wp + IVec3::Y;
        while n < 16 {
            let id = self.block_at(p);
            if id != block::CAVE_VINE && id != block::CAVE_VINE_BERRIES {
                break;
            }
            n += 1;
            p += IVec3::Y;
        }
        n
    }

    /// U9 bonemeal: instantly advance a growable block at `wp` (budding amethyst sprouts/advances,
    /// cave vines extend with berries, moss spreads to adjacent stone/dirt). Returns true if it grew.
    pub fn bonemeal(&mut self, gpu: &Gpu, renderer: &ChunkRenderer, wp: IVec3) -> bool {
        let id = self.block_at(wp);
        let mut changes: Vec<(IVec3, BlockId, u8)> = Vec::new();
        match id {
            block::BUDDING_AMETHYST => {
                for d in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z, IVec3::Y] {
                    if self.block_at(wp + d) == block::AIR {
                        changes.push((wp + d, block::SMALL_AMETHYST_BUD, 0));
                    }
                }
            }
            block::SMALL_AMETHYST_BUD => changes.push((wp, block::LARGE_AMETHYST_BUD, 0)),
            block::MEDIUM_AMETHYST_BUD | block::LARGE_AMETHYST_BUD => {
                changes.push((wp, block::AMETHYST_CLUSTER, 0))
            }
            block::CAVE_VINE | block::CAVE_VINE_BERRIES => {
                let mut p = wp - IVec3::Y;
                for _ in 0..3 {
                    if self.block_at(p) != block::AIR {
                        break;
                    }
                    changes.push((p, block::CAVE_VINE_BERRIES, 0));
                    p -= IVec3::Y;
                }
            }
            block::MOSS_BLOCK => {
                for d in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                    let np = wp + d;
                    if matches!(
                        self.block_at(np),
                        block::STONE | block::DIRT | block::DEEPSLATE | block::COBBLED_DEEPSLATE
                    ) {
                        changes.push((np, block::MOSS_BLOCK, 0));
                    }
                }
            }
            _ => return false,
        }
        if changes.is_empty() {
            return false;
        }
        self.apply_block_changes(gpu, renderer, &changes);
        true
    }

    /// U10: emit a vibration (game event) at a world position with an MC frequency class. Heard by the
    /// sculk family + the Warden (U11) over a short TTL.
    pub fn emit_vibration(&mut self, pos: Vec3, freq: u8) {
        self.vibrations.push(Vibration {
            pos,
            freq,
            ttl: VIBRATION_TTL,
        });
    }

    /// The position of the most recent live vibration within `range` of `from` (U11 Warden targeting).
    pub fn nearest_vibration(&self, from: Vec3, range: f32) -> Option<Vec3> {
        nearest_vibration_in(&self.vibrations, from, range)
    }

    /// Whether a sculk catalyst sits within `r` blocks of `pos` (scanned on mob death; deaths are rare).
    fn catalyst_near(&self, pos: IVec3, r: i32) -> bool {
        for dy in -r..=r {
            for dz in -r..=r {
                for dx in -r..=r {
                    if self.block_at(pos + IVec3::new(dx, dy, dz)) == block::SCULK_CATALYST {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// True if any of the 6 face neighbors of `p` is air (so sculk only spreads over exposed surfaces).
    fn exposed_to_air(&self, p: IVec3) -> bool {
        const DIRS: [IVec3; 6] =
            [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];
        DIRS.iter().any(|d| self.block_at(p + *d) == block::AIR)
    }

    /// U10 sculk spread tick: drain a budget of catalyst charges; each FLOODS to a few exposed
    /// convertible neighbors (so a bloom grows as a patch, not a thread), enqueuing a reduced child at
    /// each and occasionally capping it with a sculk sensor. Batched edits (persist via the edited path).
    fn step_sculk(&mut self, gpu: &Gpu, renderer: &ChunkRenderer) {
        if self.sculk_charges.is_empty() {
            return;
        }
        const DIRS: [IVec3; 6] =
            [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];
        let mut changes: Vec<(IVec3, BlockId, u8)> = Vec::new();
        let mut done: FxHashSet<IVec3> = FxHashSet::default(); // cells converted in this batch
        let mut rng = self.tick_rng;
        let mut budget = SCULK_BUDGET;
        while budget > 0 {
            let Some(charge) = self.sculk_charges.pop_front() else {
                break;
            };
            if charge.amount == 0 {
                continue;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let start = (rng % 6) as usize;
            let mut converted = 0;
            for i in 0..6 {
                if budget == 0 || converted >= 3 {
                    break;
                }
                let np = charge.pos + DIRS[(start + i) % 6];
                if done.contains(&np)
                    || !sculk_convertible(self.block_at(np))
                    || !self.exposed_to_air(np)
                {
                    continue;
                }
                done.insert(np);
                changes.push((np, block::SCULK, 0));
                budget -= 1;
                converted += 1;
                if charge.amount > 1 {
                    self.sculk_charges.push_back(SculkCharge {
                        pos: np,
                        amount: charge.amount - 1,
                    });
                }
                // A high-charge bloom occasionally sprouts a sculk sensor on the new sculk.
                if charge.amount > 8 && rng % 20 == 0 && self.block_at(np + IVec3::Y) == block::AIR {
                    changes.push((np + IVec3::Y, block::SCULK_SENSOR, 0));
                }
            }
        }
        self.tick_rng = rng;
        if !changes.is_empty() {
            self.apply_block_changes(gpu, renderer, &changes);
        }
    }

    /// Debug/headless: seed a sculk-spread charge at `pos` (simulates a catalyst bloom) so the next
    /// update ticks spread sculk over the surrounding exposed rock. Used for U10 verification.
    pub fn debug_seed_sculk(&mut self, pos: IVec3, amount: u16) {
        self.sculk_charges.push_back(SculkCharge { pos, amount });
    }

    pub fn spawn_mob(&mut self, pos: Vec3, species: crate::entity::Species) {
        self.entities.spawn_mob(pos, species);
    }

    pub fn spawn_baby(&mut self, pos: Vec3, species: crate::entity::Species) {
        self.entities.spawn_baby(pos, species);
    }

    pub fn spawn_item(&mut self, pos: Vec3, stack: crate::item::ItemStack) {
        self.entities.spawn_item(pos, stack);
    }

    pub fn spawn_xp(&mut self, pos: Vec3, amount: u32) {
        self.entities.spawn_xp(pos, amount);
    }

    pub fn spawn_arrow(&mut self, pos: Vec3, vel: Vec3) {
        self.entities.spawn_arrow(pos, vel);
    }

    /// Loose a PLAYER arrow (bow shot, P13): hits mobs, never the player.
    pub fn spawn_player_arrow(&mut self, pos: Vec3, vel: Vec3, damage: f32) {
        self.entities.spawn_player_arrow(pos, vel, damage);
    }

    fn spawn_rand(&mut self) -> u64 {
        let mut x = self.spawn_rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.spawn_rng = x;
        x
    }

    /// Default player spawn Y for a fresh world: a couple of blocks above the generated surface column
    /// so the player settles onto the ground. Needed now that the world is deepened (sea level y96) —
    /// the old fixed y96 spawn would land inside the raised terrain.
    pub fn spawn_surface_y(&self, wx: i32, wz: i32) -> f32 {
        self.worldgen.height(wx, wz) as f32 + 2.0
    }

    /// Topmost walkable ground surface at (wx,wz) with 2 air blocks above (room for a mob), else
    /// None. Only real ground counts (grass/dirt/stone/sand/snow/gravel) — never a leaf canopy or a
    /// tree trunk, so mobs don't spawn perched in the air on trees.
    fn surface_at(&self, wx: i32, wz: i32) -> Option<(i32, BlockId)> {
        for y in (1..WORLD_HEIGHT_CHUNKS * CHUNK_SIZE_I - 2).rev() {
            let here = self.block_at(IVec3::new(wx, y, wz));
            let ground = matches!(
                here,
                block::GRASS
                    | block::DIRT
                    | block::STONE
                    | block::SAND
                    | block::SNOW
                    | block::GRAVEL
            );
            if ground
                && self.block_at(IVec3::new(wx, y + 1, wz)) == block::AIR
                && self.block_at(IVec3::new(wx, y + 2, wz)) == block::AIR
            {
                return Some((y, here));
            }
        }
        None
    }

    /// Sky-light at a world cell (0 or 15, binary like the mesher): 15 if open to the sky straight up.
    fn sky_light_at(&self, wp: IVec3) -> u8 {
        sky_light(wp, WORLD_HEIGHT_CHUNKS * CHUNK_SIZE_I, &|p| self.block_at(p))
    }

    /// Block-light at a world cell (0..15). A cheap chunk-`emitter_count` precheck skips the full
    /// flood on the common no-emitter case (the open surface), so this is usually O(a few chunk reads).
    fn block_light_at(&self, wp: IVec3) -> u8 {
        const R: i32 = 15;
        let (c0, c1) = (world::chunk_of(wp - IVec3::splat(R)), world::chunk_of(wp + IVec3::splat(R)));
        let mut any = false;
        'outer: for cx in c0.x..=c1.x {
            for cy in c0.y..=c1.y {
                for cz in c0.z..=c1.z {
                    if self.world.get(IVec3::new(cx, cy, cz)).is_some_and(|c| c.emitter_count > 0) {
                        any = true;
                        break 'outer;
                    }
                }
            }
        }
        if !any {
            return 0;
        }
        block_light(wp, &|p| self.block_at(p))
    }

    /// Live hostile / passive mob counts (P16 spawn caps).
    pub fn hostile_count(&self) -> usize {
        self.entities.hostile_count()
    }
    pub fn passive_count(&self) -> usize {
        self.entities.passive_count()
    }

    /// Spawn a pack of `species` clustered near (wx,wz): each member re-scans for valid grass footing
    /// (open to sky), skips already-used cells (no stacking), and respects PASSIVE_CAP. Returns how
    /// many actually spawned (fewer near edges/obstacles). Deterministic via `spawn_rand`.
    fn spawn_passive_pack(&mut self, species: crate::entity::Species, wx: i32, wz: i32) -> usize {
        let mut spawned = 0;
        let mut used: Vec<(i32, i32)> = Vec::new();
        let mut attempts = 0;
        while spawned < PACK_SIZE && attempts < PACK_SIZE * 4 {
            attempts += 1;
            if self.entities.passive_count() >= PASSIVE_CAP {
                break;
            }
            let r = self.spawn_rand();
            let (cx, cz) = (wx + (r % 7) as i32 - PACK_RADIUS, wz + ((r >> 8) % 7) as i32 - PACK_RADIUS);
            if used.contains(&(cx, cz)) {
                continue;
            }
            let Some((sy, surf)) = self.surface_at(cx, cz) else { continue };
            if surf != block::GRASS || self.sky_light_at(IVec3::new(cx, sy + 1, cz)) < 15 {
                continue;
            }
            used.push((cx, cz));
            self.entities
                .spawn_mob(Vec3::new(cx as f32 + 0.5, (sy + 1) as f32, cz as f32 + 0.5), species);
            spawned += 1;
        }
        spawned
    }

    /// One natural-spawn attempt around the player (P16): at night, a HOSTILE on any solid surface in
    /// the dark (block-light 0, MC 1.18+); by day, a biome-appropriate PASSIVE PACK on lit grass. Each
    /// category has its own ring + cap. `day` is the day factor (0 = night, 1 = full day).
    pub fn try_spawn(&mut self, day: f32, player_pos: Vec3) -> bool {
        use crate::entity::Species;
        const HOSTILE: [Species; 6] = [
            Species::Zombie,
            Species::Skeleton,
            Species::Creeper,
            Species::Spider,
            Species::Slime, // P18 (spawn_mob makes a LARGE slime by default)
            Species::Enderman,
        ];
        let night = day < 0.35 && self.difficulty.spawns_hostiles();
        let daytime = day > 0.6;
        if !night && !daytime {
            return false; // dawn/dusk: no spawns
        }
        let r = self.spawn_rand();
        let (rmin, rmax) = if night { (HOSTILE_MIN, HOSTILE_MAX) } else { (PASSIVE_MIN, PASSIVE_MAX) };
        let angle = (r & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
        let dist = rmin + ((r >> 16) & 0xffff) as f32 / 65535.0 * (rmax - rmin);
        let wx = (player_pos.x + angle.cos() * dist).floor() as i32;
        let wz = (player_pos.z + angle.sin() * dist).floor() as i32;
        let Some((sy, surf)) = self.surface_at(wx, wz) else {
            return false;
        };
        let feet = IVec3::new(wx, sy + 1, wz);

        if night {
            // Hostiles on any solid surface, only where it's genuinely dark (no nearby torch/glowstone).
            if self.entities.hostile_count() >= HOSTILE_CAP || self.block_light_at(feet) > 0 {
                return false;
            }
            let pick = ((r >> 32) as usize) % HOSTILE.len();
            self.entities
                .spawn_mob(Vec3::new(wx as f32 + 0.5, (sy + 1) as f32, wz as f32 + 0.5), HOSTILE[pick]);
            return true;
        }

        // Daytime passive pack on lit grass, species drawn from the biome's pool.
        if surf != block::GRASS || self.entities.passive_count() >= PASSIVE_CAP {
            return false;
        }
        if self.sky_light_at(feet) < 15 {
            return false; // shaded/cave grass — animals need open sky
        }
        let pool = self.worldgen.passive_pool(wx, wz);
        if pool.is_empty() {
            return false;
        }
        let species = pool[((r >> 32) as usize) % pool.len()];
        self.spawn_passive_pack(species, wx, wz) > 0
    }

    /// Furnace state at `pos`, creating an empty one (the player just opened it).
    pub fn furnace_mut(&mut self, pos: IVec3) -> &mut FurnaceState {
        self.furnaces.entry(pos).or_default()
    }

    /// Read-only furnace state at `pos` (None if never opened / already empty).
    pub fn furnace(&self, pos: IVec3) -> Option<&FurnaceState> {
        self.furnaces.get(&pos)
    }

    /// Snapshot non-empty furnaces for saving (M24 persistence).
    pub fn furnaces_to_save(&self) -> Vec<persistence::FurnaceSave> {
        self.furnaces
            .iter()
            .filter(|(_, f)| !f.is_empty())
            .map(|(&pos, f)| persistence::FurnaceSave {
                pos,
                input: f.input,
                fuel: f.fuel,
                output: f.output,
                burn_remaining: f.burn_remaining,
                burn_max: f.burn_max,
                cook_progress: f.cook_progress,
                cook_item: f.cook_item,
            })
            .collect()
    }

    /// Restore furnaces loaded from disk into the live map (M24 persistence).
    pub fn restore_furnaces(&mut self, saved: Vec<persistence::FurnaceSave>) {
        for f in saved {
            self.furnaces.insert(
                f.pos,
                FurnaceState {
                    input: f.input,
                    fuel: f.fuel,
                    output: f.output,
                    burn_remaining: f.burn_remaining,
                    burn_max: f.burn_max,
                    cook_progress: f.cook_progress,
                    cook_item: f.cook_item,
                },
            );
        }
    }

    /// Chest container at `pos`, creating an empty 27-slot one (the player just opened/placed it).
    pub fn chest_mut(&mut self, pos: IVec3) -> &mut crate::container::Container {
        self.chests
            .entry(pos)
            .or_insert_with(|| crate::container::Container::new(crate::container::CHEST_SLOTS))
    }

    /// Read-only chest container at `pos` (None if never opened / already empty).
    pub fn chest(&self, pos: IVec3) -> Option<&crate::container::Container> {
        self.chests.get(&pos)
    }

    /// Snapshot non-empty chests for saving.
    pub fn chests_to_save(&self) -> Vec<persistence::ChestSave> {
        self.chests
            .iter()
            .filter(|(_, c)| !c.is_empty())
            .map(|(&pos, c)| persistence::ChestSave {
                pos,
                slots: c.slots.clone(),
            })
            .collect()
    }

    /// Restore chests loaded from disk into the live map.
    pub fn restore_chests(&mut self, saved: Vec<persistence::ChestSave>) {
        for c in saved {
            let mut container = crate::container::Container::new(crate::container::CHEST_SLOTS);
            for (i, s) in c.slots.into_iter().enumerate() {
                if i < container.slots.len() {
                    container.slots[i] = s;
                }
            }
            self.chests.insert(c.pos, container);
        }
    }

    /// Mirror the world difficulty (gates hostile spawning in `try_spawn`).
    pub fn set_difficulty(&mut self, d: crate::rules::Difficulty) {
        self.difficulty = d;
    }

    /// Remove hostile mobs immediately (called once when the world switches to Peaceful).
    pub fn despawn_hostiles(&mut self) {
        self.entities.despawn_hostiles();
    }

    pub fn entity_count(&self) -> usize {
        self.entities.count()
    }

    /// One-line mob AI-state tally (headless verification).
    pub fn mob_ai_summary(&self) -> String {
        self.entities.ai_summary()
    }

    /// Passive/hostile mob tally (headless spawn verification).
    pub fn mob_species_summary(&self) -> String {
        self.entities.species_summary()
    }
    pub fn despawn_all_mobs(&mut self) {
        self.entities.clear_mobs();
    }

    /// Distance to the nearest mob the ray hits within `reach` (None = no mob in the way).
    pub fn nearest_mob_hit(&self, origin: Vec3, dir: Vec3, reach: f32) -> Option<f32> {
        self.entities.nearest_mob_hit(origin, dir, reach)
    }

    /// P17 feed: ray-find a breedable adult and (if `food` is its breeding food) flag it in love.
    /// Returns true so the caller consumes one food.
    pub fn try_feed_mob(&mut self, origin: Vec3, dir: Vec3, reach: f32, food: crate::item::ItemId) -> bool {
        self.entities.try_feed(origin, dir, reach, food)
    }

    /// Melee the nearest mob the ray hits within `reach` (damage + knockback + hurt-flash). `crit`
    /// lengthens the flash, `sweep = Some(dmg)` adds the 1.9 sword AoE, `kb_mult` scales knockback
    /// (a sprint-attack passes >1).
    pub fn attack_nearest(
        &mut self,
        origin: Vec3,
        dir: Vec3,
        reach: f32,
        damage: f32,
        crit: bool,
        sweep: Option<f32>,
        kb_mult: f32,
    ) -> bool {
        self.entities.attack(origin, dir, reach, damage, crit, sweep, kb_mult)
    }

    /// Debug: flash every mob (headless hurt-flash verification).
    pub fn flash_mobs(&mut self, amount: f32) {
        self.entities.flash_all(amount);
    }

    /// Build one GPU mesh for all entities this frame (mobs + items), lit by the chunk pass.
    pub fn build_entity_mesh(&self, gpu: &Gpu, renderer: &ChunkRenderer) -> Option<GpuMesh> {
        let data = self.entities.build_mesh();
        if data.is_empty() {
            None
        } else {
            Some(renderer.upload_mesh(gpu, &data))
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Biome name at a world column (F3 debug overlay).
    pub fn biome_name_at(&self, wx: i32, wz: i32) -> &'static str {
        self.worldgen.biome_name(wx, wz)
    }

    /// Current ray-traced lighting mode name (F3 debug overlay).
    pub fn rtx_mode_name(&self) -> &'static str {
        self.volume.rtx_mode_name()
    }

    pub fn volume_bind_group(&self) -> &wgpu::BindGroup {
        self.volume.bind_group()
    }

    /// Fully populate the voxel volume (headless screenshots).
    pub fn prime_volume(&mut self, gpu: &Gpu, camera_pos: Vec3) {
        let pc = Self::center_of(camera_pos);
        self.volume.prime(gpu, &self.world, pc);
    }

    /// Cycle ray-traced lighting: off -> shadows -> shadows+GI. Returns the new mode name.
    pub fn cycle_rtx(&mut self) -> &'static str {
        self.volume.cycle_rtx();
        self.volume.rtx_mode_name()
    }

    /// Raise the GI hemisphere ray count (offscreen screenshots want many more than interactive).
    pub fn set_rtx_quality(&mut self, rays: u32) {
        self.volume.set_gi_rays(rays);
    }

    /// Water depth-clarity smoothing radius in blocks (debug knob; 0 disables smoothing).
    pub fn set_water_smooth(&mut self, r: f32) {
        self.volume.set_water_smooth(r);
    }

    /// Write the edited chunks and the level header to disk.
    pub fn save(&self, level: &Level) {
        let dir = persistence::save_dir();
        let chunks: Vec<(IVec3, &Chunk)> =
            self.saved.iter().map(|(p, c)| (*p, c.as_ref())).collect();
        if let Err(e) = persistence::save_chunks(&dir, &chunks) {
            log::error!("failed to save chunks: {e}");
        }
        if let Err(e) = persistence::save_level(&dir, level) {
            log::error!("failed to save level: {e}");
        }
    }

    pub fn visible_meshes(&self, frustum: &Frustum) -> Vec<&GpuMesh> {
        let mut out = Vec::new();
        for (pos, mesh) in &self.meshes {
            if let Some(mesh) = mesh {
                let origin = world::chunk_origin(*pos);
                let min = origin.as_vec3();
                let max = min + Vec3::splat(CHUNK_SIZE_I as f32);
                if frustum.aabb_visible(min, max) {
                    out.push(mesh);
                }
            }
        }
        out
    }

    /// Every loaded non-empty mesh (no frustum cull). Used for the hardware-RT TLAS, since
    /// off-screen geometry still casts shadows and is hit by GI rays. (M33-G5)
    pub fn all_meshes(&self) -> Vec<&GpuMesh> {
        self.meshes.values().filter_map(|m| m.as_ref()).collect()
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.world.chunks.len()
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.values().filter(|m| m.is_some()).count()
    }
}

#[cfg(test)]
mod furnace_tests {
    use super::*;
    use crate::item::ItemStack;
    use std::collections::HashMap;

    #[test]
    fn u9_growth_decisions() {
        // rnd=0 passes every probability gate (0 < threshold), so growth fires deterministically.
        // Amethyst bud stages advance in place.
        assert_eq!(
            grow_decision(block::SMALL_AMETHYST_BUD, 0, block::STONE, block::STONE, 0),
            Some((IVec3::ZERO, block::MEDIUM_AMETHYST_BUD))
        );
        assert_eq!(
            grow_decision(block::LARGE_AMETHYST_BUD, 0, block::STONE, block::STONE, 0),
            Some((IVec3::ZERO, block::AMETHYST_CLUSTER))
        );
        // Cave vine grows a new tip below only into air and under the length cap.
        assert_eq!(
            grow_decision(block::CAVE_VINE, 0, block::AIR, block::DEEPSLATE, 3),
            Some((IVec3::NEG_Y, block::CAVE_VINE_BERRIES)) // rnd=0 -> berried tip
        );
        assert_eq!(grow_decision(block::CAVE_VINE, 0, block::STONE, block::STONE, 3), None); // blocked below
        assert_eq!(grow_decision(block::CAVE_VINE, 0, block::AIR, block::STONE, 14), None); // length cap
        // A high rnd fails the probability gate (no growth this tick).
        assert_eq!(grow_decision(block::SMALL_AMETHYST_BUD, 99, block::STONE, block::STONE, 0), None);
        // Pointed dripstone only extends a hanging stalactite (dripstone above, air below).
        assert_eq!(
            grow_decision(block::POINTED_DRIPSTONE, 0, block::AIR, block::DRIPSTONE_BLOCK, 0),
            Some((IVec3::NEG_Y, block::POINTED_DRIPSTONE))
        );
        assert_eq!(grow_decision(block::POINTED_DRIPSTONE, 0, block::AIR, block::STONE, 0), None);
        assert!(random_tickable(block::BUDDING_AMETHYST) && !random_tickable(block::STONE));
    }

    #[test]
    fn u10_sculk_convertible() {
        // Exposed solid rock/ground converts to sculk; air/fluids/non-cubes + the sculk family don't.
        assert!(sculk_convertible(block::STONE));
        assert!(sculk_convertible(block::DEEPSLATE) && sculk_convertible(block::DIRT));
        assert!(sculk_convertible(block::GRANITE) && sculk_convertible(block::COBBLED_DEEPSLATE));
        assert!(!sculk_convertible(block::AIR) && !sculk_convertible(block::WATER));
        assert!(!sculk_convertible(block::BEDROCK) && !sculk_convertible(block::REINFORCED_DEEPSLATE));
        assert!(!sculk_convertible(block::SCULK) && !sculk_convertible(block::SCULK_CATALYST));
        assert!(!sculk_convertible(block::POINTED_DRIPSTONE)); // non-cube billboard
    }

    #[test]
    fn sky_light_open_vs_roofed() {
        // Open column → full sky-light; a solid block anywhere above → dark.
        let open: HashMap<IVec3, block::BlockId> = HashMap::new();
        assert_eq!(sky_light(IVec3::new(0, 64, 0), 256, &|p| *open.get(&p).unwrap_or(&block::AIR)), 15);
        let mut roofed = HashMap::new();
        roofed.insert(IVec3::new(0, 70, 0), block::STONE);
        assert_eq!(sky_light(IVec3::new(0, 64, 0), 256, &|p| *roofed.get(&p).unwrap_or(&block::AIR)), 0);
    }

    #[test]
    fn block_light_flood_and_occlusion() {
        // A torch (emission 14) 3 cells away → 14 - 3 = 11; no emitter → 0.
        let mut w = HashMap::new();
        w.insert(IVec3::new(3, 64, 0), block::TORCH);
        assert_eq!(block_light(IVec3::new(0, 64, 0), &|p| *w.get(&p).unwrap_or(&block::AIR)), 11);
        let none: HashMap<IVec3, block::BlockId> = HashMap::new();
        assert_eq!(block_light(IVec3::new(0, 64, 0), &|p| *none.get(&p).unwrap_or(&block::AIR)), 0);
        // A solid wall between the torch and the target forces a longer detour → dimmer than 11.
        let mut walled = HashMap::new();
        walled.insert(IVec3::new(3, 64, 0), block::TORCH);
        for dy in -2..=2 {
            for dz in -2..=2 {
                walled.insert(IVec3::new(1, 64 + dy, dz), block::STONE);
            }
        }
        assert!(
            block_light(IVec3::new(0, 64, 0), &|p| *walled.get(&p).unwrap_or(&block::AIR)) < 11,
            "a wall attenuates the torch light"
        );
    }

    /// One full smelt cycle: ore + fuel → an ingot, the input drops by one, fuel keeps burning.
    #[test]
    fn smelts_one_item_and_consumes_fuel() {
        let mut f = FurnaceState {
            input: Some(ItemStack::new(block::IRON_ORE, 2)),
            fuel: Some(ItemStack::new(block::PLANKS, 1)),
            ..Default::default()
        };
        let dt = 0.1;
        let mut t = 0.0;
        while t < smelting::SMELT_TIME + 0.5 {
            step_furnace(&mut f, dt);
            t += dt;
        }
        assert_eq!(f.output.map(|s| (s.item, s.count)), Some((item::IRON_INGOT, 1)));
        assert_eq!(f.input.map(|s| s.count), Some(1));
        assert!(f.burn_remaining > 0.0); // a plank (9s) still has burn left after one 6s smelt
    }

    /// Swapping the input mid-cook must not let banked progress finish a different recipe instantly.
    #[test]
    fn swapping_input_resets_cook_progress() {
        let mut f = FurnaceState {
            input: Some(ItemStack::new(block::IRON_ORE, 1)),
            fuel: Some(ItemStack::new(block::PLANKS, 2)),
            ..Default::default()
        };
        while f.cook_progress < smelting::SMELT_TIME - 0.5 {
            step_furnace(&mut f, 0.1);
        }
        assert!(f.output.is_none(), "iron should not be done yet");
        // Swap to a different smeltable: progress must reset so nothing pops out next tick.
        f.input = Some(ItemStack::new(block::COBBLESTONE, 1));
        step_furnace(&mut f, 0.1);
        assert!(f.output.is_none(), "swapped input must not smelt instantly");
        assert!(f.cook_progress < 0.5, "progress should reset on input swap");
    }

    #[test]
    fn explosion_damage_falls_off() {
        assert!(explosion_damage(0.0, 3.0) > 0.0);
        assert!(explosion_damage(0.0, 3.0) > explosion_damage(3.0, 3.0));
        assert_eq!(explosion_damage(10.0, 3.0), 0.0); // beyond reach (1.6 * 3 = 4.8)
    }

    /// No fuel → nothing smelts, and the input is untouched.
    #[test]
    fn idle_without_fuel_does_not_smelt() {
        let mut f = FurnaceState {
            input: Some(ItemStack::new(block::IRON_ORE, 1)),
            ..Default::default()
        };
        for _ in 0..200 {
            step_furnace(&mut f, 0.1);
        }
        assert!(f.output.is_none());
        assert_eq!(f.input.map(|s| s.count), Some(1));
    }
}
