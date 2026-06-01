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

/// Natural-spawn tuning (M31).
const SPAWN_INTERVAL: f32 = 2.0; // seconds between spawn attempts
const MOB_CAP: usize = 16; // stop spawning past this many live mobs
const SPAWN_MIN: f32 = 8.0; // spawn ring around the player (blocks)
const SPAWN_MAX: f32 = 24.0;

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

        // Natural spawning: one gated attempt per interval around the player (M31).
        self.spawn_timer += dt;
        if self.spawn_timer >= SPAWN_INTERVAL {
            self.spawn_timer -= SPAWN_INTERVAL;
            self.try_spawn(day, camera_pos);
        }

        // Update mobs and dropped items (AI + physics). The collision closure borrows only the
        // chunk map, leaving `self.entities` free to mutate.
        let chunks = &self.world.chunks;
        let collected = self.entities.update(dt, camera_pos, |wp| {
            let cpos = world::chunk_of(wp);
            match chunks.get(&cpos) {
                Some(c) => {
                    let o = world::chunk_origin(cpos);
                    block::is_solid(c.get(
                        (wp.x - o.x) as usize,
                        (wp.y - o.y) as usize,
                        (wp.z - o.z) as usize,
                    ))
                }
                None => false,
            }
        });

        // Apply any creeper explosions: carve the crater, then add radial blast damage to the player.
        let mut collected = collected;
        for (center, radius) in std::mem::take(&mut collected.explosions) {
            collected.player_damage += self.apply_explosion(gpu, renderer, center, radius, camera_pos);
        }

        // Feed the GPU voxel volume (ray-traced lighting) around the player.
        self.volume.update(gpu, &self.world, self.center);

        collected
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

    pub fn spawn_mob(&mut self, pos: Vec3, species: crate::entity::Species) {
        self.entities.spawn_mob(pos, species);
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

    /// One natural-spawn attempt around the player: hostiles at night, passives by day on grass.
    /// `day` is the day factor (0 = night, 1 = full day). Capped + only on solid footing.
    pub fn try_spawn(&mut self, day: f32, player_pos: Vec3) -> bool {
        if self.entities.mob_count() >= MOB_CAP {
            return false;
        }
        use crate::entity::Species;
        const HOSTILE: [Species; 4] = [
            Species::Zombie,
            Species::Skeleton,
            Species::Creeper,
            Species::Spider,
        ];
        const PASSIVE: [Species; 4] = [
            Species::Cow,
            Species::Pig,
            Species::Sheep,
            Species::Chicken,
        ];
        let r = self.spawn_rand();
        let angle = (r & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
        let dist = SPAWN_MIN + ((r >> 16) & 0xffff) as f32 / 65535.0 * (SPAWN_MAX - SPAWN_MIN);
        let wx = (player_pos.x + angle.cos() * dist).floor() as i32;
        let wz = (player_pos.z + angle.sin() * dist).floor() as i32;
        let Some((sy, surf)) = self.surface_at(wx, wz) else {
            return false;
        };
        let pos = Vec3::new(wx as f32 + 0.5, (sy + 1) as f32, wz as f32 + 0.5);
        let pick = ((r >> 32) & 3) as usize;
        if day < 0.35 && self.difficulty.spawns_hostiles() {
            self.entities.spawn_mob(pos, HOSTILE[pick]);
            true
        } else if day > 0.6 && surf == block::GRASS {
            self.entities.spawn_mob(pos, PASSIVE[pick]);
            true
        } else {
            false
        }
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

    /// Passive/hostile mob tally + live mob count (headless spawn verification).
    pub fn mob_species_summary(&self) -> String {
        self.entities.species_summary()
    }
    pub fn mob_count(&self) -> usize {
        self.entities.mob_count()
    }
    pub fn despawn_all_mobs(&mut self) {
        self.entities.clear_mobs();
    }

    /// Distance to the nearest mob the ray hits within `reach` (None = no mob in the way).
    pub fn nearest_mob_hit(&self, origin: Vec3, dir: Vec3, reach: f32) -> Option<f32> {
        self.entities.nearest_mob_hit(origin, dir, reach)
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
