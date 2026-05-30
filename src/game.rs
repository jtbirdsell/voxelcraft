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

        // Feed the GPU voxel volume (ray-traced lighting) around the player.
        self.volume.update(gpu, &self.world, self.center);

        collected
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

    /// Set a block at a world position and synchronously re-mesh the affected chunk(s).
    /// Returns true if a change was made.
    pub fn set_block(
        &mut self,
        gpu: &Gpu,
        renderer: &ChunkRenderer,
        wp: IVec3,
        id: crate::block::BlockId,
    ) -> bool {
        let cpos = world::chunk_of(wp);
        let origin = world::chunk_origin(cpos);
        let (lx, ly, lz) = (
            (wp.x - origin.x) as usize,
            (wp.y - origin.y) as usize,
            (wp.z - origin.z) as usize,
        );

        let old = {
            let Some(arc) = self.world.chunks.get_mut(&cpos) else {
                return false;
            };
            let chunk = Arc::make_mut(arc);
            let old = chunk.get(lx, ly, lz);
            if old == id {
                return false;
            }
            chunk.set(lx, ly, lz, id);
            old
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

    pub fn spawn_mob(&mut self, pos: Vec3) {
        self.entities.spawn_mob(pos);
    }

    pub fn spawn_item(&mut self, pos: Vec3, stack: crate::item::ItemStack) {
        self.entities.spawn_item(pos, stack);
    }

    pub fn spawn_xp(&mut self, pos: Vec3, amount: u32) {
        self.entities.spawn_xp(pos, amount);
    }

    /// Furnace state at `pos`, creating an empty one (the player just opened it).
    pub fn furnace_mut(&mut self, pos: IVec3) -> &mut FurnaceState {
        self.furnaces.entry(pos).or_default()
    }

    /// Read-only furnace state at `pos` (None if never opened / already empty).
    pub fn furnace(&self, pos: IVec3) -> Option<&FurnaceState> {
        self.furnaces.get(&pos)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.count()
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
