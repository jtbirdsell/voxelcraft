//! Streaming manager: keeps chunks generated and meshed around the camera within a render
//! distance using a worker pool, frustum-culls them for drawing, and unloads distant ones.
//!
//! Threading model: workers generate chunks and build meshes from immutable snapshots; the main
//! thread owns the world map (the only mutator) and performs GPU uploads (budgeted per frame).

use std::collections::VecDeque;
use std::sync::Arc;

use glam::{IVec3, Vec3};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::frustum::Frustum;
use crate::gpu::Gpu;
use crate::mesher::{self, MeshData};
use crate::renderer::{ChunkRenderer, GpuMesh};
use crate::persistence::{self, Level};
use crate::worker::{Job, JobResult, WorkerPool};
use crate::worldgen::Worldgen;
use crate::world::{self, Chunk, World, CHUNK_SIZE_I, WORLD_HEIGHT_CHUNKS};

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

    pool: WorkerPool,
    pending_gen: FxHashSet<IVec3>,
    pending_mesh: FxHashSet<IVec3>,
    ready_meshes: VecDeque<(IVec3, u32, MeshData)>,
    /// Mesh version per chunk; bumped on edit so stale in-flight mesh jobs are discarded.
    versions: FxHashMap<IVec3, u32>,

    max_inflight_gen: usize,
    max_inflight_mesh: usize,
    upload_budget: usize,
}

impl Game {
    pub fn new(
        seed: u64,
        render_distance: i32,
        saved_chunks: FxHashMap<IVec3, Chunk>,
    ) -> Self {
        let saved: FxHashMap<IVec3, Arc<Chunk>> = saved_chunks
            .into_iter()
            .map(|(pos, chunk)| (pos, Arc::new(chunk)))
            .collect();
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

    pub fn update(&mut self, gpu: &Gpu, renderer: &ChunkRenderer, camera_pos: Vec3) {
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

        {
            let Some(arc) = self.world.chunks.get_mut(&cpos) else {
                return false;
            };
            let chunk = Arc::make_mut(arc);
            if chunk.get(lx, ly, lz) == id {
                return false;
            }
            chunk.set(lx, ly, lz, id);
        }

        // Persist the edited chunk (kept resident in `saved`).
        if let Some(arc) = self.world.chunks.get(&cpos) {
            self.saved.insert(cpos, arc.clone());
            self.dirty = true;
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

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn edited_chunk_count(&self) -> usize {
        self.saved.len()
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
