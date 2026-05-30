//! Streaming manager: keeps chunks generated and meshed around the camera within a render
//! distance, frustum-culls them for drawing, and unloads distant ones. Single-threaded with a
//! per-frame work budget for now; a worker pool replaces the budget next.

use glam::{IVec3, Vec3};
use rustc_hash::FxHashMap;

use crate::frustum::Frustum;
use crate::gpu::Gpu;
use crate::mesher;
use crate::renderer::{ChunkRenderer, GpuMesh};
use crate::world::{self, World, CHUNK_SIZE_I, WORLD_HEIGHT_CHUNKS};

pub struct Game {
    world: World,
    /// Built meshes per chunk; `None` means "meshed but empty" (no geometry).
    meshes: FxHashMap<IVec3, Option<GpuMesh>>,
    render_distance: i32,
    center: IVec3,
    /// XZ offsets within (render_distance + 1), sorted nearest-first.
    offsets: Vec<(i32, i32, i32)>, // (dx, dz, dist2)
    gen_budget: usize,
    mesh_budget: usize,
}

impl Game {
    pub fn new(seed: u64, render_distance: i32) -> Self {
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

        Self {
            world: World::new(seed),
            meshes: FxHashMap::default(),
            render_distance,
            center: IVec3::new(i32::MIN, 0, i32::MIN),
            offsets,
            gen_budget: 96,
            mesh_budget: 48,
        }
    }

    fn center_of(camera_pos: Vec3) -> IVec3 {
        world::chunk_of(IVec3::new(
            camera_pos.x.floor() as i32,
            camera_pos.y.floor() as i32,
            camera_pos.z.floor() as i32,
        ))
    }

    fn neighbors_ready(&self, pos: IVec3) -> bool {
        // Horizontal face neighbors must exist.
        for off in [IVec3::X, -IVec3::X, IVec3::Z, -IVec3::Z] {
            if !self.world.is_generated(pos + off) {
                return false;
            }
        }
        // Vertical neighbors only when within the world's vertical bounds.
        if pos.y - 1 >= 0 && !self.world.is_generated(pos - IVec3::Y) {
            return false;
        }
        if pos.y + 1 < WORLD_HEIGHT_CHUNKS && !self.world.is_generated(pos + IVec3::Y) {
            return false;
        }
        true
    }

    fn build_and_upload(&mut self, gpu: &Gpu, renderer: &ChunkRenderer, pos: IVec3) {
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

    /// Stream chunks toward being generated + meshed around the camera, within a per-call budget.
    pub fn update(&mut self, gpu: &Gpu, renderer: &ChunkRenderer, camera_pos: Vec3) {
        self.center = Self::center_of(camera_pos);
        let (cx, cz) = (self.center.x, self.center.z);
        let r = self.render_distance;

        // 1. Generation (radius r+1, nearest-first).
        let mut gen_left = self.gen_budget;
        'generate: for &(dx, dz, _) in &self.offsets {
            for cy in 0..WORLD_HEIGHT_CHUNKS {
                let pos = IVec3::new(cx + dx, cy, cz + dz);
                if !self.world.is_generated(pos) {
                    self.world.ensure_generated(pos);
                    gen_left -= 1;
                    if gen_left == 0 {
                        break 'generate;
                    }
                }
            }
        }

        // 2. Meshing (radius r, nearest-first, only where neighbors are ready).
        let mut mesh_left = self.mesh_budget;
        let mut to_mesh: Vec<IVec3> = Vec::new();
        'find: for &(dx, dz, dist2) in &self.offsets {
            if dist2 > r * r {
                break;
            }
            for cy in 0..WORLD_HEIGHT_CHUNKS {
                let pos = IVec3::new(cx + dx, cy, cz + dz);
                if !self.meshes.contains_key(&pos) && self.neighbors_ready(pos) {
                    to_mesh.push(pos);
                    mesh_left -= 1;
                    if mesh_left == 0 {
                        break 'find;
                    }
                }
            }
        }
        for pos in to_mesh {
            self.build_and_upload(gpu, renderer, pos);
        }

        // 3. Unload chunks well beyond the render distance.
        let keep = r + 2;
        let center = self.center;
        self.meshes.retain(|pos, _| {
            (pos.x - center.x).abs() <= keep && (pos.z - center.z).abs() <= keep
        });
        self.world.chunks.retain(|pos, _| {
            (pos.x - center.x).abs() <= keep + 1 && (pos.z - center.z).abs() <= keep + 1
        });
    }

    /// Fully generate and mesh the current radius synchronously (used for headless screenshots).
    pub fn load_all_blocking(&mut self, gpu: &Gpu, renderer: &ChunkRenderer, camera_pos: Vec3) {
        self.center = Self::center_of(camera_pos);
        let (cx, cz) = (self.center.x, self.center.z);
        let r = self.render_distance;

        for &(dx, dz, _) in &self.offsets {
            for cy in 0..WORLD_HEIGHT_CHUNKS {
                self.world
                    .ensure_generated(IVec3::new(cx + dx, cy, cz + dz));
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
                self.build_and_upload(gpu, renderer, pos);
            }
        }
    }

    /// Collect references to all loaded, non-empty meshes whose chunk AABB is in the frustum.
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
