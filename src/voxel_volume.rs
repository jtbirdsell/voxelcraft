//! GPU-resident voxel material volume for ray-traced lighting.
//!
//! A 256^3 `r16uint` 3D texture holds the opaque block id (0 = air/transparent) for the region
//! around the player. Because 256 is a multiple of the 32-block chunk size, every chunk maps to
//! exactly one non-wrapping 32^3 sub-box, so the volume is a toroidal ring buffer: a world
//! voxel's texel is `worldPos mod 256`, and as the player moves, chunks scrolling in overwrite
//! the texels of chunks scrolling out. Fragment shaders DDA-march this for sun shadows, and —
//! because each texel now carries the block id (not just occupancy) — for ray-traced ambient
//! occlusion and one-bounce colored global illumination (the bounce reads the hit block's color).

use glam::IVec3;
use rustc_hash::FxHashMap;

use crate::block;
use crate::gpu::Gpu;
use crate::world::{Chunk, World, CHUNK_SIZE, CHUNK_SIZE_I};

pub const VOL_SIZE: i32 = 256;
pub const VOL_CHUNKS: i32 = VOL_SIZE / CHUNK_SIZE_I; // 8

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumeUniform {
    /// World-block coordinates of the volume's minimum corner.
    origin: [i32; 4],
    /// (size, rtx_mode, gi_rays, _) — rtx_mode: 0 off, 1 shadows, 2 shadows+GI.
    params: [u32; 4],
    /// (gi_dist, gi_strength, sky_boost, _)
    paramsf: [f32; 4],
}

/// Ray-traced lighting mode, cycled by the `R` key.
pub const RTX_OFF: u32 = 0;
pub const RTX_SHADOWS: u32 = 1;
pub const RTX_GI: u32 = 2;

pub struct VoxelVolume {
    texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    origin_chunk: IVec3,
    occupant: FxHashMap<(i32, i32, i32), IVec3>,
    rtx_mode: u32,
    /// Hemisphere ray count for GI (low interactive, high for offscreen screenshots).
    gi_rays: u32,
    gi_dist: f32,
    gi_strength: f32,
    sky_boost: f32,
    upload_budget: usize,
    /// Reused padded scratch (row stride = 256 texels) for texture writes.
    scratch: Vec<u16>,
}

impl VoxelVolume {
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn new(gpu: &Gpu, bgl: &wgpu::BindGroupLayout) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel-volume"),
            size: wgpu::Extent3d {
                width: VOL_SIZE as u32,
                height: VOL_SIZE as u32,
                depth_or_array_layers: VOL_SIZE as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume-uniform"),
            size: std::mem::size_of::<VolumeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume-bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        Self {
            texture,
            uniform,
            bind_group,
            origin_chunk: IVec3::new(i32::MIN, 0, i32::MIN),
            occupant: FxHashMap::default(),
            rtx_mode: RTX_GI,
            gi_rays: 4,
            gi_dist: 22.0,
            gi_strength: 1.0,
            sky_boost: 0.55,
            upload_budget: 24,
            scratch: vec![0u16; (VOL_SIZE as usize) * CHUNK_SIZE * CHUNK_SIZE],
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Human-readable name of the current ray-traced lighting mode.
    pub fn rtx_mode_name(&self) -> &'static str {
        match self.rtx_mode {
            RTX_OFF => "off",
            RTX_SHADOWS => "shadows",
            _ => "shadows + GI",
        }
    }

    /// Cycle off -> shadows -> shadows+GI; returns the new mode.
    pub fn cycle_rtx(&mut self) -> u32 {
        self.rtx_mode = (self.rtx_mode + 1) % 3;
        self.rtx_mode
    }

    /// Set the hemisphere ray count used for GI (offscreen screenshots crank this up).
    pub fn set_gi_rays(&mut self, rays: u32) {
        self.gi_rays = rays;
    }

    /// Mark a chunk so it re-uploads (e.g. after an edit), if currently in the volume.
    pub fn invalidate(&mut self, pos: IVec3) {
        let key = (pos.x.rem_euclid(VOL_CHUNKS), pos.y, pos.z.rem_euclid(VOL_CHUNKS));
        if self.occupant.get(&key) == Some(&pos) {
            self.occupant.remove(&key);
        }
    }

    /// Re-center on the player and upload a budget of in-range chunks per frame.
    pub fn update(&mut self, gpu: &Gpu, world: &World, player_chunk: IVec3) {
        self.origin_chunk = IVec3::new(player_chunk.x - VOL_CHUNKS / 2, 0, player_chunk.z - VOL_CHUNKS / 2);
        let origin_world = self.origin_chunk * CHUNK_SIZE_I;

        let uni = VolumeUniform {
            origin: [origin_world.x, 0, origin_world.z, 0],
            params: [VOL_SIZE as u32, self.rtx_mode, self.gi_rays, 0],
            paramsf: [self.gi_dist, self.gi_strength, self.sky_boost, 0.0],
        };
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uni));

        let (ox, oz) = (self.origin_chunk.x, self.origin_chunk.z);
        let mut budget = self.upload_budget;
        'outer: for cz in oz..oz + VOL_CHUNKS {
            for cx in ox..ox + VOL_CHUNKS {
                for cy in 0..VOL_CHUNKS {
                    let pos = IVec3::new(cx, cy, cz);
                    let key = (cx.rem_euclid(VOL_CHUNKS), cy, cz.rem_euclid(VOL_CHUNKS));
                    if self.occupant.get(&key) == Some(&pos) {
                        continue;
                    }
                    if let Some(chunk) = world.get(pos) {
                        self.upload_chunk(gpu, pos, chunk);
                        self.occupant.insert(key, pos);
                        budget -= 1;
                        if budget == 0 {
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    /// Upload all in-range chunks at once (used for headless screenshots).
    pub fn prime(&mut self, gpu: &Gpu, world: &World, player_chunk: IVec3) {
        let saved = self.upload_budget;
        self.upload_budget = (VOL_CHUNKS * VOL_CHUNKS * VOL_CHUNKS) as usize;
        self.update(gpu, world, player_chunk);
        self.upload_budget = saved;
    }

    fn upload_chunk(&mut self, gpu: &Gpu, pos: IVec3, chunk: &Chunk) {
        let tx = (pos.x * CHUNK_SIZE_I).rem_euclid(VOL_SIZE) as u32;
        let ty = (pos.y * CHUNK_SIZE_I).rem_euclid(VOL_SIZE) as u32;
        let tz = (pos.z * CHUNK_SIZE_I).rem_euclid(VOL_SIZE) as u32;

        // Texture data order is x (fastest), then y, then z; row stride 256 texels (= 512 bytes,
        // already a multiple of COPY_BYTES_PER_ROW_ALIGNMENT). Opaque blocks store their id so the
        // tracer can read material color; air and water (non-opaque) store 0 and cast no shadow.
        let row_stride = VOL_SIZE as usize;
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                let base = z * row_stride * CHUNK_SIZE + y * row_stride;
                for x in 0..CHUNK_SIZE {
                    let id = chunk.get(x, y, z);
                    self.scratch[base + x] = if block::is_opaque(id) { id } else { 0 };
                }
            }
        }

        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: tx, y: ty, z: tz },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.scratch),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(VOL_SIZE as u32 * 2),
                rows_per_image: Some(CHUNK_SIZE as u32),
            },
            wgpu::Extent3d {
                width: CHUNK_SIZE as u32,
                height: CHUNK_SIZE as u32,
                depth_or_array_layers: CHUNK_SIZE as u32,
            },
        );
    }
}
