//! Hardware ray-tracing spike (`VOXELCRAFT_RT_SPIKE=path.png`). A self-contained, isolated proof that
//! the RTX 4090's RT cores are reachable through wgpu — it touches none of the main render path.
//!
//! It requests a Vulkan device with `EXPERIMENTAL_RAY_QUERY`, bakes a small scene (a ground plane plus
//! three boxes) into one BLAS, wraps it in a TLAS, and runs a compute shader (`rt_spike.wgsl`) that
//! casts a hardware primary ray and a hardware shadow ray per pixel. The result is read back to a PNG.
//! Visible hard shadows == the acceleration-structure build + `rayQuery` work on real silicon, which is
//! the go/no-go for migrating the software DDA tracer (`rtx_common.wgsl`) onto the RT cores.

use std::iter;

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, vec3};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    eye: [f32; 4],
    ll: [f32; 4],
    horiz: [f32; 4],
    vert: [f32; 4],
    sun: [f32; 4],
    dims: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Tri {
    normal: [f32; 4],
    albedo: [f32; 4],
}

/// CPU-side scene: a flat vertex/index buffer (the BLAS input) plus a parallel per-triangle
/// (normal, albedo) table the shader indexes by `primitive_index`.
struct Scene {
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
    tris: Vec<Tri>,
}

impl Scene {
    fn new() -> Self {
        Self { vertices: Vec::new(), indices: Vec::new(), tris: Vec::new() }
    }

    /// Append a quad (a,b,c,d wound CCW) as two triangles sharing a flat geometric normal.
    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, albedo: Vec3) {
        let n = (b - a).cross(c - a).normalize();
        let base = self.vertices.len() as u32;
        for v in [a, b, c, d] {
            self.vertices.push([v.x, v.y, v.z]);
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        let tri = Tri { normal: [n.x, n.y, n.z, 0.0], albedo: [albedo.x, albedo.y, albedo.z, 1.0] };
        self.tris.push(tri);
        self.tris.push(tri);
    }

    /// Append an axis-aligned box with outward-facing quads.
    fn cuboid(&mut self, min: Vec3, max: Vec3, albedo: Vec3) {
        let (x0, y0, z0) = (min.x, min.y, min.z);
        let (x1, y1, z1) = (max.x, max.y, max.z);
        let p = |x: f32, y: f32, z: f32| vec3(x, y, z);
        // -Z, +Z, -X, +X, -Y, +Y faces (each wound CCW about its outward normal).
        self.quad(p(x0, y0, z0), p(x0, y1, z0), p(x1, y1, z0), p(x1, y0, z0), albedo);
        self.quad(p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1), p(x0, y0, z1), albedo);
        self.quad(p(x0, y0, z1), p(x0, y1, z1), p(x0, y1, z0), p(x0, y0, z0), albedo);
        self.quad(p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), p(x1, y0, z1), albedo);
        self.quad(p(x0, y0, z1), p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), albedo);
        self.quad(p(x0, y1, z0), p(x0, y1, z1), p(x1, y1, z1), p(x1, y1, z0), albedo);
    }
}

fn build_scene() -> Scene {
    let mut s = Scene::new();
    // Ground plane at y=0.
    s.quad(
        vec3(-14.0, 0.0, -14.0),
        vec3(-14.0, 0.0, 14.0),
        vec3(14.0, 0.0, 14.0),
        vec3(14.0, 0.0, -14.0),
        vec3(0.55, 0.55, 0.58),
    );
    // Three boxes of different heights so their hard shadows are unmistakable.
    s.cuboid(vec3(-3.0, 0.0, -1.0), vec3(-1.0, 3.0, 1.0), vec3(0.80, 0.27, 0.22)); // tall red
    s.cuboid(vec3(1.0, 0.0, -2.5), vec3(2.5, 1.4, -1.0), vec3(0.32, 0.68, 0.34)); // short green
    s.cuboid(vec3(0.4, 0.0, 1.4), vec3(2.2, 2.1, 3.2), vec3(0.30, 0.46, 0.86)); // medium blue
    s
}

/// Compute the eye-relative ray-plane basis (lower-left direction + horizontal/vertical spans) for a
/// pinhole camera, matching the reconstruction in `rt_spike.wgsl`.
fn camera_basis(eye: Vec3, target: Vec3, fov_y_deg: f32, aspect: f32) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    let half_h = (fov_y_deg.to_radians() * 0.5).tan();
    let half_w = half_h * aspect;
    let ll = forward - right * half_w - up * half_h;
    let horiz = right * (2.0 * half_w);
    let vert = up * (2.0 * half_h);
    (
        [ll.x, ll.y, ll.z, 0.0],
        [horiz.x, horiz.y, horiz.z, 0.0],
        [vert.x, vert.y, vert.z, 0.0],
    )
}

pub fn run() {
    let path = std::env::var("VOXELCRAFT_RT_SPIKE").unwrap_or_else(|_| "rt_spike.png".to_string());
    let path = if path == "1" { "rt_spike.png".to_string() } else { path };

    // The probe established that only the Vulkan backend exposes hardware ray queries on the 4090.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("RT_SPIKE: no Vulkan adapter");

    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        log::error!("RT_SPIKE: adapter does not expose EXPERIMENTAL_RAY_QUERY; aborting");
        return;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rt-spike-device"),
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        // Default limits zero out the acceleration-structure caps; adopt the adapter's real maxima.
        required_limits: adapter.limits(),
        // SAFETY: this spike is the acknowledgment — we accept that wgpu's hardware-RT path is
        // experimental and may contain bugs; that's exactly what this isolated probe is testing.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .expect("RT_SPIKE: request_device failed");

    let scene = build_scene();
    let tri_count = scene.tris.len();
    log::info!(
        "RT_SPIKE: scene has {} vertices, {} triangles",
        scene.vertices.len(),
        tri_count
    );

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-spike-vertices"),
        contents: bytemuck::cast_slice(&scene.vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
    });
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-spike-indices"),
        contents: bytemuck::cast_slice(&scene.indices),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
    });
    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-spike-tris"),
        contents: bytemuck::cast_slice(&scene.tris),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ---- Acceleration structures ----
    let geo_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: scene.vertices.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(scene.indices.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("rt-spike-blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![geo_size.clone()] },
    );

    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("rt-spike-tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    // Identity transform (3x4 row-major); the scene is baked in world space already.
    const IDENTITY: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    tlas[0] = Some(wgpu::TlasInstance::new(&blas, IDENTITY, 0, 0xff));

    let mut build_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rt-spike-build"),
    });
    build_enc.build_acceleration_structures(
        iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &geo_size,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: std::mem::size_of::<[f32; 3]>() as u64,
                index_buffer: Some(&index_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::once(&tlas),
    );
    queue.submit(Some(build_enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    log::info!("RT_SPIKE: acceleration structures built");

    // ---- Output target + uniforms ----
    let out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt-spike-out"),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let eye = vec3(7.5, 5.0, 7.5);
    let target = vec3(0.0, 1.2, 0.0);
    let (ll, horiz, vert) = camera_basis(eye, target, 45.0, WIDTH as f32 / HEIGHT as f32);
    let sun = vec3(0.55, 0.9, 0.32).normalize();
    let uniforms = Uniforms {
        eye: [eye.x, eye.y, eye.z, 0.0],
        ll,
        horiz,
        vert,
        sun: [sun.x, sun.y, sun.z, 0.0],
        dims: [WIDTH, HEIGHT, 0, 0],
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-spike-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ---- Compute pipeline ----
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rt-spike-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/rt_spike.wgsl").into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rt-spike-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rt-spike-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
            wgpu::BindGroupEntry { binding: 2, resource: tri_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&out_view),
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rt-spike-layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rt-spike-pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // ---- Dispatch + readback ----
    let bpp = 4u32;
    let unpadded = WIDTH * bpp;
    let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt-spike-readback"),
        size: (padded * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rt-spike-compute"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rt-spike-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1);
    }
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map channel closed").expect("buffer map failed");

    let data = slice.get_mapped_range();
    let mut rgba = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    for y in 0..HEIGHT as usize {
        let src_off = y * padded as usize;
        let src = &data[src_off..src_off + unpadded as usize];
        let dst_off = y * (WIDTH * 4) as usize;
        rgba[dst_off..dst_off + (WIDTH * 4) as usize].copy_from_slice(src);
    }
    drop(data);
    readback.unmap();

    image::save_buffer(&path, &rgba, WIDTH, HEIGHT, image::ColorType::Rgba8)
        .expect("RT_SPIKE: failed to save PNG");
    log::info!("RT_SPIKE: hardware-ray-traced image saved to {path} ({WIDTH}x{HEIGHT})");
}
