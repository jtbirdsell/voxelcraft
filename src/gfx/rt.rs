//! Hardware ray-tracing acceleration structures over the voxel world (M33-G4).
//!
//! For now this is an isolated self-test (`VOXELCRAFT_AS_STATS=1`, alongside `VOXELCRAFT_SHOT`): it
//! builds a BLAS per opaque chunk mesh and one TLAS over them on the real, streamed world geometry,
//! then logs counts + build time — proving the BLAS/TLAS build chain works on DX12+DXC over the
//! actual world before the live hardware trace lands in G5. Chunk vertices are already world-space,
//! so every instance uses the identity transform.

use std::iter;
use std::time::Instant;

use crate::gpu::Gpu;
use crate::mesher::Vertex;
use crate::renderer::{GpuMesh, GpuPart};

/// 3x4 row-major identity (chunk geometry is baked in world space already).
const IDENTITY: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

fn size_desc(part: &GpuPart) -> wgpu::BlasTriangleGeometrySizeDescriptor {
    wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: part.vertex_count,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(part.index_count),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    }
}

/// Build (and submit) a BLAS for one opaque chunk part. Called at mesh-upload time; the BLAS then
/// lives on the `GpuMesh` and is referenced by the per-frame TLAS. Same queue as rendering, so the
/// build is ordered before any trace that uses it — no explicit wait needed.
pub fn build_chunk_blas(gpu: &Gpu, part: &GpuPart) -> wgpu::Blas {
    let size = size_desc(part);
    let blas = gpu.device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("chunk-blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    );
    let stride = std::mem::size_of::<Vertex>() as u64;
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chunk-blas-build"),
    });
    enc.build_acceleration_structures(
        iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &part.vertex_buffer,
                first_vertex: 0,
                vertex_stride: stride,
                index_buffer: Some(&part.index_buffer),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::empty::<&wgpu::Tlas>(),
    );
    gpu.queue.submit(Some(enc.finish()));
    blas
}

/// Owns the per-frame top-level acceleration structure: a TLAS over every loaded chunk's BLAS,
/// rebuilt each frame (cheap — the BLASes are already built). The capacity grows as the loaded set
/// does. Chunk geometry is world-space, so every instance uses the identity transform; the instance
/// index is stored as custom_data for material recovery in later stages.
pub struct RtScene {
    tlas: Option<wgpu::Tlas>,
    capacity: u32,
}

impl RtScene {
    pub fn new() -> Self {
        Self {
            tlas: None,
            capacity: 0,
        }
    }

    /// Rebuild the TLAS over the BLAS-bearing meshes; returns it for binding (None if none/no RT).
    pub fn rebuild<'a>(&'a mut self, gpu: &Gpu, meshes: &[&GpuMesh]) -> Option<&'a wgpu::Tlas> {
        if !gpu.rt_enabled {
            return None;
        }
        // Always return a (possibly empty) TLAS when RT is on. The pipelines bind group(3)
        // unconditionally under use_hw_rt, so returning None here would leave the water/glass passes
        // drawing with an UNBOUND acceleration structure whenever no opaque geometry exists
        // scene-wide (e.g. spawning into open ocean before terrain streams in). An empty TLAS makes
        // every ray miss — correct. (review H1)
        let with_blas: Vec<&wgpu::Blas> = meshes.iter().filter_map(|m| m.blas.as_ref()).collect();
        let n = with_blas.len() as u32;
        if self.tlas.is_none() || self.capacity < n.max(1) {
            self.capacity = n.next_power_of_two().max(256);
            self.tlas = Some(gpu.device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("world-tlas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                max_instances: self.capacity,
            }));
        }
        let tlas = self.tlas.as_mut().unwrap();
        for (i, blas) in with_blas.iter().enumerate() {
            tlas[i] = Some(wgpu::TlasInstance::new(blas, IDENTITY, i as u32, 0xff));
        }
        for i in with_blas.len()..self.capacity as usize {
            tlas[i] = None;
        }
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("world-tlas-build"),
        });
        enc.build_acceleration_structures(iter::empty::<&wgpu::BlasBuildEntry>(), iter::once(&*tlas));
        gpu.queue.submit(Some(enc.finish()));
        self.tlas.as_ref()
    }
}

/// Build a BLAS per opaque chunk mesh + one TLAS over them on the real loaded world, and log the
/// stats. Isolated from the live render path. No-op (warns) when the backend has no hardware RT.
pub fn log_as_stats(gpu: &Gpu, meshes: &[&GpuMesh]) {
    if !gpu.rt_enabled {
        log::warn!("AS_STATS: hardware RT not enabled on this backend; skipping");
        return;
    }
    let device = &gpu.device;
    let t0 = Instant::now();

    // One BLAS per opaque chunk part. Keep the BLASes + size descriptors alive for the build below.
    struct Item<'a> {
        blas: wgpu::Blas,
        size: wgpu::BlasTriangleGeometrySizeDescriptor,
        part: &'a GpuPart,
    }
    let mut items: Vec<Item> = Vec::new();
    let mut tris: u64 = 0;
    for mesh in meshes {
        let Some(part) = &mesh.opaque else { continue };
        let size = size_desc(part);
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("chunk-blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        tris += (part.index_count / 3) as u64;
        items.push(Item { blas, size, part });
    }
    if items.is_empty() {
        log::warn!("AS_STATS: no opaque chunk meshes to build");
        return;
    }

    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("world-tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: items.len() as u32,
    });
    for (i, it) in items.iter().enumerate() {
        tlas[i] = Some(wgpu::TlasInstance::new(&it.blas, IDENTITY, i as u32, 0xff));
    }

    let stride = std::mem::size_of::<Vertex>() as u64;
    let build_entries: Vec<wgpu::BlasBuildEntry> = items
        .iter()
        .map(|it| wgpu::BlasBuildEntry {
            blas: &it.blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &it.size,
                vertex_buffer: &it.part.vertex_buffer,
                first_vertex: 0,
                vertex_stride: stride,
                index_buffer: Some(&it.part.index_buffer),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("as-stats-build"),
    });
    enc.build_acceleration_structures(build_entries.iter(), iter::once(&tlas));
    gpu.queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let ms = t0.elapsed().as_secs_f32() * 1000.0;
    log::info!(
        "AS_STATS: built {} chunk BLAS ({} tris) + 1 TLAS ({} instances) in {:.1} ms \
         — hardware acceleration structures over the real world OK (stride {} B)",
        items.len(),
        tris,
        items.len(),
        ms,
        stride,
    );
}
