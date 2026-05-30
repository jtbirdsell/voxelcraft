//! M1 mesher: per-face culling (emit a face only when the neighbor is non-opaque).
//! Out-of-chunk neighbors are treated as air so the chunk's outer shell renders.
//! Binary greedy meshing + packed vertices come in M2.

use crate::block;
use crate::world::{Chunk, CHUNK_SIZE};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub const ATTRS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// (neighbor offset, 4 corner offsets of the face quad).
/// Quads are wound so [0,1,2,0,2,3] faces outward; M1 renders with culling off regardless.
const FACES: [([i32; 3], [[f32; 3]; 4]); 6] = [
    // +X
    (
        [1, 0, 0],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
    ),
    // -X
    (
        [-1, 0, 0],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
    ),
    // +Y (top)
    (
        [0, 1, 0],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
    ),
    // -Y (bottom)
    (
        [0, -1, 0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    ),
    // +Z
    (
        [0, 0, 1],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
    ),
    // -Z
    (
        [0, 0, -1],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    ),
];

pub fn build_mesh(chunk: &Chunk) -> MeshData {
    let mut mesh = MeshData::default();
    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let id = chunk.get(x, y, z);
                if !block::is_solid(id) {
                    continue;
                }
                for (offset, corners) in FACES.iter() {
                    let nx = x as i32 + offset[0];
                    let ny = y as i32 + offset[1];
                    let nz = z as i32 + offset[2];
                    let neighbor_opaque = Chunk::in_bounds(nx, ny, nz)
                        && block::is_opaque(chunk.get(nx as usize, ny as usize, nz as usize));
                    if neighbor_opaque {
                        continue;
                    }
                    let normal = [offset[0] as f32, offset[1] as f32, offset[2] as f32];
                    let color = block::face_color(id, *offset);
                    let base = mesh.vertices.len() as u32;
                    for corner in corners.iter() {
                        mesh.vertices.push(Vertex {
                            position: [
                                x as f32 + corner[0],
                                y as f32 + corner[1],
                                z as f32 + corner[2],
                            ],
                            normal,
                            color,
                        });
                    }
                    mesh.indices
                        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
    }
    mesh
}
