//! Greedy mesher: per-axis slab sweep that merges coplanar, same-block faces into large
//! rectangles. Faces are culled against neighbor chunks (via `Neighborhood`). Produces
//! world-space vertices so all chunks share one camera transform (camera-relative coords
//! arrive in M5). Per-vertex AO + packed vertices arrive in M4/M2 optimization passes.

use crate::block;
use crate::world::{Neighborhood, CHUNK_SIZE_I};

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

/// A merged face: which block, and whether the face normal points toward +axis.
type Mask = Option<(u16, bool)>;

fn normal_vec(axis: usize, positive: bool) -> [f32; 3] {
    let s = if positive { 1.0 } else { -1.0 };
    let mut n = [0.0; 3];
    n[axis] = s;
    n
}

fn normal_offset(axis: usize, positive: bool) -> [i32; 3] {
    let s = if positive { 1 } else { -1 };
    let mut n = [0; 3];
    n[axis] = s;
    n
}

/// Greedy-mesh the center chunk of `neigh`, emitting world-space geometry offset by
/// `origin` (the chunk's minimum block corner).
pub fn build_mesh(neigh: &Neighborhood, origin: [i32; 3]) -> MeshData {
    let mut mesh = MeshData::default();
    if neigh.center.is_empty() {
        return mesh;
    }
    const S: i32 = CHUNK_SIZE_I;

    for d in 0..3usize {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;
        let mut mask: Vec<Mask> = vec![None; (S * S) as usize];

        // Sweep slabs between layer i and i+1 along axis d (i from -1..S).
        let mut i = -1i32;
        while i < S {
            // Build the mask for this slab.
            let mut x = [0i32; 3];
            x[d] = i;
            let mut n = 0usize;
            for vv in 0..S {
                x[v] = vv;
                for uu in 0..S {
                    x[u] = uu;
                    let mut xb = x;
                    xb[d] = i + 1;
                    let a_op = neigh.opaque_at(x[0], x[1], x[2]);
                    let b_op = neigh.opaque_at(xb[0], xb[1], xb[2]);
                    mask[n] = if a_op && !b_op {
                        Some((neigh.block_at(x[0], x[1], x[2]), true))
                    } else if b_op && !a_op {
                        Some((neigh.block_at(xb[0], xb[1], xb[2]), false))
                    } else {
                        None
                    };
                    n += 1;
                }
            }

            // Greedy-merge rectangles over the SxS mask.
            let mut j = 0i32;
            while j < S {
                let mut k = 0i32;
                while k < S {
                    let cell = mask[(j * S + k) as usize];
                    if cell.is_none() {
                        k += 1;
                        continue;
                    }
                    // Width along u.
                    let mut w = 1i32;
                    while k + w < S && mask[(j * S + k + w) as usize] == cell {
                        w += 1;
                    }
                    // Height along v.
                    let mut h = 1i32;
                    'grow: while j + h < S {
                        for kk in 0..w {
                            if mask[((j + h) * S + k + kk) as usize] != cell {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }

                    let (blk, positive) = cell.unwrap();
                    emit_quad(&mut mesh, origin, d, u, v, i + 1, k, j, w, h, blk, positive);

                    // Clear the consumed region.
                    for jj in 0..h {
                        for kk in 0..w {
                            mask[((j + jj) * S + k + kk) as usize] = None;
                        }
                    }
                    k += w;
                }
                j += 1;
            }

            i += 1;
        }
    }
    mesh
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    mesh: &mut MeshData,
    origin: [i32; 3],
    d: usize,
    u: usize,
    v: usize,
    d_plane: i32,
    u0: i32,
    v0: i32,
    w: i32,
    h: i32,
    block_id: u16,
    positive: bool,
) {
    let mut base = [0i32; 3];
    base[d] = d_plane;
    base[u] = u0;
    base[v] = v0;
    let mut du = [0i32; 3];
    du[u] = w;
    let mut dv = [0i32; 3];
    dv[v] = h;

    let corner = |a: [i32; 3]| -> [f32; 3] {
        [
            (origin[0] + a[0]) as f32,
            (origin[1] + a[1]) as f32,
            (origin[2] + a[2]) as f32,
        ]
    };
    let p0 = base;
    let p1 = [base[0] + du[0], base[1] + du[1], base[2] + du[2]];
    let p2 = [
        base[0] + du[0] + dv[0],
        base[1] + du[1] + dv[1],
        base[2] + du[2] + dv[2],
    ];
    let p3 = [base[0] + dv[0], base[1] + dv[1], base[2] + dv[2]];

    let normal = normal_vec(d, positive);
    let color = block::face_color(block_id, normal_offset(d, positive));
    let v = mesh.vertices.len() as u32;
    for p in [p0, p1, p2, p3] {
        mesh.vertices.push(Vertex {
            position: corner(p),
            normal,
            color,
        });
    }
    // Cull is disabled in M2 so winding is irrelevant; two triangles per quad.
    mesh.indices
        .extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
}
