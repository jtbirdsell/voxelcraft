//! Greedy mesher: per-axis slab sweep that merges coplanar, same-block faces into large
//! rectangles. Faces are culled against neighbor chunks (via `Neighborhood`). Produces
//! world-space vertices so all chunks share one camera transform (camera-relative coords
//! arrive in M5). Per-vertex AO + packed vertices arrive in M4/M2 optimization passes.

use crate::block;
use crate::world::{Neighborhood, CHUNK_SIZE_I};

/// Unified world vertex (M13). 52 bytes. `color` is gone: rgb comes from the atlas `tile`/`uv`,
/// emission/tint move to `shade`. `light` is filled by block-light (M14); until then it is (1,1).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Tiled atlas UV: one unit per block, so a greedy-merged quad repeats its tile w*h times.
    pub uv: [f32; 2],
    /// Atlas tile index (see `block::face_tile` / `block::tile`); flat-interpolated.
    pub tile: u32,
    /// (skylight, blocklight) in 0..1; (1,1) until block-light (M14) fills it.
    pub light: [f32; 2],
    /// (self-emission, tint_class) — see `block::emission` / `block::tint_class`.
    pub shade: [f32; 2],
}

impl Vertex {
    pub const ATTRS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Uint32, 4 => Float32x2, 5 => Float32x2
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[derive(Default)]
pub struct Geometry {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Geometry {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Per-chunk mesh split by render layer: opaque (solids + foliage) and translucent water.
#[derive(Default)]
pub struct MeshData {
    pub opaque: Geometry,
    pub water: Geometry,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.water.is_empty()
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
                    let a = neigh.block_at(x[0], x[1], x[2]);
                    let b = neigh.block_at(xb[0], xb[1], xb[2]);
                    mask[n] = if block::renders(a) && !block::occludes(a, b) {
                        Some((a, true))
                    } else if block::renders(b) && !block::occludes(b, a) {
                        Some((b, false))
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
                    let geom = if block::is_water(blk) {
                        &mut mesh.water
                    } else {
                        &mut mesh.opaque
                    };
                    emit_quad(geom, origin, d, u, v, i + 1, k, j, w, h, blk, positive);

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
    geom: &mut Geometry,
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
    let foff = normal_offset(d, positive);
    let tile = block::face_tile(block_id, foff);
    let shade = [block::emission(block_id), block::tint_class(block_id, foff)];
    // Tiled UV: one unit per block so a greedy w*h quad repeats the tile, not stretches it.
    let uvs = [
        [0.0, 0.0],
        [w as f32, 0.0],
        [w as f32, h as f32],
        [0.0, h as f32],
    ];
    let v = geom.vertices.len() as u32;
    for (p, uv) in [p0, p1, p2, p3].iter().zip(uvs.iter()) {
        geom.vertices.push(Vertex {
            position: corner(*p),
            normal,
            uv: *uv,
            tile,
            light: [1.0, 1.0],
            shade,
        });
    }
    // Cull is disabled so winding is irrelevant; two triangles per quad.
    geom.indices
        .extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
}
