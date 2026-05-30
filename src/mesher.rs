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

/// A merged face: which block, whether the face normal points toward +axis, and the packed light
/// (sky<<4|block) of the air voxel just outside the face. Light is part of the key so greedy merge
/// only fuses faces with identical lighting.
type Mask = Option<(u16, bool, u8)>;

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
    // Per-face light (sky + block) for this chunk, sampled at the air voxel just outside each face.
    let lightgrid = crate::light::compute(neigh);

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
                    // Cross-billboard plants are emitted separately, not as greedy cube faces.
                    let a = if block::is_cube(a) { a } else { block::AIR };
                    let b = if block::is_cube(b) { b } else { block::AIR };
                    mask[n] = if block::renders(a) && !block::occludes(a, b) {
                        // a's +face: lit by the air voxel on b's side.
                        Some((a, true, crate::light::at(&lightgrid, xb[0], xb[1], xb[2])))
                    } else if block::renders(b) && !block::occludes(b, a) {
                        // b's -face: lit by the air voxel on a's side.
                        Some((b, false, crate::light::at(&lightgrid, x[0], x[1], x[2])))
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

                    let (blk, positive, face_light) = cell.unwrap();
                    let geom = if block::is_water(blk) {
                        &mut mesh.water
                    } else {
                        &mut mesh.opaque
                    };
                    emit_quad(geom, origin, d, u, v, i + 1, k, j, w, h, blk, positive, face_light);

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

    // Cross-billboard plants: an X of two quads per plant cell, lit and non-occluding.
    if neigh.center.blocks.iter().any(|&b| block::is_plant(b)) {
        for y in 0..S {
            for z in 0..S {
                for x in 0..S {
                    let id = neigh.center.get(x as usize, y as usize, z as usize);
                    if block::is_plant(id) {
                        let l = crate::light::at(&lightgrid, x, y, z);
                        emit_cross(&mut mesh.opaque, origin, x, y, z, id, l);
                    }
                }
            }
        }
    }

    mesh
}

/// Emit a cross-billboard plant (two diagonal quads) for the cell at chunk-local (lx,ly,lz).
fn emit_cross(geom: &mut Geometry, origin: [i32; 3], lx: i32, ly: i32, lz: i32, id: u16, face_light: u8) {
    let ox = (origin[0] + lx) as f32;
    let oy = (origin[1] + ly) as f32;
    let oz = (origin[2] + lz) as f32;
    let tile = block::face_tile(id, [0, 1, 0]);
    let shade = [block::emission(id), block::tint_class(id, [0, 1, 0])];
    let light = [(face_light >> 4) as f32 / 15.0, (face_light & 0x0F) as f32 / 15.0];
    let normal = [0.0, 1.0, 0.0]; // up-ish so plants catch sky/ambient light
    let m = 0.06;
    let (x0, x1) = (ox + m, ox + 1.0 - m);
    let (z0, z1) = (oz + m, oz + 1.0 - m);
    let (y0, y1) = (oy, oy + 1.0);
    let quads = [
        [[x0, y0, z0], [x1, y0, z1], [x1, y1, z1], [x0, y1, z0]],
        [[x1, y0, z0], [x0, y0, z1], [x0, y1, z1], [x1, y1, z0]],
    ];
    // v flipped (bottom of quad = bottom of tile = stem); slight inset to avoid the fract() wrap.
    let uvs = [[0.01, 0.99], [0.99, 0.99], [0.99, 0.01], [0.01, 0.01]];
    for q in quads {
        let base = geom.vertices.len() as u32;
        for (k, p) in q.iter().enumerate() {
            geom.vertices.push(Vertex {
                position: *p,
                normal,
                uv: uvs[k],
                tile,
                light,
                shade,
            });
        }
        geom.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
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
    face_light: u8,
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
    let light = [
        (face_light >> 4) as f32 / 15.0,
        (face_light & 0x0F) as f32 / 15.0,
    ];
    let v = geom.vertices.len() as u32;
    for (p, uv) in [p0, p1, p2, p3].iter().zip(uvs.iter()) {
        geom.vertices.push(Vertex {
            position: corner(*p),
            normal,
            uv: *uv,
            tile,
            light,
            shade,
        });
    }
    // Cull is disabled so winding is irrelevant; two triangles per quad.
    geom.indices
        .extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
}
