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

/// Per-chunk mesh split by render layer: opaque (solids + foliage + partials), translucent water,
/// and translucent glass (its own alpha-blended pass, drawn over opaque, with depth write).
#[derive(Default)]
pub struct MeshData {
    pub opaque: Geometry,
    pub water: Geometry,
    pub translucent: Geometry,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.water.is_empty() && self.translucent.is_empty()
    }
}

/// A merged face: which block, whether the face normal points toward +axis, the packed light
/// (sky<<4|block) of the air voxel just outside the face, and the owner's log-axis (0 for every
/// non-log block). Light + axis are part of the key so greedy merge only fuses faces with identical
/// lighting AND orientation (so two logs of different axes never merge into one quad).
type Mask = Option<(u16, bool, u8, u8)>;

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
                        // a's +face: lit by the air voxel on b's side; axis read from the solid cell a.
                        let axis = if a == block::WOOD {
                            block::log_axis(neigh.state_at(x[0], x[1], x[2]))
                        } else {
                            0
                        };
                        Some((a, true, crate::light::at(&lightgrid, xb[0], xb[1], xb[2]), axis))
                    } else if block::renders(b) && !block::occludes(b, a) {
                        // b's -face: lit by the air voxel on a's side; axis read from the solid cell b.
                        let axis = if b == block::WOOD {
                            block::log_axis(neigh.state_at(xb[0], xb[1], xb[2]))
                        } else {
                            0
                        };
                        Some((b, false, crate::light::at(&lightgrid, x[0], x[1], x[2]), axis))
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

                    let (blk, positive, face_light, axis) = cell.unwrap();
                    let geom = if block::is_water(blk) {
                        &mut mesh.water
                    } else if block::is_glass(blk) {
                        &mut mesh.translucent
                    } else {
                        &mut mesh.opaque
                    };
                    emit_quad(geom, origin, d, u, v, i + 1, k, j, w, h, blk, positive, face_light, axis);

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

    // Non-greedy shapes emitted one cell at a time (cross plants + partial slabs/stairs). They were
    // excluded from the greedy cube sweep above, so they never participate in face culling. One pass
    // over the block array (index = x + S*(z + S*y), see world::local_index).
    for (i, &id) in neigh.center.blocks.iter().enumerate() {
        let kind = block::render_kind(id);
        if matches!(kind, block::RenderKind::Cube) {
            continue;
        }
        let x = (i as i32) % S;
        let z = (i as i32 / S) % S;
        let y = i as i32 / (S * S);
        let l = crate::light::at(&lightgrid, x, y, z);
        match kind {
            block::RenderKind::Cross => emit_cross(&mut mesh.opaque, origin, x, y, z, id, l),
            // Slab: a half-height box (bottom/top) or a full cube (double), per the state byte.
            block::RenderKind::Slab => {
                let (lo, hi) = match block::slab_half(neigh.state_at(x, y, z)) {
                    block::SLAB_TOP => ([0.0, 0.5, 0.0], [1.0, 1.0, 1.0]),
                    block::SLAB_DOUBLE => ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                    _ => ([0.0, 0.0, 0.0], [1.0, 0.5, 1.0]),
                };
                emit_box(&mut mesh.opaque, origin, x, y, z, lo, hi, id, l, 0);
            }
            // Stairs: a bottom slab + an upper half-box on the high side (per the block's facing). The
            // upper box's bottom face sits flush on the slab, so skip it (face 2 = -Y) — that removes
            // the wasted coplanar quad + its z-fighting.
            block::RenderKind::Stairs => {
                emit_box(&mut mesh.opaque, origin, x, y, z, [0.0, 0.0, 0.0], [1.0, 0.5, 1.0], id, l, 0);
                let ub = block::stair_upper_box(block::stair_facing(neigh.state_at(x, y, z)));
                emit_box(
                    &mut mesh.opaque,
                    origin,
                    x,
                    y,
                    z,
                    [ub[0], ub[1], ub[2]],
                    [ub[3], ub[4], ub[5]],
                    id,
                    l,
                    1 << 2,
                );
            }
            // Doors + trapdoors: thin oriented panel(s). Render geometry == collision (block::
            // solid_boxes), so an open door draws its swung side-panel while the doorway opens up.
            block::RenderKind::Door | block::RenderKind::Trapdoor => {
                let st = neigh.state_at(x, y, z);
                for b in block::solid_boxes(id, st) {
                    emit_box(
                        &mut mesh.opaque,
                        origin,
                        x,
                        y,
                        z,
                        [b[0], b[1], b[2]],
                        [b[3], b[4], b[5]],
                        id,
                        l,
                        0,
                    );
                }
            }
            block::RenderKind::Cube => {}
        }
    }

    mesh
}

/// Emit the faces of an axis-aligned sub-cube `[lo,hi]` (local 0..1 within cell `(lx,ly,lz)`) into
/// `geom`, textured per-face like a normal block. `skip` is a face bitmask (bit `d*2 + positive`,
/// i.e. 0:-X 1:+X 2:-Y 3:+Y 4:-Z 5:+Z) to omit faces flush against another box. Partial blocks are
/// rare and don't greedy-merge. `face_light` is the (sky<<4|block) value sampled at the cell.
#[allow(clippy::too_many_arguments)]
fn emit_box(
    geom: &mut Geometry,
    origin: [i32; 3],
    lx: i32,
    ly: i32,
    lz: i32,
    lo: [f32; 3],
    hi: [f32; 3],
    id: u16,
    face_light: u8,
    skip: u8,
) {
    let cell = [
        (origin[0] + lx) as f32,
        (origin[1] + ly) as f32,
        (origin[2] + lz) as f32,
    ];
    let light = [
        (face_light >> 4) as f32 / 15.0,
        (face_light & 0x0F) as f32 / 15.0,
    ];
    // World-space min/max corners of the sub-cube.
    let wmin = [cell[0] + lo[0], cell[1] + lo[1], cell[2] + lo[2]];
    let wmax = [cell[0] + hi[0], cell[1] + hi[1], cell[2] + hi[2]];

    for d in 0..3usize {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;
        for positive in [false, true] {
            if skip & (1 << (d * 2 + positive as usize)) != 0 {
                continue;
            }
            let foff = normal_offset(d, positive);
            let normal = normal_vec(d, positive);
            let tile = block::face_tile(id, foff);
            let shade = [block::emission(id), block::tint_class(id, foff)];
            // Plane position along d, and the face's extent along u,v (so the tile isn't stretched).
            let pd = if positive { wmax[d] } else { wmin[d] };
            let (su, sv) = (hi[u] - lo[u], hi[v] - lo[v]);
            // Four corners in (u,v) order, expressed in world space.
            let corner = |cu: f32, cv: f32| -> [f32; 3] {
                let mut p = [0.0f32; 3];
                p[d] = pd;
                p[u] = if cu < 0.5 { wmin[u] } else { wmax[u] };
                p[v] = if cv < 0.5 { wmin[v] } else { wmax[v] };
                p
            };
            let p = [
                corner(0.0, 0.0),
                corner(1.0, 0.0),
                corner(1.0, 1.0),
                corner(0.0, 1.0),
            ];
            let uvs = [[0.0, 0.0], [su, 0.0], [su, sv], [0.0, sv]];
            let base = geom.vertices.len() as u32;
            for (pos, uv) in p.iter().zip(uvs.iter()) {
                geom.vertices.push(Vertex {
                    position: *pos,
                    normal,
                    uv: *uv,
                    tile,
                    light,
                    shade,
                });
            }
            geom.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
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
    axis: u8,
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
    let tile = block::log_face_tile(block_id, foff, axis);
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
