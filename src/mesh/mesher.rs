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

/// Per-chunk mesh split by render layer: opaque (solids + partials that occlude light), `detail`
/// (opaque-drawn geometry the DDA voxel volume does NOT treat as an occluder -- plants, doors,
/// fences, torches -- excluded from the hardware-RT BLAS so both tracers shadow the same occluder
/// set, S4b), translucent water, and translucent glass (its own alpha-blended pass).
#[derive(Default)]
pub struct MeshData {
    pub opaque: Geometry,
    pub detail: Geometry,
    pub water: Geometry,
    pub translucent: Geometry,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty()
            && self.detail.is_empty()
            && self.water.is_empty()
            && self.translucent.is_empty()
    }
}

/// A merged face: which block, whether the face normal points toward +axis, the **per-corner** smooth
/// light (M35-SL: 4 `(sky, block)` pairs from `light::corner_lights`), and the owner's log-axis (0 for
/// every non-log block). All four are part of the key, so greedy merge only fuses faces with identical
/// per-corner lighting AND orientation — uniform-lit runs still merge into big quads, while a light
/// gradient (shadow edge, AO corner, torch falloff) splits into ~1×1 quads that interpolate smoothly.
type Mask = Option<(u16, bool, [[f32; 2]; 4], u8)>;

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
                        // M35-SL: smooth per-corner light at the air voxel (xb). AO only on opaque
                        // cubes — water/glass keep the smooth average but no corner darkening.
                        let ao = !block::is_water(a) && !block::is_glass(a);
                        Some((a, true, crate::light::corner_lights(&lightgrid, neigh, xb, u, v, ao), axis))
                    } else if block::renders(b) && !block::occludes(b, a) {
                        // b's -face: lit by the air voxel on a's side; axis read from the solid cell b.
                        let axis = if b == block::WOOD {
                            block::log_axis(neigh.state_at(xb[0], xb[1], xb[2]))
                        } else {
                            0
                        };
                        let ao = !block::is_water(b) && !block::is_glass(b);
                        Some((b, false, crate::light::corner_lights(&lightgrid, neigh, x, u, v, ao), axis))
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

                    let (blk, positive, corner_light, axis) = cell.unwrap();
                    let geom = if block::is_water(blk) {
                        &mut mesh.water
                    } else if block::is_glass(blk) {
                        &mut mesh.translucent
                    } else {
                        &mut mesh.opaque
                    };
                    emit_quad(geom, origin, d, u, v, i + 1, k, j, w, h, blk, positive, corner_light, axis);

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
        // S4b: route by the DDA volume's own occluder criterion. Blocks the volume stores as 0
        // (is_volume_solid = false: plants, doors/trapdoors, fences/walls, torches, dripleaf)
        // render with the opaque pipeline but go into `detail`, which is EXCLUDED from the
        // hardware-RT BLAS -- under HWRT they were full opaque occluders (a grass billboard cast
        // a solid X-quad shadow, transparent texels included), diverging from the DDA tracer.
        let vs = block::is_volume_solid(id);
        match kind {
            block::RenderKind::Cross => {
                let geom = if vs { &mut mesh.opaque } else { &mut mesh.detail };
                emit_cross(geom, origin, x, y, z, id, l)
            }
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
                        if vs { &mut mesh.opaque } else { &mut mesh.detail },
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
            // Fence/wall/pane: a post + a thin arm toward each connecting horizontal neighbor (the
            // shape is derived from neighbors here, not stored). A lone glass pane is a flat sheet.
            block::RenderKind::Connect => {
                let geom = if id == block::GLASS_PANE {
                    &mut mesh.translucent
                } else if vs {
                    &mut mesh.opaque
                } else {
                    &mut mesh.detail
                };
                let dims = block::connect_dims(id);
                let sides = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)];
                let conn = sides.map(|(dx, dz)| block::connects(id, neigh.block_at(x + dx, y, z + dz)));
                if id == block::GLASS_PANE && !conn.iter().any(|&c| c) {
                    let s = block::PANE_SHEET;
                    emit_box(geom, origin, x, y, z, [s[0], s[1], s[2]], [s[3], s[4], s[5]], id, l, 0);
                } else {
                    let p = dims.post;
                    emit_box(geom, origin, x, y, z, [p[0], p[1], p[2]], [p[3], p[4], p[5]], id, l, 0);
                    let (a0, a1) = dims.arm_perp;
                    for (i, &(dx, dz)) in sides.iter().enumerate() {
                        if !conn[i] {
                            continue;
                        }
                        for &(ry0, ry1) in dims.rails {
                            let (lo, hi) = if dx > 0 {
                                ([p[3], ry0, a0], [1.0, ry1, a1])
                            } else if dx < 0 {
                                ([0.0, ry0, a0], [p[0], ry1, a1])
                            } else if dz > 0 {
                                ([a0, ry0, p[5]], [a1, ry1, 1.0])
                            } else {
                                ([a0, ry0, 0.0], [a1, ry1, p[2]])
                            };
                            emit_box(geom, origin, x, y, z, lo, hi, id, l, 0);
                        }
                    }
                }
            }
            // Torch/lever/button: small box(es) on a floor/wall attach face (per the state byte).
            // Walk-through (solid_boxes = BOX_NONE); the torch's block-light glow is id-keyed and so
            // survives the move off RenderKind::Cube. Per-cell, never greedy.
            block::RenderKind::Attach => {
                let st = neigh.state_at(x, y, z);
                let face = block::attach_face(st);
                for b in block::attach_boxes(id, st) {
                    // Skip the box face coplanar with the support it mounts on — that face is fully
                    // hidden by the support cube, so emitting it only z-fights with the support's
                    // exposed face. Only boxes that actually reach the cell boundary get the skip
                    // (e.g. a lever's base plate, not its raised handle). emit_box skip bit =
                    // axis*2+positive (0:-X 1:+X 2:-Y 3:+Y 4:-Z 5:+Z).
                    let skip = match face {
                        block::ATTACH_PZ if b[5] >= 1.0 => 1 << 5,
                        block::ATTACH_NZ if b[2] <= 0.0 => 1 << 4,
                        block::ATTACH_PX if b[3] >= 1.0 => 1 << 1,
                        block::ATTACH_NX if b[0] <= 0.0 => 1 << 0,
                        block::ATTACH_FLOOR if b[1] <= 0.0 => 1 << 2, // base sits on the floor's +Y face
                        _ => 0,
                    };
                    emit_box(
                        if vs { &mut mesh.opaque } else { &mut mesh.detail },
                        origin,
                        x,
                        y,
                        z,
                        [b[0], b[1], b[2]],
                        [b[3], b[4], b[5]],
                        id,
                        l,
                        skip,
                    );
                }
            }
            // F2 big dripleaf: a thin standable leaf whose box droops with the tilt stage (render ==
            // collision); fully tilted = no box (the leaf has folded down and you fall through).
            block::RenderKind::Platform => {
                let st = neigh.state_at(x, y, z);
                for b in block::solid_boxes(id, st) {
                    emit_box(
                        if vs { &mut mesh.opaque } else { &mut mesh.detail },
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
            // Stand side-face tiles upright (V tracks world-Y, row 0 at the box top); top/bottom
            // faces project onto the (u,v) ground plane. Mirrors `emit_quad`. Using world-space
            // corner deltas keeps a non-unit sub-box extent from stretching the tile.
            let horiz = if u == 1 { v } else { u };
            let uv_for = |pos: [f32; 3]| -> [f32; 2] {
                if d == 1 {
                    [pos[u] - wmin[u], pos[v] - wmin[v]]
                } else {
                    [pos[horiz] - wmin[horiz], wmax[1] - pos[1]]
                }
            };
            let base = geom.vertices.len() as u32;
            for pos in p.iter() {
                geom.vertices.push(Vertex {
                    position: *pos,
                    normal,
                    uv: uv_for(*pos),
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
    corner_light: [[f32; 2]; 4],
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
    // Tiled UV: one unit per block so a greedy w*h quad repeats the tile (fract wrap), not
    // stretches it. Side faces (normal along X/Z) must stand the tile UPRIGHT — its V axis tracks
    // world-Y with row 0 (the tile top, e.g. grass's cap or bark) at the block's top, and its U
    // axis tracks the horizontal in-plane axis. Top/bottom faces project onto the ground (u,v) plane.
    let y_top = base[1] + du[1] + dv[1]; // local Y of the quad's highest corner
    let horiz = if u == 1 { v } else { u }; // the in-plane axis that isn't world-Y (side faces)
    let uv_for = |c: [i32; 3]| -> [f32; 2] {
        if d == 1 {
            [(c[u] - base[u]) as f32, (c[v] - base[v]) as f32]
        } else {
            [(c[horiz] - base[horiz]) as f32, (y_top - c[1]) as f32]
        }
    };
    // M35-SL: per-corner smooth light. p0..p3 map to the corner order produced by `corner_lights`
    // ((du,dv) = (0,0),(1,0),(1,1),(0,1) along u/v); the GPU Gouraud-interpolates between them.
    let v = geom.vertices.len() as u32;
    for (idx, p) in [p0, p1, p2, p3].iter().enumerate() {
        geom.vertices.push(Vertex {
            position: corner(*p),
            normal,
            uv: uv_for(*p),
            tile,
            light: corner_light[idx],
            shade,
        });
    }
    // Cull is disabled so winding is irrelevant; two triangles per quad.
    geom.indices
        .extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
}
