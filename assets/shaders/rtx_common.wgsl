// Shared RTX scaffolding, prepended (at pipeline build time) to chunk.wgsl, water.wgsl, and
// glass.wgsl: the camera + voxel-volume bindings, the standard lit-vertex stage, and the DDA voxel
// tracer used for sun shadows, ambient occlusion / global illumination, and water reflections.
// Keeping one copy means the tracer can't drift between the shaders.

struct Camera {
    view_proj: mat4x4<f32>,          // JITTERED render projection (under DLSS)
    prev_view_proj: mat4x4<f32>,     // previous frame, UN-jittered (motion vectors)
    view_proj_nojitter: mat4x4<f32>, // current frame, UN-jittered (motion vectors) — M33-G8
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sky_color: vec4<f32>,
    fog_color: vec4<f32>,
    params: vec4<f32>, // fog_start, fog_end, ambient, sun_intensity
    time: vec4<f32>,   // elapsed_seconds, day_fraction, _, _
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Volume {
    origin: vec4<i32>,
    params: vec4<u32>,  // size_xz, rtx_mode, gi_rays, size_y
    paramsf: vec4<f32>, // gi_dist, gi_strength, sky_boost, water_smooth
    paramsg: vec4<f32>, // sun_dist, gi_sun_dist, water_refl_max, water_depth_max (P21 tier)
};
@group(1) @binding(0) var voxels: texture_3d<u32>;
@group(1) @binding(1) var<uniform> volume: Volume;

// NOTE: the procedural block atlas (group 2) + `sample_tile()` live in `atlas.wgsl`, prepended only
// to the render shaders (chunk/water/glass). They use `textureSample`, which is fragment-only, so
// they are kept OUT of this module — `rtx_common.wgsl` is also prepended to the GI compute shader
// (`gi_compute.wgsl`), where `textureSample` is illegal.

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tile: u32,
    @location(4) light: vec2<f32>,  // (sky, block) 0..1
    @location(5) shade: vec2<f32>,  // (emission, tint_class)
};
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) tile: u32,
    @location(4) light: vec2<f32>,
    @location(5) shade: vec2<f32>,
    // Clip-space positions for screen-space motion vectors (M33-G3). `@builtin(position)` is
    // framebuffer space in the fragment stage, so the clip coords are forwarded explicitly.
    @location(6) cur_clip: vec4<f32>,
    @location(7) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let wp = vec4<f32>(in.position, 1.0);
    out.clip_pos = camera.view_proj * wp;
    // Motion vectors use the UN-jittered projections (current + previous) so they carry only
    // geometric motion; DLSS resolves the subpixel jitter itself. (M33-G8)
    out.cur_clip = camera.view_proj_nojitter * wp;
    out.prev_clip = camera.prev_view_proj * wp;
    out.normal = in.normal;
    out.world_pos = in.position;
    out.uv = in.uv;
    out.tile = in.tile;
    out.light = in.light;
    out.shade = in.shade;
    return out;
}

fn in_volume(p: vec3<i32>) -> bool {
    let sxz = i32(volume.params.x);
    let sy = i32(volume.params.w);
    let l = p - volume.origin.xyz;
    return l.x >= 0 && l.y >= 0 && l.z >= 0 && l.x < sxz && l.y < sy && l.z < sxz;
}

// Opaque block id at a world voxel (0 = air/transparent/out-of-volume). The volume is a toroidal
// ring buffer keyed by worldPos mod size per axis (XZ and Y differ), matching the chunk uploads.
fn voxel_id(p: vec3<i32>) -> u32 {
    if (!in_volume(p)) {
        return 0u;
    }
    let sxz = i32(volume.params.x);
    let sy = i32(volume.params.w);
    let t = vec3<i32>(
        ((p.x % sxz) + sxz) % sxz,
        ((p.y % sy) + sy) % sy,
        ((p.z % sxz) + sxz) % sxz,
    );
    return textureLoad(voxels, t, 0).r;
}

fn voxel_solid(p: vec3<i32>) -> bool {
    return voxel_id(p) != 0u;
}

// Representative albedo per block id (mirrors block::face_color averages) for GI/reflection color.
fn voxel_color(id: u32) -> vec3<f32> {
    switch (id) {
        case 1u:  { return vec3<f32>(0.49, 0.49, 0.52); } // stone
        case 2u:  { return vec3<f32>(0.45, 0.33, 0.21); } // dirt
        case 3u:  { return vec3<f32>(0.36, 0.60, 0.27); } // grass
        case 4u:  { return vec3<f32>(0.80, 0.75, 0.52); } // sand
        case 5u:  { return vec3<f32>(0.45, 0.34, 0.21); } // wood
        case 6u:  { return vec3<f32>(0.20, 0.42, 0.18); } // leaves
        case 8u:  { return vec3<f32>(0.92, 0.94, 0.97); } // snow
        case 9u:  { return vec3<f32>(0.28, 0.28, 0.30); } // coal ore
        case 10u: { return vec3<f32>(0.60, 0.52, 0.45); } // iron ore
        case 11u: { return vec3<f32>(1.0, 0.42, 0.06); }  // lava
        case 12u: { return vec3<f32>(0.85, 0.62, 0.28); } // torch
        case 13u: { return vec3<f32>(0.95, 0.82, 0.45); } // glowstone
        case 14u: { return vec3<f32>(0.42, 0.42, 0.44); } // cobblestone
        case 15u: { return vec3<f32>(0.62, 0.48, 0.30); } // planks
        case 16u: { return vec3<f32>(0.55, 0.28, 0.22); } // bricks
        case 17u: { return vec3<f32>(0.20, 0.20, 0.22); } // bedrock
        case 18u: { return vec3<f32>(0.50, 0.47, 0.45); } // gravel
        case 19u: { return vec3<f32>(0.12, 0.09, 0.18); } // obsidian
        case 20u: { return vec3<f32>(0.55, 0.50, 0.35); } // gold ore
        case 21u: { return vec3<f32>(0.45, 0.62, 0.62); } // diamond ore
        case 22u: { return vec3<f32>(0.45, 0.30, 0.30); } // redstone ore
        case 23u: { return vec3<f32>(0.30, 0.35, 0.55); } // lapis ore
        case 24u: { return vec3<f32>(0.22, 0.22, 0.25); } // deepslate
        case 25u: { return vec3<f32>(0.50, 0.36, 0.22); } // crafting table
        case 26u: { return vec3<f32>(0.38, 0.38, 0.40); } // furnace
        case 27u: { return vec3<f32>(0.55, 0.42, 0.24); } // chest
        case 31u: { return vec3<f32>(0.30, 0.52, 0.24); } // cactus (28-30 are non-opaque plants)
        case 36u: { return vec3<f32>(0.79, 0.50, 0.13); } // pumpkin (32-35 are non-opaque plants)
        case 37u: { return vec3<f32>(0.66, 0.80, 0.92); } // ice (38 glass is non-opaque -> stored 0)
        case 39u: { return vec3<f32>(0.49, 0.49, 0.52); } // stone slab
        case 40u, 42u, 43u, 44u: { return vec3<f32>(0.49, 0.49, 0.52); } // stone stairs (4 facings)
        case 41u: { return vec3<f32>(0.62, 0.48, 0.30); } // wood slab
        // U2 ores (opaque cubes): tile-average reflectance, in lockstep with block::face_color.
        case 49u: { return vec3<f32>(0.58, 0.49, 0.44); } // copper ore
        case 50u: { return vec3<f32>(0.46, 0.58, 0.49); } // emerald ore
        case 51u: { return vec3<f32>(0.20, 0.20, 0.22); } // deepslate coal ore
        case 52u: { return vec3<f32>(0.34, 0.30, 0.30); } // deepslate iron ore
        case 53u: { return vec3<f32>(0.36, 0.30, 0.29); } // deepslate copper ore
        case 54u: { return vec3<f32>(0.36, 0.33, 0.27); } // deepslate gold ore
        case 55u: { return vec3<f32>(0.33, 0.23, 0.25); } // deepslate redstone ore
        case 56u: { return vec3<f32>(0.24, 0.36, 0.30); } // deepslate emerald ore
        case 57u: { return vec3<f32>(0.22, 0.26, 0.38); } // deepslate lapis ore
        case 58u: { return vec3<f32>(0.27, 0.37, 0.39); } // deepslate diamond ore
        // U3 storage/compaction blocks.
        case 59u: { return vec3<f32>(0.86, 0.84, 0.80); } // iron block
        case 60u: { return vec3<f32>(0.96, 0.82, 0.25); } // gold block
        case 61u: { return vec3<f32>(0.40, 0.85, 0.84); } // diamond block
        case 62u: { return vec3<f32>(0.22, 0.78, 0.42); } // emerald block
        case 63u: { return vec3<f32>(0.18, 0.32, 0.72); } // lapis block
        case 64u: { return vec3<f32>(0.72, 0.12, 0.10); } // redstone block
        case 65u: { return vec3<f32>(0.78, 0.46, 0.34); } // copper block
        case 66u: { return vec3<f32>(0.10, 0.10, 0.12); } // coal block
        case 67u: { return vec3<f32>(0.72, 0.58, 0.46); } // raw iron block
        case 68u: { return vec3<f32>(0.74, 0.46, 0.30); } // raw copper block
        case 69u: { return vec3<f32>(0.82, 0.66, 0.28); } // raw gold block
        // U3 stone family.
        case 70u: { return vec3<f32>(0.42, 0.43, 0.40); } // tuff
        case 71u: { return vec3<f32>(0.90, 0.90, 0.88); } // calcite
        case 72u: { return vec3<f32>(0.66, 0.45, 0.38); } // granite
        case 73u: { return vec3<f32>(0.68, 0.47, 0.40); } // polished granite
        case 74u: { return vec3<f32>(0.84, 0.84, 0.85); } // diorite
        case 75u: { return vec3<f32>(0.86, 0.86, 0.87); } // polished diorite
        case 76u: { return vec3<f32>(0.55, 0.56, 0.57); } // andesite
        case 77u: { return vec3<f32>(0.57, 0.58, 0.59); } // polished andesite
        case 78u: { return vec3<f32>(0.62, 0.64, 0.70); } // clay
        case 79u: { return vec3<f32>(0.55, 0.42, 0.34); } // dripstone block
        case 80u: { return vec3<f32>(0.28, 0.27, 0.30); } // smooth basalt
        case 81u: { return vec3<f32>(0.26, 0.26, 0.29); } // cobbled deepslate
        case 82u: { return vec3<f32>(0.24, 0.24, 0.27); } // polished deepslate
        case 83u: { return vec3<f32>(0.23, 0.23, 0.26); } // deepslate bricks
        case 84u: { return vec3<f32>(0.22, 0.22, 0.25); } // deepslate tiles
        case 85u: { return vec3<f32>(0.78, 0.46, 0.34); } // cut copper
        // U4 cave-biome OPAQUE cubes only (cross-billboard crystals/plants store 0 in the volume and
        // are correctly absent here). In lockstep with block::face_color + texture.rs base_color.
        case 86u:  { return vec3<f32>(0.55, 0.40, 0.78); } // amethyst block
        case 87u:  { return vec3<f32>(0.58, 0.42, 0.80); } // budding amethyst
        case 93u:  { return vec3<f32>(0.30, 0.42, 0.20); } // moss block
        case 101u: { return vec3<f32>(0.42, 0.32, 0.22); } // rooted dirt
        case 104u: { return vec3<f32>(0.30, 0.46, 0.22); } // azalea leaves
        case 105u: { return vec3<f32>(0.06, 0.09, 0.11); } // sculk
        case 107u: { return vec3<f32>(0.10, 0.20, 0.24); } // sculk sensor
        case 108u: { return vec3<f32>(0.12, 0.16, 0.18); } // sculk shrieker
        case 109u: { return vec3<f32>(0.10, 0.14, 0.16); } // sculk catalyst
        case 110u: { return vec3<f32>(0.20, 0.21, 0.24); } // reinforced deepslate
        default:  { return vec3<f32>(0.5, 0.5, 0.5); }
    }
}

// Emissive radiance leaving a voxel (0 for ordinary blocks). Lava glows; GI and reflection rays
// that hit it pick up this light, so a lava pool illuminates and is mirrored by its surroundings.
fn voxel_emission(id: u32) -> vec3<f32> {
    if (id == 11u) { return vec3<f32>(1.5, 0.5, 0.1); }  // lava
    if (id == 12u) { return vec3<f32>(1.6, 0.9, 0.35); } // torch
    if (id == 13u) { return vec3<f32>(1.8, 1.4, 0.7); }  // glowstone
    if (id == 109u) { return vec3<f32>(0.2, 0.5, 0.5); } // sculk catalyst (dim cyan soul-fire bloom)
    return vec3<f32>(0.0);
}

fn boundary(o: f32, d: f32, v: i32) -> f32 {
    if (d > 0.0) {
        return (f32(v) + 1.0 - o) / d;
    } else if (d < 0.0) {
        return (f32(v) - o) / d;
    }
    return 1e30;
}

struct Hit {
    hit: bool,
    id: u32,
    pos: vec3<f32>,
    normal: vec3<f32>,
};

// DDA voxel traversal. Returns the first opaque voxel hit within `max_dist` blocks, with the
// face normal it was entered through. `max_steps` bounds work for the shader.
fn trace_dda(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, max_steps: i32) -> Hit {
    var h: Hit;
    h.hit = false;
    h.id = 0u;
    h.pos = origin;
    h.normal = vec3<f32>(0.0);

    var voxel = vec3<i32>(floor(origin));
    let step = vec3<i32>(sign(dir));
    let tdelta = vec3<f32>(
        select(1e30, abs(1.0 / dir.x), abs(dir.x) > 1e-6),
        select(1e30, abs(1.0 / dir.y), abs(dir.y) > 1e-6),
        select(1e30, abs(1.0 / dir.z), abs(dir.z) > 1e-6),
    );
    var tmax = vec3<f32>(
        boundary(origin.x, dir.x, voxel.x),
        boundary(origin.y, dir.y, voxel.y),
        boundary(origin.z, dir.z, voxel.z),
    );
    var dist = 0.0;
    var axis = 0;
    for (var i = 0; i < max_steps; i = i + 1) {
        if (tmax.x <= tmax.y && tmax.x <= tmax.z) {
            voxel.x = voxel.x + step.x;
            dist = tmax.x;
            tmax.x = tmax.x + tdelta.x;
            axis = 0;
        } else if (tmax.y <= tmax.z) {
            voxel.y = voxel.y + step.y;
            dist = tmax.y;
            tmax.y = tmax.y + tdelta.y;
            axis = 1;
        } else {
            voxel.z = voxel.z + step.z;
            dist = tmax.z;
            tmax.z = tmax.z + tdelta.z;
            axis = 2;
        }
        if (dist > max_dist || !in_volume(voxel)) {
            break;
        }
        let id = voxel_id(voxel);
        if (id != 0u) {
            h.hit = true;
            h.id = id;
            h.pos = origin + dir * dist;
            if (axis == 0) {
                h.normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
            } else if (axis == 1) {
                h.normal = vec3<f32>(0.0, -f32(step.y), 0.0);
            } else {
                h.normal = vec3<f32>(0.0, 0.0, -f32(step.z));
            }
            break;
        }
    }
    return h;
}

// Sun visibility (1 lit, 0 shadowed) from a surface point, via a short DDA shadow ray. The active
// `sun_visibility()` (DDA wrapper or hardware ray query) is appended by the renderer per VOXELCRAFT_TRACER.
fn sun_visibility_dda(world_pos: vec3<f32>, n: vec3<f32>, max_dist: f32) -> f32 {
    let sun = normalize(camera.sun_dir.xyz);
    if (sun.y <= 0.02) {
        return 1.0;
    }
    // Slope-scaled origin bias — byte-identical to the HW-RT copy in sun_vis_hw.wgsl (kills grazing-
    // angle self-intersection "acne" banding; capped at dot>=0.1 so steep shadows don't peter-pan).
    let bias_dot = max(dot(n, sun), 0.1);
    let origin = world_pos + n * (0.06 / bias_dot) + sun * 0.01;
    // ~2 voxel boundaries per block keeps diagonal rays from clipping short of max_dist.
    let h = trace(origin, sun, max_dist, i32(max_dist * 2.0) + 4);
    return select(1.0, 0.0, h.hit);
}

// ---- Ambient occlusion + one-bounce colored GI -------------------------------------------------
// Shared by the chunk fragment shader (in-fragment GI, the parity oracle) and the GI compute pass
// (`gi_compute.wgsl`, the deferred path), so the gather can't drift between them.

// Branchless orthonormal basis (Duff et al. 2017) with `n` as the third (z) column.
fn onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

// Cosine-weighted hemisphere rays gather sky radiance when they escape and bounced (sun-lit)
// material color when they hit. Returns the indirect irradiance (NOT yet multiplied by the surface
// albedo or skylight — the caller demodulates), so the compute path can hand a clean noisy
// irradiance buffer to the denoiser / DLSS Ray Reconstruction.
fn gather_gi(world_pos: vec3<f32>, n: vec3<f32>, px: vec2<f32>) -> vec3<f32> {
    let rays = i32(volume.params.z);
    if (rays <= 0) {
        return vec3<f32>(camera.params.z);
    }
    let gi_dist = volume.paramsf.x;
    let gi_strength = volume.paramsf.y;
    let sky_boost = volume.paramsf.z;
    // A diagonal ray crosses up to ~sqrt(3) voxel boundaries per block travelled, so size the
    // step budget above gi_dist to keep AO range isotropic rather than clipping diagonal rays.
    let max_steps = i32(gi_dist * 1.8) + 2;

    let sun = normalize(camera.sun_dir.xyz);
    let ambient = camera.params.z;
    let sun_intensity = camera.params.w;
    let sky = camera.sky_color.rgb * sky_boost;

    let basis = onb(n);
    let rnd = hash2(px);
    let origin = world_pos + n * 0.06;

    var accum = vec3<f32>(0.0);
    for (var i = 0; i < rays; i = i + 1) {
        // Stratified-ish cosine-weighted hemisphere sample via golden-ratio additive recurrence,
        // Cranley–Patterson rotated per-pixel so neighbouring pixels don't share a pattern.
        let u1 = fract(rnd.x + f32(i) * 0.7548776662);
        let u2 = fract(rnd.y + f32(i) * 0.5698402909);
        let r = sqrt(u1);
        let phi = 6.28318530718 * u2;
        let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        let dir = normalize(basis * local);

        let h = trace(origin, dir, gi_dist, max_steps);
        if (h.hit) {
            let alb = voxel_color(h.id);
            let ndl_h = max(dot(h.normal, sun), 0.0);
            var vis = 1.0;
            if (ndl_h > 0.0) {
                // Secondary sun ray from the bounce point. Its range is a tier knob (P21,
                // paramsg.y): the maxed tier sets it == gi_dist, the pre-P21 behavior.
                vis = sun_visibility(h.pos, h.normal, volume.paramsg.y);
            }
            // Bounced radiance leaving the hit surface: its own ambient + direct sun term, plus any
            // self-emission (lava) so emissive blocks cast colored indirect light.
            accum += alb * (ambient + sun_intensity * ndl_h * vis) + voxel_emission(h.id);
        } else {
            accum += sky;
        }
    }
    return (accum / f32(rays)) * gi_strength;
}
