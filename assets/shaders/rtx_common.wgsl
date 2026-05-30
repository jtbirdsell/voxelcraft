// Shared RTX scaffolding, prepended (at pipeline build time) to both chunk.wgsl and water.wgsl:
// the camera + voxel-volume bindings, the standard lit-vertex stage, and the DDA voxel tracer
// used for sun shadows, ambient occlusion / global illumination, and water reflections. Keeping
// one copy means the tracer can't drift between the two shaders.

struct Camera {
    view_proj: mat4x4<f32>,
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
    paramsf: vec4<f32>, // gi_dist, gi_strength, sky_boost, _
};
@group(1) @binding(0) var voxels: texture_3d<u32>;
@group(1) @binding(1) var<uniform> volume: Volume;

// Procedural block atlas (8x8 grid of 16px tiles), nearest-sampled. `uv` tiles one unit per block,
// so fract() repeats the tile across a greedy-merged quad.
@group(2) @binding(0) var atlas_tex: texture_2d<f32>;
@group(2) @binding(1) var atlas_samp: sampler;

fn sample_tile(tile: u32, uv: vec2<f32>) -> vec4<f32> {
    let col = f32(tile % 8u);
    let row = f32(tile / 8u);
    let local = fract(uv);
    let atlas_uv = (vec2<f32>(col, row) + local) / 8.0;
    return textureSample(atlas_tex, atlas_samp, atlas_uv);
}

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
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.world_pos = in.position;
    out.uv = in.uv;
    out.tile = in.tile;
    out.light = in.light;
    out.shade = in.shade;
    return out;
}

// Average albedo of an atlas tile (mirrors block::face_tile -> face_color). Used as the surface
// albedo until the atlas texture is sampled (M13b), and to keep GI/volume color consistent.
fn tile_average(tile: u32) -> vec3<f32> {
    switch (tile) {
        case 0u:  { return vec3<f32>(0.49, 0.49, 0.52); } // stone
        case 1u:  { return vec3<f32>(0.45, 0.33, 0.21); } // dirt
        case 2u:  { return vec3<f32>(0.36, 0.60, 0.27); } // grass top
        case 3u:  { return vec3<f32>(0.42, 0.42, 0.24); } // grass side
        case 4u:  { return vec3<f32>(0.80, 0.75, 0.52); } // sand
        case 5u:  { return vec3<f32>(0.55, 0.43, 0.27); } // wood top
        case 6u:  { return vec3<f32>(0.40, 0.30, 0.18); } // wood side
        case 7u:  { return vec3<f32>(0.20, 0.42, 0.18); } // leaves
        case 8u:  { return vec3<f32>(0.16, 0.34, 0.62); } // water
        case 9u:  { return vec3<f32>(0.92, 0.94, 0.97); } // snow
        case 10u: { return vec3<f32>(0.28, 0.28, 0.30); } // coal ore
        case 11u: { return vec3<f32>(0.60, 0.52, 0.45); } // iron ore
        case 12u: { return vec3<f32>(1.0, 0.42, 0.06); }  // lava
        case 13u: { return vec3<f32>(0.86, 0.55, 0.58); } // mob body
        case 14u: { return vec3<f32>(0.80, 0.50, 0.52); } // mob head
        case 15u: { return vec3<f32>(0.85, 0.62, 0.28); } // torch
        case 16u: { return vec3<f32>(0.95, 0.82, 0.45); } // glowstone
        case 17u: { return vec3<f32>(0.42, 0.42, 0.44); } // cobblestone
        case 18u: { return vec3<f32>(0.62, 0.48, 0.30); } // planks
        case 19u: { return vec3<f32>(0.55, 0.28, 0.22); } // bricks
        case 20u: { return vec3<f32>(0.20, 0.20, 0.22); } // bedrock
        case 21u: { return vec3<f32>(0.50, 0.47, 0.45); } // gravel
        case 22u: { return vec3<f32>(0.12, 0.09, 0.18); } // obsidian
        case 23u: { return vec3<f32>(0.55, 0.50, 0.35); } // gold ore
        case 24u: { return vec3<f32>(0.45, 0.62, 0.62); } // diamond ore
        case 25u: { return vec3<f32>(0.45, 0.30, 0.30); } // redstone ore
        case 26u: { return vec3<f32>(0.30, 0.35, 0.55); } // lapis ore
        case 27u: { return vec3<f32>(0.22, 0.22, 0.25); } // deepslate
        case 28u: { return vec3<f32>(0.50, 0.36, 0.22); } // crafting table
        case 29u: { return vec3<f32>(0.38, 0.38, 0.40); } // furnace
        case 30u: { return vec3<f32>(0.55, 0.42, 0.24); } // chest
        default:  { return vec3<f32>(1.0, 0.0, 1.0); }
    }
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
        default:  { return vec3<f32>(0.5, 0.5, 0.5); }
    }
}

// Emissive radiance leaving a voxel (0 for ordinary blocks). Lava glows; GI and reflection rays
// that hit it pick up this light, so a lava pool illuminates and is mirrored by its surroundings.
fn voxel_emission(id: u32) -> vec3<f32> {
    if (id == 11u) { return vec3<f32>(1.5, 0.5, 0.1); }  // lava
    if (id == 12u) { return vec3<f32>(1.6, 0.9, 0.35); } // torch
    if (id == 13u) { return vec3<f32>(1.8, 1.4, 0.7); }  // glowstone
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
fn trace(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, max_steps: i32) -> Hit {
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

// Sun visibility (1 lit, 0 shadowed) from a surface point, via a short DDA shadow ray.
fn sun_visibility(world_pos: vec3<f32>, n: vec3<f32>, max_dist: f32) -> f32 {
    let sun = normalize(camera.sun_dir.xyz);
    if (sun.y <= 0.02) {
        return 1.0;
    }
    let origin = world_pos + n * 0.06 + sun * 0.02;
    // ~2 voxel boundaries per block keeps diagonal rays from clipping short of max_dist.
    let h = trace(origin, sun, max_dist, i32(max_dist * 2.0) + 4);
    return select(1.0, 0.0, h.hit);
}
