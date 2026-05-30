// Chunk shader: sun + ambient lighting with ray-traced (voxel-DDA) sun shadows, distance fog.

struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sky_color: vec4<f32>,
    fog_color: vec4<f32>,
    params: vec4<f32>, // fog_start, fog_end, ambient, sun_intensity
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Volume {
    origin: vec4<i32>,
    params: vec4<u32>, // size, rtx_enabled, _, _
};
@group(1) @binding(0) var voxels: texture_3d<u32>;
@group(1) @binding(1) var<uniform> volume: Volume;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.color = in.color;
    out.world_pos = in.position;
    return out;
}

fn in_volume(p: vec3<i32>) -> bool {
    let size = i32(volume.params.x);
    let l = p - volume.origin.xyz;
    return l.x >= 0 && l.y >= 0 && l.z >= 0 && l.x < size && l.y < size && l.z < size;
}

fn voxel_solid(p: vec3<i32>) -> bool {
    if (!in_volume(p)) {
        return false;
    }
    let size = i32(volume.params.x);
    let t = ((p % size) + size) % size;
    return textureLoad(voxels, t, 0).r != 0u;
}

fn boundary(o: f32, d: f32, v: i32) -> f32 {
    if (d > 0.0) {
        return (f32(v) + 1.0 - o) / d;
    } else if (d < 0.0) {
        return (f32(v) - o) / d;
    }
    return 1e30;
}

// Returns 0 in shadow, 1 lit, via DDA voxel traversal toward the sun.
fn sun_shadow(world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    if (volume.params.y == 0u) {
        return 1.0;
    }
    let sun = normalize(camera.sun_dir.xyz);
    if (sun.y <= 0.02) {
        return 1.0;
    }
    let max_dist = 96.0;
    let origin = world_pos + n * 0.06 + sun * 0.02;
    var voxel = vec3<i32>(floor(origin));
    let step = vec3<i32>(sign(sun));
    let tdelta = vec3<f32>(
        select(1e30, abs(1.0 / sun.x), abs(sun.x) > 1e-6),
        select(1e30, abs(1.0 / sun.y), abs(sun.y) > 1e-6),
        select(1e30, abs(1.0 / sun.z), abs(sun.z) > 1e-6),
    );
    var tmax = vec3<f32>(
        boundary(origin.x, sun.x, voxel.x),
        boundary(origin.y, sun.y, voxel.y),
        boundary(origin.z, sun.z, voxel.z),
    );
    var dist = 0.0;
    for (var i = 0; i < 192; i = i + 1) {
        if (tmax.x <= tmax.y && tmax.x <= tmax.z) {
            voxel.x = voxel.x + step.x;
            dist = tmax.x;
            tmax.x = tmax.x + tdelta.x;
        } else if (tmax.y <= tmax.z) {
            voxel.y = voxel.y + step.y;
            dist = tmax.y;
            tmax.y = tmax.y + tdelta.y;
        } else {
            voxel.z = voxel.z + step.z;
            dist = tmax.z;
            tmax.z = tmax.z + tdelta.z;
        }
        if (dist > max_dist || !in_volume(voxel)) {
            break;
        }
        if (voxel_solid(voxel)) {
            return 0.0;
        }
    }
    return 1.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let sun = normalize(camera.sun_dir.xyz);
    let ambient = camera.params.z;
    let sun_intensity = camera.params.w;
    let ndl = max(dot(n, sun), 0.0);
    let shadow = sun_shadow(in.world_pos, n);
    let light = ambient + sun_intensity * ndl * shadow;
    var rgb = in.color * light;

    let dist = length(in.world_pos - camera.cam_pos.xyz);
    let fog = smoothstep(camera.params.x, camera.params.y, dist);
    rgb = mix(rgb, camera.fog_color.rgb, fog);

    return vec4<f32>(rgb, 1.0);
}
