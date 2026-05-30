// M1 chunk shader: transform by camera view-proj, simple directional + ambient lighting.

struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let sun_dir = normalize(vec3<f32>(0.45, 1.0, 0.35));
    let n = normalize(in.normal);
    let ndl = max(dot(n, sun_dir), 0.0);
    let ambient = 0.35;
    let light = ambient + (1.0 - ambient) * ndl;
    return vec4<f32>(in.color * light, 1.0);
}
