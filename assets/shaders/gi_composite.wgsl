// GI composite (M33-G6): the deferred counterpart to the chunk fragment's in-fragment GI. A
// fullscreen pass, run after the opaque pass and before glass/water, that reads the noisy irradiance
// (gi_compute.wgsl) + the G-buffer and ADDITIVELY blends the indirect contribution back onto the HDR
// scene color:
//
//     contribution = albedo * irradiance * skylight * (1 - fog) * (1 - flash*1.6)
//
// With the chunk shader writing everything-but-GI (direct + block-light + emissive, fogged then
// hurt-flashed) into HDR, this additive term algebraically reconstructs the exact G5b lit color:
//   mix(mix(D, fog, t), flash, k) + albedo*irr*sky*(1-t)*(1-k)  ==  G5b full color.
// The pipeline uses additive (One,One) color blending with the alpha channel masked off, so the
// HDR alpha stays 1.0 for the subsequent water/glass ALPHA_BLENDING.
//
// GI_RAW (a const injected by the renderer for VOXELCRAFT_GI_RAW) instead dumps the raw irradiance.

struct Camera {
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sky_color: vec4<f32>,
    fog_color: vec4<f32>,
    params: vec4<f32>, // fog_start, fog_end, ambient, sun_intensity
    time: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var g_albedo: texture_2d<f32>;  // albedo .rgb, skylight .a
@group(1) @binding(1) var g_pos: texture_2d<f32>;     // world position .xyz, hurt-flash .w
@group(1) @binding(2) var g_normal: texture_2d<f32>;  // world normal .xyz (sky test)
@group(1) @binding(3) var gi_tex: texture_2d<f32>;    // noisy irradiance

struct VsOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle (same convention as tonemap.wgsl).
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VsOut;
    out.clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // `@builtin(position).xy` is the pixel center in framebuffer space; the composite runs 1:1 with
    // the G-buffer, so the integer texel is a direct fetch (no sampler / interpolation).
    let coord = vec2<i32>(in.clip.xy);
    let irr = textureLoad(gi_tex, coord, 0).rgb;
    if (GI_RAW) {
        return vec4<f32>(irr, 1.0);
    }
    let nrm = textureLoad(g_normal, coord, 0).xyz;
    if (dot(nrm, nrm) < 0.5) {
        return vec4<f32>(0.0); // sky / background: additive no-op
    }
    let g = textureLoad(g_albedo, coord, 0);
    let albedo = g.rgb;
    let skylight = g.a;
    let gp = textureLoad(g_pos, coord, 0);
    let dist = length(gp.xyz - camera.cam_pos.xyz);
    let fog = smoothstep(camera.params.x, camera.params.y, dist);
    let flash = gp.w;
    let contrib = albedo * irr * skylight * (1.0 - fog) * (1.0 - flash * 1.6);
    return vec4<f32>(contrib, 0.0);
}
