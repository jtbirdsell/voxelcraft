// M33-G2 ACES tonemap: resolve the linear HDR scene-color buffer to the LDR target. A single
// oversized triangle covers the screen; the sRGB output target applies the final encode, so we emit
// the linear ACES result here (no manual gamma).

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle: uv in [0,2], clip in [-1,3] / [3,-1].
    var out: VsOut;
    out.uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.clip = vec4<f32>(out.uv.x * 2.0 - 1.0, 1.0 - out.uv.y * 2.0, 0.0, 1.0);
    return out;
}

// Narkowicz 2015 ACES filmic approximation (returns linear display values for the sRGB encode).
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Exposure applied before the curve. The pre-G2 scene was authored straight-to-sRGB (display-
// referred), but ACES expects scene-linear and lifts midtones — so we pre-scale by ~0.72, the point
// where ACES is ~identity through the mid-tones, leaving the frame visually neutral (the intended
// G2 scaffold) while still rolling off highlights >1. Re-tuned when real HDR lighting lands (G6+).
const EXPOSURE: f32 = 0.72;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSampleLevel(hdr_tex, hdr_samp, in.uv, 0.0).rgb;
    return vec4<f32>(aces(hdr * EXPOSURE), 1.0);
}
