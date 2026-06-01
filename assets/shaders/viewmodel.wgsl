// First-person view-model (M34-VM): the held item/hand drawn in front of the camera. Self-contained
// (NOT prefixed with rtx_common/atlas) — it samples the block atlas with its own bindings and a fixed
// view-space key light, drawn AFTER the DLSS resolve + ACES tonemap into the LDR output, so it never
// touches the HDR scene / G-buffer / motion guides. Geometry arrives already in VIEW space (the CPU
// bakes the per-frame pose transform), so only a fixed near-perspective projection is applied here.

struct VmUniform {
    proj: mat4x4<f32>,
    // x = brightness (local light scalar), y/z/w reserved.
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> vm: VmUniform;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_samp: sampler;

// Atlas grid — MUST match ATLAS_COLS/ATLAS_ROWS in src/gfx/texture.rs (mirrors atlas.wgsl).
const ATLAS_COLS: u32 = 16u;
const ATLAS_COLS_F: f32 = 16.0;
const ATLAS_ROWS_F: f32 = 16.0;

fn sample_tile(tile: u32, uv: vec2<f32>) -> vec4<f32> {
    let col = f32(tile % ATLAS_COLS);
    let row = f32(tile / ATLAS_COLS);
    let local = fract(uv);
    let atlas_uv = (vec2<f32>(col, row) + local) / vec2<f32>(ATLAS_COLS_F, ATLAS_ROWS_F);
    return textureSample(atlas_tex, atlas_samp, atlas_uv);
}

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tile: u32,
    @location(4) light: vec2<f32>,
    @location(5) shade: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) tile: u32,
    @location(3) shade: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = vm.proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.uv = in.uv;
    out.tile = in.tile;
    out.shade = in.shade;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_tile(in.tile, in.uv);
    if (texel.a < 0.5) {
        discard; // alpha-cutout for item sprites / plant-style cards
    }
    let n = normalize(in.normal);
    // Fixed view-space key light (up / right / toward the camera) — a stable, readable shape.
    let key = normalize(vec3<f32>(0.35, 0.7, 0.55));
    let ambient = 0.5;
    let ndl = max(dot(n, key), 0.0);
    let lit = (ambient + (1.0 - ambient) * ndl) * vm.params.x;
    let emissive = texel.rgb * in.shade.x; // glowstone / lava items self-glow
    let rgb = texel.rgb * lit + emissive;
    return vec4<f32>(rgb, 1.0);
}
