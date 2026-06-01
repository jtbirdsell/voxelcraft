// Procedural block atlas (16x16 grid of 16px tiles), nearest-sampled. Prepended (after rtx_common) to
// the render shaders only — chunk.wgsl, water.wgsl, glass.wgsl. `sample_tile` uses textureSample,
// which is fragment-stage-only, so this is kept separate from rtx_common.wgsl (which is also
// prepended to the GI compute shader, where textureSample is illegal).
//
// `uv` tiles one unit per block, so fract() repeats the tile across a greedy-merged quad.
//
// Grid dimensions — MUST match ATLAS_COLS / ATLAS_ROWS in src/gfx/texture.rs (a #[cfg(test)] in
// texture.rs reads this file and asserts they agree). Per-axis divisor so a non-square grid is safe.
const ATLAS_COLS: u32 = 16u;
const ATLAS_COLS_F: f32 = 16.0;
const ATLAS_ROWS_F: f32 = 16.0;
@group(2) @binding(0) var atlas_tex: texture_2d<f32>;
@group(2) @binding(1) var atlas_samp: sampler;

fn sample_tile(tile: u32, uv: vec2<f32>) -> vec4<f32> {
    let col = f32(tile % ATLAS_COLS);
    let row = f32(tile / ATLAS_COLS);
    let local = fract(uv);
    let atlas_uv = (vec2<f32>(col, row) + local) / vec2<f32>(ATLAS_COLS_F, ATLAS_ROWS_F);
    return textureSample(atlas_tex, atlas_samp, atlas_uv);
}
