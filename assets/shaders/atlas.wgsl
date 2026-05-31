// Procedural block atlas (8x8 grid of 16px tiles), nearest-sampled. Prepended (after rtx_common) to
// the render shaders only — chunk.wgsl, water.wgsl, glass.wgsl. `sample_tile` uses textureSample,
// which is fragment-stage-only, so this is kept separate from rtx_common.wgsl (which is also
// prepended to the GI compute shader, where textureSample is illegal).
//
// `uv` tiles one unit per block, so fract() repeats the tile across a greedy-merged quad.
@group(2) @binding(0) var atlas_tex: texture_2d<f32>;
@group(2) @binding(1) var atlas_samp: sampler;

fn sample_tile(tile: u32, uv: vec2<f32>) -> vec4<f32> {
    let col = f32(tile % 8u);
    let row = f32(tile / 8u);
    let local = fract(uv);
    let atlas_uv = (vec2<f32>(col, row) + local) / 8.0;
    return textureSample(atlas_tex, atlas_samp, atlas_uv);
}
