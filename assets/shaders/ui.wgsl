// Screen-space UI: NDC position + atlas UV + RGBA color + mode, alpha-blended, no depth.
//   mode 0 = flat color (rects)
//   mode 1 = font atlas sample * color (unused; the font path is mode 2)
//   mode 2 = font coverage: rgb = color.rgb, alpha = color.a * atlas.r (crisp bitmap glyphs)
//   mode 3 = block atlas sample * color (textured block-item icons)

@group(0) @binding(0) var ui_tex: texture_2d<f32>;
@group(0) @binding(1) var ui_samp: sampler;
@group(0) @binding(2) var block_tex: texture_2d<f32>;
@group(0) @binding(3) var block_samp: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mode: f32,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) mode: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.mode = u32(in.mode + 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.mode == 0u) {
        return in.color;
    }
    if (in.mode == 3u) {
        // Textured block-item icon: sample the block atlas, tinted by color (alpha for cutouts).
        let b = textureSample(block_tex, block_samp, in.uv);
        return b * in.color;
    }
    let tex = textureSample(ui_tex, ui_samp, in.uv);
    if (in.mode == 2u) {
        return vec4<f32>(in.color.rgb, in.color.a * tex.r);
    }
    return tex * in.color;
}
