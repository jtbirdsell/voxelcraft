// Supersample downscale (M33-G9): resolve a super-resolution LDR image down to the swapchain with a
// sharp Catmull-Rom (bicubic, a=-0.5) filter — the DLDSR-style "render high, downscale sharp" step.
// `src_tex` is the swapchain (sRGB) format, so the Linear sampler returns sRGB-DECODED (linear) values;
// we weight in linear light and the sRGB swapchain re-encodes on write. Bilinear would be too soft and
// throw away the detail supersampling paid for.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.uv = vec2<f32>(x, y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

// Catmull-Rom cubic weight (a = -0.5).
fn cr(x: f32) -> f32 {
    let a = abs(x);
    if (a < 1.0) {
        return 1.5 * a * a * a - 2.5 * a * a + 1.0;
    } else if (a < 2.0) {
        return -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0;
    }
    return 0.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex));
    let src_pos = in.uv * dims - 0.5;
    let base = floor(src_pos);
    let f = src_pos - base;
    // 4 separable taps per axis at offsets {-1,0,1,2} from `base`.
    let wx = vec4<f32>(cr(f.x + 1.0), cr(f.x), cr(f.x - 1.0), cr(f.x - 2.0));
    let wy = vec4<f32>(cr(f.y + 1.0), cr(f.y), cr(f.y - 1.0), cr(f.y - 2.0));

    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let texel = base + vec2<f32>(f32(i) - 1.0, f32(j) - 1.0);
            let uv = (texel + 0.5) / dims;
            let c = textureSampleLevel(src_tex, samp, uv, 0.0).rgb;
            let w = wx[i] * wy[j];
            acc = acc + c * w;
            wsum = wsum + w;
        }
    }
    // Normalize + clamp negative lobes (ringing) to keep colours valid.
    return vec4<f32>(max(acc / wsum, vec3<f32>(0.0)), 1.0);
}
