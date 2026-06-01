// GI temporal accumulation (M33-G9): reproject the previous frame's accumulated irradiance via the
// motion vectors and exponentially blend in the current noisy frame, rejecting history on disocclusion
// (depth discontinuity / off-screen). The temporal half of an SVGF-style denoiser — cuts GI variance
// before DLSS-RR. Opt-in via VOXELCRAFT_GI_ACCUM=1. No tracing, so it's self-contained on group(0)
// (unlike gi_compute.wgsl, which the renderer prefixes with the camera/volume bindings + the tracer).

@group(0) @binding(0) var gi_cur: texture_2d<f32>;    // current noisy irradiance (.rgb)
@group(0) @binding(1) var g_depth: texture_2d<f32>;   // current linear view depth (R32Float .r)
@group(0) @binding(2) var g_motion: texture_2d<f32>;  // screen-space motion (cur_uv - prev_uv)
@group(0) @binding(3) var gi_hist: texture_2d<f32>;   // prev accumulated (.rgb) + prev depth (.a)
@group(0) @binding(4) var gi_accum: texture_storage_2d<rgba16float, write>;

const ALPHA: f32 = 0.1;      // EMA weight for the current (noisy) frame (~10-frame effective history)
const DEPTH_REL: f32 = 0.05; // relative depth tolerance for the disocclusion test

@compute @workgroup_size(8, 8, 1)
fn gi_temporal_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(gi_accum);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let cur = textureLoad(gi_cur, coord, 0).rgb;
    let cur_d = textureLoad(g_depth, coord, 0).r;

    let dimf = vec2<f32>(dim);
    let cur_uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / dimf;
    let mv = textureLoad(g_motion, coord, 0).xy; // cur - prev
    let prev_uv = cur_uv - mv;                   // where this surface was last frame

    var out = cur;
    if (prev_uv.x >= 0.0 && prev_uv.x < 1.0 && prev_uv.y >= 0.0 && prev_uv.y < 1.0) {
        let prev_coord = vec2<i32>(clamp(prev_uv * dimf, vec2<f32>(0.0), dimf - vec2<f32>(1.0)));
        let hist = textureLoad(gi_hist, prev_coord, 0);
        // Reject history across a depth discontinuity (disocclusion) so edges don't smear/ghost.
        if (abs(cur_d - hist.a) <= DEPTH_REL * max(cur_d, 1.0)) {
            out = mix(hist.rgb, cur, ALPHA);
        }
    }
    textureStore(gi_accum, coord, vec4<f32>(out, cur_d));
}
