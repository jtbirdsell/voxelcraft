// Resize the render-resolution depth + motion G-buffer guides to the FG (swapchain) resolution for
// DLSS Frame Generation. M33-G9 — the supersampling counterpart of gbuf_upscale.wgsl. Bidirectional:
// when the source is LARGER (supersampling: super-res → swap) it box-reduces the footprint; when equal
// or smaller it degenerates to a nearest point sample (== the old upscale). The per-output footprint is
// derived from screen-space derivatives of the uv (= 1/dst_dims), so no extra uniform is needed.
//
// Depth uses MIN over the footprint (nearest-camera), NOT an average — averaging linear depth across a
// silhouette is meaningless and smears DLSS-G's reprojection. Motion (a UV delta) box-averages.

@group(0) @binding(0) var src_depth: texture_2d<f32>;   // R32Float  (linear view depth = clip.w)
@group(0) @binding(1) var src_motion: texture_2d<f32>;  // Rg16Float (screen-space UV motion)

// Camera near/far — MUST match Camera::new in src/gfx/camera.rs. Linear view depth -> [0,1] NDC depth
// (DLSS-G expects projected depth, depth_inverted=false). Same conversion as gbuf_upscale.wgsl.
const NEAR: f32 = 0.1;
const FAR: f32 = 4000.0;
fn linear_to_ndc_depth(d: f32) -> f32 {
    return clamp((FAR / (FAR - NEAR)) * (1.0 - NEAR / max(d, NEAR)), 0.0, 1.0);
}

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

struct FsOut {
    @location(0) depth: f32,
    @location(1) motion: vec2<f32>,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let dims = vec2<f32>(textureDimensions(src_depth));
    // Source texels covered per destination pixel (= src_dims / dst_dims). On a fullscreen triangle the
    // uv derivatives are constant and equal 1/dst_dims.
    let footprint = vec2<f32>(abs(dpdx(in.uv.x)), abs(dpdy(in.uv.y))) * dims;
    let n = clamp(vec2<i32>(ceil(footprint)), vec2<i32>(1), vec2<i32>(4));
    let center = in.uv * dims - 0.5;
    let lo = dims - vec2<f32>(1.0);

    var min_d = 1.0e30;
    var mot = vec2<f32>(0.0);
    var cnt = 0.0;
    for (var y = 0; y < n.y; y = y + 1) {
        for (var x = 0; x < n.x; x = x + 1) {
            // Sample points spread across the footprint, centered on `center`.
            let off = vec2<f32>(f32(x), f32(y)) - 0.5 * (vec2<f32>(n) - vec2<f32>(1.0));
            let st = vec2<i32>(clamp(center + off, vec2<f32>(0.0), lo));
            min_d = min(min_d, textureLoad(src_depth, st, 0).r);
            mot = mot + textureLoad(src_motion, st, 0).xy;
            cnt = cnt + 1.0;
        }
    }
    var out: FsOut;
    out.depth = linear_to_ndc_depth(min_d);
    out.motion = mot / cnt;
    return out;
}
