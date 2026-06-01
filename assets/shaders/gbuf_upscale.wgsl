// Nearest (point) upscale of the render-resolution linear-depth + motion-vector G-buffer guides to
// output resolution, for DLSS Frame Generation. M33-G8-FG (Phase 3, RR+FG stack).
//
// DLSS-G tags the depth + motion of the *presented* frame, which is at output (swapchain) resolution,
// but Ray Reconstruction only emits those guides at render resolution (it upscales colour, not the
// guides). The unified DLSS-SR+G path would upscale them internally; with RR (raw NGX) + FG
// (Streamline) as separate features we do it ourselves. `textureLoad` (no sampler) gives a true point
// upscale and sidesteps R32Float's non-filterability.

@group(0) @binding(0) var src_depth: texture_2d<f32>;   // R32Float  (linear view depth = clip.w)
@group(0) @binding(1) var src_motion: texture_2d<f32>;  // Rg16Float (screen-space UV motion)

// Camera near/far — MUST match Camera::new in src/gfx/camera.rs (znear/zfar). The G-buffer depth is
// LINEAR view depth (clip.w, for DLSS Ray Reconstruction's DepthType::Linear), but DLSS Frame
// Generation expects a standard projected [0,1] NDC depth (depth_inverted=false), so reproject here.
const NEAR: f32 = 0.1;
const FAR: f32 = 4000.0;

// Linear view depth d (= clip.w) -> [0,1] NDC depth, the inverse of glam::Mat4::perspective_rh:
// z_ndc = (far/(far-near)) * (1 - near/d). Clamped (the sky clear is beyond far).
fn linear_to_ndc_depth(d: f32) -> f32 {
    return clamp((FAR / (FAR - NEAR)) * (1.0 - NEAR / max(d, NEAR)), 0.0, 1.0);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single fullscreen triangle.
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
    // Both guides share render resolution; map this output pixel's uv to the nearest source texel.
    let dims = vec2<f32>(textureDimensions(src_depth));
    let st = vec2<i32>(clamp(in.uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    var out: FsOut;
    out.depth = linear_to_ndc_depth(textureLoad(src_depth, st, 0).r);
    out.motion = textureLoad(src_motion, st, 0).xy;
    return out;
}
