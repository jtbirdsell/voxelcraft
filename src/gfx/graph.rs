//! Render-graph scaffold (M33-G2): the size-dependent intermediate targets a frame renders through.
//! For now that's just the HDR scene-color buffer the world is drawn into before the ACES tonemap
//! resolves it to the LDR swapchain (or the screenshot readback). G3+ extend `RenderTargets` with the
//! G-buffer / motion-vector / denoiser attachments, and both render paths rebuild it on resize.

/// Linear HDR scene-color format. 16-bit float keeps highlights (sun, emissive lava) above 1.0 for
/// the tonemap to roll off, and is the buffer the future GI / denoise passes read and write.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// G-buffer: world normal (.xyz) + emission (.w), written by the opaque pass. (M33-G3)
pub const GNORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// G-buffer: screen-space motion vectors (UV delta vs the previous frame). (M33-G3)
pub const GMOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// G-buffer: world-space position of the opaque surface — the GI compute pass traces from here.
/// 32-bit float for precision at large (streamed) world coordinates. (M33-G6)
pub const GPOS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// G-buffer: surface albedo (.rgb) + skylight (.a) — the GI composite multiplies irradiance by these.
pub const GALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Per-resolution render targets, owned by each render path (the live present loop and the offscreen
/// screenshot) and rebuilt on resize. Built by `ChunkRenderer::make_targets`.
pub struct RenderTargets {
    /// HDR scene color the world renders into (then sampled by the tonemap pass).
    pub hdr_view: wgpu::TextureView,
    /// G-buffer normal+emission (written by the opaque pass; read by the GI / denoiser passes later).
    pub gnormal_view: wgpu::TextureView,
    /// G-buffer screen-space motion vectors (for temporal reprojection in the denoiser).
    pub gmotion_view: wgpu::TextureView,
    /// G-buffer world-space surface position — the GI compute pass traces hemisphere rays from here.
    pub gpos_view: wgpu::TextureView,
    /// G-buffer albedo (.rgb) + skylight (.a) — the GI composite multiplies irradiance by these.
    pub galbedo_view: wgpu::TextureView,
    /// Bind group feeding the HDR + G-buffer textures (+ debug-mode uniform) to the tonemap pass.
    pub tonemap_bg: wgpu::BindGroup,
}
