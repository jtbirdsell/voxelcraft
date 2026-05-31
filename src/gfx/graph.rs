//! Render-graph scaffold (M33-G2): the size-dependent intermediate targets a frame renders through.
//! For now that's just the HDR scene-color buffer the world is drawn into before the ACES tonemap
//! resolves it to the LDR swapchain (or the screenshot readback). G3+ extend `RenderTargets` with the
//! G-buffer / motion-vector / denoiser attachments, and both render paths rebuild it on resize.

/// Linear HDR scene-color format. 16-bit float keeps highlights (sun, emissive lava) above 1.0 for
/// the tonemap to roll off, and is the buffer the future GI / denoise passes read and write.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Per-resolution render targets, owned by each render path (the live present loop and the offscreen
/// screenshot) and rebuilt on resize. Built by `ChunkRenderer::make_targets`.
pub struct RenderTargets {
    /// HDR scene color the world renders into (then sampled by the tonemap pass).
    pub hdr_view: wgpu::TextureView,
    /// Bind group feeding `hdr_view` (+ sampler) to the tonemap pipeline.
    pub tonemap_bg: wgpu::BindGroup,
}
