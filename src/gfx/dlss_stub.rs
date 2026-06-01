//! Zero-cost stub for DLSS Ray Reconstruction, compiled when the `dlss` cargo feature is OFF
//! (`cargo build --no-default-features` — no NVIDIA SDK / libclang). `Dlss`/`DlssRender` are
//! uninhabited enums whose constructors return `None`, so the engine's `Option<DlssRender>` is always
//! `None` and every DLSS call site falls through to native rendering. The methods exist (so the
//! `Some(_)` branches in renderer.rs / app.rs / capture.rs still type-check) but are unreachable
//! (`match *self {}`). Keep these signatures in sync with the real `dlss.rs`; the no-default-features
//! build is the guard. (M33-G9b)

use crate::gfx::graph::RenderTargets;
use crate::gpu::Gpu;
use crate::renderer::ChunkRenderer;

/// Supersample factor — always 1.0 (off) without DLSS.
pub fn supersample_from_env() -> f32 {
    1.0
}

pub enum Dlss {}

impl Dlss {
    pub fn init(_gpu: &Gpu) -> Option<Dlss> {
        None
    }
}

pub enum DlssRender {}

impl DlssRender {
    pub fn new(
        _dlss: &Dlss,
        _renderer: &ChunkRenderer,
        _gpu: &Gpu,
        _swap_res: (u32, u32),
        _ss: f32,
    ) -> Option<DlssRender> {
        None
    }
    pub fn make_render_targets(
        &self,
        _renderer: &ChunkRenderer,
        _device: &wgpu::Device,
    ) -> RenderTargets {
        match *self {}
    }
    pub fn depth_view(&self) -> &wgpu::TextureView {
        match *self {}
    }
    pub fn jitter(&self) -> (f32, f32) {
        match *self {}
    }
    pub fn render_dims(&self) -> (u32, u32) {
        match *self {}
    }
    pub fn evaluate(&mut self, _targets: &RenderTargets, _queue: &wgpu::Queue) {
        match *self {}
    }
    pub fn output_tonemap_bg(&self) -> &wgpu::BindGroup {
        match *self {}
    }
    pub fn output_gdepth(&self) -> &wgpu::Texture {
        match *self {}
    }
    pub fn output_gmotion(&self) -> &wgpu::Texture {
        match *self {}
    }
    pub fn output_gdepth_view(&self) -> &wgpu::TextureView {
        match *self {}
    }
    pub fn output_gmotion_view(&self) -> &wgpu::TextureView {
        match *self {}
    }
    pub fn is_supersampled(&self) -> bool {
        match *self {}
    }
    pub fn ss_ldr_view(&self) -> &wgpu::TextureView {
        match *self {}
    }
    pub fn ss_downscale_bg(&self) -> &wgpu::BindGroup {
        match *self {}
    }
}
