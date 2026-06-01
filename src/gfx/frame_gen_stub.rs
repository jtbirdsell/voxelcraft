//! Zero-cost stub for DLSS Frame Generation, compiled when the `dlss` cargo feature is OFF. Mirrors
//! `frame_gen.rs`'s public surface with uninhabited types so the engine's `Option<FrameGen>` is always
//! `None` (FG disabled, ordinary swapchain present). Keep in sync with `frame_gen.rs`. (M33-G9b)

use crate::gpu::Gpu;

/// Frame Generation is never requested without the feature.
pub fn requested() -> bool {
    false
}

pub enum Streamline {}

/// No Streamline without the feature.
pub fn init_streamline() -> Option<Streamline> {
    None
}

pub enum FrameGen {}

impl FrameGen {
    pub fn create(
        _streamline: Option<Streamline>,
        _device: &wgpu::Device,
        _adapter: &wgpu::Adapter,
        _color_format: wgpu::TextureFormat,
        _width: u32,
        _height: u32,
    ) -> Option<FrameGen> {
        None
    }
    pub fn present_frame<F>(
        &mut self,
        _gpu: &Gpu,
        _aspect: f32,
        _depth: &wgpu::Texture,
        _motion: &wgpu::Texture,
        _hudless: &wgpu::Texture,
        _ui: &wgpu::Texture,
        _render: F,
    ) -> bool
    where
        F: FnOnce(&wgpu::TextureView),
    {
        match *self {}
    }
    pub fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        match *self {}
    }
    pub fn hudless_tex(&self) -> &wgpu::Texture {
        match *self {}
    }
    pub fn ui_tex(&self) -> &wgpu::Texture {
        match *self {}
    }
    pub fn max_presented(&self) -> u32 {
        match *self {}
    }
}
