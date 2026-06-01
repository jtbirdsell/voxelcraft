//! NVIDIA DLSS **Frame Generation** (DLSS-G via NVIDIA Streamline) for the wgpu DX12 backend
//! (M33-G8-FG). DLSS-G interpolates frames *inside* the swapchain present path (~2× FPS) on the Ada+
//! Optical-Flow hardware. It sits on top of Ray Reconstruction: RR denoises + upscales the scene,
//! FG generates intermediate frames from the presented frames + their depth/motion guides.
//!
//! Two hard ordering rules (from `dlss_wgpu_dx12`): `Streamline::init()` must run **before**
//! `wgpu::Instance::new` (the patched wgpu fork upgrades its DXGI factory to a Streamline proxy in
//! `Instance::init`, but only if `sl.interposer.dll` is already loaded), and
//! `FrameGenerationContext::new` must run **after** the device but **before** `surface.configure`
//! (it calls `slSetD3DDevice` before the swapchain exists). Everything degrades gracefully to no-FG.
//!
//! Requested via `VOXELCRAFT_FG=1`. Runtime needs the Streamline DLLs staged beside the exe (build.rs)
//! + `STREAMLINE_SDK` set, an Ada/RTX-40+ GPU, the DX12 backend, and a visible/focused window. FG can
//! only be verified interactively (it inserts frames during present); a `query_state` log line with
//! `num_frames_actually_presented >= 2` is the headless-friendly proxy that it is generating.

use dlss_wgpu_dx12::{
    FgConstants, FgResource, FgResources, FgUi, FrameGenerationContext, FrameGenerationOptions,
    Streamline, StreamlineError,
};
use glam029::Vec2;

use crate::gpu::Gpu;

/// Whether Frame Generation is requested (`VOXELCRAFT_FG` set).
pub fn requested() -> bool {
    std::env::var("VOXELCRAFT_FG").is_ok()
}

/// Load + initialize Streamline (signature-verified `sl.interposer.dll` + `slInit`) **before** the
/// wgpu instance is created, so the fork's `Instance::init` upgrades the DXGI factory to a Streamline
/// proxy. Returns `None` unless FG is requested, or on any failure — then FG is simply unavailable
/// and the factory upgrade is a no-op (stock wgpu behaviour).
pub fn init_streamline() -> Option<Streamline> {
    if !requested() {
        return None;
    }
    match Streamline::init() {
        Ok(sl) => {
            log::info!("FG: Streamline initialized (sl.interposer.dll loaded + verified)");
            Some(sl)
        }
        Err(e) => {
            log::warn!(
                "FG: Streamline init failed ({e}) — frame generation disabled (need STREAMLINE_SDK \
                 set, the SL DLLs beside the exe, an NVIDIA GPU, and HW-accelerated GPU scheduling)"
            );
            None
        }
    }
}

/// The per-output DLSS-G feature, bound after the device and before the swapchain is configured.
/// Owns the Streamline core API (moved out of the `Streamline` handle on success).
pub struct FrameGen {
    ctx: FrameGenerationContext,
    /// Monotonic frame index handed to `begin_frame` (drives Reflex/PCL + the per-frame token).
    frame: u32,
    /// Max `num_frames_actually_presented` observed (the success metric: 2 = DLSS-G is generating an
    /// interpolated frame between each rendered one). Updated by the periodic `query_state` poll.
    max_presented: u32,
    /// Swapchain colour format — used to rebuild the recomposition textures on resize.
    color_format: wgpu::TextureFormat,
    /// M33-G8-FG (Phase 4) UI recomposition: the scene is tonemapped (HUD-less) into `hudless_tex`
    /// and the HUD alone into `ui_tex` (transparent elsewhere). DLSS-G interpolates the HUD-less
    /// scene and recomposites the crisp UI on generated frames, so the HUD never shimmers. Both are
    /// swapchain-sized + swapchain-format (RR's output resolution equals the swapchain size).
    hudless_tex: wgpu::Texture,
    ui_tex: wgpu::Texture,
}

/// Create the swapchain-sized HUD-less colour + UI recomposition textures (render target + sampled).
fn make_recomp_textures(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::Texture) {
    let desc = |label| wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    (
        device.create_texture(&desc("fg-hudless-color")),
        device.create_texture(&desc("fg-ui-layer")),
    )
}

impl FrameGen {
    /// Create the DLSS-G context **after** the device, **before** `surface.configure`. Consumes the
    /// `streamline` handle. `color_format` is the swapchain (back-buffer + HUD-less) format. Returns
    /// `None` when FG isn't requested, the hardware/driver doesn't support DLSS-G, or creation fails
    /// — the caller then renders without frame generation.
    pub fn create(
        streamline: Option<Streamline>,
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<FrameGen> {
        let mut sl = streamline?;
        // Enable UI recomposition (Phase 4): DLSS-G recomposites a separately tagged UI layer over
        // generated frames so the HUD stays crisp instead of being interpolated.
        let options = FrameGenerationOptions::enabled()
            .with_color_format(color_format)
            .with_ui_recomposition(color_format);
        match FrameGenerationContext::new(&mut sl, device, adapter, &options) {
            Ok(ctx) => {
                match ctx.query_state() {
                    Ok(st) => log::info!(
                        "FG: DLSS-G bound — {} (presented={}, max_generate={}, vram={}MiB)",
                        st.status_text,
                        st.num_frames_actually_presented,
                        st.num_frames_to_generate_max,
                        st.estimated_vram_usage_in_bytes / (1024 * 1024),
                    ),
                    Err(e) => log::info!("FG: DLSS-G bound (initial query_state failed: {e})"),
                }
                let (hudless_tex, ui_tex) =
                    make_recomp_textures(device, color_format, width, height);
                Some(FrameGen {
                    ctx,
                    frame: 0,
                    max_presented: 0,
                    color_format,
                    hudless_tex,
                    ui_tex,
                })
            }
            Err(e) => {
                log::warn!(
                    "FG: DLSS-G context creation failed ({e}) — frame generation disabled \
                     (needs an Ada / RTX-40+ GPU on the DX12 backend)"
                );
                None
            }
        }
    }

    /// Present one frame through DLSS-G, generating an interpolated frame between this and the
    /// previous. Drives the crate's strict `Frame` sequence: `begin_frame` → `set_constants` →
    /// `acquire` → `render` (the closure draws the final frame into the acquired swapchain texture) →
    /// `tag` (depth + motion vectors) → submit → `end_render` → `present`. `depth`/`motion` are the
    /// output-resolution G-buffer guides; `render` must submit its own command buffers (the tag is a
    /// separate raw encoder submitted after). Errors fall back to skipping the frame, never panic.
    pub fn present_frame<F>(
        &mut self,
        gpu: &Gpu,
        aspect: f32,
        depth: &wgpu::Texture,
        motion: &wgpu::Texture,
        hudless: &wgpu::Texture,
        ui: &wgpu::Texture,
        render: F,
    ) where
        F: FnOnce(&wgpu::TextureView),
    {
        let idx = self.frame;
        self.frame = self.frame.wrapping_add(1);

        let frame = match self.ctx.begin_frame(idx) {
            Ok(f) => f,
            Err(e) => {
                log::error!("FG: begin_frame failed: {e}");
                return;
            }
        };

        // The G-buffer motion is a screen-space UV delta (cur - prev) that already carries the full
        // camera motion, so `mvec_scale = (1,1)` (no pixel→UV rescale) and `camera_motion_included`.
        // Camera matrices stay identity (DLSS-G then uses the mvecs verbatim). The exact mvec
        // sign/convention is an interactive-tuning item.
        let mut consts = FgConstants::new();
        consts.mvec_scale = Vec2::new(1.0, 1.0);
        consts.camera_motion_included = true;
        consts.camera_aspect_ratio = aspect;
        if let Err(e) = frame.set_constants(&consts) {
            log::error!("FG: set_constants failed: {e}");
            return;
        }

        // acquire performs the mandatory GetCurrentBackBufferIndex; reconfigure + skip on a transient
        // unavailable surface.
        let surface_tex = match frame.acquire(&gpu.surface) {
            Ok((tex, _bbi)) => tex,
            Err(StreamlineError::SurfaceUnavailable { .. }) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            Err(e) => {
                log::error!("FG: acquire failed: {e}");
                return;
            }
        };
        let view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Render the final frame into the acquired swapchain texture (submits its own command bufs).
        render(&view);

        // Tag the DLSS-G inputs on a dedicated raw-only encoder, submitted AFTER the render so the
        // tagged content is the freshly rendered frame.
        let mut tag_enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fg-tag") });
        if let Err(e) = frame.tag(
            &mut tag_enc,
            &FgResources {
                depth: FgResource::new(depth),
                motion_vectors: FgResource::new(motion),
                // Phase 4: HUD-less scene colour + the UI layer (colour+alpha) for recomposition, so
                // DLSS-G interpolates the scene and keeps the HUD crisp on generated frames.
                hudless_color: Some(FgResource::new(hudless)),
                ui: Some(FgUi::ColorAndAlpha(FgResource::new(ui))),
            },
        ) {
            log::error!("FG: tag failed: {e}");
            return;
        }
        gpu.queue.submit([tag_enc.finish()]);

        frame.end_render();
        frame.present(surface_tex);
        drop(frame);

        // Periodically confirm DLSS-G is actually generating (presented == 2 in classic 2x mode).
        if idx % 30 == 0 {
            match self.ctx.query_state() {
                Ok(st) => {
                    self.max_presented = self.max_presented.max(st.num_frames_actually_presented);
                    log::info!(
                        "FG: presented={} (max {}) — {}",
                        st.num_frames_actually_presented,
                        self.max_presented,
                        st.status_text
                    );
                }
                Err(e) => log::warn!("FG: query_state failed: {e}"),
            }
        }
    }

    /// Number of frames driven through `present_frame` so far (the per-frame token index).
    pub fn frames(&self) -> u32 {
        self.frame
    }

    /// Max `num_frames_actually_presented` observed (>= 2 means DLSS-G is generating).
    pub fn max_presented(&self) -> u32 {
        self.max_presented
    }

    /// HUD-less scene-colour texture (tonemapped scene, no HUD) tagged for DLSS-G interpolation.
    pub fn hudless_tex(&self) -> &wgpu::Texture {
        &self.hudless_tex
    }

    /// UI layer texture (HUD + alpha, transparent elsewhere) tagged for DLSS-G recomposition.
    pub fn ui_tex(&self) -> &wgpu::Texture {
        &self.ui_tex
    }

    /// Rebuild the recomposition textures for a new swapchain size (call from the resize handler).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (hudless, ui) = make_recomp_textures(device, self.color_format, width, height);
        self.hudless_tex = hudless;
        self.ui_tex = ui;
    }
}
