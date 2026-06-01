//! NVIDIA DLSS (Ray Reconstruction + Super Resolution) for the wgpu DX12 backend, via the
//! `dlss_wgpu_dx12` crate (M33-G8). DLSS runs on the RTX Tensor cores: Ray Reconstruction denoises
//! the noisy deferred-GI buffer (G6) AND upscales in one pass (replacing a hand-written SVGF
//! denoiser); Super Resolution upscales a clean image.
//!
//! Everything degrades gracefully to native-resolution rendering: the whole `Option<Dlss>` is
//! `None` whenever DLSS is off, the backend isn't DX12, or NGX init fails (non-RTX GPU, old driver,
//! missing `nvngx_dlss*.dll`). No DLSS code path can panic the game.

use std::sync::{Arc, Mutex};

use dlss_wgpu_dx12::{
    DepthType, DlssError, DlssFeatureFlags, DlssPerfQualityMode, DlssRayReconstructionContext,
    DlssRayReconstructionParameters, DlssSdk, DlssTexture, RoughnessMode,
};
// dlss_wgpu_dx12's API speaks glam 0.29; alias it here so our types match at the RR boundary.
use glam029::{UVec2, Vec2};

use crate::gfx::graph::{RenderTargets, GDEPTH_FORMAT, GMOTION_FORMAT, HDR_FORMAT};
use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::renderer::ChunkRenderer;

/// Stable project identifier for NGX (identifies the app to the NVIDIA driver). Must be a
/// well-formed RFC-4122 GUID — version nibble 4, variant nibble 8/9/a/b — or NGX's init rejects it
/// with FAIL_InvalidParameter.
const PROJECT_ID: uuid::Uuid = uuid::Uuid::from_u128(0x7b3e_9c12_8f4a_4d6e_9a21_c5b7_03e8_11df);

/// The requested DLSS mode. The *achievable* mode also depends on hardware/driver support — when
/// unsupported, `Dlss::init` returns `None` and the game renders at native resolution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DlssMode {
    /// Native resolution, no DLSS.
    Off,
    /// Super Resolution: upscale a clean low-resolution image (does not denoise).
    SuperResolution,
    /// Ray Reconstruction: denoise + upscale the noisy GI image. The default when available.
    RayReconstruction,
}

impl DlssMode {
    /// Desired mode from `VOXELCRAFT_DLSS` (default: Ray Reconstruction when available).
    pub fn from_env() -> Self {
        match std::env::var("VOXELCRAFT_DLSS").ok().as_deref() {
            Some("off") | Some("none") => DlssMode::Off,
            Some("sr") | Some("super") => DlssMode::SuperResolution,
            Some("rr") | Some("") | None => DlssMode::RayReconstruction,
            Some(other) => {
                log::warn!("unknown VOXELCRAFT_DLSS={other:?}; defaulting to rr");
                DlssMode::RayReconstruction
            }
        }
    }
}

/// Application-wide DLSS handle: the shared NGX SDK plus the active mode. Created once at startup;
/// the resolution-dependent feature contexts (built later) borrow `sdk`.
pub struct Dlss {
    pub sdk: Arc<Mutex<DlssSdk>>,
    pub mode: DlssMode,
}

impl Dlss {
    /// Initialize DLSS if a mode is requested and the backend/hardware/driver support it. Returns
    /// `None` (→ native-resolution rendering) on: mode off, a non-DX12 backend, or a failed NGX init
    /// (non-RTX GPU, old driver, missing DLLs). Never panics.
    pub fn init(gpu: &Gpu) -> Option<Dlss> {
        let mode = DlssMode::from_env();
        if mode == DlssMode::Off {
            log::info!("DLSS: off (native resolution)");
            return None;
        }
        if gpu.backend != wgpu::Backend::Dx12 {
            log::info!(
                "DLSS: unavailable on the {:?} backend (DX12-only) — native resolution",
                gpu.backend
            );
            return None;
        }
        match DlssSdk::new(PROJECT_ID, gpu.device.clone()) {
            Ok(sdk) => {
                log::info!("DLSS: SDK initialized — requested mode {mode:?}");
                Some(Dlss { sdk, mode })
            }
            Err(DlssError::FeatureNotSupported) => {
                log::warn!("DLSS: not supported on this GPU/driver — native resolution");
                None
            }
            Err(e) => {
                log::warn!("DLSS: init failed ({e}) — native resolution");
                None
            }
        }
    }
}

/// Quality preset from `VOXELCRAFT_DLSS_QUALITY` (default Quality). DLAA = denoise at full output
/// resolution (no upscale) — best quality, good for a GPU with headroom.
fn quality_from_env() -> DlssPerfQualityMode {
    match std::env::var("VOXELCRAFT_DLSS_QUALITY").ok().as_deref() {
        Some("dlaa") => DlssPerfQualityMode::Dlaa,
        Some("balanced") | Some("b") => DlssPerfQualityMode::Balanced,
        Some("performance") | Some("p") => DlssPerfQualityMode::Performance,
        Some("ultra") | Some("ultraperformance") | Some("up") => {
            DlssPerfQualityMode::UltraPerformance
        }
        Some("quality") | Some("q") | Some("") | None => DlssPerfQualityMode::Quality,
        Some(other) => {
            log::warn!("unknown VOXELCRAFT_DLSS_QUALITY={other:?}; defaulting to quality");
            DlssPerfQualityMode::Quality
        }
    }
}

/// Supersample factor from `VOXELCRAFT_SS` (render above native res, then downscale — DLDSR-style).
/// Default 1.0 (off), clamped `[1.0, 4.0]`. `>1` spends GPU headroom for a sharper, less grainy image;
/// when set, RR renders in DLAA at `swap * ss` and a Catmull-Rom pass resolves down to the window.
pub fn supersample_from_env() -> f32 {
    std::env::var("VOXELCRAFT_SS")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(1.0)
        .clamp(1.0, 4.0)
}


/// Per-output-resolution DLSS Ray Reconstruction state. Recreated on resize. Owns the RR feature
/// context, the render-resolution scene depth (also the RR hardware-depth guide), the two constant
/// guide textures voxels don't vary (specular albedo = 0, roughness = 1), and the upscaled output.
pub struct DlssRender {
    rr: DlssRayReconstructionContext,
    render_res: UVec2,
    frame: u32,
    /// Camera jitter is OFF by default: with jitter on, DLSS-RR fails to resolve the subpixel offset
    /// (steady ~17% frame-to-frame oscillation regardless of sign) — a tremor on static geometry.
    /// Without it, RR is rock-stable (denoise + upscale, slightly softer). Opt back in for tuning the
    /// jitter convention with VOXELCRAFT_DLSS_JITTER=1.
    jitter_enabled: bool,
    /// Sign applied to the PROJECTION jitter (camera) relative to NGX's InJitterOffset — the two must
    /// agree in NGX's convention or DLSS un-jitters the wrong way. Swept via VOXELCRAFT_DLSS_JSX/JSY.
    jsx: f32,
    jsy: f32,
    depth_view: wgpu::TextureView,
    specular_tex: wgpu::Texture,
    roughness_tex: wgpu::Texture,
    output_tex: wgpu::Texture,
    output_tonemap_bg: wgpu::BindGroup,
    /// M33-G8-FG (Phase 3): output-resolution depth + motion, point-upscaled from the render-res
    /// guides, tagged for DLSS Frame Generation (which interpolates the output-res presented frame).
    output_gdepth_tex: wgpu::Texture,
    output_gdepth_view: wgpu::TextureView,
    output_gmotion_tex: wgpu::Texture,
    output_gmotion_view: wgpu::TextureView,
    /// M33-G9 supersampling factor: when `ss > 1`, RR renders the scene at `swap * ss` (DLAA, no
    /// internal upscale) and a Catmull-Rom pass downscales the tonemapped result to the swapchain;
    /// `ss_ldr_*` is the super-res tonemapped LDR the downscale samples from.
    ss: f32,
    // The ss_ldr texture itself isn't stored — the view + bind group keep it alive (wgpu ref-counts).
    ss_ldr_view: wgpu::TextureView,
    ss_downscale_bg: wgpu::BindGroup,
}

impl DlssRender {
    /// Create the RR context + its render targets for `output_res`. Returns `None` (→ native res) if
    /// the mode isn't RR or the context can't be created.
    pub fn new(
        dlss: &Dlss,
        renderer: &ChunkRenderer,
        gpu: &Gpu,
        swap_res: (u32, u32),
        ss: f32,
    ) -> Option<DlssRender> {
        if dlss.mode != DlssMode::RayReconstruction {
            // Super Resolution isn't wired yet; render natively for any non-RR mode.
            log::warn!("DLSS: only Ray Reconstruction is implemented — native res for {:?}", dlss.mode);
            return None;
        }
        let swap = UVec2::new(swap_res.0.max(1), swap_res.1.max(1));
        let ss = ss.clamp(1.0, 4.0);
        // M33-G9: supersample → render at super_res. `output_res` is what RR upscales TO; with ss>1 we
        // force DLAA so RR render-res == super_res (true supersampling, no internal upscale), then a
        // Catmull-Rom pass downscales to the swapchain. At ss==1 this is today's behaviour.
        let output_res = UVec2::new(
            (swap.x as f32 * ss).round() as u32,
            (swap.y as f32 * ss).round() as u32,
        );
        let device = &gpu.device;
        let quality = if ss > 1.0 {
            DlssPerfQualityMode::Dlaa
        } else {
            quality_from_env()
        };
        let rr = match DlssRayReconstructionContext::new(
            output_res,
            quality,
            RoughnessMode::Unpacked,
            // Linear view depth (the G-buffer R32Float gdepth target). A wgpu Depth32Float attachment
            // isn't typeless, so NGX can't SRV it as R32_FLOAT — hence the dedicated linear target.
            DepthType::Linear,
            DlssFeatureFlags::HighDynamicRange,
            dlss.sdk.clone(),
            device,
            &gpu.queue,
        ) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("DLSS: Ray Reconstruction context creation failed ({e}) — native res");
                return None;
            }
        };
        let render_res = rr.render_resolution();
        log::info!(
            "DLSS-RR: render {}x{} -> output {}x{} -> swap {}x{} (ss {ss}, {quality:?})",
            render_res.x,
            render_res.y,
            output_res.x,
            output_res.y,
            swap.x,
            swap.y
        );

        let extent = |r: UVec2| wgpu::Extent3d {
            width: r.x,
            height: r.y,
            depth_or_array_layers: 1,
        };
        let make = |label: &str, res: UVec2, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent(res),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let attach = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        // Scene depth attachment only (z-test); the RR depth guide is the linear gdepth target.
        let depth_tex = make("dlss-depth", render_res, DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Voxels are diffuse: specular albedo is 0, roughness is 1 everywhere. Constant per frame, so
        // clear once here and reuse. (RR still requires the inputs to be present.)
        let specular_tex = make("dlss-specular", render_res, HDR_FORMAT, attach);
        let roughness_tex = make("dlss-roughness", render_res, wgpu::TextureFormat::R16Float, attach);
        let output_tex = make(
            "dlss-output",
            output_res,
            HDR_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let output_tonemap_bg = renderer.make_tonemap_bg(
            device,
            &output_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        );
        // FG depth/motion guides at SWAPCHAIN resolution (the presented frame). The renderer resizes
        // the render-res guides into these — point-upscale without supersampling, box-downscale with it.
        // Render attachment (resize target) + texture binding (NGX reads them). (M33-G9)
        let output_gdepth_tex = make("dlss-fg-gdepth", swap, GDEPTH_FORMAT, attach);
        let output_gdepth_view = output_gdepth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let output_gmotion_tex = make("dlss-fg-gmotion", swap, GMOTION_FORMAT, attach);
        let output_gmotion_view =
            output_gmotion_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Super-res tonemapped LDR (swapchain colour format) the Catmull-Rom downscale samples from.
        // Allocated at output_res (== super_res with ss>1); unused on the ss==1 fast path.
        let ss_ldr_tex = make("dlss-ss-ldr", output_res, gpu.config.format, attach);
        let ss_ldr_view = ss_ldr_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let ss_downscale_bg = renderer.make_tonemap_bg(device, &ss_ldr_view);

        // One-time clear of the constant guides: specular -> 0, roughness -> 1.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dlss-const-guides-clear"),
        });
        for (tex, value) in [
            (&specular_tex, wgpu::Color::TRANSPARENT),
            (&roughness_tex, wgpu::Color::WHITE),
        ] {
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dlss-guide-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(value),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        gpu.queue.submit([enc.finish()]);

        Some(DlssRender {
            rr,
            render_res,
            frame: 0,
            jitter_enabled: std::env::var("VOXELCRAFT_DLSS_JITTER").is_ok(),
            jsx: if std::env::var("VOXELCRAFT_DLSS_JSX").as_deref() == Ok("-1") { -1.0 } else { 1.0 },
            jsy: if std::env::var("VOXELCRAFT_DLSS_JSY").as_deref() == Ok("-1") { -1.0 } else { 1.0 },
            depth_view,
            specular_tex,
            roughness_tex,
            output_tex,
            output_tonemap_bg,
            output_gdepth_tex,
            output_gdepth_view,
            output_gmotion_tex,
            output_gmotion_view,
            ss,
            ss_ldr_view,
            ss_downscale_bg,
        })
    }

    /// Build the render-resolution scene targets for this DLSS pass.
    pub fn make_render_targets(&self, renderer: &ChunkRenderer, device: &wgpu::Device) -> RenderTargets {
        renderer.make_targets(device, self.render_res.x, self.render_res.y)
    }

    /// The render-resolution depth attachment (scene depth + RR depth guide).
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Tonemap bind group over the upscaled output (the tonemap pass resolves this to the swapchain).
    pub fn output_tonemap_bg(&self) -> &wgpu::BindGroup {
        &self.output_tonemap_bg
    }

    /// Output-resolution depth/motion guides for DLSS Frame Generation. The `_view`s are the
    /// nearest-upscale render targets; the textures are cloned (cheap, Arc-backed) for the FG tag.
    pub fn output_gdepth(&self) -> &wgpu::Texture {
        &self.output_gdepth_tex
    }
    pub fn output_gmotion(&self) -> &wgpu::Texture {
        &self.output_gmotion_tex
    }
    pub fn output_gdepth_view(&self) -> &wgpu::TextureView {
        &self.output_gdepth_view
    }
    pub fn output_gmotion_view(&self) -> &wgpu::TextureView {
        &self.output_gmotion_view
    }

    /// M33-G9: true when supersampling (ss > 1) — the renderer renders at super-res and downscales.
    pub fn is_supersampled(&self) -> bool {
        self.ss > 1.0
    }

    /// The super-res tonemapped-LDR render target (the supersample downscale writes its source here).
    pub fn ss_ldr_view(&self) -> &wgpu::TextureView {
        &self.ss_ldr_view
    }

    /// Bind group sampling the super-res LDR — the input to the Catmull-Rom downscale pass.
    pub fn ss_downscale_bg(&self) -> &wgpu::BindGroup {
        &self.ss_downscale_bg
    }

    /// The subpixel camera jitter (in render-resolution pixels) to apply this frame — must match
    /// what `evaluate` passes to NGX. Returned as a plain tuple to keep glam 0.29 out of the engine.
    pub fn jitter(&self) -> (f32, f32) {
        if !self.jitter_enabled {
            return (0.0, 0.0);
        }
        // The camera (projection) jitter may need a sign flip relative to NGX's InJitterOffset.
        let j = self.rr.suggested_jitter(self.frame, self.render_res);
        (j.x * self.jsx, j.y * self.jsy)
    }

    /// Render resolution (DLSS upscales from here to the output resolution).
    pub fn render_dims(&self) -> (u32, u32) {
        (self.render_res.x, self.render_res.y)
    }

    /// Evaluate Ray Reconstruction: the render-resolution scene + guides (`targets` at render res,
    /// plus the constant guides + depth) → the upscaled, denoised output. Submits on `queue`.
    pub fn evaluate(&mut self, targets: &RenderTargets, queue: &wgpu::Queue) {
        let frame = self.frame;
        let res = self.render_res;
        let jitter = if self.jitter_enabled {
            self.rr.suggested_jitter(frame, res)
        } else {
            Vec2::ZERO
        };
        let params = DlssRayReconstructionParameters {
            color: DlssTexture { texture: &targets.hdr_tex },
            diffuse_albedo: DlssTexture { texture: &targets.galbedo_tex },
            specular_albedo: DlssTexture { texture: &self.specular_tex },
            normals: DlssTexture { texture: &targets.gnormal_tex },
            roughness: DlssTexture { texture: &self.roughness_tex },
            depth: DlssTexture { texture: &targets.gdepth_tex },
            motion_vectors: DlssTexture { texture: &targets.gmotion_tex },
            output: DlssTexture { texture: &self.output_tex },
            reset: frame == 0,
            jitter_offset: jitter,
            partial_texture_size: Some(res),
            // gmotion is a UV delta (cur-prev), but NGX reprojects with prev-cur, so scale by the
            // NEGATED render res (= negate + convert UV→render pixels). This negate removes the motion
            // smear/ghosting on moving content — verified interactively. (M33-G9 motion-blur fix.)
            motion_vector_scale: Some(Vec2::new(-(res.x as f32), -(res.y as f32))),
        };
        if let Err(e) = self.rr.render(params, queue) {
            log::error!("DLSS Ray Reconstruction evaluate failed: {e}");
        }
        self.frame = frame.wrapping_add(1);
    }
}
