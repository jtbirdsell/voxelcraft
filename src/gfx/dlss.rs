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

use crate::gfx::graph::{RenderTargets, HDR_FORMAT};
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
}

impl DlssRender {
    /// Create the RR context + its render targets for `output_res`. Returns `None` (→ native res) if
    /// the mode isn't RR or the context can't be created.
    pub fn new(
        dlss: &Dlss,
        renderer: &ChunkRenderer,
        gpu: &Gpu,
        output_res: (u32, u32),
    ) -> Option<DlssRender> {
        if dlss.mode != DlssMode::RayReconstruction {
            // Super Resolution isn't wired yet; render natively for any non-RR mode.
            log::warn!("DLSS: only Ray Reconstruction is implemented — native res for {:?}", dlss.mode);
            return None;
        }
        let output_res = UVec2::new(output_res.0.max(1), output_res.1.max(1));
        let device = &gpu.device;
        let quality = quality_from_env();
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
            "DLSS-RR: render {}x{} -> output {}x{} ({quality:?})",
            render_res.x,
            render_res.y,
            output_res.x,
            output_res.y
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
            // gmotion is a UV delta (cur-prev); NGX wants render-pixel units. Scale by render res.
            // Sign/exact convention is an interactive-tuning item (only matters once history builds).
            motion_vector_scale: Some(Vec2::new(res.x as f32, res.y as f32)),
        };
        if let Err(e) = self.rr.render(params, queue) {
            log::error!("DLSS Ray Reconstruction evaluate failed: {e}");
        }
        self.frame = frame.wrapping_add(1);
    }
}
