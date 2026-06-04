//! GPU context: instance/surface/device/queue, swapchain config, and a depth buffer.
//! Adapter selection is platform-aware: on Windows/Linux it defaults to **DX12** — better visual
//! parity and Windows windowing on this box, with hardware RT via the DXC compiler (staged next to
//! the exe by build.rs) and Vulkan as the fallback. On **macOS** it defaults to **Metal**, the only
//! native backend there; wgpu advertises ray query on Apple GPUs but the experimental path hangs, so
//! the software DDA tracer runs (opt in via `VOXELCRAFT_TRACER=hwrt`; no DLSS). Override with
//! `VOXELCRAFT_BACKEND=dx12|vulkan|gl|metal`. GL has no hardware RT (software DDA tracer). The
//! device opts into hardware ray tracing whenever the adapter advertises it (except Metal, above).

use std::sync::Arc;
use winit::window::Window;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Gpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_view: wgpu::TextureView,
    /// True window size in physical pixels. May exceed `config.width/height` when `render_scale < 1`.
    pub size: winit::dpi::PhysicalSize<u32>,
    /// Swapchain scale relative to the window's physical size (`VOXELCRAFT_RENDER_SCALE`): `<1`
    /// renders into a smaller drawable that CAMetalLayer stretches to the window — the standard
    /// macOS reduced-resolution technique, and the ray-budget lever on Retina (2× scale = 4× the
    /// pixels of the same logical window). Metal-only; forced to 1.0 on every other backend.
    pub render_scale: f32,
    /// Which wgpu backend the adapter resolved to (Vulkan/Dx12/Gl/Metal).
    pub backend: wgpu::Backend,
    /// True when the device was created with hardware ray query (`EXPERIMENTAL_RAY_QUERY`).
    /// The RT pipeline (M33-G4+) only builds acceleration structures when this holds.
    pub rt_enabled: bool,
}

impl Gpu {
    /// Returns the GPU context plus the DLSS-G Frame Generation context (if `VOXELCRAFT_FG=1` and
    /// supported). FG lives outside `Gpu` so the present path can borrow it mutably while `Gpu` is
    /// borrowed shared.
    pub async fn new(window: Arc<Window>) -> (Self, Option<crate::frame_gen::FrameGen>) {
        let size = window.inner_size();
        let scale_factor = window.scale_factor();
        // M33-G8-FG: Streamline must initialize BEFORE the wgpu instance is created, so the patched
        // fork's `Instance::init` upgrades its DXGI factory to a Streamline proxy (only if the
        // interposer is already loaded). `None` unless `VOXELCRAFT_FG=1`. Consumed by the FG context.
        let streamline = crate::frame_gen::init_streamline();
        let (_instance, surface, adapter) = init_adapter(window).await;

        let info = adapter.get_info();
        log::info!(
            "GPU: {} | backend {:?} | driver {}",
            info.name,
            info.backend,
            info.driver_info
        );

        // Opt the real device into hardware ray tracing whenever the adapter supports it (Vulkan
        // on this 4090). Requesting EXPERIMENTAL_RAY_QUERY on an adapter that lacks it would fail
        // request_device, so gate on the feature and fall back cleanly to the software DDA path.
        // EXCEPTION — Metal: wgpu 29 advertises EXPERIMENTAL_RAY_QUERY on Apple GPUs (M3+ has RT
        // cores), but the experimental Metal path hangs this engine's first frame (the submission
        // never signals; capture's device.poll(Wait) spins forever). Only the DX12+DXC and Vulkan
        // recipes are validated (rt_spike), so on Metal hardware RT is explicit-opt-in via
        // VOXELCRAFT_TRACER=hwrt and the default is the software DDA tracer.
        let rt_supported = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let rt_enabled = rt_supported
            && (info.backend != wgpu::Backend::Metal
                || std::env::var("VOXELCRAFT_TRACER").ok().as_deref() == Some("hwrt"));
        // DLSS Ray Reconstruction's UAV output (+ some guide buffer formats) need adapter-specific
        // format capabilities; opt in when the adapter offers them so the M33-G8 DLSS path can
        // allocate those targets. Harmless when DLSS is off.
        let format_features = adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        let mut required_features = wgpu::Features::empty();
        if rt_enabled {
            required_features |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        }
        if format_features {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        // P21: per-pass GPU timestamps for the headless benchmark. Requested ONLY in bench mode so
        // ordinary runs (Windows included) ask for the exact feature set they always did; adapter-
        // gated so a backend without timestamp queries degrades to wall-time instead of failing
        // request_device (bench.rs::GpuTimer::new re-checks and returns None).
        if std::env::var("VOXELCRAFT_BENCH").is_ok()
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (required_limits, experimental_features) = if rt_enabled {
            // Default limits zero out the acceleration-structure caps; adopt the adapter's real
            // maxima (matches the validated rt_spike recipe). SAFETY: explicit acknowledgment that
            // wgpu's hardware-RT path is experimental; we pin wgpu to de-risk it (see Cargo.toml).
            (adapter.limits(), unsafe {
                wgpu::ExperimentalFeatures::enabled()
            })
        } else {
            // The deferred MRT (HDR + normal/motion/position/albedo G-buffer) needs 48 bytes per
            // sample of color attachments; the WebGPU default limit is 32. Adopt the adapter's real
            // maximum (64 on Apple GPUs) so the non-RT path (Metal/GL) can build the chunk pipeline.
            let mut limits = wgpu::Limits::default();
            limits.max_color_attachment_bytes_per_sample =
                adapter.limits().max_color_attachment_bytes_per_sample;
            (limits, wgpu::ExperimentalFeatures::disabled())
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("voxelcraft-device"),
                required_features,
                required_limits,
                experimental_features,
                ..Default::default()
            })
            .await
            .expect("Failed to create wgpu device");
        log::info!(
            "RT cores: {}",
            if rt_enabled {
                "ENABLED (hardware EXPERIMENTAL_RAY_QUERY)"
            } else if rt_supported {
                "advertised but DISABLED on Metal (experimental path hangs; VOXELCRAFT_TRACER=hwrt to try it) — software DDA tracer"
            } else {
                "unavailable on this backend/adapter — software DDA fallback"
            }
        );

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        log::info!(
            "Surface format: {:?} (sRGB={}); available: {:?}",
            format,
            format.is_srgb(),
            caps.formats
        );
        // Sub-native rendering on Metal: size the swapchain at `render_scale` × the window and let
        // CAMetalLayer stretch the drawable to the layer bounds. Everything downstream (G-buffer,
        // DDA/GI dispatch, UI, capture) keys off `config.width/height`, so this is the one choke
        // point; `self.size` keeps the true window size for cursor mapping (app.rs CursorMoved).
        let render_scale = resolve_render_scale(info.backend, scale_factor);
        let render_size = scaled_size(size, render_scale);
        if render_scale != 1.0 {
            log::info!(
                "Render scale {render_scale}: {}x{} drawable in a {}x{} window",
                render_size.width,
                render_size.height,
                size.width.max(1),
                size.height.max(1)
            );
        }
        // M33-G8-FG: create the DLSS-G context AFTER the device but BEFORE `surface.configure`
        // (`slSetD3DDevice` must precede swapchain creation). `None` unless FG was requested + bound.
        let frame_gen = crate::frame_gen::FrameGen::create(
            streamline,
            &device,
            &adapter,
            format,
            render_size.width,
            render_size.height,
        );
        // DLSS-G owns frame pacing (Reflex), so it needs a non-vsync present mode; otherwise vsync.
        let present_mode = if frame_gen.is_some() {
            if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
                wgpu::PresentMode::Immediate
            } else {
                wgpu::PresentMode::Fifo
            }
        } else {
            wgpu::PresentMode::AutoVsync
        };
        log::info!("Present mode: {present_mode:?}");
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: render_size.width,
            height: render_size.height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            // Metal runs one frame in flight: the 2026-06-03 AGXG15G wedge (windowed run hung the
            // GPU firmware irrecoverably; see GpuWatchdog below) happened with queued frames piled
            // on the submit path, so keep the queue as shallow as possible where the driver burned
            // us. Elsewhere the wgpu default (2) stands — DLSS-G frame pacing depends on it.
            desired_maximum_frame_latency: if info.backend == wgpu::Backend::Metal { 1 } else { 2 },
        };
        surface.configure(&device, &config);
        let depth_view = create_depth(&device, &config);

        // The instance must outlive the surface; the surface holds an Arc to the window and
        // an internal reference, so dropping `_instance` here is fine (kept implicitly alive).
        let gpu = Self {
            surface,
            device,
            queue,
            config,
            depth_view,
            size,
            render_scale,
            backend: info.backend,
            rt_enabled,
        };
        (gpu, frame_gen)
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.size = size;
            let render_size = scaled_size(size, self.render_scale);
            self.config.width = render_size.width;
            self.config.height = render_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth(&self.device, &self.config);
        }
    }

    /// Re-resolve the render scale when the window moves to a display with a different DPI scale.
    /// winit fires `ScaleFactorChanged` then a `Resized`, which applies the new scale via `resize`.
    pub fn rescale(&mut self, scale_factor: f64) {
        self.render_scale = resolve_render_scale(self.backend, scale_factor);
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }
}

/// Fail-safe against GPU/driver wedges (2026-06-03 incident: the first windowed Metal run hung the
/// AGXG15G firmware irrecoverably — WindowServer froze and the machine needed a power-button reset).
/// The app cannot un-wedge a driver, but it can stop feeding a queue that has stopped signaling:
/// every presented frame `arm`s an `on_submitted_work_done` callback, and `check` (called each
/// frame) reports when submitted work has made no completion progress for the timeout. The caller
/// is expected to save the world and exit *without* touching the GPU again. Tunable via
/// `VOXELCRAFT_GPU_WATCHDOG` (seconds, `0` disables; default 5).
pub struct GpuWatchdog {
    /// Highest armed frame index the GPU has signaled complete (written from wgpu's callback).
    completed: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Frames armed so far; `seen < submitted` means work is outstanding.
    submitted: u64,
    seen: u64,
    /// When the current no-progress window started, if work is outstanding.
    stall_since: Option<std::time::Instant>,
    timeout_secs: f32,
}

impl GpuWatchdog {
    pub fn new() -> Self {
        let timeout_secs = std::env::var("VOXELCRAFT_GPU_WATCHDOG")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(5.0);
        Self {
            completed: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            submitted: 0,
            seen: 0,
            stall_since: None,
            timeout_secs,
        }
    }

    /// Register completion tracking for the frame just submitted. Call once per presented frame,
    /// after its `queue.submit`.
    pub fn arm(&mut self, queue: &wgpu::Queue) {
        if self.timeout_secs <= 0.0 {
            return;
        }
        self.submitted += 1;
        let n = self.submitted;
        let completed = self.completed.clone();
        queue.on_submitted_work_done(move || {
            completed.fetch_max(n, std::sync::atomic::Ordering::Release);
        });
    }

    /// Poll completion progress; returns the stall duration once outstanding work has not advanced
    /// for the timeout. A non-blocking device poll drives wgpu's callbacks.
    pub fn check(&mut self, device: &wgpu::Device) -> Option<f32> {
        if self.timeout_secs <= 0.0 || self.submitted == 0 {
            return None;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let completed = self.completed.load(std::sync::atomic::Ordering::Acquire);
        if completed > self.seen {
            // Progress — restart the stall window.
            self.seen = completed;
            self.stall_since = None;
        }
        if self.seen >= self.submitted {
            self.stall_since = None;
            return None;
        }
        let since = *self.stall_since.get_or_insert_with(std::time::Instant::now);
        let stalled = since.elapsed().as_secs_f32();
        (stalled >= self.timeout_secs).then_some(stalled)
    }
}

/// Effective swapchain scale: `VOXELCRAFT_RENDER_SCALE` (clamped 0.25..=1.0, a fraction of the
/// window's PHYSICAL size) when set, else `METAL_LOGICAL_SCALE / scale_factor` on macOS — i.e. a
/// tier-tuned fraction of *logical* resolution, DPI-independent (P21: 0.70× logical on the M3;
/// 1.0 would be the pre-P21 logical-res default). Metal-only: CAMetalLayer composites a smaller
/// drawable stretched to the layer bounds (wgpu sets `drawableSize` from the surface config and the
/// CALayer default gravity is resize); DXGI/Vulkan make no such guarantee, so every other backend
/// is forced to 1.0 (it also keeps the knob away from the DLSS output-resolution path).
fn resolve_render_scale(backend: wgpu::Backend, scale_factor: f64) -> f32 {
    let env = std::env::var("VOXELCRAFT_RENDER_SCALE")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok());
    if backend != wgpu::Backend::Metal {
        if env.is_some_and(|s| s != 1.0) {
            log::warn!("VOXELCRAFT_RENDER_SCALE is Metal-only (drawable-stretch); ignoring on {backend:?}");
        }
        return 1.0;
    }
    let scale = env.unwrap_or_else(|| {
        if cfg!(target_os = "macos") {
            (crate::quality::METAL_LOGICAL_SCALE as f64 / scale_factor.max(0.1)) as f32
        } else {
            1.0
        }
    });
    scale.clamp(0.25, 1.0)
}

/// The swapchain size for a window of `size` physical pixels at `scale` (each axis floored, min 1).
fn scaled_size(size: winit::dpi::PhysicalSize<u32>, scale: f32) -> winit::dpi::PhysicalSize<u32> {
    winit::dpi::PhysicalSize::new(
        ((size.width as f32 * scale) as u32).max(1),
        ((size.height as f32 * scale) as u32).max(1),
    )
}

fn create_depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

async fn init_adapter(
    window: Arc<Window>,
) -> (wgpu::Instance, wgpu::Surface<'static>, wgpu::Adapter) {
    for backends in backend_order() {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        // Use DXC on DX12 (dxcompiler.dll staged next to the exe by build.rs) — FXC cannot compile
        // ray-tracing shaders, so without DXC the DX12 backend never advertises
        // EXPERIMENTAL_RAY_QUERY. Ignored by the Vulkan/GL backends. WGPU_DX12_COMPILER overrides.
        desc.backend_options.dx12.shader_compiler =
            wgpu::Dx12Compiler::from_env().unwrap_or(wgpu::Dx12Compiler::Auto);
        let instance = wgpu::Instance::new(desc);
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            return (instance, surface, adapter);
        }
    }
    panic!("No compatible GPU adapter found (tried, in order: {:?})", backend_order());
}

/// Backend preference order, platform-aware: **DX12** first on Windows/Linux (hardware RT via DXC;
/// best visual parity on the dev box) with Vulkan/GL fallbacks; **Metal** first on macOS (the only
/// native backend — no hardware RT, so the software DDA tracer runs; Vulkan/GL only matter there
/// via MoltenVK/ANGLE and are skipped harmlessly when absent). Override with
/// `VOXELCRAFT_BACKEND=dx12|vulkan|gl|metal` (a backend the OS lacks just fails adapter selection).
pub(crate) fn backend_order() -> [wgpu::Backends; 3] {
    use wgpu::Backends as B;
    #[cfg(target_os = "macos")]
    let default = [B::METAL, B::VULKAN, B::GL];
    #[cfg(not(target_os = "macos"))]
    let default = [B::DX12, B::VULKAN, B::GL];
    // The explicit-override arms fold METAL into the trailing fallback flag set so a macOS user
    // forcing a backend the OS lacks still lands on Metal instead of panicking; METAL is compiled
    // out of wgpu off-Apple (cfg target_vendor), so the extra flag is inert on Windows/Linux.
    match std::env::var("VOXELCRAFT_BACKEND").ok().as_deref() {
        Some("vulkan") | Some("vk") => [B::VULKAN, B::DX12, B::GL | B::METAL],
        Some("gl") | Some("opengl") => [B::GL, B::DX12, B::VULKAN | B::METAL],
        Some("metal") | Some("mtl") => [B::METAL, B::VULKAN, B::GL],
        Some("dx12") | Some("d3d12") => [B::DX12, B::VULKAN, B::GL | B::METAL],
        None => default,
        Some(other) => {
            log::warn!("unknown VOXELCRAFT_BACKEND={other:?}; using platform default {default:?}");
            default
        }
    }
}
