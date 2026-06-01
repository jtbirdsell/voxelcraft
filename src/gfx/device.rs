//! GPU context: instance/surface/device/queue, swapchain config, and a depth buffer.
//! Adapter selection defaults to **DX12** — better visual parity and Windows windowing on this box,
//! with hardware RT via the DXC compiler (staged next to the exe by build.rs). Vulkan is the
//! fallback: it also has hardware RT and is the only backend wgpu exposes DLSS on, but DX12 renders
//! better here so it leads. Override with `VOXELCRAFT_BACKEND=dx12|vulkan|gl`. GL has no hardware RT
//! (software DDA tracer). The device opts into hardware ray tracing whenever the adapter advertises it.

use std::sync::Arc;
use winit::window::Window;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Gpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_view: wgpu::TextureView,
    pub size: winit::dpi::PhysicalSize<u32>,
    /// Which wgpu backend the adapter resolved to (Vulkan/Dx12/Gl).
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
        let rt_enabled = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
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
        let (required_limits, experimental_features) = if rt_enabled {
            // Default limits zero out the acceleration-structure caps; adopt the adapter's real
            // maxima (matches the validated rt_spike recipe). SAFETY: explicit acknowledgment that
            // wgpu's hardware-RT path is experimental; we pin wgpu to de-risk it (see Cargo.toml).
            (adapter.limits(), unsafe {
                wgpu::ExperimentalFeatures::enabled()
            })
        } else {
            (
                wgpu::Limits::default(),
                wgpu::ExperimentalFeatures::disabled(),
            )
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
        // M33-G8-FG: create the DLSS-G context AFTER the device but BEFORE `surface.configure`
        // (`slSetD3DDevice` must precede swapchain creation). `None` unless FG was requested + bound.
        let frame_gen = crate::frame_gen::FrameGen::create(
            streamline,
            &device,
            &adapter,
            format,
            size.width.max(1),
            size.height.max(1),
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
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
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
            backend: info.backend,
            rt_enabled,
        };
        (gpu, frame_gen)
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.size = size;
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth(&self.device, &self.config);
        }
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }
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

/// Backend preference order, defaulting to **DX12** (hardware RT via DXC; best visual parity here)
/// with Vulkan/GL as fallbacks. Override with `VOXELCRAFT_BACKEND=dx12|vulkan|gl`.
pub(crate) fn backend_order() -> [wgpu::Backends; 3] {
    use wgpu::Backends as B;
    match std::env::var("VOXELCRAFT_BACKEND").ok().as_deref() {
        Some("vulkan") | Some("vk") => [B::VULKAN, B::DX12, B::GL],
        Some("gl") | Some("opengl") => [B::GL, B::DX12, B::VULKAN],
        Some("dx12") | Some("d3d12") | None => [B::DX12, B::VULKAN, B::GL],
        Some(other) => {
            log::warn!("unknown VOXELCRAFT_BACKEND={other:?}; defaulting to dx12");
            [B::DX12, B::VULKAN, B::GL]
        }
    }
}
