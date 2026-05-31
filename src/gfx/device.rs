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
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
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
        let (required_features, required_limits, experimental_features) = if rt_enabled {
            (
                wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                // Default limits zero out the acceleration-structure caps; adopt the adapter's real
                // maxima (matches the validated rt_spike recipe).
                adapter.limits(),
                // SAFETY: explicit acknowledgment that wgpu's hardware-RT path is experimental and
                // may carry bugs / breaking changes. We pin wgpu to de-risk this (see Cargo.toml).
                unsafe { wgpu::ExperimentalFeatures::enabled() },
            )
        } else {
            (
                wgpu::Features::empty(),
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
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth_view = create_depth(&device, &config);

        // The instance must outlive the surface; the surface holds an Arc to the window and
        // an internal reference, so dropping `_instance` here is fine (kept implicitly alive).
        Self {
            surface,
            device,
            queue,
            config,
            depth_view,
            size,
            backend: info.backend,
            rt_enabled,
        }
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
fn backend_order() -> [wgpu::Backends; 3] {
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
