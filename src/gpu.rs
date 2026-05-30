//! GPU context: instance/surface/device/queue, swapchain config, and a depth buffer.
//! Adapter selection prefers DX12 on this NVIDIA/Windows box (faster than Vulkan in wgpu),
//! falling back to Vulkan then GL.

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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("voxelcraft-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to create wgpu device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
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
    for backends in [
        wgpu::Backends::DX12,
        wgpu::Backends::VULKAN,
        wgpu::Backends::GL,
    ] {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
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
    panic!("No compatible GPU adapter found (tried DX12, Vulkan, GL)");
}
