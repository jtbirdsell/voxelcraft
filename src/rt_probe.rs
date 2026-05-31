//! Hardware-ray-tracing capability probe (`VOXELCRAFT_RT_PROBE=1`). Enumerates the GPU on each
//! backend and reports whether wgpu exposes the experimental ray-query / acceleration-structure
//! features — the go/no-go for moving the RTX pipeline onto the 4090's RT cores. Throwaway/no window.

pub fn probe() {
    for backends in [wgpu::Backends::DX12, wgpu::Backends::VULKAN] {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        match adapter {
            Ok(a) => {
                let info = a.get_info();
                let feats = a.features();
                let rq = feats.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
                let vr = feats.contains(wgpu::Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN);
                log::info!(
                    "RT_PROBE [{backends:?}] {} (driver {}): RAY_QUERY={rq} HIT_VERTEX_RETURN={vr}",
                    info.name,
                    info.driver_info
                );
            }
            Err(e) => log::info!("RT_PROBE [{backends:?}]: no adapter ({e})"),
        }
    }
}
