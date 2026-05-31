//! Hardware-ray-tracing capability probe (`VOXELCRAFT_RT_PROBE=1`). Enumerates the GPU on each
//! backend and reports whether wgpu exposes the experimental ray-query / acceleration-structure
//! features — the go/no-go for moving the RTX pipeline onto the 4090's RT cores. Throwaway/no window.
//!
//! Corrected premise (2026-05): wgpu's DX12 backend DID ship DXR (PR #6777, v24), but DX12
//! ray-tracing shaders require the **DXC** compiler — the default **FXC** cannot compile them, and
//! wgpu only advertises `EXPERIMENTAL_RAY_QUERY` on DX12 when DXC is available. An earlier spike's
//! "DX12 RT = false" was therefore most likely an FXC artifact, not an API limit. We probe DX12
//! twice (FXC vs DXC) to document the reality on this box. We ship on the **Vulkan** backend
//! regardless (it is the proven RT path *and* the only one DLSS supports), so the DX12 result is
//! informational only. `WGPU_DX12_COMPILER=dxc|fxc|auto` overrides the DXC pass.

pub fn probe() {
    // The proven path: the rt_spike renders hardware ray tracing here.
    probe_one(wgpu::Backends::VULKAN, None, "Vulkan");
    // DX12 with the legacy FXC compiler — expected NOT to expose ray query (FXC can't compile it).
    probe_one(wgpu::Backends::DX12, Some(wgpu::Dx12Compiler::Fxc), "DX12 + FXC");
    // DX12 asking for DXC: honour WGPU_DX12_COMPILER if set, else Auto (tries static DXC, then DXC
    // on PATH, then cleanly falls back to FXC — so an adapter is still produced when DXC is absent,
    // unlike DynamicDxc which hard-fails on a missing dxcompiler.dll). The honest "does DX12 RT
    // light up with DXC on this 4090?" test.
    let dxc = wgpu::Dx12Compiler::from_env().unwrap_or(wgpu::Dx12Compiler::Auto);
    probe_one(wgpu::Backends::DX12, Some(dxc), "DX12 + DXC(auto)");
}

fn probe_one(backends: wgpu::Backends, dx12_compiler: Option<wgpu::Dx12Compiler>, label: &str) {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    if let Some(compiler) = dx12_compiler {
        desc.backend_options.dx12.shader_compiler = compiler;
    }
    let instance = wgpu::Instance::new(desc);
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
                "RT_PROBE [{label}] {} (driver {}): RAY_QUERY={rq} HIT_VERTEX_RETURN={vr}",
                info.name,
                info.driver_info
            );
        }
        Err(e) => log::info!("RT_PROBE [{label}]: no adapter ({e})"),
    }
}
