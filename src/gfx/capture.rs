//! Offscreen screenshot: render one frame to a texture and read it back to a PNG.
//! Used for headless milestone verification (set VOXELCRAFT_SHOT=path.png).

use glam::IVec3;

use crate::camera::CameraUniform;
use crate::dlss::DlssRender;
use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::overlay::UiVertex;
use crate::renderer::{ChunkRenderer, GpuMesh, GpuPart, ViewModelDraw};

#[allow(clippy::too_many_arguments)]
pub fn screenshot(
    gpu: &Gpu,
    renderer: &ChunkRenderer,
    meshes: &[&GpuMesh],
    volume_bg: &wgpu::BindGroup,
    highlight: Option<(IVec3, f32)>,
    ui_verts: &[UiVertex],
    width: u32,
    height: u32,
    // The UN-jittered camera uniform; the DLSS path re-applies the per-frame jitter itself.
    camera_uniform: &CameraUniform,
    // M34-VM: the first-person view-model geometry + uniform for this shot (built by the caller).
    viewmodel: Option<(&GpuPart, crate::viewmodel::ViewModelUniform)>,
    dlss: Option<&mut DlssRender>,
    path: &str,
) {
    let device = &gpu.device;
    let format = gpu.config.format;

    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());

    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot-depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    // M34-VM: the view-model's private depth (output res), cleared each pass like the live path's.
    let vm_depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot-vm-depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let vm_depth_view = vm_depth.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row =
        unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shot-readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // DLSS renders the scene at the reduced render resolution (then upscales to `width`x`height`);
    // native renders straight at output resolution.
    let targets = match &dlss {
        Some(d) => d.make_render_targets(renderer, device),
        None => renderer.make_targets(device, width, height),
    };
    // Build a one-shot world TLAS for hardware shadow rays (M33-G5), bound at group 3.
    let mut rt_scene = crate::gfx::rt::RtScene::new();
    let world_tlas = if renderer.use_hw_rt() {
        rt_scene.rebuild(gpu, meshes)
    } else {
        None
    };
    let as_bg = world_tlas.map(|t| renderer.make_as_bind_group(device, t));

    // Copy the shot color texture → readback buffer, decode (BGRA→RGBA if needed), save to `p`.
    let swap_rb = matches!(
        format,
        wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm
    );
    let save_png = |p: &str| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shot-copy-encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map channel closed").expect("buffer map failed");

        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for y in 0..height as usize {
            let src_off = y * padded_bytes_per_row as usize;
            let src_row = &data[src_off..src_off + unpadded_bytes_per_row as usize];
            let dst_off = y * (width * 4) as usize;
            let dst_row = &mut rgba[dst_off..dst_off + (width * 4) as usize];
            if swap_rb {
                for x in 0..width as usize {
                    dst_row[x * 4] = src_row[x * 4 + 2];
                    dst_row[x * 4 + 1] = src_row[x * 4 + 1];
                    dst_row[x * 4 + 2] = src_row[x * 4];
                    dst_row[x * 4 + 3] = src_row[x * 4 + 3];
                }
            } else {
                dst_row.copy_from_slice(src_row);
            }
        }
        drop(data);
        readback.unmap();
        image::save_buffer(p, &rgba, width, height, image::ColorType::Rgba8)
            .expect("failed to save screenshot PNG");
        log::info!("Saved screenshot: {p} ({width}x{height})");
    };

    // render_into submits internally (the DLSS path interleaves the RR evaluate between scene +
    // resolve), so the readback runs in its own encoder after. DLSS: warm up a few frames so NGX's
    // first-evaluate network (re)creation settles before capture, and apply the SAME per-frame jitter
    // NGX is told (advancing each frame) so the headless sequence matches interactive — otherwise a
    // fixed camera jitter vs an advancing NGX jitter desyncs the reconstruction.
    // VOXELCRAFT_DLSS_FRAMES saves every warm-up frame (`<path>.N.png`) to diff for temporal stability.
    let warmup = if dlss.is_some() {
        std::env::var("VOXELCRAFT_DLSS_WARMUP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6)
    } else if renderer.gi_accum() {
        // P21: with GI temporal accumulation on (the mac tier), the first frame's irradiance is
        // raw low-ray noise — run enough frames for the EMA to converge so the PNG shows the
        // steady-state interactive look (the camera is static here, so frames are identical work).
        16
    } else {
        1
    };
    let save_frames = std::env::var("VOXELCRAFT_DLSS_FRAMES").is_ok();
    let mut dlss = dlss;
    for i in 0..warmup {
        if let Some(d) = dlss.as_deref() {
            let mut u = *camera_uniform;
            let (rw, rh) = d.render_dims();
            u.apply_jitter(d.jitter(), rw, rh);
            renderer.update_camera(gpu, &u);
        }
        let vmd = viewmodel.map(|(mesh, uniform)| ViewModelDraw {
            mesh,
            depth: &vm_depth_view,
            uniform,
        });
        renderer.render_into(
            gpu,
            &targets,
            &color_view,
            &depth_view,
            meshes,
            volume_bg,
            as_bg.as_ref(),
            highlight,
            ui_verts,
            None,
            None,
            vmd.as_ref(),
            dlss.as_deref_mut(),
        );
        if save_frames {
            save_png(&format!("{path}.{i}.png"));
        }
    }
    save_png(path);
}
