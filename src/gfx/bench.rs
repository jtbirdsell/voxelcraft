//! Headless GPU benchmark (P21): render N frames offscreen on the screenshot path and report
//! per-frame CPU/GPU timings — the measurement tool for tuning the macOS quality tier without a
//! windowed run (which is banned on the Mac after the 2026-06-03 AGX wedge; see CLAUDE.md).
//!
//! `VOXELCRAFT_BENCH=N` renders N timed frames (plus warm-up) into a private offscreen target —
//! the swapchain/present path is never touched. Per-pass GPU timings come from timestamp queries
//! written at pass boundaries (`PassTimestampWrites`): Apple GPUs only support counter sampling
//! at stage boundaries, so this is the one portable mechanism (DX12/Vulkan accept it too). When
//! the adapter lacks `TIMESTAMP_QUERY` the harness degrades to whole-frame wall time.
//!
//! Knobs: `VOXELCRAFT_BENCH_WARMUP` (default max(3, N/10)), `VOXELCRAFT_BENCH_LABEL` (tag echoed
//! in the report), `VOXELCRAFT_BENCH_CSV` (append one row per run), `VOXELCRAFT_BENCH_GPU=0`
//! (force wall-time-only). NOTE: in bench mode the headless GI-ray default is the *interactive*
//! default (8), not the screenshot default (64), so out-of-the-box numbers reflect gameplay cost.

use std::cell::Cell;
use std::io::Write as _;
use std::time::{Duration, Instant};

use glam::IVec3;

use crate::camera::CameraUniform;
use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::overlay::UiVertex;
use crate::renderer::{ChunkRenderer, GpuMesh};

/// Per-frame GPU stall deadline. A frame that takes this long is a wedge, not a slow frame —
/// fail fast instead of spinning against a dead queue (the machine has form here).
const FRAME_DEADLINE: Duration = Duration::from_secs(30);

/// The passes timed inside `ChunkRenderer::record`, in execution order. Tonemap/HUD/highlight are
/// not individually timed (they live in `record_resolve`/`record_highlight`); they show up in the
/// derived `post` bucket = whole-frame GPU time − sum of timed passes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Opaque = 0,
    GiCompute = 1,
    GiTemporal = 2,
    GiComposite = 3,
    Glass = 4,
    Water = 5,
}

pub const PASS_COUNT: usize = 6;
const PASS_NAMES: [&str; PASS_COUNT] =
    ["opaque", "gi_compute", "gi_temporal", "gi_composite", "glass", "water"];
const QUERY_SLOTS: u32 = (PASS_COUNT * 2) as u32;

/// Owns the timestamp query set + resolve/readback buffers, reused across frames. `None` when the
/// device lacks `TIMESTAMP_QUERY` (feature is only requested in bench mode — see device.rs).
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    /// Ticks → nanoseconds.
    period: f32,
    /// Bitmask of passes whose begin/end timestamps were armed this frame (a pass that never
    /// begins — e.g. gi_temporal with accumulation off — leaves stale slots; ignore them).
    armed: Cell<u16>,
}

impl GpuTimer {
    pub fn new(gpu: &Gpu) -> Option<Self> {
        if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bench-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_SLOTS,
        });
        let size = (QUERY_SLOTS as u64 * wgpu::QUERY_SIZE as u64)
            .next_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);
        let resolve_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench-ts-resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench-ts-read"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buf,
            read_buf,
            period: gpu.queue.get_timestamp_period(),
            armed: Cell::new(0),
        })
    }

    /// Forget last frame's armed passes. Call once per frame before recording.
    pub fn begin_frame(&self) {
        self.armed.set(0);
    }

    fn arm(&self, pass: Pass) -> (u32, u32) {
        self.armed.set(self.armed.get() | 1 << pass as u16);
        let base = (pass as u32) * 2;
        (base, base + 1)
    }

    /// Timestamp writes for a render pass (begin/end at the pass's stage boundaries).
    pub fn render_writes(&self, pass: Pass) -> wgpu::RenderPassTimestampWrites<'_> {
        let (begin, end) = self.arm(pass);
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        }
    }

    /// Timestamp writes for a compute pass.
    pub fn compute_writes(&self, pass: Pass) -> wgpu::ComputePassTimestampWrites<'_> {
        let (begin, end) = self.arm(pass);
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        }
    }

    /// Resolve this frame's queries into the readback buffer. Record after the timed passes,
    /// inside the same encoder/submission.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..QUERY_SLOTS, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buf,
            0,
            &self.read_buf,
            0,
            QUERY_SLOTS as u64 * wgpu::QUERY_SIZE as u64,
        );
    }

    /// Read back this frame's per-pass durations (ns), `None` for passes that never ran. Call
    /// after the frame's submission has completed (the bench loop waits per frame).
    pub fn read(&self, gpu: &Gpu) -> [Option<u64>; PASS_COUNT] {
        let slice = self.read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(FRAME_DEADLINE),
        });
        let mapped = rx
            .recv_timeout(FRAME_DEADLINE)
            .expect("bench: timestamp map timed out")
            .is_ok();
        let mut out = [None; PASS_COUNT];
        if mapped {
            let data = slice.get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            let armed = self.armed.get();
            for i in 0..PASS_COUNT {
                if armed & (1 << i) != 0 {
                    let (b, e) = (ticks[i * 2], ticks[i * 2 + 1]);
                    out[i] = Some((e.saturating_sub(b) as f64 * self.period as f64) as u64);
                }
            }
            drop(data);
            self.read_buf.unmap();
        }
        out
    }
}

struct FrameTiming {
    /// Wall time to build + submit the frame's encoder (CPU).
    cpu_record_ns: u64,
    /// Submit → on_submitted_work_done wall time (the whole-frame GPU verdict).
    gpu_frame_ns: u64,
    pass_ns: [Option<u64>; PASS_COUNT],
}

/// Render `frames` timed frames (plus warm-up) offscreen and print a timing report. Mirrors
/// `capture::screenshot`'s offscreen setup (own color/depth, never the surface), minus readback.
#[allow(clippy::too_many_arguments)]
pub fn run(
    gpu: &Gpu,
    renderer: &ChunkRenderer,
    meshes: &[&GpuMesh],
    volume_bg: &wgpu::BindGroup,
    highlight: Option<(IVec3, f32)>,
    ui_verts: &[UiVertex],
    width: u32,
    height: u32,
    camera_uniform: &CameraUniform,
    frames: usize,
) {
    let device = &gpu.device;
    let frames = frames.max(1);
    let warmup = std::env::var("VOXELCRAFT_BENCH_WARMUP")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or_else(|| 3.max(frames / 10));
    let label = std::env::var("VOXELCRAFT_BENCH_LABEL").unwrap_or_else(|_| "bench".into());

    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench-color"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench-depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let targets = renderer.make_targets(device, width, height);
    // One-shot world TLAS for hardware shadow rays, exactly like the screenshot path.
    let mut rt_scene = crate::gfx::rt::RtScene::new();
    let world_tlas = if renderer.use_hw_rt() {
        rt_scene.rebuild(gpu, meshes)
    } else {
        None
    };
    let as_bg = world_tlas.map(|t| renderer.make_as_bind_group(device, t));
    renderer.update_camera(gpu, camera_uniform);

    let timer = if std::env::var("VOXELCRAFT_BENCH_GPU").ok().as_deref() == Some("0") {
        None
    } else {
        GpuTimer::new(gpu)
    };

    let mut samples: Vec<FrameTiming> = Vec::with_capacity(frames);
    for i in 0..warmup + frames {
        if let Some(t) = &timer {
            t.begin_frame();
        }
        let t0 = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bench-encoder"),
        });
        renderer.record_full(
            gpu,
            &mut encoder,
            &targets,
            &color_view,
            &depth_view,
            meshes,
            volume_bg,
            as_bg.as_ref(),
            highlight,
            ui_verts,
            None,       // hudless_view (FG only)
            None,       // ui_view (FG only)
            None,       // viewmodel (no first-person model in the bench)
            timer.as_ref(),
        );
        if let Some(t) = &timer {
            t.resolve(&mut encoder);
        }
        let buf = encoder.finish();
        let cpu_record_ns = t0.elapsed().as_nanos() as u64;

        // Whole-frame GPU wall time: submit → completion callback, with a hard deadline so a
        // wedged queue aborts the bench instead of hanging the machine's only safe test path.
        let g0 = Instant::now();
        let (tx, rx) = std::sync::mpsc::channel();
        gpu.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        gpu.queue.submit(Some(buf));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(FRAME_DEADLINE),
        });
        if rx.recv_timeout(FRAME_DEADLINE).is_err() {
            log::error!(
                "bench: frame {i} did not complete within {FRAME_DEADLINE:?} — GPU appears wedged; aborting"
            );
            std::process::exit(70);
        }
        let gpu_frame_ns = g0.elapsed().as_nanos() as u64;

        let pass_ns = timer
            .as_ref()
            .map(|t| t.read(gpu))
            .unwrap_or([None; PASS_COUNT]);
        if i >= warmup {
            samples.push(FrameTiming { cpu_record_ns, gpu_frame_ns, pass_ns });
        }
    }

    report(gpu, &label, width, height, warmup, &samples, timer.is_some());
}

/// (p50, p95, min, max) of a metric across frames, in milliseconds.
fn stats(mut v: Vec<u64>) -> Option<(f64, f64, f64, f64)> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let ms = |n: u64| n as f64 / 1e6;
    let p = |q: f64| ms(v[((v.len() - 1) as f64 * q).round() as usize]);
    Some((p(0.50), p(0.95), ms(v[0]), ms(*v.last().unwrap())))
}

fn report(
    gpu: &Gpu,
    label: &str,
    width: u32,
    height: u32,
    warmup: usize,
    samples: &[FrameTiming],
    per_pass: bool,
) {
    let gpu_stats = stats(samples.iter().map(|f| f.gpu_frame_ns).collect()).unwrap();
    let cpu_stats = stats(samples.iter().map(|f| f.cpu_record_ns).collect()).unwrap();
    let fps = 1000.0 / gpu_stats.0;
    let budget = 16.6 - gpu_stats.0;

    println!(
        "== VOXELCRAFT BENCH ==  label={label} backend={:?} res={width}x{height} frames={} warmup={warmup} gpu_timing={}",
        gpu.backend,
        samples.len(),
        if per_pass { "per-pass(stage-boundary)" } else { "frame-wall-only" },
    );
    println!("metric            p50(ms)  p95(ms)   min(ms)   max(ms)");
    println!(
        "gpu_frame        {:8.2} {:8.2} {:9.2} {:9.2}   -> {fps:.1} FPS (16.6ms budget: {budget:+.1}ms)",
        gpu_stats.0, gpu_stats.1, gpu_stats.2, gpu_stats.3
    );
    println!(
        "cpu_record       {:8.2} {:8.2} {:9.2} {:9.2}",
        cpu_stats.0, cpu_stats.1, cpu_stats.2, cpu_stats.3
    );
    let mut pass_p50 = [None::<(f64, f64, f64, f64)>; PASS_COUNT];
    if per_pass {
        let mut timed_sum_p50 = 0.0;
        for i in 0..PASS_COUNT {
            let s = stats(samples.iter().filter_map(|f| f.pass_ns[i]).collect());
            if let Some(s) = s {
                println!(
                    "  {:<14} {:8.2} {:8.2} {:9.2} {:9.2}",
                    PASS_NAMES[i], s.0, s.1, s.2, s.3
                );
                timed_sum_p50 += s.0;
            }
            pass_p50[i] = s;
        }
        // Whatever the timed passes don't account for: tonemap + HUD + highlight + barriers +
        // submit overhead. On TBDR, pass boundaries include tile load/store, so treat per-pass
        // numbers as relative weights and gpu_frame as the absolute verdict.
        println!("  {:<14} {:8.2}", "post(derived)", (gpu_stats.0 - timed_sum_p50).max(0.0));
    }

    if let Ok(csv) = std::env::var("VOXELCRAFT_BENCH_CSV") {
        let new = !std::path::Path::new(&csv).exists();
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&csv) {
            if new {
                let _ = writeln!(
                    f,
                    "label,backend,width,height,frames,gpu_frame_p50_ms,gpu_frame_p95_ms,cpu_record_p50_ms,{},fps",
                    PASS_NAMES.map(|n| format!("{n}_p50_ms")).join(",")
                );
            }
            let passes = pass_p50
                .iter()
                .map(|s| s.map_or(String::new(), |s| format!("{:.3}", s.0)))
                .collect::<Vec<_>>()
                .join(",");
            let _ = writeln!(
                f,
                "{label},{:?},{width},{height},{},{:.3},{:.3},{:.3},{passes},{fps:.2}",
                gpu.backend,
                samples.len(),
                gpu_stats.0,
                gpu_stats.1,
                cpu_stats.0,
            );
            log::info!("bench: appended results to {csv}");
        }
    }
}
