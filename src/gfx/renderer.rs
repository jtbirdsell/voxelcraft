//! Chunk render pipeline: a camera uniform bind group + a textured/lit triangle pipeline,
//! plus helpers to upload a `MeshData` to GPU buffers and draw it.

use std::cell::Cell;

use glam::IVec3;
use wgpu::util::DeviceExt;

use crate::camera::CameraUniform;
use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::mesher::{Geometry, MeshData, Vertex};
use crate::gfx::graph::{
    RenderTargets, GALBEDO_FORMAT, GDEPTH_FORMAT, GMOTION_FORMAT, GNORMAL_FORMAT, GPOS_FORMAT,
    HDR_FORMAT, IRRADIANCE_FORMAT,
};
use crate::overlay::{self, LineVertex, UiVertex};
use crate::voxel_volume::VoxelVolume;

fn upload_geometry(device: &wgpu::Device, geom: &Geometry, blas_input: bool) -> Option<GpuPart> {
    if geom.is_empty() {
        return None;
    }
    // When hardware RT is available, the buffers double as BLAS inputs (M33-G4). Gated because the
    // BLAS_INPUT usage is only valid on a device created with the ray-query feature.
    let (vusage, iusage) = if blas_input {
        (
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        )
    } else {
        (wgpu::BufferUsages::VERTEX, wgpu::BufferUsages::INDEX)
    };
    Some(GpuPart {
        vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-vbuf"),
            contents: bytemuck::cast_slice(&geom.vertices),
            usage: vusage,
        }),
        index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-ibuf"),
            contents: bytemuck::cast_slice(&geom.indices),
            usage: iusage,
        }),
        vertex_count: geom.vertices.len() as u32,
        index_count: geom.indices.len() as u32,
    })
}

pub struct GpuPart {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
}

pub struct GpuMesh {
    pub opaque: Option<GpuPart>,
    pub water: Option<GpuPart>,
    pub translucent: Option<GpuPart>,
    /// Hardware-RT bottom-level acceleration structure built from the opaque geometry (M33-G5),
    /// referenced by the per-frame TLAS. `None` when RT is unavailable or the chunk is empty.
    pub blas: Option<wgpu::Blas>,
}

pub struct ChunkRenderer {
    pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    glass_pipeline: wgpu::RenderPipeline,
    highlight_pipeline: wgpu::RenderPipeline,
    ui_pipeline: wgpu::RenderPipeline,
    tonemap_pipeline: wgpu::RenderPipeline,
    tonemap_bgl: wgpu::BindGroupLayout,
    tonemap_sampler: wgpu::Sampler,
    /// M33-G8-FG (Phase 3): nearest-upscale render-res depth + motion → output-res, so DLSS Frame
    /// Generation can tag output-resolution guides for the RR-upscaled frame.
    guide_upscale_pipeline: wgpu::RenderPipeline,
    guide_upscale_bgl: wgpu::BindGroupLayout,
    /// M33-G9 supersampling: Catmull-Rom downscale (super-res LDR → swapchain) + bidirectional guide
    /// resize (super-res depth/motion → swap-res FG guides).
    ss_downscale_pipeline: wgpu::RenderPipeline,
    guide_downscale_pipeline: wgpu::RenderPipeline,
    ui_bind_group: wgpu::BindGroup,
    atlas_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    volume_bgl: wgpu::BindGroupLayout,
    as_bgl: Option<wgpu::BindGroupLayout>,
    use_hw_rt: bool,
    /// M33-G6: when set (VOXELCRAFT_GI != "fragment", the default), the hemisphere GI is gathered in
    /// the compute pass + composite below instead of inline in the chunk fragment. The pipelines /
    /// layouts are `Some` only in this mode.
    defer_gi: bool,
    gi_pipeline: Option<wgpu::ComputePipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    gi_io_bgl: Option<wgpu::BindGroupLayout>,
    composite_bgl: Option<wgpu::BindGroupLayout>,
    /// M33-G9 (opt-in, `VOXELCRAFT_GI_ACCUM=1`): temporally accumulate the GI irradiance before the
    /// composite (reproject prev via motion vectors + EMA blend, disocclusion-rejected). Default off
    /// so it can't ghost the already-clean supersampled default; the pipeline/layout are `Some` only
    /// when both deferred GI and this are enabled.
    gi_accum: bool,
    gi_temporal_bgl: Option<wgpu::BindGroupLayout>,
    gi_temporal_pipeline: Option<wgpu::ComputePipeline>,
    /// M34-VM first-person view-model: a tiny LDR 3D pass drawn AFTER the DLSS resolve + ACES tonemap
    /// (so it never touches the HDR scene / G-buffer / motion guides). Has its own output-resolution
    /// depth, cleared each pass + recreated on resize.
    viewmodel_pipeline: wgpu::RenderPipeline,
    viewmodel_buffer: wgpu::Buffer,
    viewmodel_bind_group: wgpu::BindGroup,
    viewmodel_depth_view: wgpu::TextureView,
    /// P21: GI compute workgroup shape — must match the `GI_WG_X/Y` consts injected into the
    /// gi_compute + gi_temporal shaders (one source: ChunkRenderer::new). (8,4) on Metal, (8,8) else.
    gi_wg: (u32, u32),
    sky_color: Cell<wgpu::Color>,
    /// Consecutive swapchain-acquire failures (2026-06-03 wedge hardening): after a few, each retry
    /// backs off so a persistently broken surface is never hammered with reconfigure/acquire spins.
    /// Reset on every successful acquire.
    surface_errors: Cell<u32>,
}

/// One frame's view-model draw: the (already view-space) geometry, the projection + brightness
/// uniform, and the private depth target to clear + test against.
pub struct ViewModelDraw<'a> {
    pub mesh: &'a GpuPart,
    pub depth: &'a wgpu::TextureView,
    pub uniform: crate::viewmodel::ViewModelUniform,
}

/// Create the view-model's private depth texture at the given output resolution.
fn make_viewmodel_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewmodel-depth"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

impl ChunkRenderer {
    pub fn new(gpu: &Gpu, quality: &crate::quality::Quality) -> Self {
        let device = &gpu.device;

        // Both the chunk and water shaders are prefixed with the shared RTX scaffolding
        // (camera/volume bindings, vertex stage, the DDA voxel tracer) so there is one copy.
        // M33-G5: select the shadow tracer at build time. Hardware ray query when the backend
        // exposes it and VOXELCRAFT_TRACER != "dda"; else the software DDA. Re-run with
        // VOXELCRAFT_TRACER=dda to A/B against the DDA golden. The renderer appends the active
        // `sun_visibility()` (+ the TLAS binding at group 3 in the hardware case).
        let use_hw_rt =
            gpu.rt_enabled && std::env::var("VOXELCRAFT_TRACER").map_or(true, |v| v != "dda");
        // M33-G6: where the hemisphere GI is gathered. Default = a deferred compute pass + composite
        // (the path DLSS Ray Reconstruction feeds off in G8). VOXELCRAFT_GI=fragment restores the
        // in-fragment G5b gather as a switchable parity oracle / non-deferred fallback.
        let defer_gi = std::env::var("VOXELCRAFT_GI").map_or(true, |v| v != "fragment");
        let gi_raw = std::env::var("VOXELCRAFT_GI_RAW").is_ok();
        // M33-G9: GI temporal accumulation; only meaningful with deferred GI. P21: the default is
        // the platform tier's (off maxed — the supersampled default is already clean and temporal
        // accumulation risks ghosting; ON for the low-ray mac tier, where it does the denoising).
        // VOXELCRAFT_GI_ACCUM still overrides either way (folded into Quality::resolve).
        let gi_accum = defer_gi && quality.gi_accum;
        log::info!(
            "tracer: {} | GI: {}{}",
            if use_hw_rt { "hardware ray query (shadows + GI)" } else { "software DDA" },
            if defer_gi { "deferred compute pass" } else { "in-fragment (G5b oracle)" },
            if gi_accum { " + temporal accumulation" } else { "" }
        );
        let rtx_common = include_str!("../../assets/shaders/rtx_common.wgsl");
        // Atlas bindings + sample_tile (textureSample, fragment-only) — prepended to the render
        // shaders only, NOT to rtx_common, so rtx_common can also feed the GI compute shader.
        let atlas = include_str!("../../assets/shaders/atlas.wgsl");
        let rt_prefix = if use_hw_rt {
            "enable wgpu_ray_query;\n@group(3) @binding(0) var rt_acc: acceleration_structure;\n"
        } else {
            ""
        };
        let sun_vis = if use_hw_rt {
            include_str!("../../assets/shaders/sun_vis_hw.wgsl")
        } else {
            "fn trace(o: vec3<f32>, d: vec3<f32>, md: f32, ms: i32) -> Hit { return trace_dda(o, d, md, ms); }\nfn sun_visibility(p: vec3<f32>, n: vec3<f32>, md: f32) -> f32 { return sun_visibility_dda(p, n, md); }\n"
        };
        let gi_defines = format!("const DEFER_GI: bool = {defer_gi};\n");
        // P21: GI compute workgroup shape. AGX schedules 32-wide SIMD-groups, and one 8×4 = 32
        // workgroup maps the divergent DDA gather onto exactly one of them; every other backend
        // keeps the original 8×8. Single-sourced here and injected into BOTH compute shaders so
        // the `@workgroup_size` and the dispatch `div_ceil` below can never drift apart.
        let gi_wg: (u32, u32) = if gpu.backend == wgpu::Backend::Metal {
            (8, 4)
        } else {
            (8, 8)
        };
        let wg_defines = format!(
            "const GI_WG_X: u32 = {}u;\nconst GI_WG_Y: u32 = {}u;\n",
            gi_wg.0, gi_wg.1
        );
        let chunk_src = format!(
            "{rt_prefix}{gi_defines}{rtx_common}{atlas}{sun_vis}{}",
            include_str!("../../assets/shaders/chunk.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chunk-shader"),
            source: wgpu::ShaderSource::Wgsl(chunk_src.into()),
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // + COMPUTE so the same camera bind group feeds the M33-G6 GI compute pass.
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Camera-only layout (highlight) and camera+volume layout (chunks/water).
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("camera-pipeline-layout"),
            bind_group_layouts: &[Some(&camera_bgl)],
            immediate_size: 0,
        });
        let volume_bgl = VoxelVolume::bind_group_layout(device);

        // Procedural block texture atlas (group 2), shared by the chunk + water pipelines.
        let atlas_img = crate::texture::build_atlas();
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("block-atlas"),
            size: wgpu::Extent3d {
                width: crate::texture::ATLAS_W,
                height: crate::texture::ATLAS_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(crate::texture::ATLAS_W * 4),
                rows_per_image: Some(crate::texture::ATLAS_H),
            },
            wgpu::Extent3d {
                width: crate::texture::ATLAS_W,
                height: crate::texture::ATLAS_H,
                depth_or_array_layers: 1,
            },
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas-sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let atlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atlas-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas-bg"),
            layout: &atlas_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        // M34-VM: first-person view-model pipeline. group(0) = its proj+brightness uniform, group(1) =
        // the (reused) block atlas. Reuses the unified Vertex so the geometry builders feed it directly.
        let viewmodel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewmodel-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/viewmodel.wgsl").into(),
            ),
        });
        let viewmodel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewmodel-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let viewmodel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewmodel-uniform"),
            size: std::mem::size_of::<crate::viewmodel::ViewModelUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let viewmodel_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewmodel-bg"),
            layout: &viewmodel_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewmodel_buffer.as_entire_binding(),
            }],
        });
        let viewmodel_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewmodel-pipeline-layout"),
            bind_group_layouts: &[Some(&viewmodel_bgl), Some(&atlas_bgl)],
            immediate_size: 0,
        });
        let viewmodel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewmodel-pipeline"),
            layout: Some(&viewmodel_layout),
            vertex: wgpu::VertexState {
                module: &viewmodel_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &viewmodel_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let viewmodel_depth_view =
            make_viewmodel_depth(device, gpu.config.width, gpu.config.height);

        // Optional group(3): the world TLAS, for hardware shadow rays (M33-G5).
        let as_bgl = if use_hw_rt {
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rt-as-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // FRAGMENT for the chunk/water/glass shadow + GI rays; COMPUTE for the G6 GI pass.
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    count: None,
                }],
            }))
        } else {
            None
        };
        let mut chunk_bgls: Vec<Option<&wgpu::BindGroupLayout>> =
            vec![Some(&camera_bgl), Some(&volume_bgl), Some(&atlas_bgl)];
        if let Some(b) = &as_bgl {
            chunk_bgls.push(Some(b));
        }
        let chunk_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chunk-pipeline-layout"),
            bind_group_layouts: &chunk_bgls,
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk-pipeline"),
            layout: Some(&chunk_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // M1: cull disabled to guarantee visibility regardless of winding.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // MRT: HDR scene color (0) + G-buffer normal/emission (1) + motion vectors (2).
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GNORMAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GMOTION_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GPOS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GALBEDO_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GDEPTH_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Translucent water pipeline: same vertex format, alpha blend, depth test but no write.
        let water_src = format!(
            "{rt_prefix}{rtx_common}{atlas}{sun_vis}{}",
            include_str!("../../assets/shaders/water.wgsl")
        );
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-shader"),
            source: wgpu::ShaderSource::Wgsl(water_src.into()),
        });
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water-pipeline"),
            layout: Some(&chunk_layout),
            vertex: wgpu::VertexState {
                module: &water_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &water_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Glass pipeline: a translucent clone of the water pipeline (depth-tested, NO depth write so
        // water + farther glass behind a pane still show through) with no UV animation (glass.wgsl).
        let glass_src = format!(
            "{rt_prefix}{rtx_common}{atlas}{sun_vis}{}",
            include_str!("../../assets/shaders/glass.wgsl")
        );
        let glass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glass-shader"),
            source: wgpu::ShaderSource::Wgsl(glass_src.into()),
        });
        let glass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glass-pipeline"),
            layout: Some(&chunk_layout),
            vertex: wgpu::VertexState {
                module: &glass_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &glass_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Highlight wireframe pipeline (line list, depth-tested, reuses the camera bind group).
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/line.wgsl").into()),
        });
        let highlight_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                buffers: &[LineVertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // 2D HUD pipeline (no depth, alpha-blended).
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/ui.wgsl").into()),
        });
        // Bitmap-font atlas (R8 coverage) + nearest sampler for the textured UI pipeline.
        let font_img = crate::font::bake_r8();
        let font_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui-font-atlas"),
            size: wgpu::Extent3d {
                width: crate::font::ATLAS_W,
                height: crate::font::ATLAS_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &font_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &font_img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(crate::font::ATLAS_W),
                rows_per_image: Some(crate::font::ATLAS_H),
            },
            wgpu::Extent3d {
                width: crate::font::ATLAS_W,
                height: crate::font::ATLAS_H,
                depth_or_array_layers: 1,
            },
        );
        let font_view = font_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let ui_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui-sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let ui_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The block atlas + sampler, so the UI can draw textured block-item icons (mode 3).
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ui_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui-bg"),
            layout: &ui_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&font_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&ui_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });
        let ui_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui-pipeline-layout"),
            bind_group_layouts: &[Some(&ui_bgl)],
            immediate_size: 0,
        });
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui-pipeline"),
            layout: Some(&ui_layout),
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: Some("vs_main"),
                buffers: &[UiVertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // ACES tonemap pass: a fullscreen triangle that resolves the HDR scene-color buffer to the
        // LDR output. The world renders into an Rgba16Float target; this samples and tone-maps it.
        let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/tonemap.wgsl").into(),
            ),
        });
        let tonemap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tonemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tonemap-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tonemap_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap-pipeline-layout"),
            bind_group_layouts: &[Some(&tonemap_bgl)],
            immediate_size: 0,
        });
        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap-pipeline"),
            layout: Some(&tonemap_layout),
            vertex: wgpu::VertexState {
                module: &tonemap_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &tonemap_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // M33-G8-FG (Phase 3): point-upscale the render-res depth/motion guides to output res for the
        // DLSS Frame Generation tag. Two input textures (no sampler — `textureLoad`), two MRT outputs
        // (R32Float depth, Rg16Float motion).
        let guide_upscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("guide-upscale-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/gbuf_upscale.wgsl").into(),
            ),
        });
        let guide_in = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let guide_upscale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("guide-upscale-bgl"),
            entries: &[guide_in(0), guide_in(1)],
        });
        let guide_upscale_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("guide-upscale-pipeline-layout"),
            bind_group_layouts: &[Some(&guide_upscale_bgl)],
            immediate_size: 0,
        });
        let guide_upscale_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("guide-upscale-pipeline"),
                layout: Some(&guide_upscale_layout),
                vertex: wgpu::VertexState {
                    module: &guide_upscale_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &guide_upscale_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: crate::gfx::graph::GDEPTH_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: crate::gfx::graph::GMOTION_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        // M33-G9 supersampling: Catmull-Rom downscale (super-res LDR → swapchain), reusing the tonemap
        // BGL/sampler (texture + filtering sampler); and a bidirectional guide resize (super-res
        // depth/motion → swap-res FG guides), reusing the guide-upscale BGL (2 textures, MRT out).
        let ss_downscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ss-downscale-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/ss_downscale.wgsl").into(),
            ),
        });
        let fullscreen_primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        };
        let no_multisample = wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };
        let ss_downscale_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ss-downscale-pipeline"),
            layout: Some(&tonemap_layout),
            vertex: wgpu::VertexState {
                module: &ss_downscale_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: fullscreen_primitive,
            depth_stencil: None,
            multisample: no_multisample,
            fragment: Some(wgpu::FragmentState {
                module: &ss_downscale_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let guide_downscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("guide-downscale-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/gbuf_downscale.wgsl").into(),
            ),
        });
        let guide_downscale_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("guide-downscale-pipeline"),
                layout: Some(&guide_upscale_layout),
                vertex: wgpu::VertexState {
                    module: &guide_downscale_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: fullscreen_primitive,
                depth_stencil: None,
                multisample: no_multisample,
                fragment: Some(wgpu::FragmentState {
                    module: &guide_downscale_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: crate::gfx::graph::GDEPTH_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: crate::gfx::graph::GMOTION_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        // M33-G6 deferred GI: a compute pass traces the hemisphere GI from the G-buffer into a noisy
        // irradiance texture; a fullscreen composite additively blends it onto the HDR scene. Built
        // only when GI is deferred (the default). The compute shader reuses rtx_common + the active
        // tracer so the gather matches the in-fragment oracle bit-for-bit; the world TLAS, when
        // present, binds at group 3 exactly as in the render path (group 2 is the GI io instead of
        // the atlas — the compute path needs no texture atlas).
        let (gi_pipeline, composite_pipeline, gi_io_bgl, composite_bgl) = if defer_gi {
            let compute_rt_prefix = if use_hw_rt {
                "enable wgpu_ray_query;\n@group(3) @binding(0) var rt_acc: acceleration_structure;\n"
            } else {
                ""
            };
            let gi_src = format!(
                "{compute_rt_prefix}{wg_defines}{rtx_common}{sun_vis}{}",
                include_str!("../../assets/shaders/gi_compute.wgsl")
            );
            let gi_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gi-compute-shader"),
                source: wgpu::ShaderSource::Wgsl(gi_src.into()),
            });
            // group 2: gpos (in), gnormal (in), irradiance (out). Reads are textureLoad-only, so the
            // inputs are declared non-filterable (gpos is Rgba32Float, which isn't filterable anyway).
            let in_tex = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            let gi_io_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-io-bgl"),
                entries: &[
                    in_tex(0),
                    in_tex(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: IRRADIANCE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
            let mut gi_bgls: Vec<Option<&wgpu::BindGroupLayout>> =
                vec![Some(&camera_bgl), Some(&volume_bgl), Some(&gi_io_bgl)];
            if let Some(b) = &as_bgl {
                gi_bgls.push(Some(b)); // rt_acc at group 3
            }
            let gi_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gi-compute-layout"),
                bind_group_layouts: &gi_bgls,
                immediate_size: 0,
            });
            let gi_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gi-compute-pipeline"),
                layout: Some(&gi_layout),
                module: &gi_shader,
                entry_point: Some("gi_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let composite_src = format!(
                "const GI_RAW: bool = {gi_raw};\n{}",
                include_str!("../../assets/shaders/gi_composite.wgsl")
            );
            let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gi-composite-shader"),
                source: wgpu::ShaderSource::Wgsl(composite_src.into()),
            });
            // group 1: albedo, pos, normal, irradiance (all textureLoad). camera is group 0.
            let fin_tex = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-composite-bgl"),
                entries: &[fin_tex(0), fin_tex(1), fin_tex(2), fin_tex(3)],
            });
            let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gi-composite-layout"),
                bind_group_layouts: &[Some(&camera_bgl), Some(&composite_bgl)],
                immediate_size: 0,
            });
            // Additive (One,One): the indirect contribution adds onto the opaque HDR color. Alpha is
            // masked off (write_mask COLOR) so the HDR alpha stays 1.0 for the following water/glass
            // ALPHA_BLENDING. GI_RAW instead REPLACES, dumping the raw irradiance for debugging.
            let (blend, write_mask) = if gi_raw {
                (Some(wgpu::BlendState::REPLACE), wgpu::ColorWrites::ALL)
            } else {
                (
                    Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    wgpu::ColorWrites::COLOR,
                )
            };
            let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gi-composite-pipeline"),
                layout: Some(&composite_layout),
                vertex: wgpu::VertexState {
                    module: &composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend,
                        write_mask,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

            (
                Some(gi_pipeline),
                Some(composite_pipeline),
                Some(gi_io_bgl),
                Some(composite_bgl),
            )
        } else {
            (None, None, None, None)
        };

        // M33-G9: GI temporal-accumulation compute pipeline (opt-in). Group 0: gi_cur, g_depth,
        // g_motion, gi_hist (all textureLoad) + gi_accum (storage write).
        let (gi_temporal_bgl, gi_temporal_pipeline) = if gi_accum {
            let in_tex = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-temporal-bgl"),
                entries: &[
                    in_tex(0),
                    in_tex(1),
                    in_tex(2),
                    in_tex(3),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: IRRADIANCE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gi-temporal-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "{wg_defines}{}",
                        include_str!("../../assets/shaders/gi_temporal.wgsl")
                    )
                    .into(),
                ),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gi-temporal-layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gi-temporal-pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("gi_temporal_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            (Some(bgl), Some(pipeline))
        } else {
            (None, None)
        };

        Self {
            pipeline,
            water_pipeline,
            glass_pipeline,
            highlight_pipeline,
            ui_pipeline,
            tonemap_pipeline,
            tonemap_bgl,
            tonemap_sampler,
            guide_upscale_pipeline,
            guide_upscale_bgl,
            ss_downscale_pipeline,
            guide_downscale_pipeline,
            ui_bind_group,
            atlas_bind_group,
            camera_buffer,
            camera_bind_group,
            volume_bgl,
            as_bgl,
            use_hw_rt,
            defer_gi,
            gi_pipeline,
            composite_pipeline,
            gi_io_bgl,
            composite_bgl,
            gi_accum,
            gi_temporal_bgl,
            gi_temporal_pipeline,
            viewmodel_pipeline,
            viewmodel_buffer,
            viewmodel_bind_group,
            viewmodel_depth_view,
            gi_wg,
            sky_color: Cell::new(wgpu::Color {
                r: 0.46,
                g: 0.64,
                b: 0.92,
                a: 1.0,
            }),
            surface_errors: Cell::new(0),
        }
    }

    pub fn set_sky(&self, color: wgpu::Color) {
        self.sky_color.set(color);
    }

    /// Recreate output-resolution-bound targets on window resize. Currently just the view-model's
    /// private depth (the scene targets are owned by the app / DLSS). Call from the resize handler.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.viewmodel_depth_view = make_viewmodel_depth(device, width, height);
    }

    /// Upload one frame's view-model geometry into a `GpuPart` (no BLAS — it never enters the world AS).
    pub fn upload_viewmodel(&self, device: &wgpu::Device, geom: &Geometry) -> Option<GpuPart> {
        upload_geometry(device, geom, false)
    }

    /// M34-VM: draw the first-person view-model into `color_view` over its own freshly-cleared depth.
    /// `clear_color` clears the colour first (for the FG UI-recomposition layer); otherwise it draws
    /// over the already-resolved frame. Runs after the tonemap, before the 2D HUD.
    fn record_viewmodel(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        vm: &ViewModelDraw,
        clear_color: bool,
    ) {
        gpu.queue
            .write_buffer(&self.viewmodel_buffer, 0, bytemuck::bytes_of(&vm.uniform));
        let load = if clear_color {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewmodel-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: vm.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.viewmodel_pipeline);
        pass.set_bind_group(0, &self.viewmodel_bind_group, &[]);
        pass.set_bind_group(1, &self.atlas_bind_group, &[]);
        pass.set_vertex_buffer(0, vm.mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(vm.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..vm.mesh.index_count, 0, 0..1);
    }

    pub fn volume_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.volume_bgl
    }

    /// Whether hardware ray query is active (backend supports it and VOXELCRAFT_TRACER != dda). When
    /// true the render path must rebuild + bind the world TLAS each frame (group 3).
    pub fn use_hw_rt(&self) -> bool {
        self.use_hw_rt
    }

    /// Whether GI temporal accumulation is active (P21: tier-driven; mac tier default). Capture
    /// uses this to warm up screenshots so the accumulator converges before the saved frame.
    pub fn gi_accum(&self) -> bool {
        self.gi_accum
    }

    /// Build the group(3) bind group binding the world TLAS for hardware shadow rays. Only valid
    /// when `use_hw_rt`; rebuilt each frame from the freshly-built TLAS.
    pub fn make_as_bind_group(&self, device: &wgpu::Device, tlas: &wgpu::Tlas) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt-as-bg"),
            layout: self.as_bgl.as_ref().expect("as_bgl exists when use_hw_rt"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(tlas),
            }],
        })
    }

    pub fn upload_mesh(&self, gpu: &Gpu, mesh: &MeshData) -> GpuMesh {
        // Gate BLAS (+ its BLAS_INPUT buffer usage) on the active tracer, not just the backend: in
        // VOXELCRAFT_TRACER=dda mode there is no TLAS/trace, so per-chunk BLASes are pure waste. (review M1)
        let rt = self.use_hw_rt;
        let opaque = upload_geometry(&gpu.device, &mesh.opaque, rt);
        let blas = if rt {
            opaque.as_ref().map(|p| crate::gfx::rt::build_chunk_blas(gpu, p))
        } else {
            None
        };
        GpuMesh {
            opaque,
            water: upload_geometry(&gpu.device, &mesh.water, rt),
            translucent: upload_geometry(&gpu.device, &mesh.translucent, rt),
            blas,
        }
    }

    pub fn update_camera(&self, gpu: &Gpu, uniform: &CameraUniform) {
        gpu.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[*uniform]));
    }

    /// Build the per-resolution HDR scene-color target (+ its tonemap bind group) for one render
    /// path. Called at startup and on resize by the present loop, and per-shot by the screenshot.
    pub fn make_targets(&self, device: &wgpu::Device, width: u32, height: u32) -> RenderTargets {
        let (width, height) = (width.max(1), height.max(1));
        let target = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        // Keep the textures for the four DLSS RR guides (rr.render needs &Texture); gpos is GI-only
        // so a view suffices (the texture stays alive via the view).
        let hdr_tex = target("hdr-scene-color", HDR_FORMAT);
        let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let gnormal_tex = target("gbuf-normal", GNORMAL_FORMAT);
        let gnormal_view = gnormal_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let gmotion_tex = target("gbuf-motion", GMOTION_FORMAT);
        let gmotion_view = gmotion_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let gpos_view = target("gbuf-pos", GPOS_FORMAT).create_view(&wgpu::TextureViewDescriptor::default());
        let galbedo_tex = target("gbuf-albedo", GALBEDO_FORMAT);
        let galbedo_view = galbedo_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let gdepth_tex = target("gbuf-depth-linear", GDEPTH_FORMAT);
        let gdepth_view = gdepth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // GI compute (group 2: gpos + gnormal in, irradiance out) + composite (group 1: albedo, pos,
        // normal, irradiance) bind groups — built only in deferred-GI mode (the layouts are `Some`).
        // The irradiance texture (compute-written storage, composite-sampled) is created here and
        // kept alive by the bind groups, so it isn't allocated at all in the in-fragment path.
        let (gi_compute_bg, composite_bg, gi_temporal_bg, gi_accum_tex, gi_history_tex) =
            match (&self.gi_io_bgl, &self.composite_bgl) {
                (Some(gi_io_bgl), Some(composite_bgl)) => {
                    let irr_desc = |label, usage| wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: IRRADIANCE_FORMAT,
                        usage,
                        view_formats: &[],
                    };
                    let irradiance_view = device
                        .create_texture(&irr_desc(
                            "gi-irradiance",
                            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                        ))
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let gi_compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gi-compute-bg"),
                        layout: gi_io_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&gpos_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&gnormal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&irradiance_view),
                            },
                        ],
                    });
                    // M33-G9 opt-in: temporal-accumulation buffers. `gi_accum` is the temporal output
                    // (composite reads it); `gi_history` is the previous frame's accum (copied each
                    // frame after the pass). When off, the composite reads the raw irradiance.
                    let (gi_temporal_bg, gi_accum_tex, gi_history_tex, accum_view) = if self.gi_accum {
                        let tbgl = self
                            .gi_temporal_bgl
                            .as_ref()
                            .expect("gi_temporal_bgl exists when gi_accum");
                        let accum = device.create_texture(&irr_desc(
                            "gi-accum",
                            wgpu::TextureUsages::STORAGE_BINDING
                                | wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_SRC,
                        ));
                        let accum_view = accum.create_view(&wgpu::TextureViewDescriptor::default());
                        let history = device.create_texture(&irr_desc(
                            "gi-history",
                            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        ));
                        let history_view =
                            history.create_view(&wgpu::TextureViewDescriptor::default());
                        let view_entry = |binding, v| wgpu::BindGroupEntry {
                            binding,
                            resource: wgpu::BindingResource::TextureView(v),
                        };
                        let tbg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("gi-temporal-bg"),
                            layout: tbgl,
                            entries: &[
                                view_entry(0, &irradiance_view),
                                view_entry(1, &gdepth_view),
                                view_entry(2, &gmotion_view),
                                view_entry(3, &history_view),
                                view_entry(4, &accum_view),
                            ],
                        });
                        (Some(tbg), Some(accum), Some(history), Some(accum_view))
                    } else {
                        (None, None, None, None)
                    };
                    // Composite samples the accumulated irradiance (temporal on) or the raw (off).
                    let composite_irr = accum_view.as_ref().unwrap_or(&irradiance_view);
                    let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gi-composite-bg"),
                        layout: composite_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&galbedo_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&gpos_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&gnormal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(composite_irr),
                            },
                        ],
                    });
                    (
                        Some(gi_compute_bg),
                        Some(composite_bg),
                        gi_temporal_bg,
                        gi_accum_tex,
                        gi_history_tex,
                    )
                }
                _ => (None, None, None, None, None),
            };

        let tonemap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap-bg"),
            layout: &self.tonemap_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.tonemap_sampler),
                },
            ],
        });
        RenderTargets {
            hdr_view,
            gnormal_view,
            gmotion_view,
            gpos_view,
            galbedo_view,
            hdr_tex,
            gnormal_tex,
            gmotion_tex,
            galbedo_tex,
            gdepth_view,
            gdepth_tex,
            width,
            height,
            gi_compute_bg,
            composite_bg,
            gi_temporal_bg,
            gi_accum_tex,
            gi_history_tex,
            tonemap_bg,
        }
    }

    /// Build a tonemap bind group over an arbitrary HDR source view (e.g. the DLSS output texture),
    /// reusing the tonemap layout + sampler. Lets the DLSS path resolve its upscaled output through
    /// the same ACES pass. (M33-G8) — only used by the DLSS path, which compiles only under
    /// all(feature, Windows) (src/gfx.rs dual gate), so the allow mirrors that full gate.
    #[cfg_attr(not(all(feature = "dlss", target_os = "windows")), allow(dead_code))]
    pub fn make_tonemap_bg(&self, device: &wgpu::Device, hdr: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap-bg-dlss"),
            layout: &self.tonemap_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.tonemap_sampler),
                },
            ],
        })
    }


    /// Record the chunk pass into an existing encoder against arbitrary color/depth targets,
    /// drawing every mesh in `meshes`. Shared by the present and screenshot paths. `timer` is the
    /// headless benchmark's per-pass timestamp hook (P21) — `None` (every non-bench path) records
    /// `timestamp_writes: None` exactly as before.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &RenderTargets,
        depth_view: &wgpu::TextureView,
        meshes: &[&GpuMesh],
        volume_bg: &wgpu::BindGroup,
        as_bg: Option<&wgpu::BindGroup>,
        timer: Option<&crate::bench::GpuTimer>,
    ) {
        // Opaque pass: clears the HDR color + G-buffer (normal, motion) + depth, then draws.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("opaque-pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.hdr_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.sky_color.get()),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.gnormal_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.gmotion_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.gpos_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.galbedo_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.gdepth_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            // Far linear depth for sky / no-geometry pixels.
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 10000.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.map(|t| t.render_writes(crate::bench::Pass::Opaque)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, volume_bg, &[]);
            pass.set_bind_group(2, &self.atlas_bind_group, &[]);
            if let Some(b) = as_bg {
                pass.set_bind_group(3, b, &[]);
            }
            for mesh in meshes {
                if let Some(part) = &mesh.opaque {
                    pass.set_vertex_buffer(0, part.vertex_buffer.slice(..));
                    pass.set_index_buffer(part.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..part.index_count, 0, 0..1);
                }
            }
        }

        // M33-G6 deferred GI: trace the hemisphere GI from the just-written G-buffer into the noisy
        // irradiance texture (compute), then additively composite `albedo * irradiance * skylight *
        // (1-fog) * (1-flash)` onto the opaque HDR color — BEFORE glass/water blend over it. wgpu
        // inserts the storage-write → sampled-read barrier between the passes. Skipped entirely in
        // the in-fragment (VOXELCRAFT_GI=fragment) oracle path.
        if self.defer_gi {
            if let (Some(gi_pipeline), Some(gi_bg), Some(composite_pipeline), Some(composite_bg)) = (
                self.gi_pipeline.as_ref(),
                targets.gi_compute_bg.as_ref(),
                self.composite_pipeline.as_ref(),
                targets.composite_bg.as_ref(),
            ) {
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gi-compute-pass"),
                        timestamp_writes: timer.map(|t| t.compute_writes(crate::bench::Pass::GiCompute)),
                    });
                    cpass.set_pipeline(gi_pipeline);
                    cpass.set_bind_group(0, &self.camera_bind_group, &[]);
                    cpass.set_bind_group(1, volume_bg, &[]);
                    cpass.set_bind_group(2, gi_bg, &[]);
                    if let Some(b) = as_bg {
                        cpass.set_bind_group(3, b, &[]);
                    }
                    cpass.dispatch_workgroups(
                        targets.width.div_ceil(self.gi_wg.0),
                        targets.height.div_ceil(self.gi_wg.1),
                        1,
                    );
                }
                // M33-G9 opt-in: temporally accumulate the noisy irradiance into `gi_accum` (which the
                // composite then samples), reprojecting the previous frame via motion vectors; then
                // copy accum → history for next frame. wgpu inserts the storage-write/read barriers.
                if let (Some(tp), Some(tbg), Some(accum), Some(history)) = (
                    self.gi_temporal_pipeline.as_ref(),
                    targets.gi_temporal_bg.as_ref(),
                    targets.gi_accum_tex.as_ref(),
                    targets.gi_history_tex.as_ref(),
                ) {
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("gi-temporal-pass"),
                            timestamp_writes: timer
                                .map(|t| t.compute_writes(crate::bench::Pass::GiTemporal)),
                        });
                        cpass.set_pipeline(tp);
                        cpass.set_bind_group(0, tbg, &[]);
                        cpass.dispatch_workgroups(
                            targets.width.div_ceil(self.gi_wg.0),
                            targets.height.div_ceil(self.gi_wg.1),
                            1,
                        );
                    }
                    encoder.copy_texture_to_texture(
                        accum.as_image_copy(),
                        history.as_image_copy(),
                        wgpu::Extent3d {
                            width: targets.width,
                            height: targets.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("gi-composite-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &targets.hdr_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: timer
                            .map(|t| t.render_writes(crate::bench::Pass::GiComposite)),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(composite_pipeline);
                    pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    pass.set_bind_group(1, composite_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }

        // Glass pass (loads color + depth; alpha-blends over the opaque world, depth-tested but no
        // depth write, so water + farther glass behind a pane still show through).
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("glass-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.hdr_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.map(|t| t.render_writes(crate::bench::Pass::Glass)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.glass_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, volume_bg, &[]);
            pass.set_bind_group(2, &self.atlas_bind_group, &[]);
            if let Some(b) = as_bg {
                pass.set_bind_group(3, b, &[]);
            }
            for mesh in meshes {
                if let Some(part) = &mesh.translucent {
                    pass.set_vertex_buffer(0, part.vertex_buffer.slice(..));
                    pass.set_index_buffer(part.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..part.index_count, 0, 0..1);
                }
            }
        }

        // Translucent water pass (loads color + depth; blends, no depth write).
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.hdr_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.map(|t| t.render_writes(crate::bench::Pass::Water)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.water_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, volume_bg, &[]);
            pass.set_bind_group(2, &self.atlas_bind_group, &[]);
            if let Some(b) = as_bg {
                pass.set_bind_group(3, b, &[]);
            }
            for mesh in meshes {
                if let Some(part) = &mesh.water {
                    pass.set_vertex_buffer(0, part.vertex_buffer.slice(..));
                    pass.set_index_buffer(part.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..part.index_count, 0, 0..1);
                }
            }
        }
    }

    /// The block-highlight wireframe + crack overlay, drawn into the HDR scene buffer (over the
    /// opaque world, depth-tested). Factored out of the frame so the native and DLSS paths share it.
    fn record_highlight(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        highlight: Option<(IVec3, f32)>,
    ) {
        let Some((block, progress)) = highlight else {
            return;
        };
        let mut lines = overlay::highlight_lines(block);
        if progress > 0.0 {
            lines.extend(overlay::crack_lines(block, progress));
        }
        let count = lines.len() as u32;
        if count == 0 {
            return;
        }
        // The buffer is reference-counted into the command buffer, so it survives this scope until
        // the encoder is submitted (matches the previous in-function lifetime).
        let buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("highlight-vbuf"),
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("highlight-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.highlight_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, buf.slice(..));
        pass.draw(0..count, 0..1);
    }

    /// Resolve an HDR source (`tonemap_bg`) to `final_view` via the fullscreen ACES pass, then draw
    /// the LDR HUD on top. `tonemap_bg` is `targets.tonemap_bg` (native) or the DLSS output's.
    fn record_resolve(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        tonemap_bg: &wgpu::BindGroup,
        final_view: &wgpu::TextureView,
        ui_verts: &[UiVertex],
        // M33-G8-FG (Phase 4): also write the HUD-less scene + the UI layer for DLSS-G recomposition.
        hudless_view: Option<&wgpu::TextureView>,
        ui_view: Option<&wgpu::TextureView>,
        // M34-VM: the first-person held item, drawn over the tonemapped scene + into the FG UI layer.
        viewmodel: Option<&ViewModelDraw>,
    ) {
        // Fullscreen ACES resolve (HDR → LDR). The triangle covers every pixel; the clear is just a
        // defined initial state. Reused for both the swapchain and (Phase 4) the HUD-less buffer.
        let tonemap = |encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, tonemap_bg, &[]);
            pass.draw(0..3, 0..1);
        };
        tonemap(encoder, final_view);
        // Phase 4: the HUD-less scene colour DLSS-G interpolates from (same tonemapped scene). The
        // view-model is NOT drawn here — it has no motion vectors, so it rides the UI layer instead.
        if let Some(hv) = hudless_view {
            tonemap(encoder, hv);
        }

        // View-model over the real (presented) frame, then HUD on top.
        if let Some(vm) = viewmodel {
            self.record_viewmodel(gpu, encoder, final_view, vm, false);
        }
        if !ui_verts.is_empty() {
            self.draw_hud(gpu, encoder, final_view, ui_verts, false);
        }
        // The FG UI-recomposition layer: clear → view-model → HUD (so the hand recomposites crisply on
        // generated frames like the HUD, instead of ghosting against world motion in the scene layer).
        if let Some(uv) = ui_view {
            match viewmodel {
                Some(vm) => {
                    self.record_viewmodel(gpu, encoder, uv, vm, true); // clears the layer, then the hand
                    self.draw_hud(gpu, encoder, uv, ui_verts, false);
                }
                None => self.draw_hud(gpu, encoder, uv, ui_verts, true),
            }
        }
    }

    /// Fullscreen blit: clear `view` and draw the fullscreen triangle with `pipeline` + `bg`. Used by
    /// the supersample resolve (tonemap into the super-res LDR, then Catmull-Rom downscale). (M33-G9)
    fn blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bg: &wgpu::BindGroup,
        view: &wgpu::TextureView,
        label: &str,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Draw the HUD (`ui_verts`) into `view`. `clear` clears to transparent first (the FG
    /// UI-recomposition layer) — otherwise loads + alpha-blends over the presented frame.
    fn draw_hud(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ui_verts: &[UiVertex],
        clear: bool,
    ) {
        let buf = (!ui_verts.is_empty()).then(|| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ui-vbuf"),
                contents: bytemuck::cast_slice(ui_verts),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Some(buf) = &buf {
            pass.set_pipeline(&self.ui_pipeline);
            pass.set_bind_group(0, &self.ui_bind_group, &[]);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..ui_verts.len() as u32, 0..1);
        }
    }

    /// M35: draw a full-screen menu (its own UI-vertex list) into `view`, clearing first. Used by the
    /// app's menu scenes (no world render) and the headless `VOXELCRAFT_MENU` screenshot.
    pub fn render_menu(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, ui_verts: &[UiVertex]) {
        self.draw_hud(gpu, encoder, view, ui_verts, true);
    }

    /// M35-N3 interactive menu present: acquire the swapchain, clear it, draw the menu UI, present.
    /// World-independent (no DLSS / scene targets) — used for every menu/paused scene. Mirrors
    /// `render_frame`'s acquire/backoff handling so a broken surface backs off instead of spinning.
    pub fn present_menu(&self, gpu: &Gpu, ui_verts: &[UiVertex]) -> bool {
        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.surface_errors.set(0);
                t
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.surface_backoff();
                return false;
            }
            _ => {
                self.surface_backoff();
                log::warn!("swapchain acquire failed (lost/outdated) — reconfiguring");
                gpu.surface.configure(&gpu.device, &gpu.config);
                return false;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("menu") });
        self.render_menu(gpu, &mut enc, &view, ui_verts);
        gpu.queue.submit(Some(enc.finish()));
        frame.present();
        true
    }

    /// M33-G9 supersample resolve: tonemap the super-res HDR (`hdr_tonemap_bg`) into the super-res LDR
    /// (`ss_ldr_view`), Catmull-Rom downscale that to the swapchain `final_view` (+ the FG hudless
    /// layer), then draw the crisp HUD at swapchain resolution.
    #[allow(clippy::too_many_arguments)]
    fn record_resolve_supersampled(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        hdr_tonemap_bg: &wgpu::BindGroup,
        ss_ldr_view: &wgpu::TextureView,
        ss_downscale_bg: &wgpu::BindGroup,
        final_view: &wgpu::TextureView,
        ui_verts: &[UiVertex],
        hudless_view: Option<&wgpu::TextureView>,
        ui_view: Option<&wgpu::TextureView>,
        viewmodel: Option<&ViewModelDraw>,
    ) {
        // 1. ACES tonemap the super-res HDR → super-res LDR (swapchain colour format).
        self.blit(encoder, &self.tonemap_pipeline, hdr_tonemap_bg, ss_ldr_view, "ss-tonemap");
        // 2. Catmull-Rom downscale → swapchain (and the FG HUD-less layer, also at swap res).
        self.blit(encoder, &self.ss_downscale_pipeline, ss_downscale_bg, final_view, "ss-downscale");
        if let Some(hv) = hudless_view {
            self.blit(encoder, &self.ss_downscale_pipeline, ss_downscale_bg, hv, "ss-downscale-hudless");
        }
        // 3. View-model (swapchain res, never supersampled) → crisp HUD on top.
        if let Some(vm) = viewmodel {
            self.record_viewmodel(gpu, encoder, final_view, vm, false);
        }
        if !ui_verts.is_empty() {
            self.draw_hud(gpu, encoder, final_view, ui_verts, false);
        }
        // 4. The FG UI-recomposition layer: clear → view-model → HUD.
        if let Some(uv) = ui_view {
            match viewmodel {
                Some(vm) => {
                    self.record_viewmodel(gpu, encoder, uv, vm, true);
                    self.draw_hud(gpu, encoder, uv, ui_verts, false);
                }
                None => self.draw_hud(gpu, encoder, uv, ui_verts, true),
            }
        }
    }

    /// Record a complete frame into `encoder` (native, no DLSS): scene → HDR, ACES tonemap to
    /// `final_view`, then the HUD. Shared by the present + screenshot paths via `render_into`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_full(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        targets: &RenderTargets,
        final_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        meshes: &[&GpuMesh],
        volume_bg: &wgpu::BindGroup,
        as_bg: Option<&wgpu::BindGroup>,
        highlight: Option<(IVec3, f32)>,
        ui_verts: &[UiVertex],
        hudless_view: Option<&wgpu::TextureView>,
        ui_view: Option<&wgpu::TextureView>,
        viewmodel: Option<&ViewModelDraw>,
        timer: Option<&crate::bench::GpuTimer>,
    ) {
        self.record(encoder, targets, depth_view, meshes, volume_bg, as_bg, timer);
        self.record_highlight(gpu, encoder, &targets.hdr_view, depth_view, highlight);
        self.record_resolve(
            gpu, encoder, &targets.tonemap_bg, final_view, ui_verts, hudless_view, ui_view, viewmodel,
        );
    }

    /// Render one full frame to `final_view`, managing its own encoders + submits. With `dlss`, the
    /// scene renders at `targets`' (render) resolution, Ray Reconstruction denoises + upscales it,
    /// and the upscaled output is tonemapped to `final_view`; otherwise the native single-encoder
    /// path. Shared by the present + screenshot paths.
    #[allow(clippy::too_many_arguments)]
    /// M33-G8-FG (Phase 3): point-upscale the render-resolution depth + motion guides in `src` into
    /// the output-resolution `dst_depth` / `dst_motion` views, for the DLSS Frame Generation tag
    /// (DLSS-G tags the output-res guides of the presented frame; RR only emits them at render res).
    pub fn resize_guides(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &RenderTargets,
        dst_depth: &wgpu::TextureView,
        dst_motion: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("guide-resize-bg"),
            layout: &self.guide_upscale_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src.gdepth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src.gmotion_view),
                },
            ],
        });
        let attach = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("guide-resize-pass"),
            color_attachments: &[attach(dst_depth), attach(dst_motion)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn render_into(
        &self,
        gpu: &Gpu,
        targets: &RenderTargets,
        final_view: &wgpu::TextureView,
        scene_depth: &wgpu::TextureView,
        meshes: &[&GpuMesh],
        volume_bg: &wgpu::BindGroup,
        as_bg: Option<&wgpu::BindGroup>,
        highlight: Option<(IVec3, f32)>,
        ui_verts: &[UiVertex],
        // M33-G8-FG (Phase 4): when present (FG active), the scene is also tonemapped HUD-less into
        // `hudless_view` and the HUD alone into `ui_view`, for DLSS-G UI recomposition.
        hudless_view: Option<&wgpu::TextureView>,
        ui_view: Option<&wgpu::TextureView>,
        viewmodel: Option<&ViewModelDraw>,
        dlss: Option<&mut crate::dlss::DlssRender>,
    ) {
        match dlss {
            Some(dlss) => {
                // Scene (+ GI + highlight) at render resolution into the G-buffer + HDR; submit so
                // RR's resource transitions observe the finished scene (as the dlss example does).
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dlss-scene-encoder"),
                });
                self.record(&mut enc, targets, dlss.depth_view(), meshes, volume_bg, as_bg, None);
                self.record_highlight(gpu, &mut enc, &targets.hdr_view, dlss.depth_view(), highlight);
                gpu.queue.submit(Some(enc.finish()));

                // Debug knob: VOXELCRAFT_DLSS_BYPASS tonemaps the render-res scene directly (sampler-
                // upscaled), skipping RR — isolates "is the scene lit?" from "is RR producing output?".
                let bypass = std::env::var("VOXELCRAFT_DLSS_BYPASS").is_ok();
                let resolve_bg = if bypass {
                    &targets.tonemap_bg
                } else {
                    // Ray Reconstruction: denoise + upscale into the DLSS output texture (own submits).
                    dlss.evaluate(targets, &gpu.queue);
                    dlss.output_tonemap_bg()
                };

                // Resolve → swapchain/readback + HUD at output resolution.
                let mut enc2 = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dlss-resolve-encoder"),
                });
                // Resize the render-res depth/motion guides to SWAPCHAIN res for DLSS-G: point-upscale
                // without supersampling, box-downscale with it (M33-G9). Harmless when FG is off.
                let supersampled = dlss.is_supersampled();
                let guide_pipeline = if supersampled {
                    &self.guide_downscale_pipeline
                } else {
                    &self.guide_upscale_pipeline
                };
                self.resize_guides(
                    &gpu.device,
                    &mut enc2,
                    targets,
                    dlss.output_gdepth_view(),
                    dlss.output_gmotion_view(),
                    guide_pipeline,
                );
                if supersampled {
                    // Tonemap the super-res RR output, Catmull-Rom downscale to the swapchain, crisp HUD.
                    self.record_resolve_supersampled(
                        gpu,
                        &mut enc2,
                        resolve_bg,
                        dlss.ss_ldr_view(),
                        dlss.ss_downscale_bg(),
                        final_view,
                        ui_verts,
                        hudless_view,
                        ui_view,
                        viewmodel,
                    );
                } else {
                    self.record_resolve(
                        gpu, &mut enc2, resolve_bg, final_view, ui_verts, hudless_view, ui_view,
                        viewmodel,
                    );
                }
                gpu.queue.submit(Some(enc2.finish()));
            }
            None => {
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame-encoder"),
                });
                self.record_full(
                    gpu, &mut enc, targets, final_view, scene_depth, meshes, volume_bg, as_bg,
                    highlight, ui_verts, hudless_view, ui_view, viewmodel, None,
                );
                gpu.queue.submit(Some(enc.finish()));
            }
        }
    }

    /// Present a full frame to the swapchain.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &self,
        gpu: &Gpu,
        targets: &RenderTargets,
        meshes: &[&GpuMesh],
        volume_bg: &wgpu::BindGroup,
        as_bg: Option<&wgpu::BindGroup>,
        highlight: Option<(IVec3, f32)>,
        ui_verts: &[UiVertex],
        // M34-VM: this frame's view-model geometry + its projection/brightness uniform (the renderer
        // supplies the private depth target). `None` when nothing is held / VM disabled.
        viewmodel: Option<(&GpuPart, crate::viewmodel::ViewModelUniform)>,
        mut dlss: Option<&mut crate::dlss::DlssRender>,
        fg: Option<&mut crate::frame_gen::FrameGen>,
    ) -> bool {
        let vmd = viewmodel.map(|(mesh, uniform)| ViewModelDraw {
            mesh,
            depth: &self.viewmodel_depth_view,
            uniform,
        });
        // M33-G8-FG: present through DLSS Frame Generation when active — the scene renders into the
        // acquired swapchain texture, FG tags the output-res depth/motion and inserts a generated
        // frame. With RR (Phase 3) the scene is rendered at render res and the guides are point-
        // upscaled into DlssRender's output-res textures (by render_into); standalone the native
        // output-res targets are tagged directly.
        if let Some(fg) = fg {
            let aspect = gpu.aspect();
            // Clone the guide texture handles (cheap, Arc-backed) so `dlss` can still move into the
            // render closure as `&mut` (the upscale that fills these runs inside render_into).
            let (depth, motion) = match dlss.as_deref() {
                Some(d) => (d.output_gdepth().clone(), d.output_gmotion().clone()),
                None => (targets.gdepth_tex.clone(), targets.gmotion_tex.clone()),
            };
            // Phase 4: clone the HUD-less + UI recomposition textures (Arc-backed) so `dlss` can still
            // move into the render closure; render_into fills them via `record_resolve`.
            let hudless = fg.hudless_tex().clone();
            let ui = fg.ui_tex().clone();
            let hudless_view = hudless.create_view(&wgpu::TextureViewDescriptor::default());
            let ui_view = ui.create_view(&wgpu::TextureViewDescriptor::default());
            // Reborrow dlss into the closure so it stays usable for the plain-present fallback if FG
            // fails to present this frame — keeps the window alive instead of freezing (review #7).
            let presented = fg.present_frame(gpu, aspect, &depth, &motion, &hudless, &ui, |view| {
                self.render_into(
                    gpu, targets, view, &gpu.depth_view, meshes, volume_bg, as_bg, highlight,
                    ui_verts, Some(&hudless_view), Some(&ui_view), vmd.as_ref(), dlss.as_deref_mut(),
                );
            });
            if presented {
                return true;
            }
            // FG did not present (begin_frame/set_constants/acquire/tag failure) — fall through to the
            // ordinary swapchain present below so the frame is still shown (RR still applies).
        }

        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.surface_errors.set(0);
                t
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.surface_backoff();
                return false;
            }
            _ => {
                // Lost/Outdated (or a future variant): reconfigure once and let the next frame
                // retry — with backoff, so a surface that stays broken is not respun at full rate.
                self.surface_backoff();
                log::warn!("swapchain acquire failed (lost/outdated) — reconfiguring");
                gpu.surface.configure(&gpu.device, &gpu.config);
                return false;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.render_into(
            gpu,
            targets,
            &view,
            &gpu.depth_view,
            meshes,
            volume_bg,
            as_bg,
            highlight,
            ui_verts,
            None,
            None,
            vmd.as_ref(),
            dlss,
        );
        frame.present();
        true
    }

    /// Escalating sleep after repeated swapchain-acquire failures (2026-06-03 wedge hardening):
    /// the first few failures retry immediately (a minimized/occluded window is normal), then each
    /// further one sleeps a little longer (capped at 250ms) so `about_to_wait`'s unconditional
    /// `request_redraw` cannot spin acquire/reconfigure against a wedged or occluded surface.
    fn surface_backoff(&self) {
        let n = self.surface_errors.get().saturating_add(1);
        self.surface_errors.set(n);
        if n > 3 {
            let ms = (25 * u64::from(n - 3)).min(250);
            std::thread::sleep(std::time::Duration::from_millis(ms));
        }
    }
}
