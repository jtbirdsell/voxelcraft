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
    sky_color: Cell<wgpu::Color>,
}

impl ChunkRenderer {
    pub fn new(gpu: &Gpu) -> Self {
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
        log::info!(
            "tracer: {} | GI: {}",
            if use_hw_rt { "hardware ray query (shadows + GI)" } else { "software DDA" },
            if defer_gi { "deferred compute pass" } else { "in-fragment (G5b oracle)" }
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
                "{compute_rt_prefix}{rtx_common}{sun_vis}{}",
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

        Self {
            pipeline,
            water_pipeline,
            glass_pipeline,
            highlight_pipeline,
            ui_pipeline,
            tonemap_pipeline,
            tonemap_bgl,
            tonemap_sampler,
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
            sky_color: Cell::new(wgpu::Color {
                r: 0.46,
                g: 0.64,
                b: 0.92,
                a: 1.0,
            }),
        }
    }

    pub fn set_sky(&self, color: wgpu::Color) {
        self.sky_color.set(color);
    }

    pub fn volume_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.volume_bgl
    }

    /// Whether hardware ray query is active (backend supports it and VOXELCRAFT_TRACER != dda). When
    /// true the render path must rebuild + bind the world TLAS each frame (group 3).
    pub fn use_hw_rt(&self) -> bool {
        self.use_hw_rt
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
        let (gi_compute_bg, composite_bg) = match (&self.gi_io_bgl, &self.composite_bgl) {
            (Some(gi_io_bgl), Some(composite_bgl)) => {
                let irradiance_view = device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("gi-irradiance"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: IRRADIANCE_FORMAT,
                        usage: wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
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
                            resource: wgpu::BindingResource::TextureView(&irradiance_view),
                        },
                    ],
                });
                (Some(gi_compute_bg), Some(composite_bg))
            }
            _ => (None, None),
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
            tonemap_bg,
        }
    }

    /// Build a tonemap bind group over an arbitrary HDR source view (e.g. the DLSS output texture),
    /// reusing the tonemap layout + sampler. Lets the DLSS path resolve its upscaled output through
    /// the same ACES pass. (M33-G8)
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
    /// drawing every mesh in `meshes`. Shared by the present and screenshot paths.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &RenderTargets,
        depth_view: &wgpu::TextureView,
        meshes: &[&GpuMesh],
        volume_bg: &wgpu::BindGroup,
        as_bg: Option<&wgpu::BindGroup>,
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
                timestamp_writes: None,
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
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(gi_pipeline);
                    cpass.set_bind_group(0, &self.camera_bind_group, &[]);
                    cpass.set_bind_group(1, volume_bg, &[]);
                    cpass.set_bind_group(2, gi_bg, &[]);
                    if let Some(b) = as_bg {
                        cpass.set_bind_group(3, b, &[]);
                    }
                    cpass.dispatch_workgroups(
                        targets.width.div_ceil(8),
                        targets.height.div_ceil(8),
                        1,
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
                        timestamp_writes: None,
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
                timestamp_writes: None,
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
                timestamp_writes: None,
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
    ) {
        // Resolve HDR → LDR (fullscreen ACES). The triangle covers every pixel; clear is just for a
        // defined initial state on a fresh swapchain view.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_view,
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
        }

        if !ui_verts.is_empty() {
            let buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ui-vbuf"),
                contents: bytemuck::cast_slice(ui_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.ui_pipeline);
            pass.set_bind_group(0, &self.ui_bind_group, &[]);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..ui_verts.len() as u32, 0..1);
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
    ) {
        self.record(encoder, targets, depth_view, meshes, volume_bg, as_bg);
        self.record_highlight(gpu, encoder, &targets.hdr_view, depth_view, highlight);
        self.record_resolve(gpu, encoder, &targets.tonemap_bg, final_view, ui_verts);
    }

    /// Render one full frame to `final_view`, managing its own encoders + submits. With `dlss`, the
    /// scene renders at `targets`' (render) resolution, Ray Reconstruction denoises + upscales it,
    /// and the upscaled output is tonemapped to `final_view`; otherwise the native single-encoder
    /// path. Shared by the present + screenshot paths.
    #[allow(clippy::too_many_arguments)]
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
        dlss: Option<&mut crate::dlss::DlssRender>,
    ) {
        match dlss {
            Some(dlss) => {
                // Scene (+ GI + highlight) at render resolution into the G-buffer + HDR; submit so
                // RR's resource transitions observe the finished scene (as the dlss example does).
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dlss-scene-encoder"),
                });
                self.record(&mut enc, targets, dlss.depth_view(), meshes, volume_bg, as_bg);
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
                self.record_resolve(gpu, &mut enc2, resolve_bg, final_view, ui_verts);
                gpu.queue.submit(Some(enc2.finish()));
            }
            None => {
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame-encoder"),
                });
                self.record_full(
                    gpu, &mut enc, targets, final_view, scene_depth, meshes, volume_bg, as_bg,
                    highlight, ui_verts,
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
        dlss: Option<&mut crate::dlss::DlssRender>,
    ) {
        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            _ => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
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
            dlss,
        );
        frame.present();
    }
}
