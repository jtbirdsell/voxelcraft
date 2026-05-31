//! Chunk render pipeline: a camera uniform bind group + a textured/lit triangle pipeline,
//! plus helpers to upload a `MeshData` to GPU buffers and draw it.

use std::cell::Cell;

use glam::IVec3;
use wgpu::util::DeviceExt;

use crate::camera::CameraUniform;
use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::mesher::{Geometry, MeshData, Vertex};
use crate::gfx::graph::{RenderTargets, GMOTION_FORMAT, GNORMAL_FORMAT, HDR_FORMAT};
use crate::overlay::{self, LineVertex, UiVertex};
use crate::voxel_volume::VoxelVolume;

fn upload_geometry(device: &wgpu::Device, geom: &Geometry) -> Option<GpuPart> {
    if geom.is_empty() {
        return None;
    }
    Some(GpuPart {
        vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-vbuf"),
            contents: bytemuck::cast_slice(&geom.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-ibuf"),
            contents: bytemuck::cast_slice(&geom.indices),
            usage: wgpu::BufferUsages::INDEX,
        }),
        index_count: geom.indices.len() as u32,
    })
}

pub struct GpuPart {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub struct GpuMesh {
    pub opaque: Option<GpuPart>,
    pub water: Option<GpuPart>,
    pub translucent: Option<GpuPart>,
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
    sky_color: Cell<wgpu::Color>,
}

impl ChunkRenderer {
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;

        // Both the chunk and water shaders are prefixed with the shared RTX scaffolding
        // (camera/volume bindings, vertex stage, the DDA voxel tracer) so there is one copy.
        let rtx_common = include_str!("../../assets/shaders/rtx_common.wgsl");
        let chunk_src = format!("{rtx_common}{}", include_str!("../../assets/shaders/chunk.wgsl"));
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
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

        let chunk_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chunk-pipeline-layout"),
            bind_group_layouts: &[Some(&camera_bgl), Some(&volume_bgl), Some(&atlas_bgl)],
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
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Translucent water pipeline: same vertex format, alpha blend, depth test but no write.
        let water_src = format!("{rtx_common}{}", include_str!("../../assets/shaders/water.wgsl"));
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
        let glass_src = format!("{rtx_common}{}", include_str!("../../assets/shaders/glass.wgsl"));
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

    pub fn upload_mesh(&self, gpu: &Gpu, mesh: &MeshData) -> GpuMesh {
        GpuMesh {
            opaque: upload_geometry(&gpu.device, &mesh.opaque),
            water: upload_geometry(&gpu.device, &mesh.water),
            translucent: upload_geometry(&gpu.device, &mesh.translucent),
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
            device
                .create_texture(&wgpu::TextureDescriptor {
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let hdr_view = target("hdr-scene-color", HDR_FORMAT);
        let gnormal_view = target("gbuf-normal", GNORMAL_FORMAT);
        let gmotion_view = target("gbuf-motion", GMOTION_FORMAT);
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
            tonemap_bg,
        }
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
            for mesh in meshes {
                if let Some(part) = &mesh.opaque {
                    pass.set_vertex_buffer(0, part.vertex_buffer.slice(..));
                    pass.set_index_buffer(part.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..part.index_count, 0, 0..1);
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
            for mesh in meshes {
                if let Some(part) = &mesh.water {
                    pass.set_vertex_buffer(0, part.vertex_buffer.slice(..));
                    pass.set_index_buffer(part.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..part.index_count, 0, 0..1);
                }
            }
        }
    }

    /// Record a complete frame into `encoder`: the world renders into the HDR scene buffer
    /// (`targets`), an ACES tonemap resolves it to `final_view`, then the HUD is drawn on top in
    /// LDR. Shared by the on-screen present path and the offscreen screenshot path.
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
        highlight: Option<(IVec3, f32)>,
        ui_verts: &[UiVertex],
    ) {
        // Scene passes render into the HDR buffer + G-buffer: opaque (clears), then glass + water.
        self.record(encoder, targets, depth_view, meshes, volume_bg);

        // Build overlay buffers (kept alive until this function returns; wgpu retains the
        // underlying resources in the command buffer until execution).
        let highlight_buf = highlight.map(|(block, progress)| {
            let mut lines = overlay::highlight_lines(block);
            if progress > 0.0 {
                lines.extend(overlay::crack_lines(block, progress));
            }
            let count = lines.len() as u32;
            let buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("highlight-vbuf"),
                contents: bytemuck::cast_slice(&lines),
                usage: wgpu::BufferUsages::VERTEX,
            });
            (buf, count)
        });
        let ui_buf = if ui_verts.is_empty() {
            None
        } else {
            let buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ui-vbuf"),
                contents: bytemuck::cast_slice(ui_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            Some((buf, ui_verts.len() as u32))
        };

        if let Some((buf, count)) = &highlight_buf {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("highlight-pass"),
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
            pass.set_pipeline(&self.highlight_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..*count, 0..1);
        }

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
            pass.set_bind_group(0, &targets.tonemap_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if let Some((buf, count)) = &ui_buf {
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
            pass.draw(0..*count, 0..1);
        }
    }

    /// Present a full frame to the swapchain.
    pub fn render_frame(
        &self,
        gpu: &Gpu,
        targets: &RenderTargets,
        meshes: &[&GpuMesh],
        volume_bg: &wgpu::BindGroup,
        highlight: Option<(IVec3, f32)>,
        ui_verts: &[UiVertex],
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
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        self.record_full(
            gpu,
            &mut encoder,
            targets,
            &view,
            &gpu.depth_view,
            meshes,
            volume_bg,
            highlight,
            ui_verts,
        );
        gpu.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
