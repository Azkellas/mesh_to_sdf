use anyhow::Result;

use crate::camera::CameraData;
use crate::pbr::shadow_map;
use crate::texture::Texture;
use crate::utility::shader_builder::ShaderBuilder;

use crate::pbr::mesh::MeshVertex;

pub struct ModelRenderPass {
    pub render_pipeline: wgpu::RenderPipeline,
    pub model_bind_group_layout: wgpu::BindGroupLayout,
    pub textures_bind_group_layout: wgpu::BindGroupLayout,
    pub shadow_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub shadow_bind_group: Option<wgpu::BindGroup>,
    backface_culling: bool,
}

impl ModelRenderPass {
    fn create_pipeline(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        model_bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraData,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        textures_bind_group_layout: &wgpu::BindGroupLayout,
        backface_culling: bool,
    ) -> Result<wgpu::RenderPipeline> {
        let draw_shader = ShaderBuilder::create_module(device, "draw_model.wgsl")?;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[
                    model_bind_group_layout,
                    &camera.bind_group_layout,
                    shadow_bind_group_layout,
                    textures_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: backface_culling.then_some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        };

        ShaderBuilder::create_render_pipeline(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("ModelRenderPass::render_pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: "main_vs",
                    buffers: &[MeshVertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &draw_shader,
                    entry_point: "main_fs",
                    targets: &[Some(view_format.into())],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive,
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        )
    }

    pub fn update_pipeline(
        &mut self,
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        backface_culling: bool,
    ) -> Result<()> {
        self.render_pipeline = Self::create_pipeline(
            device,
            view_format,
            &self.model_bind_group_layout,
            camera,
            self.shadow_bind_group_layout.as_ref().unwrap(),
            &self.textures_bind_group_layout,
            backface_culling,
        )?;
        self.backface_culling = backface_culling;
        Ok(())
    }

    pub fn is_culling_backfaces(&self) -> bool {
        self.backface_culling
    }
    fn create_shadow_bind_group(
        device: &wgpu::Device,
        shadow_map: &shadow_map::ShadowMap,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_map.light.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_map.texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_map.texture.sampler),
                },
            ],
            label: Some("Model shadow bind group"),
        })
    }

    fn create_shadow_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
            label: Some("Model shadow bind group layout"),
        })
    }

    pub fn create_textures_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            label: Some("Model textures bind group layout"),
        })
    }

    pub fn create_model_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("ModelRenderPass::model bind group layout"),
        })
    }

    pub fn new(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        shadow_map: &shadow_map::ShadowMap,
        backface_culling: bool,
    ) -> Result<Self> {
        let render_shadow_bind_group_layout = Self::create_shadow_bind_group_layout(device);

        let render_shadow_bind_group =
            Self::create_shadow_bind_group(device, shadow_map, &render_shadow_bind_group_layout);

        let render_textures_bind_group_layout = Self::create_textures_bind_group_layout(device);

        let model_bind_group_layout = Self::create_model_bind_group_layout(device);

        let render_pipeline = ModelRenderPass::create_pipeline(
            device,
            view_format,
            &model_bind_group_layout,
            camera,
            &render_shadow_bind_group_layout,
            &render_textures_bind_group_layout,
            backface_culling,
        )?;

        Ok(ModelRenderPass {
            render_pipeline,
            model_bind_group_layout,
            textures_bind_group_layout: render_textures_bind_group_layout,
            shadow_bind_group_layout: Some(render_shadow_bind_group_layout),
            shadow_bind_group: Some(render_shadow_bind_group),
            backface_culling,
        })
    }

    pub fn run(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_map: &Texture,
        camera: &CameraData,
        model: &crate::pbr::model::Model,
    ) {
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("ModelRenderPass::run::render_pass_descriptor"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_map.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        // render pass
        let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &model.transform_bind_group, &[]);
        rpass.set_bind_group(1, &camera.bind_group, &[]);
        rpass.set_bind_group(2, self.shadow_bind_group.as_ref().unwrap(), &[]);
        rpass.set_bind_group(3, &model.textures_bind_group, &[]);

        rpass.set_vertex_buffer(0, model.mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(model.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..model.mesh.index_count, 0, 0..1);
    }
}
