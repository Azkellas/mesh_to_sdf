use anyhow::Result;

use crate::camera::CameraData;
use crate::pbr::model::Model;
use crate::pbr::model_instance::ModelInstance;
use crate::texture::Texture;
use crate::utility::shader_builder::ShaderBuilder;

use crate::pbr::mesh::MeshVertex;

pub struct CubemapGenerationPass {
    pub render_pipeline: wgpu::RenderPipeline,
    pub model_bind_group_layout: wgpu::BindGroupLayout,
    pub textures_bind_group_layout: wgpu::BindGroupLayout,
}

impl CubemapGenerationPass {
    fn create_pipeline(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        model_bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraData,
        textures_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline> {
        let draw_shader = ShaderBuilder::create_module(device, "draw_cubemap.wgsl")?;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[
                    model_bind_group_layout,
                    &camera.bind_group_layout,
                    textures_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
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
                    depth_compare: wgpu::CompareFunction::Greater, // this way we don't clear the depth buffer to 1.0.
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
    ) -> Result<()> {
        self.render_pipeline = Self::create_pipeline(
            device,
            view_format,
            &self.model_bind_group_layout,
            camera,
            &self.textures_bind_group_layout,
        )?;
        Ok(())
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
    ) -> Result<Self> {
        let render_textures_bind_group_layout = Self::create_textures_bind_group_layout(device);

        let model_bind_group_layout = Self::create_model_bind_group_layout(device);

        let render_pipeline = Self::create_pipeline(
            device,
            view_format,
            &model_bind_group_layout,
            camera,
            &render_textures_bind_group_layout,
        )?;

        Ok(Self {
            render_pipeline,
            model_bind_group_layout,
            textures_bind_group_layout: render_textures_bind_group_layout,
        })
    }

    pub fn run(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &CameraData,
        model: &Model,
        model_instance: &ModelInstance,
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
                view: depth_view,
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
        rpass.set_bind_group(0, &model_instance.transform_bind_group, &[]);
        rpass.set_bind_group(1, &camera.bind_group, &[]);
        rpass.set_bind_group(2, &model.textures_bind_group, &[]);

        rpass.set_vertex_buffer(0, model.mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(model.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..model.mesh.index_count, 0, 0..1);
    }
}
