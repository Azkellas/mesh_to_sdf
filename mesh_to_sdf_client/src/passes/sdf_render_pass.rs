use anyhow::Result;

use crate::camera::CameraData;
use crate::sdf::{Sdf, SdfUniforms};
use crate::texture::Texture;
use crate::utility::shader_builder::ShaderBuilder;

pub struct SdfRenderPass {
    pub render_pipeline: wgpu::RenderPipeline,
    pub sdf_bind_group_layout: wgpu::BindGroupLayout,
}

impl SdfRenderPass {
    fn create_pipeline(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        sdf_bind_group_layout: &wgpu::BindGroupLayout,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline> {
        let draw_shader = ShaderBuilder::create_module(device, "draw_sdf.wgsl")?;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SdfRenderPass::PipelineLayout"),
                bind_group_layouts: &[
                    sdf_bind_group_layout,
                    &camera.bind_group_layout,
                    settings_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        };

        ShaderBuilder::create_render_pipeline(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("SdfRenderPass::RenderPipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: "main_vs",
                    buffers: &[],
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

    pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sdf Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            core::mem::size_of::<SdfUniforms>() as _,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn update_pipeline(
        &mut self,
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<()> {
        self.render_pipeline = Self::create_pipeline(
            device,
            view_format,
            camera,
            &self.sdf_bind_group_layout,
            settings_bind_group_layout,
        )?;
        Ok(())
    }
    pub fn new(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        let sdf_bind_group_layout = Self::get_bind_group_layout(device);

        let render_pipeline = Self::create_pipeline(
            device,
            view_format,
            camera,
            &sdf_bind_group_layout,
            settings_bind_group_layout,
        )?;

        Ok(Self {
            render_pipeline,
            sdf_bind_group_layout,
        })
    }

    pub fn run(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_map: &Texture,
        camera: &CameraData,
        sdf: &Sdf,
        settings_bind_group: &wgpu::BindGroup,
    ) {
        // need to draw it each frame to update depth map.
        command_encoder.push_debug_group("render particles");
        {
            let render_pass_descriptor = wgpu::RenderPassDescriptor {
                label: Some("SdfRenderPass::run::render_pass_descriptor"),
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
            rpass.set_bind_group(0, &sdf.bind_group, &[]);
            rpass.set_bind_group(1, &camera.bind_group, &[]);
            rpass.set_bind_group(2, settings_bind_group, &[]);
            // vertices are computed in the shader directly to save bandwidth
            rpass.draw(0..3, 0..sdf.get_cell_count());
        }
        command_encoder.pop_debug_group();
    }
}
