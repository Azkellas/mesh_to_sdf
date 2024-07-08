use anyhow::Result;

use crate::camera::CameraData;
use crate::pbr::shadow_map;
use crate::sdf::{Sdf, SdfUniforms};
use crate::sdf_program::SettingsData;
use crate::utility::shader_builder::ShaderBuilder;

pub struct RaymarchRenderPass {
    pub render_pipeline: wgpu::RenderPipeline,

    pub raymarch_bind_group_layout: wgpu::BindGroupLayout,
    pub shadow_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub shadow_bind_group: Option<wgpu::BindGroup>,
}

impl RaymarchRenderPass {
    fn create_pipeline(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        raymarch_bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraData,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline> {
        let draw_shader = ShaderBuilder::create_module(device, "draw_raymarching.wgsl")?;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RaymarchRenderPass::PipelineLayout"),
                bind_group_layouts: &[
                    raymarch_bind_group_layout,
                    &camera.bind_group_layout,
                    settings_bind_group_layout,
                    shadow_bind_group_layout,
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
                label: Some("RaymarchRenderPass::RenderPipeline"),
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        )
    }

    pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RaymarchRenderPass::Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SdfUniforms>() as _
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
            &self.raymarch_bind_group_layout,
            camera,
            settings_bind_group_layout,
            self.shadow_bind_group_layout.as_ref().unwrap(),
        )?;
        Ok(())
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

    pub fn new(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_map: &shadow_map::ShadowMap,
    ) -> Result<Self> {
        let render_shadow_bind_group_layout = Self::create_shadow_bind_group_layout(device);

        let render_shadow_bind_group =
            Self::create_shadow_bind_group(device, shadow_map, &render_shadow_bind_group_layout);

        let raymarch_bind_group_layout = Self::get_bind_group_layout(device);

        let render_pipeline = RaymarchRenderPass::create_pipeline(
            device,
            view_format,
            &raymarch_bind_group_layout,
            camera,
            settings_bind_group_layout,
            &render_shadow_bind_group_layout,
        )?;

        Ok(RaymarchRenderPass {
            render_pipeline,
            raymarch_bind_group_layout,
            shadow_bind_group_layout: Some(render_shadow_bind_group_layout),
            shadow_bind_group: Some(render_shadow_bind_group),
        })
    }

    pub fn run(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &CameraData,
        sdf: &Sdf,
        settings: &SettingsData,
    ) {
        // need to draw it each frame to update depth map.
        command_encoder.push_debug_group("render particles");
        {
            let render_pass_descriptor = wgpu::RenderPassDescriptor {
                label: Some("RaymarchRenderPass::run::render_pass_descriptor"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };

            // render pass
            let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &sdf.bind_group, &[]);
            rpass.set_bind_group(1, &camera.bind_group, &[]);
            rpass.set_bind_group(2, &settings.bind_group, &[]);
            rpass.set_bind_group(3, self.shadow_bind_group.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }
        command_encoder.pop_debug_group();
    }
}
