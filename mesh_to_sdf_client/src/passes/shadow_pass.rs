use anyhow::Result;

use crate::pbr::model::ModelVertex;
use crate::texture::Texture;
use crate::utility::shader_builder::ShaderBuilder;

use crate::pbr::shadow_map::ShadowMap;

use crate::passes::model_render_pass::ModelRenderPass;

pub struct ShadowPass {
    pub render_pipeline: wgpu::RenderPipeline,
    pub map: ShadowMap,
}

impl ShadowPass {
    fn create_pipeline(
        device: &wgpu::Device,
        shadow_map: &ShadowMap,
    ) -> Result<wgpu::RenderPipeline> {
        let draw_shader = ShaderBuilder::create_module(device, "draw_model.wgsl")?;

        let model_bind_group_layout = ModelRenderPass::create_model_bind_group_layout(device);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow map"),
                bind_group_layouts: &[
                    &model_bind_group_layout,
                    &shadow_map.light.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        ShaderBuilder::create_render_pipeline(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("shadow pass"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: "main_vs",
                    buffers: &[ModelVertex::desc()],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState::default(),
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

    pub fn update_pipeline(&mut self, device: &wgpu::Device) -> Result<()> {
        self.render_pipeline = Self::create_pipeline(device, &self.map)?;
        Ok(())
    }

    pub fn new(device: &wgpu::Device, shadow_map: ShadowMap) -> Result<Self> {
        Ok(ShadowPass {
            render_pipeline: ShadowPass::create_pipeline(device, &shadow_map)?,
            map: shadow_map,
        })
    }

    pub fn run(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        model: &crate::pbr::model::Model,
    ) {
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("ShadowPass::run::render_pass_descriptor"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.map.texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &model.transform_bind_group, &[]);
        rpass.set_bind_group(1, &self.map.light.bind_group, &[]);
        rpass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
        rpass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..model.index_count, 0, 0..1);
    }
}
