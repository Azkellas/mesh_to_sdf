use anyhow::Result;

use crate::utility::shader_builder::ShaderBuilder;

#[derive(Debug)]
pub struct MipGenerationPass {
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
}

impl MipGenerationPass {
    pub fn create_pipeline(device: &wgpu::Device) -> Result<wgpu::RenderPipeline> {
        let shader = ShaderBuilder::create_module(device, "utility/mipmap_generation.wgsl")?;

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(pipeline)
    }

    pub fn update_pipeline(&mut self, device: &wgpu::Device) -> Result<()> {
        self.pipeline = Self::create_pipeline(device)?;
        Ok(())
    }

    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let pipeline = Self::create_pipeline(device)?;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self { pipeline, sampler })
    }

    pub fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        from_view: &wgpu::TextureView,
        to_view: &wgpu::TextureView,
    ) {
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(from_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("generate_mipmaps::bind_group"),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("generate_mipmaps::rpass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: to_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
