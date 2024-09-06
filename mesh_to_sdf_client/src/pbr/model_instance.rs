use wgpu::util::DeviceExt;

use crate::passes::model_render_pass::ModelRenderPass;

pub struct ModelInstance {
    pub model_id: usize,
    pub transform: glam::Mat4,
    pub transform_buffer: wgpu::Buffer,
    pub transform_bind_group: wgpu::BindGroup,
    pub transform_bind_group_layout: wgpu::BindGroupLayout,
}

impl ModelInstance {
    pub fn new(device: &wgpu::Device, model_id: usize, transform: glam::Mat4) -> Self {
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::cast_slice(&[transform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let transform_bind_group_layout = ModelRenderPass::create_model_bind_group_layout(device);

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
            label: Some("Model::Transform bind group"),
        });

        Self {
            model_id,
            transform,
            transform_buffer,
            transform_bind_group,
            transform_bind_group_layout,
        }
    }
}
