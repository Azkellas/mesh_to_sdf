use wgpu::util::DeviceExt;

use anyhow::Result;

use crate::passes::sdf_render_pass::SdfRenderPass;

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfUniforms {
    start: [f32; 4],
    end: [f32; 4],
    cell_size: [f32; 4],
    cell_count: [u32; 4],
}

#[derive(Debug)]
pub struct Sdf {
    pub uniforms: SdfUniforms,
    pub uniforms_buffer: wgpu::Buffer,
    pub data: Vec<f32>,
    pub data_buffer: wgpu::Buffer,

    pub cell_radius: [f32; 3],
    pub cell_count: u32,

    pub bind_group: wgpu::BindGroup,
}

impl Sdf {
    pub fn new(
        device: &wgpu::Device,
        vertices: &[[f32; 3]],
        indices: &[u32],
        start_cell: &[f32; 3],
        cell_radius: &[f32; 3],
        cell_count: &[u32; 3],
    ) -> Result<Self> {
        let data = mesh_to_sdf::generate_grid_sdf(
            vertices,
            mesh_to_sdf::Topology::TriangleList(Some(indices)),
            start_cell,
            cell_radius,
            cell_count,
        );

        let uniforms = SdfUniforms {
            start: [start_cell[0], start_cell[1], start_cell[2], 0.0],
            end: [
                start_cell[0] + cell_radius[0] * cell_count[0] as f32,
                start_cell[1] + cell_radius[1] * cell_count[1] as f32,
                start_cell[2] + cell_radius[2] * cell_count[2] as f32,
                0.0,
            ],
            cell_size: [cell_radius[0], cell_radius[1], cell_radius[2], 0.0],
            cell_count: [cell_count[0], cell_count[1], cell_count[2], 0],
        };

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sdf Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sdf Data Buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = SdfRenderPass::get_bind_group_layout(device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sdf Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            uniforms,
            uniforms_buffer,
            data,
            data_buffer,
            bind_group,

            cell_radius: *cell_radius,
            cell_count: cell_count[0] * cell_count[1] * cell_count[2],
        })
    }
}
