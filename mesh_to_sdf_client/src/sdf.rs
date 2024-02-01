use itertools::Itertools;
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
    pub ordered_indices: Vec<u32>,
    pub ordered_buffer: wgpu::Buffer,

    pub bind_group: wgpu::BindGroup,
}

impl Sdf {
    pub fn new(
        device: &wgpu::Device,
        vertices: &[[f32; 3]],
        indices: &[u32],
        start_cell: &[f32; 3],
        end_cell: &[f32; 3],
        cell_count: &[u32; 3],
    ) -> Result<Self> {
        let ucell_count = [
            cell_count[0] as usize,
            cell_count[1] as usize,
            cell_count[2] as usize,
        ];
        let grid = mesh_to_sdf::Grid::from_bounding_box(start_cell, end_cell, &ucell_count);
        let data = mesh_to_sdf::generate_grid_sdf(
            vertices,
            mesh_to_sdf::Topology::TriangleList(Some(indices)),
            &grid,
        );

        // sort cells by their (absolute) distance to surface.
        // used in the voxel render pass to only draw valid cells.
        let ordered_indices = (0..data.len())
            .sorted_by(|i, j| data[*i].abs().partial_cmp(&data[*j].abs()).unwrap())
            .map(|i| i as u32)
            .collect_vec();

        let cell_size = grid.get_cell_size();
        let uniforms = SdfUniforms {
            start: [start_cell[0], start_cell[1], start_cell[2], 0.0],
            end: [end_cell[0], end_cell[1], end_cell[2], 0.0],
            cell_size: [cell_size[0], cell_size[1], cell_size[2], 0.0],
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

        let ordered_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sdf Ordered Indices Buffer"),
            contents: bytemuck::cast_slice(&ordered_indices),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ordered_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            uniforms,
            uniforms_buffer,
            data,
            data_buffer,
            ordered_indices,
            ordered_buffer,
            bind_group,
        })
    }

    pub fn get_cell_count(&self) -> u32 {
        self.uniforms.cell_count[0] * self.uniforms.cell_count[1] * self.uniforms.cell_count[2]
    }
}
