use itertools::Itertools;
use mesh_to_sdf::SignMethod;
use wgpu::util::DeviceExt;

use anyhow::Result;

use crate::passes::sdf_render_pass::SdfRenderPass;

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfUniforms {
    pub start: [f32; 4],
    pub end: [f32; 4],
    pub cell_size: [f32; 4],
    pub cell_count: [u32; 4],
}

#[derive(Debug)]
pub struct Sdf {
    pub uniforms: SdfUniforms,
    pub uniforms_buffer: wgpu::Buffer,
    pub data: Vec<f32>,
    pub data_buffer: wgpu::Buffer,
    pub ordered_indices: Vec<u32>,
    pub ordered_buffer: wgpu::Buffer,

    pub grid: mesh_to_sdf::Grid<glam::Vec3>,
    pub iso_limits: (f32, f32),

    pub bind_group: wgpu::BindGroup,
}

impl Sdf {
    pub fn new(
        device: &wgpu::Device,
        vertices: &[glam::Vec3],
        indices: &[u32],
        start_cell: &glam::Vec3,
        end_cell: &glam::Vec3,
        cell_count: &[u32; 3],
        sign_method: SignMethod,
    ) -> Result<Self> {
        let ucell_count = [
            cell_count[0] as usize,
            cell_count[1] as usize,
            cell_count[2] as usize,
        ];
        let grid = mesh_to_sdf::Grid::from_bounding_box(start_cell, end_cell, ucell_count);
        let data = mesh_to_sdf::generate_grid_sdf(
            vertices,
            mesh_to_sdf::Topology::TriangleList(Some(indices)),
            &grid,
            sign_method,
        );

        // sort cells by their distance to surface.
        // used in the voxel render pass to only draw valid cells.
        let ordered_indices = (0..data.len())
            .sorted_by(|i, j| data[*i].total_cmp(&data[*j]))
            .map(|i| i as u32)
            .collect_vec();

        let cell_size = grid.get_cell_size();
        let first_cell = grid.get_first_cell();
        let last_cell = grid.get_last_cell();
        let uniforms = SdfUniforms {
            start: [first_cell[0], first_cell[1], first_cell[2], 0.0],
            end: [last_cell[0], last_cell[1], last_cell[2], 0.0],
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

        let iso_limits = data.iter().copied().minmax().into_option().unwrap();

        Ok(Self {
            uniforms,
            uniforms_buffer,
            data,
            data_buffer,
            ordered_indices,
            ordered_buffer,
            bind_group,
            grid,
            iso_limits,
        })
    }

    pub fn get_cell_count(&self) -> u32 {
        self.uniforms.cell_count[0] * self.uniforms.cell_count[1] * self.uniforms.cell_count[2]
    }
}
