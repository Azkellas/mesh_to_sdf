use itertools::{Itertools, MinMaxResult};
use wgpu::util::DeviceExt;

use anyhow::Result;

use crate::sdf_render_pass::SdfRenderPass;

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
        vertices: &[easy_gltf::model::Vertex],
        indices: &[u32],
        cell_radius: [f32; 3],
    ) -> Result<Self> {
        let MinMaxResult::MinMax(xmin, xmax) = vertices.iter().map(|v| v.position.x).minmax()
        else {
            anyhow::bail!("Bounding box is ill-defined")
        };
        let MinMaxResult::MinMax(ymin, ymax) = vertices.iter().map(|v| v.position.y).minmax()
        else {
            anyhow::bail!("Bounding box is ill-defined")
        };
        let MinMaxResult::MinMax(zmin, zmax) = vertices.iter().map(|v| v.position.z).minmax()
        else {
            anyhow::bail!("Bounding box is ill-defined")
        };

        // generate points in x,y,z bounds with a cell radius of cell_radius
        let mut query_points = vec![];
        let xsize = ((xmax - xmin) / cell_radius[0]).ceil();
        let ysize = ((ymax - ymin) / cell_radius[1]).ceil();
        let zsize = ((zmax - zmin) / cell_radius[2]).ceil();

        for xi in 0..xsize as u32 {
            for yi in 0..ysize as u32 {
                for zi in 0..zsize as u32 {
                    let x = xmin + xi as f32 * cell_radius[0];
                    let y = ymin + yi as f32 * cell_radius[1];
                    let z = zmin + zi as f32 * cell_radius[2];
                    query_points.push([x, y, z]);
                }
            }
        }

        let vertices = vertices
            .iter()
            .map(|v| [v.position[0], v.position[1], v.position[2]])
            .collect_vec();

        let data = mesh_to_sdf::generate_sdf(
            &vertices,
            mesh_to_sdf::Topology::TriangleList(Some(indices)),
            &query_points,
        );

        let uniforms = SdfUniforms {
            start: [xmin, ymin, zmin, 0.0],
            end: [
                xmin + cell_radius[0] * xsize,
                ymin + cell_radius[1] * ysize,
                zmin + cell_radius[2] * zsize,
                0.0,
            ],
            cell_size: [cell_radius[0], cell_radius[1], cell_radius[2], 0.0],
            cell_count: [xsize as u32, ysize as u32, zsize as u32, 0],
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

            cell_radius,
            cell_count: query_points.len() as u32,
        })
    }
}
