use itertools::Itertools;
use wgpu::util::DeviceExt;

use crate::texture::Texture;

use crate::passes::model_render_pass::ModelRenderPass;

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl ModelVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 8 * 4,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct Model {
    pub name: Option<String>,
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,

    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,

    pub index_count: u32,

    // TODO Material struct.
    pub albedo: Texture,

    pub transform: glam::Mat4,
    pub transform_buffer: wgpu::Buffer,
    pub transform_bind_group: wgpu::BindGroup,
    pub transform_bind_group_layout: wgpu::BindGroupLayout,

    pub textures_bind_group: wgpu::BindGroup,
}

impl Model {
    pub fn from_gtlf(device: &wgpu::Device, queue: &wgpu::Queue, model: &easy_gltf::Model) -> Self {
        let indices = model.indices().unwrap().iter().copied().collect_vec();
        let vertices = model
            .vertices()
            .iter()
            .map(|v| ModelVertex {
                position: [v.position.x, v.position.y, v.position.z],
                normal: [v.normal.x, v.normal.y, v.normal.z],
                tex_coords: [v.tex_coords.x, v.tex_coords.y],
            })
            .collect_vec();

        println!("model: {:?}", model.mesh_name());

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let index_count = indices.len() as u32;

        // If no albedo is present, render as grey.
        let grey_albedo: image::ImageBuffer<image::Rgba<u8>, std::vec::Vec<u8>> =
            image::ImageBuffer::from_pixel(2, 2, image::Rgba([128, 128, 128, 255]));

        let albedo = Texture::from_image(
            device,
            queue,
            model
                .material()
                .pbr
                .base_color_texture
                .as_ref()
                .map_or(&grey_albedo, |x| x.as_ref()),
            Some("albedo"),
        );

        let textures_bind_group_layout = ModelRenderPass::create_textures_bind_group_layout(device);

        let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &textures_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&albedo.sampler),
                },
            ],
            label: Some("Model Bind group"),
        });

        // In easy_gltf, the transform is already applied to the vertices.
        // This is not an issue with this app since we don't want to move the model (yet).
        let transform = glam::Mat4::IDENTITY;

        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::cast_slice(&[transform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let model_bind_group_layout = ModelRenderPass::create_model_bind_group_layout(device);

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
            label: Some("Model::Transform bind group"),
        });

        Self {
            name: model.mesh_name().map(|x| x.to_owned()),
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
            index_count,
            albedo,
            transform,
            transform_buffer,
            transform_bind_group,
            transform_bind_group_layout: model_bind_group_layout,
            textures_bind_group,
        }
    }
}
