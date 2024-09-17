use itertools::Itertools;

use crate::{
    gltf::{GltfModel, Material},
    texture::Texture,
};

use super::mesh::{Mesh, MeshVertex};
use crate::passes::model_render_pass::ModelRenderPass;

pub struct Model {
    pub name: Option<String>,

    pub mesh: Mesh,

    // TODO Material struct.
    pub albedo: Texture,

    pub textures_bind_group: wgpu::BindGroup,
}

impl Model {
    pub fn from_gtlf(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &GltfModel,
        material: Option<&Material>,
    ) -> Self {
        let indices = model
            .indices()
            .cloned()
            .unwrap_or_else(|| (0..(model.vertices().len() as u32)).collect());
        let vertices = model
            .vertices()
            .iter()
            .map(|v| MeshVertex {
                position: [v.position.x, v.position.y, v.position.z],
                normal: [v.normal.x, v.normal.y, v.normal.z],
                tex_coords: [v.tex_coords.x, v.tex_coords.y],
            })
            .collect_vec();

        println!("model: {:?}", model.mesh_name());

        let mesh = Mesh::new(device, vertices, indices);

        // If no albedo is present, render as grey.
        let grey_albedo: image::ImageBuffer<image::Rgba<u8>, std::vec::Vec<u8>> =
            image::ImageBuffer::from_pixel(2, 2, image::Rgba([128, 128, 128, 255]));

        let albedo = Texture::from_image(
            device,
            queue,
            material.map_or(&grey_albedo, |material| {
                material
                    .pbr
                    .base_color_texture
                    .as_ref()
                    .map_or(&grey_albedo, |x| x.as_ref())
            }),
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

        Self {
            name: model.mesh_name().map(std::borrow::ToOwned::to_owned),
            mesh,
            albedo,
            textures_bind_group,
        }
    }
}
