use anyhow::Result;

use crate::{
    camera::CameraData, passes::cubemap_generation_pass::CubemapGenerationPass, pbr::model::Model,
    texture::Texture,
};

/// A cubemap is a 6-sided texture that is used to render a model from outside.
/// It is used to project its material onto the SDF/voxel visualization.
pub struct Cubemap {
    /// Material albedo cubemap. The texture has 6 layers, one for each face of the cube.
    albedo: Texture,
    /// Model depth cubemap. The texture has 6 layers, one for each face of the cube.
    depth: Texture,
    /// Uniform buffer for the camera data. Contains an array of 6 view-projection matrices.
    uniforms: wgpu::Buffer,
    /// Bind group for the cubemap.
    /// 0: albedo texture
    /// 1: albedo sampler
    /// 2: depth texture
    /// 3: depth sampler
    /// 4: camera uniform buffer
    bind_group: wgpu::BindGroup,
    /// Bind group layout for the cubemap.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Cubemap {
    /// Create a new cubemap of a given size.
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let size3d = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        };

        let albedo = Texture::create_render_target(device, size3d, Some("mesh_cubemap"), false);
        let depth = Texture::create_depth_texture(device, size3d, "mesh_cubemap_depth");
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cubemap Uniform Buffer"),
            size: 6 * 16 * std::mem::size_of::<f32>() as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniforms.size()),
                    },
                    count: None,
                },
            ],
            label: Some("cubemap_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&albedo.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&depth.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniforms.as_entire_binding(),
                },
            ],
            label: Some("cubemap_bind_group"),
        });

        Self {
            albedo,
            depth,
            uniforms,
            bind_group,
            bind_group_layout,
        }
    }

    /// Get the bind group layout for the cubemap.
    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the bind group for the cubemap.
    pub fn get_bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Get the albedo texture of the cubemap.
    pub fn get_albedo(&self) -> &Texture {
        &self.albedo
    }

    /// Get the depth texture of the cubemap.
    pub fn get_depth(&self) -> &Texture {
        &self.depth
    }

    /// Get the uniform buffer for the cubemap.
    pub fn get_uniforms(&self) -> &wgpu::Buffer {
        &self.uniforms
    }

    /// Generate the cubemap by rendering the models from the outside.
    pub fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounding_box: [f32; 6],
        models: &[Model],
        cubemap_pass: &CubemapGenerationPass,
    ) -> Result<()> {
        let size = [2048, 2048];

        let mut camera = CameraData::new(device);

        let bbx = 0.5 * (bounding_box[3] - bounding_box[0]);
        let bby = 0.5 * (bounding_box[4] - bounding_box[1]);
        let bbz = 0.5 * (bounding_box[5] - bounding_box[2]);
        let bb_center = glam::Vec3::new(
            (bounding_box[0] + bounding_box[3]) * 0.5,
            (bounding_box[1] + bounding_box[4]) * 0.5,
            (bounding_box[2] + bounding_box[5]) * 0.5,
        );

        let mut uniforms = [glam::Mat4::IDENTITY; 6];

        // Render the cubemap from 6 directions.
        // For each face we start by computing the eye position, then the view and projection matrices.
        (0..6).for_each(|i| {
            let (eye, proj, view) = match i {
                0 => {
                    // axis +X
                    let eye = bb_center - bbx * glam::Vec3::X;
                    (
                        eye,
                        glam::Mat4::orthographic_rh(-bbz, bbz, -bby, bby, 0.0, 2.0 * bbx),
                        glam::Mat4::look_at_rh(eye, bb_center, glam::Vec3::Y),
                    )
                }
                1 => {
                    // axis -x
                    let eye = bb_center + bbx * glam::Vec3::X;
                    (
                        eye,
                        glam::Mat4::orthographic_rh(-bbz, bbz, -bby, bby, 0.0, 2.0 * bbx),
                        glam::Mat4::look_at_rh(eye, bb_center, glam::Vec3::Y),
                    )
                }
                2 => {
                    // axis +z
                    let eye = bb_center + bbz * glam::Vec3::Z;
                    (
                        eye,
                        glam::Mat4::orthographic_rh(-bbx, bbx, -bby, bby, 0.0, 2.0 * bbz),
                        glam::Mat4::look_at_rh(eye, bb_center, glam::Vec3::Y),
                    )
                }
                3 => {
                    // axis -z
                    let eye = bb_center - bbz * glam::Vec3::Z;
                    (
                        eye,
                        glam::Mat4::orthographic_rh(-bbx, bbx, -bby, bby, 0.0, 2.0 * bbz),
                        glam::Mat4::look_at_rh(eye, bb_center, glam::Vec3::Y),
                    )
                }
                4 => {
                    // axis +y
                    let eye = bb_center - bby * glam::Vec3::Y;
                    (
                        eye,
                        glam::Mat4::orthographic_rh(-bbx, bbx, -bbz, bbz, 0.0, 2.0 * bby),
                        glam::Mat4::look_at_rh(eye, bb_center, glam::Vec3::Z),
                    )
                }
                5 => {
                    // axis -y
                    let eye = bb_center + bby * glam::Vec3::Y;
                    (
                        eye,
                        glam::Mat4::orthographic_rh(-bbx, bbx, -bbz, bbz, 0.0, 2.0 * bby),
                        glam::Mat4::look_at_rh(eye, bb_center, glam::Vec3::Z),
                    )
                }
                _ => unreachable!(),
            };

            // Update the camera data.
            camera.uniform.view_proj = proj * view;
            camera.uniform.view = view;
            camera.uniform.proj = proj;
            camera.uniform.view_inv = view.inverse();
            camera.uniform.proj_inv = proj.inverse();
            camera.uniform.eye = eye.extend(1.0);
            camera.uniform.resolution = size;
            queue.write_buffer(&camera.buffer, 0, bytemuck::cast_slice(&[camera.uniform]));

            uniforms[i as usize] = camera.uniform.view_proj;

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("generate_cubemap"),
            });

            // Create a view for the current layer of the cubemap albedo.
            let layer_view = self
                .albedo
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                });

            // Create a view for the current layer of the cubemap depth.
            let depth_view = self
                .depth
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                });

            // Clear the textures.
            let whole_texture_range = wgpu::ImageSubresourceRange {
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i,
                array_layer_count: Some(1),
            };
            encoder.clear_texture(&self.albedo.texture, &whole_texture_range);
            encoder.clear_texture(&self.depth.texture, &whole_texture_range);

            // Run the cubemap pass for each model for the current layer.
            for model in models {
                cubemap_pass.run(&mut encoder, &layer_view, &depth_view, &camera, model);
            }

            queue.submit(Some(encoder.finish()));
        });

        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));

        Ok(())
    }
}
