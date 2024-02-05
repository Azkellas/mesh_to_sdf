use anyhow::Result;

use crate::camera::CameraData;
use crate::pbr::mesh::primitives::create_box;
use crate::pbr::mesh::{Mesh, MeshVertex};
use crate::pbr::shadow_map;
use crate::sdf::{voxels_indices_cmp, Sdf, SdfUniforms};
use crate::sdf_program::SettingsData;
use crate::texture::Texture;
use crate::utility::shader_builder::ShaderBuilder;

pub struct VoxelRenderPass {
    pub render_pipeline: wgpu::RenderPipeline,
    pub voxel_bind_group_layout: wgpu::BindGroupLayout,

    pub shadow_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub shadow_bind_group: Option<wgpu::BindGroup>,

    voxel: Mesh,
}

impl VoxelRenderPass {
    fn create_pipeline(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        voxel_bind_group_layout: &wgpu::BindGroupLayout,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline> {
        let draw_shader = ShaderBuilder::create_module(device, "draw_voxels.wgsl")?;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("VoxelRenderPass::PipelineLayout"),
                bind_group_layouts: &[
                    voxel_bind_group_layout,
                    &camera.bind_group_layout,
                    settings_bind_group_layout,
                    shadow_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        };

        ShaderBuilder::create_render_pipeline(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("VoxelRenderPass::RenderPipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: "main_vs",
                    buffers: &[MeshVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &draw_shader,
                    entry_point: "main_fs",
                    targets: &[Some(view_format.into())],
                }),
                primitive,
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

    pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sdf Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SdfUniforms>() as _
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn update_pipeline(
        &mut self,
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<()> {
        self.render_pipeline = Self::create_pipeline(
            device,
            view_format,
            camera,
            &self.voxel_bind_group_layout,
            settings_bind_group_layout,
            self.shadow_bind_group_layout.as_ref().unwrap(),
        )?;
        Ok(())
    }

    fn create_shadow_bind_group(
        device: &wgpu::Device,
        shadow_map: &shadow_map::ShadowMap,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_map.light.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_map.texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_map.texture.sampler),
                },
            ],
            label: Some("Model shadow bind group"),
        })
    }

    fn create_shadow_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
            label: Some("Model shadow bind group layout"),
        })
    }

    pub fn new(
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
        camera: &CameraData,
        settings_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_map: &shadow_map::ShadowMap,
    ) -> Result<Self> {
        // TODO: copy pasted from model_render_pass, maybe we can merge the two.
        let voxel_bind_group_layout = Self::get_bind_group_layout(device);

        let render_shadow_bind_group_layout = Self::create_shadow_bind_group_layout(device);

        let render_shadow_bind_group =
            Self::create_shadow_bind_group(device, shadow_map, &render_shadow_bind_group_layout);

        let render_pipeline = VoxelRenderPass::create_pipeline(
            device,
            view_format,
            camera,
            &voxel_bind_group_layout,
            settings_bind_group_layout,
            &render_shadow_bind_group_layout,
        )?;

        let voxel = create_box(device);

        Ok(VoxelRenderPass {
            render_pipeline,
            voxel_bind_group_layout,
            voxel,

            shadow_bind_group_layout: Some(render_shadow_bind_group_layout),
            shadow_bind_group: Some(render_shadow_bind_group),
        })
    }

    pub fn run(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_map: &Texture,
        camera: &CameraData,
        sdf: &Sdf,
        settings: &SettingsData,
    ) {
        // need to draw it each frame to update depth map.
        command_encoder.push_debug_group("render particles");
        {
            let render_pass_descriptor = wgpu::RenderPassDescriptor {
                label: Some("VoxelRenderPass::run::render_pass_descriptor"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_map.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            };

            // find the last index that is less than the surface width
            // all indices before this index are valid and should be drawn.
            // note: if several cells are at exactly the surface width, we might miss some.
            // see `voxels_indices_cmp` for more details on how the comparison is done.
            let cell_size = [
                sdf.uniforms.cell_size[0],
                sdf.uniforms.cell_size[1],
                sdf.uniforms.cell_size[2],
            ];
            let index = sdf.ordered_indices.binary_search_by(|i| {
                voxels_indices_cmp(
                    sdf.data[*i as usize],
                    settings.settings.surface_width,
                    cell_size,
                )
            });
            let index = match index {
                Ok(i) | Err(i) => i,
            };

            // render pass
            let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &sdf.bind_group, &[]);
            rpass.set_bind_group(1, &camera.bind_group, &[]);
            rpass.set_bind_group(2, &settings.bind_group, &[]);
            rpass.set_bind_group(3, self.shadow_bind_group.as_ref().unwrap(), &[]);
            rpass.set_vertex_buffer(0, self.voxel.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.voxel.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.voxel.index_count, 0, 0..(index as u32));
        }
        command_encoder.pop_debug_group();
    }
}
