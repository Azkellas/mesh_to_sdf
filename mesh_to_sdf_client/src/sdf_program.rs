use anyhow::Result;

use itertools::{Itertools, MinMaxResult};
use wgpu::util::DeviceExt;
use wgpu::Extent3d;
use winit_input_helper::WinitInputHelper;

use egui_gizmo::GizmoMode;

use crate::camera::*;
use crate::frame_rate::FrameRate;

use crate::gltf;
use crate::texture::{self, Texture};

use crate::sdf::Sdf;

use crate::camera_control::CameraLookAt;

mod command_stack;
mod ui;

struct ModelInfo {
    pub vertex_count: usize,
    pub index_count: usize,
    pub triangle_count: usize,

    pub bounding_box: [f32; 6],
}

struct LastRunInfo {
    pub time: f32,
    pub size: [u32; 3],
}

#[derive(Debug, Clone, PartialEq)]
enum RenderMode {
    Model,
    Sdf,
    ModelAndSdf,
    Voxels,
    Raymarch,
}

#[derive(Debug, Clone, PartialEq)]
enum RaymarchMode {
    Snap,
    Trilinear,
    Tetrahedral,
    SnapStylized,
}

impl TryFrom<u32> for RaymarchMode {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RaymarchMode::Snap),
            1 => Ok(RaymarchMode::Trilinear),
            2 => Ok(RaymarchMode::Tetrahedral),
            3 => Ok(RaymarchMode::SnapStylized),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
struct Parameters {
    file_name: Option<String>,
    gizmo_mode: GizmoMode,
    cell_count: [u32; 3],
    render_mode: RenderMode,
    sdf_sign_method: mesh_to_sdf::SignMethod,
    enable_shadows: bool,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Settings {
    // TODO: remove padding?
    pub positive_color: [f32; 3],
    pub _positive_padding: f32,
    pub negative_color: [f32; 3],
    pub _negative_padding: f32,
    pub surface_color: [f32; 3],
    pub _surface_padding: f32,
    pub positives_power: f32,
    pub negatives_power: f32,
    pub surface_iso: f32,
    pub surface_power: f32,
    pub surface_width: f32,
    pub point_size: f32,
    pub raymarch_mode: u32,
    pub bounding_box_extent: f32,
    pub mesh_bbox_min: [f32; 4],
    pub mesh_bbox_max: [f32; 4],
    pub map_material: u32, // 0: no, > 0: add material to voxels and raymarch
    pub _padding: [f32; 3],
}

pub struct SettingsData {
    pub settings: Settings,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl SettingsData {
    pub fn new(device: &wgpu::Device, settings: Settings) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_program::SDF Settings Buffer"),
            contents: bytemuck::cast_slice(&[settings]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(buffer.size()),
                },
                count: None,
            }],
            label: Some("sdf_program::settings_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("sdf_program::settings_bind_group"),
        });

        Self {
            settings,
            buffer,
            bind_group,
            bind_group_layout,
        }
    }
}

struct Passes {
    sdf: crate::passes::sdf_render_pass::SdfRenderPass,
    mip_gen: crate::passes::mip_generation_pass::MipGenerationPass,
    model: crate::passes::model_render_pass::ModelRenderPass,
    shadow: crate::passes::shadow_pass::ShadowPass,
    voxels: crate::passes::voxel_render_pass::VoxelRenderPass,
    raymarch: crate::passes::raymarch_pass::RaymarchRenderPass,
    cube_map: crate::passes::cubemap_generation_pass::CubemapGenerationPass,
}

pub struct SdfProgram {
    parameters: Parameters,
    settings: SettingsData,
    pass: Passes,
    frame_rate: FrameRate,
    last_update: web_time::Instant,
    camera: CameraData,
    depth_map: Texture,
    sdf: Option<Sdf>,
    models: Vec<crate::pbr::model::Model>,
    model_info: Option<ModelInfo>,
    last_run_info: Option<LastRunInfo>,
    command_stack: command_stack::CommandStack,
    alert_message: Option<(String, web_time::Instant)>,

    cubemap: Texture,
    cubemap_depth: Texture,
    cubemap_uniforms: wgpu::Buffer,
    cubemap_bind_group: wgpu::BindGroup,
    cubemap_bind_group_layout: wgpu::BindGroupLayout,
    sdf_vertices: Vec<[f32; 3]>,
    sdf_indices: Vec<u32>,
}

impl SdfProgram {
    pub fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    pub fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::VERTEX_STORAGE,
            shader_model: wgpu::ShaderModel::Sm5,
            ..wgpu::DownlevelCapabilities::default()
        }
    }

    pub fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_bind_groups: 8,
            ..Default::default()
        }
    }

    pub fn required_features() -> wgpu::Features {
        wgpu::Features::CLEAR_TEXTURE
    }

    #[allow(clippy::unused_self)]
    pub fn process_input(&mut self, input: &WinitInputHelper) -> bool {
        let mut captured = false;

        if input.held_control() && input.key_released(winit::keyboard::KeyCode::KeyZ) {
            if let Some(command) = self.command_stack.undo() {
                self.parameters = command.old_state.parameters;
                self.settings.settings = command.old_state.settings;
            }
            captured = true;
        }
        if input.held_control() && input.key_released(winit::keyboard::KeyCode::KeyY) {
            if let Some(command) = self.command_stack.redo() {
                self.parameters = command.new_state.parameters;
                self.settings.settings = command.new_state.settings;
            }
            captured = true;
        }

        captured
    }

    /// Get program name.
    pub fn get_name() -> &'static str {
        "SDF Client"
    }

    pub fn init(
        surface: &wgpu::Surface,
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        _queue: &wgpu::Queue,
    ) -> Result<Self> {
        let swapchain_capabilities = surface.get_capabilities(adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let size = surface.get_current_texture().unwrap().texture.size();

        let depth_map = texture::Texture::create_depth_texture(device, size, "depth_texture");
        let camera = Self::create_camera(device);

        let settings = SettingsData::new(
            device,
            Settings {
                positive_color: [0.0, 1.0, 0.0],
                _positive_padding: 0.0,
                negative_color: [0.0, 0.0, 1.0],
                _negative_padding: 0.0,
                surface_color: [1.0, 0.0, 0.0],
                _surface_padding: 0.0,
                positives_power: 0.1,
                negatives_power: 1.0,
                surface_iso: 0.0,
                surface_power: 1.0,
                surface_width: 0.02,
                point_size: 0.3,
                raymarch_mode: RaymarchMode::Trilinear as u32,
                bounding_box_extent: 1.1,
                mesh_bbox_min: [0.0, 0.0, 0.0, 0.0],
                mesh_bbox_max: [0.0, 0.0, 0.0, 0.0],
                map_material: 1, // map material on voxels and raymarch
                _padding: [0.0, 0.0, 0.0],
            },
        );

        let sdf_pass = crate::passes::sdf_render_pass::SdfRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &settings.bind_group_layout,
        )?;

        let mip_gen_pass = crate::passes::mip_generation_pass::MipGenerationPass::new(device)?;

        let shadow_map = crate::pbr::shadow_map::ShadowMap::new(device);

        let shadow_pass = crate::passes::shadow_pass::ShadowPass::new(device, shadow_map)?;

        let model_render_pass = crate::passes::model_render_pass::ModelRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &shadow_pass.map,
        )?;

        let parameters = Parameters {
            file_name: None,
            gizmo_mode: GizmoMode::Translate,
            cell_count: [16, 16, 16],
            render_mode: RenderMode::Sdf,
            sdf_sign_method: mesh_to_sdf::SignMethod::default(),
            enable_shadows: false, // deactivating shadows for now.
        };

        let size3d = wgpu::Extent3d {
            width: 2048,
            height: 2048,
            depth_or_array_layers: 6,
        };
        let cubemap = Texture::create_render_target(device, size3d, Some("mesh_cubemap"), false);
        let cubemap_depth = Texture::create_depth_texture(device, size3d, "mesh_cubemap_depth");
        let cubemap_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cubemap Uniform Buffer"),
            size: 6 * 16 * std::mem::size_of::<f32>() as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cubemap_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                            min_binding_size: wgpu::BufferSize::new(cubemap_uniforms.size()),
                        },
                        count: None,
                    },
                ],
                label: Some("cubemap_bind_group_layout"),
            });

        let cubemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &cubemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cubemap.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&cubemap.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&cubemap_depth.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&cubemap_depth.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cubemap_uniforms.as_entire_binding(),
                },
            ],
            label: Some("cubemap_bind_group"),
        });

        let voxels = crate::passes::voxel_render_pass::VoxelRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &settings.bind_group_layout,
            &shadow_pass.map,
            &cubemap_bind_group_layout,
        )?;


        let raymarch = crate::passes::raymarch_pass::RaymarchRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &settings.bind_group_layout,
            &shadow_pass.map,
            &cubemap_bind_group_layout,
        )?;

        let cubemap_generation_pass =
            crate::passes::cubemap_generation_pass::CubemapGenerationPass::new(
                device,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                &camera,
            )?;
        let pass = Passes {
            sdf: sdf_pass,
            mip_gen: mip_gen_pass,
            model: model_render_pass,
            shadow: shadow_pass,
            voxels,
            raymarch,
            cube_map: cubemap_generation_pass,
        };

        Ok(SdfProgram {
            pass,
            settings,
            frame_rate: FrameRate::new(100),
            last_update: web_time::Instant::now(),
            camera,
            depth_map,
            parameters,
            sdf: None,
            models: vec![],
            model_info: None,
            last_run_info: None,
            command_stack: command_stack::CommandStack::new(20),
            alert_message: None,
            cubemap,
            cubemap_bind_group,
            cubemap_bind_group_layout,
            cubemap_depth,
            cubemap_uniforms,
            sdf_vertices: vec![],
            sdf_indices: vec![],
        })
    }

    /// update is called for any `WindowEvent` not handled by the framework
    pub fn update_passes(
        &mut self,
        surface: &wgpu::Surface,
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
    ) -> Result<()> {
        let swapchain_capabilities = surface.get_capabilities(adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        self.pass.mip_gen.update_pipeline(device)?;
        self.pass.shadow.update_pipeline(device)?;

        self.pass
            .model
            .update_pipeline(device, swapchain_format, &self.camera)?;

        self.pass.sdf.update_pipeline(
            device,
            swapchain_format,
            &self.camera,
            &self.settings.bind_group_layout,
        )?;

        self.pass.voxels.update_pipeline(
            device,
            swapchain_format,
            &self.camera,
            &self.settings.bind_group_layout,
            &self.cubemap_bind_group_layout,
        )?;

        self.pass.raymarch.update_pipeline(
            device,
            swapchain_format,
            &self.camera,
            &self.settings.bind_group_layout,
            &self.cubemap_bind_group_layout,
        )?;

        Ok(())
    }

    /// resize is called on `WindowEvent::Resized` events
    pub fn resize(
        &mut self,
        surface_configuration: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let size = Extent3d {
            width: surface_configuration.width,
            height: surface_configuration.height,
            depth_or_array_layers: 1,
        };

        self.depth_map = texture::Texture::create_depth_texture(device, size, "depth_texture");
        self.camera.update_resolution([size.width, size.height]);
    }

    pub fn update(&mut self, queue: &wgpu::Queue) {
        let last_frame_duration = self.last_update.elapsed().as_secs_f32();
        self.frame_rate.update(last_frame_duration);
        self.last_update = web_time::Instant::now();

        queue.write_buffer(
            &self.settings.buffer,
            0,
            bytemuck::cast_slice(&[self.settings.settings]),
        );

        let camera = &mut self.camera;
        camera.uniform.update_view_proj(&camera.camera);
        queue.write_buffer(&camera.buffer, 0, bytemuck::cast_slice(&[camera.uniform]));

        let shadow_map = &mut self.pass.shadow.map;
        shadow_map
            .light
            .uniform
            .update_view_proj(&shadow_map.light.camera);
        queue.write_buffer(
            &shadow_map.light.buffer,
            0,
            bytemuck::cast_slice(&[shadow_map.light.uniform]),
        );
    }

    pub fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Clear textures.
        {
            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // clear depth and render target
            command_encoder.clear_texture(
                &self.depth_map.texture,
                &wgpu::ImageSubresourceRange::default(),
            );

            command_encoder.clear_texture(
                &self.pass.shadow.map.texture.texture,
                &wgpu::ImageSubresourceRange::default(),
            );

            queue.submit(Some(command_encoder.finish()));
        }

        // shadow pass
        if self.parameters.enable_shadows {
            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            command_encoder.push_debug_group("render models shadow");
            {
                for model in &self.models {
                    self.pass.shadow.run(&mut command_encoder, model);
                }
            }
            command_encoder.pop_debug_group();

            queue.submit(Some(command_encoder.finish()));
        }

        // render models
        if self.parameters.render_mode == RenderMode::Model
            || self.parameters.render_mode == RenderMode::ModelAndSdf
        {
            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            command_encoder.push_debug_group("render models");
            {
                for model in &self.models {
                    self.pass.model.run(
                        &mut command_encoder,
                        view,
                        &self.depth_map,
                        &self.camera,
                        model,
                    );
                }
            }
            command_encoder.pop_debug_group();

            queue.submit(Some(command_encoder.finish()));
        }

        // render sdf
        if self.sdf.is_some()
            && (self.parameters.render_mode == RenderMode::Sdf
                || self.parameters.render_mode == RenderMode::ModelAndSdf)
        {
            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            self.pass.sdf.run(
                &mut command_encoder,
                view,
                &self.depth_map,
                &self.camera,
                self.sdf.as_ref().unwrap(),
                &self.settings.bind_group,
            );

            queue.submit(Some(command_encoder.finish()));
        }

        // render voxels
        if self.sdf.is_some() && self.parameters.render_mode == RenderMode::Voxels {
            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            self.pass.voxels.run(
                &mut command_encoder,
                view,
                &self.depth_map,
                &self.camera,
                self.sdf.as_ref().unwrap(),
                &self.settings,
                &self.cubemap_bind_group,
            );

            queue.submit(Some(command_encoder.finish()));
        }

        // render raymarch
        if self.sdf.is_some() && self.parameters.render_mode == RenderMode::Raymarch {
            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            self.pass.raymarch.run(
                &mut command_encoder,
                view,
                &self.camera,
                self.sdf.as_ref().unwrap(),
                &self.settings,
                &self.cubemap_bind_group,
            );

            queue.submit(Some(command_encoder.finish()));
        }
    }

    pub fn get_camera(&mut self) -> Option<&mut crate::camera_control::CameraLookAt> {
        Some(&mut self.camera.camera.look_at)
    }

    fn create_camera(device: &wgpu::Device) -> CameraData {
        let camera = Camera {
            look_at: CameraLookAt {
                center: glam::Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                longitude: 6.06,
                latitude: 0.37,
                distance: 1.66,
            },
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.1,
        };

        let camera_uniform = CameraUniform::from_camera(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(camera_buffer.size()),
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        CameraData {
            camera,
            uniform: camera_uniform,
            buffer: camera_buffer,
            bind_group: camera_bind_group,
            bind_group_layout: camera_bind_group_layout,
        }
    }

    fn load_gltf(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        match self.parameters.file_name {
            None => anyhow::bail!("No file to load"),
            Some(ref path) => {
                // a gltf scene can contain multiple models.
                // we merge them in a single sdf.
                self.models = gltf::load_scene(device, queue, path)?;

                let (vertices, indices) = self.models.iter().fold(
                    (vec![], vec![]),
                    |(mut vertices, mut indices), model| {
                        // we need to offset the indices by the number of vertices we already have.
                        let len = vertices.len();
                        vertices.extend(model.mesh.vertices.iter().map(|v| v.position));
                        indices.extend(model.mesh.indices.iter().map(|i| *i + len as u32));
                        (vertices, indices)
                    },
                );

                // TODO: do it in a single pass.
                let MinMaxResult::MinMax(xmin, xmax) = vertices.iter().map(|v| v[0]).minmax()
                else {
                    anyhow::bail!("Bounding box is ill-defined")
                };
                let MinMaxResult::MinMax(ymin, ymax) = vertices.iter().map(|v| v[1]).minmax()
                else {
                    anyhow::bail!("Bounding box is ill-defined")
                };
                let MinMaxResult::MinMax(zmin, zmax) = vertices.iter().map(|v| v[2]).minmax()
                else {
                    anyhow::bail!("Bounding box is ill-defined")
                };

                let model_info = ModelInfo {
                    vertex_count: vertices.len(),
                    index_count: indices.len(),
                    triangle_count: indices.len() / 3,
                    bounding_box: [xmin, ymin, zmin, xmax, ymax, zmax],
                };
                self.settings.settings.mesh_bbox_min = [xmin, ymin, zmin, 0.0];
                self.settings.settings.mesh_bbox_max = [xmax, ymax, zmax, 0.0];
                self.model_info = Some(model_info);

                self.sdf_vertices = vertices;
                self.sdf_indices = indices;

                // Adapt surface width to the size of the model.
                self.settings.settings.surface_width =
                    (xmax - xmin).max(ymax - ymin).max(zmax - zmin) / 100.0;

                // Place the camera eye at the center of the model.
                self.camera.camera.look_at.center = glam::Vec3::new(
                    (xmin + xmax) / 2.0,
                    (ymin + ymax) / 2.0,
                    (zmin + zmax) / 2.0,
                );
                // And at a pertinent distance.
                self.camera.camera.look_at.distance =
                    (xmax - xmin).max(ymax - ymin).max(zmax - zmin) * 2.0;

                self.generate_sdf(device)?;
                self.generate_cubemap(device, queue)?;
                Ok(())
            }
        }
    }

    fn generate_sdf(&mut self, device: &wgpu::Device) -> Result<()> {
        let Some(model_info) = &self.model_info else {
            anyhow::bail!("No model to generate SDF from")
        };

        let start = web_time::Instant::now();
        let middle = [
            (model_info.bounding_box[0] + model_info.bounding_box[3]) / 2.0,
            (model_info.bounding_box[1] + model_info.bounding_box[4]) / 2.0,
            (model_info.bounding_box[2] + model_info.bounding_box[5]) / 2.0,
        ];
        let half_size = [
            (model_info.bounding_box[3] - model_info.bounding_box[0]) / 2.0,
            (model_info.bounding_box[4] - model_info.bounding_box[1]) / 2.0,
            (model_info.bounding_box[5] - model_info.bounding_box[2]) / 2.0,
        ];

        let bounding_box_extent = self.settings.settings.bounding_box_extent;
        let xmin = middle[0] - half_size[0] * bounding_box_extent;
        let xmax = middle[0] + half_size[0] * bounding_box_extent;
        let ymin = middle[1] - half_size[1] * bounding_box_extent;
        let ymax = middle[1] + half_size[1] * bounding_box_extent;
        let zmin = middle[2] - half_size[2] * bounding_box_extent;
        let zmax = middle[2] + half_size[2] * bounding_box_extent;

        let start_cell = [xmin, ymin, zmin];
        let end_cell = [xmax, ymax, zmax];

        self.sdf = Some(Sdf::new(
            device,
            &self.sdf_vertices,
            &self.sdf_indices,
            &start_cell,
            &end_cell,
            &self.parameters.cell_count,
            self.parameters.sdf_sign_method,
        )?);

        self.last_run_info = Some(LastRunInfo {
            time: start.elapsed().as_secs_f32() * 1000.0,
            size: self.parameters.cell_count,
        });

        Ok(())
    }

    fn generate_cubemap(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        let size = [2048, 2048];

        let mut camera = Self::create_camera(device);

        let bounding_box = self.model_info.as_ref().unwrap().bounding_box;
        let bbx = 0.5 * (bounding_box[3] - bounding_box[0]);
        let bby = 0.5 * (bounding_box[4] - bounding_box[1]);
        let bbz = 0.5 * (bounding_box[5] - bounding_box[2]);
        let bb_center = glam::Vec3::new(
            (bounding_box[0] + bounding_box[3]) * 0.5,
            (bounding_box[1] + bounding_box[4]) * 0.5,
            (bounding_box[2] + bounding_box[5]) * 0.5,
        );

        let mut uniforms = [glam::Mat4::IDENTITY; 6];

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

            camera.uniform.view_proj = proj * view;
            camera.uniform.view = view;
            camera.uniform.proj = proj;
            camera.uniform.view_inv = view.inverse();
            camera.uniform.proj_inv = proj.inverse();
            camera.uniform.eye = eye.extend(1.0);
            camera.uniform.resolution = size;

            uniforms[i as usize] = camera.uniform.view_proj;

            queue.write_buffer(&camera.buffer, 0, bytemuck::cast_slice(&[camera.uniform]));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("generate_cubemap"),
            });

            let layer_view = self
                .cubemap
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                });

            let depth_view = self
                .cubemap_depth
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                });

            let whole_texture_range = wgpu::ImageSubresourceRange {
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i,
                array_layer_count: Some(1),
            };

            encoder.clear_texture(&self.cubemap.texture, &whole_texture_range);
            encoder.clear_texture(&self.cubemap_depth.texture, &whole_texture_range);

            for model in &self.models {
                self.pass
                    .cube_map
                    .run(&mut encoder, &layer_view, &depth_view, &camera, model);
            }

            queue.submit(Some(encoder.finish()));
        });

        queue.write_buffer(&self.cubemap_uniforms, 0, bytemuck::bytes_of(&uniforms));

        self.cubemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.cubemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.cubemap.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.cubemap.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.cubemap_depth.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.cubemap_depth.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.cubemap_uniforms.as_entire_binding(),
                },
            ],
            label: Some("cubemap_bind_group"),
        });

        Ok(())
    }
}
