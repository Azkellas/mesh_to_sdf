use anyhow::Result;

use hashbrown::HashMap;
use itertools::{Itertools, MinMaxResult};
use wgpu::util::DeviceExt;
use wgpu::Extent3d;
use winit_input_helper::WinitInputHelper;

use egui_gizmo::GizmoMode;

use crate::camera::CameraData;
use crate::cubemap::Cubemap;
use crate::frame_rate::FrameRate;

use crate::gltf;
use crate::pbr::model::Model;
use crate::pbr::model_instance::ModelInstance;
use crate::texture::{self, Texture};

use crate::sdf::Sdf;

mod command_stack;
mod ui;

struct ModelInfo {
    pub vertex_count: usize,
    pub index_count: usize,
    pub triangle_count: usize,

    pub bounding_box: [f32; 6],
}

struct LastRunInfo {
    pub time: core::time::Duration,
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
            0 => Ok(Self::Snap),
            1 => Ok(Self::Trilinear),
            2 => Ok(Self::Tetrahedral),
            3 => Ok(Self::SnapStylized),
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
    enable_backface_culling: bool,
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
    models: HashMap<usize, Model>,
    model_instances: Vec<ModelInstance>,
    model_info: Option<ModelInfo>,
    last_run_info: Option<LastRunInfo>,
    command_stack: command_stack::CommandStack,
    alert_message: Option<(String, web_time::Instant)>,

    cubemap: Cubemap,

    sdf_vertices: Vec<glam::Vec3>,
    sdf_indices: Vec<u32>,
}

impl SdfProgram {
    pub const fn optional_features() -> wgpu::Features {
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

    pub const fn required_features() -> wgpu::Features {
        wgpu::Features::CLEAR_TEXTURE
    }

    pub fn process_input(&mut self, input: &WinitInputHelper) -> bool {
        #[expect(clippy::useless_let_if_seq)]
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
    pub const fn get_name() -> &'static str {
        "SDF Client"
    }

    #[expect(clippy::too_many_lines)]
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
        let camera = CameraData::new(device);

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

        let parameters = Parameters {
            file_name: None,
            gizmo_mode: GizmoMode::Translate,
            cell_count: [16, 16, 16],
            render_mode: RenderMode::Sdf,
            sdf_sign_method: mesh_to_sdf::SignMethod::default(),
            enable_shadows: false, // deactivating shadows for now.
            enable_backface_culling: false,
        };

        let model_render_pass = crate::passes::model_render_pass::ModelRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &shadow_pass.map,
            parameters.enable_backface_culling,
        )?;

        let cubemap = Cubemap::new(device, 2048);

        let voxels = crate::passes::voxel_render_pass::VoxelRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &settings.bind_group_layout,
            &shadow_pass.map,
            cubemap.get_bind_group_layout(),
        )?;

        let raymarch = crate::passes::raymarch_pass::RaymarchRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &settings.bind_group_layout,
            &shadow_pass.map,
            cubemap.get_bind_group_layout(),
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

        Ok(Self {
            pass,
            settings,
            frame_rate: FrameRate::new(100),
            last_update: web_time::Instant::now(),
            camera,
            depth_map,
            parameters,
            sdf: None,
            models: HashMap::new(),
            model_instances: vec![],
            model_info: None,
            last_run_info: None,
            command_stack: command_stack::CommandStack::new(20),
            alert_message: None,
            cubemap,
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

        self.pass.model.update_pipeline(
            device,
            swapchain_format,
            &self.camera,
            self.parameters.enable_backface_culling,
        )?;

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
            self.cubemap.get_bind_group_layout(),
        )?;

        self.pass.raymarch.update_pipeline(
            device,
            swapchain_format,
            &self.camera,
            &self.settings.bind_group_layout,
            self.cubemap.get_bind_group_layout(),
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

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        surface: &wgpu::Surface,
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
    ) {
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

        if self.parameters.enable_backface_culling != self.pass.model.is_culling_backfaces() {
            let swapchain_capabilities = surface.get_capabilities(adapter);
            let swapchain_format = swapchain_capabilities.formats[0];
            self.pass
                .model
                .update_pipeline(
                    device,
                    swapchain_format,
                    &self.camera,
                    self.parameters.enable_backface_culling,
                )
                .unwrap();
        }
    }

    pub fn render(&self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
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
                for model_instance in &self.model_instances {
                    let model = &self.models[&model_instance.model_id];
                    self.pass
                        .shadow
                        .run(&mut command_encoder, model, model_instance);
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
                for model_instance in &self.model_instances {
                    let model = &self.models[&model_instance.model_id];

                    self.pass.model.run(
                        &mut command_encoder,
                        view,
                        &self.depth_map,
                        &self.camera,
                        model,
                        model_instance,
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
                self.cubemap.get_bind_group(),
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
                self.cubemap.get_bind_group(),
            );

            queue.submit(Some(command_encoder.finish()));
        }
    }

    pub fn get_camera(&mut self) -> &mut crate::camera_control::CameraLookAt {
        &mut self.camera.camera.look_at
    }

    fn load_gltf(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        match &self.parameters.file_name {
            None => anyhow::bail!("No file to load"),
            Some(path) => {
                // a gltf scene can contain multiple models.
                // we merge them in a single sdf.
                let (models, instances) = gltf::load_scene(device, queue, path)?;
                self.models = models;
                self.model_instances = instances;

                let (vertices, indices) = self.model_instances.iter().fold(
                    (vec![], vec![]),
                    |(mut vertices, mut indices), model_instance| {
                        let model = &self.models[&model_instance.model_id];
                        let transform = model_instance.transform;
                        // we need to offset the indices by the number of vertices we already have.
                        let len = vertices.len();
                        vertices.extend(model.mesh.vertices.iter().map(|v| {
                            transform.transform_point3(glam::Vec3::from_array(v.position))
                        }));
                        indices.extend(model.mesh.indices.iter().map(|i| *i + len as u32));
                        (vertices, indices)
                    },
                );

                // TODO: do it in a single pass.
                let MinMaxResult::MinMax(xmin, xmax) = vertices.iter().map(|v| v.x).minmax() else {
                    anyhow::bail!("Bounding box is ill-defined")
                };
                let MinMaxResult::MinMax(ymin, ymax) = vertices.iter().map(|v| v.y).minmax() else {
                    anyhow::bail!("Bounding box is ill-defined")
                };
                let MinMaxResult::MinMax(zmin, zmax) = vertices.iter().map(|v| v.z).minmax() else {
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

                let Some(model_info) = &self.model_info else {
                    anyhow::bail!("No model to generate SDF from")
                };

                self.cubemap.generate(
                    device,
                    queue,
                    model_info.bounding_box,
                    &self.models,
                    &self.model_instances,
                    &self.pass.cube_map,
                );
            }
        }
        Ok(())
    }

    fn generate_sdf(&mut self, device: &wgpu::Device) -> Result<()> {
        let Some(model_info) = &self.model_info else {
            anyhow::bail!("No model to generate SDF from")
        };

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

        let start_cell = glam::Vec3::new(xmin, ymin, zmin);
        let end_cell = glam::Vec3::new(xmax, ymax, zmax);

        self.sdf = Some(Sdf::new(
            device,
            &self.sdf_vertices,
            &self.sdf_indices,
            &start_cell,
            &end_cell,
            &self.parameters.cell_count,
            self.parameters.sdf_sign_method,
        ));

        self.last_run_info = Some(LastRunInfo {
            time: self.sdf.as_ref().unwrap().time_taken,
            size: self.parameters.cell_count,
        });

        Ok(())
    }
}
