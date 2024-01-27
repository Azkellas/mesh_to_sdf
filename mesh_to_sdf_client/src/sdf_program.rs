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
use crate::sdf_render_pass::SdfRenderPass;

use crate::camera_control::CameraLookAt;

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

struct Parameters {
    file_name: Option<String>,
    gizmo_mode: GizmoMode,
    cell_count: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Settings {
    // TODO: remove padding?
    positive_color: [f32; 3],
    _positive_padding: f32,
    negative_color: [f32; 3],
    _negative_padding: f32,
    surface_color: [f32; 3],
    _surface_padding: f32,
    positives_power: f32,
    negatives_power: f32,
    surface_power: f32,
    surface_width: f32,
    point_size: f32,
    _padding: [f32; 3],
}

struct SettingsData {
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
    sdf: SdfRenderPass,
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
    model_info: Option<ModelInfo>,
    last_run_info: Option<LastRunInfo>,

    alert_message: Option<(String, web_time::Instant)>,
}

impl SdfProgram {
    pub fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    pub fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }

    pub fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    pub fn required_features() -> wgpu::Features {
        wgpu::Features::empty() | wgpu::Features::CLEAR_TEXTURE
    }

    pub fn process_input(&mut self, _input: &WinitInputHelper) -> bool {
        // nothing to do here for now.
        false
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

        let depth_map = texture::Texture::create_depth_texture(device, &size, "depth_texture");
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
                positives_power: 1.0,
                negatives_power: 1.0,
                surface_power: 1.0,
                surface_width: 0.02,
                point_size: 0.3,
                _padding: [0.0; 3],
            },
        );

        let sdf_pass = SdfRenderPass::new(
            device,
            swapchain_format,
            &camera,
            &settings.bind_group_layout,
        )?;

        let pass = Passes { sdf: sdf_pass };

        let parameters = Parameters {
            file_name: None,
            gizmo_mode: GizmoMode::Translate,
            cell_count: [16, 16, 16],
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
            model_info: None,
            last_run_info: None,
            alert_message: None,
        })
    }

    /// update is called for any `WindowEvent` not handled by the framework
    pub fn update_passes(
        &mut self,
        surface: &wgpu::Surface,
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
    ) -> Result<()> {
        // self.passes.render_model.update_pipeline(
        //     device,
        //     &self.camera,
        //     &self.passes.hdri.bind_group_layout,
        // )?;

        let swapchain_capabilities = surface.get_capabilities(adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        self.pass.sdf.update_pipeline(
            device,
            swapchain_format,
            &self.camera,
            &self.settings.bind_group_layout,
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

        self.depth_map = texture::Texture::create_depth_texture(device, &size, "depth_texture");
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

        // TODO: do not recompute this each frame.
        let camera = &mut self.camera;
        camera.uniform.update_view_proj(&camera.camera);
        queue.write_buffer(&camera.buffer, 0, bytemuck::cast_slice(&[camera.uniform]));
    }

    /// Draw ui with egui.
    pub fn draw_gizmos(&mut self, _ui: &mut egui::Ui) {
        // TODO: backport transform.
    }

    /// Draw ui with egui.
    pub fn draw_ui(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, ui: &mut egui::Ui) {
        egui::Grid::new("settings").show(ui, |ui| {
            ui.separator();
            ui.end_row();

            ui.label("Framerate:");
            ui.label(std::format!("{:.0}fps", self.frame_rate.get()));
            ui.end_row();

            ui.separator();
            ui.end_row();

            match self.parameters.file_name {
                None => {
                    ui.label("No file loaded");
                }
                Some(ref file_name) => {
                    ui.label("Current file");
                    ui.label(file_name);
                }
            };
            ui.end_row();

            if let Some(ref model_info) = &self.model_info {
                ui.label("Vertex count");
                ui.label(model_info.vertex_count.to_string());
                ui.end_row();

                ui.label("Index count");
                ui.label(model_info.index_count.to_string());
                ui.end_row();

                ui.label("Triangle count");
                ui.label(model_info.triangle_count.to_string());
                ui.end_row();

                ui.label("Bounding box min");
                ui.label(format!(
                    "x: {:.2} y: {:.2} z: {:.2}",
                    model_info.bounding_box[0],
                    model_info.bounding_box[1],
                    model_info.bounding_box[2]
                ));
                ui.end_row();

                ui.label("Bounding box max");
                ui.label(format!(
                    "x: {:.2} y: {:.2} z: {:.2}",
                    model_info.bounding_box[3],
                    model_info.bounding_box[4],
                    model_info.bounding_box[4]
                ));
                ui.end_row();
            }

            if ui.button("Open file…").clicked() {
                if let Some(path) = pollster::block_on(
                    rfd::AsyncFileDialog::new()
                        .add_filter("gltf", &["gltf", "glb"])
                        .pick_file(),
                ) {
                    #[cfg(target_arch = "wasm32")]
                    {
                        let path = path.inner().to_string().as_string().unwrap();
                        self.parameters.file_name = Some(path);
                        if self.load_gltf(device, queue).is_err() {
                            self.alert_message = Some((
                                "Failed to load file. Make sure it is a valid gltf file."
                                    .to_owned(),
                                web_time::Instant::now(),
                            ));
                        }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let path = path.inner().to_str().unwrap().to_owned();
                        self.parameters.file_name = Some(path);
                        if self.load_gltf(device, queue).is_err() {
                            self.alert_message = Some((
                                "Failed to load file. Make sure it is a valid gltf file."
                                    .to_owned(),
                                web_time::Instant::now(),
                            ));
                        }
                    }
                }
            }

            ui.end_row();

            ui.separator();

            ui.end_row();

            ui.label("Positive Color");
            egui::color_picker::color_edit_button_rgb(
                ui,
                &mut self.settings.settings.positive_color,
            );

            ui.end_row();

            ui.label("Negative Color");
            egui::color_picker::color_edit_button_rgb(
                ui,
                &mut self.settings.settings.negative_color,
            );

            ui.end_row();

            ui.label("Surface Color");
            egui::color_picker::color_edit_button_rgb(
                ui,
                &mut self.settings.settings.surface_color,
            );

            ui.end_row();

            ui.label("Positive power");
            ui.add(egui::Slider::new(
                &mut self.settings.settings.positives_power,
                0.0..=10.0,
            ));

            ui.end_row();

            ui.label("Negative power");
            ui.add(egui::Slider::new(
                &mut self.settings.settings.negatives_power,
                0.0..=10.0,
            ));

            ui.end_row();

            ui.label("Surface power");
            ui.add(egui::Slider::new(
                &mut self.settings.settings.surface_power,
                0.0..=10.0,
            ));

            ui.end_row();

            ui.label("Surface width");
            ui.add(egui::Slider::new(
                &mut self.settings.settings.surface_width,
                0.001..=0.1,
            ));

            ui.end_row();

            ui.label("Point size");
            ui.add(egui::Slider::new(
                &mut self.settings.settings.point_size,
                0.1..=1.,
            ));
            ui.end_row();

            ui.separator();
            ui.end_row();

            ui.label("Cell count");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.parameters.cell_count[0])
                        .clamp_range(1..=64)
                        .prefix("x: "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.parameters.cell_count[1])
                        .clamp_range(1..=64)
                        .prefix("y: "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.parameters.cell_count[2])
                        .clamp_range(1..=64)
                        .prefix("z: "),
                );
            });
            ui.end_row();

            ui.label("Cell size");
            ui.horizontal(|ui| {
                let (x_min, x_max) = if let Some(ref model_info) = &self.model_info {
                    (model_info.bounding_box[0], model_info.bounding_box[3])
                } else {
                    (0.0, 1.0)
                };
                let (y_min, y_max) = if let Some(ref model_info) = &self.model_info {
                    (model_info.bounding_box[1], model_info.bounding_box[4])
                } else {
                    (0.0, 1.0)
                };
                let (z_min, z_max) = if let Some(ref model_info) = &self.model_info {
                    (model_info.bounding_box[2], model_info.bounding_box[5])
                } else {
                    (0.0, 1.0)
                };

                // TODO: depends on the bounding box of the sdf.
                let mut size_x = (x_max - x_min) / self.parameters.cell_count[0] as f32;
                ui.add(
                    egui::DragValue::new(&mut size_x)
                        .prefix("x: ")
                        .speed(0.001)
                        .max_decimals(3),
                );
                self.parameters.cell_count[0] = ((x_max - x_min) / size_x) as u32;

                let mut size_y = (y_max - y_min) / self.parameters.cell_count[1] as f32;
                ui.add(
                    egui::DragValue::new(&mut size_y)
                        .prefix("y: ")
                        .speed(0.001)
                        .max_decimals(3),
                );
                self.parameters.cell_count[1] = ((y_max - y_min) / size_y) as u32;

                let mut size_z = (z_max - z_min) / self.parameters.cell_count[2] as f32;
                ui.add(
                    egui::DragValue::new(&mut size_z)
                        .prefix("z: ")
                        .speed(0.001)
                        .max_decimals(3),
                );
                self.parameters.cell_count[2] = ((z_max - z_min) / size_z) as u32;
            });
            ui.end_row();

            ui.separator();
            ui.end_row();

            if ui.button("Generate").clicked() {
                let _ = self.load_gltf(device, queue); // TODO: don't ignore error.
            }
            ui.end_row();

            if let Some(ref run_info) = self.last_run_info {
                ui.label(format!(
                    "Last generation: {:.0}ms with size {}x{}x{} = {}",
                    run_info.time,
                    run_info.size[0],
                    run_info.size[1],
                    run_info.size[2],
                    run_info.size[0] * run_info.size[1] * run_info.size[2]
                ));
            }
            ui.end_row();
        });

        if ui.button("test alert").clicked() {
            self.alert_message = Some((
                "This is a test alert message".to_owned(),
                web_time::Instant::now(),
            ));
        }

        if let Some((ref msg, ref start)) = self.alert_message {
            let timeout_s = 3.0_f32;
            if start.elapsed().as_secs_f32() > timeout_s {
                self.alert_message = None;
            } else {
                // TODO: chose one?
                ui.label(egui::RichText::new(msg).color(egui::Color32::RED));
                egui::Window::new("")
                    .title_bar(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                    .fixed_size(egui::Vec2::new(300.0, 100.0))
                    .show(ui.ctx(), |ui| {
                        ui.label(egui::RichText::new(msg).color(egui::Color32::RED));
                    });
            }
        }
    }

    pub fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Clear depth.
        {
            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            command_encoder.clear_texture(
                &self.depth_map.texture,
                &wgpu::ImageSubresourceRange::default(),
            );

            queue.submit(Some(command_encoder.finish()));
        }

        // render models
        // {
        //     let mut command_encoder =
        //         device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        //     // need to draw it each frame to update depth map.
        //     command_encoder.push_debug_group("render models");
        //     {
        //         for model in &self.models {
        //             self.passes.render_model.run(
        //                 &mut command_encoder,
        //                 &self.render_target,
        //                 &self.depth_map,
        //                 &self.camera,
        //                 &self.passes.draw_shadow_map.bind_group,
        //                 &self.passes.hdri.bind_group,
        //                 model,
        //             );
        //         }
        //     }
        //     command_encoder.pop_debug_group();

        //     queue.submit(Some(command_encoder.finish()));
        // }

        // render sdf
        if self.sdf.is_some() {
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
                let start = web_time::Instant::now();

                // a gltf scene can contain multiple models.
                // we merge them in a single sdf.
                let models = gltf::load_scene(device, queue, &path)?;

                let (vertices, indices) =
                    models
                        .iter()
                        .fold((vec![], vec![]), |(mut vertices, mut indices), model| {
                            // we need to offset the indices by the number of vertices we already have.
                            let len = vertices.len();
                            vertices.extend(model.vertices().iter().map(|v| *v));
                            indices
                                .extend(model.indices().unwrap().iter().map(|i| *i + len as u32));
                            (vertices, indices)
                        });

                // TODO: c/c'd from sdf.rs
                let MinMaxResult::MinMax(xmin, xmax) =
                    vertices.iter().map(|v| v.position.x).minmax()
                else {
                    anyhow::bail!("Bounding box is ill-defined")
                };
                let MinMaxResult::MinMax(ymin, ymax) =
                    vertices.iter().map(|v| v.position.y).minmax()
                else {
                    anyhow::bail!("Bounding box is ill-defined")
                };
                let MinMaxResult::MinMax(zmin, zmax) =
                    vertices.iter().map(|v| v.position.z).minmax()
                else {
                    anyhow::bail!("Bounding box is ill-defined")
                };

                let model_info = ModelInfo {
                    vertex_count: vertices.len(),
                    index_count: indices.len(),
                    triangle_count: indices.len() / 3,
                    bounding_box: [xmin, ymin, zmin, xmax, ymax, zmax],
                };
                self.model_info = Some(model_info);

                let cell_radius = [
                    (xmax - xmin) / self.parameters.cell_count[0] as f32,
                    (ymax - ymin) / self.parameters.cell_count[1] as f32,
                    (zmax - zmin) / self.parameters.cell_count[2] as f32,
                ];
                self.sdf = Sdf::new(device, &vertices, &indices, cell_radius).ok();

                self.last_run_info = Some(LastRunInfo {
                    time: start.elapsed().as_secs_f32() * 1000.0,
                    size: self.parameters.cell_count,
                });

                Ok(())
            }
        }
    }
}
