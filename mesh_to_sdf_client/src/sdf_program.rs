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
}

#[derive(Debug, Clone)]
struct Parameters {
    file_name: Option<String>,
    gizmo_mode: GizmoMode,
    cell_count: [u32; 3],
    render_mode: RenderMode,
    enable_shadows: bool,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
    sdf: crate::passes::sdf_render_pass::SdfRenderPass,
    mip_gen: crate::passes::mip_generation_pass::MipGenerationPass,
    model: crate::passes::model_render_pass::ModelRenderPass,
    shadow: crate::passes::shadow_pass::ShadowPass,
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
        // Stricter than default.
        wgpu::Limits::downlevel_defaults()
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

        let pass = Passes {
            sdf: sdf_pass,
            mip_gen: mip_gen_pass,
            model: model_render_pass,
            shadow: shadow_pass,
        };

        let parameters = Parameters {
            file_name: None,
            gizmo_mode: GizmoMode::Translate,
            cell_count: [16, 16, 16],
            render_mode: RenderMode::Sdf,
            enable_shadows: false, // deactivating shadows for now.
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

    /// Draw ui with egui.
    #[allow(clippy::unused_self)]
    pub fn draw_gizmos(&mut self, _ui: &mut egui::Ui) {
        // TODO: backport transform.
    }

    /// Draw ui with egui.
    pub fn draw_ui(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, ui: &mut egui::Ui) {
        egui::Grid::new("settings").show(ui, |ui| {
            ui.separator();
            ui.end_row();

            ui.label("Render Mode");
            egui::ComboBox::from_label("")
                .selected_text(format!("{:?}", self.parameters.render_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.parameters.render_mode,
                        RenderMode::Model,
                        "Model",
                    );
                    ui.selectable_value(&mut self.parameters.render_mode, RenderMode::Sdf, "SDF");
                    ui.selectable_value(
                        &mut self.parameters.render_mode,
                        RenderMode::ModelAndSdf,
                        "Model and SDF",
                    );
                });
            ui.end_row();

            ui.separator();
            ui.end_row();

            match self.parameters.file_name {
                None => {
                    ui.label("No file loaded");
                }
                Some(ref file_name) => {
                    ui.label("Current file");
                    ui.label(
                        file_name
                            .rsplit_once('\\')
                            .map_or(file_name.as_str(), |(_, name)| name),
                    );
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

            for label in ["Positive Color", "Negative Color", "Surface Color"] {
                let color = match label {
                    "Positive Color" => self.settings.settings.positive_color,
                    "Negative Color" => self.settings.settings.negative_color,
                    "Surface Color" => self.settings.settings.surface_color,
                    _ => continue,
                };

                if let Some(new_color) = Self::add_color_widget(ui, label, color) {
                    // Save old state.
                    let old_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    println!("diffenrence: {} {:?} {:?}", label, color, new_color);

                    match label {
                        "Positive Color" => self.settings.settings.positive_color = new_color,
                        "Negative Color" => self.settings.settings.negative_color = new_color,
                        "Surface Color" => self.settings.settings.surface_color = new_color,
                        _ => continue,
                    };

                    // Get new state.
                    let new_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    // Push with the old and new state.
                    self.command_stack.push(
                        label,
                        command_stack::Command {
                            old_state,
                            new_state,
                        },
                    );
                }
                ui.end_row();
            }

            for (label, range) in [
                ("Positive power", 0.0..=10.0),
                ("Negative power", 0.0..=10.0),
                ("Surface power", 0.0..=10.0),
                ("Surface width", 0.001..=0.1),
                ("Point size", 0.1..=1.),
            ] {
                let value = match label {
                    "Positive power" => self.settings.settings.positives_power,
                    "Negative power" => self.settings.settings.negatives_power,
                    "Surface power" => self.settings.settings.surface_power,
                    "Surface width" => self.settings.settings.surface_width,
                    "Point size" => self.settings.settings.point_size,
                    _ => continue,
                };

                let mut new_value = value;
                ui.label(label);
                ui.add(egui::Slider::new(&mut new_value, range));

                if !float_cmp::approx_eq!(f32, new_value, value, ulps = 2, epsilon = 1e-6) {
                    // Save old state.
                    let old_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    match label {
                        "Positive power" => self.settings.settings.positives_power = new_value,
                        "Negative power" => self.settings.settings.negatives_power = new_value,
                        "Surface power" => self.settings.settings.surface_power = new_value,
                        "Surface width" => self.settings.settings.surface_width = new_value,
                        "Point size" => self.settings.settings.point_size = new_value,
                        _ => continue,
                    };

                    // Get new state.
                    let new_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    // Push with the old and new state.
                    self.command_stack.push(
                        label,
                        command_stack::Command {
                            old_state,
                            new_state,
                        },
                    );
                }
                ui.end_row();
            }

            ui.separator();
            ui.end_row();

            ui.label("Cell count");
            ui.horizontal(|ui| {
                let mut new_value = self.parameters.cell_count;
                ui.add(
                    egui::DragValue::new(&mut new_value[0])
                        .clamp_range(2..=100)
                        .prefix("x: "),
                );
                ui.add(
                    egui::DragValue::new(&mut new_value[1])
                        .clamp_range(2..=100)
                        .prefix("y: "),
                );
                ui.add(
                    egui::DragValue::new(&mut new_value[2])
                        .clamp_range(2..=100)
                        .prefix("z: "),
                );

                if new_value != self.parameters.cell_count {
                    // Save old state.
                    let old_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    self.parameters.cell_count = new_value;

                    // Get new state.
                    let new_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    // Push with the old and new state.
                    self.command_stack.push(
                        "Cell size",
                        command_stack::Command {
                            old_state,
                            new_state,
                        },
                    );
                }
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

                let size_x = (x_max - x_min) / self.parameters.cell_count[0] as f32;
                let size_y = (y_max - y_min) / self.parameters.cell_count[1] as f32;
                let size_z = (z_max - z_min) / self.parameters.cell_count[2] as f32;

                let value = [size_x, size_y, size_z];
                let mut new_value = value;

                ui.add(
                    egui::DragValue::new(&mut new_value[0])
                        .prefix("x: ")
                        .speed(0.001)
                        .max_decimals(3),
                );

                ui.add(
                    egui::DragValue::new(&mut new_value[1])
                        .prefix("y: ")
                        .speed(0.001)
                        .max_decimals(3),
                );

                ui.add(
                    egui::DragValue::new(&mut new_value[2])
                        .prefix("z: ")
                        .speed(0.001)
                        .max_decimals(3),
                );

                if !float_cmp::approx_eq!(f32, value[0], new_value[0], ulps = 2, epsilon = 1e-6)
                    || !float_cmp::approx_eq!(f32, value[1], new_value[1], ulps = 2, epsilon = 1e-6)
                    || !float_cmp::approx_eq!(f32, value[2], new_value[2], ulps = 2, epsilon = 1e-6)
                {
                    // Save old state.
                    let old_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    self.parameters.cell_count[0] = ((x_max - x_min) / new_value[0]) as u32;
                    self.parameters.cell_count[1] = ((y_max - y_min) / new_value[1]) as u32;
                    self.parameters.cell_count[2] = ((z_max - z_min) / new_value[2]) as u32;

                    // Get new state.
                    let new_state = command_stack::State {
                        parameters: self.parameters.clone(),
                        settings: self.settings.settings,
                    };

                    // Push with the old and new state.
                    self.command_stack.push(
                        "Cell size",
                        command_stack::Command {
                            old_state,
                            new_state,
                        },
                    );
                }
            });
            ui.end_row();

            ui.separator();
            ui.end_row();

            if self.parameters.enable_shadows {
                ui.label("Light");
                ui.end_row();
                ui.add(
                    egui::Slider::new(
                        &mut self.pass.shadow.map.light.camera.look_at.distance,
                        0.0..=30.0,
                    )
                    .text("Distance"),
                );
                ui.end_row();
                ui.add(
                    egui::Slider::new(
                        &mut self.pass.shadow.map.light.camera.look_at.longitude,
                        0.0..=std::f32::consts::TAU,
                    )
                    .text("Longitude"),
                );
                ui.end_row();
                ui.add(
                    egui::Slider::new(
                        &mut self.pass.shadow.map.light.camera.look_at.latitude,
                        -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
                    )
                    .text("Latitude"),
                );
                ui.end_row();

                ui.separator();
                ui.end_row();
            }

            if ui.button("Generate").clicked() {
                if self.load_gltf(device, queue).is_err() {
                    self.alert_message = Some((
                        "Failed to load file. Make sure it is a valid gltf file.".to_owned(),
                        web_time::Instant::now(),
                    ));
                }
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
        // TODO: we don't need to load the gltf file each time we generate the sdf.
        match self.parameters.file_name {
            None => anyhow::bail!("No file to load"),
            Some(ref path) => {
                let start = web_time::Instant::now();

                // a gltf scene can contain multiple models.
                // we merge them in a single sdf.
                self.models = gltf::load_scene(device, queue, path)?;

                let (vertices, indices) = self.models.iter().fold(
                    (vec![], vec![]),
                    |(mut vertices, mut indices), model| {
                        // we need to offset the indices by the number of vertices we already have.
                        let len = vertices.len();
                        vertices.extend(model.vertices.iter().map(|v| v.position));
                        indices.extend(model.indices.iter().map(|i| *i + len as u32));
                        (vertices, indices)
                    },
                );

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
                self.model_info = Some(model_info);

                // Adapt surface width to the size of the model.
                self.settings.settings.surface_width =
                    (xmax - xmin).max(ymax - ymin).max(zmax - zmin) / 100.0;

                let start_cell = [xmin, ymin, zmin];
                let end_cell = [xmax, ymax, zmax];

                self.sdf = Sdf::new(
                    device,
                    &vertices,
                    &indices,
                    &start_cell,
                    &end_cell,
                    &self.parameters.cell_count,
                )
                .ok();

                self.last_run_info = Some(LastRunInfo {
                    time: start.elapsed().as_secs_f32() * 1000.0,
                    size: self.parameters.cell_count,
                });

                Ok(())
            }
        }
    }
}
