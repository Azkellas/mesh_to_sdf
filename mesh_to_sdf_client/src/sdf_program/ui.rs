use super::*;

impl SdfProgram {
    fn add_color_widget(ui: &mut egui::Ui, label: &str, color: [f32; 3]) -> Option<[f32; 3]> {
        ui.label(label);
        let mut new_color = color;
        egui::color_picker::color_edit_button_rgb(ui, &mut new_color);

        if !float_cmp::approx_eq!(f32, color[0], new_color[0], ulps = 2, epsilon = 1e-6)
            || !float_cmp::approx_eq!(f32, color[1], new_color[1], ulps = 2, epsilon = 1e-6)
            || !float_cmp::approx_eq!(f32, color[2], new_color[2], ulps = 2, epsilon = 1e-6)
        {
            Some(new_color)
        } else {
            None
        }
    }

    fn end_category(ui: &mut egui::Ui) {
        ui.separator();
        ui.end_row();
    }

    /// Draw ui with egui.
    #[allow(clippy::unused_self)]
    pub fn draw_gizmos(&mut self, _ui: &mut egui::Ui) {
        // TODO: backport transform.
    }

    /// Draw ui with egui.
    pub fn draw_ui(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, ui: &mut egui::Ui) {
        egui::Grid::new("settings").show(ui, |ui| {
            Self::end_category(ui);

            #[cfg(debug_assertions)]
            {
                ui.label(std::format!("Frame rate: {:.2}", self.frame_rate.get()));
                ui.separator();
                ui.end_row();
            }

            self.ui_render_mode(ui);

            Self::end_category(ui);

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

            self.ui_model_info(ui);

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

                    // TODO: fix this.
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

            Self::end_category(ui);

            if self.parameters.render_mode == RenderMode::Sdf
                || self.parameters.render_mode == RenderMode::ModelAndSdf
            {
                self.ui_colors(ui);
                Self::end_category(ui);

                self.ui_powers(ui);
                Self::end_category(ui);
            }

            if self.parameters.render_mode == RenderMode::Voxels {
                self.ui_surface(ui);
                Self::end_category(ui);
            }

            self.ui_cells(ui);
            Self::end_category(ui);

            self.ui_light(ui);

            Self::end_category(ui);

            if ui.button("Generate").clicked() {
                #[allow(clippy::collapsible_if)]
                if self.generate_sdf(device).is_err() {
                    self.alert_message = Some((
                        "Failed to generate sdf.".to_owned(),
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

    fn ui_render_mode(&mut self, ui: &mut egui::Ui) {
        ui.label("Render Mode");
        egui::ComboBox::from_id_source("render_mode")
            .selected_text(format!("{:?}", self.parameters.render_mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.parameters.render_mode, RenderMode::Model, "Model");
                ui.selectable_value(&mut self.parameters.render_mode, RenderMode::Sdf, "SDF");
                ui.selectable_value(
                    &mut self.parameters.render_mode,
                    RenderMode::ModelAndSdf,
                    "Model and SDF",
                );
                ui.selectable_value(
                    &mut self.parameters.render_mode,
                    RenderMode::Voxels,
                    "Voxels",
                );
                ui.selectable_value(
                    &mut self.parameters.render_mode,
                    RenderMode::Raymarch,
                    "Raymarch",
                );
            });
        ui.end_row();

        if self.parameters.render_mode == RenderMode::Raymarch {
            ui.label("Raymarch Mode");
            egui::ComboBox::from_id_source("raymarch_mode")
                .selected_text(format!(
                    "{:?}",
                    RaymarchMode::try_from(self.settings.settings.raymarch_mode).unwrap()
                ))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.settings.settings.raymarch_mode,
                        RaymarchMode::Snap as _,
                        "Snap",
                    );
                    ui.selectable_value(
                        &mut self.settings.settings.raymarch_mode,
                        RaymarchMode::Trilinear as _,
                        "Trilinear",
                    );
                    ui.selectable_value(
                        &mut self.settings.settings.raymarch_mode,
                        RaymarchMode::Tetrahedral as _,
                        "Tetrahedral",
                    );
                });
            ui.end_row();
        }
    }

    fn ui_model_info(&mut self, ui: &mut egui::Ui) {
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
                model_info.bounding_box[0], model_info.bounding_box[1], model_info.bounding_box[2]
            ));
            ui.end_row();

            ui.label("Bounding box max");
            ui.label(format!(
                "x: {:.2} y: {:.2} z: {:.2}",
                model_info.bounding_box[3], model_info.bounding_box[4], model_info.bounding_box[4]
            ));
            ui.end_row();
        }
    }

    fn ui_colors(&mut self, ui: &mut egui::Ui) {
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
    }

    fn ui_powers(&mut self, ui: &mut egui::Ui) {
        for (label, range) in [
            ("Positive power", 0.0..=10.0),
            ("Negative power", 0.0..=10.0),
            ("Surface power", 0.0..=10.0),
            ("Surface width", 0.0001..=0.1),
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
    }

    fn ui_surface(&mut self, ui: &mut egui::Ui) {
        let value = self.settings.settings.surface_width;

        let mut new_value = value;
        ui.label("Surface width");
        ui.add(egui::Slider::new(&mut new_value, 0.0005..=0.5));

        if !float_cmp::approx_eq!(f32, new_value, value, ulps = 2, epsilon = 1e-6) {
            // Save old state.
            let old_state = command_stack::State {
                parameters: self.parameters.clone(),
                settings: self.settings.settings,
            };

            self.settings.settings.surface_width = new_value;

            // Get new state.
            let new_state = command_stack::State {
                parameters: self.parameters.clone(),
                settings: self.settings.settings,
            };

            // Push with the old and new state.
            self.command_stack.push(
                "Surface width",
                command_stack::Command {
                    old_state,
                    new_state,
                },
            );
        }
        ui.end_row();
    }

    fn ui_cells(&mut self, ui: &mut egui::Ui) {
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

        {
            ui.label("Bounding Box Extent");
            let mut new_value = self.parameters.bounding_box_extent;
            ui.add(egui::Slider::new(&mut new_value, 1.0..=3.0));

            if new_value != self.parameters.bounding_box_extent {
                // Save old state.
                let old_state = command_stack::State {
                    parameters: self.parameters.clone(),
                    settings: self.settings.settings,
                };

                self.parameters.bounding_box_extent = new_value;

                // Get new state.
                let new_state = command_stack::State {
                    parameters: self.parameters.clone(),
                    settings: self.settings.settings,
                };

                // Push with the old and new state.
                self.command_stack.push(
                    "Bounding Box Extent",
                    command_stack::Command {
                        old_state,
                        new_state,
                    },
                );
            }
        }
        ui.end_row();
    }

    fn ui_light(&mut self, ui: &mut egui::Ui) {
        ui.label("Light");
        ui.end_row();

        ui.label("Distance");
        ui.add(egui::Slider::new(
            &mut self.pass.shadow.map.light.camera.look_at.distance,
            0.0..=30.0,
        ));
        ui.end_row();

        ui.label("Longitude");
        ui.add(egui::Slider::new(
            &mut self.pass.shadow.map.light.camera.look_at.longitude,
            0.0..=std::f32::consts::TAU,
        ));
        ui.end_row();

        ui.label("Latitude");
        ui.add(egui::Slider::new(
            &mut self.pass.shadow.map.light.camera.look_at.latitude,
            -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
        ));
        ui.end_row();
    }
}
