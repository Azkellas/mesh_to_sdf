use egui_wgpu::renderer::{Renderer, ScreenDescriptor};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use winit::{
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use winit_input_helper::WinitInputHelper;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;

use crate::sdf_program::SdfProgram;

/// Initialize wgpu and run the app.
async fn run(
    event_loop: EventLoop<()>,
    window: Rc<Window>,
    data: Arc<Mutex<crate::reload_flags::ReloadFlags>>,
) {
    let mut input = WinitInputHelper::new();

    // Create the instance and surface.
    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        dx12_shader_compiler,
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::default(),
    });

    let surface =
        unsafe { instance.create_surface(window.as_ref()) }.expect("Could not create surface");

    // Select an adapter and a surface configuration.
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .await
        .expect("No suitable GPU adapters found on the system!");

    let optional_features = SdfProgram::optional_features();
    let required_features = SdfProgram::required_features();
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features for this example: {:?}",
        required_features - adapter_features
    );

    let required_downlevel_capabilities = SdfProgram::required_downlevel_capabilities();
    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    assert!(
        downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
        "Adapter does not support the minimum shader model required to run this example: {:?}",
        required_downlevel_capabilities.shader_model
    );
    assert!(
        downlevel_capabilities
            .flags
            .contains(required_downlevel_capabilities.flags),
        "Adapter does not support the downlevel capabilities required to run this example: {:?}",
        required_downlevel_capabilities.flags - downlevel_capabilities.flags
    );

    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
    let needed_limits = SdfProgram::required_limits().using_resolution(adapter.limits());

    let trace_dir = std::env::var("WGPU_TRACE");
    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: (optional_features & adapter_features) | required_features,
                limits: needed_limits,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    // Configure surface.
    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .expect("Surface isn't supported by the adapter.");

    // Comment to disable freerun and enable v-sync. Note that this is only valid in native.
    #[cfg(not(target_arch = "wasm32"))]
    {
        config.present_mode = wgpu::PresentMode::Immediate;
    }

    surface.configure(&device, &config);

    // Create our program.
    let mut program =
        SdfProgram::init(&surface, &device, &adapter, &queue).expect("Failed to create program");

    // Create egui state.
    let mut egui_state = egui_winit::State::new(
        egui::Context::default(),
        egui::ViewportId::default(),
        &event_loop,
        None,
        None,
    );

    let mut egui_renderer = Renderer::new(&device, config.format, None, 1);

    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent {
                event: ref window_event,
                ..
            } = &event
            {
                // ignore event response.
                let _ = egui_state.on_window_event(&window, window_event);
            }

            if input.update(&event) {
                // Have the closure take ownership of the resources.
                // `event_loop.run` never returns, therefore we must do this to ensure
                // the resources are properly cleaned up.
                let _ = (&instance, &adapter, &program, &egui_renderer, &egui_state);

                // Poll all events to ensure a maximum framerate.
                target.set_control_flow(ControlFlow::Poll);

                //TODO: this needs a huge refacto.

                if input.close_requested() {
                    target.exit();
                }
                if let Some(new_size) = input.window_resized() {
                    // Resize with 0 width and height is used by winit to signal a minimize event on Windows.
                    // See: https://github.com/rust-windowing/winit/issues/208
                    // This solves an issue where the app would panic when minimizing on Windows.
                    if new_size.width > 0 && new_size.height > 0 {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        surface.configure(&device, &config);
                        program.resize(&config, &device, &queue);
                    }
                }

                program.process_input(&input);

                if let Some(camera) = program.get_camera() {
                    camera.update(&input, [size.width as f32, size.height as f32]);
                };

                window.request_redraw();
                let mut data = data.lock().unwrap();
                // Reload shaders if needed
                if !data.shaders.is_empty() {
                    log::info!("rebuild shaders {:?}", data.shaders);
                    if let Err(program_error) = program.update_passes(&surface, &device, &adapter) {
                        log::error!("{program_error:?}");
                    }
                    data.shaders.clear();
                }

                // Rebuild render pipeline if needed
                if data.lib == crate::reload_flags::LibState::Reloaded {
                    log::info!("reload lib");
                    if let Err(program_error) = program.update_passes(&surface, &device, &adapter) {
                        log::error!("{program_error}");
                    }
                    data.lib = crate::reload_flags::LibState::Stable;
                }

                // Render a frame if the lib is stable.
                if data.lib == crate::reload_flags::LibState::Stable {
                    // Get the next frame and view.
                    let texture = surface.get_current_texture();
                    let frame = match texture {
                        Ok(f) => f,
                        Err(e) => {
                            log::warn!("surface lost: window is probably minimized: {e}");
                            return;
                        }
                    };

                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // Update the program before drawing.
                    program.update(&queue);

                    // Render the program first so the ui is on top.
                    program.render(&view, &device, &queue);

                    // Update the ui before drawing.
                    let input = egui_state.take_egui_input(&window);

                    let egui_context = egui_state.egui_ctx();

                    // let pixels_per_point = egui_context.pixels_per_point();
                    // if pixels_per_point != 1.0 {
                    //     egui_context.set_zoom_factor(1.0 / pixels_per_point);
                    // }

                    egui_context.begin_frame(input);

                    egui::panel::SidePanel::new(
                        egui::panel::Side::Left,
                        egui::Id::new("control_panel"),
                    )
                    .default_width(size.width as f32 * 0.15)
                    .show(egui_context, |ui| {
                        program.draw_ui(&device, &queue, ui);
                    });

                    egui::Area::new("Viewport")
                        .fixed_pos((0.0, 0.0))
                        .show(egui_context, |ui| {
                            ui.with_layer_id(egui::LayerId::background(), |ui| {
                                program.draw_gizmos(ui);
                            })
                        });

                    let output = egui_context.end_frame();
                    let paint_jobs =
                        egui_context.tessellate(output.shapes, egui_context.pixels_per_point());
                    let screen_descriptor = ScreenDescriptor {
                        size_in_pixels: [config.width, config.height],
                        pixels_per_point: egui_context.pixels_per_point(),
                    };

                    // Create a command encoder.
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    // Update the egui renderer.
                    {
                        for (id, image_delta) in &output.textures_delta.set {
                            egui_renderer.update_texture(&device, &queue, *id, image_delta);
                        }
                        for id in &output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }

                        {
                            egui_renderer.update_buffers(
                                &device,
                                &queue,
                                &mut encoder,
                                &paint_jobs,
                                &screen_descriptor,
                            );
                        }
                    }

                    // Render ui.
                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });

                        egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
                    }

                    // Present the frame.
                    queue.submit(Some(encoder.finish()));
                    frame.present();
                }
            }
        })
        .unwrap();
}

/// Create the window depending on the platform.
pub fn start_app(data: Arc<Mutex<crate::reload_flags::ReloadFlags>>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        let event_loop = EventLoop::new().unwrap();
        let builder = winit::window::WindowBuilder::new()
            .with_title(SdfProgram::get_name())
            .with_maximized(true);
        let window = Rc::new(builder.build(&event_loop).unwrap());

        pollster::block_on(run(event_loop, window, data));
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Initialize logging.
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");

        // // Create event_loop and window.
        let event_loop = EventLoop::new().expect("Could not create event loop");
        let window =
            Rc::new(winit::window::Window::new(&event_loop).expect("Could not create window"));

        // Add canvas to document body.
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(
                    window.canvas().expect("Could not find canvas"),
                ))
                .ok()
            })
            .expect("couldn't append canvas to document body");

        // start the app.
        wasm_bindgen_futures::spawn_local(run(event_loop, window, data));
    }
}
