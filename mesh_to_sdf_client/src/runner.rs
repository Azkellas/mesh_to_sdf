use egui_wgpu::renderer::{Renderer, ScreenDescriptor};
use std::sync::{Arc, Mutex};
use winit::dpi::PhysicalSize;
use winit::event::StartCause;
use winit::{
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use winit_input_helper::WinitInputHelper;

use crate::sdf_program::SdfProgram;

struct EventLoopWrapper {
    event_loop: EventLoop<()>,
    window: Arc<Window>,
}

impl EventLoopWrapper {
    pub fn new(title: &str) -> Self {
        let event_loop = EventLoop::new().unwrap();
        let mut builder = winit::window::WindowBuilder::new();
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowBuilderExtWebSys;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            builder = builder.with_canvas(Some(canvas));
        }
        builder = builder.with_title(title);

        #[cfg(not(target_arch = "wasm32"))]
        {
            builder = builder.with_maximized(true);
        }

        let window = Arc::new(builder.build(&event_loop).unwrap());

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;

            let get_full_size = || {
                // TODO Not sure how to get scrollbar dims
                let scrollbars = 4.0;
                let win = web_sys::window().unwrap();
                // `inner_width` corresponds to the browser's `self.innerWidth` function, which are in
                // Logical, not Physical, pixels
                winit::dpi::LogicalSize::new(
                    win.inner_width().unwrap().as_f64().unwrap() - scrollbars,
                    win.inner_height().unwrap().as_f64().unwrap() - scrollbars,
                )
            };

            let size = get_full_size();
            let _ = window.request_inner_size(size);

            let websys_window = web_sys::window().unwrap();
            let window = window.clone();
            let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move |_: web_sys::Event| {
                let size = get_full_size();
                let _ = window.request_inner_size(size);
            }) as Box<dyn FnMut(_)>);
            websys_window
                .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
                .unwrap();
            closure.forget();
        }

        Self { event_loop, window }
    }
}

/// Wrapper type which manages the surface and surface configuration.
///
/// As surface usage varies per platform, wrapping this up cleans up the event loop code.
struct SurfaceWrapper {
    surface: Option<wgpu::Surface>,
    config: Option<wgpu::SurfaceConfiguration>,
}

impl SurfaceWrapper {
    /// Create a new surface wrapper with no surface or configuration.
    fn new() -> Self {
        Self {
            surface: None,
            config: None,
        }
    }

    /// Called after the instance is created, but before we request an adapter.
    ///
    /// On wasm, we need to create the surface here, as the WebGL backend needs
    /// a surface (and hence a canvas) to be present to create the adapter.
    ///
    /// We cannot unconditionally create a surface here, as Android requires
    /// us to wait until we recieve the `Resumed` event to do so.
    fn pre_adapter(&mut self, instance: &wgpu::Instance, window: Arc<Window>) {
        if cfg!(target_arch = "wasm32") {
            unsafe {
                self.surface = Some(instance.create_surface(&window).unwrap());
            }
        }
    }

    /// Check if the event is the start condition for the surface.
    fn start_condition(event: &Event<()>) -> bool {
        event == &Event::NewEvents(StartCause::Init)
    }

    /// Called when an event which matches [`Self::start_condition`] is recieved.
    ///
    /// On all native platforms, this is where we create the surface.
    ///
    /// Additionally, we configure the surface based on the (now valid) window size.
    fn resume(&mut self, context: &ExampleContext, window: Arc<Window>, srgb: bool) {
        // Window size is only actually valid after we enter the event loop.
        let window_size = window.inner_size();
        let width = window_size.width.max(1);
        let height = window_size.height.max(1);

        log::info!("Surface resume {window_size:?}");

        // We didn't create the surface in pre_adapter, so we need to do so now.
        if !cfg!(target_arch = "wasm32") {
            unsafe {
                self.surface = Some(context.instance.create_surface(&window).unwrap());
            }
        }

        // From here on, self.surface should be Some.

        let surface = self.surface.as_ref().unwrap();

        // Get the default configuration,
        let mut config = surface
            .get_default_config(&context.adapter, width, height)
            .expect("Surface isn't supported by the adapter.");
        if srgb {
            // Not all platforms (WebGPU) support sRGB swapchains, so we need to use view formats
            let view_format = config.format.add_srgb_suffix();
            config.view_formats.push(view_format);
        } else {
            // All platforms support non-sRGB swapchains, so we can just use the format directly.
            let format = config.format.remove_srgb_suffix();
            config.format = format;
            config.view_formats.push(format);
        };

        surface.configure(&context.device, &config);
        self.config = Some(config);
    }

    /// Resize the surface, making sure to not resize to zero.
    fn resize(&mut self, context: &ExampleContext, size: PhysicalSize<u32>) {
        log::info!("Surface resize {size:?}");

        let config = self.config.as_mut().unwrap();
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        let surface = self.surface.as_ref().unwrap();
        surface.configure(&context.device, config);
    }

    /// Acquire the next surface texture.
    fn acquire(&mut self, context: &ExampleContext) -> wgpu::SurfaceTexture {
        let surface = self.surface.as_ref().unwrap();

        match surface.get_current_texture() {
            Ok(frame) => frame,
            // If we timed out, just try again
            Err(wgpu::SurfaceError::Timeout) => surface
                .get_current_texture()
                .expect("Failed to acquire next surface texture!"),
            Err(
                // If the surface is outdated, or was lost, reconfigure it.
                wgpu::SurfaceError::Outdated
                | wgpu::SurfaceError::Lost
                // If OutOfMemory happens, reconfiguring may not help, but we might as well try
                | wgpu::SurfaceError::OutOfMemory,
            ) => {
                surface.configure(&context.device, self.config());
                surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        }
    }

    /// On suspend on android, we drop the surface, as it's no longer valid.
    ///
    /// A suspend event is always followed by at least one resume event.
    fn suspend(&mut self) {
        self.surface = None;
    }

    fn get(&self) -> Option<&wgpu::Surface> {
        self.surface.as_ref()
    }

    fn config(&self) -> &wgpu::SurfaceConfiguration {
        self.config.as_ref().unwrap()
    }
}

/// Context containing global wgpu resources.
struct ExampleContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl ExampleContext {
    /// Initializes the example context.
    async fn init_async(surface: &mut SurfaceWrapper, window: Arc<Window>) -> Self {
        log::info!("Initializing wgpu...");

        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all());
        let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
        let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler,
            gles_minor_version,
        });
        log::info!("Created instance: {:?}", instance);

        surface.pre_adapter(&instance, window);
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, surface.get())
            .await
            .expect("No suitable GPU adapters found on the system!");

        let adapter_info = adapter.get_info();
        log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

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
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device Descriptor"),
                    features: (optional_features & adapter_features) | required_features,
                    limits: needed_limits,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }
}

/// Initialize wgpu and run the app.
async fn run(
    // event_loop: EventLoop<()>,
    // window: Rc<Window>,
    data: Arc<Mutex<crate::reload_flags::ReloadFlags>>,
) {
    let window_loop = EventLoopWrapper::new(SdfProgram::get_name());
    let mut surface = SurfaceWrapper::new();
    let context = ExampleContext::init_async(&mut surface, window_loop.window.clone()).await;

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::EventLoopExtWebSys;
            let event_loop_function = EventLoop::spawn;
        } else {
            let event_loop_function = EventLoop::run;
        }
    }

    let mut input = WinitInputHelper::new();
    let mut program = None;

    // Create egui state.
    let mut egui_state = egui_winit::State::new(
        egui::Context::default(),
        egui::ViewportId::default(),
        &window_loop.event_loop,
        None,
        None,
    );

    let mut egui_renderer: Option<Renderer> = None;

    let size = window_loop.window.inner_size();

    #[allow(clippy::let_unit_value)]
    let _ = (event_loop_function)(
        window_loop.event_loop,
        move |event: Event<()>, target: &winit::event_loop::EventLoopWindowTarget<()>| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            // let _ = (&instance, &adapter, &runner_data, &egui_state);

            if let Event::WindowEvent {
                event: ref window_event,
                ..
            } = &event
            {
                // ignore event response.
                let _ = egui_state.on_window_event(&window_loop.window, window_event);

                if window_event == &winit::event::WindowEvent::CloseRequested {
                    target.exit();
                }
            }

            if SurfaceWrapper::start_condition(&event) {
                surface.resume(&context, window_loop.window.clone(), true);

                if program.is_none() {
                    program = Some(
                        SdfProgram::init(
                            surface.surface.as_ref().unwrap(),
                            &context.device,
                            &context.adapter,
                            &context.queue,
                        )
                        .unwrap(),
                    );
                }

                if egui_renderer.is_none() {
                    egui_renderer = Some(Renderer::new(
                        &context.device,
                        surface.config.as_ref().unwrap().format,
                        None,
                        1,
                    ));
                }
            }

            if event == Event::Suspended {
                surface.suspend();
            }

            if input.update(&event) {
                let Some(program) = &mut program else {
                    return;
                };
                let Some(config) = surface.config.as_mut() else {
                    return;
                };
                let Some(surface) = surface.surface.as_ref() else {
                    return;
                };
                let Some(egui_renderer) = egui_renderer.as_mut() else {
                    return;
                };

                // Poll all events to ensure a maximum framerate.
                target.set_control_flow(ControlFlow::Poll);

                //TODO: this needs a huge refacto.

                if input.close_requested() {
                    target.exit();
                }
                if let Some(new_size) = input.window_resized() {
                    log::info!("Window resized: {:?}", new_size);
                    // Resize with 0 width and height is used by winit to signal a minimize event on Windows.
                    // See: https://github.com/rust-windowing/winit/issues/208
                    // This solves an issue where the app would panic when minimizing on Windows.
                    if new_size.width > 0 && new_size.height > 0 {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        surface.configure(&context.device, config);
                        program.resize(config, &context.device, &context.queue);
                    }
                }

                program.process_input(&input);

                if let Some(camera) = program.get_camera() {
                    camera.update(&input, [size.width as f32, size.height as f32]);
                };

                // window_loop.window.request_redraw();

                let mut data = data.lock().unwrap();
                // Reload shaders if needed
                if !data.shaders.is_empty() {
                    log::info!("rebuild shaders {:?}", data.shaders);
                    if let Err(program_error) =
                        program.update_passes(surface, &context.device, &context.adapter)
                    {
                        log::error!("{program_error:?}");
                    }
                    data.shaders.clear();
                }

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
                program.update(&context.queue);

                // Render the program first so the ui is on top.
                program.render(&view, &context.device, &context.queue);

                // Update the ui before drawing.
                let input = egui_state.take_egui_input(&window_loop.window);

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
                    program.draw_ui(&context.device, &context.queue, ui);
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
                let mut encoder = context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // Update the egui renderer.
                {
                    for (id, image_delta) in &output.textures_delta.set {
                        egui_renderer.update_texture(
                            &context.device,
                            &context.queue,
                            *id,
                            image_delta,
                        );
                    }
                    for id in &output.textures_delta.free {
                        egui_renderer.free_texture(id);
                    }

                    {
                        egui_renderer.update_buffers(
                            &context.device,
                            &context.queue,
                            &mut encoder,
                            &paint_jobs,
                            &screen_descriptor,
                        );
                    }
                }

                // Render ui.
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui render pass"),
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
                context.queue.submit(Some(encoder.finish()));
                frame.present();
            }
        },
    );
}

/// Create the window depending on the platform.
pub fn start_app(data: Arc<Mutex<crate::reload_flags::ReloadFlags>>) {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("could not initialize logger");
            wasm_bindgen_futures::spawn_local(async move { run(data).await })
        } else {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
            pollster::block_on(run(data));
        }
    }
}
