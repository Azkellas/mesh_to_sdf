//! Client application for the `mesh_to_sdf` project.
//!
//! ![example](https://raw.githubusercontent.com/Azkellas/mesh_to_sdf/main/client.gif)
//!
//! A visualization client for the SDF of a mesh.
//! Generates a SDF from a mesh and renders it.
mod camera;
mod camera_control;
mod cubemap;
mod frame_rate;
mod gltf;
mod passes;
mod pbr;
mod reload_flags;
mod runner;
mod sdf;
mod sdf_program;
mod texture;
mod utility;

use std::sync::{Arc, Mutex};

#[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};

#[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
use std::path::Path;

/// App entry point.
fn main() {
    let data = Arc::new(Mutex::new(crate::reload_flags::ReloadFlags {
        shaders: vec![],
    }));

    // Watch shaders folder.
    // When a shader is saved, the pipelines will be recreated.
    // Only enabled in native debug mode.
    #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
    {
        let data = Arc::clone(&data);
        std::thread::spawn(move || {
            // first try to watch the mesh_to_sdf_client/shaders folder (runnin from the root of the project)
            // if that fails, try to watch the shaders folder (running from the client folder)
            let paths = ["mesh_to_sdf_client/shaders", "shaders"];
            for path in paths {
                log::info!("Watching {path}");
                if let Err(error) = watch(path, &data) {
                    log::error!("Could not watch shaders folder: {error:?}");
                }
            }
        });
    }

    runner::start_app(data);
}

/// Watch shader folder. Only done in native debug mode.
/// Everytime a shader is modified/added/deleted,
/// it will update the `ReloadFlags` so the program can reload them.
#[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
fn watch<P: AsRef<Path>>(
    path: P,
    data: &Arc<Mutex<crate::reload_flags::ReloadFlags>>,
) -> notify::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();

    // Automatically select the best implementation for your platform.
    // You can also access each implementation directly e.g. INotifyWatcher.
    let mut watcher = RecommendedWatcher::new(tx, Config::default())?;

    // Add a path to be watched. All files and directories at that path and
    // below will be monitored for changes.
    watcher.watch(path.as_ref(), RecursiveMode::Recursive)?;

    for res in rx {
        match res {
            Ok(event) => {
                log::info!("Change: {:?}", event.paths);
                let mut data = data.lock().unwrap();
                event.paths.iter().for_each(|p| {
                    let shader_path = p.to_str().unwrap().to_owned();
                    data.shaders.push(shader_path);
                });
            }
            Err(error) => log::error!("Error: {error:?}"),
        }
    }

    Ok(())
}
