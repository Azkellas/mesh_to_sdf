## Mesh to SDF Client

This README contains references for developpers. For user documentation, see the [repo README](../README.md).

---

The program will hot reload shaders in debug native mode. In release mode, they are bundled in the executable.
While the project started with[rust_wgpu_hot_reload](https://github.com/Azkellas/rust_wgpu_hot_reload/) as template, the rust hot reload code has been removed to keep it in a single package.

---

The project is a bit of a mess currently as it was exported from another. Here's a breakdown of the architecture until it gets cleaned up:
- `main`: Watch `shaders` folder in debug mode and start `runner``
- `runner`: Initializes winit, egui, and the renderer, run the event loop and calls `sdf_program` when needed
- `sdf_program`: Contains the logic the program. Draws ui, renders the sdf, etc.
- `sdf`: Call `mesh_to_sdf` to generate the sdf and stores it in a `Sdf` struct. Contains the uniforms for the render pass
- other files should be small and self explanatory