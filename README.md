# Mesh to SDF

https://github.com/Azkellas/mesh_to_sdf/assets/29731210/9d87350c-f491-4de1-aa34-0922c8739e7a

This repository contains two crates:
- `mesh_to_sdf`: a library to convert a mesh to a signed distance field (SDF).
- `mesh_to_sdf_client`: a visualization client for the SDF of a mesh.

---

## `mesh_to_sdf`: Convert a mesh to a signed distance field (SDF).

⚠️ This crate is still in its early stages. Expect the API to change.

---

This crate provides two entry points:

- `mesh_to_sdf::generate_sdf(vertices, indices, query_points)` returns the signed distance field for the mesh defined by `vertices` and `indices` at the points `query_points`.
- `mesh_to_sdf::generate_grid_sdf(vertices, indices, start_pos, cell_radius, cell_count)` returns the signed distance field for the mesh defined by `vertices` and `indices` on a grid with `cell_count` cells of size `cell_radius` starting at `start_pos`.

```rust
let vertices: Vec<[f32; 3]> = ...
let indices: Vec<u32> = ...

let query_points: Vec<[f32; 3]> = ...

// Query points are expected to be in the same space as the mesh.
let sdf: Vec<f32> = mesh_to_sdf::generate_sdf(&vertices, &indices, &query_points);
for point in query_points.iter().zip(sdf.iter()) {
    println!("Distance to {:?}: {}", point.0, point.1);
}

// If you can, generate a grid sdf instead.
let bounding_box = get_bounding_box(&vertices); // fake api
let (min, max) = bounding_box.min_max();
let start = min;
let cell_count = [10, 10, 10];
let cell_radius = (max - min) / (cell_count - 1); // to query the whole bounding box.

let sdf: Vec<f32> = mesh_to_sdf::generate_grid_sdf(&vertices, &indices, &start, &cell_radius, &cell_count);

for x in 0..cell_count[0] {
    for y in 0..cell_count[1] {
        for z in 0..cell_count[2] {
            let index = z + y * cell_count[2] + x * cell_count[1] * cell_count[2];
            println!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index]);
        }
    }
}
```

Indices can be of any type that implements `Into<u32>`, e.g. `u16` and `u32`. Topology can be list or strip. Triangle orientation does not matter.
For vertices, this library aims to be as generic as possible by providing a trait `Point` that can be implemented for any type. In a near future, this crate will provide implementation for most common libraries (`glam`, `nalgebra`, etc.). Such implementations are gated behind feature flags. By default, only `[f32; 3]` is provided. If you do not find your favorite library, feel free to implement the trait for it and submit a PR or open an issue.

#### Using your favorite library

To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependencies. For example, to use `glam`:
```toml
[dependencies]
mesh_to_sdf = { version = "0.1", features = ["glam"] }
```

Currently, the following libraries are supported:
- **cgmath** (`cgmath::Vector3<f32>`)
- **glam** (`glam::Vec3`)
- **mint** (`mint::Vector3<f32>` and `mint::Point3<f32>`)
- **nalgebra** (`nalgebra::Vector3<f32>` and `nalgebra::Point3<f32>`)
- and `[f32; 3]`

#### Determining inside/outside

As of now, sign is computed by checking the normals of the triangles. This is not robust and might lead to negative distances leaking outside the mesh in pyramidal shapes. A more robust solution is planned for the future.

#### Benchmarks

`generate_grid_sdf` is much faster than `generate_sdf` and should be used whenever possible. `generate_sdf` does not allocate memory (except for the result array) but is slow. A faster implementation is planned for the future.

---

TODO: Add benchmarks against other libraries.

## `mesh_to_sdf_client` Mesh to SDF visualization client.

Load a gltf/glb file and visualize its signed distance field.

Credits:
- this app started with [rust_wgpu_hot_reload](https://github.com/Azkellas/rust_wgpu_hot_reload/) as template.
- [wgpu examples](https://github.com/gfx-rs/wgpu/tree/trunk/examples) were also extensively used to keep track of `wgpu <-> winit` integration after each `winit` update.

## TODO

This project is still in its early stages. Here is a list of things that are planned for (near) the future:
- [ ] [lib] Robust inside/outside checking with raycast
- [x] [lib] Implement `Point` for common libraries (`cgmath`, `nalgebra`, `mint`, ...)
- [ ] [lib] Optimize `mesh_to_sdf` with a bvh
- [ ] [lib] Optimize `mesh_to_sdf` by computing on the gpu
- [ ] [lib] Load/Save vf files
- [ ] [lib] General optimizations
- [ ] [lib] Tests/Benchmarks/Examples
- [ ] [lib] Methods to sample the surface
- [x] [client] Display meshes
- [ ] [client] Raymarch the surface
- [x] [client] Voxelize the mesh
- [x] [client] Undo/Redo
- [ ] [client] CLI