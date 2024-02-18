# Mesh to SDF

![](client.gif)

This repository contains two crates:
- `mesh_to_sdf`: a library to convert a mesh to a signed distance field (SDF).
- `mesh_to_sdf_client`: a visualization client for the SDF of a mesh.

documentation: https://docs.rs/mesh_to_sdf/latest/mesh_to_sdf/

---

## `mesh_to_sdf`: Convert a mesh to a signed distance field (SDF).

⚠️ This crate is still in its early stages. Expect the API to change.

---

This crate provides two entry points:

- [`generate_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` at the points `query_points`.
- [`generate_grid_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` on a [Grid].

```rust
use mesh_to_sdf::{generate_sdf, generate_grid_sdf, SignMethod, Topology, Grid};
// vertices are [f32; 3], but can be cgmath::Vector3<f32>, glam::Vec3, etc.
let vertices: Vec<[f32; 3]> = vec![[0.5, 1.5, 0.5], [1., 2., 3.], [1., 3., 7.]];
let indices: Vec<u32> = vec![0, 1, 2];

// query points must be of the same type as vertices
let query_points: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]];

// Query points are expected to be in the same space as the mesh.
let sdf: Vec<f32> = generate_sdf(
    &vertices,
    Topology::TriangleList(Some(&indices)), // TriangleList as opposed to TriangleStrip
    &query_points,
    SignMethod::Raycast, // How the sign is computed.
);                       // Raycast is robust but requires the mesh to be watertight.

for point in query_points.iter().zip(sdf.iter()) {
    // distance is positive outside the mesh and negative inside.
    println!("Distance to {:?}: {}", point.0, point.1);
}

// if you can, use generate_grid_sdf instead of generate_sdf as it's optimized and much faster.
let bounding_box_min = [0., 0., 0.];
let bounding_box_max = [10., 10., 10.];
let cell_count = [10, 10, 10];

let grid = Grid::from_bounding_box(&bounding_box_min, &bounding_box_max, cell_count);

let sdf: Vec<f32> = generate_grid_sdf(
    &vertices,
    Topology::TriangleList(Some(&indices)),
    &grid,
    SignMethod::Raycast, // How the sign is computed.
);                       // Raycast is robust but requires the mesh to be watertight.

for x in 0..cell_count[0] {
    for y in 0..cell_count[1] {
        for z in 0..cell_count[2] {
            let index = grid.get_cell_idx(&[x, y, z]);
            log::info!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index as usize]);
        }
    }
}
```

---

##### Mesh Topology

Indices can be of any type that implements `Into<u32>`, e.g. `u16` and `u32`. Topology can be list or strip.
If the indices are not provided, they are supposed to be `0..vertices.len()`.

For vertices, this library aims to be as generic as possible by providing a trait `Point` that can be implemented for any type.
Implementations for most common math libraries are gated behind feature flags. By default, only `[f32; 3]` is provided.
If you do not find your favorite library, feel free to implement the trait for it and submit a PR or open an issue.

---

##### Computing sign

This crate provides two methods to compute the sign of the distance:
- `SignMethod::Raycast` (default): a robust method to compute the sign of the distance. It counts the number of intersection between a ray starting from the query point and the triangles of the mesh.
    It only works for watertight meshes, but garantees the sign is correct.
- `SignMethod::Normal`: It uses the normals of the triangles to estimate the sign by doing a dot product with the direction of the query point.
    It works for non-watertight meshes but might leak negative distances outside the mesh.

For grid generation, `Raycast` is ~1% slower.
For query points, `Raycast` is ~10% slower.
Note that it depends on the query points / grid size to triangle ratio, but this gives a rough idea.

---

##### Using your favorite library

To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependency. For example, to use `glam`:
```toml
[dependencies]
mesh_to_sdf = { version = "0.2.0", features = ["glam"] }
```

Currently, the following libraries are supported:
- `cgmath` (`cgmath::Vector3<f32>`)
- `glam` (`glam::Vec3`)
- `mint` (`mint::Vector3<f32>` and `mint::Point3<f32>`)
- `nalgebra` (`nalgebra::Vector3<f32>` and `nalgebra::Point3<f32>`)
- `[f32; 3]`

---

##### Benchmarks

[`generate_grid_sdf`] is much faster than [`generate_sdf`] and should be used whenever possible.
[`generate_sdf`] does not allocate memory (except for the result array) but is slow. A faster implementation is planned for the future.
[`SignMethod::Raycast`] is slightly slower than [`SignMethod::Normal`] but is robust and should be used whenever possible (~1% in [`generate_grid_sdf`], ~10% in [`generate_sdf`]).

---

TODO: Add benchmarks against other libraries.

## `mesh_to_sdf_client` Mesh to SDF visualization client.

Load a gltf/glb file and visualize its signed distance field.

Credits:
- this app started with [rust_wgpu_hot_reload](https://github.com/Azkellas/rust_wgpu_hot_reload/) as template.
- [wgpu examples](https://github.com/gfx-rs/wgpu/tree/trunk/examples) were also extensively used to keep track of `wgpu <-> winit` integration after each `winit` update.

## TODO

This project is still in its early stages. Here is a list of things that are planned for (near) the future:
- [x] [lib] Robust inside/outside checking with raycast
- [x] [lib] Implement `Point` for common libraries (`cgmath`, `nalgebra`, `mint`, ...)
- [ ] [lib] Optimize `mesh_to_sdf` with a bvh
- [ ] [lib] Optimize `mesh_to_sdf` by computing on the gpu
- [ ] [lib] Load/Save vf files
- [ ] [lib] General optimizations
- [x] [lib] Tests/Benchmarks/Examples
- [ ] [lib] Methods to sample the surface
- [x] [client] Display meshes
- [x] [client] Raymarch the surface
- [x] [client] Voxelize the mesh
- [x] [client] Undo/Redo
- [ ] [client] CLI
