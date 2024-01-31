# mesh_to_sdf

⚠️ This crate is still in its early stages. Expect the API to change.

---

This crate provides two entry points:

- [`generate_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` at the points `query_points`.
- [`generate_grid_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` on a grid with `cell_count` cells of size `cell_radius` starting at `start_pos`.

```rust
let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
let indices: Vec<u32> = vec![0, 1, 2];

let query_points: Vec<[f32; 3]> = vec![[0., 0., 0.]];

// Query points are expected to be in the same space as the mesh.
let sdf: Vec<f32> = generate_sdf(
    &vertices,
    Topology::TriangleList(Some(&indices)),
    &query_points);

for point in query_points.iter().zip(sdf.iter()) {
    println!("Distance to {:?}: {}", point.0, point.1);
}

// if you can, use generate_grid_sdf instead of generate_sdf.
let bounding_box_min = [0., 0., 0.];
let bounding_box_max = [10., 10., 10.];
let cell_radius = [1., 1., 1.];
let cell_count = [11, 11, 11]; // 0 1 2 .. 10 = 11 samples

let sdf: Vec<f32> = generate_grid_sdf(
    &vertices,
    Topology::TriangleList(Some(&indices)),
    &bounding_box_min,
    &cell_radius,
    &cell_count);

for x in 0..cell_count[0] {
    for y in 0..cell_count[1] {
        for z in 0..cell_count[2] {
            let index = z + y * cell_count[2] + x * cell_count[1] * cell_count[2];
            println!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index as usize]);
        }
    }
}
```

Indices can be of any type that implements `Into<u32>`, e.g. `u16` and `u32`. Topology can be list or strip. Triangle orientation does not matter.
For vertices, this library aims to be as generic as possible by providing a trait `Point` that can be implemented for any type. In a near future, this crate will provide implementation for most common libraries (`glam`, `nalgebra`, etc.). Such implementations are gated behind feature flags. By default, only `[f32; 3]` is provided. If you do not find your favorite library, feel free to implement the trait for it and submit a PR or open an issue.

##### Using your favorite library

To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependencies. For example, to use `glam`:
```toml
[dependencies]
mesh_to_sdf = { version = "0.1", features = ["glam"] }
```

Currently, the following libraries are supported:
- `cgmath` (`cgmath::Vector3<f32>`)
- `glam` (`glam::Vec3`)
- `mint` (`mint::Vector3<f32>` and `mint::Point3<f32>`)
- `nalgebra` (`nalgebra::Vector3<f32>` and `nalgebra::Point3<f32>`)
- and `[f32; 3]`

##### Determining inside/outside

As of now, sign is computed by checking the normals of the triangles. This is not robust and might lead to negative distances leaking outside the mesh in pyramidal shapes. A more robust solution is planned for the future.

##### Benchmarks

[`generate_grid_sdf`] is much faster than [`generate_sdf`] and should be used whenever possible. [`generate_sdf`] does not allocate memory (except for the result array) but is slow. A faster implementation is planned for the future.

License: MIT OR Apache-2.0
