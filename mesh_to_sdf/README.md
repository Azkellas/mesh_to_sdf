# mesh_to_sdf

This crate provides two entry points:

- [`generate_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` at the points `query_points`.
- [`generate_grid_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` on a [Grid].

```rust
use mesh_to_sdf::{generate_sdf, generate_grid_sdf, SignMethod, AccelerationMethod, Topology, Grid};
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
    AccelerationMethod::RtreeBvh, // Use an r-tree and a bvh to accelerate queries.
);

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
                         // Raycast is robust but requires the mesh to be watertight.
                         // and is more expensive.
                         // Normal might leak negative distances outside the mesh
);                       // but works for all meshes, even surfaces.

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
Implementations for most common math libraries are gated behind feature flags. By default, `[f32; 3]` and `nalgebra::[Point3, Vector3]` are provided.
If you do not find your favorite library, feel free to implement the trait for it and submit a PR or open an issue.

---

##### Computing sign

This crate provides two methods to compute the sign of the distance:
- [`SignMethod::Raycast`] (default): a robust method to compute the sign of the distance. It counts the number of intersections between a ray starting from the query point and the triangles of the mesh.
    It only works for watertight meshes, but guarantees the sign is correct.
- [`SignMethod::Normal`]: uses the normals of the triangles to estimate the sign by doing a dot product with the direction of the query point.
    It works for non-watertight meshes but might leak negative distances outside the mesh.

Using `Raycast` is slower than `Normal` but gives better results. Performances depends on the triangle count and method used.
On big dataset, `Raycast` is 5-10% slower for grid generation and rtree based methods. On smaller dataset, the difference can be worse
depending on whether the query is triangle intensive or query point intensive.
For bvh the difference is negligible between the two methods.

---

##### Acceleration structures

For generic queries, you can use acceleration structures to speed up the computation.
- [`AccelerationMethod::None`]: no acceleration structure. This is the slowest method but requires no extra memory. Scales really poorly.
- [`AccelerationMethod::Bvh`]: Bounding Volume Hierarchy. Accepts a `SignMethod`.
- [`AccelerationMethod::Rtree`]: R-tree. Uses `SignMethod::Normal`. The fastest method assuming you have more than a couple thousands of queries.
- [`AccelerationMethod::RtreeBvh`] (default): Uses R-tree for nearest neighbor search and Bvh for `SignMethod::Raycast`. 5-10% slower than `Rtree` on big datasets.

If your mesh is watertight and you have more than a thousand queries/triangles, you should use `AccelerationMethod::RtreeBvh` for best performances.
If it's not watertight, you can use `AccelerationMethod::Rtree` instead.

`Rtree` methods are 3-4x faster than `Bvh` methods for big enough data. On small meshes, the difference is negligible.
`AccelerationMethod::None` scales really poorly and should be avoided unless for small datasets or if you're really tight on memory.

---

##### Using your favorite library

To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependency. For example, to use `glam`:
```toml
[dependencies]
mesh_to_sdf = { version = "0.2.1", features = ["glam"] }
```

Currently, the following libraries are supported:
- [cgmath] ([`cgmath::Vector3<f32>`])
- [glam] ([`glam::Vec3`])
- [mint] ([`mint::Vector3<f32>`] and [`mint::Point3<f32>`])
- [nalgebra] ([`nalgebra::Vector3<f32>`] and [`nalgebra::Point3<f32>`])
- `[f32; 3]`

[nalgebra] is always available as it's used internally in the bvh tree.

---

##### Serialization

If you want to serialize and deserialize signed distance fields, you need to enable the `serde` feature.
This features also provides helpers to save and load signed distance fields to and from files via `save_to_file` and `read_from_file`.

License: MIT OR Apache-2.0
