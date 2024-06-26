# Changelog

## [Unreleased]

### Added

- `generate_sdf` takes an additional `AccelerationMethod` parameter, that can be either `AccelerationMethod::Bvh` (default) or `AccelerationMethod::None`.
    `Bvh` is recommended unless you're really tight on memory or your query is very small (couple hundreds of queries and triangles).
    For example, a query of 15k points with 100k triangles is 10 times faster with a bvh, and twice faster for 500 queries and 10k triangles.
    `Bvh` is also optimised for the `SignMethod::Raycast` sign method. https://github.com/Azkellas/mesh_to_sdf/pull/75

### Fixed

- Fix https://github.com/Azkellas/mesh_to_sdf/issues/25 where `generate_grid_sdf` might panic if the grid does not contain the mesh.


## [0.2.1] - 2024-02-18

### Changed

- `generate_grid_sdf` with `SignMethod::Raycast` now tests in the three directions (+x, +y, +z). This does not slow the computation down, but it makes the result more robust. This does not affect `generate_sdf`. 
- `Grid::from_bounding_box` takes a `[usize; 3]` instead of `&[usize; 3]` for `cell_count`


## [0.2.0] - 2024-02-16

### Added

- `SignMethod` enum to represent the sign method used to calculate the signed distance.
- `generate_sdf` and `generate_grid_sdf` now take a `SignMethod` parameter.
- `SignMethod::Raycast` to use raycasting to determine the sign of the distance.

## [0.1.0] - 2024-02-05

First release of `mesh_to_sdf`.

### Added

- `generate_sdf` function to get the signed distance from query points to a mesh.
- `generate_grid_sdf` function to get the signed distance from a grid to a mesh.
- `Point` trait to allow the use of any type as vertices.
- `Topology` enum to represent the topology of the mesh.
- `Grid` struct to represent a grid.
- `[f32; 3]` implementation for `Point` trait.
- `cgmath` implementation for `Point` trait.
- `glam` implementation for `Point` trait.
- `mint` implementation for `Point` trait.
- `nalgebra` implementation for `Point` trait.
