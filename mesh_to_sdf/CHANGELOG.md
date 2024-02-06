# Changelog

## [Unreleased]


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
