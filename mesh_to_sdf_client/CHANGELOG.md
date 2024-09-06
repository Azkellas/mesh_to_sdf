# Changelog

## [0.3.0]

### Added

- Voxel and Raymarch visualizations now map the material of the mesh onto the new geometry.

## [0.2.1] - 2024-02-18

Update `mesh_to_sdf` dependency to `0.2.1` since it had a breaking change.


## [0.2.0] - 2024-02-16

### Added

- Isosurface control for point cloud, voxels and raymarching visualization.
- Expose `SignMethod` enum in UI.

## [0.1.0] - 2024-02-05

First release of `mesh_to_sdf_client`.

### Added

- Model visualization with a Blinn-Phong shader.
- SDF visualization with a point cloud.
- SDF point size and color customization.
- Model + SDF visualization.
- Raymarching visualization (snap, trilinear and tetrahedral interpolations).
- Voxels visualization.
- Model and SDF metrics.
- Light control.
- Grid parametrization (cell count, size, and bounding box extension).
