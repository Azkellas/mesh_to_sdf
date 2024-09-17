//! Module containing the `generate_sdf_bvh` function.

use std::cmp::Ordering;

use bvh::{bounding_hierarchy::BoundingHierarchy, bvh::Bvh};
use itertools::Itertools;
use rayon::prelude::*;

use crate::{bvh_ext::BvhDistance, compare_distances, geo, Point, SignMethod, Topology};

/// A node in the BVH tree containing the data for a triangle.
/// Public in the crate so that the grid generation can use it.
/// `RtreeBvh` uses its own version of this struct to be able to implement the `RTreeObject` trait.
#[derive(Clone)]
pub(crate) struct BvhNode<V: Point> {
    pub vertex_indices: (usize, usize, usize),
    pub node_index: usize,
    pub bounding_box: (V, V),
}

impl<V: Point> bvh::aabb::Bounded<f32, 3> for BvhNode<V> {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        let min = nalgebra::Point3::new(
            self.bounding_box.0.x(),
            self.bounding_box.0.y(),
            self.bounding_box.0.z(),
        );
        let max = nalgebra::Point3::new(
            self.bounding_box.1.x(),
            self.bounding_box.1.y(),
            self.bounding_box.1.z(),
        );
        bvh::aabb::Aabb::with_bounds(min, max)
    }
}

impl<V: Point> bvh::bounding_hierarchy::BHShape<f32, 3> for BvhNode<V> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// Generate a signed distance field from a mesh using a bvh.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
pub(crate) fn generate_sdf_bvh<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
    sign_method: SignMethod,
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    let mut bvh_nodes = Topology::get_triangles(vertices, indices)
        .map(|triangle| BvhNode {
            vertex_indices: triangle,
            node_index: 0,
            bounding_box: geo::triangle_bounding_box(
                &vertices[triangle.0],
                &vertices[triangle.1],
                &vertices[triangle.2],
            ),
        })
        .collect_vec();

    let bvh = Bvh::build_par(&mut bvh_nodes);

    query_points
        .par_iter()
        .map(|point| {
            let bvh_indices = bvh.nearest_candidates(point, &bvh_nodes);

            let mut min_dist = f32::MAX;
            if sign_method == SignMethod::Normal {
                for index in &bvh_indices {
                    let triangle = &bvh_nodes[*index];
                    let a = &vertices[triangle.vertex_indices.0];
                    let b = &vertices[triangle.vertex_indices.1];
                    let c = &vertices[triangle.vertex_indices.2];
                    let distance = geo::point_triangle_signed_distance(point, a, b, c);

                    if compare_distances(min_dist, distance) == Ordering::Greater {
                        min_dist = distance;
                    }
                }
                min_dist
            } else {
                for index in &bvh_indices {
                    let triangle = &bvh_nodes[*index];
                    let a = &vertices[triangle.vertex_indices.0];
                    let b = &vertices[triangle.vertex_indices.1];
                    let c = &vertices[triangle.vertex_indices.2];
                    let distance = geo::point_triangle_distance(point, a, b, c);

                    min_dist = min_dist.min(distance);
                }

                let alignments = [
                    (geo::GridAlign::X, nalgebra::Vector3::new(1.0, 0.0, 0.0)),
                    (geo::GridAlign::Y, nalgebra::Vector3::new(0.0, 1.0, 0.0)),
                    (geo::GridAlign::Z, nalgebra::Vector3::new(0.0, 0.0, 1.0)),
                ];

                let mut insides = 0;
                for (alignment, direction) in alignments {
                    let ray = bvh::ray::Ray::new(
                        nalgebra::Point3::new(point.x(), point.y(), point.z()),
                        direction,
                    );
                    let mut intersection_count = 0;
                    let hitcast = bvh.traverse(&ray, &bvh_nodes);
                    for bvh_node in hitcast {
                        let a = &vertices[bvh_node.vertex_indices.0];
                        let b = &vertices[bvh_node.vertex_indices.1];
                        let c = &vertices[bvh_node.vertex_indices.2];
                        let intersect =
                            geo::ray_triangle_intersection_aligned(point, [a, b, c], alignment);
                        if intersect.is_some() {
                            intersection_count += 1;
                        }
                    }

                    if intersection_count % 2 == 1 {
                        insides += 1;
                    }
                }

                // Return inside if at least two are insides.
                if insides > 1 {
                    -min_dist
                } else {
                    min_dist
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{generate_grid_sdf, generate_sdf, AccelerationMethod, Grid};

    use super::*;

    #[test]
    fn test_generate_bvh() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();
        let query_points = [
            cgmath::Vector3::new(0.01, 0.01, 0.5),
            cgmath::Vector3::new(1.0, 1.0, 1.0),
            cgmath::Vector3::new(0.1, 0.2, 0.2),
            cgmath::Vector3::new(1.1, 2.2, 5.2),
            cgmath::Vector3::new(-0.1, 0.2, -0.2),
        ];

        let bvh_sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh(SignMethod::Raycast),
        );

        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::None(SignMethod::Raycast),
        );

        for (idx, (bvh, sdf)) in bvh_sdf.iter().zip(sdf.iter()).enumerate() {
            assert!(
                (bvh - sdf).abs() < 0.01,
                "{:?}: {} != {}",
                query_points[idx],
                bvh,
                sdf
            );
        }
    }

    #[test]
    fn test_generate_bvh_big() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();

        let bbox_min = vertices.iter().fold(
            cgmath::Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.min(v.x),
                y: acc.y.min(v.y),
                z: acc.z.min(v.z),
            },
        );
        let bbox_max = vertices.iter().fold(
            cgmath::Vector3::new(-f32::MAX, -f32::MAX, -f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.max(v.x),
                y: acc.y.max(v.y),
                z: acc.z.max(v.z),
            },
        );

        let grid = Grid::from_bounding_box(&bbox_min, &bbox_max, [32, 32, 32]);
        let mut query_points = Vec::new();
        for x in 0..grid.get_cell_count()[0] {
            for y in 0..grid.get_cell_count()[1] {
                for z in 0..grid.get_cell_count()[2] {
                    query_points.push(grid.get_cell_center(&[x, y, z]));
                }
            }
        }
        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh(SignMethod::Raycast),
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Raycast,
        );

        // Test against generate_sdf
        // TODO: sometimes fails
        // thread 'tests::test_generate_bvh_big' panicked at mesh_to_sdf\src\lib.rs:1232:13:
        // i: 17435: 0.0076956493 0.030284861
        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
            assert!(
                (sdf - grid_sdf).abs() < 0.01,
                "cell: {:?}: {} {}",
                grid.get_cell_integer_coordinates(i),
                sdf,
                grid_sdf
            );
        }
    }

    #[test]
    fn test_generate_bvh_normal() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();

        let bbox_min = vertices.iter().fold(
            cgmath::Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.min(v.x),
                y: acc.y.min(v.y),
                z: acc.z.min(v.z),
            },
        );
        let bbox_max = vertices.iter().fold(
            cgmath::Vector3::new(-f32::MAX, -f32::MAX, -f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.max(v.x),
                y: acc.y.max(v.y),
                z: acc.z.max(v.z),
            },
        );

        let grid = Grid::from_bounding_box(&bbox_min, &bbox_max, [16, 16, 16]);
        let mut query_points = Vec::new();
        for x in 0..grid.get_cell_count()[0] {
            for y in 0..grid.get_cell_count()[1] {
                for z in 0..grid.get_cell_count()[2] {
                    query_points.push(grid.get_cell_center(&[x, y, z]));
                }
            }
        }
        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh(SignMethod::Normal),
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Normal,
        );

        // Test against generate_sdf
        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
            // TODO: sometimes fails
            // thread 'tests::test_generate_bvh_big' panicked at mesh_to_sdf\src\lib.rs:1232:13:
            // i: 17435: 0.0076956493 0.030284861
            // i: 1742: 0.09342232 -0.094851956
            assert!(
                (sdf - grid_sdf).abs() < 0.01,
                "i: {}: {} {}",
                i,
                sdf,
                grid_sdf
            );
        }
    }
}
