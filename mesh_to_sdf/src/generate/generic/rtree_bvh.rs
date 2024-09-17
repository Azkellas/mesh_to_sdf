//! Module containing the `generate_sdf_rtree_bvh` function.

use bvh::{bounding_hierarchy::BoundingHierarchy, bvh::Bvh};
use itertools::Itertools;
use rayon::prelude::*;

use crate::{geo, Point, Topology};

use super::rtree::PointWrapper;

/// `RtreeBvhNode` is a node for the r-tree and bvh acceleration structures.
#[derive(Clone)]
struct RtreeBvhNode<V: Point> {
    vertices: (V, V, V),
    bounding_box: (V, V),
    node_index: usize,
}

impl<V: Point> rstar::RTreeObject for RtreeBvhNode<V> {
    type Envelope = rstar::AABB<PointWrapper<V>>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_corners(
            PointWrapper(self.bounding_box.0),
            PointWrapper(self.bounding_box.1),
        )
    }
}

impl<V: Point> rstar::PointDistance for RtreeBvhNode<V> {
    // Required method
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        geo::point_triangle_distance2(
            &point.0,
            &self.vertices.0,
            &self.vertices.1,
            &self.vertices.2,
        )
    }
}

impl<V: Point> bvh::aabb::Bounded<f32, 3> for RtreeBvhNode<V> {
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

impl<V: Point> bvh::bounding_hierarchy::BHShape<f32, 3> for RtreeBvhNode<V> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// Generate a signed distance field from a mesh using an r-tree for nearest neighbor search and a bvh for ray intersection.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
pub fn generate_sdf_rtree_bvh<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    let mut rtree_bvh_nodes = Topology::get_triangles(vertices, indices)
        .map(|triangle| RtreeBvhNode {
            vertices: (
                vertices[triangle.0],
                vertices[triangle.1],
                vertices[triangle.2],
            ),
            node_index: 0,
            bounding_box: geo::triangle_bounding_box(
                &vertices[triangle.0],
                &vertices[triangle.1],
                &vertices[triangle.2],
            ),
        })
        .collect_vec();

    if rtree_bvh_nodes.is_empty() {
        return vec![];
    }

    let rtree = rstar::RTree::bulk_load(rtree_bvh_nodes.clone());
    let bvh = Bvh::build_par(&mut rtree_bvh_nodes);

    query_points
        .par_iter()
        .map(|point| {
            let nearest = rtree.nearest_neighbor(&PointWrapper(*point));
            let nearest = nearest.unwrap(); // tree isn't empty.

            let dist = geo::point_triangle_distance(
                point,
                &nearest.vertices.0,
                &nearest.vertices.1,
                &nearest.vertices.2,
            );

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
                let hitcast = bvh.traverse(&ray, &rtree_bvh_nodes);
                for bvh_node in hitcast {
                    let a = &bvh_node.vertices.0;
                    let b = &bvh_node.vertices.1;
                    let c = &bvh_node.vertices.2;
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
                -dist
            } else {
                dist
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{generate_grid_sdf, generate_sdf, AccelerationMethod, Grid, SignMethod};

    use super::*;

    #[test]
    fn test_generate_rtree_bvh() {
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

        let rtree_sdf = generate_sdf_rtree_bvh(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
        );

        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh(SignMethod::Raycast),
        );

        for (idx, (rtree, sdf)) in rtree_sdf.iter().zip(sdf.iter()).enumerate() {
            assert!(
                (rtree - sdf).abs() < 0.01,
                "{:?}: {} != {}",
                query_points[idx],
                rtree,
                sdf
            );
        }
    }

    #[test]
    fn test_generate_rtree_bvh_big() {
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
            AccelerationMethod::RtreeBvh,
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Raycast,
        );

        // Test against generate_sdf
        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
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
