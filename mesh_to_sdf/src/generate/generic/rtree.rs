//! Module containing the `generate_sdf_rtree` function.

use itertools::Itertools;
use rayon::prelude::*;

use crate::{geo, Point, Topology};

/// Wrapper around a point to make it compatible with the r-tree.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Hash)]
pub(super) struct PointWrapper<V>(pub V);

impl<V> rstar::Point for PointWrapper<V>
where
    V: Point,
{
    type Scalar = f32;

    const DIMENSIONS: usize = 3;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        let x = generator(0);
        let y = generator(1);
        let z = generator(2);
        PointWrapper(V::new(x, y, z))
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.0.x(),
            1 => self.0.y(),
            2 => self.0.z(),
            _ => panic!("Index out of bounds"),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => self.0.x_mut(),
            1 => self.0.y_mut(),
            2 => self.0.z_mut(),
            _ => panic!("Index out of bounds"),
        }
    }
}

/// `RtreeNode` is a node for the r-tree acceleration structure.
#[derive(Clone)]
struct RtreeNode<V: Point> {
    vertices: (V, V, V),
    bounding_box: (V, V),
}

impl<V: Point> rstar::RTreeObject for RtreeNode<V> {
    type Envelope = rstar::AABB<PointWrapper<V>>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_corners(
            PointWrapper(self.bounding_box.0),
            PointWrapper(self.bounding_box.1),
        )
    }
}

impl<V: Point> rstar::PointDistance for RtreeNode<V> {
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

/// Generate a signed distance field from a mesh using an r-tree.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
///
/// This method is only compatible with normal sign method.
/// If you want to use raycasting, use `AccelerationMethod::RtreeBvh` instead.
pub(crate) fn generate_sdf_rtree<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    let bvh_nodes = Topology::get_triangles(vertices, indices)
        .map(|triangle| RtreeNode {
            vertices: (
                vertices[triangle.0],
                vertices[triangle.1],
                vertices[triangle.2],
            ),
            bounding_box: geo::triangle_bounding_box(
                &vertices[triangle.0],
                &vertices[triangle.1],
                &vertices[triangle.2],
            ),
        })
        .collect_vec();

    let rtree = rstar::RTree::bulk_load(bvh_nodes);

    query_points
        .par_iter()
        .map(|point| {
            let nearest = rtree.nearest_neighbor(&PointWrapper(*point));
            let nearest = nearest.unwrap();
            geo::point_triangle_signed_distance(
                point,
                &nearest.vertices.0,
                &nearest.vertices.1,
                &nearest.vertices.2,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{generate_grid_sdf, generate_sdf, AccelerationMethod, Grid, SignMethod};

    use super::*;

    #[test]
    fn test_generate_rtree() {
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

        let rtree_sdf = generate_sdf_rtree(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
        );

        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh(SignMethod::Normal),
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
    fn test_generate_rtree_big() {
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
            AccelerationMethod::Rtree,
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Normal,
        );

        let mut fails = 0;
        // Test against generate_sdf
        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
            // Assert we have the same absolute value.
            assert!(
                (sdf.abs() - grid_sdf.abs()).abs() < 0.01,
                "i: {}: {} {}",
                i,
                sdf,
                grid_sdf
            );

            // Count sign issues.
            if (sdf - grid_sdf).abs() > 0.01 {
                fails += 1;
            }
        }

        // We can expect sign to be different for some cells.
        // This is because the rtree only outputs the closest triangle
        // while bvh works with a subset of close triangles and refines the sign based on those.
        assert!(
            (fails as f32 / sdf.len() as f32) < 0.01,
            "fails: {fails}/{}",
            sdf.len()
        );
    }
}
