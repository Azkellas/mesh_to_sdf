use bvh::aabb::Aabb;
use bvh::bvh::{Bvh, BvhNode};
use nalgebra::SimdPartialOrd;

use crate::Point;

/// Returns the signed distance to a box.
/// See <https://iquilezles.org/articles/distfunctions/>
fn box_sdf<V: Point>(p: &V, aabb: &Aabb<f32, 3>) -> f32 {
    let p = nalgebra::Point3::new(p.x(), p.y(), p.z());
    let p = p - aabb.center();
    let q = p.abs() - aabb.size() * 0.5;

    // lhs is the outside part of the box
    let lhs = q.simd_max(nalgebra::Vector3::zeros());
    // rhs is the inside part of the box
    let rhs = q.max().min(0.0);
    (lhs + nalgebra::Vector3::from_element(rhs)).norm()
}

pub trait AabbExt {
    /// Returns the minimum and maximum distances to the box.
    /// The minimum distance is the distance to the closest point on the box,
    /// 0 if the point is inside the box.
    /// The maximum distance is the distance to the furthest point in the box.
    fn get_min_max_distances<V: Point>(&self, point: &V) -> (f32, f32);
}

impl AabbExt for Aabb<f32, 3> {
    /// Returns the minimum and maximum distances to the box.
    /// The minimum distance is the distance to the closest point on the box,
    /// 0 if the point is inside the box.
    /// The maximum distance is the distance to the furthest point in the box.
    fn get_min_max_distances<V: Point>(&self, point: &V) -> (f32, f32) {
        // Convert point to nalgebra point
        let n_point: nalgebra::OPoint<f32, nalgebra::Const<3>> =
            nalgebra::Point3::new(point.x(), point.y(), point.z());

        let half_size = self.size() * 0.5;
        let center = self.center();

        let delta = n_point - center;

        // See <https://iquilezles.org/articles/distfunctions/>
        let q = delta.abs() - half_size;
        let min_dist = q.map(|x| x.max(0.0)).norm();

        // The signum helps to determine the furthest point
        let signum = delta
            // Invert the signum to get the furthest vertex
            .map(|x| -x.signum())
            // Make sure we're always on a vertex and not on a face if the point is aligned with the box
            .map(|x| if x != 0.0 { x } else { 1.0 });

        let furthest = center + signum.component_mul(&half_size);
        let max_dist = (n_point - furthest).norm();

        (min_dist, max_dist)
    }
}

pub trait BvhDistance<V: Point> {
    /// Traverses the [`Bvh`].
    /// Returns a subset of `shapes` which are candidates for being the closest to `point`.
    ///
    fn nearest_candidates(&self, origin: &V) -> Vec<usize>
    where
        Self: std::marker::Sized;
}

impl<V: Point> BvhDistance<V> for Bvh<f32, 3> {
    /// Traverses the [`Bvh`].
    /// Returns a subset of `shapes` which are candidates for being the closest to `point`.
    ///
    fn nearest_candidates(&self, origin: &V) -> Vec<usize> {
        let mut indices = Vec::new();
        let mut best_min_distance = f32::MAX;
        let mut best_max_distance = f32::MAX;
        BvhNode::nearest_candidates_recursive(
            &self.nodes,
            0,
            origin,
            &mut indices,
            &mut best_min_distance,
            &mut best_max_distance,
        );

        indices
            .into_iter()
            .filter(|(_, node_min)| *node_min <= best_max_distance)
            .map(|(i, _)| i)
            .collect()
    }
}
pub trait BvhTraverseDistance<V: Point> {
    /// Traverses the [`Bvh`] recursively and returns all shapes whose [`Aabb`] countains
    /// a candidate shape for being the nearest to the given point.
    ///
    fn nearest_candidates_recursive(
        nodes: &[Self],
        node_index: usize,
        origin: &V,
        indices: &mut Vec<(usize, f32)>,
        best_min_distance: &mut f32,
        best_max_distance: &mut f32,
    ) where
        Self: std::marker::Sized;
}

impl<V: Point> BvhTraverseDistance<V> for BvhNode<f32, 3> {
    /// Traverses the [`Bvh`] recursively and returns all shapes whose [`Aabb`] countains
    /// a candidate shape for being the nearest to the given point.
    ///
    fn nearest_candidates_recursive(
        nodes: &[Self],
        node_index: usize,
        origin: &V,
        indices: &mut Vec<(usize, f32)>,
        best_min_distance: &mut f32,
        best_max_distance: &mut f32,
    ) {
        match nodes[node_index] {
            BvhNode::Node {
                ref child_l_aabb,
                child_l_index,
                ref child_r_aabb,
                child_r_index,
                ..
            } => {
                // Compute min/max dists for both children
                let [child_l_dists, child_r_dists] =
                    [child_l_aabb, child_r_aabb].map(|aabb| aabb.get_min_max_distances(origin));

                // Update best max distance before traversing children to avoid unnecessary traversals
                // where right node prunes left node.
                *best_max_distance = best_max_distance.min(child_l_dists.1.min(child_r_dists.1));

                // Traverse children
                for ((dist_min, dist_max), index) in [
                    (child_l_dists, child_l_index),
                    (child_r_dists, child_r_index),
                ] {
                    // Node is better by a margin.
                    if dist_max <= *best_min_distance {
                        indices.clear();
                    }

                    // Node might contain a candidate
                    if dist_min <= *best_max_distance {
                        Self::nearest_candidates_recursive(
                            nodes,
                            index,
                            origin,
                            indices,
                            best_min_distance,
                            best_max_distance,
                        );
                    }
                }
            }
            BvhNode::Leaf {
                shape_index,
                parent_index,
            } => {
                // Try to compute bounding box via parent node to update best_min/max_distances
                let parent_node = &nodes[parent_index];
                let min_dist = if let BvhNode::Node {
                    ref child_l_aabb,
                    child_l_index,
                    ref child_r_aabb,
                    ..
                } = parent_node
                {
                    // We found a parent node, we can compute the min/max distances for the leaf shape.
                    let aabb = if *child_l_index == node_index {
                        child_l_aabb
                    } else {
                        child_r_aabb
                    };
                    let (min_dist, max_dist) = aabb.get_min_max_distances(origin);

                    if !indices.is_empty() && max_dist < *best_min_distance {
                        // Node is better by a margin
                        indices.clear();
                    }

                    // Also update min_dist here since we have a credible (small) bounding box.
                    *best_min_distance = best_min_distance.min(min_dist);
                    *best_max_distance = best_max_distance.min(max_dist);

                    min_dist
                } else {
                    // The parent is a leaf if the tree is a single node (ie there is only one shape in the tree).
                    *best_min_distance
                };

                // we reached a leaf, we add it to the list of indices since it is a potential candidate
                indices.push((shape_index, min_dist));
            }
        }
    }
}
