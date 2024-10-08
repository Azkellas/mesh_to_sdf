use bvh::aabb::{Aabb, Bounded};
use bvh::bvh::{Bvh, BvhNode};

use crate::Point;
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
            .map(|x| if x == 0.0 { 1.0 } else { x });

        let furthest = center + signum.component_mul(&half_size);
        let max_dist = (n_point - furthest).norm();

        (min_dist, max_dist)
    }
}

pub trait BvhDistance<V: Point, Shape: Bounded<f32, 3>> {
    /// Traverses the [`Bvh`].
    /// Returns a subset of `shapes` which are candidates for being the closest to `point`.
    ///
    fn nearest_candidates(&self, origin: &V, shapes: &[Shape]) -> Vec<usize>
    where
        Self: core::marker::Sized;
}

impl<V: Point, Shape: Bounded<f32, 3>> BvhDistance<V, Shape> for Bvh<f32, 3> {
    /// Traverses the [`Bvh`].
    /// Returns a subset of `shapes` which are candidates for being the closest to `point`.
    ///
    fn nearest_candidates(&self, origin: &V, shapes: &[Shape]) -> Vec<usize> {
        let mut indices = Vec::new();
        let mut best_min_distance = f32::MAX;
        let mut best_max_distance = f32::MAX;
        BvhNode::nearest_candidates_recursive(
            &self.nodes,
            0,
            origin,
            shapes,
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

pub trait BvhTraverseDistance<V: Point, Shape: Bounded<f32, 3>> {
    /// Traverses the [`Bvh`] recursively and returns all shapes whose [`Aabb`] countains
    /// a candidate shape for being the nearest to the given point.
    ///
    fn nearest_candidates_recursive(
        nodes: &[Self],
        node_index: usize,
        origin: &V,
        shapes: &[Shape],
        indices: &mut Vec<(usize, f32)>,
        best_min_distance: &mut f32,
        best_max_distance: &mut f32,
    ) where
        Self: core::marker::Sized;
}

impl<V: Point, Shape: Bounded<f32, 3>> BvhTraverseDistance<V, Shape> for BvhNode<f32, 3> {
    /// Traverses the [`Bvh`] recursively and returns all shapes whose [`Aabb`] countains
    /// a candidate shape for being the nearest to the given point.
    ///
    #[expect(clippy::similar_names)]
    fn nearest_candidates_recursive(
        nodes: &[Self],
        node_index: usize,
        origin: &V,
        shapes: &[Shape],
        indices: &mut Vec<(usize, f32)>,
        best_min_distance: &mut f32,
        best_max_distance: &mut f32,
    ) {
        match &nodes[node_index] {
            Self::Node {
                child_l_aabb,
                child_l_index,
                child_r_aabb,
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
                            *index,
                            origin,
                            shapes,
                            indices,
                            best_min_distance,
                            best_max_distance,
                        );
                    }
                }
            }
            Self::Leaf { shape_index, .. } => {
                let aabb = shapes[*shape_index].aabb();
                let (min_dist, max_dist) = aabb.get_min_max_distances(origin);

                if !indices.is_empty() && max_dist < *best_min_distance {
                    // Node is better by a margin
                    indices.clear();
                }

                // Also update min_dist here since we have a credible (small) bounding box.
                *best_min_distance = best_min_distance.min(min_dist);
                *best_max_distance = best_max_distance.min(max_dist);

                // we reached a leaf, we add it to the list of indices since it is a potential candidate
                indices.push((*shape_index, min_dist));
            }
        }
    }
}
