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
    fn point_distance<V: Point>(&self, point: &V) -> f32;
    fn body_diagonal(&self) -> f32;
}

impl AabbExt for Aabb<f32, 3> {
    /// Returns the signed distance to the aabb.
    /// See <https://iquilezles.org/articles/distfunctions/>
    fn point_distance<V: Point>(&self, point: &V) -> f32 {
        let p = nalgebra::Point3::new(point.x(), point.y(), point.z());
        let p = p - self.center();
        let q = p.abs() - self.size() * 0.5;

        // lhs is the outside part of the box
        let lhs = nalgebra::Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
        // rhs is the inside part of the box
        let rhs = q.max().min(0.0);
        (lhs + nalgebra::Vector3::from_element(rhs)).norm()
    }

    /// Returns the body diagonal of the aabb.
    fn body_diagonal(&self) -> f32 {
        self.size().norm()
    }
}

pub trait BvhDistance<V: Point> {
    fn traverse_distance(&self, origin: &V) -> Vec<usize>
    where
        Self: std::marker::Sized;
}

impl<V: Point> BvhDistance<V> for Bvh<f32, 3> {
    fn traverse_distance(&self, origin: &V) -> Vec<usize> {
        let mut indices = Vec::new();
        let mut best_min_distance = f32::MAX;
        let mut best_max_distance = f32::MAX;
        BvhNode::traverse_recursive_distance(
            &self.nodes,
            0,
            origin,
            &mut indices,
            &mut best_min_distance,
            &mut best_max_distance,
        );

        indices
            .into_iter()
            .filter(|(_, node_min, _)| *node_min <= best_max_distance)
            .map(|(i, _, _)| i)
            .collect()
    }
}
pub trait BvhTraverseDistance<V: Point> {
    fn traverse_recursive_distance(
        nodes: &[Self],
        node_index: usize,
        origin: &V,
        indices: &mut Vec<(usize, f32, f32)>,
        best_min_distance: &mut f32,
        best_max_distance: &mut f32,
    ) where
        Self: std::marker::Sized;
}

impl<V: Point> BvhTraverseDistance<V> for BvhNode<f32, 3> {
    fn traverse_recursive_distance(
        nodes: &[Self],
        node_index: usize,
        origin: &V,
        indices: &mut Vec<(usize, f32, f32)>,
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
                let l_dist = child_l_aabb.point_distance(origin);
                let r_dist = child_r_aabb.point_distance(origin);

                let l_min_dist = l_dist.max(0.0);
                let r_min_dist = r_dist.max(0.0);

                // if l_dist is positive, we add the dist to aabb to the body diagonal
                // if it is negative, we still add it and it will be a subtraction since it is negative.
                // in both cases it keeps the triangle inequality.
                let l_max_dist = l_dist + child_l_aabb.body_diagonal();
                let r_max_dist = r_dist + child_r_aabb.body_diagonal();

                if !indices.is_empty() && l_max_dist < *best_min_distance {
                    // left node is better by a margin
                    indices.clear();
                }

                if !indices.is_empty() && r_max_dist < *best_min_distance {
                    // right node is better by a margin
                    indices.clear();
                }

                *best_max_distance = best_max_distance.min(l_max_dist.min(r_max_dist));

                if l_min_dist <= *best_max_distance {
                    Self::traverse_recursive_distance(
                        nodes,
                        child_l_index,
                        origin,
                        indices,
                        best_min_distance,
                        best_max_distance,
                    );
                }

                if r_min_dist <= *best_max_distance {
                    Self::traverse_recursive_distance(
                        nodes,
                        child_r_index,
                        origin,
                        indices,
                        best_min_distance,
                        best_max_distance,
                    );
                }
            }
            BvhNode::Leaf {
                shape_index,
                parent_index,
            } => {
                let parent_node = &nodes[parent_index];
                if let BvhNode::Node {
                    ref child_l_aabb,
                    child_l_index,
                    ref child_r_aabb,
                    child_r_index,
                    ..
                } = parent_node
                {
                    let mut min_dist = f32::MAX;
                    let mut max_dist = f32::MAX;
                    if *child_l_index == node_index {
                        let l_dist = child_l_aabb.point_distance(origin);
                        min_dist = l_dist.max(0.0);
                        max_dist = l_dist + child_l_aabb.body_diagonal();
                    }
                    if *child_r_index == node_index {
                        let r_dist = child_r_aabb.point_distance(origin);
                        min_dist = r_dist.max(0.0);
                        max_dist = r_dist + child_r_aabb.body_diagonal();
                    }

                    if !indices.is_empty() && max_dist < *best_min_distance {
                        // left node is better by a margin
                        indices.clear();
                    }
                    *best_min_distance = best_min_distance.min(min_dist);
                    *best_max_distance = best_max_distance.min(max_dist);

                    // we reached a leaf, we add it to the list of indices since it is a potential candidate
                    indices.push((shape_index, min_dist, max_dist));
                } else {
                    // The parent is a leaf if the tree is a single node (ie there is only one shape in the tree).
                    indices.push((shape_index, *best_min_distance, *best_max_distance));
                }
            }
        }
    }
}
