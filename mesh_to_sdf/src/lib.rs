use std::boxed::Box;

use itertools::Itertools;
use rayon::prelude::*;

mod geo;
mod point;

pub use point::Point;

/// Mesh Topology
pub enum Topology<'a, I>
where
    // I should be a u32 or u16
    I: Into<u32>,
{
    /// Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
    TriangleList(Option<&'a [I]>),
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create four triangles `0 1 2`, `2 1 3`, `2 3 4`, and `4 3 5`
    TriangleStrip(Option<&'a [I]>),
}

/// Generate a signed distance field from a mesh.
/// Compute the signed distance for each query point.
pub fn generate_sdf<V, I>(vertices: &[V], indices: Topology<I>, query_points: &[V]) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    query_points
        .par_iter()
        .map(|query| {
            let triangles: Box<dyn Iterator<Item = (usize, usize, usize)>> = match indices {
                Topology::TriangleList(Some(triangles)) => {
                    Box::new(triangles.iter().map(|x| (*x).into() as usize).tuples())
                }
                // TODO: test
                Topology::TriangleList(None) => Box::new((0..vertices.len()).into_iter().tuples()),
                // TODO: test
                Topology::TriangleStrip(Some(triangles)) => Box::new(
                    triangles
                        .into_iter()
                        .map(|x| (*x).into() as usize)
                        .tuple_windows(),
                ),
                // TODO: test
                Topology::TriangleStrip(None) => {
                    Box::new((0..vertices.len()).into_iter().tuple_windows())
                }
            };

            triangles
                .map(|(i, j, k)| (&vertices[i], &vertices[j], &vertices[k]))
                .map(|(a, b, c)| {
                    // unsigned distance.
                    let mut distance = geo::point_triangle_distance(query, a, b, c);

                    // signed distance: positive if the point is outside the mesh, negative if inside.
                    // assume all normals are pointing outside the mesh.
                    let barycenter = geo::triangle_barycenter(a, b, c);
                    let direction = query.sub(&barycenter);
                    // No need for it to be normalized.
                    let normal = geo::triangle_normal(a, b, c);
                    if direction.dot(&normal) < 0.0 {
                        distance = -distance;
                    }

                    distance
                })
                // find the closest triangle
                .min_by(|a, b| {
                    // for a point to be inside, it has to be inside all normals of nearest triangles.
                    // if one distance is positive, then the point is outside.
                    // this check is sensible to floating point errors though
                    // so it's not perfect, but it reduces the number of false positives considerably.
                    if float_cmp::approx_eq!(f32, a.abs(), b.abs(), ulps = 2, epsilon = 1e-6) {
                        // return the one with the smallest distance, privileging positive distances.
                        match (a.is_sign_negative(), b.is_sign_negative()) {
                            (true, false) => std::cmp::Ordering::Greater,
                            (false, true) => std::cmp::Ordering::Less,
                            _ => a.abs().partial_cmp(&b.abs()).unwrap(),
                        }
                    } else {
                        a.abs().partial_cmp(&b.abs()).unwrap()
                    }
                })
                .unwrap()
        })
        .collect()
}
