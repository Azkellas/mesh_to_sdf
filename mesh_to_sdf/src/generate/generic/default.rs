//! Module containing the `generate_sdf_default` function.
use rayon::prelude::*;

use crate::{compare_distances, geo, Point, SignMethod, Topology};

/// Generate a signed distance field from a mesh.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
pub fn generate_sdf_default<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
    sign_method: SignMethod,
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    // For each query point, we compute the distance to each triangle.
    // sign is estimated by comparing the normal to the direction.
    // when two triangles give the same distance (wrt floating point errors),
    // we keep the one with positive distance since to be inside means to be inside all triangles.
    // whereas to be outside means to be outside at least one triangle.
    // see `compare_distances` for more details.
    query_points
        .par_iter()
        .map(|query| {
            Topology::get_triangles(vertices, indices)
                .map(|(i, j, k)| (&vertices[i], &vertices[j], &vertices[k]))
                .map(|(a, b, c)| match sign_method {
                    // Raycast: returns (distance, ray_intersection)
                    SignMethod::Raycast => (
                        geo::point_triangle_distance(query, a, b, c),
                        geo::ray_triangle_intersection_aligned(query, [a, b, c], geo::GridAlign::X)
                            .is_some(),
                    ),
                    // Normal: returns (signed_distance, false)
                    SignMethod::Normal => {
                        (geo::point_triangle_signed_distance(query, a, b, c), false)
                    }
                })
                .fold(
                    (f32::MAX, 0),
                    |(min_distance, intersection_count), (distance, ray_intersection)| {
                        match sign_method {
                            SignMethod::Raycast => (
                                min_distance.min(distance),
                                intersection_count + u32::from(ray_intersection),
                            ),
                            SignMethod::Normal => (
                                match compare_distances(distance, min_distance) {
                                    core::cmp::Ordering::Less => distance,
                                    core::cmp::Ordering::Equal | core::cmp::Ordering::Greater => {
                                        min_distance
                                    }
                                },
                                intersection_count,
                            ),
                        }
                    },
                )
        })
        .map(|(distance, intersection_count)| {
            if intersection_count % 2 == 0 {
                distance
            } else {
                // can only be odd if in raycast mode
                -distance
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::Itertools;

    #[test]
    fn test_generate() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();
        let query_points = [
            cgmath::Vector3::new(0.0, 0.0, 0.0),
            cgmath::Vector3::new(1.0, 1.0, 1.0),
            cgmath::Vector3::new(0.1, 0.2, 0.2),
        ];
        let sdf = generate_sdf_default(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            SignMethod::Normal,
        );

        // pysdf [0.45216727 -0.6997909   0.45411023] # negative is outside in pysdf
        // mesh_to_sdf [-0.40961263  0.6929414  -0.46345082] # negative is inside in mesh_to_sdf
        let baseline = [-0.42, 0.69, -0.46];

        // make sure the results are close enough.
        // the results are not exactly the same because the algorithm is not the same and baselines might not be exact.
        // this is mostly to make sure the results are not completely off.
        for (sdf, baseline) in sdf.iter().zip(baseline.iter()) {
            assert!((sdf - baseline).abs() < 0.1);
        }
    }
}
