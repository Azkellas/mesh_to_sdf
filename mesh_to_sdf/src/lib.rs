use itertools::Itertools;
use rayon::prelude::*;

use std::boxed::Box;

/// Mesh Topology
pub enum Topology<'a, T>
where
    // T should be a u32 or u16
    T: Into<u32>,
{
    /// Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
    TriangleList(Option<&'a [T]>),
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create four triangles `0 1 2`, `2 1 3`, `2 3 4`, and `4 3 5`
    TriangleStrip(Option<&'a [T]>),
}

/// Generate a signed distance field from a mesh.
/// Compute the signed distance for each query point.
pub fn generate_sdf<T>(
    vertices: &[[f32; 3]],
    indices: Topology<T>,
    query_points: &[[f32; 3]],
) -> Vec<f32>
where
    T: Copy + Into<u32> + Sync + Send,
{
    query_points
        .par_iter()
        .map(|query| {
            //TODO: handle triangle strips
            let triangles: Box<dyn Iterator<Item = usize>> = match indices {
                Topology::TriangleList(Some(triangles)) => {
                    Box::new(triangles.iter().map(|x| (*x).into() as usize))
                }
                Topology::TriangleList(None) => Box::new((0..vertices.len()).into_iter()),
                Topology::TriangleStrip(_) => todo!(),
            };

            triangles
                .tuples()
                .map(|(ia, ib, ic)| {
                    let a: &[f32; 3] = &vertices[ia as usize];
                    let b: &[f32; 3] = &vertices[ib as usize];
                    let c: &[f32; 3] = &vertices[ic as usize];

                    // No need for it to be normalized.
                    let normal = [
                        (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
                        (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
                    ];

                    let barycenter = [
                        (a[0] + b[0] + c[0]) / 3.0,
                        (a[1] + b[1] + c[1]) / 3.0,
                        (a[2] + b[2] + c[2]) / 3.0,
                    ];

                    let direction = sub(query, &barycenter);

                    // unsigned distance.
                    let mut distance = point_triangle_distance(query, a, b, c);

                    // signed distance: positive if the point is outside the mesh, negative if inside.
                    // assume all normals are pointing outside the mesh.
                    if dot(&direction, &normal) < 0.0 {
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

/// Note: this is adapted from https://github.com/christopherbatty/SDFGen
// find distance x0 is from triangle x1-x2-x3
fn point_triangle_distance(x0: &[f32; 3], x1: &[f32; 3], x2: &[f32; 3], x3: &[f32; 3]) -> f32 {
    // first find barycentric coordinates of closest point on infinite plane
    let x13 = sub(x1, x3);
    let x23 = sub(x2, x3);
    let x03 = sub(x0, x3);

    let m13 = dot(&x13, &x13);
    let m23 = dot(&x23, &x23);

    let d = dot(&x13, &x23);

    let invdet = 1.0 / f32::max(m13 * m23 - d * d, 1e-30);

    let a = dot(&x13, &x03);
    let b = dot(&x23, &x03);

    // the barycentric coordinates themselves
    let w23 = invdet * (m23 * a - d * b);
    let w31 = invdet * (m13 * b - d * a);
    let w12 = 1.0 - w23 - w31;

    if w23 >= 0.0 && w31 >= 0.0 && w12 >= 0.0 {
        // if we're inside the triangle
        let projected = [
            w23 * x1[0] + w31 * x2[0] + w12 * x3[0],
            w23 * x1[1] + w31 * x2[1] + w12 * x3[1],
            w23 * x1[2] + w31 * x2[2] + w12 * x3[2],
        ];

        dist(x0, &projected)
    } else {
        // we have to clamp to one of the edges
        if w23 > 0.0 {
            // this rules out edge 2-3 for us
            f32::min(
                point_segment_distance(x0, x1, x2),
                point_segment_distance(x0, x1, x3),
            )
        } else if w31 > 0.0 {
            // this rules out edge 1-3
            f32::min(
                point_segment_distance(x0, x1, x2),
                point_segment_distance(x0, x2, x3),
            )
        } else {
            // w12 must be >0, ruling out edge 1-2
            f32::min(
                point_segment_distance(x0, x1, x3),
                point_segment_distance(x0, x2, x3),
            )
        }
    }
}

// find distance x0 is from segment x1-x2
fn point_segment_distance(x0: &[f32; 3], x1: &[f32; 3], x2: &[f32; 3]) -> f32 {
    let dx = sub(x2, x1);
    let m2 = dot(&dx, &dx);

    // find parameter value of closest point on segment
    let mut s12 = dot(&sub(x2, x0), &dx) / m2;
    s12 = s12.clamp(0.0, 1.0);

    // and find the distance
    dist(x0, &add(&fmul(x1, s12), &fmul(x2, 1.0 - s12)))
}

fn add(x: &[f32; 3], y: &[f32; 3]) -> [f32; 3] {
    [x[0] + y[0], x[1] + y[1], x[2] + y[2]]
}

fn sub(x: &[f32; 3], y: &[f32; 3]) -> [f32; 3] {
    [x[0] - y[0], x[1] - y[1], x[2] - y[2]]
}

fn dot(x: &[f32; 3], y: &[f32; 3]) -> f32 {
    x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
}

fn length(x: &[f32; 3]) -> f32 {
    dot(x, x).sqrt()
}

fn dist(x: &[f32; 3], y: &[f32; 3]) -> f32 {
    length(&sub(x, y))
}

fn fmul(x: &[f32; 3], y: f32) -> [f32; 3] {
    [x[0] * y, x[1] * y, x[2] * y]
}
