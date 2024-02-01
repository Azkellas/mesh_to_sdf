use crate::point::Point;

/// Compute the bounding box of a triangle.
pub fn triangle_bounding_box<V: Point>(a: &V, b: &V, c: &V) -> (V, V) {
    let min = V::new(
        f32::min(a.x(), f32::min(b.x(), c.x())),
        f32::min(a.y(), f32::min(b.y(), c.y())),
        f32::min(a.z(), f32::min(b.z(), c.z())),
    );
    let max = V::new(
        f32::max(a.x(), f32::max(b.x(), c.x())),
        f32::max(a.y(), f32::max(b.y(), c.y())),
        f32::max(a.z(), f32::max(b.z(), c.z())),
    );
    (min, max)
}

/// Compute the signed distance between a point and a triangle.
/// Sign is positive if the point is outside the mesh, negative if inside.
/// Assume all normals are pointing outside the mesh.
/// Sign is computed by checking if the point is on the same side of the triangle as the normal.
pub fn point_triangle_signed_distance<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
    // Compute the unsigned distance from the point to the plane of the triangle
    let nearest = closest_point_triangle(x0, x1, x2, x3);
    let direction = x0.sub(&nearest);
    let normal = triangle_normal(x1, x2, x3);

    let distance = x0.dist(&nearest);

    if direction.dot(&normal) > 0.0 {
        distance
    } else {
        -distance
    }
}

/// Return the normal, which is NOT normalized.
/// TODO: this might be better with the mesh normals if we add them to the api.
fn triangle_normal<V: Point>(a: &V, b: &V, c: &V) -> V {
    let ab = b.sub(a);
    let ac = c.sub(a);
    V::new(
        ab.y() * ac.z() - ab.z() * ac.y(),
        ab.z() * ac.x() - ab.x() * ac.z(),
        ab.x() * ac.y() - ab.y() * ac.x(),
    )
}

/// Project a point onto a triangle.
/// Adapted from Embree.
/// <https://github.com/embree/embree/blob/master/tutorials/common/math/closest_point.h#L10>
fn closest_point_triangle<V: Point>(p: &V, a: &V, b: &V, c: &V) -> V {
    // Add safety checks for degenerate triangles
    match (a.eq(b), b.eq(c), a.eq(c)) {
        (true, true, true) => {
            return *a;
        }
        (true, _, _) => {
            return closest_point_segment(p, a, c);
        }
        (_, true, _) => {
            return closest_point_segment(p, a, b);
        }
        (_, _, true) => {
            return closest_point_segment(p, a, b);
        }
        // they are all different
        _ => {}
    }

    // Actual embree code.
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ap = p.sub(a);

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }

    let bp = p.sub(b);
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }

    let cp = p.sub(c);
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a.add(&ab.fmul(v));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let v = d2 / (d2 - d6);
        return a.add(&ac.fmul(v));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = c.sub(b);
        return b.add(&bc.fmul(v));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a.add(&ab.fmul(v)).add(&ac.fmul(w))
}

/// Project p on \[ab\].
fn closest_point_segment<V: Point>(p: &V, a: &V, b: &V) -> V {
    let ab = b.sub(a);
    let m = ab.dot(&ab);

    // find parameter value of closest point on segment
    let pb = p.sub(b);
    let mut s12 = ab.dot(&pb) / m;
    s12 = s12.clamp(0.0, 1.0);

    a.add(&ab.fmul(s12))
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;
    use proptest::test_runner::Config;

    proptest! {
        #![proptest_config(Config::with_cases(1000))]

        #[test]
        fn test_methods(
            p in prop::array::uniform3(-10.0f32..10.0),
            a in prop::array::uniform3(-10.0f32..10.0),
            b in prop::array::uniform3(-10.0f32..10.0),
            c in prop::array::uniform3(-10.0f32..10.0))
        {
            use float_cmp::approx_eq;

            let cmp = |a: [f32; 3], b: [f32;3]| {
                approx_eq!(f32, a[0], b[0], ulps = 5, epsilon = 1e-3) ||
                approx_eq!(f32, a[1], b[1], ulps = 5, epsilon = 1e-3) ||
                approx_eq!(f32, a[2], b[2], ulps = 5, epsilon = 1e-3)
            };

            if cmp(a, b) || cmp(a, c) || cmp(b, c) {
                // baseline_point_triangle_distance does not work if the triangle is degenerate
                return Ok(());
            }

            let closest = closest_point_triangle(&p, &a, &b, &c);
            let dist = p.dist(&closest);
            let baseline_dist = baseline_point_triangle_distance(&p, &a, &b, &c);

            println!("p: {:?}, a: {:?}, b: {:?}, c: {:?} - {} {}", p, a, b, c, dist, baseline_dist);
            assert!(!dist.is_nan());
            assert!(float_cmp::approx_eq!(f32, dist, baseline_dist, ulps = 5, epsilon = 1e-3));
        }
    }

    /// Find the distance x0 is from triangle x1-x2-x3.
    /// Note: this is adapted from <https://github.com/christopherbatty/SDFGen>
    /// It is slower than the Embree version, but we can use it as a baseline to test the Embree version.
    /// This version does not work for degenerate triangles.
    fn baseline_point_triangle_distance<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
        // first find barycentric coordinates of closest point on infinite plane
        let x03 = x0.sub(x3);
        let x13 = x1.sub(x3);
        let x23 = x2.sub(x3);

        let m13 = x13.dot(&x13);
        let m23 = x23.dot(&x23);

        let d = x13.dot(&x23);

        let invdet = 1.0 / f32::max(m13 * m23 - d * d, 1e-30);

        let a = x13.dot(&x03);
        let b = x23.dot(&x03);

        // the barycentric coordinates themselves
        let w23 = invdet * (m23 * a - d * b);
        let w31 = invdet * (m13 * b - d * a);
        let w12 = 1.0 - w23 - w31;

        if w23 >= 0.0 && w31 >= 0.0 && w12 >= 0.0 {
            // if we're inside the triangle
            let x1p = x1.fmul(w23);
            let x2p = x2.fmul(w31);
            let x3p = x3.fmul(w12);
            let projected = x1p.add(&x2p).add(&x3p);
            x0.dist(&projected)
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

    /// Find the distance x0 is from segment x1-x2.
    fn point_segment_distance<V: Point>(x0: &V, x1: &V, x2: &V) -> f32 {
        let dx = x2.sub(x1);
        let m2 = dx.dot(&dx);

        // find parameter value of closest point on segment
        let mut s12 = dx.dot(&x2.sub(x0)) / m2;
        s12 = s12.clamp(0.0, 1.0);

        // and find the distance
        x0.dist(&x1.fmul(s12).add(&x2.fmul(1.0 - s12)))
    }
}
