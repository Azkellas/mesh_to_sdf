use crate::point::Point;

pub fn point_triangle_signed_distance<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
    let mut distance = point_triangle_distance(x0, x1, x2, x3);

    // signed distance: positive if the point is outside the mesh, negative if inside.
    // assume all normals are pointing outside the mesh.
    let barycenter = triangle_barycenter(x1, x2, x3);
    let direction = x0.sub(&barycenter);
    // No need for it to be normalized.
    let normal = triangle_normal(x1, x2, x3);
    if direction.dot(&normal) < 0.0 {
        distance = -distance;
    }

    distance
}

fn triangle_barycenter<V: Point>(a: &V, b: &V, c: &V) -> V {
    (a.add(b).add(c)).mul(1.0 / 3.0)
}

/// Note: the normal is NOT normalized.
fn triangle_normal<V: Point>(a: &V, b: &V, c: &V) -> V {
    let ab = b.sub(a);
    let ac = c.sub(a);
    V::new(
        ab.y() * ac.z() - ab.z() * ac.y(),
        ab.z() * ac.x() - ab.x() * ac.z(),
        ab.x() * ac.y() - ab.y() * ac.x(),
    )
}

/// Note: this is adapted from https://github.com/christopherbatty/SDFGen
// find distance x0 is from triangle x1-x2-x3
fn point_triangle_distance<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
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
        let x1p = x1.mul(w23);
        let x2p = x2.mul(w31);
        let x3p = x3.mul(w12);
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

// find distance x0 is from segment x1-x2
pub fn point_segment_distance<V: Point>(x0: &V, x1: &V, x2: &V) -> f32 {
    let dx = x2.sub(x1);
    let m2 = dx.dot(&dx);

    // find parameter value of closest point on segment
    let mut s12 = dx.dot(&x2.sub(x0)) / m2;
    s12 = s12.clamp(0.0, 1.0);

    // and find the distance
    x0.dist(&x1.mul(s12).add(&x2.mul(1.0 - s12)))
}

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
