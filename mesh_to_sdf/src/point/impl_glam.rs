use super::Point;

impl Point for glam::Vec3 {
    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self {
        glam::Vec3::new(x, y, z)
    }

    /// Get the x coordinate.
    fn x(&self) -> f32 {
        self.x
    }

    /// Get the y coordinate.
    fn y(&self) -> f32 {
        self.y
    }

    /// Get the z coordinate.
    fn z(&self) -> f32 {
        self.z
    }

    /// Add two points.
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }
    /// Subtract two points.
    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }
    /// Dot product of two points.
    fn dot(&self, other: &Self) -> f32 {
        glam::Vec3::dot(*self, *other)
    }
    /// Cross product of two points.
    fn cross(&self, other: &Self) -> Self {
        glam::Vec3::cross(*self, *other)
    }
    /// Length of the point.
    fn length(&self) -> f32 {
        glam::Vec3::length(*self)
    }
    /// Distance between two points.
    fn dist(&self, other: &Self) -> f32 {
        glam::Vec3::distance(*self, *other)
    }
    /// Multiply a point by a scalar.
    fn fmul(&self, other: f32) -> Self {
        *self * other
    }
    /// Divide two points by components.
    fn comp_div(&self, other: &Self) -> Self {
        *self / *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::proptest;

    proptest! {
        #[test]
        fn test_point_glam(x: f32, y: f32, z: f32) {
            let cmp = |a: glam::Vec3, b: &[f32;3]| {
                if a.x.is_nan() && b[0].is_nan() {
                    return true;
                }
                if a.y.is_nan() && b[1].is_nan() {
                    return true;
                }
                if a.z.is_nan() && b[2].is_nan() {
                    return true;
                }
                float_cmp::approx_eq!(f32, a.x, b[0]) && float_cmp::approx_eq!(f32, a.y, b[1]) && float_cmp::approx_eq!(f32, a.z, b[2])
            };

            let p1 = glam::Vec3::new(x, y, z);
            let p2 = glam::Vec3::new(x, y, z);

            let p3: glam::Vec3 = Point::new(p1.x(), p1.y(), p1.z());
            assert_eq!(p3.x(), x);
            assert_eq!(p3.y(), y);
            assert_eq!(p3.z(), z);

            let ap1 = [x, y, z];
            let ap2 = [x, y, z];
            assert!(cmp(Point::add(&p1, &p2), &ap1.add(&ap2)));
            assert!(cmp(Point::sub(&p1, &p2), &ap1.sub(&ap2)));
            assert!(Point::dot(&p1, &p2) == ap1.dot(&ap2));
            assert!(cmp(Point::cross(&p1, &p2), &ap1.cross(&ap2)));
            assert!(Point::length(&p1) == ap1.length());
            assert!(Point::dist(&p1, &p2) == ap1.dist(&ap2));
            assert!(cmp(Point::fmul(&p1, 2.0), &ap1.fmul(2.0)));
            if ap2.x() != 0.0 && ap2.y() != 0.0 && ap2.z() != 0.0 {
                assert!(cmp(Point::comp_div(&p1, &p2), &ap1.comp_div(&ap2)));
            }
        }
    }
}
