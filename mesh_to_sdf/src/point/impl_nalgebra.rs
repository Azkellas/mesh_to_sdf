use super::Point;

impl Point for nalgebra::Point3<f32> {
    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self {
        nalgebra::Point3::new(x, y, z)
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
}

impl Point for nalgebra::Vector3<f32> {
    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self {
        nalgebra::Vector3::new(x, y, z)
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
        self + other
    }
    /// Subtract two points.
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    /// Dot product of two points.
    fn dot(&self, other: &Self) -> f32 {
        self.dot(other)
    }
    /// Length of the point.
    fn length(&self) -> f32 {
        self.norm()
    }
    /// Distance between two points.
    fn dist(&self, other: &Self) -> f32 {
        (self - other).norm()
    }
    /// Multiply a point by a scalar.
    fn fmul(&self, other: f32) -> Self {
        self * other
    }
    /// Divide two points by components.
    fn comp_div(&self, other: &Self) -> Self {
        self.component_div(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::proptest;

    proptest! {
        #[test]
        fn test_point_nalgebra_point3(x: f32, y: f32, z: f32) {
            let cmp = |a: nalgebra::Point3<f32>, b: [f32;3]| {
                a.x == b[0] && a.y == b[1] && a.z == b[2]
            };
            let p1 = nalgebra::Point3::new(x, y, z);
            let p2 = nalgebra::Point3::new(x, y, z);

            let p3: nalgebra::Point3<f32> = Point::new(p1.x(), p1.y(), p1.z());
            assert_eq!(p3.x(), x);
            assert_eq!(p3.y(), y);
            assert_eq!(p3.z(), z);

            let ap1 = [x, y, z];
            let ap2 = [x, y, z];
            assert!(cmp(Point::add(&p1, &p2), ap1.add(&ap2)));
            assert!(cmp(Point::sub(&p1, &p2), ap1.sub(&ap2)));
            assert!(Point::dot(&p1, &p2) == ap1.dot(&ap2));
            assert!(Point::length(&p1) == ap1.length());
            assert!(Point::dist(&p1, &p2) == ap1.dist(&ap2));
            assert!(cmp(Point::fmul(&p1, 2.0), ap1.fmul(2.0)));
            if ap2[0] != 0.0 && ap2[1] != 0.0 && ap2[2] != 0.0 {
                assert!(cmp(Point::comp_div(&p1, &p2), ap1.comp_div(&ap2)));
            }
        }
    }

    proptest! {
        #[test]
        fn test_point_nalgebra_vector3(x: f32, y: f32, z: f32) {
            let cmp = |a: nalgebra::Vector3<f32>, b: [f32;3]| {
                a.x == b[0] && a.y == b[1] && a.z == b[2]
            };
            let p1 = nalgebra::Vector3::new(x, y, z);
            let p2 = nalgebra::Vector3::new(x, y, z);

            let p3: nalgebra::Vector3<f32> = Point::new(p1.x(), p1.y(), p1.z());
            assert_eq!(p3.x(), x);
            assert_eq!(p3.y(), y);
            assert_eq!(p3.z(), z);

            let ap1 = [x, y, z];
            let ap2 = [x, y, z];
            assert!(cmp(Point::add(&p1, &p2), ap1.add(&ap2)));
            assert!(cmp(Point::sub(&p1, &p2), ap1.sub(&ap2)));
            assert!(Point::dot(&p1, &p2) == ap1.dot(&ap2));
            assert!(Point::length(&p1) == ap1.length());
            assert!(Point::dist(&p1, &p2) == ap1.dist(&ap2));
            assert!(cmp(Point::fmul(&p1, 2.0), ap1.fmul(2.0)));
            if ap2[0] != 0.0 && ap2[1] != 0.0 && ap2[2] != 0.0 {
                assert!(cmp(Point::comp_div(&p1, &p2), ap1.comp_div(&ap2)));
            }
        }
    }
}
