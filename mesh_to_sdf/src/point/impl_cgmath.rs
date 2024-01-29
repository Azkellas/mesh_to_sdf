use cgmath::{ElementWise, InnerSpace, MetricSpace};

use super::Point;

impl Point for cgmath::Vector3<f32> {
    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self {
        cgmath::Vector3::new(x, y, z)
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
        cgmath::InnerSpace::dot(*self, *other)
    }
    /// Length of the point.
    fn length(&self) -> f32 {
        self.magnitude()
    }
    /// Distance between two points.
    fn dist(&self, other: &Self) -> f32 {
        self.distance(*other)
    }
    /// Multiply a point by a scalar.
    fn mul(&self, other: f32) -> Self {
        *self * other
    }
    /// Divide two points by components.
    fn comp_div(&self, other: &Self) -> Self {
        self.div_element_wise(*other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::proptest;

    proptest! {
        #[test]
        fn test_point_cgmath(x: f32, y: f32, z: f32) {
            let cmp = |a: cgmath::Vector3<f32>, b: [f32;3]| {
                a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
            };
            let p1 = cgmath::Vector3::new(x, y, z);
            let p2 = cgmath::Vector3::new(x, y, z);

            let ap1 = [x, y, z];
            let ap2 = [x, y, z];
            assert!(cmp(Point::add(&p1, &p2), ap1.add(&ap2)));
            assert!(cmp(Point::sub(&p1, &p2), ap1.sub(&ap2)));
            assert!(Point::dot(&p1, &p2) == ap1.dot(&ap2));
            assert!(Point::length(&p1) == ap1.length());
            assert!(Point::dist(&p1, &p2) == ap1.dist(&ap2));
            assert!(cmp(Point::mul(&p1, 2.0), ap1.mul(2.0)));
            if ap2[0] != 0.0 && ap2[1] != 0.0 && ap2[2] != 0.0 {
                assert!(cmp(Point::comp_div(&p1, &p2), ap1.comp_div(&ap2)));
            }
        }
    }
}
