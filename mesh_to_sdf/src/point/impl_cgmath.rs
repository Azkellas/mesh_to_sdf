use cgmath::{ElementWise, InnerSpace, MetricSpace};

use super::Point;

impl Point for cgmath::Vector3<f32> {
    #[cfg(feature = "serde")]
    type Serde = Self;

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
    /// Cross product of two points.
    fn cross(&self, other: &Self) -> Self {
        cgmath::Vector3::cross(*self, *other)
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
    fn fmul(&self, other: f32) -> Self {
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
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_point_cgmath_vector3(
            p1 in prop::array::uniform3(-100.0f32..100.0),
            p2 in prop::array::uniform3(-100.0f32..100.0),

        ) {
            let cmp = |a: cgmath::Vector3<f32>, b: [f32;3]| {
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

            let ap1 = p1;
            let ap2 = p2;

            let p1 = cgmath::Vector3::new(p1[0], p1[1], p1[2]);
            let p2 = cgmath::Vector3::new(p2[0], p2[1], p2[2]);

            let p3: cgmath::Vector3<f32> = Point::new(p1.x(), p1.y(), p1.z());
            assert_eq!(p3.x(), ap1[0]);
            assert_eq!(p3.y(), ap1[1]);
            assert_eq!(p3.z(), ap1[2]);

            assert!(cmp(Point::add(&p1, &p2), ap1.add(&ap2)));
            assert!(cmp(Point::sub(&p1, &p2), ap1.sub(&ap2)));
            assert!(Point::dot(&p1, &p2) == ap1.dot(&ap2));
            assert!(cmp(Point::cross(&p1, &p2), ap1.cross(&ap2)));
            assert!(Point::length(&p1) == ap1.length());
            assert!(Point::dist(&p1, &p2) == ap1.dist(&ap2));
            assert!(cmp(Point::fmul(&p1, 2.0), ap1.fmul(2.0)));
            if ap2[0] != 0.0 && ap2[1] != 0.0 && ap2[2] != 0.0 {
                assert!(cmp(Point::comp_div(&p1, &p2), ap1.comp_div(&ap2)));
            }
        }
    }
}
