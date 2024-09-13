use super::Point;

impl Point for mint::Point3<f32> {
    #[cfg(feature = "serde")]
    type Serde = Self;

    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self {
        mint::Point3 { x, y, z }
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

    fn x_mut(&mut self) -> &mut f32 {
        &mut self.x
    }

    fn y_mut(&mut self) -> &mut f32 {
        &mut self.y
    }

    fn z_mut(&mut self) -> &mut f32 {
        &mut self.z
    }
}

impl Point for mint::Vector3<f32> {
    #[cfg(feature = "serde")]
    type Serde = Self;

    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self {
        mint::Vector3 { x, y, z }
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

    fn x_mut(&mut self) -> &mut f32 {
        &mut self.x
    }

    fn y_mut(&mut self) -> &mut f32 {
        &mut self.y
    }

    fn z_mut(&mut self) -> &mut f32 {
        &mut self.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_point_mint_point3(
            p1 in prop::array::uniform3(-100.0f32..100.0),
            p2 in prop::array::uniform3(-100.0f32..100.0),

        ) {
            let cmp = |a: mint::Point3<f32>, b: [f32;3]| {
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

            let p1 = mint::Point3::new(p1[0], p1[1], p1[2]);
            let p2 = mint::Point3::new(p2[0], p2[1], p2[2]);

            let p3: mint::Point3<f32> = Point::new(p1.x(), p1.y(), p1.z());
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

    proptest! {
        #[test]
        fn test_point_mint_vector3(
            p1 in prop::array::uniform3(-100.0f32..100.0),
            p2 in prop::array::uniform3(-100.0f32..100.0),

        ) {
            let cmp = |a: mint::Vector3<f32>, b: [f32;3]| {
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

            let p1 = mint::Vector3::new(p1[0], p1[1], p1[2]);
            let p2 = mint::Vector3::new(p2[0], p2[1], p2[2]);

            let p3: mint::Vector3<f32> = Point::new(p1.x(), p1.y(), p1.z());
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
