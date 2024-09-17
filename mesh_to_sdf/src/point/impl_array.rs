use super::Point;

impl Point for [f32; 3] {
    #[cfg(feature = "serde")]
    type Serde = Self;

    fn new(x: f32, y: f32, z: f32) -> Self {
        [x, y, z]
    }

    fn x(&self) -> f32 {
        self[0]
    }

    fn y(&self) -> f32 {
        self[1]
    }

    fn z(&self) -> f32 {
        self[2]
    }

    fn x_mut(&mut self) -> &mut f32 {
        &mut self[0]
    }

    fn y_mut(&mut self) -> &mut f32 {
        &mut self[1]
    }

    fn z_mut(&mut self) -> &mut f32 {
        &mut self[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_aray() {
        let p1 = [1.0, 2.0, 3.0];
        let p2 = [4.0, 5.0, 6.0];

        let p3: [f32; 3] = Point::new(p1.x(), p1.y(), p1.z());
        assert_eq!(p3.x(), 1.0);
        assert_eq!(p3.y(), 2.0);
        assert_eq!(p3.z(), 3.0);

        assert_eq!(p1.add(&p2), [5.0, 7.0, 9.0]);
        assert_eq!(p1.sub(&p2), [-3.0, -3.0, -3.0]);
        assert_eq!(p1.dot(&p2), 32.0);
        assert_eq!(p1.length(), 3.7416575);
        assert_eq!(p1.dist(&p2), 5.196152);
        assert_eq!(p1.fmul(2.0), [2.0, 4.0, 6.0]);
        assert_eq!(p1.comp_div(&p2), [0.25, 0.4, 0.5]);

        let mut p = p1;
        *p.x_mut() = 10.0;
        *p.y_mut() = 20.0;
        *p.z_mut() = 30.0;
        assert_eq!(p, [10.0, 20.0, 30.0]);
    }
}
