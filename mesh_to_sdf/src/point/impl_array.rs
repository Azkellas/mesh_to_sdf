use super::Point;

impl Point for [f32; 3] {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_aray() {
        let p1 = [1.0, 2.0, 3.0];
        let p2 = [4.0, 5.0, 6.0];

        assert_eq!(p1.add(&p2), [5.0, 7.0, 9.0]);
        assert_eq!(p1.sub(&p2), [-3.0, -3.0, -3.0]);
        assert_eq!(p1.dot(&p2), 32.0);
        assert_eq!(p1.length(), 3.7416575);
        assert_eq!(p1.dist(&p2), 5.196152);
        assert_eq!(p1.mul(2.0), [2.0, 4.0, 6.0]);
        assert_eq!(p1.comp_div(&p2), [0.25, 0.4, 0.5]);
    }
}
