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

    fn add(&self, other: &Self) -> Self {
        [self[0] + other[0], self[1] + other[1], self[2] + other[2]]
    }

    fn sub(&self, other: &Self) -> Self {
        [self[0] - other[0], self[1] - other[1], self[2] - other[2]]
    }

    fn dot(&self, other: &Self) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    fn length(&self) -> f32 {
        self.dot(self).sqrt()
    }

    fn dist(&self, other: &Self) -> f32 {
        self.sub(other).length()
    }

    fn mul(&self, other: f32) -> Self {
        [self[0] * other, self[1] * other, self[2] * other]
    }

    fn comp_div(&self, other: &Self) -> Self {
        [self[0] / other[0], self[1] / other[1], self[2] / other[2]]
    }
}
