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
