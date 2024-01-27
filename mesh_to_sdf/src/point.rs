mod impl_array;

/// Point is the trait that represents a point in 3D space.
/// It is an abstraction over the type of point used in the client math library.
/// While everything could be done with new/x/y/z only,
/// the other methods are provided for optimization purposes,
/// relying on the client math library to provide optimized implementations when possible.
pub trait Point: Sized + Copy + Sync + Send {
    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self;

    /// Get the x coordinate.
    fn x(&self) -> f32;
    /// Get the y coordinate.
    fn y(&self) -> f32;
    /// Get the z coordinate.
    fn z(&self) -> f32;

    /// Add two points.
    fn add(&self, other: &Self) -> Self;
    /// Subtract two points.
    fn sub(&self, other: &Self) -> Self;
    /// Dot product of two points.
    fn dot(&self, other: &Self) -> f32;
    /// Length of the point.
    fn length(&self) -> f32;
    /// Distance between two points.
    fn dist(&self, other: &Self) -> f32;
    /// Multiply a point by a scalar.
    fn mul(&self, other: f32) -> Self;
    /// Divide two points by components.
    fn comp_div(&self, other: &Self) -> Self;
}
