#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Serialize};

mod impl_array;

#[cfg(feature = "cgmath")]
mod impl_cgmath;
#[cfg(feature = "glam")]
mod impl_glam;
#[cfg(feature = "mint")]
mod impl_mint;
//#[cfg(feature = "nalgebra")]
mod impl_nalgebra;

/// Point is the trait that represents a point in 3D space.
/// It is an abstraction over the type of point used in the client math library.
///
/// While everything could be done with new/x/y/z only,
/// the other methods are provided for optimization purposes,
/// relying on the client math library to provide optimized implementations when possible.
pub trait Point: Sized + Copy + Sync + Send {
    /// If the `serde` feature is enabled, the Point should be serializable and deserializable.
    /// You should set Serde to Self:
    /// ```ignore
    /// use mesh_to_sdf::Point;
    /// #[derive(Serialize, Deserialize)]
    /// struct MyPoint {
    ///     x: f32,
    ///     y: f32,
    ///     z: f32,
    /// }
    ///
    /// impl Point for MyPoint {
    ///     #[cfg(feature = "serde")]
    ///    type Serde = Self;
    ///
    ///    // ...
    /// }
    /// ```
    #[cfg(feature = "serde")]
    type Serde: Serialize + DeserializeOwned;

    /// Create a new point.
    fn new(x: f32, y: f32, z: f32) -> Self;

    /// Get the x coordinate.
    fn x(&self) -> f32;
    /// Get the y coordinate.
    fn y(&self) -> f32;
    /// Get the z coordinate.
    fn z(&self) -> f32;

    /// Get the coordinate at index `i`.
    fn get(&self, i: usize) -> f32 {
        match i {
            0 => self.x(),
            1 => self.y(),
            2 => self.z(),
            _ => panic!("Index out of bounds"),
        }
    }

    // Past this point, all methods are optional.
    // You are encouraged to implement them if your math library provides equivalent methods for optimization purposes,
    // but a default implementation is provided that uses new/x/y/z as a fallback.

    /// Compare two points for equality.
    fn eq(&self, other: &Self) -> bool {
        self.x() == other.x() && self.y() == other.y() && self.z() == other.z()
    }

    /// Add two points.
    fn add(&self, other: &Self) -> Self {
        Self::new(
            self.x() + other.x(),
            self.y() + other.y(),
            self.z() + other.z(),
        )
    }
    /// Subtract two points.
    fn sub(&self, other: &Self) -> Self {
        Self::new(
            self.x() - other.x(),
            self.y() - other.y(),
            self.z() - other.z(),
        )
    }
    /// Dot product of two points.
    fn dot(&self, other: &Self) -> f32 {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
    /// Cross product of two points.
    fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        )
    }
    /// Length of the point.
    fn length(&self) -> f32 {
        self.dot(self).sqrt()
    }
    /// Distance between two points.
    fn dist(&self, other: &Self) -> f32 {
        self.sub(other).length()
    }
    /// Multiply a point by a scalar.
    fn fmul(&self, other: f32) -> Self {
        Self::new(self.x() * other, self.y() * other, self.z() * other)
    }
    /// Divide two points by components.
    fn comp_div(&self, other: &Self) -> Self {
        Self::new(
            self.x() / other.x(),
            self.y() / other.y(),
            self.z() / other.z(),
        )
    }
}
