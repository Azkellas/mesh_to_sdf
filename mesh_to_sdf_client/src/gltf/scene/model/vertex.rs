use glam::{Vec2, Vec3, Vec4};

/// Represents the 3 vertices of a triangle.
pub type Triangle = [Vertex; 3];

/// Represents the 2 vertices of a line.
pub type Line = [Vertex; 2];

/// Contains a position, normal and texture coordinates vectors.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    /// Position
    pub position: Vec3,
    /// Normalized normal
    pub normal: Vec3,
    /// Tangent normal
    /// The w component is the handedness of the tangent basis (can be -1 or 1)
    pub tangent: Vec4,
    /// Texture coordinates
    pub tex_coords: Vec2,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::ZERO,
            tangent: Vec4::ZERO,
            tex_coords: Vec2::ZERO,
        }
    }
}
