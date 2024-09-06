use std::fmt;

/// The type of primitives to render.
///
/// To find more information for each mode and how to render them check
/// [Khronos Primitive Documentation](https://www.khronos.org/opengl/wiki/Primitive).
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Mode {
    /// Corresponds to `GL_POINTS`.
    Points,
    /// Corresponds to `GL_LINES`.
    Lines,
    /// Corresponds to `GL_LINE_LOOP`.
    LineLoop,
    /// Corresponds to `GL_LINE_STRIP`.
    LineStrip,
    /// Corresponds to `GL_TRIANGLES`.
    #[default]
    Triangles,
    /// Corresponds to `GL_TRIANGLE_STRIP`.
    TriangleStrip,
    /// Corresponds to `GL_TRIANGLE_FAN`.
    TriangleFan,
}

impl From<gltf::mesh::Mode> for Mode {
    fn from(mode: gltf::mesh::Mode) -> Self {
        match mode {
            gltf::mesh::Mode::Points => Self::Points,
            gltf::mesh::Mode::Lines => Self::Lines,
            gltf::mesh::Mode::LineLoop => Self::LineLoop,
            gltf::mesh::Mode::LineStrip => Self::LineStrip,
            gltf::mesh::Mode::Triangles => Self::Triangles,
            gltf::mesh::Mode::TriangleFan => Self::TriangleFan,
            gltf::mesh::Mode::TriangleStrip => Self::TriangleStrip,
        }
    }
}

/// Represents a runtime error. This error is triggered when an expected mode
/// doesn't match the model mode .
#[derive(Clone, Debug)]
pub struct BadMode {
    /// The current mode of the model.
    pub mode: Mode,
}

impl fmt::Display for BadMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid mode \"{:?}\"", self.mode,)
    }
}
