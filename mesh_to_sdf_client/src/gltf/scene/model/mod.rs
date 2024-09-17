use glam::{Vec2, Vec3, Vec4};
use mode::BadMode;
use vertex::{Line, Triangle, Vertex};

use crate::gltf::GltfData;

mod material;
mod mode;
mod vertex;

pub use material::*;
pub use mode::Mode;

/// Geometry to be rendered with the given material.
///
/// # Examples
///
/// ### Classic rendering
///
/// In most cases you want to use `triangles()`, `lines()` and `points()`
/// to get the geometry of the model.
///
/// ```
/// # use easy_gltf::*;
/// # use easy_gltf::model::Mode;
/// # let model = Model::default();
/// match model.mode() {
///   Mode::Triangles | Mode::TriangleFan | Mode::TriangleStrip => {
///     let triangles = model.triangles().unwrap();
///     // Render triangles...
///   },
///   Mode::Lines | Mode::LineLoop | Mode::LineStrip => {
///     let lines = model.lines().unwrap();
///     // Render lines...
///   }
///   Mode::Points => {
///     let points = model.points().unwrap();
///     // Render points...
///   }
/// }
/// ```
///
/// ### OpenGL style rendering
///
/// You will need the vertices and the indices if existing.
///
/// ```
/// # use easy_gltf::*;
/// # use easy_gltf::model::Mode;
/// # let model = Model::default();
/// let vertices = model. vertices();
/// let indices = model.indices();
/// match model.mode() {
///   Mode::Triangles => {
///     if let Some(indices) = indices.as_ref() {
///       // glDrawElements(GL_TRIANGLES, indices.len(), GL_UNSIGNED_INT, 0);
///     } else {
///       // glDrawArrays(GL_TRIANGLES, 0, vertices.len());
///     }
///   },
///   // ...
/// # _ => unimplemented!(),
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct GltfModel {
    pub(crate) mesh_name: Option<String>,
    pub(crate) mesh_extras: gltf::json::extras::Extras,
    pub(crate) primitive_extras: gltf::json::extras::Extras,

    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Option<Vec<u32>>,
    pub(crate) mode: Mode,
    pub(crate) material_index: Option<usize>,
    pub(crate) has_normals: bool,
    pub(crate) has_tangents: bool,
    pub(crate) has_tex_coords: bool,
}

impl GltfModel {
    /// Mesh name. Requires the `names` feature.
    ///
    /// A `Model` in easy-gltf represents a primitive in gltf, so if a mesh has multiple primitives, you will
    /// get multiple `Model`s with the same name.
    pub fn mesh_name(&self) -> Option<&str> {
        self.mesh_name.as_deref()
    }

    /// Mesh extra data. Requires the `extras` feature.
    pub const fn mesh_extras(&self) -> &gltf::json::extras::Extras {
        &self.mesh_extras
    }

    /// Primitive extra data. Requires the `extras` feature.
    pub const fn primitive_extras(&self) -> &gltf::json::extras::Extras {
        &self.primitive_extras
    }

    /// Material to apply to the whole model.
    pub const fn material_index(&self) -> Option<usize> {
        self.material_index
    }

    /// List of raw `vertices` of the model. You might have to use the `indices`
    /// to render the model.
    ///
    /// **Note**: If you're not rendering with **OpenGL** you probably want to use
    /// `triangles()`, `lines()` or `points()` instead.
    pub const fn vertices(&self) -> &Vec<Vertex> {
        &self.vertices
    }

    /// Potential list of `indices` to render the model using raw `vertices`.
    ///
    /// **Note**: If you're **not** rendering with **OpenGL** you probably want to use
    /// `triangles()`, `lines()` or `points()` instead.
    pub const fn indices(&self) -> Option<&Vec<u32>> {
        self.indices.as_ref()
    }

    /// The type of primitive to render.
    /// You have to check the `mode` to render the model correctly.
    ///
    /// Then you can either use:
    /// * `vertices()` and `indices()` to arrange the data yourself (useful for **OpenGL**).
    /// * `triangles()` or `lines()` or `points()` according to the returned mode.
    pub fn mode(&self) -> Mode {
        self.mode.clone()
    }

    /// List of triangles ready to be rendered.
    ///
    /// **Note**: This function will return an error if the mode isn't `Triangles`, `TriangleFan`
    /// or `TriangleStrip`.
    pub fn triangles(&self) -> Result<Vec<Triangle>, BadMode> {
        let mut triangles = vec![];
        let indices = (0..self.vertices.len() as u32).collect();
        let indices = self.indices().unwrap_or(&indices);

        #[expect(clippy::wildcard_enum_match_arm)]
        match self.mode {
            Mode::Triangles => {
                for i in (0..indices.len()).step_by(3) {
                    triangles.push([
                        self.vertices[indices[i] as usize],
                        self.vertices[indices[i + 1] as usize],
                        self.vertices[indices[i + 2] as usize],
                    ]);
                }
            }
            Mode::TriangleStrip => {
                for i in 0..(indices.len() - 2) {
                    triangles.push([
                        self.vertices[indices[i] as usize + i % 2],
                        self.vertices[indices[i + 1 - i % 2] as usize],
                        self.vertices[indices[i + 2] as usize],
                    ]);
                }
            }
            Mode::TriangleFan => {
                for i in 1..(indices.len() - 1) {
                    triangles.push([
                        self.vertices[indices[0] as usize],
                        self.vertices[indices[i] as usize],
                        self.vertices[indices[i + 1] as usize],
                    ]);
                }
            }
            _ => return Err(BadMode { mode: self.mode() }),
        }
        Ok(triangles)
    }

    /// List of lines ready to be rendered.
    ///
    /// **Note**: This function will return an error if the mode isn't `Lines`, `LineLoop`
    /// or `LineStrip`.
    pub fn lines(&self) -> Result<Vec<Line>, BadMode> {
        let mut lines = vec![];
        let indices = (0..self.vertices.len() as u32).collect();
        let indices = self.indices().unwrap_or(&indices);
        #[expect(clippy::wildcard_enum_match_arm)]
        match self.mode {
            Mode::Lines => {
                for i in (0..indices.len()).step_by(2) {
                    lines.push([
                        self.vertices[indices[i] as usize],
                        self.vertices[indices[i + 1] as usize],
                    ]);
                }
            }
            Mode::LineStrip | Mode::LineLoop => {
                for i in 0..(indices.len() - 1) {
                    lines.push([
                        self.vertices[indices[i] as usize],
                        self.vertices[indices[i + 1] as usize],
                    ]);
                }
            }
            _ => return Err(BadMode { mode: self.mode() }),
        }
        if self.mode == Mode::LineLoop {
            lines.push([
                self.vertices[indices[0] as usize],
                self.vertices[indices[indices.len() - 1] as usize],
            ]);
        }

        Ok(lines)
    }

    /// List of points ready to be renderer.
    ///
    /// **Note**: This function will return an error if the mode isn't `Points`.
    pub fn points(&self) -> Result<&Vec<Vertex>, BadMode> {
        if self.mode == Mode::Points {
            Ok(&self.vertices)
        } else {
            Err(BadMode { mode: self.mode() })
        }
    }

    /// Indicate if the vertices contains normal information.
    ///
    /// **Note**: If this function return `false` all vertices has a normal field
    /// initialized to `zero`.
    pub const fn has_normals(&self) -> bool {
        self.has_normals
    }

    /// Indicate if the vertices contains tangents information.
    ///
    /// **Note**: If this function return `false` all vertices has a tangent field
    /// initialized to `zero`.
    pub const fn has_tangents(&self) -> bool {
        self.has_tangents
    }

    /// Indicate if the vertices contains texture coordinates information.
    ///
    /// **Note**: If this function return `false` all vertices has a `tex_coord` field
    /// initialized to `zero`.
    pub const fn has_tex_coords(&self) -> bool {
        self.has_tex_coords
    }

    pub(crate) fn load(mesh: &gltf::Mesh, primitive: &gltf::Primitive, data: &GltfData) -> Self {
        let buffers = &data.buffers;
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let indices = reader
            .read_indices()
            .map(|indices| indices.into_u32().collect());

        // Init vertices with the position
        let mut vertices: Vec<_> = reader
            .read_positions()
            .unwrap_or_else(|| panic!("The model primitive doesn't contain positions"))
            .map(|pos| Vertex {
                position: Vec3::from(pos),
                ..Default::default()
            })
            .collect();

        // Fill normals
        let has_normals = reader.read_normals().map_or(false, |normals| {
            for (i, normal) in normals.enumerate() {
                vertices[i].normal = Vec3::from(normal).normalize();
            }
            true
        });

        // Fill tangents
        let has_tangents = reader.read_tangents().map_or(false, |tangents| {
            for (i, tangent) in tangents.enumerate() {
                let tangent = Vec4::from(tangent);
                vertices[i].tangent = tangent.truncate().normalize().extend(tangent.w);
            }
            true
        });

        // Texture coordinates
        let has_tex_coords = reader.read_tex_coords(0).map_or(false, |tex_coords| {
            for (i, tex_coords) in tex_coords.into_f32().enumerate() {
                vertices[i].tex_coords = Vec2::from(tex_coords);
            }
            true
        });

        Self {
            mesh_name: mesh.name().map(String::from),
            mesh_extras: mesh.extras().clone(),
            primitive_extras: primitive.extras().clone(),
            vertices,
            indices,
            material_index: primitive.material().index(),
            mode: primitive.mode().into(),
            has_normals,
            has_tangents,
            has_tex_coords,
        }
    }
}
