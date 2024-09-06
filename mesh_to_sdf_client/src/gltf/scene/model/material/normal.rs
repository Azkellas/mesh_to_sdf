use super::GltfData;
use image::RgbImage;
use std::sync::Arc;

#[derive(Clone, Debug)]
/// Defines the normal texture of a material.
pub struct NormalMap {
    /// A tangent space normal map.
    /// The texture contains RGB components in linear space. Each texel
    /// represents the XYZ components of a normal vector in tangent space.
    ///
    /// * Red [0 to 255] maps to X [-1 to 1].
    /// * Green [0 to 255] maps to Y [-1 to 1].
    /// * Blue [128 to 255] maps to Z [1/255 to 1].
    ///
    /// The normal vectors use OpenGL conventions where +X is right, +Y is up,
    /// and +Z points toward the viewer.
    pub texture: Arc<RgbImage>,

    /// The `normal_factor` is the normal strength to be applied to the
    /// texture value.
    pub factor: f32,
}

impl NormalMap {
    pub(crate) fn load(gltf_mat: &gltf::Material, data: &GltfData) -> Option<Self> {
        gltf_mat.normal_texture().map(|texture| Self {
            texture: data.get_rgb_image(&texture.texture()),
            factor: texture.scale(),
        })
    }
}
