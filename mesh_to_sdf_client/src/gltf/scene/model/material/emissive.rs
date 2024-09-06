use glam::*;
use image::RgbImage;
use std::sync::Arc;

use crate::gltf::GltfData;

#[derive(Clone, Debug)]
/// The emissive color of the material.
pub struct Emissive {
    /// The `emissive_texture` refers to a texture that may be used to illuminate parts of the
    /// model surface: It defines the color of the light that is emitted from the surface
    pub texture: Option<Arc<RgbImage>>,

    /// The `emissive_factor` contains scaling factors for the red, green and
    /// blue components of this texture.
    pub factor: Vec3,
}

impl Emissive {
    pub(crate) fn load(gltf_mat: &gltf::Material, data: &GltfData) -> Self {
        Self {
            texture: gltf_mat
                .emissive_texture()
                .map(|texture| data.get_rgb_image(&texture.texture())),
            factor: gltf_mat.emissive_factor().into(),
        }
    }
}

impl Default for Emissive {
    fn default() -> Self {
        Self {
            texture: None,
            factor: Vec3::ZERO,
        }
    }
}
