mod gltf_data;

use std::sync::{Arc, RwLock};

use glam::Mat4;
pub use gltf_data::GltfData;

use gltf::scene::Transform;
use image::GrayImage;
use itertools::Itertools;
use rayon::prelude::*;

pub fn transform_to_matrix(transform: Transform) -> Mat4 {
    Mat4::from_cols_array_2d(&transform.matrix())
}

/// Get all rgb images from the glTF document.
/// rgb textures are:
/// - normal textures
/// - emissive textures
pub fn get_rgb_textures(doc: &gltf::Document) -> Vec<gltf::Texture> {
    let normal_textures = doc
        .materials()
        .filter_map(|mat| mat.normal_texture())
        .map(|normal| normal.texture());
    let emissive_textures = doc
        .materials()
        .filter_map(|mat| mat.emissive_texture())
        .map(|emissive| emissive.texture());

    // append normal and emissives
    normal_textures
        .chain(emissive_textures)
        .unique_by(gltf::Texture::index)
        .collect()
}

/// Load all rgb images from the glTF document.
pub fn load_rgb_images(doc: &gltf::Document, data: &Arc<RwLock<GltfData>>) {
    // Load rgb images
    get_rgb_textures(doc).par_iter().for_each(|tex| {
        let texture = {
            let data = data.read().unwrap();
            data.load_texture(tex)
        };

        let texture = Arc::new(texture.to_rgb8());
        data.write()
            .unwrap()
            .rgb_images
            .insert(tex.index(), texture);
    });
}

/// Get all rgba images from the glTF document.
/// rgba textures are:
/// - base color textures
pub fn get_rgba_textures(doc: &gltf::Document) -> Vec<gltf::Texture> {
    doc.materials()
        .filter_map(|mat| mat.pbr_metallic_roughness().base_color_texture())
        .map(|info| info.texture())
        .collect()
}

/// Load all rgba images from the glTF document.
pub fn load_rbga_images(doc: &gltf::Document, data: &Arc<RwLock<GltfData>>) {
    // Load rgb images
    get_rgba_textures(doc).par_iter().for_each(|tex| {
        let texture = {
            let data = data.read().unwrap();
            data.load_texture(tex)
        };

        let texture = Arc::new(texture.to_rgba8());
        data.write()
            .unwrap()
            .rgba_images
            .insert(tex.index(), texture);
    });
}

/// Get all rgba images from the glTF document.
/// rgba textures are:
/// - occlusion textures
/// - metallic roughness textures
pub fn get_gray_textures(doc: &gltf::Document) -> Vec<gltf::Texture> {
    let occlusion_textures = doc
        .materials()
        .filter_map(|mat| mat.occlusion_texture())
        .map(|normal| normal.texture());
    let metallic_roughness_textures = doc
        .materials()
        .filter_map(|mat| mat.pbr_metallic_roughness().metallic_roughness_texture())
        .map(|emissive| emissive.texture());

    // Only keep unique textures indices.
    occlusion_textures
        .chain(metallic_roughness_textures)
        .unique_by(gltf::Texture::index)
        .collect()
}

/// Load all rgba images from the glTF document.
pub fn load_gray_images(doc: &gltf::Document, data: &Arc<RwLock<GltfData>>) {
    // Load rgb images
    get_gray_textures(doc).par_iter().for_each(|tex| {
        let texture = {
            let data = data.read().unwrap();
            data.load_texture(tex)
        };

        let img = texture.to_rgba8();
        for channel in 0..4 {
            let mut extract_img = GrayImage::new(img.width(), img.height());
            for (x, y, px) in img.enumerate_pixels() {
                extract_img[(x, y)][0] = px[channel];
            }

            let img = Arc::new(extract_img);

            data.write()
                .unwrap()
                .gray_images
                .insert((tex.index(), channel), img);
        }
    });
}

/// Load all images from the glTF document.
pub fn load_all_images(doc: &gltf::Document, data: &Arc<RwLock<GltfData>>) {
    load_rgb_images(doc, data);
    load_rbga_images(doc, data);
    load_gray_images(doc, data);
}
