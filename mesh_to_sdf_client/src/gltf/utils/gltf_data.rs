use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use gltf::image::Source;
use hashbrown::HashMap;
use image::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gltf::scene::GltfModel;
use crate::gltf::Material;

/// Helps to simplify the signature of import related functions.
#[derive(Debug, Clone)]
pub struct GltfData {
    /// Buffers of the glTF document.
    pub buffers: Vec<gltf::buffer::Data>,
    /// Base directory of the glTF document.
    pub base_dir: PathBuf,
    /// Models of the glTF document.
    pub models: HashMap<usize, Arc<GltfModel>>,
    /// Materials of the glTF document.
    pub materials: HashMap<Option<usize>, Arc<Material>>,
    /// RGB images of the glTF document.
    pub rgb_images: HashMap<usize, Arc<RgbImage>>,
    /// RGBA images of the glTF document.
    pub rgba_images: HashMap<usize, Arc<RgbaImage>>,
    /// Gray images of the glTF document.
    pub gray_images: HashMap<(usize, usize), Arc<GrayImage>>,
}

impl GltfData {
    /// Create a new `GltfData` instance.
    pub fn new<P>(buffers: Vec<gltf::buffer::Data>, path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let mut base_dir = PathBuf::from(path.as_ref());
        base_dir.pop();
        GltfData {
            buffers,
            base_dir,
            models: Default::default(),
            materials: Default::default(),
            rgb_images: Default::default(),
            rgba_images: Default::default(),
            gray_images: Default::default(),
        }
    }

    /// Get a rgb image from the glTF document.
    pub fn get_rgb_image(&self, texture: &gltf::Texture<'_>) -> Arc<RgbImage> {
        self.rgb_images
            .get(&texture.index())
            .expect("Didn't preload this image")
            .clone()
    }

    /// Get a base color image from the glTF document.
    pub fn get_base_color_image(&self, texture: &gltf::Texture<'_>) -> Arc<RgbaImage> {
        self.rgba_images
            .get(&texture.index())
            .expect("Didn't preload this image")
            .clone()
    }

    /// Get a gray image from the glTF document.
    pub fn get_gray_image(&self, texture: &gltf::Texture<'_>, channel: usize) -> Arc<GrayImage> {
        self.gray_images
            .get(&(texture.index(), channel))
            .expect("Didn't preload this image")
            .clone()
    }

    /// Load a texture from the glTF document.
    pub fn load_texture(&self, texture: &gltf::Texture<'_>) -> DynamicImage {
        let g_img = texture.source();
        let buffers = &self.buffers;
        match g_img.source() {
            Source::View { view, mime_type } => {
                let parent_buffer_data = &buffers[view.buffer().index()].0;
                let data = &parent_buffer_data[view.offset()..view.offset() + view.length()];
                let mime_type = mime_type.replace('/', ".");
                image::load_from_memory_with_format(
                    data,
                    ImageFormat::from_path(mime_type).unwrap(),
                )
                .unwrap()
            }
            Source::Uri { uri, mime_type } => {
                if uri.starts_with("data:") {
                    let encoded = uri.split(',').nth(1).unwrap();
                    let data = URL_SAFE_NO_PAD.decode(encoded).unwrap();
                    let mime_type = if let Some(ty) = mime_type {
                        ty
                    } else {
                        uri.split(',')
                            .next()
                            .unwrap()
                            .split(':')
                            .nth(1)
                            .unwrap()
                            .split(';')
                            .next()
                            .unwrap()
                    };
                    let mime_type = mime_type.replace('/', ".");
                    image::load_from_memory_with_format(
                        &data,
                        ImageFormat::from_path(mime_type).unwrap(),
                    )
                    .unwrap()
                } else {
                    let path = self.base_dir.join(uri);
                    open(path).unwrap()
                }
            }
        }
    }
}
