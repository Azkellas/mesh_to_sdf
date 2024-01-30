use anyhow::Result;

use crate::pbr::model::Model;

/// Load scene via `easy_gltf`.
/// `easy_gltf` is a bottleneck here since we need to convert their model to our model (they don't support bytemucks).
/// and their api, while convenient at first is very limiting (poor material support, etc.)
/// TODO: replace by bare `gltf` crate.
pub fn load_scene(device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> Result<Vec<Model>> {
    let gltf = easy_gltf::load(path).map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(gltf
        .iter()
        .flat_map(|scene| scene.models.iter())
        .map(|model| Model::from_gtlf(device, queue, model))
        .collect())
}
