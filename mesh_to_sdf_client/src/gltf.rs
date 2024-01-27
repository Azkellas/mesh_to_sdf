use anyhow::Result;

pub fn load_scene(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    path: &str,
) -> Result<Vec<easy_gltf::Model>> {
    let gltf = easy_gltf::load(path).map_err(|e| anyhow::anyhow!("{}", e))?;

    // TODO: clones meshes.
    Ok(gltf
        .iter()
        .flat_map(|scene| scene.models.iter())
        .map(|m| m.clone())
        .collect())
}
