use anyhow::Result;

use crate::passes::mip_generation_pass::MipGenerationPass;

pub fn generate_mipmaps(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> Result<()> {
    // TODO: only create the pass once.
    let pass = MipGenerationPass::new(device)?;

    let mip_count = texture.mip_level_count();

    let views = (0..mip_count)
        .map(|mip| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("generate_mipmaps::mip_view"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            })
        })
        .collect::<Vec<_>>();

    for target_mip in 1..mip_count as usize {
        pass.run(device, encoder, &views[target_mip - 1], &views[target_mip]);
    }

    Ok(())
}
